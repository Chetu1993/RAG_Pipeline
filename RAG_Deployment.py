from langchain_ollama import ChatOllama
from fastapi import FastAPI,UploadFile,File,HTTPException
from pydantic import BaseModel
from langchain_text_splitters import RecursiveCharacterTextSplitter
import pytesseract
from langchain_core.documents import Document
from langchain_chroma import Chroma
import OnnxEmbeddings
import uvicorn
pytesseract.pytesseract.tesseract_cmd=(r'C:\Program Files\Tesseract-OCR\tesseract.exe')
from pdf2image import convert_from_path
import os
os.environ.pop('OLLAMA_HOST',None)

poppler_path=r'C:\poppler\Library\bin'
chroma_db=r'.\chroma_db'

embeddings=OnnxEmbeddings.OnnxClass()
upload_dir=os.path.abspath('data/upload')
os.makedirs(upload_dir,exist_ok=True)
db=Chroma(persist_directory=chroma_db,embedding_function=embeddings)

class Query(BaseModel):
    question:str

app=FastAPI()

def ingest_pdf(filepath):
    if not os.path.exists(filepath) or os.path.getsize(filepath)==0:
        raise ValueError("pdf file not found or invalid")

    documents=[]
    splitter=RecursiveCharacterTextSplitter(chunk_size=900,chunk_overlap=200)
    pages=convert_from_path(filepath,poppler_path=poppler_path)
    for page_no,page in enumerate(pages):

        text=pytesseract.image_to_string(page,lang="eng")
        if not text.split():
            continue
        chunks=splitter.split_text(text)
        for chunk in chunks:
            documents.append(Document(page_content=chunk,metadata={'source':os.path.basename(filepath),'page':page_no}))


    if not documents:
        raise ValueError("cannout extracted texts from pdf")

    db.add_documents(documents)


@app.post("/ingest")
async def upload_file(file:UploadFile=File(...)):
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400,detail='only pdf files are allowed')
    content=await file.read()

    if not content:
        raise HTTPException(status_code=400,detail='empty file')

    max_size=20*1024*1024
    if len(content)>max_size:
        raise HTTPException(status_code=400,detail='file size should be less than 20MB')

    filepath=os.path.join(upload_dir,file.filename)

    with open(filepath,'wb') as f:
        f.write(content)

    try:
        ingest_pdf(filepath)
    except Exception as e:
        raise HTTPException(status_code=400,detail=str(e))

    return {'status':'success','file':file.filename}

@app.post("/rag")
async def get_response(query:Query):
    try:
        result=db.similarity_search(query.question,k=3)
    except Exception as e:
        raise HTTPException(status_code=400,detail=str(e))
    if not result:
        return {'answer':'i dont know the answer based on the provided document','sources':[]}

    context="\n".join([doc.page_content for doc in result])


    prompt=f"""you are a helpful assistent, read the context, give me the answer for the question

    Rules:
    - read the context, provide the answer
    - answer should be clear and concise
    - if there is no answer from the context, say I don't have any information based on the provided context
    - dont repeat the context once the answer is provided
    
    Context
    {context}
    
    Question
    {query.question}
    
    Answer
    """

    llm=ChatOllama(model='mistral:latest')
    response=llm.invoke(prompt)

    sources=[{'sources':doc.metadata.get("source"),'page':doc.metadata.get("page")} for doc in result]

    return {"answer":response.content,'sources':sources}

if __name__=="__main__":
    uvicorn.run("RAG_Deployment:app",host="0.0.0.0",port=8000,reload=True)

