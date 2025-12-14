from pdf2image import convert_from_path
import pytesseract

pytesseract.pytesseract.tesseract_cmd = (
    r"C:\Program Files\Tesseract-OCR\tesseract.exe"
)
# from langchain_community.vectorstores import Chroma
from langchain_chroma import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import ChatOllama
from fastapi import FastAPI
from pydantic import BaseModel
import OnnxClass
import os
os.environ.pop("OLLAMA_HOST",None)
import uvicorn

class Query(BaseModel):
    question: str
app=FastAPI()
file_path=r"/data/upload\\Universe_info.pdf"
pdf_format=convert_from_path(file_path,poppler_path=r'C:\poppler\Library\bin')
texts=[]
for i,page in enumerate(pdf_format):
    text=pytesseract.image_to_string(page)
    texts.append(text)


chunks=[]
text_split=RecursiveCharacterTextSplitter(chunk_size=900,chunk_overlap=200)
for i in texts:
    chunks.extend(text_split.split_text(i))

embeddings=OnnxClass.OnnxMiniEmbeddings()
db=Chroma.from_texts(chunks,embeddings,persist_directory='./chroma.db')

db=Chroma(persist_directory='./chroma.db',embedding_function=embeddings)

@app.post("/rag")
def ask_question(query:Query):
    result=db.similarity_search(query.question,k=3)
    context="\n".join([doc.page_content for doc in result])
    response= prompt = f"""
    You are a knowledgeable assistant.
    Answer the question strictly using the context provided below.

    Rules:
    - Use only the information in the context.
    - If the answer is not present in the context, say "I don't know based on the provided document."
    - Be concise and clear.
    - Do NOT repeat the context.

    Context:
    {context}

    Question:
    {query.question}

    Answer:
    """
    llm=ChatOllama(model="mistral:latest")
    final_response=llm.invoke(response)
    return {"answer":final_response.content}

if __name__=="__main__":
    uvicorn.run("RAG_Request:app",host="0.0.0.0",port=8000)





