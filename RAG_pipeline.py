from langchain_ollama import ChatOllama, OllamaEmbeddings
from fastapi import FastAPI, UploadFile, File, HTTPException, Header, Depends
from pydantic import BaseModel
from langchain_text_splitters import RecursiveCharacterTextSplitter
import pytesseract
from typing_extensions import NotRequired
from datetime import datetime, timedelta, timezone
from sentence_transformers import CrossEncoder
from typing import Any, Literal, Optional,Dict,TypedDict,List
import asyncio
from langchain_core.documents import Document
from langchain_chroma import Chroma
import uvicorn
import hashlib
from contextlib import asynccontextmanager
from pdf2image import convert_from_path
import os
import logging
import json

logging.basicConfig(level=logging.INFO,format='%(asctime)s [%(levelname)s] [%(message)s]')
logger=logging.getLogger(__name__)

# -----------------------------
# SYSTEM SETUP
#
# -----------------------------
os.environ.pop("OLLAMA_HOST",None)
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
poppler_path = r"C:\poppler\Library\bin"

upload_dir = os.path.abspath("data/upload")
os.makedirs(upload_dir, exist_ok=True)

chroma_db = "./chroma_db"
llm=ChatOllama(model='mistral:latest')
class RAGclass(TypedDict):
    question:str
    tenant_key: str
    filters: dict
    retrieved_docs:NotRequired[List[Document]]
    ranked_docs:NotRequired[[Document]]
    answer:NotRequired[str]
    need_retry:NotRequired[bool]

def verify_tenant_id(tenant_id:Literal["tenant_a_key","tenant_b_key"]=Header(...,description='tenant key for multi-tenant isolation')):
    tenant_keys={'tenant_a_key':'tenant_a',
                 'tenant_b_key':'tenant_b'}

    tenant_key=tenant_keys.get(tenant_id)
    if not tenant_key:
        return HTTPException(status_code=401,detail='unauthorized')
    return tenant_key



# -----------------------------
# MODELS
# -----------------------------
reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
embeddings = OllamaEmbeddings(model="nomic-embed-text")

db = Chroma(
    collection_name="docs",
    persist_directory=chroma_db,
    embedding_function=embeddings,
)

# -----------------------------
# API MODELS
# -----------------------------
class Query(BaseModel):
    question: str
    source: Optional[Literal["pdf", "db", "web"]] = None

# -----------------------------
# UTILITIES
# -----------------------------
def cleanup_expired_documents():
    now_ts = datetime.now(timezone.utc).timestamp()
    expired = db.get(where={"expires_at": {"$lt": now_ts}})
    if expired.get("ids"):
        db.delete(ids=expired["ids"])

def rerank_documents(question: str, docs: list, top_k: int = 3):
    pairs = [(question, doc.page_content) for doc in docs]
    scores = reranker.predict(pairs)
    ranked_docs = sorted(zip(scores, docs), key=lambda x: x[0], reverse=True)
    return [doc for _, doc in ranked_docs[:top_k]]

def file_sha256(path: str) -> str:
    with open(path, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()

# -----------------------------
# INGESTION
# -----------------------------
def ingest_pdf(filepath: str,tenant_key:str):
    if not os.path.exists(filepath) or os.path.getsize(filepath) == 0:
        return {"status": "error", "message": "Invalid PDF"}

    file_hash = file_sha256(filepath)
    existing = db.get(where={"file_hash": file_hash})

    if existing.get("ids"):
        return {
            "status": "skipped",
            "message": "PDF already exists. You can ask questions.",
        }

    splitter = RecursiveCharacterTextSplitter(chunk_size=900, chunk_overlap=200)
    pages = convert_from_path(filepath, poppler_path=poppler_path)

    documents = []
    expires_at_ts = (datetime.now(timezone.utc) + timedelta(days=90)).timestamp()

    for page_no, page in enumerate(pages):
        text = pytesseract.image_to_string(page, lang="eng")
        if not text.strip():
            continue

        for chunk in splitter.split_text(text):
            documents.append(
                Document(
                    page_content=chunk,
                    metadata={
                        "source":'pdf',
                        "filename": os.path.basename(filepath),
                        "page": page_no,
                        "file_hash": file_hash,
                        "is_active": True,
                        "expires_at": expires_at_ts,
                        "uploaded_at": datetime.now(timezone.utc).timestamp(),
                        "tenant_id":tenant_key
                    },
                )
            )

    if not documents:
        return {"status": "error", "message": "No text extracted"}

    db.add_documents(documents)
    return {"status": "success", "chunks": len(documents)}

# -----------------------------
# LIFESPAN
# -----------------------------

def retrived_docs_k(retrived,relavant):
    return len(set(retrived) & set(relavant))/len(relavant)

def hit_rate(retrived,relavant):
    return int(len(set(retrived) &set(relavant))>0)

@asynccontextmanager
async def lifespan(app: FastAPI):
    try:
        await asyncio.to_thread(cleanup_expired_documents)
        print("Expired documents cleaned successfully.")
    except Exception as e:
        print(f"Startup cleanup failed: {e}")
    yield
    print("application got shutdown")

app = FastAPI(lifespan=lifespan)

# -----------------------------
# ROUTES
# -----------------------------
@app.post("/ingest")
async def upload_file(file: UploadFile = File(...), tenant_key: str = Depends(verify_tenant_id)):
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(400, "Only PDF allowed")

    content = await file.read()
    if not content:
        raise HTTPException(400, "Empty file")

    filepath = os.path.join(upload_dir, file.filename)
    with open(filepath, "wb") as f:
        f.write(content)

    return ingest_pdf(filepath,tenant_key)


def retrieve_node(state:RAGclass)->RAGclass:
    results=db.max_marginal_relevance_search(
        state['question'],fetch_k=5,k=5,lambda_mult=0.5,filters=state['filters'])
    return {**state,'retrieved_docs':results}

def rerank_node(state:RAGclass)->RAGclass:
    if not state.get("retrieved_docs"):
        return {**state,"ranked_docs":[]}
    ranked=rerank_documents(state['question'],state['retrieved_docs'],top_k=5)
    return {**state,'ranked_docs':ranked}

def judge_context_node(state:RAGclass)->RAGclass:
    context="\n".join(d.page_content[:300] for d in state['ranked_docs'])
    decision=llm.invoke(f"""you are a strict judge.
    answer only YES or NO.
    context:
    {context}
    question:
    {state['question']}
    """)
    need_retry=decision.content.strip().upper()=="NO"
    logger.info(f"[JUDGE] decision: {decision.content.strip()} retry={need_retry}")
    return {**state,"need_retry":need_retry}

def retry_retrieval_node(state: RAGclass) -> RAGclass:
    rewrite = llm.invoke(f"""
Rewrite this question to retrieve better documents:

Original question:
{state['question']}
""").content.strip()
    logger.info(f"[RETRY] old_q='{state['question']}' new_q='{rewrite}'")

    new_filters = dict(state["filters"])
    new_filters.pop("source", None)

    results = db.max_marginal_relevance_search(
        rewrite,
        fetch_k=20,
        k=8,
        lambda_mult=0.3,
        filters=new_filters
    )

    return {
        **state,
        "question": rewrite,
        "retrieved_docs": results,
        "filters": new_filters
    }


def generate_node(state: RAGclass) -> RAGclass:
    context = "\n".join(d.page_content for d in state["ranked_docs"])

    prompt = f"""
Answer ONLY from context.
If not found, say "I don't know".

Context:
{context}

Question:
{state['question']}
"""

    response = llm.invoke(prompt)

    return {
        **state,
        "answer": response.content
    }

from langgraph.graph import StateGraph, END

graph = StateGraph(RAGclass)

graph.add_node("retrieve", retrieve_node)
graph.add_node("rerank", rerank_node)
graph.add_node("judge", judge_context_node)
graph.add_node("retry", retry_retrieval_node)
graph.add_node("generate", generate_node)

graph.set_entry_point("retrieve")

graph.add_edge("retrieve", "rerank")
graph.add_edge("rerank", "judge")

graph.add_conditional_edges(
    "judge",
    lambda s: "retry" if s["need_retry"] else "generate",
    {
        "retry": "retry",
        "generate": "generate",
    }
)

graph.add_edge("retry", "rerank")
graph.add_edge("generate", END)

rag_graph = graph.compile()



@app.post("/rag")
async def rag(query: Query, tenant_key: str = Depends(verify_tenant_id)):

    filters = {
        "expires_at": {"$gt": datetime.now(timezone.utc).timestamp()},
        "tenant_id": {"$eq": tenant_key}
    }
    if query.source:
        filters["source"] = {"$eq": query.source}

    state = {
        "question": query.question,
        "tenant_key": tenant_key,
        "filters": filters,
        "retrieved_docs": [],
        "ranked_docs": [],
        "answer": "",
        "need_retry": False
    }

    final_state = await asyncio.to_thread(rag_graph.invoke, state)

    sources = [
        {
            "source": d.metadata.get("filename"),
            "page": d.metadata.get("page")
        }
        for d in final_state["ranked_docs"]
    ]

    return {
        "answer": final_state["answer"],
        "sources": sources
    }


if __name__ == "__main__":
    uvicorn.run("testing2:app", host="0.0.0.0", port=8000, reload=True)
