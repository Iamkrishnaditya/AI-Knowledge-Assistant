
import os
from pathlib import Path
from typing import List, Optional

from fastapi import FastAPI, Body
from pydantic import BaseModel
from dotenv import load_dotenv
USER_AGENT = os.getenv("USER_AGENT", "Mozilla/5.0 (compatible; RAGBot/0.1)")

# LangChain + ecosystem
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_community.document_loaders import (
    PyPDFLoader, TextLoader, WebBaseLoader
)
from langchain.schema import Document
from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain

# ---------- Config ----------
load_dotenv()
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3.1:8b")
CHROMA_DIR = os.getenv("CHROMA_DIR", "./storage")

DATA_DIR = Path("./data")
DATA_DIR.mkdir(exist_ok=True)
Path(CHROMA_DIR).mkdir(exist_ok=True)

# ---------- App ----------
app = FastAPI(title="RAG Knowledge Assistant", version="0.1.0")

# Lazy globals
_vs = None
_retriever = None
_llm = None
_rag_chain = None

# ---------- Helpers ----------
def get_embeddings():
    # CPU-friendly HF embeddings
    return HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)

def build_vectorstore():
    embeddings = get_embeddings()
    vs = Chroma(collection_name="docs", persist_directory=CHROMA_DIR, embedding_function=embeddings)
    return vs

def init_rag_chain():
    global _vs, _retriever, _llm, _rag_chain
    if _vs is None:
        _vs = build_vectorstore()
    _retriever = _vs.as_retriever(search_kwargs={"k": 4})
    _llm = ChatOllama(model=OLLAMA_MODEL, temperature=0.2)

    system_prompt = (
        "You are a precise assistant. Use ONLY the provided context to answer. "
        "If the answer is not in context, say you don't know briefly. "
        "Always return a short answer, then bullet citations as source titles or URLs."
    )
    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", "Question: {input}\n\nContext:\n{context}")
    ])

    doc_chain = create_stuff_documents_chain(_llm, prompt)
    _rag_chain = create_retrieval_chain(_retriever, doc_chain)
    return _rag_chain

def load_local_files(paths: List[str]) -> List[Document]:
    docs: List[Document] = []
    for p in paths:
        pth = Path(p)
        if not pth.exists():
            continue
        if pth.suffix.lower() in [".pdf"]:
            loader = PyPDFLoader(str(pth))
            docs.extend(loader.load())
        elif pth.suffix.lower() in [".txt", ".md"]:
            loader = TextLoader(str(pth), autodetect_encoding=True)
            docs.extend(loader.load())
    return docs

def load_urls(urls: List[str]) -> List[Document]:
    docs: List[Document] = []
    for url in urls:
        try:
            loader = WebBaseLoader(url)
            docs.extend(loader.load())
        except Exception:
            pass
    return docs

def upsert_docs(docs: List[Document]):
    global _vs
    if _vs is None:
        _vs = build_vectorstore()
    if not docs:
        return 0
    _vs.add_documents(docs)
    _vs.persist()
    return len(docs)

def ensure_chain_ready():
    if not all([_vs, _retriever, _llm, _rag_chain]):
        init_rag_chain()

# ---------- Schemas ----------
class IngestRequest(BaseModel):
    file_paths: Optional[List[str]] = []
    urls: Optional[List[str]] = []

class ChatRequest(BaseModel):
    query: str

# ---------- Endpoints ----------
@app.get("/health")
def health():
    return {"status": "ok", "embedding_model": EMBEDDING_MODEL, "ollama_model": OLLAMA_MODEL}

@app.post("/ingest")
def ingest(req: IngestRequest):
    files = req.file_paths or []
    urls = req.urls or []

    docs = []
    if files:
        docs += load_local_files(files)
    if urls:
        docs += load_urls(urls)

    added = upsert_docs(docs)
    return {"ingested_docs": added, "total_chunks": added}

@app.post("/chat")
def chat(req: ChatRequest):
    ensure_chain_ready()
    result = _rag_chain.invoke({"input": req.query})

    # Attach simple citations from source metadata
    sources = []
    for d in result.get("context", []):
        src = d.metadata.get("source") or d.metadata.get("url") or d.metadata.get("file_path")
        if src and src not in sources:
            sources.append(src)

    return {
        "answer": result.get("answer", "").strip(),
        "citations": sources[:6]
    }

# ---------- Dev helper: seed ingest ----------
@app.on_event("startup")
def _warmup():
    init_rag_chain()
    # Optional: auto-seed from data/seed_urls.txt and any PDFs in data/
    seed_urls_file = DATA_DIR / "seed_urls.txt"
    batch_files = [str(p) for p in DATA_DIR.glob("*.pdf")] + [str(p) for p in DATA_DIR.glob("*.txt")]
    docs = []
    if seed_urls_file.exists():
        with open(seed_urls_file, "r", encoding="utf-8") as f:
            urls = [line.strip() for line in f if line.strip()]
        docs += load_urls(urls)
    docs += load_local_files(batch_files)
    if docs:
        upsert_docs(docs)
