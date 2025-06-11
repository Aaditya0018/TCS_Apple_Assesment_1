import os
import uuid
import logging
import shutil
import fitz
import requests

from fastapi import FastAPI, UploadFile, File
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer, CrossEncoder
from transformers import pipeline

# ---- Config & Setup ----
app = FastAPI()
app.mount("/static", StaticFiles(directory="static"), name="static")

UPLOAD_DIR = "uploaded_files"
VECTOR_DIR = "chroma_store"
COLLECTION_NAME = "rag_qna"

os.makedirs(UPLOAD_DIR, exist_ok=True)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)

# ---- Models & DB ----
model = SentenceTransformer("all-MiniLM-L6-v2")
reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
query_rewriter = pipeline("text2text-generation", model="google/flan-t5-small")

chroma_client = chromadb.Client(Settings(
    persist_directory=VECTOR_DIR,
    anonymized_telemetry=False
))
collection = chroma_client.get_or_create_collection(name=COLLECTION_NAME)

@app.get("/")
def read_index():
    return FileResponse("static/index.html")

@app.post("/upload")
async def upload_pdf(file: UploadFile = File(...)):
    file_path = os.path.join(UPLOAD_DIR, file.filename)

    existing = collection.get()
    existing_ids = existing.get("ids", [])
    if existing_ids:
        collection.delete(ids=existing_ids)
        logging.info(f"[UPLOAD] Deleted {len(existing_ids)} records from collection.")

    with open(file_path, "wb") as f:
        shutil.copyfileobj(file.file, f)

    text = extract_text_from_pdf(file_path)
    chunks = split_text(text)
    embeddings = model.encode(chunks).tolist()
    ids = [f"{file.filename}_{i}" for i in range(len(chunks))]

    collection.add(
        documents=chunks,
        embeddings=embeddings,
        ids=ids,
        metadatas=[{"file": file.filename}] * len(chunks)
    )

    logging.info(f"[UPLOAD] Indexed {len(chunks)} chunks for: {file.filename}")
    return {"message": f"{file.filename} uploaded and processed successfully."}

def extract_text_from_pdf(file_path):
    text = ""
    doc = fitz.open(file_path)
    for page in doc:
        text += page.get_text()
    return text

def split_text(text, chunk_size=500, overlap=100):
    words = text.split()
    chunks = []
    for i in range(0, len(words), chunk_size - overlap):
        chunk = " ".join(words[i:i + chunk_size])
        chunks.append(chunk)
    return chunks

def generate_queries(original_query):
    prompts = [
        f"Rewrite the question in a simpler way: {original_query}",
        f"Ask this in another format: {original_query}",
        f"Make this question more specific: {original_query}"
    ]
    rewritten = [original_query]
    for p in prompts:
        output = query_rewriter(p, max_length=64, do_sample=False)[0]["generated_text"]
        rewritten.append(output)
    return rewritten

class Question(BaseModel):
    query: str
    strategy: str = "basic" 
    # Basic RAG is used if not mentioned

@app.post("/ask")
def ask_question(q: Question):
    request_id = str(uuid.uuid4())
    logging.info(f"[{request_id}] Query: {q.query} | Strategy: {q.strategy}")

    try:
        query = q.query
        strategy = q.strategy.lower()

        # ---- BASIC Strategy ----
        if strategy == "basic":
            query_embedding = model.encode([query]).tolist()[0]
            results = collection.query(query_embeddings=[query_embedding], n_results=3)
            top_chunks = results['documents'][0]

        # ---- RERANK Strategy ----
        elif strategy == "rerank":
            query_embedding = model.encode([query]).tolist()[0]
            results = collection.query(query_embeddings=[query_embedding], n_results=10)
            retrieved_chunks = results['documents'][0]
            pairs = [(query, chunk) for chunk in retrieved_chunks]
            scores = reranker.predict(pairs)
            reranked = sorted(zip(retrieved_chunks, scores), key=lambda x: x[1], reverse=True)
            top_chunks = [text for text, _ in reranked[:3]]

        # ---- MULTIQUERY Strategy ----
        elif strategy == "multiquery":
            queries = generate_queries(query)
            all_chunks = []

            for q_ in queries:
                emb = model.encode([q_])[0].tolist()
                result = collection.query(query_embeddings=[emb], n_results=3)
                docs = result["documents"][0]
                all_chunks.extend(docs)

            seen = set()
            top_chunks = []
            for chunk in all_chunks:
                if chunk not in seen:
                    top_chunks.append(chunk)
                    seen.add(chunk)
                if len(top_chunks) >= 5:
                    break
        else:
            return {"error": f"Invalid strategy: {strategy}", "request_id": request_id}

        if not top_chunks:
            return {"request_id": request_id, "answer": "No relevant context found."}

        context = "\n".join(top_chunks)

        prompt = f"""Use the following context to answer the question.\n\nContext:\n{context}\n\nQuestion: {query}\n\nAnswer:"""
        logging.info(f"[{request_id}] Final Prompt Sent to LLM:\n{prompt}")

        response = requests.post("http://localhost:11434/api/generate", json={
            "model": "tinyllama",
            "prompt": prompt,
            "stream": False
        })

        if response.status_code == 200:
            llm_answer = response.json()["response"].strip()
            return {
                "request_id": request_id,
                "answer": llm_answer,
                "strategy_used": strategy,
                "chunks_used": top_chunks
            }
        else:
            return {
                "error": "LLM backend error",
                "status_code": response.status_code,
                "response": response.text
            }

    except Exception as e:
        logging.error(f"[{request_id}] Error: {str(e)}")
        return {"error": str(e), "request_id": request_id}
