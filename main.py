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
from sentence_transformers import SentenceTransformer

# ---- Config & Setup ----
app = FastAPI()
app.mount("/static", StaticFiles(directory="static"), name="static")

UPLOAD_DIR = "uploaded_files"
VECTOR_DIR = "chroma_store"
COLLECTION_NAME = "rag_qna"

os.makedirs(UPLOAD_DIR, exist_ok=True)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # loosened for testing
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)

# ---- Vector Store ----
model = SentenceTransformer("all-MiniLM-L6-v2")
chroma_client = chromadb.Client(Settings(
    persist_directory=VECTOR_DIR,
    anonymized_telemetry=False
))
collection = chroma_client.get_or_create_collection(name=COLLECTION_NAME)

@app.get("/")
def read_index():
    return FileResponse("static/index.html")

# ---- Upload PDF and Index Vectors ----
@app.post("/upload")
async def upload_pdf(file: UploadFile = File(...)):
    file_path = os.path.join(UPLOAD_DIR, file.filename)

    with open(file_path, "wb") as f:
        shutil.copyfileobj(file.file, f)

    text = extract_text_from_pdf(file_path)
    chunks = split_text(text)

    embeddings = model.encode(chunks).tolist()
    ids = [f"{file.filename}_{i}" for i in range(len(chunks))]

    collection.delete(where={"file": file.filename})

    collection.add(
        documents=chunks,
        embeddings=embeddings,
        ids=ids,
        metadatas=[{"file": file.filename}] * len(chunks)
    )

    logging.info(f"Uploaded and indexed: {file.filename}")
    return {"message": f"{file.filename} uploaded and processed successfully."}

# ---- Text Extraction from file ----
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

class Question(BaseModel):
    query: str

@app.post("/ask")
def ask_question(q: Question):
    request_id = str(uuid.uuid4())
    logging.info(f"Request ID: {request_id}")
    try:
        query = q.query
        query_embedding = model.encode([query]).tolist()[0]

        results = collection.query(
            query_embeddings=[query_embedding],
            n_results=3
        )

        context = "\n".join(results['documents'][0])

        prompt = f"""Use the following context to answer the question.\n\nContext:\n{context}\n\nQuestion: {query}\n\nAnswer:"""

        logging.info(f"[{request_id}] Query: {query}")
        logging.info(f"[{request_id}] Context: {context}")

        response = requests.post("http://localhost:11434/api/generate", json={
            "model": "tinyllama",
            "prompt": prompt,
            "stream": False,
        })

        logging.info(f"[{request_id}] Response Code: {response.status_code}")

        if response.status_code == 200:
            llm_answer = response.json()["response"].strip()
            logging.info(f"[{request_id}] Answer: {llm_answer}")
            return {"request_id": request_id, "answer": llm_answer}
        else:
            return {"error": "Error from LLM backend", "request_id": request_id}
    except Exception as e:
        logging.error(f"[{request_id}] Exception: {str(e)}")
        return {"error": str(e), "request_id": request_id}
