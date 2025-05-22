import uuid
from fastapi import FastAPI
from pydantic import BaseModel
import requests
import logging

from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse

app = FastAPI()
app.mount("/static", StaticFiles(directory="static"), name="static")

@app.get("/")
def read_index():
    return FileResponse("static/index.html")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)

class Question(BaseModel):
    query: str

@app.post("/ask")
def ask_question(q: Question):
    request_id = str(uuid.uuid4())
    logging.info(f"Request ID: {request_id}")
    try:
        logging.info(f"[{request_id}] User Query: {q.query}")
        response = requests.post("http://localhost:11434/api/generate", json={
            "model": "tinyllama",
            "prompt": q.query,
            "stream": False,
        })
        logging.info(f"[{request_id}] Response Code: {response.status_code}")

        if response.status_code == 200:
            llm_answer = response.json()["response"].strip()
            logging.info(f"[{request_id}] LLM Answer: {llm_answer}")
            return {"request_id": request_id, "answer": llm_answer}
        else:
            return {"error": "Error Occurred", "request_id": request_id}
    except Exception as e:
        return {"error": str(e), "request_id": request_id}
