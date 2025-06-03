# TCS_Apple_Assesment_1

## ✅ Task 1: General Knowledge Q&A using FastAPI + TinyLLaMA
1. FastAPI application to query general knowledge questions.

2. Uses TinyLLaMA (running locally via Ollama) as the language model.

3. Logs every API request with a unique identifier, user query, and LLM response at API level.

Steps to Run (Task 1)
1. Install Ollama: https://ollama.com

2. Pull the TinyLLaMA model
```bash
ollama pull tinyllama
```
3. Run the model
```bash
ollama run tinyllama
```
4. Clone the repository
```bash
git clone <your_repo_url> cd <your_repo_name>
```
5. Install Python dependencies
```bash
pip install -r requirements.txt
```
6. Run the FastAPI app
```bash
uvicorn main:app --reload
```
7. Visit the app in browser
http://127.0.0.1:8000


## ✅ Task 2: RAG System for File Ingestion + Question Answering
1. Users can upload a file (e.g., a resume).

2. Text is extracted, chunked, embedded, and stored in a ChromaDB vector database.

3. Users can ask questions specific to the content of that file.

4. Relevant chunks are retrieved and passed as context to TinyLLaMA to generate an answer.
5. Updated Readme file
   
## 🔍 The steps to run Task 2 are the same as Task 1 (Steps 1–7).
Once the app is running in the browser:

1. Upload a PDF file using the upload option in the UI (e.g., your resume).

2. After upload, type a question related to the content of that file.

3. Click on the Ask button to receive a contextual answer based on the uploaded document.
## 🔍 RAG Pipeline
1. Text Extraction: via PyMuPDF (fitz)

2. Chunking: ~500-word chunks with 100-word overlap

3. Embedding: sentence-transformers (all-MiniLM-L6-v2)

4. Vector Store: ChromaDB

5. Question Encoding + Retrieval: Top relevant chunks are used

6. LLM Response: Answer generated using local TinyLLaMA via Ollama

## 🧠 Usage Flow
1. Start Ollama + TinyLLaMA

2. Run FastAPI (uvicorn main:app --reload)

3. Open browser → upload file → ask document-specific questions
