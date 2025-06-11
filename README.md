# TCS_Apple_Assesment_1

## ✅ Task 1: General Knowledge Q&A using FastAPI + TinyLLaMA
1. FastAPI application to query general knowledge questions.

2. Uses TinyLLaMA (running locally via Ollama) as the language model.

3. Logs every API request with a unique identifier, user query, and LLM response at API level.

4. Now can ask general questions to the model.

## ✅ Task 2: RAG System for File Ingestion + Question Answering

Steps to Run:
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
   
## 🔍 Getting Started After Launch
Once the application is running in your browser:
1. Upload a PDF Document
Use the upload option in the interface to upload a PDF file (e.g., your resume or any document you want to query).

2. Ask a Question
After the file is successfully uploaded and processed, enter a question related to the content of the document.

3. Receive Contextual Answers
Click the Ask button to get a precise, context-aware answer generated using the content of your uploaded file.

## 🔍 RAG Startegies Used.
### Basic RAG
1. Embeds the user query using Sentence Transformers.
2. Retrieves top k (default: 3) relevant text chunks from the vector database (ChromaDB).
3. Simple, fast, and efficient for straightforward questions.

### Reranked RAG
1. Initially retrieves more chunks (default: 10) using the query embedding.

2. Then re-scores these chunks using a CrossEncoder (semantic similarity between query and chunk).

3. The top 3 most relevant chunks are selected based on the reranking score.

### Multi-query RAG
1. Generates multiple reformulations of the original question using a T5-based query rewriter.

2. Runs vector search for each reformulated query.

3. Aggregates results from all queries, removes duplicates, and selects top relevant chunks (up to 5).
   
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
