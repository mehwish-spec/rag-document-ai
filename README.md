# RAG Document Intelligence API

## Overview
A production-ready Retrieval-Augmented Generation (RAG) pipeline that allows users to upload any PDF document and ask natural language questions about it. Built with LangChain, FAISS vector store, and FastAPI.

## Demo
Upload any PDF → Ask questions → Get answers with source citations in <100ms

## Features
- PDF document ingestion and chunking
- Semantic search using FAISS vector store
- HuggingFace sentence embeddings (all-MiniLM-L6-v2)
- FastAPI REST API with automatic docs
- Interactive Streamlit UI
- Source citations with page numbers
- <100ms retrieval latency

## Tech Stack
Python, LangChain, FAISS, HuggingFace Transformers, FastAPI, Streamlit, Sentence-Transformers

## Project Structure
rag-document-ai/
├── app/
│   ├── main.py          # FastAPI server
│   ├── rag_pipeline.py  # RAG pipeline
│   └── utils.py
├── streamlit_app.py     # Streamlit UI
└── requirements.txt

## API Endpoints
- POST /upload — Upload and process PDF
- POST /ask — Ask a question about the document
- GET /health — Health check
- GET /docs — Swagger UI

## Quick Start
```bash
# Install dependencies
pip install -r requirements.txt

# Start FastAPI server
uvicorn app.main:app --port 8005

# Start Streamlit UI (new terminal)
streamlit run streamlit_app.py
```

## Example
```bash
# Upload a PDF
curl -X POST http://localhost:8005/upload \
  -F "file=@document.pdf"

# Ask a question
curl -X POST http://localhost:8005/ask \
  -H "Content-Type: application/json" \
  -d '{"question": "What are the main topics covered?"}'

# Response
{
  "question": "What are the main topics covered?",
  "answer": "The document covers...",
  "sources": [{"page": 0, "content": "..."}],
  "latency_ms": 99.35
}
```
