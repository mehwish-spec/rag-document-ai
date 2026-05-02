from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from app.rag_pipeline import RAGPipeline
import shutil
import os
import time

app = FastAPI(title="RAG Document Intelligence API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global pipeline instance
rag = RAGPipeline()
pipeline_ready = False

class QuestionRequest(BaseModel):
    question: str

class AnswerResponse(BaseModel):
    question: str
    answer: str
    sources: list
    latency_ms: float

@app.on_event("startup")
async def startup():
    global pipeline_ready
    print("Loading embeddings on startup...")
    rag.load_embeddings()
    rag.load_llm()
    
    # Load existing vectorstore if available
    if os.path.exists("data/faiss_index"):
        rag.load_vectorstore()
        rag.build_qa_chain()
        pipeline_ready = True
        print("Pipeline ready with existing index!")

@app.post("/upload")
async def upload_document(file: UploadFile = File(...)):
    global pipeline_ready
    
    if not file.filename.endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files supported")
    
    # Save uploaded file
    os.makedirs("data", exist_ok=True)
    file_path = f"data/{file.filename}"
    with open(file_path, "wb") as f:
        shutil.copyfileobj(file.file, f)
    
    # Process document
    chunks = rag.load_documents(file_path)
    rag.build_vectorstore(chunks)
    rag.save_vectorstore()
    rag.build_qa_chain()
    pipeline_ready = True
    
    return {
        "message": f"Document '{file.filename}' processed successfully!",
        "chunks": len(chunks),
        "status": "ready"
    }

@app.post("/ask", response_model=AnswerResponse)
async def ask_question(req: QuestionRequest):
    if not pipeline_ready:
        raise HTTPException(
            status_code=400,
            detail="No document uploaded yet. Please upload a PDF first."
        )
    
    t0 = time.perf_counter()
    result = rag.answer(req.question)
    latency = (time.perf_counter() - t0) * 1000
    
    return AnswerResponse(
        question=req.question,
        answer=result["answer"],
        sources=result["sources"],
        latency_ms=round(latency, 2)
    )

@app.get("/health")
async def health():
    return {
        "status": "ok",
        "pipeline_ready": pipeline_ready
    }

@app.get("/")
async def root():
    return {
        "message": "RAG Document Intelligence API",
        "docs": "/docs",
        "endpoints": ["/upload", "/ask", "/health"]
    }
