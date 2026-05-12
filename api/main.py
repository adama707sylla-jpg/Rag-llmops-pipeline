"""
FastAPI — RAG LLMOps API
Endpoints : /query · /health · /sources
"""

import time
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from rag.pipeline import RAGPipeline

app = FastAPI(
    title="RAG LLMOps API",
    description="Semantic search + LLM generation on arXiv ML papers",
    version="1.0.0",
)

# Init pipeline once at startup
pipeline = RAGPipeline(top_k=5)

# ── Schemas ──────────────────────────────────────────────

class QueryRequest(BaseModel):
    question: str = Field(..., min_length=5, max_length=500)
    top_k   : int = Field(default=5, ge=1, le=10)

class SourceSchema(BaseModel):
    paper_id: str
    title   : str
    authors : list[str]
    url     : str
    score   : float

class QueryResponse(BaseModel):
    question   : str
    answer     : str
    sources    : list[SourceSchema]
    model      : str
    latency_sec: float
    retrieved_k: int

# ── Endpoints ────────────────────────────────────────────

@app.get("/health")
def health():
    return {"status": "ok", "timestamp": time.time()}

@app.post("/query", response_model=QueryResponse)
def query(req: QueryRequest):
    try:
        pipeline.retriever.top_k = req.top_k
        result = pipeline.query(req.question)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/sources")
def get_sources(q: str, top_k: int = 3):
    """Retrieval only — no LLM generation."""
    try:
        papers = pipeline.retriever.retrieve(q)
        return {"query": q, "sources": papers[:top_k]}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
