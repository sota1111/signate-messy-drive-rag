"""FastAPI backend exposing the messy-drive RAG on Cloud Run (Vertex Gemini).

Mirrors toddler-private-rag's serving shape: a lean Cloud Run service backed by Vertex.
The retrieval index is baked into the image at /app/index_store; the glossary is served
from the minimal /app/corpus/社内管理 folder. The full corpus is NOT shipped (image stays
lean), so figure-image attachment is disabled server-side — text retrieval only.
"""
from __future__ import annotations

import os

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

app = FastAPI(title="signate-messy-drive-rag", version="0.1.0")


class AskRequest(BaseModel):
    question: str
    hard: bool = False


class AskResponse(BaseModel):
    answer: str
    confidence: str
    evidence_files: list[str]


@app.get("/health")
def health() -> dict:
    return {"status": "ok", "project": os.getenv("GCP_PROJECT_ID", "")}


@app.post("/ask", response_model=AskResponse)
def ask(req: AskRequest) -> AskResponse:
    if not req.question.strip():
        raise HTTPException(status_code=400, detail="question is required")
    # Imported lazily so /health works even if the index is still warming.
    from src.rag import generate

    res = generate.answer_question(req.question, hard=req.hard)
    return AskResponse(
        answer=res["answer"],
        confidence=res["confidence"],
        evidence_files=res["evidence_files"],
    )
