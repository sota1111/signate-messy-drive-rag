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
    confidence: float
    evidence: str
    method: str
    gate_status: str
    reason: str
    evidence_files: list[str]


@app.get("/health")
def health() -> dict:
    return {"status": "ok", "project": os.getenv("GCP_PROJECT_ID", "")}


@app.post("/ask", response_model=AskResponse)
def ask(req: AskRequest) -> AskResponse:
    if not req.question.strip():
        raise HTTPException(status_code=400, detail="question is required")
    # The agent stack imports Office/image extraction dependencies. Keep it lazy so
    # /health remains available while the serving process is warming up.
    from src.rag.agent import gate

    # ``hard`` is retained for API compatibility. The Gemini agent stack already
    # performs iterative investigation and independent verification for every query.
    decision = gate.gate_question(req.question.strip())
    res = decision.to_dict()
    return AskResponse(
        answer=res["answer"],
        confidence=res["confidence"],
        evidence=res["evidence"],
        method=res["method"],
        gate_status=res["gate_status"],
        reason=res["reason"],
        # Kept for backward compatibility with the original endpoint schema. The
        # agent emits auditable free-form evidence rather than retrieval chunk paths.
        evidence_files=[],
    )
