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
    # SOT-2490: the default answer path is the investigator single pass. Set ``mode="resolve"``
    # (or ``mode="investigator"`` to force the default) to opt into the heavy 合議 (resolve) path
    # per-request. When omitted, the env fallback (ASK_MODE / ASK_RESOLVE) decides. See ``ask``.
    mode: str | None = None


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


_RESOLVE_TRUTHY = {"1", "true", "yes", "on"}


def _resolve_requested(req: AskRequest) -> bool:
    """Decide whether to run the heavy 合議 (resolve) path instead of the investigator default.

    Precedence (SOT-2490): an explicit request ``mode`` field wins over the env fallback. The
    default — no ``mode`` field, no env override — is the investigator single pass (Vertex-only,
    ~4x cheaper than 合議). Opt into 合議 with ``mode="resolve"``, ``ASK_MODE=resolve`` or a truthy
    ``ASK_RESOLVE``. ``mode="investigator"`` (or any other value) forces the default.
    """
    if req.mode is not None and req.mode.strip():
        return req.mode.strip().lower() == "resolve"
    if os.getenv("ASK_MODE", "").strip().lower() == "resolve":
        return True
    return os.getenv("ASK_RESOLVE", "").strip().lower() in _RESOLVE_TRUTHY


def _investigator_answer(question: str) -> AskResponse:
    """Default answer path: the investigator single pass (調査AG + self-confidence abstention).

    The investigator already decides commit/棄権 from its own confidence, so we surface that as the
    ``gate_status``/``reason`` fields without running the verifier/tie-break/gate 合議.
    """
    from src.rag.agent import investigator

    inv = investigator.answer_question(question)
    ans = inv.answer
    committed = not investigator.is_abstain(ans.answer)
    reason = (
        f"investigator単一パス・自己確信 {ans.confidence:.2f}(合議なし)"
        if committed
        else f"investigator棄権 — 低確信/不明(self-confidence {ans.confidence:.2f}, "
        f"stop={inv.stop_reason})"
    )
    return AskResponse(
        answer=ans.answer,
        confidence=ans.confidence,
        evidence=ans.evidence,
        method=ans.method,
        gate_status="commit" if committed else "abstain",
        reason=reason,
        evidence_files=[],
    )


def _resolve_answer(question: str) -> AskResponse:
    """Opt-in answer path: the full 合議 (investigator → verifier → tie-break) + confidence gate."""
    from src.rag.agent import gate

    res = gate.gate_question(question).to_dict()
    return AskResponse(
        answer=res["answer"],
        confidence=res["confidence"],
        evidence=res["evidence"],
        method=res["method"],
        gate_status=res["gate_status"],
        reason=res["reason"],
        evidence_files=[],
    )


@app.post("/ask", response_model=AskResponse)
def ask(req: AskRequest) -> AskResponse:
    question = req.question.strip()
    if not question:
        raise HTTPException(status_code=400, detail="question is required")
    # The agent stack imports Office/image extraction dependencies. Keep the import lazy (inside the
    # path helpers) so /health remains available while the serving process is warming up.
    #
    # ``hard`` is retained for API compatibility. SOT-2490: the default answer path is the Gemini
    # investigator single pass (Vertex-only, Claude-independent); the heavier 合議 (resolve =
    # investigator → verifier → tie-break + gate) is opt-in per-request/env — see ``_resolve_requested``.
    if _resolve_requested(req):
        return _resolve_answer(question)
    return _investigator_answer(question)
