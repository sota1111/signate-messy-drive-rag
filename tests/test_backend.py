"""Offline contract tests for the Cloud Run ``/ask`` endpoint."""
from __future__ import annotations

import pytest
from fastapi import HTTPException

from backend.app import main


class _FakeDecision:
    def to_dict(self) -> dict:
        return {
            "answer": "42",
            "confidence": 0.93,
            "evidence": "report.xlsx の集計値",
            "method": "find_files → extract_office → compute",
            "gate_status": "commit",
            "reason": "合議一致・高確信",
        }


def test_ask_uses_gemini_agent_gate_and_returns_evidence(monkeypatch):
    from src.rag.agent import gate

    seen: list[str] = []

    def fake_gate_question(question: str):
        seen.append(question)
        return _FakeDecision()

    monkeypatch.setattr(gate, "gate_question", fake_gate_question)

    response = main.ask(main.AskRequest(question="  representative question  "))

    assert seen == ["representative question"]
    assert response.answer == "42"
    assert response.confidence == 0.93
    assert response.evidence == "report.xlsx の集計値"
    assert response.method == "find_files → extract_office → compute"
    assert response.gate_status == "commit"
    assert response.reason == "合議一致・高確信"
    assert response.evidence_files == []


def test_ask_does_not_use_legacy_or_claude_backends(monkeypatch):
    from src.rag import generate, opus_gen
    from src.rag.agent import gate

    monkeypatch.setattr(generate, "answer_question",
                        lambda *a, **k: pytest.fail("legacy generate backend must not run"))
    monkeypatch.setattr(opus_gen, "answer_question",
                        lambda *a, **k: pytest.fail("Claude backend must not run"))
    monkeypatch.setattr(gate, "gate_question", lambda question: _FakeDecision())

    assert main.ask(main.AskRequest(question="Q")).answer == "42"


def test_ask_rejects_blank_question_before_loading_agent():
    with pytest.raises(HTTPException) as exc:
        main.ask(main.AskRequest(question=" \n "))
    assert exc.value.status_code == 400


def test_health_does_not_load_agent(monkeypatch):
    monkeypatch.setenv("GCP_PROJECT_ID", "test-project")
    assert main.health() == {"status": "ok", "project": "test-project"}
