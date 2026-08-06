"""Offline contract tests for the Cloud Run ``/ask`` endpoint.

SOT-2490: the default answer path is the investigator single pass (Vertex-only, Claude-independent);
the heavier 合議 (resolve = investigator → verifier → tie-break + gate) is opt-in per-request/env.
"""
from __future__ import annotations

import pytest
from fastapi import HTTPException

from backend.app import main


class _FakeAnswer:
    def __init__(self, answer: str, confidence: float, evidence: str, method: str):
        self.answer = answer
        self.confidence = confidence
        self.evidence = evidence
        self.method = method


class _FakeInvestigation:
    def __init__(self, answer: _FakeAnswer, stop_reason: str = "answered"):
        self.answer = answer
        self.stop_reason = stop_reason


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


def _stub_investigator(monkeypatch, answer: _FakeAnswer, stop_reason: str = "answered") -> list[str]:
    """Route ``/ask`` through a fake investigator; returns the list that records seen questions."""
    from src.rag.agent import investigator

    seen: list[str] = []

    def fake_answer_question(question: str):
        seen.append(question)
        return _FakeInvestigation(answer, stop_reason=stop_reason)

    monkeypatch.setattr(investigator, "answer_question", fake_answer_question)
    return seen


def _fail_gate(monkeypatch) -> None:
    from src.rag.agent import gate

    monkeypatch.setattr(gate, "gate_question",
                        lambda *a, **k: pytest.fail("合議 gate must not run on the investigator default"))


def test_ask_defaults_to_investigator_single_pass(monkeypatch):
    """The default path calls the investigator (not the 合議 gate) and surfaces its answer."""
    seen = _stub_investigator(
        monkeypatch,
        _FakeAnswer("42", 0.93, "report.xlsx の集計値", "find_files → extract_office → compute"),
    )
    _fail_gate(monkeypatch)

    response = main.ask(main.AskRequest(question="  representative question  "))

    assert seen == ["representative question"]
    assert response.answer == "42"
    assert response.confidence == 0.93
    assert response.evidence == "report.xlsx の集計値"
    assert response.method == "find_files → extract_office → compute"
    assert response.gate_status == "commit"
    assert "investigator" in response.reason
    assert response.evidence_files == []


def test_ask_investigator_abstain_maps_to_abstain_gate_status(monkeypatch):
    """An investigator abstention surfaces as gate_status='abstain'."""
    _stub_investigator(
        monkeypatch,
        _FakeAnswer("わかりません", 0.0, "", ""),
        stop_reason="timeout",
    )
    _fail_gate(monkeypatch)

    response = main.ask(main.AskRequest(question="hard question"))
    assert response.gate_status == "abstain"
    assert response.confidence == 0.0


def test_ask_resolve_opt_in_via_request_field(monkeypatch):
    """``mode='resolve'`` routes through the 合議 gate; the investigator default must not run."""
    from src.rag.agent import gate, investigator

    seen: list[str] = []

    def fake_gate_question(question: str):
        seen.append(question)
        return _FakeDecision()

    monkeypatch.setattr(gate, "gate_question", fake_gate_question)
    monkeypatch.setattr(investigator, "answer_question",
                        lambda *a, **k: pytest.fail("investigator must not run when resolve is requested"))

    response = main.ask(main.AskRequest(question="  q  ", mode="resolve"))
    assert seen == ["q"]
    assert response.answer == "42"
    assert response.gate_status == "commit"
    assert response.reason == "合議一致・高確信"


@pytest.mark.parametrize("env", [("ASK_RESOLVE", "1"), ("ASK_MODE", "resolve")])
def test_ask_resolve_opt_in_via_env(monkeypatch, env):
    """``ASK_RESOLVE=1`` / ``ASK_MODE=resolve`` opt into the 合議 gate."""
    from src.rag.agent import gate, investigator

    key, value = env
    monkeypatch.setenv(key, value)
    monkeypatch.setattr(gate, "gate_question", lambda question: _FakeDecision())
    monkeypatch.setattr(investigator, "answer_question",
                        lambda *a, **k: pytest.fail("investigator must not run when resolve env is set"))

    assert main.ask(main.AskRequest(question="Q")).answer == "42"


def test_request_mode_overrides_env(monkeypatch):
    """An explicit request ``mode`` wins over the env fallback (mode=investigator beats ASK_RESOLVE)."""
    monkeypatch.setenv("ASK_RESOLVE", "1")
    seen = _stub_investigator(monkeypatch, _FakeAnswer("inv", 0.8, "e", "m"))
    _fail_gate(monkeypatch)

    response = main.ask(main.AskRequest(question="Q", mode="investigator"))
    assert seen == ["Q"]
    assert response.answer == "inv"


def test_ask_does_not_use_legacy_or_claude_backends(monkeypatch):
    from src.rag import generate, opus_gen

    monkeypatch.setattr(generate, "answer_question",
                        lambda *a, **k: pytest.fail("legacy generate backend must not run"))
    monkeypatch.setattr(opus_gen, "answer_question",
                        lambda *a, **k: pytest.fail("Claude backend must not run"))
    _stub_investigator(monkeypatch, _FakeAnswer("42", 0.9, "e", "m"))

    assert main.ask(main.AskRequest(question="Q")).answer == "42"


def test_ask_rejects_blank_question_before_loading_agent():
    with pytest.raises(HTTPException) as exc:
        main.ask(main.AskRequest(question=" \n "))
    assert exc.value.status_code == 400


def test_health_does_not_load_agent(monkeypatch):
    monkeypatch.setenv("GCP_PROJECT_ID", "test-project")
    assert main.health() == {"status": "ok", "project": "test-project"}
