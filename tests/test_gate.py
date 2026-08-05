"""SOT-2473 — offline tests for the precision-first confidence gate on the verifier consensus.

Network-free: :class:`~src.rag.agent.tiebreak.Resolution` objects are produced from *scripted* fake
models (same shape as :mod:`tests.test_tiebreak`, no Vertex) and fed to :func:`src.rag.agent.gate.apply_gate`.
Covers the acceptance criterion — **低確信/不一致は棄権** — and the Incorrect-minimisation property that
motivates the gate (Incorrect=−1 / Missing=0): a low-confidence agreement that would be *wrong* is
converted to an abstention, never committed.
"""
from __future__ import annotations

from config import settings
from src.rag.agent.investigator import (
    ABSTAIN,
    SUBMIT_ANSWER,
    Answer,
    Call,
    Investigation,
    Step,
    Usage,
)
from src.rag.agent import gate
from src.rag.agent.gate import GateDecision, apply_gate
from src.rag.agent.tiebreak import (
    STATUS_AGREED,
    STATUS_ABSTAINED,
    STATUS_TIEBREAK_INVESTIGATOR,
    Resolution,
    resolve_answer,
)


# --------------------------------------------------------------------------- scripted resolution build
class _ScriptedModel:
    """Fake :class:`~src.rag.agent.investigator.Model` replaying one ``submit_answer`` step."""

    def __init__(self, answer, *, confidence=0.9):
        self._answer, self._confidence = answer, confidence
        self.model_name = "fake"
        self.calls = 0

    def next(self, tool_responses):
        self.calls += 1
        return Step(function_calls=(Call(SUBMIT_ANSWER, {
            "answer": self._answer, "confidence": self._confidence,
            "evidence": "e", "method": "m"}),), usage=Usage(50, 10))


def _investigation(answer, *, confidence=0.9) -> Investigation:
    return Investigation(
        question="Q?", answer=Answer(answer=answer, confidence=confidence, evidence="ie", method="im"),
        iterations=1, tool_calls=[], usage=Usage(100, 20), model="gemini-2.5-pro",
        elapsed_s=0.1, stop_reason="answered",
    )


def _resolution(inv_answer, ver_answer, judge_answer=None, *, confidence=0.9) -> Resolution:
    """A realistic :class:`Resolution` from the scripted 合議 (agree / tie-break / abstain per inputs)."""
    return resolve_answer(
        "Q?", _investigation(inv_answer, confidence=confidence),
        verifier_model=_ScriptedModel(ver_answer),
        judge_model=(_ScriptedModel(judge_answer) if judge_answer is not None else None),
        max_turns=3,
    )


# --------------------------------------------------------------------------- 合議一致 → 高確信 commit
def test_agreement_high_confidence_commits():
    r = _resolution("1526", "1526.0", confidence=0.9)
    assert r.status == STATUS_AGREED
    d = apply_gate(r, commit_confidence=0.7)
    assert isinstance(d, GateDecision)
    assert d.commit and d.answer == "1526" and d.confidence == 0.9
    assert not d.abstained
    assert "commit" in d.reason


def test_agreement_confidence_exactly_at_threshold_commits():
    d = apply_gate(_resolution("A", "A", confidence=0.7), commit_confidence=0.7)
    assert d.commit and d.answer == "A"  # boundary is inclusive (≥)


# --------------------------------------------------------------------------- 合議一致 → 低確信 棄権
def test_agreement_low_confidence_abstains():
    r = _resolution("1526", "1526", confidence=0.4)
    d = apply_gate(r, commit_confidence=0.7)
    assert not d.commit and d.answer == settings.ABSTAIN and d.confidence == 0.0
    assert d.abstained
    assert "低確信" in d.reason


# --------------------------------------------------------------------------- 合議棄権 → 棄権を維持
def test_consensus_abstention_stays_abstained():
    # investigator/verifier disagree and the judge lands on a third value → 決着不能 → 棄権.
    r = _resolution("1530", "1526", judge_answer="1600")
    assert r.status == STATUS_ABSTAINED and r.abstained
    d = apply_gate(r)
    assert not d.commit and d.answer == settings.ABSTAIN
    assert "合議が棄権" in d.reason


# --------------------------------------------------------------------------- tie-break → 既定棄権
def test_tiebreak_abstains_by_default_even_when_confident():
    # investigator was wrong-but-confident; judge sided with investigator → tie-broken commit candidate.
    r = _resolution("1530", "1526", judge_answer="1530", confidence=0.99)
    assert r.status == STATUS_TIEBREAK_INVESTIGATOR and not r.abstained
    d = apply_gate(r)  # commit_on_tiebreak defaults OFF (precision-first)
    assert not d.commit and d.answer == settings.ABSTAIN
    assert "不一致" in d.reason and "棄権" in d.reason


def test_tiebreak_opt_in_commits_above_stricter_bar():
    r = _resolution("1530", "1526", judge_answer="1530", confidence=0.9)
    d = apply_gate(r, commit_on_tiebreak=True, tiebreak_confidence=0.85)
    assert d.commit and d.answer == "1530" and d.confidence == 0.9


def test_tiebreak_opt_in_still_abstains_below_stricter_bar():
    r = _resolution("1530", "1526", judge_answer="1530", confidence=0.8)
    d = apply_gate(r, commit_on_tiebreak=True, tiebreak_confidence=0.85)
    assert not d.commit and d.answer == settings.ABSTAIN
    assert "低確信" in d.reason


# --------------------------------------------------------------------------- Incorrect minimisation
def test_gate_minimises_incorrect_by_abstaining_low_confidence_wrong_agreements():
    """受け入れ条件 + 検証内容: Incorrect(-1) を最小化する。

    A slate where both AGs *agree* but on a WRONG value at low confidence (a correlated near-miss) — the
    consensus alone would commit it and score −1. The precision-first gate turns each such low-confidence
    agreement into Missing(0), strictly improving the official score, while leaving high-confidence
    correct agreements committed.
    """
    def score(served: str, truth: str) -> int:
        if served == settings.ABSTAIN:
            return 0                       # Missing
        return 1 if served == truth else -1  # Correct / Incorrect

    slate = [
        # (inv, ver, confidence, ground truth)
        ("0.8999", "0.8999", 0.4, "0.9000"),  # low-conf agreement on a near-miss WRONG value
        ("8", "8", 0.5, "9"),                 # low-conf agreement, wrong count
        ("東京", "東京", 0.95, "東京"),          # high-conf agreement, correct → must stay committed
    ]
    consensus_total = 0
    gated_total = 0
    for inv, ver, conf, truth in slate:
        r = _resolution(inv, ver, confidence=conf)
        assert r.status == STATUS_AGREED
        consensus_total += score(r.answer, truth)             # raw consensus commits everything
        gated_total += score(apply_gate(r, commit_confidence=0.7).answer, truth)

    # Raw consensus: -1 -1 +1 = -1 ; gated: 0 0 +1 = +1. Gate minimises Incorrect and lifts the score.
    assert consensus_total == -1
    assert gated_total == 1
    assert gated_total > consensus_total


# --------------------------------------------------------------------------- schema / helpers
def test_gate_decision_to_dict_and_to_answer_commit():
    d = apply_gate(_resolution("42", "42", confidence=0.9), commit_confidence=0.7)
    dd = d.to_dict()
    assert dd["answer"] == "42" and dd["commit"] is True and dd["gate_status"] == "commit"
    assert dd["confidence"] == 0.9 and dd["status"] == STATUS_AGREED and dd["agree"] is True
    assert dd["resolution"]["answer"] == "42"
    a = d.to_answer()
    assert a.answer == "42" and a.confidence == 0.9 and a.evidence == "ie"


def test_gate_decision_to_dict_and_to_answer_abstain():
    d = apply_gate(_resolution("42", "42", confidence=0.1), commit_confidence=0.7)
    dd = d.to_dict()
    assert dd["answer"] == settings.ABSTAIN and dd["commit"] is False
    assert dd["gate_status"] == "abstain" and dd["confidence"] == 0.0
    assert dd["evidence"] == ""            # abstention carries no evidence/method
    a = d.to_answer()
    assert a.answer == settings.ABSTAIN and a.confidence == 0.0


# --------------------------------------------------------------------------- live wiring / thresholds
def test_gate_question_wires_resolve_then_gates(monkeypatch):
    """``gate_question`` runs the consensus then gates it (network-free via monkeypatched resolve)."""
    r = _resolution("1526", "1526", confidence=0.9)
    monkeypatch.setattr(gate, "resolve_question", lambda q, **kw: r)
    d = gate.gate_question("Q?", commit_confidence=0.7)
    assert d.commit and d.answer == "1526"


def test_gate_question_abstains_low_confidence(monkeypatch):
    r = _resolution("1526", "1526", confidence=0.3)
    monkeypatch.setattr(gate, "resolve_question", lambda q, **kw: r)
    d = gate.gate_question("Q?", commit_confidence=0.7)
    assert not d.commit and d.answer == settings.ABSTAIN


def test_default_thresholds_are_precision_first():
    # Defaults: commit bar >0.5 EV break-even, tie-break commit OFF (disagreement abstains).
    assert gate.GATE_COMMIT_CONFIDENCE > 0.5
    assert gate.GATE_COMMIT_ON_TIEBREAK is False
    assert gate.GATE_TIEBREAK_CONFIDENCE >= gate.GATE_COMMIT_CONFIDENCE


def test_run_gated_backend_routes_to_the_gate(monkeypatch):
    """``run.make_worker('gated')`` serves the gate decision through the run-log schema."""
    from src.rag import run

    d = apply_gate(_resolution("1526", "1526", confidence=0.9), commit_confidence=0.7)
    monkeypatch.setattr(gate, "gate_question", lambda q: d)
    res = run.make_worker("gated", False)(1, "q")[1]
    assert res["answer"] == "1526" and res["commit"] is True
    assert res["used_images"] == 0  # agent backends get the uniform run-log key
    assert "gated" in run.GEN_CHOICES
