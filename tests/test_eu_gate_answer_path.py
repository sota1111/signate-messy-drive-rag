"""SOT-2635 — offline tests for the answer-path (``investigator``) commit-time EU gate wiring.

Network-free: an :class:`~src.rag.agent.investigator.Investigation` is constructed directly and fed to the
pure :func:`~src.rag.agent.investigator._apply_answer_eu_gate` / ``_eu_signals_from_investigation`` — no
Vertex, no live loop. Covers the four wiring guarantees:

  1. **Default OFF is byte-identical** — with RAG_EU_GATE unset the helper is a no-op (answer unchanged,
     no ``eu_gate`` telemetry key added).
  2. **A committed-but-EV-negative answer is倒された to 棄権** when the gate is on (wrong-suppression).
  3. **A well-supported answer (deterministic lane / evidence) commits** — no over-abstention.
  4. **The decision telemetry (tier / U / commit / flip / signals) is recorded for every case** (answered
     AND already-abstained — SOT-2629 全ケース記録), and a numeric execution-disagreement forces ABSTAIN.
"""
from __future__ import annotations

import pytest

from src.rag.agent import investigator
from src.rag.agent.investigator import (
    Answer,
    Investigation,
    Usage,
    _apply_answer_eu_gate,
    _eu_signals_from_investigation,
)


def _inv(answer: str, *, confidence: float = 0.5, evidence: str = "", method: str = "",
         model: str = "gemini-3.6-flash", contract: str | None = None,
         calc_record: dict | None = None, pot_lane: dict | None = None,
         stop_reason: str = "answered") -> Investigation:
    return Investigation(
        question="Q?",
        answer=Answer(answer=answer, confidence=confidence, evidence=evidence, method=method),
        iterations=1,
        tool_calls=["x"],
        usage=Usage(),
        model=model,
        elapsed_s=0.0,
        stop_reason=stop_reason,
        contract=contract,
        calc_record=calc_record,
        pot_lane=pot_lane,
    )


# --------------------------------------------------------------------------- 1. default OFF byte-identical
def test_off_is_noop_byte_identical(monkeypatch):
    monkeypatch.delenv("RAG_EU_GATE", raising=False)
    inv = _inv("some weak answer", evidence="", confidence=0.5)
    out = _apply_answer_eu_gate(inv, "Q?")
    assert out is inv
    assert out.answer.answer == "some weak answer"      # not flipped
    assert "eu_gate" not in out.interventions            # no telemetry key added when OFF


# --------------------------------------------------------------------------- 2. wrong-suppression (flip)
def test_on_flips_ev_negative_commit_to_abstain(monkeypatch):
    monkeypatch.setenv("RAG_EU_GATE", "1")
    monkeypatch.delenv("RAG_EU_GATE_TAU", raising=False)
    # LLM answer, no grounding evidence, no canonical doc (registry absent in test) ⇒ EV-negative.
    inv = _inv("42%", evidence="", confidence=0.5, model="gemini-3.6-flash")
    out = _apply_answer_eu_gate(inv, "Q?")
    assert investigator.is_abstain(out.answer.answer)    # 倒された to 棄権
    rec = out.interventions["eu_gate"]
    assert rec["enabled"] is True and rec["flipped"] is True and rec["commit"] is False
    assert rec["utility"] <= 0.0
    assert "signals" in rec and rec["already_abstain"] is False


# --------------------------------------------------------------------------- 3. no over-abstention
def test_on_keeps_deterministic_lane_answer(monkeypatch):
    monkeypatch.setenv("RAG_EU_GATE", "1")
    monkeypatch.delenv("RAG_EU_GATE_TAU", raising=False)
    inv = _inv("42", evidence="セルA1=42", confidence=1.0, model="deterministic")
    out = _apply_answer_eu_gate(inv, "Q?")
    assert out.answer.answer == "42"                     # kept (not flipped)
    rec = out.interventions["eu_gate"]
    assert rec["commit"] is True and rec["flipped"] is False and rec["utility"] > 0.0


# --------------------------------------------------------------------------- 4. telemetry on all cases
def test_already_abstain_is_recorded_not_resurrected(monkeypatch):
    monkeypatch.setenv("RAG_EU_GATE", "1")
    inv = _inv("わかりません", confidence=0.0, model="gemini-3.6-flash")
    out = _apply_answer_eu_gate(inv, "Q?")
    assert investigator.is_abstain(out.answer.answer)    # still abstain (never resurrected)
    rec = out.interventions["eu_gate"]
    assert rec["already_abstain"] is True and rec["flipped"] is False and rec["enabled"] is True


def test_numeric_execution_disagreement_forces_abstain(monkeypatch):
    monkeypatch.setenv("RAG_EU_GATE", "1")
    monkeypatch.delenv("RAG_EU_GATE_TAU", raising=False)
    inv = _inv("123", evidence="計算根拠", confidence=0.9, contract="numeric",
               calc_record={"op": "sum"}, pot_lane={"verdict": "MISMATCH"})
    out = _apply_answer_eu_gate(inv, "Q?")
    assert investigator.is_abstain(out.answer.answer)
    rec = out.interventions["eu_gate"]
    assert rec["flipped"] is True and rec["commit"] is False
    assert rec["signals"]["execution_engines_agree"] is False   # PoT三層 disagreement → hard blocker
    assert "execution_disagreement" in rec["tier"] or rec["tier"] == "ABSTAIN"


def test_tau_probe_does_not_flip_but_records_utility(monkeypatch):
    """The calibration probe (τ very negative) commits everything so answers stay baseline-identical while
    the per-question U is recorded for offline τ selection."""
    monkeypatch.setenv("RAG_EU_GATE", "1")
    monkeypatch.setenv("RAG_EU_GATE_TAU", "-9")
    inv = _inv("42%", evidence="", confidence=0.5, model="gemini-3.6-flash")
    out = _apply_answer_eu_gate(inv, "Q?")
    assert out.answer.answer == "42%"                    # NOT flipped under the probe threshold
    rec = out.interventions["eu_gate"]
    assert rec["flipped"] is False and rec["commit"] is True
    assert rec["utility"] <= 0.0                          # true U still recorded (not distorted)


# --------------------------------------------------------------------------- signal builder
def test_signal_builder_credits_deterministic_lane(monkeypatch):
    inv = _inv("42", evidence="", confidence=0.0, model="deterministic")
    s = _eu_signals_from_investigation(inv, "Q?")
    assert s.deterministic_lane is True
    assert s.canonical_doc_resolved is True              # deterministic ⇒ grounded by construction
    assert s.evidence_slots_complete is True


def test_signal_builder_llm_without_evidence_is_bare(monkeypatch):
    inv = _inv("42", evidence="", confidence=0.5, model="gemini-3.6-flash")
    s = _eu_signals_from_investigation(inv, "Q?")
    assert s.deterministic_lane is False
    assert s.evidence_slots_complete is False
    assert s.verbal_confidence == pytest.approx(0.5)
