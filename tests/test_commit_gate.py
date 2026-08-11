"""SOT-2637 — unit tests for the backend-agnostic commit gate.

The matrix covers 契約型 × (検証済/未検証・universe 証明有無・slot 欠落・書式差・棄権ポリシー) and asserts the
equivalence between the reused parts (enum_scan guard) and the commit_gate verdict. The gate is pure, so
every case is offline (no network / no live model).
"""

from __future__ import annotations

import types

import pytest

from src.rag.agent import commit_gate as cg
from src.rag.agent import enum_scan as es
from src.rag.agent import exec_verifier as ev


# --------------------------------------------------------------------------- helpers
def _hist(name: str, value, ok: bool = True):
    return {"name": name, "ok": ok, "value": value}


def _packet(missing):
    return types.SimpleNamespace(missing=tuple(missing))


def _enum_result(*, kind: str, matched=(), scanned=("a",), applicable=("a",), unsupported=()):
    """Build a real EnumScanResult so the equivalence check exercises the actual SOT-2600 guard."""
    universe = es.Universe(documents_total=len(applicable), applicable_ids=tuple(applicable),
                           scanned_ids=tuple(scanned), unsupported_ids=tuple(unsupported), scope="project")
    return es.EnumScanResult(matched=tuple(matched), universe=universe, population_kind=kind,
                             ordering_rule="doc_order")


# --------------------------------------------------------------------------- flag / config
def test_enabled_default_off(monkeypatch):
    monkeypatch.delenv("RAG_COMMIT_GATE", raising=False)
    assert cg.enabled() is False


def test_enabled_on(monkeypatch):
    monkeypatch.setenv("RAG_COMMIT_GATE", "1")
    assert cg.enabled() is True


def test_abstain_after_default_and_override(monkeypatch):
    monkeypatch.delenv("RAG_COMMIT_GATE_ABSTAIN_AFTER", raising=False)
    assert cg.abstain_after() == 2
    monkeypatch.setenv("RAG_COMMIT_GATE_ABSTAIN_AFTER", "1")
    assert cg.abstain_after() == 1
    monkeypatch.setenv("RAG_COMMIT_GATE_ABSTAIN_AFTER", "nonsense")
    assert cg.abstain_after() == 2


# --------------------------------------------------------------------------- 1. numeric grounding
def test_numeric_grounded_commits():
    d = cg.evaluate("平均は？", "numeric", "42", evidence={},
                    session_tool_history=[_hist("compute", 42.0)])
    assert d.verdict == cg.COMMIT
    assert ev._leading_number(d.final_answer) == 42.0


def test_numeric_ungrounded_rejects():
    d = cg.evaluate("平均は？", "numeric", "42", evidence={}, session_tool_history=[])
    assert d.verdict == cg.REJECT
    assert any("numeric_ungrounded" in r for r in d.reasons)


def test_numeric_wrong_value_in_history_still_rejects():
    d = cg.evaluate("平均は？", "numeric", "42", evidence={},
                    session_tool_history=[_hist("compute", 99.0)])
    assert d.verdict == cg.REJECT


def test_cross_aggregate_grounded_via_response_shape():
    hist = [{"name": "corpus_aggregate", "response": {"value": 7}}]
    d = cg.evaluate("合計は？", "cross_aggregate", "7", evidence={}, session_tool_history=hist)
    assert d.verdict == cg.COMMIT


def test_non_numeric_answer_is_not_grounding_gated():
    # cross_aggregate argmax returns a column *label*, not a number ⇒ numeric grounding N/A.
    d = cg.evaluate("最も相関が高い列は？", "cross_aggregate", "bmi", evidence={}, session_tool_history=[])
    assert d.verdict == cg.COMMIT


def test_failed_compute_record_does_not_ground():
    d = cg.evaluate("平均は？", "numeric", "42", evidence={},
                    session_tool_history=[_hist("compute", 42.0, ok=False)])
    assert d.verdict == cg.REJECT


def test_non_numeric_contract_skips_numeric_check():
    d = cg.evaluate("担当者は？", "simple_lookup", "42", evidence={}, session_tool_history=[])
    assert d.verdict == cg.COMMIT
    assert "numeric" not in d.telemetry["checks"]


# --------------------------------------------------------------------------- 2. enum universe guard
def test_enum_none_uncertified_rejects():
    # person population, empty scan → index absence ≠ non-existence → 該当なし forbidden.
    res = _enum_result(kind="person", matched=())
    d = cg.evaluate("該当する担当者は？", "full_enumeration", "該当者はいません",
                    evidence={}, enum_result=res)
    assert d.verdict == cg.REJECT
    assert any("enum_none_uncertified" in r for r in d.reasons)


def test_enum_none_certified_commits():
    # document population, complete universe, empty scan → negative self-certifies → 該当なし allowed.
    res = _enum_result(kind="document", matched=(), scanned=("a",), applicable=("a",))
    d = cg.evaluate("該当する文書は？", "full_enumeration", "該当なし", evidence={}, enum_result=res)
    assert d.verdict == cg.COMMIT


def test_enum_positive_answer_not_none_gated():
    res = _enum_result(kind="person", matched=())
    d = cg.evaluate("該当する担当者は？", "full_enumeration", "田中, 佐藤", evidence={}, enum_result=res)
    assert d.verdict == cg.COMMIT
    assert "enum" not in d.telemetry["checks"]


def test_enum_none_without_result_skips_guard():
    d = cg.evaluate("該当する担当者は？", "full_enumeration", "該当なし", evidence={}, enum_result=None)
    assert d.verdict == cg.COMMIT
    assert d.telemetry["checks"]["enum"]["reason"] == "no_enum_result"


@pytest.mark.parametrize("kind,matched,unsupported,expect_forbidden", [
    ("person", (), (), True),            # evidence-based empty scan — uncertified negative
    ("document", (), (), False),         # complete document scan — certified negative
    ("document", (), ("enc.pdf",), True),  # unsupported doc blocked coverage — uncertified
])
def test_enum_guard_equivalence_with_forbids_none_answer(kind, matched, unsupported, expect_forbidden):
    """commit_gate の enum 判定は enum_scan.forbids_none_answer と 1:1 で一致する (部品等価性)."""
    applicable = ("a",) if not unsupported else ("a", "enc.pdf")
    scanned = ("a",)
    res = _enum_result(kind=kind, matched=matched, scanned=scanned, applicable=applicable,
                       unsupported=unsupported)
    assert es.forbids_none_answer(res) is expect_forbidden
    d = cg.evaluate("該当は？", "full_enumeration", "該当なし", evidence={}, enum_result=res)
    assert (d.verdict == cg.REJECT) is expect_forbidden


# --------------------------------------------------------------------------- 3. evidence completeness
def test_evidence_missing_downgrades_to_abstain():
    d = cg.evaluate("平均は？", "numeric", "42", evidence={},
                    session_tool_history=[_hist("compute", 42.0)],
                    packet=_packet(("operands",)))
    assert d.verdict == cg.ABSTAIN_V
    assert d.final_answer == cg.ABSTAIN
    assert any("evidence_incomplete" in r for r in d.reasons)


def test_evidence_complete_no_downgrade():
    d = cg.evaluate("平均は？", "numeric", "42", evidence={},
                    session_tool_history=[_hist("compute", 42.0)],
                    packet=_packet(()))
    assert d.verdict == cg.COMMIT


def test_no_packet_skips_evidence_check():
    d = cg.evaluate("平均は？", "numeric", "42", evidence={},
                    session_tool_history=[_hist("compute", 42.0)], packet=None)
    assert d.verdict == cg.COMMIT
    assert "evidence" not in d.telemetry["checks"]


# --------------------------------------------------------------------------- 5. 棄権ポリシー (asymmetry)
def test_reject_escalates_to_abstain_at_threshold(monkeypatch):
    monkeypatch.delenv("RAG_COMMIT_GATE_ABSTAIN_AFTER", raising=False)  # default 2
    first = cg.evaluate("平均は？", "numeric", "42", evidence={}, session_tool_history=[],
                        prior_rejects=0)
    assert first.verdict == cg.REJECT
    second = cg.evaluate("平均は？", "numeric", "42", evidence={}, session_tool_history=[],
                         prior_rejects=1)
    assert second.verdict == cg.ABSTAIN_V
    assert second.final_answer == cg.ABSTAIN
    assert any("reject_streak_abstain" in r for r in second.reasons)


def test_reject_escalation_threshold_override(monkeypatch):
    monkeypatch.setenv("RAG_COMMIT_GATE_ABSTAIN_AFTER", "1")
    d = cg.evaluate("平均は？", "numeric", "42", evidence={}, session_tool_history=[], prior_rejects=0)
    assert d.verdict == cg.ABSTAIN_V


# --------------------------------------------------------------------------- 4. formatting (value-invariant)
def test_already_abstain_passthrough():
    d = cg.evaluate("平均は？", "numeric", "わかりません", evidence={}, session_tool_history=[])
    assert d.verdict == cg.ABSTAIN_V
    assert d.final_answer == cg.ABSTAIN


def test_commit_formatting_is_value_invariant():
    d = cg.evaluate("平均年齢は？", "numeric", "42.5", evidence={},
                    session_tool_history=[_hist("compute", 42.5)])
    assert d.verdict == cg.COMMIT
    # the underlying numeric fact must survive the 書式契約 unchanged
    assert ev._leading_number(d.final_answer) == 42.5


# --------------------------------------------------------------------------- 6. telemetry (SOT-2629 shape)
def test_telemetry_shape_on_reject():
    d = cg.evaluate("平均は？", "numeric", "42", evidence={}, session_tool_history=[])
    tel = d.telemetry
    assert tel["enabled"] is True
    assert tel["verdict"] == cg.REJECT
    assert tel["route"] == "numeric"
    assert isinstance(tel["reasons"], list) and tel["reasons"]
    assert "numeric" in tel["checks"]
    # the record is a plain JSON-able dict suitable for interventions['commit_gate']
    assert set(tel) >= {"enabled", "verdict", "route", "reasons", "checks", "prior_rejects"}


def test_decision_to_dict_roundtrip():
    d = cg.evaluate("平均は？", "numeric", "42", evidence={},
                    session_tool_history=[_hist("compute", 42.0)])
    payload = d.to_dict()
    assert payload["verdict"] == cg.COMMIT
    assert payload["telemetry"]["verdict"] == cg.COMMIT
    assert isinstance(payload["reasons"], list)


def test_contract_accepts_question_contract_object():
    from src.rag.agent import question_contract as qc
    contract = qc.QuestionContract(contract="numeric", label="数値", archetype="numeric",
                                   completion_conditions=(), route="compute", method="deterministic",
                                   confidence=0.9)
    d = cg.evaluate("平均は？", contract, "42", evidence={},
                    session_tool_history=[_hist("compute", 42.0)])
    assert d.telemetry["route"] == "numeric"
    assert d.verdict == cg.COMMIT
