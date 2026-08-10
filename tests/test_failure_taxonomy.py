"""SOT-2584 — tests for the failure taxonomy (BUDGET_EXHAUSTED separated from epistemic abstain)."""
from __future__ import annotations

from src.rag.agent import failure_taxonomy as ft


def test_budget_exhausted_is_the_only_operational_code():
    assert ft.is_operational(ft.BUDGET_EXHAUSTED)
    for code in ft.TAXONOMY:
        if code != ft.BUDGET_EXHAUSTED:
            assert not ft.is_operational(code), code
            assert ft.is_epistemic(code), code


def test_ledger_states_map_onto_taxonomy():
    assert ft.from_ledger_state("BUDGET_EXHAUSTED") == ft.BUDGET_EXHAUSTED
    assert ft.from_ledger_state("SPIN_CUTOFF") == ft.BUDGET_EXHAUSTED  # dead-end = control failure
    assert ft.from_ledger_state("UNANSWERABLE") == ft.CORPUS_ABSENT
    assert ft.from_ledger_state("NOT_RETRIEVED") == ft.NOT_RETRIEVED
    assert ft.from_ledger_state("RETRIEVED_NOT_PARSED") == ft.PARSER_CAPABILITY_MISS
    assert ft.from_ledger_state("PARSED_AMBIGUOUS") == ft.DOC_RESOLUTION_FAILED
    assert ft.from_ledger_state("EVIDENCE_INCOMPLETE") == ft.EVIDENCE_INCOMPLETE


def test_unknown_state_is_none():
    assert ft.from_ledger_state("something_else") is None
    assert ft.from_ledger_state(None) is None


def test_stop_reason_boundary_is_budget_exhausted():
    assert ft.from_stop_reason("max_turns") == ft.BUDGET_EXHAUSTED
    assert ft.from_stop_reason("timeout") == ft.BUDGET_EXHAUSTED
    assert ft.from_stop_reason("model_error") == ft.BUDGET_EXHAUSTED


def test_registry_hard_constraint_is_doc_resolution_failed():
    assert ft.from_stop_reason(
        "DOC_NOT_FOUND_AFTER_EXHAUSTIVE_MANIFEST_SCAN") == ft.DOC_RESOLUTION_FAILED


def test_ledger_wins_over_stop_reason():
    # A deliberate UNANSWERABLE abstain that also hit the turn cap is epistemic, not budget.
    assert ft.classify(ledger_state="UNANSWERABLE", stop_reason="max_turns") == ft.CORPUS_ABSENT


def test_classify_falls_back_to_stop_reason():
    assert ft.classify(ledger_state=None, stop_reason="timeout") == ft.BUDGET_EXHAUSTED
    assert ft.classify(ledger_state=None, stop_reason="answered") is None  # committed = no code


def test_classify_investigation_object():
    class _Inv:
        stop_reason = "max_turns"
    assert ft.classify_investigation(_Inv()) == ft.BUDGET_EXHAUSTED
    assert ft.classify_investigation({"stop_reason": "timeout"}) == ft.BUDGET_EXHAUSTED


def test_tally_splits_operational_from_epistemic():
    codes = [ft.BUDGET_EXHAUSTED, ft.BUDGET_EXHAUSTED, ft.CORPUS_ABSENT, ft.NOT_RETRIEVED]
    result = ft.tally(codes)
    assert result["total"] == 4
    assert result["operational"] == 2   # the two BUDGET_EXHAUSTED
    assert result["epistemic"] == 2
    assert result["per_code"][ft.BUDGET_EXHAUSTED] == 2
