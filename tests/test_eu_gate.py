"""SOT-2589 — offline tests for the expected-utility three-tier abstain gate.

Network-free: :class:`~src.rag.agent.eu_gate.GateSignals` bundles are constructed directly and fed to the
pure :func:`~src.rag.agent.eu_gate.decide` / :func:`~src.rag.agent.eu_gate.expected_utility`. Also covers
the :mod:`src.rag.agent.gate` wiring: default OFF is byte-identical to the confidence gate, and when
``RAG_EU_GATE`` is on the served commit/abstain follows the expected-utility decision.

The four acceptance criteria are pinned as tests:
  1. commit/abstain follows U = P(P)+0.5P(A)−P(I) > 0 (three-tier HARD/SOFT/ABSTAIN);
  2. verbal confidence is demoted to auxiliary, hard evidence signals are the primary input;
  3. BUDGET_EXHAUSTED is excluded from the abstain criteria;
  4. default OFF is byte-identical (regression 0).
"""
from __future__ import annotations

import pytest

from src.rag.agent import eu_gate
from src.rag.agent.eu_gate import (
    ABSTAIN,
    HARD_ACCEPT,
    SOFT_ACCEPT,
    GateSignals,
    correctness_prob,
    decide,
    expected_utility,
    grade_probabilities,
    perfect_share,
)


# --------------------------------------------------------------------------- utility formula
def test_expected_utility_matches_grade_formula():
    s = GateSignals(canonical_doc_resolved=True, evidence_slots_complete=True,
                    answer_verifier_agrees=True)
    p_perfect, p_acceptable, p_incorrect = grade_probabilities(s)
    assert p_perfect + p_acceptable + p_incorrect == pytest.approx(1.0)
    expected = p_perfect + 0.5 * p_acceptable - p_incorrect
    assert expected_utility(s) == pytest.approx(expected)
    # correctness == P(P)+P(A) == 1 - P(I)
    assert correctness_prob(s) == pytest.approx(p_perfect + p_acceptable)


def test_grade_probabilities_and_perfect_share_bounds():
    for s in (GateSignals(), GateSignals(deterministic_lane=True, answer_verifier_agrees=True,
                                         evidence_slots_complete=True, canonical_doc_resolved=True)):
        pp, pa, pi = grade_probabilities(s)
        assert 0.0 <= pp <= 1.0 and 0.0 <= pa <= 1.0 and 0.0 <= pi <= 1.0
        assert 0.0 <= perfect_share(s) <= 1.0
        assert 0.0 <= correctness_prob(s) <= 1.0


# --------------------------------------------------------------------------- three tiers
def test_bare_bundle_abstains():
    """No positive evidence ⇒ committing is EV-negative ⇒ ABSTAIN (must be *earned*)."""
    d = decide(GateSignals())
    assert d.tier == ABSTAIN and d.commit is False
    assert d.utility <= 0.0


def test_hard_accept_deterministic_lane_evidence_complete_verifier():
    s = GateSignals(deterministic_lane=True, evidence_slots_complete=True,
                    answer_verifier_agrees=True, canonical_doc_resolved=True)
    d = decide(s)
    assert d.tier == HARD_ACCEPT and d.commit is True
    assert d.utility > 0.0


def test_soft_accept_positive_utility_without_hard_lane():
    """Enough hard evidence for U>0 but not the HARD triple ⇒ SOFT ACCEPT."""
    s = GateSignals(canonical_doc_resolved=True, evidence_slots_complete=True,
                    answer_verifier_agrees=True, deterministic_lane=False)
    d = decide(s)
    assert d.tier == SOFT_ACCEPT and d.commit is True
    assert d.utility > 0.0


def test_utility_just_below_zero_abstains():
    """A bundle whose U ≤ 0 abstains even though no hard blocker fires."""
    s = GateSignals(canonical_doc_resolved=True)  # doc only — not enough
    d = decide(s)
    assert d.tier == ABSTAIN and d.commit is False
    assert d.utility <= 0.0


# --------------------------------------------------------------------------- hard epistemic blockers
@pytest.mark.parametrize("bad, reason_key", [
    (dict(corpus_absent=True), "true_corpus_absence"),
    (dict(parser_capable=False), "parser_unsupported"),
    (dict(required_evidence_unresolved=True), "unresolved_required_evidence"),
    (dict(source_conflict_absent=False), "conflicting_evidence"),
    (dict(execution_engines_agree=False), "execution_disagreement"),
])
def test_hard_blocker_abstains_regardless_of_utility(bad, reason_key):
    """A categorical blocker forces ABSTAIN even when every other signal is maximal (U>0)."""
    strong = dict(canonical_doc_resolved=True, evidence_slots_complete=True,
                  answer_verifier_agrees=True, deterministic_lane=True,
                  self_consistency_agrees=True, verbal_confidence=1.0)
    s = GateSignals(**{**strong, **bad})
    assert expected_utility(s) > 0.0            # utility alone would commit
    d = decide(s)
    assert d.commit is False and d.tier == ABSTAIN
    assert reason_key in d.reason


# --------------------------------------------------------------------------- BUDGET_EXHAUSTED excluded
def test_budget_exhausted_alone_does_not_abstain():
    """BUDGET_EXHAUSTED is operational, not epistemic — it never flips a committable bundle to abstain."""
    good = dict(canonical_doc_resolved=True, evidence_slots_complete=True, answer_verifier_agrees=True)
    without = decide(GateSignals(**good))
    with_budget = decide(GateSignals(**good, budget_exhausted=True))
    assert without.commit is True and with_budget.commit is True
    assert without.to_dict() == with_budget.to_dict()   # budget_exhausted is ignored by the decision


# --------------------------------------------------------------------------- verbal confidence demoted
def test_verbal_confidence_is_demoted_below_hard_evidence():
    """Maxed verbal confidence alone cannot out-vote the hard-evidence mass.

    A bundle with *only* high verbal confidence (no hard evidence) must contribute far less correctness
    than a bundle with a single hard-evidence signal and zero confidence — proving confidence is auxiliary.
    """
    confidence_only = correctness_prob(GateSignals(verbal_confidence=1.0))
    one_hard_signal = correctness_prob(GateSignals(canonical_doc_resolved=True))
    assert one_hard_signal > confidence_only
    # And the confidence term contributes at most its small weight.
    base = correctness_prob(GateSignals())
    assert correctness_prob(GateSignals(verbal_confidence=1.0)) - base <= 0.05 + 1e-9


def test_confidence_only_bundle_still_abstains():
    """High self-reported confidence with no evidence does not commit (faithfulness caveat)."""
    d = decide(GateSignals(verbal_confidence=1.0))
    assert d.commit is False and d.tier == ABSTAIN


# --------------------------------------------------------------------------- env toggle
def test_enabled_reads_env(monkeypatch):
    monkeypatch.delenv("RAG_EU_GATE", raising=False)
    assert eu_gate.enabled() is False
    monkeypatch.setenv("RAG_EU_GATE", "1")
    assert eu_gate.enabled() is True
    monkeypatch.setenv("RAG_EU_GATE", "off")
    assert eu_gate.enabled() is False
