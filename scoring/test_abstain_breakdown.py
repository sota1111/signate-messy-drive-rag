"""Network-free regressions for the gold abstention-reason breakdown (SOT-2483).

All deterministic: :func:`classify_record` / :func:`breakdown` are pure over constructed resolve
records, and :func:`recovery_records` is exercised with a stub judge so no LLM / network is touched.
"""
from __future__ import annotations

import json

import pytest

from scoring import abstain_breakdown as AB


def _rec(index, *, answer="わかりません", inv=None, ver=None, jud=None, jconf=None,
         status="abstained", category="", agree=None, archetype="fact_lookup",
         verifier_confidence=None):
    """Build a resolve details record. ``inv``/``ver``/``jud`` are proposed values (None → abstain)."""
    verdict = {"category": category}
    if verifier_confidence is not None:
        verdict["verifier_confidence"] = verifier_confidence
    return {
        "index": index, "answer": answer,
        "investigator_answer": inv if inv is not None else "わかりません",
        "verifier_answer": ver if ver is not None else "わかりません",
        "judge_answer": jud if jud is not None else "わかりません",
        "judge_confidence": jconf, "status": status, "agree": agree,
        "archetype": archetype, "verdict": verdict,
    }


# --------------------------------------------------------------------------- classification
def test_committed_record_is_not_classified():
    assert AB.classify_record(_rec(0, answer="42ページ")) is None


def test_both_abstain_is_unrecoverable():
    a = AB.classify_record(_rec(1, inv=None, ver=None))
    assert a.reason == AB.BOTH_ABSTAIN
    assert a.recoverable is False
    assert a.candidate is None and a.candidate_source == ""


def test_one_side_abstain_recovers_the_proposing_side():
    # verifier proposed, investigator abstained, no judge value → candidate is the verifier's.
    a = AB.classify_record(_rec(2, inv=None, ver="東京", verifier_confidence=0.8))
    assert a.reason == AB.ONE_SIDE_ABSTAIN
    assert a.recoverable is True
    assert a.candidate == "東京" and a.candidate_source == "verifier"
    assert a.confidence == 0.8


def test_judge_adjudication_is_preferred_candidate():
    # Both sides disagreed; the third judge left a value at high confidence → that is the candidate.
    a = AB.classify_record(_rec(3, inv="A社", ver="B社", jud="A社", jconf=1.0, agree=False))
    assert a.reason == AB.VERIFIER_DISAGREEMENT
    assert a.recoverable is True
    assert a.candidate == "A社" and a.candidate_source == "judge"
    assert a.confidence == 1.0


def test_tiebreak_unable_zero_conf_judge_is_not_a_candidate():
    # judge_confidence 0.0 means the third judge did NOT adjudicate a usable value; fall back to a
    # proposing side if one exists, else unrecoverable.
    a = AB.classify_record(_rec(4, inv="X", ver="Y", jud="Z", jconf=0.0, agree=False))
    assert a.candidate_source != "judge"
    assert a.recoverable is True  # verifier value survives as the fallback candidate


def test_enumeration_mismatch_reason():
    a = AB.classify_record(
        _rec(5, inv="a,b", ver="a,b,c", jud="a,b,c", jconf=1.0, category="enumeration", agree=False))
    assert a.reason == AB.ENUMERATION_MISMATCH
    assert a.recoverable is True


def test_abstain_sentinel_and_empty_both_count_as_abstained():
    assert AB.is_abstained({"answer": "わかりません"})
    assert AB.is_abstained({"answer": ""})
    assert AB.is_abstained({"answer": "   "})
    assert not AB.is_abstained({"answer": "該当なし"})  # a real "no such data" answer


# --------------------------------------------------------------------------- breakdown aggregation
def test_breakdown_counts_and_split():
    records = [
        _rec(0, answer="42"),                                   # committed
        _rec(1, inv=None, ver=None),                            # both_abstain (unrecoverable)
        _rec(2, inv=None, ver="東京", verifier_confidence=0.8),  # one_side_abstain (recoverable)
        _rec(3, inv="A", ver="B", jud="A", jconf=1.0, agree=False),  # disagreement (recoverable)
    ]
    rep = AB.breakdown(records)
    assert rep["n_total"] == 4
    assert rep["n_committed"] == 1
    assert rep["n_abstained"] == 3
    assert rep["n_recoverable"] == 2
    assert rep["n_unrecoverable"] == 1
    assert rep["by_reason"][AB.BOTH_ABSTAIN] == 1
    assert rep["recoverable_by_reason"][AB.ONE_SIDE_ABSTAIN] == 1
    assert rep["candidate_source_counts"] == {"judge": 1, "verifier": 1}
    assert [it["index"] for it in rep["recoverable_items"]] == [2, 3]


# --------------------------------------------------------------------------- recovery EV (stub judge)
def test_recovery_records_join_candidates_with_gold_via_stub_judge():
    records = [
        _rec(1, inv=None, ver=None),                            # unrecoverable → skipped
        _rec(2, inv=None, ver="東京", verifier_confidence=0.9),  # recoverable
        _rec(3, inv="A", ver="B", jud="正解", jconf=1.0, agree=False),  # recoverable
    ]
    gold = {2: "東京", 3: "正解"}
    # Stub judge: Perfect iff the candidate equals gold, else Incorrect.
    def judge(pairs):
        return ["Perfect" if p == g else "Incorrect" for p, g in pairs]

    recs = AB.recovery_records(records, gold, judge=judge)
    assert {r["index"] for r in recs} == {2, 3}
    assert all(r["verdict"] == "Perfect" for r in recs)
    assert all(r["points"] == 1.0 for r in recs)


def test_recovery_ev_marginal_gain_and_negative_when_wrong():
    # Two recovered candidates: one Perfect (+1), one Incorrect (−1) → commit-all EV = 0.
    records = [
        _rec(2, inv=None, ver="良", verifier_confidence=0.9),
        _rec(3, inv=None, ver="悪", verifier_confidence=0.9),
    ]
    gold = {2: "良", 3: "正"}
    def judge(pairs):
        return ["Perfect" if p == g else "Incorrect" for p, g in pairs]

    recs = AB.recovery_records(records, gold, judge=judge)
    ev = AB.recovery_ev(recs)
    assert ev["n_recoverable_judged"] == 2
    assert ev["commit_all_recovery_ev"] == 0.0
    assert ev["recovery_verdict_counts"] == {"Incorrect": 1, "Perfect": 1}
    assert ev["calibrated_recovery"] is not None


def test_recovery_records_raises_on_judge_length_mismatch():
    records = [_rec(2, inv=None, ver="x", verifier_confidence=0.9)]
    with pytest.raises(ValueError):
        AB.recovery_records(records, {2: "x"}, judge=lambda pairs: [])


# --------------------------------------------------------------------------- CLI smoke (deterministic)
def test_cli_writes_report(tmp_path):
    details = tmp_path / "d.jsonl"
    details.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in [
        _rec(0, answer="42"), _rec(1, inv=None, ver=None),
        _rec(2, inv=None, ver="東京", verifier_confidence=0.8),
    ]), encoding="utf-8")
    out = tmp_path / "rep.json"
    rc = AB.main(["--details", str(details), "--out", str(out)])
    assert rc == 0
    rep = json.loads(out.read_text(encoding="utf-8"))
    assert rep["n_abstained"] == 2 and rep["n_recoverable"] == 1
