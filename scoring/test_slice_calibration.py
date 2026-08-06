"""SOT-2503 — offline tests for the per-contract-slice gate EV calibration.

Network-free: :func:`scoring.slice_calibration.calibrate_slices` is a pure function over constructed
records. Covers the adoption rule (relaxation / EV>0 / WRONG非増加 / judge-noise robustness), the
WRONG-total non-increase invariant (受け入れ条件①), and the record builders.
"""
from __future__ import annotations

import math

from scoring import slice_calibration as sc
from src.rag.agent import question_contract as qc


def _rec(contract: str, confidence: float, verdict: str, index: int = 0) -> dict:
    from scoring.calibrate import points_for_verdict
    return {"index": index, "contract": contract, "confidence": confidence,
            "verdict": verdict, "points": points_for_verdict(verdict)}


# --------------------------------------------------------------------------- adoption: EV>0 relaxation
def test_positive_ev_slice_is_adopted_and_relaxed():
    # 6 correct agreements all just below the global 0.7 bar → today abstained, but all Perfect: a
    # clear EV>0 relaxation with zero new WRONG.
    records = [_rec("simple_lookup", 0.6, "Perfect", i) for i in range(6)]
    model = sc.calibrate_slices(records, baseline_threshold=0.7)
    s = model["slices"][0]
    assert s["contract"] == "simple_lookup"
    assert s["adopt"] is True
    assert s["chosen_threshold"] == 0.6            # relaxed below the global bar
    assert s["baseline"]["committed"] == 0 and s["relaxed"]["committed"] == 6
    assert s["added_wrong"] == 0
    assert model["adopted_thresholds"] == {"simple_lookup": 0.6}
    assert sc.to_gate_thresholds(model) == {"simple_lookup": 0.6}


# --------------------------------------------------------------------------- rejection: relaxation adds WRONG
def test_slice_rejected_when_relaxation_adds_incorrect():
    # Lowering the bar would commit 3 Perfect but ALSO 2 Incorrect → WRONG rises → must NOT adopt.
    records = ([_rec("chart_read", 0.6, "Perfect", i) for i in range(3)] +
               [_rec("chart_read", 0.6, "Incorrect", 10 + i) for i in range(2)])
    model = sc.calibrate_slices(records, baseline_threshold=0.7)
    s = model["slices"][0]
    assert s["adopt"] is False
    assert "誤答増加" in s["reason"]
    assert model["adopted_thresholds"] == {}
    # WRONG total must be non-increasing: the slice keeps the global bar (commits nothing here).
    assert model["wrong_total_relaxed"] <= model["wrong_total_baseline"]
    assert model["wrong_nonincreasing"] is True


# --------------------------------------------------------------------------- rejection: no relaxation
def test_slice_not_relaxed_when_ev_max_is_at_or_above_baseline():
    # Everything already commits above the bar → the EV-max threshold is not a relaxation → keep.
    records = [_rec("numeric", 0.9, "Perfect", i) for i in range(4)]
    model = sc.calibrate_slices(records, baseline_threshold=0.7)
    s = model["slices"][0]
    assert s["adopt"] is False
    assert s["chosen_threshold"] == 0.7           # unchanged (global baseline)
    assert "緩和対象外" in s["reason"]


# --------------------------------------------------------------------------- rejection: judge-noise fragile
def test_slice_rejected_when_positive_ev_is_within_judge_noise():
    # 2 Perfect just below the bar: raw EV +2, but a single ≈1/30 judge flip (−2) wipes it out.
    records = [_rec("format_check", 0.6, "Perfect", i) for i in range(2)]
    model = sc.calibrate_slices(records, baseline_threshold=0.7, judge_noise=1 / 30)
    s = model["slices"][0]
    assert s["relaxed"]["ev"] == 2.0
    assert s["robust_ev"] == 0.0                  # 2 - 2*ceil(2/30) = 0
    assert s["adopt"] is False
    assert "判定ゆらぎ" in s["reason"]


# --------------------------------------------------------------------------- WRONG-total invariant (mixed)
def test_wrong_total_non_increasing_across_mixed_slices():
    records = (
        # adoptable: 5 Perfect below the bar
        [_rec("simple_lookup", 0.6, "Perfect", i) for i in range(5)] +
        # not adoptable: adding commits would add an Incorrect
        [_rec("chart_read", 0.6, "Perfect", 20), _rec("chart_read", 0.6, "Incorrect", 21)] +
        # already-committed correct answers above the bar (baseline wrong 0)
        [_rec("version_diff", 0.9, "Perfect", 30)]
    )
    model = sc.calibrate_slices(records, baseline_threshold=0.7)
    assert model["wrong_nonincreasing"] is True
    assert model["wrong_total_relaxed"] <= model["wrong_total_baseline"]
    # Only the safe slice is relaxed.
    assert set(model["adopted_thresholds"]) == {"simple_lookup"}
    # EV strictly improves by exactly the recovered correct commits.
    assert model["ev_gain"] == 5.0


# --------------------------------------------------------------------------- Acceptable counts as correct
def test_acceptable_verdict_counts_toward_ev_and_correct():
    records = [_rec("multi_hop", 0.6, "Acceptable", i) for i in range(6)]
    model = sc.calibrate_slices(records, baseline_threshold=0.7)
    s = model["slices"][0]
    assert s["relaxed"]["correct"] == 6 and s["relaxed"]["ev"] == 3.0   # 6 * 0.5
    assert s["adopt"] is True


# --------------------------------------------------------------------------- empty slice / no records
def test_empty_records_yield_no_adoption():
    model = sc.calibrate_slices([])
    assert model["n"] == 0 and model["n_slices"] == 0
    assert model["adopted_thresholds"] == {}
    assert model["wrong_nonincreasing"] is True


# --------------------------------------------------------------------------- record builder: contract attach
def test_records_from_report_json_attaches_contract():
    report_json = {
        "n": 3,
        "verdicts": {"Perfect": 1, "Missing": 1, "Incorrect": 1},
        "wrong_items": [{"index": 2}],
        "abstain_items": [{"index": 1}],
    }
    details = {
        0: {"index": 0, "confidence": 0.8, "question": "山田さんの隣に座っているのは誰ですか"},
        1: {"index": 1, "confidence": 0.5, "question": "train.csv の平均値は"},
        2: {"index": 2, "confidence": 0.9, "question": "グラフ1の最大の系列は"},
    }
    records = sc.records_from_report_json(report_json, details)
    assert len(records) == 3
    by_index = {r["index"]: r for r in records}
    assert by_index[0]["contract"] == qc.SPATIAL          # 隣に座 → spatial cue
    assert by_index[2]["contract"] == qc.CHART_READ       # グラフ → chart cue
    for r in records:
        assert r["contract"] in qc.CONTRACTS
        assert "confidence" in r and "points" in r


# --------------------------------------------------------------------------- record builder: live report
def test_slice_records_from_report_skips_missing_confidence():
    class _Item:
        def __init__(self, index, question, verdict):
            self.index, self.question, self.verdict = index, question, verdict

    class _Report:
        items = [
            _Item(0, "会社Aの着手金はいくらですか", "Perfect"),
            _Item(1, "会社Bの報酬は", "Incorrect"),
        ]

    details = {0: {"confidence": 0.8}, 1: {}}   # index 1 has no confidence → skipped
    records = sc.slice_records_from_report(_Report(), details)
    assert [r["index"] for r in records] == [0]
    assert records[0]["contract"] in qc.CONTRACTS
