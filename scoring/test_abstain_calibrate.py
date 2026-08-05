"""Network-free regressions for the gold-EV abstention-threshold calibration (SOT-2474).

All tests are deterministic: the EV sweep / LOO / record-building logic is exercised over
constructed records and a stub gold_offline Report, so no LLM / network is touched.
"""
import json

import pytest

from scoring import calibrate as C
from scoring import gold_offline as GO


def _rec(conf, verdict):
    return {"confidence": conf, "verdict": verdict, "points": C.points_for_verdict(verdict)}


def test_points_for_verdict_matches_official_rubric():
    assert C.points_for_verdict("Perfect") == 1.0
    assert C.points_for_verdict("Acceptable") == 0.5
    assert C.points_for_verdict("Missing") == 0.0
    assert C.points_for_verdict("Incorrect") == -1.0
    assert C.points_for_verdict("something-unknown") == 0.0  # conservative


def test_coerce_confidence_numeric_label_and_missing():
    assert C.coerce_confidence(0.8) == 0.8
    assert C.coerce_confidence(1.7) == 1.0          # clamped
    assert C.coerce_confidence(-0.3) == 0.0         # clamped
    assert C.coerce_confidence("0.42") == 0.42      # numeric string
    assert C.coerce_confidence("high") == 0.9       # known label
    assert C.coerce_confidence("weird") == C.DEFAULT_LABEL_CONFIDENCE  # unknown label
    assert C.coerce_confidence("") is None          # empty -> skip
    assert C.coerce_confidence(None) is None
    assert C.coerce_confidence(True) is None        # bool is not a confidence


def test_ev_max_threshold_when_confidence_separates():
    # Correct answers span 0.6–0.9, wrong ones sit at 0.3. The default gate (0.7) is too strict —
    # it abstains the correct 0.6 answer — so the EV-max threshold (0.6) beats it. Confidence
    # separates cleanly, so the calibrator recommends adopting the lower threshold.
    records = [_rec(0.9, "Perfect"), _rec(0.9, "Perfect"), _rec(0.9, "Perfect"),
               _rec(0.6, "Perfect"), _rec(0.3, "Incorrect"), _rec(0.3, "Incorrect")]
    model = C.calibrate_abstention(records, baseline_threshold=0.7)
    # commit-all EV = 4*(+1) + 2*(-1) = +2 ; abstain-all = 0 ; best commits the 4 Perfects = +4.
    assert model["commit_all_ev"] == 2.0
    assert model["chosen_ev"] == 4.0
    assert model["chosen_threshold"] == pytest.approx(0.6)
    assert model["chosen_committed"] == 4
    assert model["chosen_precision"] == 1.0
    assert model["signal_separates"] is True
    # default gate 0.7 only commits the 3 high-conf Perfects (EV +3) -> calibration improves on it.
    assert model["baseline_ev"] == 3.0
    assert model["improvement_over_baseline"] == 1.0
    assert model["improves_over_baseline"] is True
    assert model["adopt"] is True


def test_non_separating_signal_recommends_abstain_all_but_not_adopt():
    # Confidence does not separate: equal high-confidence Perfect and Incorrect. EV-max = abstain-all.
    records = [_rec(0.9, "Perfect")] * 2 + [_rec(0.9, "Incorrect")] * 3
    model = C.calibrate_abstention(records, baseline_threshold=0.0)  # baseline = commit-all
    assert model["commit_all_ev"] == -1.0            # 2*(+1) + 3*(-1)
    assert model["chosen_ev"] == 0.0                 # abstain everything
    assert model["chosen_committed"] == 0
    assert model["signal_separates"] is False
    # Improves over the commit-all baseline (0 > -1) but must NOT recommend adopting a live change,
    # because the signal is degenerate (abstain-all).
    assert model["improves_over_baseline"] is True
    assert model["adopt"] is False


def test_tie_break_prefers_abstention():
    # Two Perfects at different confidences: committing either subset yields the same +? but the
    # conservative tie-break prefers the higher threshold (fewer commits) at equal EV.
    records = [_rec(0.6, "Missing"), _rec(0.9, "Missing")]  # all zero-point -> every EV is 0
    model = C.calibrate_abstention(records)
    assert model["chosen_ev"] == 0.0
    # highest candidate threshold -> commit nothing
    assert model["chosen_committed"] == 0


def test_leave_one_out_generalisation_reported():
    records = [_rec(0.9, "Perfect"), _rec(0.9, "Perfect"), _rec(0.8, "Perfect"),
               _rec(0.2, "Incorrect"), _rec(0.3, "Incorrect")]
    model = C.calibrate_abstention(records)
    loo = model["leave_one_out"]
    assert loo["n"] == 5
    assert loo["loo_ev"] is not None
    # A well-separating set should generalise: LOO EV stays positive.
    assert loo["loo_ev"] > 0


def test_empty_records_raise():
    with pytest.raises(ValueError, match="no records"):
        C.calibrate_abstention([])


def _stub_report(verdict_by_index):
    items = [GO.Item(index=i, question="q", answer="a", gold="g", archetype="unknown",
                     verdict=v) for i, v in verdict_by_index.items()]
    return GO.Report(items=items)


def test_records_from_report_joins_confidence_and_skips_missing_conf():
    report = _stub_report({0: "Perfect", 1: "Incorrect", 2: "Missing"})
    details = {0: {"confidence": 0.9}, 1: {"confidence": "low"}, 2: {"confidence": ""}}
    records = C.records_from_report(report, details)
    # index 2 has empty confidence -> skipped
    assert [r["index"] for r in records] == [0, 1]
    assert records[0]["points"] == 1.0 and records[0]["confidence"] == 0.9
    assert records[1]["points"] == -1.0 and records[1]["confidence"] == 0.2  # "low" label


def test_records_from_report_json_reconstructs_verdicts_offline():
    report_json = {
        "n": 4,
        "verdicts": {"Perfect": 1, "Incorrect": 1, "Missing": 2},
        "wrong_items": [{"index": 1}],
        "abstain_items": [{"index": 2}, {"index": 3}],
    }
    details = {0: {"confidence": 0.95}, 1: {"confidence": 0.9},
               2: {"confidence": 0.0}, 3: {"confidence": 0.1}}
    records = C.records_from_report_json(report_json, details)
    by_idx = {r["index"]: r for r in records}
    assert by_idx[0]["verdict"] == "Perfect" and by_idx[0]["points"] == 1.0
    assert by_idx[1]["verdict"] == "Incorrect" and by_idx[1]["points"] == -1.0
    assert by_idx[2]["verdict"] == "Missing" and by_idx[3]["verdict"] == "Missing"


def test_records_from_report_json_rejects_ambiguous_acceptable():
    report_json = {"n": 2, "verdicts": {"Acceptable": 1, "Missing": 1},
                   "wrong_items": [], "abstain_items": [{"index": 1}]}
    with pytest.raises(ValueError, match="Acceptable"):
        C.records_from_report_json(report_json, {0: {"confidence": 0.9}})


def test_load_records_jsonl(tmp_path):
    p = tmp_path / "records.jsonl"
    p.write_text(
        json.dumps({"confidence": 0.9, "verdict": "Perfect"}) + "\n"
        + json.dumps({"confidence": "low", "verdict": "Incorrect"}) + "\n"
        + json.dumps({"confidence": 0.5, "points": 0.5}) + "\n"
        + json.dumps({"confidence": "", "verdict": "Missing"}) + "\n",  # skipped
        encoding="utf-8")
    records = C.load_records(p)
    assert len(records) == 3
    assert records[0]["points"] == 1.0
    assert records[1]["confidence"] == 0.2
    assert records[2]["points"] == 0.5


def test_load_records_requires_verdict_or_points(tmp_path):
    p = tmp_path / "bad.jsonl"
    p.write_text(json.dumps({"confidence": 0.9}) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="verdict.*points"):
        C.load_records(p)


def test_save_abstain_model_roundtrip(tmp_path):
    records = [_rec(0.9, "Perfect"), _rec(0.2, "Incorrect")]
    model = C.calibrate_abstention(records)
    out = tmp_path / "abstain.json"
    C.save_abstain_model(model, out)
    loaded = json.loads(out.read_text(encoding="utf-8"))
    assert loaded["kind"] == "abstention_threshold"
    assert loaded["chosen_threshold"] == model["chosen_threshold"]


def test_legacy_ledger_calibration_untouched():
    # The bare (ledger→public-score) calibration must be unchanged by the new abstention code.
    rows = [
        {"submission": "a", "local_score": 0.0, "real_public_score": -0.2,
         "archetype_committed": {"unknown": 10}},
        {"submission": "b", "local_score": 0.5, "real_public_score": 0.1,
         "archetype_committed": {"metric_score": 10}},
    ]
    model = C.calibrate(rows)
    assert model["n_submissions"] == 2
    assert "backtest" in model
