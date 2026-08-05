"""Network-free regressions for the submission ledger and judge-fidelity metrics."""
import json

import pytest

from scoring import calibrate, judge_fidelity, ledger


def test_record_prediction_and_set_actual_round_trip(tmp_path):
    path = tmp_path / "ledger.jsonl"
    prediction = {"estimated_real_public_score": -0.1234,
                  "confidence_interval_95": [-0.29, 0.05], "local_score": 0.9,
                  "archetype_committed": {"fact_lookup": 3, "enum_set": 2}}
    row = ledger.record_prediction("#6", prediction, date="2026-08-05",
                                   config="strict", commit="abc1234", path=path)
    # Ex-ante half is populated; the realised half is pending.
    assert row["predicted"] == -0.1234
    assert row["ci_95"] == [-0.29, 0.05]
    assert row["real_public_score"] is None and row["absolute_error"] is None
    assert row["archetype_committed"] == {"fact_lookup": 3, "enum_set": 2}

    filled = ledger.set_actual("#6", -0.1, path=path)
    assert filled["real_public_score"] == -0.1
    assert filled["absolute_error"] == pytest.approx(abs(-0.1234 - -0.1), abs=1e-4)

    # Re-recording the same submission upserts rather than duplicating.
    ledger.record_prediction("#6", prediction, date="2026-08-05", real_public_score=-0.1, path=path)
    assert sum(r["submission"] == "#6" for r in ledger.load(path)) == 1


def test_pending_rows_are_skipped_by_calibration(tmp_path):
    path = tmp_path / "ledger.jsonl"
    scored = [
        {"submission": "#1", "local_score": 0.0, "real_public_score": -0.2,
         "archetype_committed": {"unknown": 10}},
        {"submission": "#2", "local_score": 0.3, "real_public_score": 0.0,
         "archetype_committed": {"unknown": 8}},
    ]
    pending = {"submission": "#3", "local_score": 0.5, "real_public_score": None,
               "archetype_committed": {"unknown": 5}}
    path.write_text("".join(json.dumps(r) + "\n" for r in scored + [pending]), encoding="utf-8")
    assert len(calibrate.load_ledger(path)) == 2                     # pending skipped
    assert len(calibrate.load_ledger(path, scored_only=False)) == 3  # full ledger


def test_real_ledger_is_complete_and_loo_mae_not_worse():
    """Every real submission in the committed ledger has predicted/actual/error, and the
    calibration leave-one-out MAE is no worse than the pre-completion n=4 baseline (0.0848)."""
    rows = calibrate.load_ledger()
    assert len(rows) >= 5  # the ledger only grows as submissions are made
    for r in rows:
        assert r["real_public_score"] is not None
        assert r["predicted"] is not None and r["absolute_error"] is not None
        assert "archetype_committed" in r and r["archetype_committed"]
    report = judge_fidelity.ledger_fidelity_report()
    assert report["n_submissions"] == len(rows)
    assert report["loo_mae"] <= 0.0848  # must not regress vs the recorded n=4 baseline


def test_strict_rubric_narrows_leniency_gap():
    report = judge_fidelity.overjudge_report()
    # Every idx17/23/28-type over-judgement is pulled off "Perfect" by the strict rubric...
    assert report["all_pulled_off_perfect"]
    # ...and the mean points reduction more than covers the ~0.2 Gemini leniency gap.
    assert report["mean_leniency_reduction"] >= 0.2


def test_hand_labelled_agreement_not_regressed():
    from scoring import crag
    report = judge_fidelity.evaluate(crag.judge)
    assert report["accuracy"] == pytest.approx(1.0)  # deterministic cases all agree
