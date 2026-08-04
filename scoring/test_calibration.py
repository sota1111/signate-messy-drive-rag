import json

import pytest

from scoring import calibrate, predict


def _rows():
    return [
        {"submission": "a", "local_score": 0.0, "real_public_score": -0.2,
         "archetype_committed": {"unknown": 10}},
        {"submission": "b", "local_score": 0.2, "real_public_score": -0.05,
         "archetype_committed": {"metric_score": 5, "unknown": 5}},
        {"submission": "c", "local_score": 0.5, "real_public_score": 0.1,
         "archetype_committed": {"metric_score": 10}},
    ]


def test_calibration_reports_kpi_and_backtest():
    model = calibrate.calibrate(_rows())
    assert model["proxy_real_spearman"] == pytest.approx(1.0)
    assert len(model["backtest"]) == 3
    assert len(model["leave_one_out"]) == 3
    assert model["interval_half_width"] >= 0.05


def test_ledger_validation(tmp_path):
    path = tmp_path / "ledger.jsonl"
    path.write_text(json.dumps(_rows()[0]) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="at least two"):
        calibrate.load_ledger(path)


def test_answer_summary_and_prediction_interval():
    rows = [{"question": "accuracyのスコアは", "answer": "0.9"},
            {"question": "不明", "answer": "該当なし"},
            {"question": "何か", "answer": "回答", "archetype": "custom"}]
    counts = predict.summarize_answers(rows)
    assert counts == {"metric_score": 1, "custom": 1}
    result = predict.predict(calibrate.calibrate(_rows()), 0.3, counts)
    assert result["committed"] == 2
    assert result["confidence_interval_95"][0] <= result["estimated_real_public_score"]
    assert result["estimated_real_public_score"] <= result["confidence_interval_95"][1]


def test_read_headerless_submission_reattaches_questions(tmp_path, monkeypatch):
    answers = tmp_path / "predictions.csv"
    answers.write_text("7,0.9\n8,わかりません\n", encoding="utf-8")
    questions = tmp_path / "questions_test.csv"
    questions.write_text("index,question\n7,accuracyのスコアは\n8,別の質問\n", encoding="utf-8")
    monkeypatch.setattr(predict.settings, "QUESTIONS_TEST", questions)
    rows = predict.read_answers(answers)
    assert rows[0]["question"] == "accuracyのスコアは"
    assert predict.summarize_answers(rows) == {"metric_score": 1}
