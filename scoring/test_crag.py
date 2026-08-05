"""Network-free regressions for the hybrid CRAG judge."""
from scoring import crag
from scoring import deterministic as D


def test_official_rounding_examples():
    assert D.score_numeric("12.5", "12.47") == "Acceptable"
    assert D.score_numeric("12.5", "12.45") == "Acceptable"
    assert D.score_numeric("12.4", "12.47") == "Incorrect"


def test_known_misjudgements_are_deterministic():
    assert crag.deterministic_judge("Recall", "Recall") == "Perfect"
    assert crag.deterministic_judge("100円（税込）", "100円（税抜）") == "Incorrect"
    assert crag.deterministic_judge("20", "20日") == "Perfect"


def test_enumeration_is_order_independent_and_complete():
    assert crag.deterministic_judge("A、B、C", "C、A、B") == "Perfect"
    assert crag.deterministic_judge("A、B", "A、B、C") == "Incorrect"


def test_deterministic_route_never_calls_llm(monkeypatch):
    monkeypatch.setattr(crag, "_judge_gemini", lambda *_a, **_k: (_ for _ in ()).throw(AssertionError("LLM called")))
    assert crag.judge("Recall", "Recall") == "Perfect"


def test_strict_free_text_uses_worst_case_not_majority(monkeypatch):
    # Two "Perfect" and one "Incorrect": strict worst-case aggregation returns Incorrect,
    # whereas plain majority would have returned Perfect. This is the leniency-narrowing lever.
    calls = iter(["Incorrect", "Perfect", "Perfect"])
    monkeypatch.setattr(crag, "_judge_gemini", lambda *_a, **_k: next(calls))
    assert crag.judge("主要な内容を自然文で説明しています。",
                      "同じ意味を別の自然文で説明しています。", strict=True) == "Incorrect"


def test_nonstrict_falls_back_to_majority_three(monkeypatch):
    calls = iter(["Incorrect", "Perfect", "Perfect"])
    monkeypatch.setattr(crag, "_judge_gemini", lambda *_a, **_k: next(calls))
    assert crag.judge("主要な内容を自然文で説明しています。",
                      "同じ意味を別の自然文で説明しています。", strict=False) == "Perfect"


def test_strict_downgrade_rejects_verbose_extra_numeric():
    # idx17-type: short ground truth, padded answer inventing an unconfirmable count → not Perfect.
    verdict = crag.strict_downgrade(
        "pdaysの値-1は、これまで一度も連絡実績がなく未連絡であり、該当件数は約40件です。",
        "未連絡", "Perfect")
    assert verdict == "Incorrect"


def test_strict_downgrade_rejects_verbose_paraphrase():
    # idx28-type: over-long paraphrase without new numeric specifics → Acceptable, not Perfect.
    truth = ("object、string、categoricaldtype の列を候補とし、欠損を除いたユニーク数が50未満なら"
             "カテゴリ特徴量として採用している。")
    pred = ("この分析コードでは、データ型がobject、string、categoricaldtypeのいずれかである列を候補として"
            "抽出し、さらに欠損値を除外した上でユニークな値の数が50未満である場合に限り、カテゴリ特徴量として"
            "自動的に採用する実装になっています。")
    assert crag.strict_downgrade(pred, truth, "Perfect") == "Acceptable"


def test_strict_downgrade_never_upgrades_or_touches_clean_answers():
    assert crag.strict_downgrade("Recall", "Recall", "Perfect") == "Perfect"      # clean short
    assert crag.strict_downgrade("x" * 80, "y", "Incorrect") == "Incorrect"       # never upgrades
    assert crag.strict_downgrade("わかりません", "Recall", "Missing") == "Missing"


def test_strict_downgrade_applied_even_when_llm_says_perfect(monkeypatch):
    # Full judge path: LLM lenient "Perfect" on a verbose+extra-numeric answer is still not Perfect.
    monkeypatch.setattr(crag, "deterministic_judge", lambda *_a, **_k: None)
    monkeypatch.setattr(crag, "_judge_gemini", lambda *_a, **_k: "Perfect")
    verdict = crag.judge(
        "pdaysの値-1は、これまで一度も連絡実績がなく未連絡であり、該当件数は約40件です。",
        "未連絡", strict=True)
    assert verdict != "Perfect"
