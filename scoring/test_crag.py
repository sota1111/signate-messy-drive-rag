"""Network-free regressions for the hybrid CRAG judge."""
import pytest

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


def test_structured_notation_equivalence_is_perfect_without_llm():
    # SOT-2505 idx61: Japanese copula and assignment notation are semantically identical.
    assert crag.deterministic_judge(
        "n_estimatorsは300、learning_rateは0.1、random_stateは42です。",
        "n_estimators=300、learning_rate=0.1、random_state=42",
    ) == "Perfect"


def test_structured_gold_containment_is_acceptable_without_llm():
    # SOT-2505 idx62: the answer contains every ranked fact and only adds explanation.
    assert crag.deterministic_judge(
        "上位2つのextra_treesモデルのスコア差を生んでいる設定差分は、n_estimators（決定木の数）です。"
        "1位のモデルは500、2位のモデルは300に設定されています。",
        "n_estimators（1位=500、2位=300）",
    ) == "Acceptable"


def test_subject_particle_ga_reads_as_symbolic_assignment():
    # SOT-2544 idx62: the real answer uses が ("1位のモデルが500"), not は — the same ranked
    # facts as the symbolic gold, only with explanation. Must resolve without the LLM.
    assert crag.deterministic_judge(
        "上位2モデルの設定差分は n_estimators の値です。"
        "1位のモデルが500、2位のモデルが300に設定されています。",
        "n_estimators（1位=500、2位=300）",
    ) == "Acceptable"


def test_empty_set_paraphrase_matches_gold_none():
    # SOT-2544 idx85: gold "該当なし"; the answer concludes the same nonexistence in prose.
    assert crag.deterministic_judge(
        "青葉バイオメディカル機器の最終報告書によると、設定された6項目のKPIはすべて"
        "「達成」と評価されており、未達成とされている項目はありません。",
        "該当なし",
    ) == "Acceptable"


@pytest.mark.parametrize(("pred", "truth"), [
    # An answer that enumerates concrete items for a 該当なし gold is a real wrong answer.
    ("未達成項目は、売上目標とコスト削減の2項目です。", "該当なし"),
    # A positive existence claim is never an empty-set paraphrase.
    ("対象は3件あります。", "なし"),
])
def test_empty_set_path_does_not_rescue_enumerated_wrong_answers(pred, truth):
    assert crag.deterministic_judge(pred, truth) not in ("Perfect", "Acceptable")


@pytest.mark.parametrize(("pred", "truth"), [
    ("n_estimatorsは500です。", "n_estimators=300"),
    ("1位のモデルは500です。", "n_estimators（1位=500、2位=300）"),
    ("1位のモデルは500、2位のモデルは300、3位は200です。",
     "n_estimators（1位=500、2位=300）"),
])
def test_structured_prepass_rejects_contradiction_omission_and_extra_number(pred, truth):
    assert crag.deterministic_equivalence(pred, truth) is None


@pytest.mark.parametrize(("pred", "truth"), [
    ("費用見積が変更されています。契約金額(税込)が7,480,000円から8,250,000円に増額されました。",
     "提案書スライド6に、4.1データ理解〜4.5ガバナンスの作業内容が追記された"),
    ("age", "bmi"),
    ("抽出条件は行ラベルが8、列が2です。集計値は0.78です。",
     "hr=8、weekday=2で抽出されたデータに対する最大 / temp"),
    ("1473", "958"),
    ("4.23%", "1.18"),
    ("池田", "鈴木、藤田"),
    ("案件開始から第7週目と第8週目", "第6週目から第8週目"),
    ("データ移行支援（実績データ、周辺データ）", "監視ダッシュボード構築"),
    ("特別規定はなく、想定総工数170時間、時間単価25,000円で精算します。",
     "ACTHが200時間を超えた場合の特別規定はなく、時間単価25,000円で月次精算する。"),
    ("14人", "19"),
    ("最終モデル確定・再評価", "最終報告・成果物提出・検収会"),
])
def test_structured_prepass_does_not_promote_other_known_wrong_answers(pred, truth):
    assert crag.deterministic_judge(pred, truth) not in ("Perfect", "Acceptable")


def test_majority_option_keeps_strict_prompt_and_downgrade(monkeypatch):
    calls = iter(["Incorrect", "Perfect", "Perfect"])
    seen = []

    def fake_gemini(_pred, _truth, strict):
        seen.append(strict)
        return next(calls)

    monkeypatch.setattr(crag, "_judge_gemini", fake_gemini)
    verdict = crag.judge("主要な内容を自然文で説明しています。",
                         "同じ意味を別の自然文で説明しています。",
                         strict=True, majority=True)
    assert verdict == "Perfect"
    assert seen == [True, True, True]


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
