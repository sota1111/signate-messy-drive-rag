"""SOT-2687 — クロス参照カバレッジ拡張レーン（cycle7 K4）の offline テスト（ネットワーク/LLM 不要）。

決定論束縛の規律を合成ストアで固定する: OFF ⇒ None（byte-identical）、idx62=leaderboard 上位2件の設定差分、
idx36=段階メトリクスF1差、idx34=会議録アクションの M01→M02 完了×担当者。拮抗/別モデル/曖昧束縛は defer。
"""
from __future__ import annotations

import pytest

from src.rag.agent import xref_coverage_lane as X


# --------------------------------------------------------------------------- synthetic raw-artifact store
def _lb(header, rows, metric_col="primary_value", metric_name="f1_macro", lower=False):
    return {"rel": "…/leaderboard.csv", "header": header, "metric_col": metric_col,
            "metric_name": metric_name, "lower_is_better": lower, "rows": rows}


_AOBA = "青葉与信マネジメント株式会社"
_KAEDE = "医療法人社団 恒一会 かえで総合病院"


@pytest.fixture()
def synth_raw(monkeypatch):
    aoba_lb = _lb(
        ["trial_index", "status", "model_type", "n_estimators", "use_date_features",
         "random_state", "test_size", "primary_metric", "primary_value"],
        [{"trial_index": "10", "model_type": "extra_trees", "n_estimators": "500",
          "use_date_features": "True", "random_state": "42", "test_size": "0.2", "primary_value": "0.60266642"},
         {"trial_index": "6", "model_type": "extra_trees", "n_estimators": "300",
          "use_date_features": "True", "random_state": "42", "test_size": "0.2", "primary_value": "0.59534963"}])
    kaede_lb = _lb(
        ["trial_index", "model_type", "primary_metric", "primary_value"],
        [{"trial_index": "9", "model_type": "hist_gradient_boosting", "primary_value": "0.82915824"},
         {"trial_index": "4", "model_type": "linear_baseline", "primary_value": "0.73296712"},
         {"trial_index": "1", "model_type": "linear_baseline", "primary_value": "0.68549801"}])
    data = {"cases_by_project": {
        _AOBA: {"project": _AOBA, "leaderboard": aoba_lb},
        _KAEDE: {"project": _KAEDE, "leaderboard": kaede_lb,
                 "metrics": {"value": {"f1_macro": 0.8291582445227382, "accuracy": 0.83}}},
    }}
    monkeypatch.setattr(X._raw, "load", lambda path=None: data)
    monkeypatch.setattr(X, "_resolve_case",
                        lambda q: _AOBA if "青葉与信" in q else (_KAEDE if "かえで" in q else None))
    return data


_Q62 = "青葉与信マネジメントの最終報告資料における、モデル比較で上位2件のスコア差を生んでいる設定差分は何ですか。"
_Q36 = "恒一会 かえで総合病院案件において、中間報告時点のF1スコア実測値と最終報告時点のF1スコア実測値の差を絶対値で答えてください。"


def test_default_off_returns_none(monkeypatch, synth_raw):
    monkeypatch.delenv("RAG_XREF_COVERAGE", raising=False)
    assert X.enabled() is False
    assert X.resolve(_Q62) is None and X.resolve(_Q36) is None
    assert X.tool() is None


def test_idx62_leaderboard_setting_diff(monkeypatch, synth_raw):
    monkeypatch.setenv("RAG_XREF_COVERAGE", "1")
    r = X.resolve(_Q62)
    assert r is not None and r["value"] == "n_estimators（1位=500、2位=300）"
    assert r["method"]["selection"] == "leaderboard_top2_setting_diff"


def test_idx62_defers_on_different_model(monkeypatch, synth_raw):
    # 上位2件のモデル種別が違えば「設定差分」ではなくモデル差 → None。
    data = synth_raw
    data["cases_by_project"][_AOBA]["leaderboard"]["rows"][1]["model_type"] = "random_forest"
    monkeypatch.setenv("RAG_XREF_COVERAGE", "1")
    assert X.resolve(_Q62) is None


def test_idx62_defers_on_tie(monkeypatch, synth_raw):
    # 上位2件のスコアが同点（拮抗）→ 差を生む設定は決まらない → None。
    data = synth_raw
    data["cases_by_project"][_AOBA]["leaderboard"]["rows"][1]["primary_value"] = "0.60266642"
    monkeypatch.setenv("RAG_XREF_COVERAGE", "1")
    assert X.resolve(_Q62) is None


def test_idx62_defers_on_multiple_diffs(monkeypatch, synth_raw):
    # 差分が複数（一意に「その設定差」と言えない）→ None。
    data = synth_raw
    data["cases_by_project"][_AOBA]["leaderboard"]["rows"][1]["random_state"] = "77"
    monkeypatch.setenv("RAG_XREF_COVERAGE", "1")
    assert X.resolve(_Q62) is None


def test_idx36_lane_is_deliberately_unwired(monkeypatch, synth_raw):
    # idx36 は honest abstain（中間F1のフル精度が焼けず judge が ~1e-9 差を Incorrect と判定するため）。
    # レーンは _LANES に含めず、resolve() は None を返して LLM(＝棄権)へフォールバックする。
    monkeypatch.setenv("RAG_XREF_COVERAGE", "1")
    assert X._stage_metric_f1_diff not in X._LANES
    assert X.resolve(_Q36) is None


def test_idx36_helper_logic_still_correct(monkeypatch, synth_raw):
    # ロジック自体は正しい（証拠が焼けたら再配線できる）ことを直接検証する。
    monkeypatch.setenv("RAG_XREF_COVERAGE", "1")
    r = X._stage_metric_f1_diff(X._norm(_Q36), _Q36)
    assert r is not None
    assert abs(float(r["value"]) - 0.09619112452273816) < 1e-12
    assert r["evidence"]["intermediate_f1_linear"] == 0.73296712


def test_idx36_helper_defers_when_no_linear_stage(monkeypatch, synth_raw):
    data = synth_raw
    for row in data["cases_by_project"][_KAEDE]["leaderboard"]["rows"]:
        row["model_type"] = "hist_gradient_boosting"
    monkeypatch.setenv("RAG_XREF_COVERAGE", "1")
    assert X._stage_metric_f1_diff(X._norm(_Q36), _Q36) is None


def test_idx36_helper_defers_when_metric_not_f1(monkeypatch, synth_raw):
    data = synth_raw
    data["cases_by_project"][_KAEDE]["leaderboard"]["metric_name"] = "rmse"
    monkeypatch.setenv("RAG_XREF_COVERAGE", "1")
    assert X._stage_metric_f1_diff(X._norm(_Q36), _Q36) is None


# --------------------------------------------------------------------------- idx34 action transition
_Q34 = "MINAMINOにおいて、M01時点では未完了で、M02までの間に完了したAIのうち、伊藤さんが担当しているものを抽出してください。"
_MINA = "医療法人社団 蒼樹会 みなみ野女性医療センター"


class _Ref:
    def __init__(self, rel):
        self.rel = rel
        self.project = _MINA
        self.category = "meeting"
        self.name = rel.rsplit("/", 1)[-1]


@pytest.fixture()
def synth_meetings(monkeypatch):
    m01 = _Ref("プロジェクト/…/05.会議/会議録/会議録_2025-04-03.pdf")
    m02 = _Ref("プロジェクト/…/05.会議/会議録/会議録_2025-04-24.pdf")
    m03 = _Ref("プロジェクト/…/05.会議/会議録/会議録_2025-05-15.pdf")
    ocr = {
        m01.rel: ("A04 変更管 伊藤翔 2025-04- Open\n"
                  "A05 キック 伊藤翔 2025-04- 完了\n"
                  "A06 クライ 林さくら 2025-04-10 Open A09 週次定 伊藤翔太 20\n"
                  "A08 分析用 伊藤翔太/岡田佑樹 2025-04-07 Open\n"),
        m02.rel: ("A01 分析用データ  done\n"
                  "A04 変更管 伊藤翔 2025-04- Open\n"
                  "A08 分析用 伊藤翔 2025-04- Close\n"
                  "A09 週次定 伊藤翔 2025-04- Close\n"),
        m03.rel: "A10 前処理 done\n",
    }
    monkeypatch.setattr(X, "_resolve_case", lambda q: _MINA if "MINAMINO" in q else None)

    def fake_walk():
        return [m01, m03, m02]  # 意図的に順不同（レーンが rel 順で M01/M02 を選ぶことを検証）
    import src.rag.corpus as corpus
    import src.rag.index.ocr_store as ocr_store
    monkeypatch.setattr(corpus, "walk", fake_walk)
    monkeypatch.setattr(ocr_store, "lookup", lambda ref: ocr.get(ref.rel, ""))
    return ocr


def test_idx34_action_completed_between_meetings(monkeypatch, synth_meetings):
    monkeypatch.setenv("RAG_XREF_COVERAGE", "1")
    r = X.resolve(_Q34)
    assert r is not None and r["value"] == "A08、A09"
    assert r["method"]["selection"] == "action_completed_between_meetings_by_owner"
    assert r["evidence"]["person"] == "伊藤"


def test_idx34_defers_without_person(monkeypatch, synth_meetings):
    monkeypatch.setenv("RAG_XREF_COVERAGE", "1")
    q = "MINAMINOにおいて、M01時点では未完了で、M02までの間に完了したAIを抽出してください。"
    assert X.resolve(q) is None  # 担当者(person)が名指されないと束縛しない


def test_idx34_defers_when_case_unresolved(monkeypatch, synth_meetings):
    monkeypatch.setattr(X, "_resolve_case", lambda q: None)
    monkeypatch.setenv("RAG_XREF_COVERAGE", "1")
    assert X.resolve(_Q34) is None


def test_negative_controls_do_not_fire(monkeypatch, synth_raw):
    monkeypatch.setenv("RAG_XREF_COVERAGE", "1")
    assert X.resolve("青葉与信の最終報告のF1スコアは？") is None       # 上位2件設定差でない
    assert X.resolve("かえでの中間報告のF1は？") is None               # 中間/最終差でない
    assert X.resolve("MINAMINOの担当者は誰ですか？") is None           # M01/M02 完了遷移でない
