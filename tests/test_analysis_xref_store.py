"""SOT-2691 — 分析成果物クロス参照ストア（cycle8 C2）の決定論ヘルパの offline テスト（LLM/corpus 不要）。"""
from __future__ import annotations

from src.rag.index import analysis_xref_store as S


def test_extract_incomplete_ids_section_scoped():
    # 「要アクション（未完事項）」節の直後に列挙された ID のみを節スコープで拾い、昇順一意化する。
    # 別節（残課題）の "AI-08/09/10等" は拾わない。
    text = "\n".join([
        "[スライド3]",
        "01 エグゼクティブサマリ",
        "重要観察事項",
        "要アクション（未完事項）",
        "AI-05: 着手金支払の事後確認",
        "AI-09: Attr37の最終採否比較",
        "AI-08: 前処理仕様の確定",
        "※ 本書は確認済事項と仮定を分離して記載している。",
        "[スライド4]",
        "残課題",
        "文書化されたアクションリスト（AI-08/09/10等）で管理されている。",
    ])
    got = S.extract_incomplete_ids(text)
    assert got is not None
    assert got["ids"] == ["AI-05", "AI-08", "AI-09"]  # 昇順・重複なし・節外の 10 は含まない


def test_extract_incomplete_ids_absent_section():
    assert S.extract_incomplete_ids("見出し\n本文のみで未完事項節は無い") is None


def test_id_sort_key_prefix_then_number():
    ids = ["AI-9", "AI-08", "AI-05"]
    assert sorted(ids, key=S._id_sort_key) == ["AI-05", "AI-08", "AI-9"]


def test_onehot_threshold_from_features_src():
    src = "MAX_CATEGORICAL_UNIQUE = 100\n"
    assert S._onehot_threshold(src, None, None) == 100
    # fallback to config categorical_unique_limit when the source constant is absent
    assert S._onehot_threshold("", {"categorical_unique_limit": 50}, None) == 50
    assert S._onehot_threshold("", None, None) is None


def test_extract_selected_features_eng_ft_classification():
    # 選択特徴量節: 原列(train.csv 列)に無い派生列 = ENG-FT。図形塗りマーカーと注記は落とす。
    text = "\n".join([
        "選択特徴量（14変数）",
        "Major: 112件（欠損率 ≒ 0.971%）",
        "→ カテゴリ化等で対応済",
        "Gender", "Age", "Country", "Education", "Major", "Profession", "Industry", "Experience",
        "Age_ord", "【図形塗り:赤】Age_ord",
        "除外列: id（identifier_like_name）",
        "Exp_ord", "【図形塗り:赤】Exp_ord",
        "Edu_ord", "【図形塗り:赤】Edu_ord",
        "Age×Exp", "【図形塗り:赤】Age×Exp",
        "Age-Exp", "【図形塗り:赤】Age-Exp",
        "Edu×Exp", "【図形塗り:赤】Edu×Exp",
        "■ 原特徴量    ■ エンジニアリング特徴量",
        "4 / 15",
    ])
    original = ["id", "Gender", "Age", "Country", "Education", "Major", "Profession",
                "Industry", "Experience", "target"]
    got = S.extract_selected_features(text, original)
    assert got is not None
    assert got["selected_count"] == 14
    assert got["eng_ft"] == ["Age_ord", "Exp_ord", "Edu_ord", "Age×Exp", "Age-Exp", "Edu×Exp"]
    assert got["eng_ft_count"] == 6


# --------------------------------------------------------------------------- SOT-2699 staged metrics
_INTERIM_REPORT = "\n".join([
    "分析進捗報告書",
    "本報告の分析段階: interim（中間）",
    "モデル評価は f1_macro を主指標とし、T04 の f1_macro = 0.7329671168078127、"
    "accuracy = 0.7357142857142858 を記録しています。",
    "ベスト（可視範囲）: trial_index = 4",
    "f1_macro (primary): 0.7329671168078127",
    "他の可視試行の f1_macro:",
    "T01: 0.6854980146919636",
    "T02: 0.7126899909960438",
    "auc_roc: 0.8250532501536466",
])


def test_extract_report_metrics_assigns_nearest_metric():
    got = S.extract_report_metrics(_INTERIM_REPORT)
    # f1_macro の全出現が f1_macro へ、accuracy/auc_roc は別メトリクスへ振り分けられる。
    f1 = [h["value"] for h in got["f1_macro"]]
    assert 0.7329671168078127 in f1 and 0.6854980146919636 in f1
    assert 0.7357142857142858 not in f1  # accuracy は f1_macro に混ざらない
    assert got["accuracy"][0]["value"] == 0.7357142857142858
    assert got["auc_roc"][0]["value"] == 0.8250532501536466
    # ベスト（max）はフル精度（16 桁）。
    best = max(got["f1_macro"], key=lambda h: h["value"])
    assert best["raw"] == "0.7329671168078127"
    assert best["decimals"] >= S.FULL_PRECISION_MIN_DECIMALS


def test_report_stage_and_date():
    assert S._report_stage(_INTERIM_REPORT) == "interim"
    assert S._report_stage("本報告の分析段階: final（最終）") == "final"
    assert S._report_stage("段階の記載なし") is None
    assert S._report_date("報告資料_2025-09-16.docx") == "2025-09-16"
    assert S._report_date("no date here") is None


def test_full_precision_min_decimals_rejects_rounded():
    # 8 桁丸め leaderboard 値は full precision と認めない（SOT-2687 の 1e-9 差 Incorrect 回避）。
    rounded = S.extract_report_metrics("f1_macro = 0.73296712")
    best = max(rounded["f1_macro"], key=lambda h: h["value"])
    assert best["decimals"] < S.FULL_PRECISION_MIN_DECIMALS


def test_load_missing_is_empty(tmp_path):
    assert S.load(tmp_path / "nope.jsonl") == []


def test_enabled_default_off(monkeypatch):
    monkeypatch.delenv("RAG_ANALYSIS_XREF", raising=False)
    assert S.enabled() is False
    monkeypatch.setenv("RAG_ANALYSIS_XREF", "1")
    assert S.enabled() is True
