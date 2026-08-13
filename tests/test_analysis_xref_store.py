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


def test_load_missing_is_empty(tmp_path):
    assert S.load(tmp_path / "nope.jsonl") == []


def test_enabled_default_off(monkeypatch):
    monkeypatch.delenv("RAG_ANALYSIS_XREF", raising=False)
    assert S.enabled() is False
    monkeypatch.setenv("RAG_ANALYSIS_XREF", "1")
    assert S.enabled() is True
