"""SOT-2691 — 分析成果物クロス参照レーン（cycle8 C2, idx60/73/53）の offline テスト（LLM/corpus 不要）。

決定論束縛の規律を合成ストアで固定する: OFF ⇒ None（byte-identical）、束縛できない案件は defer、
idx36 は成果物にフル精度が無いため常に未配線（発火しない）。
"""
from __future__ import annotations

import pytest

from src.rag.agent import analysis_xref_lane as L

_Q60 = "白峰信用リスク評価の最終報告資料内で未完事項として挙げられているIDをすべて抽出してください。"
_Q73 = ("恒一会のPPで言及されている One-Hot Encoding のカテゴリ数閾値を実装設定から確認したうえで、"
        "その条件により One-Hot Encoding の対象となるカテゴリ列をすべて答えてください。")
_Q53 = "TOTOのFR書にて記載のある選択特徴量のうち、ENG-FTはいくつありますか。"
_Q36 = ("恒一会 かえで総合病院案件において、中間報告時点のF1スコア実測値と最終報告時点のF1スコア実測値の"
        "差を絶対値で答えてください。")


def _rows():
    return [
        {"project": "白峰信用リスク評価株式会社", "report_rel": "x/最終報告.pptx",
         "incomplete_ids": {"ids": ["AI-05", "AI-08", "AI-09"], "section_header": "要アクション（未完事項）"},
         "staged_metrics": {"final_f1_macro": 0.85, "intermediate_full_precision_available": False}},
        {"project": "医療法人社団 恒一会 かえで総合病院", "report_rel": "y/最終報告.pptx",
         "onehot": {"onehot_targets": ["Gender"], "threshold": 100,
                    "threshold_source": "features.py:MAX_CATEGORICAL_UNIQUE",
                    "categorical": {"Gender": {"nunique": 2}}, "train_rel": "y/train.csv"},
         "staged_metrics": {"final_f1_macro": 0.8291582445227382,
                            "intermediate_full_precision_available": False}},
        {"project": "株式会社東都人材プラットフォーム", "report_rel": "z/最終報告.pptx",
         "selected_features": {"selected": ["Gender", "Age", "Age_ord", "Exp_ord", "Edu_ord",
                                            "Age×Exp", "Age-Exp", "Edu×Exp"],
                               "selected_count": 8,
                               "eng_ft": ["Age_ord", "Exp_ord", "Edu_ord", "Age×Exp", "Age-Exp", "Edu×Exp"],
                               "eng_ft_count": 6}},
    ]


@pytest.fixture()
def wired(monkeypatch):
    monkeypatch.setenv("RAG_ANALYSIS_XREF", "1")
    monkeypatch.setattr(L._store, "load", lambda *a, **k: _rows())

    # glossary.company_of を合成ストアの project 名へ一意束縛するスタブ。
    class _G:
        def company_of(self, q):
            if "白峰" in q:
                return "白峰信用リスク評価株式会社"
            if "恒一会" in q:
                return "医療法人社団 恒一会 かえで総合病院"
            if "TOTO" in q or "東都" in q:
                return "株式会社東都人材プラットフォーム"
            return None

    import src.rag.extract.glossary as g
    monkeypatch.setattr(g, "load", lambda *a, **k: _G())
    return L


def test_off_is_none(monkeypatch):
    monkeypatch.delenv("RAG_ANALYSIS_XREF", raising=False)
    monkeypatch.setattr(L._store, "load", lambda *a, **k: _rows())
    assert L.resolve(_Q60) is None  # OFF ⇒ byte-identical fallback


def test_idx60_incomplete_ids(wired):
    r = wired.resolve(_Q60)
    assert r is not None
    assert r["value"] == "AI-05、AI-08、AI-09"
    assert r["method"]["contract"] == "document_extract"


def test_idx73_onehot_targets(wired):
    r = wired.resolve(_Q73)
    assert r is not None
    assert r["value"] == "Gender"
    assert r["method"]["contract"] == "enum_set"


def test_idx53_eng_ft_count(wired):
    r = wired.resolve(_Q53)
    assert r is not None
    assert r["value"] == "6"
    assert r["method"]["contract"] == "derived_calculation"


def test_idx36_deliberately_unwired(wired):
    # 中間段階フル精度が無いため idx36 は発火しない（honest abstain）。
    assert wired.resolve(_Q36) is None


def test_unbound_case_defers(wired):
    assert wired.resolve("未収録の会社の未完事項IDをすべて抽出してください") is None


def test_empty_targets_defer(monkeypatch):
    # onehot_targets が空リスト（該当なし）や None（閾値不明）は確定しない → LLM へ。
    monkeypatch.setenv("RAG_ANALYSIS_XREF", "1")
    rows = [{"project": "P", "onehot": {"onehot_targets": [], "threshold": 100}}]
    monkeypatch.setattr(L._store, "load", lambda *a, **k: rows)
    import src.rag.extract.glossary as g

    class _G:
        def company_of(self, q):
            return "P"
    monkeypatch.setattr(g, "load", lambda *a, **k: _G())
    assert L.resolve(_Q73) is None
