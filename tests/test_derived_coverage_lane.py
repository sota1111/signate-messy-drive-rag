"""SOT-2679 — 派生メトリクス網羅拡張 serve レーンの offline テスト（ネットワーク/LLM 不要）。

決定論レーンの束縛規律を合成ストア行で固定する: OFF ⇒ None（byte-identical）、idx4=相関最大特徴量、
idx24=案件横断欠損行数 argmax（full-coverage かつ厳密単独最大のみ）、拮抗/部分母集団/曖昧束縛は defer。
"""
from __future__ import annotations

import pytest

from src.rag.agent import derived_coverage_lane as L


def _row(case_id, *, with_target=None, top_feature=None, missing_row_count=None):
    return {
        "case_id": case_id,
        "sources": {"train_file": f"プロジェクト/{case_id}/03.データ/train.csv"},
        "correlations": {"target": "charges", "with_target": with_target or {},
                         "top_feature": top_feature},
        "missing": {"row_count": 100, "missing_row_count": missing_row_count,
                    "complete_row_count": None, "missing_cell_total": 0},
    }


@pytest.fixture()
def synth_store(monkeypatch):
    rows = [
        _row("医療法人社団 蒼泉会 ひがし丘総合病院",
             with_target={"age": 0.10, "bmi": 0.17, "children": 0.03, "id": 0.01},
             top_feature={"feature": "bmi", "r": 0.17, "abs_r": 0.17},
             missing_row_count=0),
        _row("株式会社青嶺不動産アセットマネジメント",
             with_target={"BOROUGH": -0.35}, top_feature={"feature": "BOROUGH", "r": -0.35, "abs_r": 0.35},
             missing_row_count=13556),
        _row("白峰信用リスク評価株式会社", with_target={}, top_feature=None, missing_row_count=3932),
    ]
    monkeypatch.setattr(L._dm, "load", lambda path=None: rows)

    def _abbrev(cid):
        return {"株式会社青嶺不動産アセットマネジメント": "AOMINE",
                "白峰信用リスク評価株式会社": "SHR"}.get(cid)
    monkeypatch.setattr(L, "_abbrev_for", _abbrev)
    return rows


_Q4 = "蒼泉会 ひがし丘総合病院の01_eda.ipynbを確認して、目的変数と相関が最も高い数値特徴量を教えてください。"
_Q24 = "分析データの中で、1つでも欠損値がある行数が最も多い案件を、主略称で答えてください。"


def test_default_off_returns_none(monkeypatch, synth_store):
    monkeypatch.delenv("RAG_DERIVED_COVERAGE", raising=False)
    assert L.enabled() is False
    assert L.resolve(_Q4) is None and L.resolve(_Q24) is None


def test_idx4_top_correlated_feature(monkeypatch, synth_store):
    monkeypatch.setenv("RAG_DERIVED_COVERAGE", "1")
    r = L.resolve(_Q4)
    assert r is not None and r["value"] == "bmi"
    assert r["method"]["selection"] == "top_correlated_feature"
    assert r["evidence"]["feature"] == "bmi"


def test_idx24_missing_row_argmax_returns_abbrev(monkeypatch, synth_store):
    monkeypatch.setenv("RAG_DERIVED_COVERAGE", "1")
    r = L.resolve(_Q24)
    assert r is not None and r["value"] == "AOMINE"          # 青嶺(13556) — 白峰(3932) ではない
    assert r["method"]["selection"] == "cross_case_missing_row_argmax"
    assert r["evidence"]["missing_row_count"] == 13556
    assert r["evidence"]["runner_up"]["missing_row_count"] == 3932


def test_idx24_defers_on_partial_coverage(monkeypatch):
    # 1件でも missing_row_count が None（母集団未確定）なら argmax を確定しない。
    rows = [_row("A", missing_row_count=10), _row("B", missing_row_count=None)]
    monkeypatch.setattr(L._dm, "load", lambda path=None: rows)
    monkeypatch.setenv("RAG_DERIVED_COVERAGE", "1")
    assert L.resolve(_Q24) is None


def test_idx24_defers_on_tie(monkeypatch):
    rows = [_row("A", missing_row_count=10), _row("B", missing_row_count=10)]
    monkeypatch.setattr(L._dm, "load", lambda path=None: rows)
    monkeypatch.setenv("RAG_DERIVED_COVERAGE", "1")
    assert L.resolve(_Q24) is None


def test_top_corr_defers_on_tie(monkeypatch):
    rows = [_row("蒼泉会 ひがし丘総合病院",
                 with_target={"age": 0.20, "bmi": 0.20}, top_feature={"feature": "age"})]
    monkeypatch.setattr(L._dm, "load", lambda path=None: rows)
    monkeypatch.setenv("RAG_DERIVED_COVERAGE", "1")
    assert L.resolve(_Q4) is None                            # |r| 拮抗 ⇒ 単独最大でない


def test_negative_controls_do_not_fire(monkeypatch, synth_store):
    monkeypatch.setenv("RAG_DERIVED_COVERAGE", "1")
    # 相関だが特徴量ランキングでない（F1 閾値の質問）
    assert L.resolve("青葉のTXでF1最大化閾値でのF1は？") is None
    # 欠損だが「最も多い案件」でない（単一案件の値）
    assert L.resolve("白峰の欠損値がある行数は何行ですか") is None


def test_tool_missing_ranking_and_off(monkeypatch, synth_store):
    monkeypatch.delenv("RAG_DERIVED_COVERAGE", raising=False)
    assert L.tool() is None                                  # OFF ⇒ surface byte-identical
    monkeypatch.setenv("RAG_DERIVED_COVERAGE", "1")
    name, desc, schema, handler = L.tool()
    assert name == "derived_coverage_lookup"
    out = handler(metric="missing_ranking")
    ranked = out["value"]
    assert ranked[0]["abbrev"] == "AOMINE" and ranked[0]["missing_row_count"] == 13556
