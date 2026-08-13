"""SOT-2694 — 文書内数値の式適用レーン（cycle8 C5, idx68/50）の offline テスト（LLM/corpus 不要）。

決定論束縛の規律を合成ストアで固定する:
* OFF ⇒ None（byte-identical）。
* idx68 = 記載式への数値代入 ⇒ 事前計算値をそのまま返す。
* idx50 = 統計量表の 行キー×2 統計量列の差分（``|上位90% − 中央値|``）。
* doc/行キー/式名が曖昧・非該当なら defer（None）。
"""
from __future__ import annotations

import pytest

from src.rag.agent import formula_apply_lane as L

_Q68 = ("東都人材プラットフォームのデータサイエンス市場の未来予測.pdfにおいて、投資実装係数の計算式が"
        "記載されているページの数値情報を式に代入し、投資実装係数を小数で答えてください。")
_Q50 = ("東都人材プラットフォームのデータサイエンティスト調査において、Salary.com が公表している"
        "データサイエンティストの年間基本給について、上位90%の層と中央値の差はいくらですか。")


def _formula_rec():
    return {
        "kind": "formula", "doc_id": "x/データサイエンス市場の未来予測.pdf", "project": "東都人材プラットフォーム",
        "doc_name": "データサイエンス市場の未来予測.pdf", "locus": "ページ5",
        "formula_name": "投資実装係数", "expression": "(生産性向上率+コスト削減率)×ROI倍率",
        "bindings": {"生産性向上率": {"value": "0.226"}, "コスト削減率": {"value": "0.152"},
                     "ROI倍率": {"value": "3.7"}},
        "value": "1.3986",
    }


def _stat_rec():
    def cell(header, value):
        return {"header": header, "value": value}
    rows = [
        {"key": "米国労働省統計局 (BLS)", "key_norm": "米国労働省統計局(bls)",
         "cells": [cell("中央値・平均値(米ドル)", 112590), cell("下位10%・最低水準", 63000),
                   cell("上位90%・最高水準", 190000)]},
        {"key": "Salary.com", "key_norm": "salary.com",
         "cells": [cell("中央値・平均値(米ドル)", 123778), cell("下位10%・最低水準", 112000),
                   cell("上位90%・最高水準", 137000)]},
    ]
    return {"kind": "stat_table", "doc_id": "x/データサイエンティスト調査.docx",
            "project": "東都人材プラットフォーム", "doc_name": "データサイエンティスト調査.docx",
            "header": ["中央値・平均値(米ドル)", "下位10%・最低水準", "上位90%・最高水準"],
            "unit": "ドル", "rows": rows}


def _records_of(kind, path=None):
    return [_formula_rec()] if kind == "formula" else [_stat_rec()]


@pytest.fixture()
def wired(monkeypatch):
    monkeypatch.setenv("RAG_FORMULA_APPLY", "1")
    monkeypatch.setattr(L._store, "enabled", lambda: True)
    monkeypatch.setattr(L._store, "records_of", _records_of)
    return L


def test_off_is_none(monkeypatch):
    monkeypatch.setattr(L._store, "enabled", lambda: False)
    monkeypatch.setattr(L._store, "records_of", _records_of)
    assert L.resolve(_Q68) is None  # OFF ⇒ byte-identical fallback
    assert L.resolve(_Q50) is None


def test_idx68_formula_apply(wired):
    res = wired.resolve(_Q68)
    assert res is not None
    assert res["value"] == "1.3986"
    assert res["method"]["selection"] == "document_formula_apply"
    assert res["method"]["naturalize"] is False


def test_idx50_stat_diff(wired):
    res = wired.resolve(_Q50)
    assert res is not None
    assert res["value"] == "13,222ドル"  # |137,000 − 123,778|
    assert res["method"]["selection"] == "stat_table_pairwise_diff"


def test_formula_defers_without_apply_cue(wired):
    # 式適用のキュー（計算式/代入）が無ければ defer。
    assert wired.resolve("投資実装係数はいくつですか。") is None


def test_stat_defers_without_diff_cue(wired):
    # 「差」を問うていなければ defer。
    q = ("データサイエンティスト調査において、Salary.com の年間基本給の上位90%と中央値はそれぞれいくらですか。")
    assert wired.resolve(q) is None


def test_stat_defers_unknown_source(wired):
    # 行キー（情報源）が質問に無ければ defer（案件横断の誤束縛防止）。
    q = ("データサイエンティスト調査において、Nikkei の年間基本給の上位90%と中央値の差はいくらですか。")
    assert wired.resolve(q) is None


def test_stat_defers_single_stat_column(wired):
    # 参照される統計量列が 1 つだけなら差分は定義できず defer。
    q = ("データサイエンティスト調査において、Salary.com の年間基本給の中央値と何かの差はいくらですか。")
    # 「中央値」しか統計量キューが無い → refs != 2 → None
    assert wired.resolve(q) is None


def test_formula_defers_wrong_doc(wired):
    # 別文書名を指定しても式名レコードが1件なら doc_hits 空→named にフォールバックし1件bindするが、
    # 式名が質問に無ければ defer。
    assert wired.resolve("別の指数の計算式に代入してください。") is None
