"""SOT-2686 — xlsx 数式トレース/回帰適用レーン（cycle7 K3）の offline テスト（ネットワーク/LLM 不要）。

決定論束縛の規律を合成ストアで固定する: OFF ⇒ None（byte-identical）、idx47=ハイライト数式が参照するデータ行の
属性（建設年→YEAR BUILT）、idx83=係数表を index=N 行へ当てはめた予測値。曖昧束縛（複数ハイライト/複数参照行/
属性不一致/index 不在）は defer。
"""
from __future__ import annotations

import pytest

from src.rag.agent import xlsx_formula_lane as X

_AOMINE = "株式会社青嶺不動産アセットマネジメント"
_MINAMINO = "医療法人社団 蒼樹会 みなみ野女性医療センター"

_Q47 = ("青嶺不動産アセットマネジメントのtrain.xlsxにおいて、黄色ハイライトセルは予測と実際の誤差を"
        "計算していますが、その予測値の対象となっている不動産の建設年を算出してください。")
_Q83 = ("蒼樹会 みなみ野女性医療センターのtrain.xlsxにおいて、回帰分析の結果として記載されている係数を"
        "index=1770のデータに当てはめたときの予測値はいくつですか。小数第5位まで答えてください。")


def _aomine_doc():
    return {
        "project": _AOMINE, "doc_name": "train.xlsx",
        "highlight_formulas": [{
            "sheet": "Sheet2", "cell": "B22", "fill": "FFFFFF00",
            "formula": "=(B18*Sheet1!U26118+Sheet2!B19*Sheet1!V26118+Sheet2!B17-Sheet1!W26118)^2",
            "referenced_rows": [{"sheet": "Sheet1", "row": 26118, "id": "train_26116",
                                 "attributes": {"id": "train_26116", "BOROUGH": 3,
                                                "YEAR BUILT": 1899, "YEAR BUILT_fillna": 127,
                                                "SALE PRICE": 1200000}}],
        }],
        "regressions": [],
    }


def _minamino_doc():
    return {
        "project": _MINAMINO, "doc_name": "train.xlsx", "highlight_formulas": [],
        "regressions": [{
            "coef_sheet": "Sheet1", "coef_cell": "B16", "intercept": -0.6569464764590836,
            "coefficients": {"BMI": 0.012622074783422584, "Age": 0.00971358116067913},
            "data_sheet": "train", "index_column": "index",
            "predictions": {"1770": 0.3831705891880006, "0": 0.12},
        }],
    }


@pytest.fixture()
def synth(monkeypatch):
    def _docs(question):
        if "青嶺" in question:
            return [_aomine_doc()]
        if "みなみ野" in question:
            return [_minamino_doc()]
        return []
    monkeypatch.setattr(X, "_resolve_case_docs", _docs)


def test_default_off_returns_none(monkeypatch, synth):
    monkeypatch.delenv("RAG_XLSX_FORMULA_TRACE", raising=False)
    assert X.enabled() is False
    assert X.resolve(_Q47) is None and X.resolve(_Q83) is None


def test_idx47_highlight_formula_trace(monkeypatch, synth):
    monkeypatch.setenv("RAG_XLSX_FORMULA_TRACE", "1")
    r = X.resolve(_Q47)
    assert r is not None and r["value"] == "1899年"
    assert r["method"]["selection"] == "highlight_formula_trace"
    assert r["evidence"]["attribute"] == "YEAR BUILT"


def test_idx83_documented_regression_apply(monkeypatch, synth):
    monkeypatch.setenv("RAG_XLSX_FORMULA_TRACE", "1")
    r = X.resolve(_Q83)
    assert r is not None and r["value"] == "0.38317"
    assert r["method"]["selection"] == "documented_regression_apply"
    assert r["evidence"]["index"] == "1770" and r["evidence"]["decimals"] == 5


def test_idx83_defers_when_index_absent(monkeypatch, synth):
    monkeypatch.setenv("RAG_XLSX_FORMULA_TRACE", "1")
    q = _Q83.replace("index=1770", "index=99999")
    assert X.resolve(q) is None


def test_idx47_defers_on_ambiguous_referenced_rows(monkeypatch):
    doc = _aomine_doc()
    doc["highlight_formulas"][0]["referenced_rows"].append(
        {"sheet": "Sheet1", "row": 100, "id": "x", "attributes": {"YEAR BUILT": 2000}})
    monkeypatch.setattr(X, "_resolve_case_docs", lambda q: [doc])
    monkeypatch.setenv("RAG_XLSX_FORMULA_TRACE", "1")
    assert X.resolve(_Q47) is None


def test_idx47_defers_on_multiple_highlight_formulas(monkeypatch):
    doc = _aomine_doc()
    doc["highlight_formulas"].append(dict(doc["highlight_formulas"][0], cell="B23"))
    monkeypatch.setattr(X, "_resolve_case_docs", lambda q: [doc])
    monkeypatch.setenv("RAG_XLSX_FORMULA_TRACE", "1")
    assert X.resolve(_Q47) is None


def test_defers_when_case_unbound(monkeypatch):
    monkeypatch.setattr(X, "_resolve_case_docs", lambda q: [])
    monkeypatch.setenv("RAG_XLSX_FORMULA_TRACE", "1")
    assert X.resolve(_Q47) is None and X.resolve(_Q83) is None


def test_investigator_tool_surface_unchanged_on_vs_off(monkeypatch):
    """This lane is a deterministic resolve-only lane: enabling it must NOT add an investigator tool,
    so LLM-route questions (incl. Sonnet sentinels) see a byte-identical tool surface (no idx21 perturbation)."""
    from src.rag.agent import fact_layer
    monkeypatch.setenv("RAG_FACT_LAYER", "1")
    monkeypatch.setenv("RAG_XLSX_FORMULA_TRACE", "0")
    names_off = sorted(t[0] for t in fact_layer.tools())
    monkeypatch.setenv("RAG_XLSX_FORMULA_TRACE", "1")
    names_on = sorted(t[0] for t in fact_layer.tools())
    assert names_off == names_on
    assert "xlsx_formula_lookup" not in names_on
