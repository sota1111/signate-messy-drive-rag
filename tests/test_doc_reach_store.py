"""SOT-2677 — 文書テーブル/全文チャンクストア + lookup レーンの offline テスト（ネット/LLM 不要）。

固定する不変量:
* OFF（``RAG_DOC_REACH_STORE`` unset）⇒ :func:`doc_lookup_lane.tool` は [] ⇒ tool 集合/MCP surface byte-identical。
* fact_layer.tools() は OFF で doc_reach ツールを一切足さない（surface byte-identical）。
* ON ⇒ table lookup が出自付きで表を返し、fulltext search が 8000 字切り詰めの向こう側（末尾）へ到達する。
* case/file の絞り込み、keyword 一致、末尾窓（keyword 省略）の各分岐。
"""
from __future__ import annotations

import pytest

from src.rag.agent import doc_lookup_lane as L
from src.rag.index import doc_reach_store as S


# --------------------------------------------------------------------------- synth store
def _doc(project, rel, *, ext="docx", tables=None, full_text=""):
    return {
        "project": project, "rel": rel, "name": rel.split("/")[-1], "ext": ext,
        "category": "proposal", "encrypted": False, "n_chars": len(full_text),
        "n_tables": len(tables or []), "tables": tables or [], "full_text": full_text,
    }


# 8000 字境界の向こう側にランキング表テキストを置いた長文（idx99 型）。
_TAIL = "死亡率が高い都道府県ランキング。1位 青森県 18.2%、2位 秋田県 16.3%。"
_LONG = ("あ" * 8200) + _TAIL

_RANK_TABLE = {
    "table_id": 3, "locus": "表3", "caption": "都道府県別 死亡率ランキング",
    "rows": [["順位", "都道府県（ワースト）", "死亡率（%）"], ["1位", "青森県", "18.2"],
             ["2位", "秋田県", "16.3"]],
}


@pytest.fixture()
def synth_store(monkeypatch):
    data = {
        "docs": [
            _doc("医療法人社団 蒼樹会 みなみ野女性医療センター",
                 "プロジェクト/みなみ野/00.提案/糖尿病統計情報.docx",
                 tables=[_RANK_TABLE], full_text=_LONG),
            _doc("株式会社東都人材プラットフォーム",
                 "プロジェクト/東都/00.提案/データサイエンティスト調査.docx",
                 tables=[{"table_id": 2, "locus": "表2", "caption": "産業別 年間賃金中央値",
                          "rows": [["産業", "年間賃金中央値（米ドル）"], ["科学研究開発", "120,090"]]}],
                 full_text=("x" * 9000) + " ML/DEの給与比較: MLエンジニア中央値139K。"),
        ],
    }
    data["by_project"] = {}
    for rec in data["docs"]:
        data["by_project"].setdefault(rec["project"], []).append(rec)
    monkeypatch.setattr(S, "load", lambda path=None: data)
    # glossary company_of を合成（会社名の一部 → canonical project）。
    monkeypatch.setattr(L, "_bind_project", lambda text: (
        "医療法人社団 蒼樹会 みなみ野女性医療センター" if "みなみ野" in text else
        "株式会社東都人材プラットフォーム" if "東都" in text else None))
    return data


# --------------------------------------------------------------------------- OFF byte-identical
def test_default_off_tool_empty(monkeypatch):
    monkeypatch.delenv("RAG_DOC_REACH_STORE", raising=False)
    assert S.enabled() is False
    assert L.enabled() is False
    assert L.tool() == []


def test_fact_layer_off_excludes_doc_reach(monkeypatch):
    """RAG_DOC_REACH_STORE OFF ⇒ fact_layer.tools() に doc_reach ツールが現れない。"""
    from src.rag.agent import fact_layer
    monkeypatch.setenv("RAG_FACT_LAYER", "1")
    monkeypatch.delenv("RAG_DOC_REACH_STORE", raising=False)
    names = {t[0] for t in fact_layer.tools()}
    assert "doc_table_lookup" not in names and "doc_fulltext_search" not in names


def test_fact_layer_on_includes_doc_reach(monkeypatch):
    from src.rag.agent import fact_layer
    monkeypatch.setenv("RAG_FACT_LAYER", "1")
    monkeypatch.setenv("RAG_DOC_REACH_STORE", "1")
    names = {t[0] for t in fact_layer.tools()}
    assert {"doc_table_lookup", "doc_fulltext_search"} <= names


# --------------------------------------------------------------------------- table lookup
def test_table_lookup_reaches_ranking_table(monkeypatch, synth_store):
    monkeypatch.setenv("RAG_DOC_REACH_STORE", "1")
    r = L._table_lookup(case="みなみ野の死亡率", keyword="死亡率")
    assert r["value"]["count"] == 1
    t = r["value"]["tables"][0]
    assert t["locus"] == "表3"
    assert ["1位", "青森県", "18.2"] in t["rows"]
    assert r["evidence"]["found"] is True


def test_table_lookup_keyword_filters(monkeypatch, synth_store):
    monkeypatch.setenv("RAG_DOC_REACH_STORE", "1")
    # keyword が一致しない → 該当なし
    r = L._table_lookup(case="みなみ野", keyword="存在しない語XYZ")
    assert r["value"]["count"] == 0 and r["evidence"]["found"] is False


def test_table_lookup_file_scope(monkeypatch, synth_store):
    monkeypatch.setenv("RAG_DOC_REACH_STORE", "1")
    r = L._table_lookup(file="データサイエンティスト調査", keyword="中央値")
    assert r["value"]["count"] == 1
    assert "東都" in r["value"]["tables"][0]["file"]


# --------------------------------------------------------------------------- fulltext search
def test_fulltext_reaches_past_truncation(monkeypatch, synth_store):
    monkeypatch.setenv("RAG_DOC_REACH_STORE", "1")
    r = L._fulltext_search(case="みなみ野", keyword="青森県")
    docs = r["value"]["documents"]
    assert docs and docs[0]["windows"]
    win = docs[0]["windows"][0]
    # 窓は keyword 前後 radius を含むので開始は 8000 手前でも、末尾は切り詰めの向こう側へ到達する。
    assert win["offset"] + len(win["excerpt"]) > 8000
    assert "青森県" in win["excerpt"]      # 8000 字切り詰めで落ちる内容へ到達


def test_fulltext_no_keyword_returns_tail(monkeypatch, synth_store):
    monkeypatch.setenv("RAG_DOC_REACH_STORE", "1")
    r = L._fulltext_search(case="みなみ野")
    win = r["value"]["documents"][0]["windows"][0]
    assert win["offset"] >= L._TAIL_FROM
    assert "青森県" in win["excerpt"]     # 末尾窓に切り詰め対象が含まれる


def test_fulltext_de_salary_reach(monkeypatch, synth_store):
    monkeypatch.setenv("RAG_DOC_REACH_STORE", "1")
    r = L._fulltext_search(case="東都", keyword="ML")
    win = r["value"]["documents"][0]["windows"][0]
    assert win["offset"] + len(win["excerpt"]) > 8000 and "給与比較" in win["excerpt"]


# --------------------------------------------------------------------------- store build (structural)
def test_build_doc_bounds_and_shape(tmp_path, monkeypatch):
    """build_doc は上限内で {tables, full_text} を返す（合成 FileRef、実 IO なし）。"""
    from src.rag.corpus import FileRef

    class _Cell:
        def __init__(self, text): self.text = text

    class _Row:
        def __init__(self, cells): self.cells = [_Cell(c) for c in cells]

    class _Table:
        def __init__(self, rows): self.rows = [_Row(r) for r in rows]

    class _Para:
        def __init__(self, text): self.text = text

    class _Doc:
        paragraphs = [_Para("都道府県別 死亡率ランキング")]
        tables = [_Table([["順位", "都道府県"], ["1位", "青森県"]])]

    ref = FileRef(path=tmp_path / "x.docx", project="P", category="proposal",
                  rel="P/x.docx", name="x.docx", ext="docx")
    monkeypatch.setattr(S.passwords, "is_encrypted", lambda p: False)
    import src.rag.index.doc_reach_store as mod
    monkeypatch.setattr(mod.office, "extract_docx", lambda r, d: "本文\n[表1]\n順位 | 都道府県")
    import docx as _docx
    monkeypatch.setattr(_docx, "Document", lambda *a, **k: _Doc())
    rec = S.build_doc(ref)
    assert rec is not None
    assert rec["n_tables"] == 1
    assert rec["tables"][0]["rows"][1] == ["1位", "青森県"]
    assert rec["full_text"].startswith("本文")
