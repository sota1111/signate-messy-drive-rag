"""Offline tests for fact-level indexing (SOT-2449 / R1) — no LLM / GCP / corpus needed.

    .venv/bin/python -m pytest scoring/test_facts.py -q

Cover the two acceptance-critical behaviours deterministically:
  * `facts.fact_rows` distils the extract-layer markers (highlight / bold / pptx / code param)
    into one-fact-per-line rows with metadata, and is a no-op when disabled;
  * `retrieve.Retriever` surfaces & boosts a fact row for a representative highlight/extract
    question (embeddings stubbed to zeros so BM25 drives ranking — no network).
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from src.rag import facts
from src.rag.corpus import FileRef


def _ref(name: str, ext: str, project: str = "青葉", rel: str | None = None,
         category: str = "") -> FileRef:
    return FileRef(path=Path("/nonexistent") / name, project=project, category=category,
                   rel=rel or f"{project}/{name}", name=name, ext=ext)


# ------------------------------- version_of --------------------------------------------------
@pytest.mark.parametrize("name,expected", [
    ("スケジュール_r2.xlsx", "r2"),
    ("提案書_最新.docx", "最新"),
    ("旧版_契約.docx", "旧版"),
    ("data_20240115.csv", "20240115"),
    ("modeling.py", ""),
    ("report.pdf", ""),
    ("figure_06.png", ""),
])
def test_version_of(name, expected):
    assert facts.version_of(name) == expected


# ------------------------------- marker facts (docx / xlsx / pptx) ---------------------------
def test_docx_bold_and_highlight_facts():
    ref = _ref("契約書.docx", "docx")
    text = ("【太字箇所】契約金額 / 支払条件 / 秘密保持\n"
            "本文がここに続く。\n"
            "【ハイライト】更新期限は自動更新とする\n")
    rows = facts.fact_rows(ref, text)
    joined = "\n".join(rows)
    assert any("種別: 太字] 契約金額" in r for r in rows)
    assert any("種別: 太字] 支払条件" in r for r in rows)
    assert any("種別: 太字] 秘密保持" in r for r in rows)
    assert any("種別: ハイライト] 更新期限は自動更新とする" in r for r in rows)
    assert "案件: 青葉" in joined and "ファイル: 青葉/契約書.docx" in joined


def test_xlsx_highlight_cell_facts():
    ref = _ref("train.xlsx", "xlsx")
    text = ("[シート: data]  範囲 A1:C10\n"
            "id | 名称 | 値\n"
            "【ハイライトされたセル】\n"
            "  B3(オレンジ): 売上高\n"
            "  C5(黄): 1200\n"
            "[シート: other]\n"
            "無関係\n")
    rows = facts.fact_rows(ref, text)
    assert any("種別: ハイライトセル] B3(オレンジ): 売上高" in r for r in rows)
    assert any("種別: ハイライトセル] C5(黄): 1200" in r for r in rows)
    # the block terminates at the next non-indented line — "無関係" must not become a fact
    assert not any("無関係" in r for r in rows)


def test_pptx_highlight_and_fill_facts():
    ref = _ref("提案.pptx", "pptx")
    text = ("[スライド1]\n"
            "【ハイライト:黄】期限厳守\n"
            "【図形塗り:オレンジ】重点施策\n")
    rows = facts.fact_rows(ref, text)
    assert any("種別: ハイライト] 黄: 期限厳守" in r for r in rows)
    assert any("種別: 図形塗り] オレンジ: 重点施策" in r for r in rows)


def test_code_param_facts_only_curated_keys():
    ref = _ref("modeling.py", "py")
    text = ("[コード modeling.py]\n"
            "random_state = 42\n"
            'model_type = "lightgbm"\n'
            "test_size = 0.2\n"
            "unrelated_var = 7\n")
    rows = facts.fact_rows(ref, text)
    assert any("種別: パラメータ] random_state = 42" in r for r in rows)
    assert any("種別: パラメータ] model_type = lightgbm" in r for r in rows)
    assert any("種別: パラメータ] test_size = 0.2" in r for r in rows)
    assert not any("unrelated_var" in r for r in rows)


def test_code_facts_not_emitted_for_non_code_ext():
    ref = _ref("notes.md", "md")
    rows = facts.fact_rows(ref, "random_state = 42\n")
    assert rows == []  # markdown text is not scanned for code params


def test_flag_default_off_and_opt_in(monkeypatch):
    monkeypatch.delenv("RAG_FACT_INDEX", raising=False)
    assert facts.enabled() is False  # opt-in: production stays byte-identical by default
    monkeypatch.setenv("RAG_FACT_INDEX", "1")
    assert facts.enabled() is True
    monkeypatch.setenv("RAG_FACT_INDEX", "0")
    assert facts.enabled() is False


def test_fact_rows_is_pure_regardless_of_flag(monkeypatch):
    # fact_rows extracts unconditionally; the flag gates the call sites (index.build/retrieve).
    monkeypatch.setenv("RAG_FACT_INDEX", "0")
    ref = _ref("契約書.docx", "docx")
    rows = facts.fact_rows(ref, "【太字箇所】契約金額 / 支払条件\n")
    assert any("契約金額" in r for r in rows)


def test_facts_are_deduped_and_capped():
    ref = _ref("x.docx", "docx")
    text = "\n".join(["【ハイライト】同じ事実"] * 5)
    rows = facts.fact_rows(ref, text)
    assert len(rows) == 1  # identical facts collapse


# ------------------------------- retrieval integration ---------------------------------------
class _Glossary:
    def expand_terms(self, q):
        return []

    def company_of(self, q):
        return None


def _chunks():
    return [
        {"id": 0, "project": "青葉", "category": "contract", "file": "契約書.docx",
         "rel": "青葉/契約書.docx", "kind": "text",
         "text": "[案件: 青葉 | 区分: contract | ファイル: 青葉/契約書.docx]\n"
                 "本契約は甲乙間の一般的な取引条件を定めるものである。"},
        {"id": 1, "project": "青葉", "category": "data", "file": "train.xlsx",
         "rel": "青葉/train.xlsx", "kind": "fact",
         "text": "[fact | 案件: 青葉 | ファイル: 青葉/train.xlsx | 種別: ハイライトセル] "
                 "B3(オレンジ): 売上高1200万円"},
        {"id": 2, "project": "東都", "category": "report", "file": "報告.docx",
         "rel": "東都/報告.docx", "kind": "text",
         "text": "[案件: 東都 | 区分: report | ファイル: 東都/報告.docx]\n無関係な報告書の本文。"},
    ]


def _make_retriever(monkeypatch):
    from src.rag import index, llm, retrieve

    chunks = _chunks()
    monkeypatch.setattr(index, "load_chunks", lambda: chunks)
    monkeypatch.setattr(index, "load_embeddings",
                        lambda: np.zeros((len(chunks), 4), dtype=np.float32))
    monkeypatch.setattr(retrieve.glossary, "load", lambda: _Glossary())
    monkeypatch.setattr(llm, "embed", lambda texts, **k: [[0.0] * 4 for _ in texts])
    return retrieve.Retriever()


_Q_HL = "青葉のtrain.xlsxでオレンジでハイライトされたセルの値をすべて答えてください。"


def test_fact_row_surfaces_for_highlight_question(monkeypatch):
    from src.rag import archetype

    monkeypatch.setenv("RAG_FACT_INDEX", "1")  # opt-in feature under test
    assert archetype.classify(_Q_HL) == "highlight_set"  # fact-favoring archetype
    r = _make_retriever(monkeypatch)
    out = r.retrieve(_Q_HL, k=3, pool=10)
    kinds = {c["rel"]: c["kind"] for c in out}
    assert kinds.get("青葉/train.xlsx") == "fact"  # the fact row was retrieved
    # and it ranks at the top for a highlight question
    assert out[0]["rel"] == "青葉/train.xlsx"


def test_fact_boost_raises_score(monkeypatch):
    from src.rag import retrieve

    monkeypatch.setenv("RAG_FACT_INDEX", "1")
    assert facts.enabled() is True
    r = _make_retriever(monkeypatch)
    with_boost = {c["rel"]: c["score"] for c in r.retrieve(_Q_HL, k=3, pool=10)}
    monkeypatch.setattr(retrieve.facts, "enabled", lambda: False)
    without = {c["rel"]: c["score"] for c in r.retrieve(_Q_HL, k=3, pool=10)}
    # same fact row, strictly higher score when the boost is active
    assert with_boost["青葉/train.xlsx"] > without["青葉/train.xlsx"]
