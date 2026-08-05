"""Offline tests for the R3 4-signal RRF fusion (SOT-2450) — no LLM / GCP / corpus needed.

    .venv/bin/python -m pytest scoring/test_rrf4.py -q

Cover the two new signals deterministically (embeddings stubbed to zeros so BM25 + the new
signals drive ranking — no network):
  * IDF-rarity: a chunk matching a RARE query token surfaces / scores higher than with the flag off;
  * version-decay: a chunk from an explicitly OLD-version file is down-ranked for a normal question
    but NOT for a version-diff question (旧版参照設問を壊さない);
  * the whole layer is a structural no-op when the flag is unset (champion path byte-identical).
"""
from __future__ import annotations

import numpy as np
import pytest

from src.rag import retrieve


# ------------------------------- flags -------------------------------------------------------
def test_rrf4_flag_default_off_and_opt_in(monkeypatch):
    monkeypatch.delenv("RAG_RRF4", raising=False)
    assert retrieve.rrf4_enabled() is False  # opt-in: production stays byte-identical by default
    monkeypatch.setenv("RAG_RRF4", "1")
    assert retrieve.rrf4_enabled() is True
    monkeypatch.setenv("RAG_RRF4", "0")
    assert retrieve.rrf4_enabled() is False


@pytest.mark.parametrize("name,expected", [
    ("提案書_旧版.docx", True),
    ("契約_旧ファイル.xlsx", True),
    ("report_old.pptx", True),
    ("提案書_最新.docx", False),
    ("スケジュール_r2.xlsx", False),   # ambiguous rev token — never decayed
    ("modeling.py", False),
])
def test_is_old_version(name, expected):
    assert retrieve._is_old_version(name) == expected


# ------------------------------- retriever stub ----------------------------------------------
class _Glossary:
    def expand_terms(self, q):
        return []

    def company_of(self, q):
        return None


def _make_retriever(monkeypatch, chunks):
    from src.rag import index, llm

    monkeypatch.setattr(index, "load_chunks", lambda: chunks)
    monkeypatch.setattr(index, "load_embeddings",
                        lambda: np.zeros((len(chunks), 4), dtype=np.float32))
    monkeypatch.setattr(retrieve.glossary, "load", lambda: _Glossary())
    monkeypatch.setattr(llm, "embed", lambda texts, **k: [[0.0] * 4 for _ in texts])
    return retrieve.Retriever()


def _chunk(cid, project, file, kind, text, rel=None):
    return {"id": cid, "project": project, "category": "", "file": file,
            "rel": rel or f"{project}/{file}", "kind": kind, "text": text}


# ------------------------------- IDF-rarity (R3a) --------------------------------------------
def test_idf_rarity_raises_score_of_rare_token_match(monkeypatch):
    # Many generic chunks + one chunk containing a rare part number. Low idf_min so the mechanism
    # is exercised independent of the tiny corpus size.
    monkeypatch.setenv("RAG_IDF_MIN", "0.5")
    generic = [_chunk(i, "青葉", f"doc{i}.txt", "text",
                      f"[案件: 青葉 | ファイル: 青葉/doc{i}.txt]\n一般的な業務資料の本文です。")
               for i in range(12)]
    rare = _chunk(99, "青葉", "spec.txt", "text",
                  "[案件: 青葉 | ファイル: 青葉/spec.txt]\n対象機器の型番は ZX9000 である。")
    chunks = generic + [rare]
    r = _make_retriever(monkeypatch, chunks)
    q = "型番 ZX9000 の対象機器は何ですか"

    monkeypatch.setenv("RAG_RRF4", "1")
    on = {c["rel"]: c["score"] for c in r.retrieve(q, k=13, pool=20)}
    monkeypatch.setenv("RAG_RRF4", "0")
    off = {c["rel"]: c["score"] for c in r.retrieve(q, k=13, pool=20)}

    # the rare-token chunk gets an extra fused contribution when the signal is on
    assert on["青葉/spec.txt"] > off["青葉/spec.txt"]


def test_idf_rarity_is_noop_when_no_rare_query_token(monkeypatch):
    monkeypatch.setenv("RAG_IDF_MIN", "50")  # nothing is rare enough
    chunks = [_chunk(i, "青葉", f"doc{i}.txt", "text", f"共通の一般語 {i}") for i in range(6)]
    r = _make_retriever(monkeypatch, chunks)
    monkeypatch.setenv("RAG_RRF4", "1")
    on = {c["rel"]: c["score"] for c in r.retrieve("一般語", k=6, pool=10)}
    monkeypatch.setenv("RAG_RRF4", "0")
    off = {c["rel"]: c["score"] for c in r.retrieve("一般語", k=6, pool=10)}
    assert on == off  # no rare token → the IDF-rarity signal adds nothing


# ------------------------------- version-decay (R3b) -----------------------------------------
_OLD = "青葉/提案書_旧版.docx"
_NEW = "青葉/提案書_最新.docx"


def _versioned_chunks():
    return [
        _chunk(0, "青葉", "提案書_旧版.docx", "text",
               "[案件: 青葉 | ファイル: 青葉/提案書_旧版.docx]\n提案書の契約金額と条件について記載。"),
        _chunk(1, "青葉", "提案書_最新.docx", "text",
               "[案件: 青葉 | ファイル: 青葉/提案書_最新.docx]\n提案書の契約金額と条件について記載。"),
    ]


def test_version_decay_downranks_old_file_for_normal_question(monkeypatch):
    r = _make_retriever(monkeypatch, _versioned_chunks())
    q = "提案書の契約金額と条件は何ですか"  # not a diff question

    monkeypatch.setenv("RAG_RRF4", "1")
    on = {c["rel"]: c["score"] for c in r.retrieve(q, k=2, pool=10)}
    monkeypatch.setenv("RAG_RRF4", "0")
    off = {c["rel"]: c["score"] for c in r.retrieve(q, k=2, pool=10)}

    # old file is decayed when the signal is on; the current edition is untouched
    assert on[_OLD] < off[_OLD]
    assert on[_NEW] == pytest.approx(off[_NEW])
    assert on[_NEW] > on[_OLD]


def test_version_decay_released_for_diff_question(monkeypatch):
    from src.rag import diffpair

    r = _make_retriever(monkeypatch, _versioned_chunks())
    q = "提案書の旧版と最新版の差分を教えてください"  # a version-diff question
    assert diffpair.is_diff_question(q) is True

    monkeypatch.setenv("RAG_RRF4", "1")
    on = {c["rel"]: c["score"] for c in r.retrieve(q, k=2, pool=10)}
    monkeypatch.setenv("RAG_RRF4", "0")
    off = {c["rel"]: c["score"] for c in r.retrieve(q, k=2, pool=10)}

    # decay is released for a diff question → the old version is NOT down-ranked
    assert on[_OLD] == pytest.approx(off[_OLD])
