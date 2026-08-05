"""Offline tests for the R4 LLM rerank + per-file cap + dedup + context expansion (SOT-2450).

    .venv/bin/python -m pytest scoring/test_rerank.py -q

The LLM is stubbed (``llm.rerank`` / ``llm.generate``) so ranking is deterministic and offline.
Cover: the reranker parses scores robustly; a high-scored candidate rises to the top; a per-file
cap + text dedup diversify the picks; adjacent body chunks are appended as context; and a rerank
failure degrades to the plain fusion order (safe no-op). Flag defaults OFF (champion path unchanged).
"""
from __future__ import annotations

import numpy as np

from src.rag import retrieve


# ------------------------------- flags -------------------------------------------------------
def test_rerank_flag_default_off_and_opt_in(monkeypatch):
    monkeypatch.delenv("RAG_RERANK", raising=False)
    assert retrieve.rerank_enabled() is False
    monkeypatch.setenv("RAG_RERANK", "1")
    assert retrieve.rerank_enabled() is True
    monkeypatch.setenv("RAG_RERANK", "0")
    assert retrieve.rerank_enabled() is False


# ------------------------------- llm.rerank parsing ------------------------------------------
def test_llm_rerank_parses_scores_and_handles_bad_indices(monkeypatch):
    from src.rag import llm

    payload = '{"scores": [{"index": 0, "score": 7}, {"index": 2, "score": 9}, ' \
              '{"index": 5, "score": 3}, {"index": 1, "score": "x"}]}'
    monkeypatch.setattr(llm, "generate", lambda *a, **k: payload)
    scores = llm.rerank("q", ["a", "b", "c"])
    assert scores == [7.0, 0.0, 9.0]  # idx5 out of range dropped; idx1 unparceable → 0.0


def test_llm_rerank_failure_returns_zeros(monkeypatch):
    from src.rag import llm

    def _boom(*a, **k):
        raise RuntimeError("no LLM")

    monkeypatch.setattr(llm, "generate", _boom)
    assert llm.rerank("q", ["a", "b"]) == [0.0, 0.0]
    assert llm.rerank("q", []) == []


# ------------------------------- retriever stub ----------------------------------------------
class _Glossary:
    def expand_terms(self, q):
        return []

    def company_of(self, q):
        return None


def _chunk(cid, file, text, kind="text", project="青葉", rel=None):
    return {"id": cid, "project": project, "category": "", "file": file,
            "rel": rel or f"{project}/{file}", "kind": kind, "text": text}


def _make_retriever(monkeypatch, chunks):
    from src.rag import index, llm

    monkeypatch.setattr(index, "load_chunks", lambda: chunks)
    monkeypatch.setattr(index, "load_embeddings",
                        lambda: np.zeros((len(chunks), 4), dtype=np.float32))
    monkeypatch.setattr(retrieve.glossary, "load", lambda: _Glossary())
    monkeypatch.setattr(llm, "embed", lambda texts, **k: [[0.0] * 4 for _ in texts])
    monkeypatch.setenv("RAG_RERANK", "1")
    return retrieve.Retriever()


def _primaries(out):
    return [c for c in out if not c.get("context_expanded")]


# ------------------------------- rerank reorders ---------------------------------------------
def test_rerank_lifts_high_scored_candidate_to_top(monkeypatch):
    chunks = [
        _chunk(0, "a.txt", "共通語 の 一般的な 文書 A"),
        _chunk(1, "b.txt", "共通語 の 一般的な 文書 B ANSWER ここに答え"),
        _chunk(2, "c.txt", "共通語 の 一般的な 文書 C"),
    ]
    r = _make_retriever(monkeypatch, chunks)
    # score the chunk containing ANSWER highest, regardless of fusion order
    monkeypatch.setattr(retrieve.llm, "rerank",
                        lambda q, texts, **k: [10.0 if "ANSWER" in t else 1.0 for t in texts])
    out = _primaries(r.retrieve("共通語", k=3, pool=10))
    assert out[0]["rel"] == "青葉/b.txt"


def test_rerank_disabled_keeps_fusion_order(monkeypatch):
    chunks = [_chunk(i, f"{c}.txt", f"共通語 文書 {c} ANSWER" if c == "c" else f"共通語 文書 {c}")
              for i, c in enumerate(["a", "b", "c"])]
    r = _make_retriever(monkeypatch, chunks)
    monkeypatch.setattr(retrieve.llm, "rerank",
                        lambda q, texts, **k: [10.0 if "ANSWER" in t else 1.0 for t in texts])
    monkeypatch.setenv("RAG_RERANK", "0")  # off → rerank must not run
    out = r.retrieve("共通語", k=3, pool=10)
    assert not any(c.get("context_expanded") for c in out)  # no expansion when disabled
    top_off = [c["rel"] for c in out]
    monkeypatch.setenv("RAG_RERANK", "1")
    top_on = [c["rel"] for c in _primaries(r.retrieve("共通語", k=3, pool=10))]
    assert top_on[0] == "青葉/c.txt" and top_on != top_off  # rerank changed the order


# ------------------------------- per-file cap + dedup ----------------------------------------
def test_per_file_cap_diversifies_picks(monkeypatch):
    monkeypatch.setenv("RAG_PER_FILE_CAP", "1")
    chunks = [
        _chunk(0, "a.txt", "共通語 文書 A 第一段落"),
        _chunk(1, "a.txt", "共通語 文書 A 第二段落"),   # same file as chunk0
        _chunk(2, "b.txt", "共通語 文書 B"),
    ]
    # note: a.txt has two consecutive chunks → they are neighbours; b.txt is standalone.
    r = _make_retriever(monkeypatch, chunks)
    monkeypatch.setattr(retrieve.llm, "rerank",
                        lambda q, texts, **k: [10.0, 9.0, 5.0])  # both a.txt chunks score highest
    prim = _primaries(r.retrieve("共通語", k=2, pool=10))
    rels = [c["rel"] for c in prim]
    assert rels.count("青葉/a.txt") == 1        # cap=1 → only one a.txt among primary picks
    assert "青葉/b.txt" in rels                  # diversified to include b.txt


def test_dedup_drops_identical_text(monkeypatch):
    dup = "共通語 まったく同じ本文"
    chunks = [
        _chunk(0, "a.txt", dup),
        _chunk(1, "b.txt", dup),          # identical text, different file
        _chunk(2, "c.txt", "共通語 別の本文"),
    ]
    r = _make_retriever(monkeypatch, chunks)
    monkeypatch.setattr(retrieve.llm, "rerank", lambda q, texts, **k: [10.0, 10.0, 8.0])
    prim = _primaries(r.retrieve("共通語", k=3, pool=10))
    texts = [retrieve._dedup_key(c["text"]) for c in prim]
    assert len(texts) == len(set(texts))                 # no duplicate text survives
    assert retrieve._dedup_key(dup) in texts             # one copy kept
    assert retrieve._dedup_key("共通語 別の本文") in texts


# ------------------------------- context expansion -------------------------------------------
def test_context_expansion_appends_adjacent_body_chunk(monkeypatch):
    chunks = [
        _chunk(0, "長文.docx", "共通語 長い文書の前半 ANSWER"),
        _chunk(1, "長文.docx", "共通語 長い文書の後半（表の続き）"),  # neighbour of chunk0
        _chunk(2, "他.txt", "共通語 無関係"),
    ]
    r = _make_retriever(monkeypatch, chunks)
    # only the first chunk is picked (k=1); its neighbour must be pulled in as context
    monkeypatch.setattr(retrieve.llm, "rerank",
                        lambda q, texts, **k: [10.0 if "ANSWER" in t else 1.0 for t in texts])
    out = r.retrieve("共通語", k=1, pool=10)
    prim = _primaries(out)
    expanded = [c for c in out if c.get("context_expanded")]
    assert [c["rel"] for c in prim] == ["青葉/長文.docx"]
    assert any(c["id"] == 1 for c in expanded)           # the adjacent second chunk was appended
    assert all(c["score"] == 0.0 for c in expanded)      # expansion chunks don't re-rank


def test_fact_rows_have_no_neighbours(monkeypatch):
    # A fact row (kind=="fact") is not a neighbouring paragraph, so it breaks the adjacency run.
    chunks = [
        _chunk(0, "d.docx", "本文チャンク"),
        _chunk(1, "d.docx", "[fact | 種別: 太字] 重要語", kind="fact"),
        _chunk(2, "d.docx", "続きの本文チャンク"),
    ]
    r = _make_retriever(monkeypatch, chunks)
    assert r._neighbors.get(1, []) == []                 # the fact row has no neighbours
    assert 1 not in r._neighbors.get(0, [])              # and isn't anyone else's neighbour
    assert 1 not in r._neighbors.get(2, [])


def test_rerank_failure_falls_back_to_fusion_order(monkeypatch):
    chunks = [_chunk(i, f"{c}.txt", f"共通語 文書 {c}") for i, c in enumerate(["a", "b", "c"])]
    r = _make_retriever(monkeypatch, chunks)
    monkeypatch.setattr(retrieve.llm, "rerank", lambda q, texts, **k: [0.0] * len(texts))
    prim = [c["rel"] for c in _primaries(r.retrieve("共通語", k=3, pool=10))]
    # all-zero rerank → stable sort preserves fusion order (same as the disabled top-k order)
    monkeypatch.setenv("RAG_RERANK", "0")
    fusion = [c["rel"] for c in r.retrieve("共通語", k=3, pool=10)]
    assert prim == fusion
