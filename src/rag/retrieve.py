"""Hybrid retrieval: dense (Vertex embeddings) + sparse (BM25) with glossary-aware
query expansion and project-scoped boosting.

Fusion is Reciprocal Rank Fusion (RRF) over the two rankings — robust to score-scale
differences. When the question clearly names a project (via glossary), chunks from that
project are boosted so cross-project noise doesn't drown the right case.
"""
from __future__ import annotations

import functools

import numpy as np
from rank_bm25 import BM25Okapi

from src.rag import index, llm
from src.rag.corpus import nfc
from src.rag.extract import glossary


class Retriever:
    def __init__(self) -> None:
        self.chunks = index.load_chunks()
        self.emb = index.load_embeddings()
        self.bm25 = BM25Okapi([index.tokenize(c["text"]) for c in self.chunks])
        self.glossary = glossary.load()

    def _expanded_query(self, question: str) -> str:
        extra = self.glossary.expand_terms(question)
        return question + (" " + " ".join(extra) if extra else "")

    def retrieve(self, question: str, k: int = 12, pool: int = 60) -> list[dict]:
        q_expanded = self._expanded_query(question)

        # dense
        qv = np.asarray(llm.embed([q_expanded], task_type="RETRIEVAL_QUERY")[0], dtype=np.float32)
        qv /= (np.linalg.norm(qv) + 1e-8)
        dense_scores = self.emb @ qv
        dense_rank = np.argsort(-dense_scores)[:pool]

        # sparse
        sparse_scores = self.bm25.get_scores(index.tokenize(q_expanded))
        sparse_rank = np.argsort(-sparse_scores)[:pool]

        # RRF fusion
        rrf: dict[int, float] = {}
        for r, i in enumerate(dense_rank):
            rrf[int(i)] = rrf.get(int(i), 0.0) + 1.0 / (60 + r)
        for r, i in enumerate(sparse_rank):
            rrf[int(i)] = rrf.get(int(i), 0.0) + 1.0 / (60 + r)

        # project boost
        target = self.glossary.company_of(question)
        if target:
            for i in list(rrf):
                if nfc(self.chunks[i]["project"]) == nfc(target):
                    rrf[i] *= 1.6

        ranked = sorted(rrf, key=lambda i: -rrf[i])[:k]
        out = []
        for i in ranked:
            c = dict(self.chunks[i])
            c["score"] = rrf[i]
            out.append(c)
        return out


@functools.lru_cache(maxsize=1)
def get() -> Retriever:
    return Retriever()
