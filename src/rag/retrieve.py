"""Hybrid retrieval: dense (Vertex embeddings) + sparse (BM25) with glossary-aware
query expansion and project-scoped boosting.

Fusion is Reciprocal Rank Fusion (RRF) over the two rankings — robust to score-scale
differences. When the question clearly names a project (via glossary), chunks from that
project are boosted so cross-project noise doesn't drown the right case.
"""
from __future__ import annotations

import functools
import re

import numpy as np
from rank_bm25 import BM25Okapi

from src.rag import index, llm
from src.rag.corpus import nfc
from src.rag.extract import glossary

# Filenames / identifiers a question may name explicitly (train.xlsx, figure_06.png,
# modeling.py, 01_eda.ipynb, スケジュール_r2.xlsx, T09 …). Chunks from a matching file are
# strongly boosted — questions very often pin the exact document.
_FILE_TOKEN = re.compile(r"[A-Za-z0-9_]+\.[A-Za-z0-9]+|[A-Za-z0-9_]{3,}")


class Retriever:
    def __init__(self) -> None:
        self.chunks = index.load_chunks()
        self.emb = index.load_embeddings()
        self.bm25 = BM25Okapi([index.tokenize(c["text"]) for c in self.chunks])
        self.glossary = glossary.load()

    def _expanded_query(self, question: str) -> str:
        extra = self.glossary.expand_terms(question)
        return question + (" " + " ".join(extra) if extra else "")

    def retrieve(self, question: str, k: int = 16, pool: int = 90) -> list[dict]:
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

        # explicit-file boost: question names a file/identifier that appears in the chunk's path
        q_tokens = {t.lower() for t in _FILE_TOKEN.findall(question) if not t.isdigit() or len(t) >= 2}
        q_files = {t for t in q_tokens if "." in t}
        if q_tokens:
            for i in list(rrf):
                rel = nfc(self.chunks[i]["rel"]).lower()
                fname = nfc(self.chunks[i]["file"]).lower()
                if q_files and any(f in fname for f in q_files):
                    rrf[i] *= 2.2                      # exact filename hit
                elif any(t in rel for t in q_tokens if len(t) >= 4):
                    rrf[i] *= 1.3                      # path/identifier hit

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
