"""Hybrid retrieval: dense (Vertex embeddings) + sparse (BM25) with glossary-aware
query expansion and project-scoped boosting.

Fusion is Reciprocal Rank Fusion (RRF) over the two rankings — robust to score-scale
differences. When the question clearly names a project (via glossary), chunks from that
project are boosted so cross-project noise doesn't drown the right case.
"""
from __future__ import annotations

import functools
import os
import re

import numpy as np
from rank_bm25 import BM25Okapi

from src.rag import archetype, diffpair, facts, index, llm
from src.rag.corpus import nfc
from src.rag.extract import glossary

_ON = {"1", "true", "yes", "on"}


# --- R3 (4-signal RRF fusion) & R4 (LLM rerank + context expansion) feature flags (SOT-2450) ---
# Both default OFF (opt-in), independently toggleable. With a flag unset the corresponding layer is
# a structural no-op, so the champion serving path (dense + BM25 RRF + existing boosts) stays
# byte-identical. They ship opt-in for the same hard reason as R1 (facts): reranking retrieval can
# change committed answers, and this project's lesson is that local proxies (valid30 / holdout) are
# non-predictive — #4/#5 regressed the real leaderboard despite good proxies — so the production path
# must not change until confirmed on 関門3 (the real SIGNATE leaderboard), currently blocked on human
# re-auth. See docs/ai/experiment_ledger.jsonl (SOT-2450).
def rrf4_enabled() -> bool:
    """R3: add the IDF-rarity + version-decay signals to the dense+BM25 RRF fusion."""
    return os.getenv("RAG_RRF4", "0").strip().lower() in _ON


def rerank_enabled() -> bool:
    """R4: LLM rerank the fused top, cap per file, dedup, then expand adjacent-chunk context."""
    return os.getenv("RAG_RERANK", "0").strip().lower() in _ON


# R3 tunables (env-overridable; read at call time so tests can vary them).
def _idf_min() -> float:
    """Minimum BM25 IDF for a query token to count as a rare needle worth an independent signal."""
    return float(os.getenv("RAG_IDF_MIN", "2.0"))


def _version_decay() -> float:
    """Multiplier applied to chunks from an explicitly OLD-version file (unless a diff question)."""
    return float(os.getenv("RAG_VERSION_DECAY", "0.8"))


# R4 tunables.
def _rerank_pool() -> int:
    return int(os.getenv("RAG_RERANK_POOL", "24"))


def _per_file_cap() -> int:
    return int(os.getenv("RAG_PER_FILE_CAP", "3"))


_RRF_K = 60

# Explicit older-version markers in a filename. Only these decay — an ambiguous r2/v3 token is NOT
# decayed (it may be the newest edition), and 最新/新版 never match, so the current file is never
# demoted. Kept narrow on purpose to avoid false down-ranking.
_OLD_VERSION = re.compile(r"(旧版|旧ファイル|旧バージョン|前回版|初版|old)", re.IGNORECASE)


def _is_old_version(name: str) -> bool:
    return bool(_OLD_VERSION.search(nfc(name)))


def _dedup_key(text: str) -> str:
    return re.sub(r"\s+", "", nfc(text))

# Filenames / identifiers a question may name explicitly (train.xlsx, figure_06.png,
# modeling.py, 01_eda.ipynb, スケジュール_r2.xlsx, T09 …). Chunks from a matching file are
# strongly boosted — questions very often pin the exact document.
_FILE_TOKEN = re.compile(r"[A-Za-z0-9_]+\.[A-Za-z0-9]+|[A-Za-z0-9_]{3,}")

# Archetypes whose answers live in an extracted artifact (a highlighted cell / bold run /
# Pivot condition / code parameter). A matching fact row is exactly the evidence they need,
# so it gets the stronger lift; other archetypes get only a mild lift (a precise short fact
# is still worth surfacing, but must not displace a genuinely better chunk). See SOT-2449.
_FACT_FAVORING = frozenset({
    "highlight_set", "enum_set", "document_extract", "fact_lookup", "pivot_condition",
    "config_model_type", "config_hyperparam",
})
_FACT_BOOST_STRONG = 1.8
_FACT_BOOST_MILD = 1.25


class Retriever:
    def __init__(self) -> None:
        self.chunks = index.load_chunks()
        self.emb = index.load_embeddings()
        self._chunk_tokens = [index.tokenize(c["text"]) for c in self.chunks]
        self.bm25 = BM25Okapi(self._chunk_tokens)
        self.glossary = glossary.load()
        # Inverted postings (token -> chunk indices) for the R3 IDF-rarity signal, so a rare query
        # token is scored without scanning every chunk. Built once; a structural cost only.
        self._postings: dict[str, list[int]] = {}
        for i, toks in enumerate(self._chunk_tokens):
            for t in set(toks):
                self._postings.setdefault(t, []).append(i)
        self._neighbors = self._build_neighbors()

    def _build_neighbors(self) -> dict[int, list[int]]:
        """Map each body chunk id -> its adjacent same-file body chunk ids (R4 context expansion).

        Body chunks of a document are contiguous in build order; fact rows (kind=="fact") are not
        neighbouring paragraphs, so they break a run and get no neighbours. The result lets R4 pull
        the previous/next paragraph (a split table row, a continued sentence) around a picked chunk.
        """
        neighbors: dict[int, list[int]] = {}

        def flush(run: list[int]) -> None:
            for pos, cid in enumerate(run):
                nb: list[int] = []
                if pos > 0:
                    nb.append(run[pos - 1])
                if pos < len(run) - 1:
                    nb.append(run[pos + 1])
                neighbors[cid] = nb

        run: list[int] = []
        prev_rel: str | None = None
        for i, c in enumerate(self.chunks):
            if c.get("kind") == "fact":
                flush(run)
                run, prev_rel = [], None
                continue
            if c["rel"] != prev_rel:
                flush(run)
                run, prev_rel = [i], c["rel"]
            else:
                run.append(i)
        flush(run)
        return neighbors

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

        # R3 signal (a): IDF-rarity. A third fused ranking that surfaces chunks matching RARE query
        # tokens (型番 / パラメータ名 / codes) independently of dense+BM25, so a precise lexical needle
        # isn't averaged away when the rest of the query is generic. Each chunk scores by the summed
        # BM25 IDF of the rare query tokens it contains; the ranking joins the RRF at k=60. Opt-in
        # (RAG_RRF4) — a structural no-op otherwise.
        if rrf4_enabled():
            q_toks = set(index.tokenize(q_expanded))
            rare = [(t, self.bm25.idf.get(t, 0.0)) for t in q_toks]
            rare = [(t, w) for t, w in rare if w >= _idf_min()]
            if rare:
                rarity = np.zeros(len(self.chunks), dtype=np.float32)
                for t, w in rare:
                    for i in self._postings.get(t, ()):
                        rarity[i] += w
                for r, i in enumerate(np.argsort(-rarity)[:pool]):
                    if rarity[int(i)] <= 0.0:
                        break
                    rrf[int(i)] = rrf.get(int(i), 0.0) + 1.0 / (_RRF_K + r)

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

        # fact-row boost (SOT-2449): lift high-signal 1-fact-per-line rows so the exact
        # highlighted cell / bold term / Pivot condition / code parameter isn't buried under
        # long chunks. Additive — only affects fact rows already retrieved into the pool, and a
        # structural no-op when fact indexing is disabled (no kind=="fact" rows exist).
        if facts.enabled():
            boost = _FACT_BOOST_STRONG if archetype.classify(question) in _FACT_FAVORING \
                else _FACT_BOOST_MILD
            for i in list(rrf):
                if self.chunks[i].get("kind") == "fact":
                    rrf[i] *= boost

        # R3 signal (b): version-decay. Down-rank chunks from an explicitly OLD-version file so a
        # superseded document doesn't outrank the current edition — RELEASED when the question is a
        # version-diff question, where the old version is exactly what's being compared, so 旧版参照
        # 設問を壊さない. Opt-in (RAG_RRF4); narrow OLD markers only (最新/r-tokens never decay).
        if rrf4_enabled() and not diffpair.is_diff_question(question):
            decay = _version_decay()
            for i in list(rrf):
                if _is_old_version(self.chunks[i]["file"]):
                    rrf[i] *= decay

        order = sorted(rrf, key=lambda i: -rrf[i])

        # R4 (SOT-2450): LLM rerank the fused top, cap per file + dedup for diversity, then expand
        # adjacent-chunk context. Opt-in (RAG_RERANK); when off, keep the exact champion behaviour
        # (top-k by fused score, no rerank/cap/expansion) so the serving path is byte-identical.
        if rerank_enabled():
            chosen = self._rerank_and_diversify(question, order, k, rrf)
            out = self._materialize(chosen, rrf)
            return self._expand_context(chosen, out)

        return self._materialize(order[:k], rrf)

    def _materialize(self, indices: list[int], rrf: dict[int, float]) -> list[dict]:
        out = []
        for i in indices:
            c = dict(self.chunks[i])
            c["score"] = rrf.get(i, 0.0)
            out.append(c)
        return out

    def _rerank_and_diversify(self, question: str, order: list[int], k: int,
                              rrf: dict[int, float]) -> list[int]:
        """LLM-rerank the fused top, then select k with a per-file cap and text dedup for diversity.

        Returns global chunk indices. Ties in the rerank score keep the fusion order (stable sort).
        If the cap/dedup leave fewer than k picks, backfill from the remaining fusion order so the
        evidence count stays stable. A rerank failure yields all-zero scores → pure fusion order.
        """
        cand = order[:_rerank_pool()]
        texts = [self.chunks[i]["text"] for i in cand]
        scores = llm.rerank(question, texts)
        ranked_pos = sorted(range(len(cand)),
                            key=lambda p: -(scores[p] if p < len(scores) else 0.0))
        per_file: dict[str, int] = {}
        seen_text: set[str] = set()
        cap = _per_file_cap()
        chosen: list[int] = []
        for p in ranked_pos:
            i = cand[p]
            rel = nfc(self.chunks[i]["rel"])
            key = _dedup_key(self.chunks[i]["text"])
            if key in seen_text or per_file.get(rel, 0) >= cap:
                continue
            seen_text.add(key)
            per_file[rel] = per_file.get(rel, 0) + 1
            chosen.append(i)
            if len(chosen) >= k:
                return chosen
        if len(chosen) < k:  # backfill so k stays stable under a tight per-file cap (cap relaxed,
            picked = set(chosen)  # but dedup still honoured — a duplicate slot adds no evidence)
            for i in order:
                if i in picked:
                    continue
                key = _dedup_key(self.chunks[i]["text"])
                if key in seen_text:
                    continue
                seen_text.add(key)
                chosen.append(i)
                if len(chosen) >= k:
                    break
        return chosen

    def _expand_context(self, chosen: list[int], out: list[dict]) -> list[dict]:
        """Append the adjacent same-file body chunks (prev/next paragraph, split table row) of each
        picked chunk, so a fact whose context straddled a chunk boundary isn't read out of context.
        Appended chunks are marked ``context_expanded`` and carry score 0 (they don't re-rank)."""
        have = set(chosen)
        extra: list[dict] = []
        for gi in chosen:
            for nb in self._neighbors.get(gi, ()):
                if nb in have:
                    continue
                have.add(nb)
                nc = dict(self.chunks[nb])
                nc["score"] = 0.0
                nc["context_expanded"] = True
                extra.append(nc)
        return out + extra


@functools.lru_cache(maxsize=1)
def get() -> Retriever:
    return Retriever()
