"""unified_search — one-call fan-out retrieval with RRF fusion + context re-expansion (SOT-2659).

Cerebras knowledge-base 教訓移植 3/5 (parent PLAN SOT-2602). The Sonnet investigator lane exhausts its
turn budget by trying the deterministic retrievers *one at a time* — text_search, then id_lookup, then
diff_lookup, then registry routing — a 12〜18 ターンの逐次往復 that BUDGET_EXHAUSTED 25/30 の主犯である。
This tool compresses that N-turn probe into ONE call: it fans the query out over every query-addressable
retriever **in parallel**, fuses their ranked lists with Reciprocal Rank Fusion (``Σ weight/(K+rank)``,
K=60 — Cerebras と同定数), merges duplicate hits, caps per-file contribution for diversity, and re-expands
the winning hits with neighbouring context, returning a single "evidence packet".

Cerebras の核心 — 「どの単一スコアラーも単独では信用しない」 — をそのまま採る: a hit that several retrievers
independently rank highly beats a hit that only one retriever puts first. The fusion is where that consensus
is computed.

Serve-time invariants (mirrors every other opt-in tool on this corpus):

* **``RAG_UNIFIED_SEARCH`` default OFF ⇒ byte-identical.** With the flag unset :func:`enabled` is False, the
  investigator does not register the ``search`` tool, and the champion serve tool set / function-call schema
  / MCP surface are unchanged.
* **LLM-free / Gemini-free at serve time.** Every internal retriever is a deterministic index/store lookup or
  file router — no query embedding (a Gemini call), no generation. 本コーパスは ID・固有語支配なので Cerebras
  自身の「リテラルは字句一致が最良」に整合する。
* **Fail-open, never raises into the answer path.** A retriever that is disabled, absent, or errors simply
  contributes zero hits; the packet is still returned (with that retriever recorded as run-with-0 or skipped).

The internal retrievers reuse the EXISTING read helpers (no reimplementation): the FTS lexical index
(:mod:`src.rag.index.text_fts`), the distillation store (:mod:`src.rag.index.distill_store`), the ID master
reverse-lookup (:mod:`src.rag.index.id_master`), the version-pair diff store (via
:func:`src.rag.agent.fact_layer._diff_lookup`), and canonical registry routing
(:func:`src.rag.tools.canonical_route.canonical_route`). Store lookups that need a *structured* bind the query
does not carry (case_filter=属性等式, metric_lookup=案件+指標, format_events=対象ファイル) stay individual tools —
per the tool discipline they are used to *narrow* a ``search`` result, not to start the probe.
"""
from __future__ import annotations

import os
import re
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Any, Callable, Iterable, Mapping, Sequence

from src.rag.tools import contract as _contract

_ON = {"1", "true", "yes", "on"}

# Fusion constants (Cerebras: K=60). All env-tunable so a cycle can retune without a code change.
_DEFAULT_RRF_K = 60
_DEFAULT_LIMIT = 10
_MAX_LIMIT = 50
_DEFAULT_PER_FILE_CAP = 3
# Per-retriever hits pulled before fusion (a deep-enough candidate pool for RRF to find consensus).
_RETRIEVER_LIMIT = 20


def enabled() -> bool:
    """True when the unified ``search`` tool should be active (default OFF — opt-in).

    Gated by ``RAG_UNIFIED_SEARCH``. Off (the default) ⇒ :func:`search` is a no-op returning an empty
    packet and the investigator never registers the tool, so the champion serve surface is byte-identical.
    """
    return os.getenv("RAG_UNIFIED_SEARCH", "0").strip().lower() in _ON


def _env_float(key: str, default: float) -> float:
    try:
        return float(os.getenv(key, "").strip())
    except (TypeError, ValueError):
        return default


def _env_int(key: str, default: int) -> int:
    try:
        return int(os.getenv(key, "").strip())
    except (TypeError, ValueError):
        return default


def _rrf_k() -> int:
    return max(1, _env_int("RAG_UNIFIED_RRF_K", _DEFAULT_RRF_K))


def _per_file_cap() -> int:
    return max(1, _env_int("RAG_UNIFIED_PER_FILE_CAP", _DEFAULT_PER_FILE_CAP))


# --------------------------------------------------------------------------- retriever abstraction
@dataclass
class Retriever:
    """A single query-addressable retriever bound into the fan-out.

    ``fetch(query, hints)`` returns a RANKED list of hit dicts (best first). Each hit needs at least a
    ``doc_id`` and ``text``; ``locator`` defaults to ``doc:<doc_id>`` and ``context`` is optional. ``weight``
    scales this retriever's RRF contribution (``weight/(K+rank)``) and is read from
    ``RAG_UNIFIED_W_<NAME>`` at build time (default 1.0).
    """

    name: str
    fetch: Callable[[str, Mapping[str, Any]], "list[dict[str, Any]]"]
    weight: float = 1.0


def _weight_for(name: str, default: float = 1.0) -> float:
    return _env_float(f"RAG_UNIFIED_W_{name.upper()}", default)


def _norm_hit(raw: Mapping[str, Any]) -> "dict[str, Any] | None":
    """Normalise a retriever record into ``{doc_id, locator, text, context, extra}`` (drop if no doc_id)."""
    doc_id = raw.get("doc_id")
    if not doc_id:
        return None
    locator = str(raw.get("locator") or f"doc:{doc_id}")
    return {
        "doc_id": str(doc_id),
        "locator": locator,
        "text": str(raw.get("text") or "")[:2000],
        "context": str(raw.get("context") or "")[:2000],
    }


# --------------------------------------------------------------------------- default serve retrievers
def _extract_id_tokens(query: str, *, limit: int = 5) -> "list[str]":
    """Pull ID-shaped literals (letter+digit / hyphen-number / APR-Mx …) from the query for id_master.

    Conservative on purpose: only alnum tokens that mix letters and digits, or a hyphen-joined code, so a
    generic word never triggers an id_master scan. Separator-insensitive matching is id_master's own job.
    """
    toks: list[str] = []
    seen: set[str] = set()
    # one-or-more letter groups joined by -_. , ending in digits (EXT1234 / APR-M3 / M-01 / WBS_12 / AI08).
    for m in re.finditer(r"[A-Za-z]+(?:[-_.][A-Za-z]+)*[-_.]?\d{1,5}", query):
        t = m.group(0).strip()
        key = re.sub(r"[-_.]", "", t).lower()
        if key and key not in seen:
            seen.add(key)
            toks.append(t)
        if len(toks) >= limit:
            break
    return toks


def _fts_retriever() -> "Retriever | None":
    from src.rag.index import text_fts
    if not text_fts.enabled():
        return None

    def fetch(query: str, hints: Mapping[str, Any]) -> "list[dict[str, Any]]":
        hits = text_fts.search(query, limit=_RETRIEVER_LIMIT,
                               project=hints.get("project"), ext=hints.get("ext"))
        return [{"doc_id": h.get("doc_id"), "locator": h.get("locator"), "text": h.get("text")}
                for h in hits]

    return Retriever("fts", fetch, _weight_for("fts"))


def _distill_retriever() -> "Retriever | None":
    from src.rag.index import distill_store
    if not distill_store.enabled():
        return None

    def fetch(query: str, hints: Mapping[str, Any]) -> "list[dict[str, Any]]":
        pairs = distill_store.search(query, limit=_RETRIEVER_LIMIT)
        if not pairs:
            return []
        by_doc: dict[str, Mapping[str, Any]] = {}
        for rec in distill_store.load():
            by_doc.setdefault(str(rec.get("doc_id")), rec)
        out: list[dict[str, Any]] = []
        for doc_id, _score in pairs:
            rec = by_doc.get(str(doc_id), {})
            text = rec.get("summary") or "; ".join(rec.get("key_facts") or [])
            context = "; ".join(rec.get("answerable_questions") or [])
            unit = rec.get("unit_key") or rec.get("unit") or "doc"
            out.append({"doc_id": doc_id, "locator": f"unit:{unit}", "text": text, "context": context})
        return out

    return Retriever("distill", fetch, _weight_for("distill"))


def _id_master_retriever() -> "Retriever | None":
    from src.rag.index import id_master
    if not id_master.enabled():
        return None

    def fetch(query: str, hints: Mapping[str, Any]) -> "list[dict[str, Any]]":
        out: list[dict[str, Any]] = []
        for tok in _extract_id_tokens(query):
            for o in id_master.lookup(tok, project=hints.get("project")):
                loc = (f"cell:{o.get('sheet')}!{o.get('row')}" if o.get("sheet")
                       else f"slide:{o.get('slide')}" if o.get("slide")
                       else f"page:{o.get('page')}" if o.get("page")
                       else f"row:{o.get('row')}" if o.get("row") is not None
                       else f"line:{o.get('line')}")
                cols = o.get("columns") or []
                out.append({"doc_id": o.get("doc_id"), "locator": loc,
                            "text": o.get("line") or o.get("label") or str(o.get("id")),
                            "context": " | ".join(str(c) for c in cols)})
        return out

    return Retriever("id_master", fetch, _weight_for("id_master"))


def _diff_retriever() -> "Retriever | None":
    from src.rag.index import diff_store
    if not diff_store.enabled():
        return None
    from src.rag.agent import fact_layer

    def fetch(query: str, hints: Mapping[str, Any]) -> "list[dict[str, Any]]":
        res = fact_layer._diff_lookup(question=query, project=hints.get("project") or "")
        val = (res or {}).get("value")
        if not isinstance(val, list):
            return []
        ev = res.get("evidence") or {}
        doc = ev.get("new_file") or ev.get("old_file")
        out: list[dict[str, Any]] = []
        for c in val:
            out.append({"doc_id": doc, "locator": str(c.get("structural_location") or f"rank:{c.get('rank')}"),
                        "text": f"{c.get('before')} → {c.get('after')}",
                        "context": str(c.get("intent") or "")})
        return out

    return Retriever("diff_store", fetch, _weight_for("diff_store"))


def _registry_retriever() -> "Retriever | None":
    # canonical_route has no serve flag — it is always-available deterministic file routing; fold it in when
    # unified search is enabled so the query's target 案件ファイル (train.xlsx/分析コード/notebook) is one of
    # the fused sources (exact→alias→family→entity→lexical resolution).
    from src.rag.tools import canonical_route

    def fetch(query: str, hints: Mapping[str, Any]) -> "list[dict[str, Any]]":
        res = canonical_route.canonical_route(question=query, project=hints.get("project"))
        val = (res or {}).get("value")
        if not isinstance(val, list):
            return []
        out: list[dict[str, Any]] = []
        for f in val:
            if not isinstance(f, Mapping):
                continue
            rel = f.get("rel")
            out.append({"doc_id": rel, "locator": f"file:{f.get('kind') or f.get('ext') or ''}",
                        "text": str(rel or ""),
                        "context": f"project={f.get('project')} category={f.get('category')} "
                                   f"kind={f.get('kind')}"})
        return out

    return Retriever("registry", fetch, _weight_for("registry", 0.7))


def _default_retrievers() -> "list[Retriever]":
    """Build the serve-path retriever set (each guarded by its own flag; fail-open on import error)."""
    builders = (_fts_retriever, _distill_retriever, _id_master_retriever,
                _diff_retriever, _registry_retriever)
    out: list[Retriever] = []
    for build in builders:
        try:
            r = build()
        except Exception:  # noqa: BLE001 — a broken retriever must never break the fan-out
            r = None
        if r is not None:
            out.append(r)
    return out


# --------------------------------------------------------------------------- fusion core
@dataclass
class _Fused:
    doc_id: str
    locator: str
    text: str = ""
    context: str = ""
    score: float = 0.0
    sources: set = field(default_factory=set)


def _run_retrievers(retrievers: Sequence[Retriever], query: str, hints: Mapping[str, Any],
                    ) -> "tuple[dict[str, list[dict[str, Any]]], list[str]]":
    """Run every retriever concurrently, fail-open. Returns (name→normalised hits, names_run)."""
    results: dict[str, list[dict[str, Any]]] = {}
    names_run: list[str] = []

    def _one(r: Retriever) -> "tuple[str, list[dict[str, Any]]]":
        try:
            raw = r.fetch(query, hints) or []
            hits = [h for h in (_norm_hit(x) for x in raw if isinstance(x, Mapping)) if h]
        except Exception:  # noqa: BLE001 — retriever error ⇒ zero contribution, never a raised answer path
            hits = []
        return r.name, hits

    if not retrievers:
        return results, names_run
    with ThreadPoolExecutor(max_workers=len(retrievers)) as pool:
        for name, hits in pool.map(_one, retrievers):
            results[name] = hits
            names_run.append(name)
    return results, names_run


def _fuse(results: Mapping[str, Sequence[Mapping[str, Any]]], weights: Mapping[str, float], k: int,
          ) -> "list[_Fused]":
    """Reciprocal Rank Fusion: ``score(hit) = Σ_retriever weight/(k + rank)`` (rank 1-based).

    Duplicate hits (same doc_id+locator across retrievers) merge into one entry whose score is the SUM of
    each retriever's contribution and whose ``sources`` records every retriever that surfaced it — so a
    multi-retriever consensus hit outscores any single retriever's rank-1.
    """
    fused: dict[tuple[str, str], _Fused] = {}
    for name, hits in results.items():
        w = weights.get(name, 1.0)
        for rank, hit in enumerate(hits, start=1):
            key = (hit["doc_id"], hit["locator"])
            entry = fused.get(key)
            if entry is None:
                entry = _Fused(doc_id=hit["doc_id"], locator=hit["locator"],
                               text=hit.get("text", ""), context=hit.get("context", ""))
                fused[key] = entry
            entry.score += w / (k + rank)
            entry.sources.add(name)
            # keep the first non-empty text/context seen (retrievers ordered by trust in _default_retrievers)
            if not entry.text and hit.get("text"):
                entry.text = hit["text"]
            if not entry.context and hit.get("context"):
                entry.context = hit["context"]
    return list(fused.values())


def _cap_and_limit(entries: Sequence[_Fused], per_file_cap: int, limit: int) -> "list[_Fused]":
    """Order by fused score (desc, deterministic tie-break) then enforce a per-doc cap for diversity."""
    ordered = sorted(entries, key=lambda e: (-e.score, e.doc_id, e.locator))
    per_doc: dict[str, int] = {}
    out: list[_Fused] = []
    for e in ordered:
        if per_doc.get(e.doc_id, 0) >= per_file_cap:
            continue
        per_doc[e.doc_id] = per_doc.get(e.doc_id, 0) + 1
        out.append(e)
        if len(out) >= limit:
            break
    return out


def _reexpand_context(winners: Sequence[_Fused], all_entries: Sequence[_Fused]) -> None:
    """Cerebras context re-expansion: attach neighbouring evidence lost to chunking.

    For each winning hit we keep any retriever-supplied context; then we append the text of the OTHER fused
    hits from the SAME doc_id (its siblings in the retrieved set) as surrounding 近傍 context. This restores
    the 見出し/前提/注意書き a single chunk drops — deterministically, from the already-retrieved neighbours,
    with no extra corpus re-read (LLM-free).
    """
    by_doc: dict[str, list[_Fused]] = {}
    for e in all_entries:
        by_doc.setdefault(e.doc_id, []).append(e)
    for w in winners:
        sibs = [s for s in sorted(by_doc.get(w.doc_id, []), key=lambda e: (-e.score, e.locator))
                if (s.doc_id, s.locator) != (w.doc_id, w.locator) and s.text]
        neighbours = " ⋯ ".join(dict.fromkeys(s.text for s in sibs[:2]))
        parts = [p for p in (w.context, neighbours) if p]
        w.context = (" ⋯ ".join(parts))[:2000]


# --------------------------------------------------------------------------- public entrypoint
def search(query: str, *, project: str | None = None, ext: str | Iterable[str] | None = None,
           limit: int = _DEFAULT_LIMIT, hints: Mapping[str, Any] | None = None,
           retrievers: Sequence[Retriever] | None = None) -> "dict[str, Any]":
    """One-call fan-out search → RRF fusion → context re-expansion, as a ``{value, evidence, method}`` packet.

    Parameters
    ----------
    query
        A literal / natural-language probe. Passed to every retriever as-is (each tokenises per its own rules).
    project, ext
        Optional filters folded into ``hints`` and forwarded to the retrievers that accept them.
    limit
        Max fused results returned (default 10, capped at 50). A per-file cap (``RAG_UNIFIED_PER_FILE_CAP``,
        default 3) is applied first for source diversity.
    hints
        Extra retriever hints (merged over ``project``/``ext``).
    retrievers
        Override the retriever set (tests inject deterministic mocks); ``None`` ⇒ :func:`_default_retrievers`.

    ``value`` is the fused result list ``[{doc_id, locator, text, context, score, sources[]}]`` (best first);
    ``evidence.coverage`` reports ``{retrievers_run, hits_per_retriever}`` so the LLM can see WHICH method hit
    (and pick the next narrowing tool). Disabled (flag OFF) or an empty fan-out ⇒ empty ``value`` so the
    caller falls back to the individual tools — the packet is a fast path, never a truncation of truth.
    """
    try:
        n = int(limit)
    except (TypeError, ValueError):
        n = _DEFAULT_LIMIT
    n = max(1, min(_MAX_LIMIT, n))

    if not enabled():
        return _contract.make([], engine="unified_search", query=query,
                              evidence={"returned": 0, "reason": "disabled",
                                        "coverage": {"retrievers_run": [], "hits_per_retriever": {}}})

    merged_hints: dict[str, Any] = {"project": project, "ext": ext}
    if hints:
        merged_hints.update(hints)

    rs = list(retrievers) if retrievers is not None else _default_retrievers()
    weights = {r.name: r.weight for r in rs}
    k = _rrf_k()

    results, names_run = _run_retrievers(rs, query, merged_hints)
    all_entries = _fuse(results, weights, k)
    winners = _cap_and_limit(all_entries, _per_file_cap(), n)
    _reexpand_context(winners, all_entries)

    value = [{"doc_id": e.doc_id, "locator": e.locator, "text": e.text, "context": e.context,
              "score": round(e.score, 6), "sources": sorted(e.sources)} for e in winners]
    coverage = {"retrievers_run": names_run,
                "hits_per_retriever": {name: len(hits) for name, hits in results.items()}}
    evidence: dict[str, Any] = {"returned": len(value), "coverage": coverage,
                                "filters": {"project": project or None,
                                            "ext": ext if isinstance(ext, str) else (
                                                list(ext) if ext else None)}}
    if not value:
        evidence["reason"] = "no_match"
    return _contract.make(value, engine="unified_search", evidence=evidence,
                          query=query, fusion="rrf", rrf_k=k,
                          per_file_cap=_per_file_cap(),
                          source="fan_out+rrf+context_reexpansion")


# The agent-tool schema (single source of truth: :mod:`src.rag.agent.investigator` registers it and
# :mod:`src.rag.mcp.server` re-publishes the same AgentTool list to MCP).
TOOL_NAME = "search"
TOOL_DESCRIPTION = (
    "統合検索: 1呼びで全リトリーバ(字句FTS・蒸留・IDマスタ逆引き・版差分・案件ファイル解決)を内部並列実行し、"
    "RRF(Σ weight/(60+rank))で融合・重複統合・ファイル毎上限・文脈復元した証拠パケットを返す。探索はまず"
    "この search を1発。返る results[{doc_id,locator,text,context,score,sources}] と coverage(どの方式が当てたか)を"
    "見て、case_filter / metric_lookup / format_events 等の個別ツールは search 結果を絞る用途に限って使う。"
)
TOOL_PARAMETERS = {
    "type": "object",
    "properties": {
        "query": {"type": "string"},
        "project": {"type": "string"},
        "ext": {"type": "string"},
        "limit": {"type": "integer"},
    },
    "required": ["query"],
}
