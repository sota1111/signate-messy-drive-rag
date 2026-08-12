"""text_search — indexed lexical search over the full-corpus FTS5 index (SOT-2657).

The agent-callable *first choice* for literal/字句 discovery ("この文字列/フラグ名/ID/固有語は
コーパスのどこにあるか"). Where :func:`src.rag.tools.file_grep.file_grep` re-extracts every document
on each call (BUDGET32 診断: 全ツール呼びの ~70% が file_grep 全走査で、針が無いと再 grep する spin が
棄権の主犯)、``text_search`` は build 時に全文を転記済みの SQLite FTS5 索引を引くだけなので、ミリ秒で
IDF ランク付きの上位ヒットを返す(LLM 関与ゼロ)。

Returns the unified :mod:`src.rag.tools.contract` shape ``{value, evidence, method}`` so the generation
layer calls it exactly like the other tools. ``value`` is a ranked list of hit records
``{doc_id, locator, text, score, project, ext}``; the locator vocabulary
(``page:N`` / ``cell:A1@sheet:…`` / ``slide:N`` / ``line:N``) is identical to file_grep's, so a hit
names the same place either way.

Serve-time gating: :func:`enabled` (``RAG_TEXT_FTS``, default OFF) — with the flag unset the index is
never consulted, the tool is not registered, and the champion serve path is byte-identical. Any miss
(disabled / absent / unreadable index, or an empty result) returns an empty ``value`` so the agent
falls back to file_grep (the index is a fast path, never a truncation of truth).
"""
from __future__ import annotations

from typing import Any, Iterable

from src.rag.index import text_fts
from src.rag.tools import contract

# Default number of ranked hits returned to the agent (kept small: the point is a precise needle,
# not a re-derivation of the whole corpus). The agent may request more via ``limit``.
_DEFAULT_LIMIT = 20
_MAX_LIMIT = 100


def enabled() -> bool:
    """True when the lexical index should back the ``text_search`` tool (default OFF — opt-in)."""
    return text_fts.enabled()


def text_search(query: str, *, project: str | None = None,
                ext: str | Iterable[str] | None = None,
                limit: int = _DEFAULT_LIMIT) -> dict[str, Any]:
    """Search the full-corpus lexical index and return contract ``{value, evidence, method}``.

    Parameters
    ----------
    query
        A literal / 字句 query. Tokenized with the same rules used at build time (ASCII words keep
        internal ``_.-`` so flag names / dotted ids stay intact; CJK/kana matched via 2-char shingles),
        then IDF-ranked so a rare literal (ID / flag / 固有名) dominates a generic substring.
    project, ext
        Optional filters (NFC company name; extension or list of extensions).
    limit
        Max ranked hits to return (default 20, capped at 100).

    ``value`` is a list of ``{doc_id, locator, text, score, project, ext}`` hit records ordered by
    descending IDF score. Empty ``value`` (with ``evidence.reason``) means the caller should fall back
    to ``file_grep`` — the index answered nothing, never a wrong/truncated answer.
    """
    try:
        n = int(limit)
    except (TypeError, ValueError):
        n = _DEFAULT_LIMIT
    n = max(1, min(_MAX_LIMIT, n))

    if not enabled():
        return contract.make([], engine="text_search",
                             evidence={"returned": 0, "reason": "disabled"},
                             query=query, source="lexical_fts")

    # Pre-search query distillation (SOT-2672, RAG_QUERY_DISTILL default OFF ⇒ eff_query/eff_project
    # unchanged ⇒ byte-identical). Company→scope, condition-phrase removal, rare-token retention.
    from src.rag.tools import query_distill as _qd

    eff_query, eff_project = query, project
    distill_diag: dict[str, Any] | None = None
    if _qd.enabled():
        d = _qd.distill(query, project=project)
        eff_query = d.query or query
        if project is None and d.scope_project:
            eff_project = d.scope_project
        distill_diag = d.as_diagnostic()

    hits = text_fts.search(eff_query, limit=n, project=eff_project, ext=ext)

    # Corpus-vocabulary retry (deterministic, once): on a zero-hit distilled query, rewrite its
    # out-of-vocabulary tokens to body-present near-tokens via the IDF table and re-search.
    retry_diag: dict[str, Any] | None = None
    if not hits and _qd.enabled():
        bridged, subs = _qd.corpus_vocab_retry(eff_query)
        if subs:
            retry_hits = text_fts.search(bridged, limit=n, project=eff_project, ext=ext)
            retry_diag = {"query": bridged, "substitutions": subs, "returned": len(retry_hits)}
            if retry_hits:
                hits = retry_hits

    evidence: dict[str, Any] = {
        "returned": len(hits),
        "source": "lexical_fts",
        "filters": {"project": eff_project or None,
                    "ext": sorted({e.lower().lstrip(".") for e in (
                        [ext] if isinstance(ext, str) else list(ext or [])) if e}) or None},
    }
    if distill_diag is not None:
        evidence["distill"] = distill_diag
    if retry_diag is not None:
        evidence["vocab_retry"] = retry_diag
    if not hits:
        evidence["reason"] = "no_match"
    return contract.make(hits, engine="text_search", evidence=evidence,
                         query=query, source="lexical_fts",
                         ranking="idf-weighted", index="fts5")


# The agent-tool schema (single source of truth: :mod:`src.rag.agent.investigator` registers it and
# :mod:`src.rag.mcp.server` re-publishes the same AgentTool list to MCP).
TOOL_NAME = "text_search"
TOOL_DESCRIPTION = (
    "コーパス全文の字句インデックス(FTS5+IDF)を引き、クエリ(エラー文字列・フラグ名・ID・固有語などの"
    "リテラル)に一致する箇所を所在(doc_id・ページ/シート/セル/行のlocator)・抜粋・スコア付きで即返す。"
    "希少トークンほど上位。file_grep の全走査より先に使い、返り値が空のときだけ file_grep にフォールバックする。"
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
