"""SOT-2587 — symbolic exhaustive-scan ENUM lane + completeness certificate.

Parent SOT-2568 deep-research comment (実装順 5/7, P1). The ``enum_set`` route scores MATCH 2 / 9 on
gold-100; the structural cause is that **top-k semantic retrieval can never prove that what it did *not*
retrieve does not exist**. An enumeration ("列挙") is a *database query*, not a similarity search: the
correct answer is obtained by resolving the target **universe** of documents deterministically and then
**scanning every one of them**, not by trusting the retrieval top-k to be complete.

This module promotes enumeration from a post-retrieval helper (:mod:`src.rag.agent.enumeration`'s closure
protocol) to a **router-directly-called primary lane** that:

1. **Resolves the universe** from the SOT-2583 document registry (not chunk similarity):
   ``documents_total`` (whole corpus), ``documents_applicable`` (the scope the question defines —
   explicitly-named files, else a named project, else the whole corpus), ``documents_scanned`` (of the
   applicable set, those a parser can actually read) and ``unsupported_documents`` (applicable but
   unreadable — encrypted / image-only). :class:`Universe`.

2. **Scans the applicable-and-scannable documents deterministically** over the persisted normalized
   evidence table (:mod:`src.rag.index.evidence_index` — typed cells / rows / entities / format events,
   full-corpus coverage) — a full-scan predicate filter, never a retrieval cutoff — and returns the
   **complete** candidate set in canonical order (:mod:`src.rag.agent.enumeration`'s ordering).

3. **Attaches a completeness certificate**::

       {"matched": [...],
        "universe": {"documents_total": N, "documents_applicable": A,
                     "documents_scanned": S, "unsupported_documents": U},
        "complete": (S == A and U == 0)}

4. **Forbids a "該当なし" (no-match) answer when the scan was not complete** — specifically when
   ``unsupported_documents > 0`` and ``matched`` is empty (the idx16-style "見えていない=存在しない" error,
   applied to enumeration): the lane returns a :data:`~src.rag.agent.failure_taxonomy.PARSER_CAPABILITY_MISS`
   signal instead of an empty "no such element" result. This is :func:`forbids_none_answer`.

Design invariants (mirroring the sibling opt-ins — document_registry #1, evidence_packet / pot_lane):
  * **Opt-in, byte-identical OFF.** :func:`enabled` (``RAG_ENUM_SCAN``) gates the whole lane; the
    default-OFF serve path never builds a scan and is byte-identical.
  * **No corpus fact / no answer injected into the prompt.** The lane is an agent-callable *tool* that
    returns a certificate the model reads; the committed answer is still the model's, subject to every
    existing commit guard. The pre-inject *directive* names only the lane and the guard rule — no value.
  * **Deterministic & network-free.** Universe resolution is the registry's structural chain; the scan is
    a predicate filter over the persisted typed evidence rows. No LLM call, no network, reproducible.
"""
from __future__ import annotations

import os
import re
from dataclasses import dataclass
from typing import Any, Iterable, Sequence

from src.rag.agent import enumeration as _enum
from src.rag.agent import failure_taxonomy as _tax
from src.rag.corpus import nfc
from src.rag.index import document_registry as _dr
from src.rag.index import evidence_index as _ev

_ON = {"1", "true", "yes", "on"}


def enabled() -> bool:
    """True when the serve path should dispatch ENUM through the symbolic scan lane (default OFF)."""
    return os.getenv("RAG_ENUM_SCAN", "0").strip().lower() in _ON


# --------------------------------------------------------------------------- universe model
@dataclass(frozen=True)
class Universe:
    """The deterministically-resolved document universe for one enumeration.

    ``documents_applicable`` is the scope the question defines; ``documents_scanned`` is the subset a
    parser can actually read (text/structural extraction available); ``unsupported_documents`` is the
    remainder (encrypted / image-only) that the scan could not cover — the source of an *incomplete*
    certificate and of the no-match guard.
    """

    documents_total: int
    applicable_ids: tuple[str, ...]
    scanned_ids: tuple[str, ...]
    unsupported_ids: tuple[str, ...]
    scope: str = ""                 # how the applicable set was resolved (explicit_file/project/corpus)

    @property
    def documents_applicable(self) -> int:
        return len(self.applicable_ids)

    @property
    def documents_scanned(self) -> int:
        return len(self.scanned_ids)

    @property
    def unsupported_documents(self) -> int:
        return len(self.unsupported_ids)

    @property
    def complete(self) -> bool:
        """The universe was fully covered: every applicable document was scanned, none unsupported."""
        return self.unsupported_documents == 0 and self.documents_scanned == self.documents_applicable

    def to_dict(self) -> dict[str, int]:
        """The certificate's ``universe`` block — the four counts only (no doc ids / no corpus fact)."""
        return {
            "documents_total": self.documents_total,
            "documents_applicable": self.documents_applicable,
            "documents_scanned": self.documents_scanned,
            "unsupported_documents": self.unsupported_documents,
        }


# A document is *unsupported* for a text/structural enumeration scan when it cannot be read: it was
# encrypted (or otherwise failed) at extraction time, or it is image-only (its sole parser is vision, so
# no textual rows exist to scan). Both are recorded on the registry record at build time.
_ENCRYPTED_WARNINGS = frozenset({"encrypted"})


def _is_unsupported(row: dict[str, Any]) -> bool:
    warnings = {str(w).strip().lower() for w in (row.get("extraction_warnings") or [])}
    if warnings & _ENCRYPTED_WARNINGS:
        return True
    caps = {str(c).strip().lower() for c in (row.get("parser_capabilities") or [])}
    # image-only: the only declared capability is vision → no text/structural rows to scan.
    if caps and caps <= {"vision"}:
        return True
    return False


def _scope_project(rows: Sequence[dict[str, Any]], project: str | None) -> list[dict[str, Any]]:
    if not project:
        return list(rows)
    p = _dr.fold(project)
    out = [r for r in rows
           if p in _dr.fold(r.get("case_id", "")) or _dr.fold(r.get("case_id", "")) in p]
    return out or list(rows)     # fail-open: an unrecognised project scopes to the whole corpus


def resolve_universe(question: str, *, project: str | None = None,
                     registry_rows: Sequence[dict[str, Any]] | None = None) -> Universe:
    """Resolve the enumeration's document universe from the registry (deterministic, network-free).

    Scope precedence (widest-recall first, never a lossy narrowing that could drop the target):
      1. **explicit files** — when the question names files that the registry resolves, the universe is
         exactly those documents (a single-file / few-file enumeration such as idx19's スケジュール_r2.xlsx).
      2. **named project** — else, when the question names a project (案件), the universe is that project's
         documents.
      3. **whole corpus** — else every document.

    ``documents_scanned`` / ``unsupported_documents`` split the applicable set by parser reachability
    (:func:`_is_unsupported`), which drives the certificate's ``complete`` flag and the no-match guard.
    """
    rows = list(registry_rows) if registry_rows is not None else _dr.load()
    total = len(rows)
    by_id = {r.get("doc_id"): r for r in rows}
    scope = "corpus"
    applicable: list[dict[str, Any]] = []

    # (1) explicit-file scope — resolve every named file through the registry's trusted structural tiers.
    if _dr.has_explicit_file_reference(question):
        resolver = _dr.Resolver(rows) if rows else None
        if resolver is not None:
            hits = resolver.resolve_explicit(question, project=project)
            seen: set[str] = set()
            for h in hits:
                if h.doc_id in by_id and h.doc_id not in seen:
                    seen.add(h.doc_id)
                    applicable.append(by_id[h.doc_id])
            if applicable:
                scope = "explicit_file"

    # (2) project scope — else the named project's documents.
    if not applicable:
        proj = project
        if proj is None:
            try:  # canonical_route reuse is best-effort; a failure just widens the scope to the corpus.
                from src.rag.tools.canonical_route import resolve_project as _resolve_project
                proj = _resolve_project(question, None)
            except Exception:  # noqa: BLE001
                proj = None
        applicable = _scope_project(rows, proj)
        scope = "project" if proj else "corpus"

    applicable_ids = tuple(str(r.get("doc_id")) for r in applicable)
    unsupported_ids = tuple(str(r.get("doc_id")) for r in applicable if _is_unsupported(r))
    unsupported_set = set(unsupported_ids)
    scanned_ids = tuple(d for d in applicable_ids if d not in unsupported_set)
    return Universe(documents_total=total, applicable_ids=applicable_ids, scanned_ids=scanned_ids,
                    unsupported_ids=unsupported_ids, scope=scope)


# --------------------------------------------------------------------------- element scan
# Which typed evidence-index rows carry the members of each enumeration population (SOT-2500 kinds).
# ``document`` is handled separately (its members are the universe's own filenames, not evidence rows).
_KIND_ENTRY_TYPES: dict[str, tuple[str, ...]] = {
    _enum.PERSON: ("person",),
    _enum.SEAT: ("person",),
    _enum.ABBREV: ("alias",),
    _enum.PROJECT: ("alias",),
    _enum.TASK: ("column", "number", "heading"),
    _enum.GENERIC: ("person", "alias", "number", "column", "heading", "highlight", "bold"),
}


def _document_members(universe: Universe,
                      registry_rows: Sequence[dict[str, Any]] | None) -> list[str]:
    """Members of a ``document`` enumeration = the scanned universe's own filenames (a true full scan)."""
    rows = list(registry_rows) if registry_rows is not None else _dr.load()
    by_id = {r.get("doc_id"): r for r in rows}
    out: list[str] = []
    for did in universe.scanned_ids:
        row = by_id.get(did, {})
        name = str(row.get("title") or row.get("normalized_filename") or did)
        out.append(nfc(name))
    return out


def _evidence_members(universe: Universe, entry_types: Sequence[str],
                      predicate: "re.Pattern[str] | None",
                      index_rows: Sequence[dict[str, Any]] | None) -> list[str]:
    """Candidate element values from the persisted typed evidence rows within the scanned universe.

    A member qualifies when its evidence row (a) belongs to a scanned document, (b) is one of the
    population's ``entry_types`` and (c) matches ``predicate`` (applied to the row's ``value``; ``None``
    = accept all). The scan is a full pass over the persisted table, filtered — never a retrieval cutoff.
    """
    rows = list(index_rows) if index_rows is not None else _ev.load()
    scanned = set(universe.scanned_ids)
    want = set(entry_types)
    out: list[str] = []
    for entry in rows:
        rel = entry.get("rel") or entry.get("file") or ""
        if nfc(rel) not in scanned:
            continue
        if entry.get("type") not in want:
            continue
        value = str(entry.get("value", "")).strip()
        if not value:
            continue
        if predicate is not None and not predicate.search(nfc(value)):
            continue
        out.append(value)
    return out


@dataclass(frozen=True)
class EnumScanResult:
    """The outcome of one symbolic enumeration scan — the matched set plus its completeness certificate."""

    matched: tuple[str, ...]
    universe: Universe
    population_kind: str
    ordering_rule: str
    predicate: str = ""
    state_code: str = ""            # "" (complete/covered) | PARSER_CAPABILITY_MISS | EVIDENCE_INCOMPLETE

    @property
    def complete(self) -> bool:
        return self.universe.complete

    @property
    def guard_triggered(self) -> bool:
        """The no-match guard: unsupported documents blocked coverage AND nothing matched (idx16-type)."""
        return self.universe.unsupported_documents > 0 and not self.matched

    def certificate(self) -> dict[str, Any]:
        """The research-shape completeness certificate: ``{matched, universe, complete}``."""
        return {
            "matched": list(self.matched),
            "universe": self.universe.to_dict(),
            "complete": self.complete,
        }

    def to_dict(self) -> dict[str, Any]:
        cert = self.certificate()
        cert.update({
            "population_kind": self.population_kind,
            "ordering_rule": self.ordering_rule,
            "predicate": self.predicate,
            "scope": self.universe.scope,
            "guard_triggered": self.guard_triggered,
            "state_code": self.state_code,
            "may_answer_none": self.may_answer_none,
        })
        return cert

    @property
    def may_answer_none(self) -> bool:
        """False when the scan is not trustworthy enough to conclude "該当なし" (the guard forbids it)."""
        return not self.guard_triggered


def scan(question: str, *, predicate: str | None = None, project: str | None = None,
         entry_types: Sequence[str] | None = None, population_kind: str | None = None,
         registry_rows: Sequence[dict[str, Any]] | None = None,
         index_rows: Sequence[dict[str, Any]] | None = None) -> EnumScanResult:
    """Run the symbolic exhaustive scan for an enumeration question and build its certificate.

    ``predicate`` is an optional regex applied to each candidate element's value (``None`` = accept all
    members of the population). ``entry_types`` / ``population_kind`` may be supplied to override the
    deterministic population detection (:mod:`src.rag.agent.enumeration`). Every argument is deterministic;
    pass ``registry_rows`` / ``index_rows`` to scan an in-memory table in tests (no artifact needed).
    """
    universe = resolve_universe(question, project=project, registry_rows=registry_rows)
    kind = population_kind or _enum.detect_population_kind(question)
    ordering = _enum.detect_ordering_rule(question, kind)

    pat: "re.Pattern[str] | None" = None
    if predicate:
        try:
            pat = re.compile(nfc(predicate))
        except re.error:
            pat = re.compile(re.escape(nfc(predicate)))

    if kind == _enum.DOCUMENT:
        raw_members = _document_members(universe, registry_rows)
        if pat is not None:
            raw_members = [m for m in raw_members if pat.search(nfc(m))]
    else:
        types = tuple(entry_types) if entry_types is not None else _KIND_ENTRY_TYPES.get(
            kind, _KIND_ENTRY_TYPES[_enum.GENERIC])
        raw_members = _evidence_members(universe, types, pat, index_rows)

    matched = tuple(_enum.apply_ordering(raw_members, ordering))

    # Certificate state: an unsupported-document-blocked empty result must NOT become "該当なし"
    # (idx16-type "見えていない=存在しない"): flag it PARSER_CAPABILITY_MISS. An otherwise-incomplete
    # universe with no match is EVIDENCE_INCOMPLETE (still not a trustworthy "no such element").
    state_code = ""
    if universe.unsupported_documents > 0 and not matched:
        state_code = _tax.PARSER_CAPABILITY_MISS
    elif not universe.complete and not matched:
        state_code = _tax.EVIDENCE_INCOMPLETE

    return EnumScanResult(
        matched=matched, universe=universe, population_kind=kind, ordering_rule=ordering,
        predicate=predicate or "", state_code=state_code)


def forbids_none_answer(result: EnumScanResult) -> bool:
    """True when the certificate forbids a "該当なし" (no-match) conclusion.

    The idx16-type guard, applied to enumeration: when the applicable universe contained documents the
    scan could not read (``unsupported_documents > 0``) and the scan matched nothing, the empty result is
    an *artifact of incomplete coverage*, not evidence of non-existence — so the lane must not let the
    agent answer "該当なし".
    """
    return result.guard_triggered


# --------------------------------------------------------------------------- agent tool wiring
TOOL_NAME = "enum_scan"

TOOL_DESCRIPTION = (
    "完全列挙(ENUM)専用の決定論『全数走査』レーン。top-k 検索は『取得しなかったものが存在しないこと』を証明できない"
    "ため、列挙は database query として扱う。質問文から対象 universe(母集団の全文書)を registry で決定論解決し、"
    "走査可能な全文書を評価して predicate に一致する要素を漏れなく返す。返り値には completeness certificate "
    "{matched, universe:{documents_total, documents_applicable, documents_scanned, unsupported_documents}, "
    "complete} が付く。complete=false かつ unsupported_documents>0 で 0件のときは『該当なし』と答えてはならない"
    "(見えていない=存在しない の誤りを防ぐ。抽出不能な文書があるだけで不存在ではない)。\n"
    "引数: question=質問文(必須)、predicate=各要素値に対する任意の正規表現フィルタ、"
    "entry_types=走査する証拠種別の上書き(person/alias/number/column/heading/highlight/bold)、project=案件スコープ。")

_STR: dict[str, Any] = {"type": "string"}
TOOL_PARAMETERS: dict[str, Any] = {
    "type": "object",
    "properties": {
        "question": _STR,
        "predicate": _STR,
        "project": _STR,
        "entry_types": {"type": "array", "items": _STR},
    },
    "required": ["question"],
}


def enum_scan_tool(question: str, predicate: str | None = None,
                   entry_types: Sequence[str] | None = None,
                   project: str | None = None) -> dict[str, Any]:
    """Agent-callable entry: run the scan and return the completeness certificate (never raises).

    Errors are returned as ``{"error": ...}`` so a malformed call surfaces as feedback rather than
    crashing the loop. The returned dict carries the certificate plus ``may_answer_none`` (the guard) and
    a ``state_code`` when coverage was incomplete."""
    if not question or not str(question).strip():
        return {"error": "question が空です。列挙対象を含む質問文を渡してください。"}
    try:
        result = scan(str(question), predicate=predicate, project=project,
                      entry_types=list(entry_types) if entry_types else None)
    except Exception as exc:  # noqa: BLE001 — a tool call must never crash the loop
        return {"error": f"enum_scan 失敗: {type(exc).__name__}: {exc}"}
    return result.to_dict()


def enum_lane_directive(question: str) -> str:
    """The extra pre-inject directive appended to an ENUM Evidence Packet when the lane is enabled.

    Tells the agent the ENUM route is a *full-scan* lane (retrieval を経由しない): it must resolve the
    universe and scan every applicable document via :data:`TOOL_NAME`, adopt only a set backed by the
    completeness certificate, and — the idx16 guard — never answer "該当なし" when the certificate is
    incomplete because documents were unreadable. Injects no corpus fact and no answer — protocol only."""
    kind = _enum.detect_population_kind(question)
    label = _enum.POPULATION_LABELS.get(kind, kind)
    order = _enum.ORDERING_LABELS.get(_enum.detect_ordering_rule(question, kind), "")
    return (
        "【ENUM 全数走査レーン(SOT-2587)】この質問は完全列挙型です。検索上位k件を母集団と見なしてはいけません"
        f"(母集団種別: {label})。答えは必ず `{TOOL_NAME}` ツールで確定してください:\n"
        "1) `enum_scan(question=質問全文[, predicate=絞り込み正規表現])` を呼び、対象 universe(全文書)を"
        "registry で決定論解決し、走査可能な全文書を全数走査した matched を得る。\n"
        "2) 返り値の completeness certificate を必ず確認する: complete=true(documents_scanned == "
        "documents_applicable かつ unsupported_documents==0)のときだけ matched を最終列挙として採用してよい。\n"
        "3) **禁止**: complete=false かつ unsupported_documents>0 かつ matched が空のとき『該当なし』と答えない"
        "(抽出不能な文書があるだけで不存在ではない)。その場合は unsupported を解消する経路(復号/別抽出)を試すか、"
        "根拠不足として棄権する。\n"
        f"4) 列挙は重複0・順序規則【{order}】で submit_answer する。ENUM の自由探索予算は0(全数走査で確定する)。")


# --------------------------------------------------------------------------- offline diagnostics
@dataclass(frozen=True)
class EnumCase:
    """One labelled enumeration case for the offline diagnostic metrics (SOT-2587 検証)."""

    index: int
    question: str
    project: str | None = None
    gold_universe: tuple[str, ...] = ()        # doc_ids that MUST be in the applicable universe
    gold_set: tuple[str, ...] = ()             # gold enumeration elements (for set precision/recall)
    predicate: str | None = None
    entry_types: tuple[str, ...] = ()
    population_kind: str | None = None


def _set_tokens(values: Iterable[str]) -> set[str]:
    """Normalized element tokens for set precision/recall — reuses the gold set scorer's normalization."""
    from scoring.deterministic import _tokens_set
    joined = "、".join(str(v) for v in values)
    return _tokens_set(joined)


def _covered(token: str, pool: set[str]) -> bool:
    """Element-containment tolerance identical to :func:`scoring.deterministic.score_set`."""
    return any(token == x or token in x or x in token for x in pool)


def measure_enum(cases: Sequence[EnumCase], *,
                 registry_rows: Sequence[dict[str, Any]] | None = None,
                 index_rows: Sequence[dict[str, Any]] | None = None) -> dict[str, Any]:
    """Compute the four SOT-2587 diagnostic metrics over labelled enumeration cases.

    * ``universe_resolution_accuracy`` — fraction of cases whose resolved *applicable* universe contains
      every gold-universe document (the enumeration cannot be complete if the target doc is out of scope).
    * ``exhaustive_scan_completion`` — mean, over cases, of ``documents_scanned / documents_applicable``
      (how much of the resolved universe the scan could actually cover).
    * ``set_precision`` / ``set_recall`` — micro-averaged over cases carrying a ``gold_set``, using the
      same element-containment normalization as the gold set judge (:func:`scoring.deterministic.score_set`).

    Deterministic and network-free; pass ``registry_rows`` / ``index_rows`` to measure an in-memory table.
    """
    detail: list[dict[str, Any]] = []
    univ_hits = 0
    graded_universe = 0
    completion_sum = 0.0
    tp = fp = fn = 0
    graded_sets = 0
    for case in cases:
        result = scan(case.question, predicate=case.predicate, project=case.project,
                      entry_types=list(case.entry_types) or None,
                      population_kind=case.population_kind,
                      registry_rows=registry_rows, index_rows=index_rows)
        u = result.universe
        gold_u = {nfc(d) for d in case.gold_universe}
        applicable = set(u.applicable_ids)
        # Only cases carrying a labelled gold universe count toward universe_resolution_accuracy.
        universe_ok: bool | None = None
        if gold_u:
            graded_universe += 1
            universe_ok = gold_u <= applicable
            if universe_ok:
                univ_hits += 1
        completion = (u.documents_scanned / u.documents_applicable) if u.documents_applicable else 0.0
        completion_sum += completion

        case_tp = case_fp = case_fn = None
        if case.gold_set:
            graded_sets += 1
            gold = _set_tokens(case.gold_set)
            pred = _set_tokens(result.matched)
            case_tp = sum(1 for p in pred if _covered(p, gold))
            case_fp = len(pred) - case_tp
            case_fn = sum(1 for g in gold if not _covered(g, pred))
            tp += case_tp
            fp += case_fp
            fn += case_fn

        detail.append({
            "index": case.index,
            "scope": u.scope,
            "universe": u.to_dict(),
            "universe_resolution_ok": universe_ok,
            "scan_completion": round(completion, 4),
            "complete": result.complete,
            "matched_count": len(result.matched),
            "state_code": result.state_code,
            "guard_triggered": result.guard_triggered,
            "set_tp": case_tp, "set_fp": case_fp, "set_fn": case_fn,
        })

    n = len(cases)
    precision = tp / (tp + fp) if (tp + fp) else None
    recall = tp / (tp + fn) if (tp + fn) else None
    return {
        "n_cases": n,
        "graded_universe_cases": graded_universe,
        "graded_set_cases": graded_sets,
        "universe_resolution_accuracy": round(univ_hits / graded_universe, 4)
        if graded_universe else None,
        "exhaustive_scan_completion": round(completion_sum / n, 4) if n else None,
        "set_precision": round(precision, 4) if precision is not None else None,
        "set_recall": round(recall, 4) if recall is not None else None,
        "set_micro": {"tp": tp, "fp": fp, "fn": fn},
        "detail": detail,
    }
