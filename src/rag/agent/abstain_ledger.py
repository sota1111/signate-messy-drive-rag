"""SOT-2492 — abstain ledger: record a state code + unmet evidence obligation for every abstain.

Parent SOT-2460. The investigator (:mod:`src.rag.agent.investigator`) abstains on a large share of
gold-100 questions (~82 in one run). SOT-2486's *one-off* diagnosis split those into
``retrieval_miss=22 / derivation_miss=18``; this module makes that split **permanent and runtime** —
every time the investigator abstains, a per-question record is appended to
``artifacts/abstain_ledger.jsonl`` carrying:

* a **state code** — one of the seven fixed causes (:data:`STATE_CODES`) so the 82 abstains are never
  collapsed into an undifferentiated "answered nothing";
* the **unmet evidence obligation** — free text (what was needed but not obtained) plus a structured
  ``missing`` block (the observed retrieval/extraction/derivation signal counts);
* the **explored paths** — the sequence of tools the investigator actually tried.

Design invariant — *observer, never decider*
--------------------------------------------
This module is a pure **observer**: it classifies an already-finished :class:`Investigation` and writes
a record. It NEVER influences the commit-vs-abstain decision (受け入れ条件②: 本番回答は不変). The
investigator collects lightweight per-tool signals as it dispatches, and only after the answer is fixed
does it (when a code path is an abstain) build and write the record here.

Total classification — *code-less abstain is structurally impossible*
---------------------------------------------------------------------
:func:`classify` is a **total function**: for any :class:`AbstainSignals` it returns exactly one member
of :data:`STATE_CODES` (defaulting to :data:`UNANSWERABLE`). Combined with the investigator always
routing an abstain through :func:`record_from_investigation`, this makes a coded record the only
possible outcome of an abstain (受け入れ条件①).
"""
from __future__ import annotations

import json
import sys
import threading
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Mapping

from config import settings
from src.rag.tools import contract as _contract

# --------------------------------------------------------------------------- the seven state codes
NOT_RETRIEVED = "NOT_RETRIEVED"              # 関連文書/データを検索で特定できなかった (retrieval miss)
RETRIEVED_NOT_PARSED = "RETRIEVED_NOT_PARSED"  # 文書は見つかったが必要な値を抽出/解析できなかった
PARSED_AMBIGUOUS = "PARSED_AMBIGUOUS"        # 候補が複数で一意に確定できなかった
EVIDENCE_CONFLICT = "EVIDENCE_CONFLICT"      # 複数の根拠が矛盾し確定できなかった
EVIDENCE_INCOMPLETE = "EVIDENCE_INCOMPLETE"  # 根拠が部分的で回答に必要な情報が揃わなかった
UNANSWERABLE = "UNANSWERABLE"                # コーパス内に根拠が存在せず原理的に導出できなかった
BUDGET_EXHAUSTED = "BUDGET_EXHAUSTED"        # 反復上限/タイムアウトで確定前に打ち切った

STATE_CODES: tuple[str, ...] = (
    NOT_RETRIEVED, RETRIEVED_NOT_PARSED, PARSED_AMBIGUOUS, EVIDENCE_CONFLICT,
    EVIDENCE_INCOMPLETE, UNANSWERABLE, BUDGET_EXHAUSTED,
)

# Free-text obligation template per code (the "what was NOT obtained" statement).
_DEFAULT_OBLIGATION: dict[str, str] = {
    NOT_RETRIEVED: "関連する文書/データを検索で特定できなかった(retrieval miss)。",
    RETRIEVED_NOT_PARSED: "文書は見つかったが、必要な値を抽出/解析できなかった。",
    PARSED_AMBIGUOUS: "候補が複数あり、質問が指す対象を一意に確定できなかった。",
    EVIDENCE_CONFLICT: "複数の根拠が矛盾し、正しい値を確定できなかった。",
    EVIDENCE_INCOMPLETE: "根拠が部分的で、回答に必要な情報が揃わなかった。",
    UNANSWERABLE: "コーパス内に回答の根拠が存在せず、原理的に導出できなかった。",
    BUDGET_EXHAUSTED: "反復上限/タイムアウトに達し、根拠を確定する前に打ち切った。",
}

# The "kind" of the structured obligation per code (machine-groupable diagnosis bucket).
_OBLIGATION_KIND: dict[str, str] = {
    NOT_RETRIEVED: "retrieval",
    RETRIEVED_NOT_PARSED: "parse",
    PARSED_AMBIGUOUS: "disambiguation",
    EVIDENCE_CONFLICT: "reconciliation",
    EVIDENCE_INCOMPLETE: "completeness",
    UNANSWERABLE: "existence",
    BUDGET_EXHAUSTED: "budget",
}

# --------------------------------------------------------------------------- tool → phase buckets
# Which investigation phase each investigator tool belongs to. Kept here (not imported from the
# investigator) so this module has no dependency on it — the investigator depends on us, not vice versa.
_RETRIEVAL_TOOLS = frozenset({"find_files", "file_grep"})
_EXTRACTION_TOOLS = frozenset({
    "read_office", "decrypt", "read_chart_values", "caption_image",
    "pdf_emphasis", "pptx_pivot", "highlight_extract",
})
_DERIVATION_TOOLS = frozenset({"compute", "version_diff", "seating_lookup", "corpus_aggregate"})

# Substrings that mark a tool error as an *ambiguity* (multiple candidates) rather than a hard failure.
_AMBIGUOUS_MARKS = ("曖昧", "ambiguous", "disambiguate", "存在プロジェクト", "複数", "multiple")
# Keyword signals mined from the model's own evidence/method text (lower-cased; JP unaffected by lower).
_CONFLICT_KW = ("矛盾", "食い違", "不一致", "conflict", "inconsisten")
_INCOMPLETE_KW = ("不足", "一部", "足りな", "揃わ", "そろわ", "incomplete", "partial", "insufficient")


class LedgerError(ValueError):
    """Raised when a ledger record fails schema validation."""


def _looks_ambiguous(text: str) -> bool:
    low = text.lower()
    return any(mark in low for mark in _AMBIGUOUS_MARKS)


def _outcome(resp: Any) -> str:
    """Coarse outcome of one dispatched tool response: ``'error' | 'ambiguous' | 'empty' | 'ok'``.

    ``dispatch`` returns either an ``{"error": ...}`` mapping, the ``{value, evidence, method}``
    contract, or a bare list/mapping/scalar. We only need a per-phase success signal, so we bucket
    them coarsely: an error whose message names an ambiguity is ``ambiguous``; a contract whose
    ``value`` is empty (``None`` / ``""`` / ``[]`` / ``{}``) or an empty list/mapping is ``empty``.
    """
    if isinstance(resp, Mapping):
        if "error" in resp:
            return "ambiguous" if _looks_ambiguous(str(resp.get("error", ""))) else "error"
        if _contract.is_contract(resp):
            return "empty" if resp.get("value") in (None, "", [], {}) else "ok"
        return "ok" if resp else "empty"
    if isinstance(resp, (list, tuple)):
        return "ok" if len(resp) else "empty"
    if resp in (None, ""):
        return "empty"
    return "ok"


@dataclass
class AbstainSignals:
    """Lightweight, side-effect-free tally the investigator fills while dispatching tools.

    Every field is derived only by *observing* tool responses (never by changing them), so populating
    an :class:`AbstainSignals` cannot alter the answer path. ``stop_reason`` and ``evidence_text`` are
    set once, after the loop, from the finished investigation.
    """

    stop_reason: str = "answered"
    evidence_text: str = ""            # model's own evidence/method/answer free text (keyword mining)
    retrieval_attempts: int = 0
    retrieval_ok: int = 0
    extraction_attempts: int = 0
    extraction_ok: int = 0
    derivation_attempts: int = 0
    derivation_ok: int = 0
    ambiguous: int = 0
    errors: int = 0

    def observe(self, tool_name: str, resp: Any) -> None:
        """Fold one dispatched tool response into the phase tallies. Pure bookkeeping."""
        oc = _outcome(resp)
        if oc == "ambiguous":
            self.ambiguous += 1
        elif oc == "error":
            self.errors += 1
        ok = 1 if oc == "ok" else 0
        if tool_name in _RETRIEVAL_TOOLS:
            self.retrieval_attempts += 1
            self.retrieval_ok += ok
        elif tool_name in _EXTRACTION_TOOLS:
            self.extraction_attempts += 1
            self.extraction_ok += ok
        elif tool_name in _DERIVATION_TOOLS:
            self.derivation_attempts += 1
            self.derivation_ok += ok

    def to_dict(self) -> dict[str, int]:
        """The structured signal counts embedded in a record's ``missing`` block."""
        return {
            "retrieval_attempts": self.retrieval_attempts, "retrieval_ok": self.retrieval_ok,
            "extraction_attempts": self.extraction_attempts, "extraction_ok": self.extraction_ok,
            "derivation_attempts": self.derivation_attempts, "derivation_ok": self.derivation_ok,
            "ambiguous": self.ambiguous, "errors": self.errors,
        }


def classify(signals: AbstainSignals) -> str:
    """Map observed signals to exactly one :data:`STATE_CODES` member (a **total** function).

    Precedence (first match wins):

    1. ``max_turns`` / ``timeout`` stop → :data:`BUDGET_EXHAUSTED` (infra cutoff, not a content cause).
    2. ``model_error`` stop → :data:`UNANSWERABLE` (transport crash produced no evidence; the raw cause
       is preserved in the record's ``stop_reason`` / ``error`` fields).
    3. any ambiguous tool outcome → :data:`PARSED_AMBIGUOUS`.
    4. a conflict keyword in the model's evidence text → :data:`EVIDENCE_CONFLICT`.
    5. nothing succeeded in any phase → :data:`NOT_RETRIEVED` (could not even retrieve).
    6. retrieval succeeded but extraction was tried and all failed → :data:`RETRIEVED_NOT_PARSED`.
    7. an incompleteness keyword, or a derivation that was tried and all failed → :data:`EVIDENCE_INCOMPLETE`.
    8. otherwise → :data:`UNANSWERABLE`.

    Never returns ``None`` — this is what makes a code-less abstain structurally impossible.
    """
    if signals.stop_reason in ("max_turns", "timeout"):
        return BUDGET_EXHAUSTED
    if signals.stop_reason == "model_error":
        return UNANSWERABLE
    hay = (signals.evidence_text or "").lower()
    if signals.ambiguous:
        return PARSED_AMBIGUOUS
    if any(kw in hay for kw in _CONFLICT_KW):
        return EVIDENCE_CONFLICT
    if not (signals.retrieval_ok or signals.extraction_ok or signals.derivation_ok):
        return NOT_RETRIEVED
    if signals.retrieval_ok and signals.extraction_attempts and not signals.extraction_ok:
        return RETRIEVED_NOT_PARSED
    if any(kw in hay for kw in _INCOMPLETE_KW):
        return EVIDENCE_INCOMPLETE
    if signals.derivation_attempts and not signals.derivation_ok:
        return EVIDENCE_INCOMPLETE
    return UNANSWERABLE


def _obligation_text(code: str, evidence_text: str) -> str:
    """Free-text unmet-obligation statement: the per-code template + the model's own note (if any)."""
    base = _DEFAULT_OBLIGATION[code]
    note = (evidence_text or "").strip()
    return f"{base} 調査メモ: {note}" if note else base


def _missing(code: str, signals: AbstainSignals) -> tuple[dict[str, Any], ...]:
    """The structured obligation block: the diagnosis ``kind``, its detail, and the raw signal counts."""
    return ({
        "kind": _OBLIGATION_KIND[code],
        "detail": _DEFAULT_OBLIGATION[code],
        "signals": signals.to_dict(),
    },)


@dataclass(frozen=True)
class AbstainRecord:
    """One per-question abstain diagnosis line (serialized to JSONL)."""

    question: str
    state_code: str
    evidence_obligation: str            # free text: what was needed but not obtained
    missing: tuple[dict[str, Any], ...]  # structured obligation(s): kind + detail + signal counts
    explored_paths: tuple[str, ...]      # the tool calls actually tried, in order
    stop_reason: str
    iterations: int
    tool_calls: tuple[str, ...]
    model: str
    elapsed_s: float
    cost_usd: float
    confidence: float
    error: str | None
    recorded_at: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "recorded_at": self.recorded_at,
            "question": self.question,
            "state_code": self.state_code,
            "evidence_obligation": self.evidence_obligation,
            "missing": [dict(m) for m in self.missing],
            "explored_paths": list(self.explored_paths),
            "stop_reason": self.stop_reason,
            "iterations": self.iterations,
            "tool_calls": list(self.tool_calls),
            "model": self.model,
            "elapsed_s": self.elapsed_s,
            "cost_usd": self.cost_usd,
            "confidence": self.confidence,
            "error": self.error,
        }


# Fields a valid record dict must carry (schema check for tests / defensive read-back).
_REQUIRED_FIELDS = (
    "recorded_at", "question", "state_code", "evidence_obligation", "missing",
    "explored_paths", "stop_reason", "iterations", "tool_calls", "model",
)


def validate(record: Mapping[str, Any]) -> None:
    """Raise :class:`LedgerError` unless ``record`` is a well-formed ledger dict with a valid code."""
    if not isinstance(record, Mapping):
        raise LedgerError(f"record must be a mapping, got {type(record).__name__}")
    missing = [k for k in _REQUIRED_FIELDS if k not in record]
    if missing:
        raise LedgerError(f"record missing required fields: {missing}")
    if record["state_code"] not in STATE_CODES:
        raise LedgerError(f"invalid state_code {record['state_code']!r}; expected one of {STATE_CODES}")
    if not isinstance(record["missing"], (list, tuple)) or not record["missing"]:
        raise LedgerError("record 'missing' must be a non-empty list")
    if not isinstance(record["explored_paths"], (list, tuple)):
        raise LedgerError("record 'explored_paths' must be a list")


def _utcnow() -> str:
    return datetime.now(timezone.utc).isoformat()


def record_from_investigation(investigation: Any, signals: AbstainSignals, *,
                              now: Callable[[], str] | None = None) -> AbstainRecord:
    """Build the coded :class:`AbstainRecord` for a finished, abstaining ``investigation``.

    Duck-typed on the investigator's ``Investigation`` (``question``, ``answer``, ``iterations``,
    ``tool_calls``, ``usage``, ``model``, ``elapsed_s``, ``stop_reason``, ``error``) so this module
    never imports the investigator. ``signals.stop_reason`` / ``signals.evidence_text`` are expected to
    already reflect the finished run.
    """
    code = classify(signals)
    answer = investigation.answer
    return AbstainRecord(
        question=investigation.question,
        state_code=code,
        evidence_obligation=_obligation_text(code, signals.evidence_text),
        missing=_missing(code, signals),
        explored_paths=tuple(investigation.tool_calls),
        stop_reason=investigation.stop_reason,
        iterations=investigation.iterations,
        tool_calls=tuple(investigation.tool_calls),
        model=investigation.model,
        elapsed_s=round(float(investigation.elapsed_s), 3),
        cost_usd=round(investigation.usage.cost_usd(investigation.model), 6),
        confidence=answer.confidence,
        error=investigation.error,
        recorded_at=(now or _utcnow)(),
    )


_WRITE_LOCK = threading.Lock()


def default_path() -> Path:
    """The production ledger location: ``artifacts/abstain_ledger.jsonl`` (gitignored)."""
    return settings.ARTIFACTS_DIR / "abstain_ledger.jsonl"


def write_record(record: AbstainRecord, path: str | Path | None = None) -> Path:
    """Append ``record`` as one JSON line to the ledger; return the path written.

    Thread-safe (run.py answers gold-100 with a ThreadPoolExecutor): a process-level lock serializes
    the single-line append so concurrent investigator abstains never interleave a line.
    """
    p = Path(path) if path is not None else default_path()
    p.parent.mkdir(parents=True, exist_ok=True)
    line = json.dumps(record.to_dict(), ensure_ascii=False)
    with _WRITE_LOCK:
        with p.open("a", encoding="utf-8") as fh:
            fh.write(line + "\n")
    return p


def record_abstain(investigation: Any, signals: AbstainSignals,
                   path: str | Path | None = None, *,
                   now: Callable[[], str] | None = None) -> AbstainRecord:
    """Build and persist an abstain record in one call; return the record.

    Best-effort *persistence*: a filesystem error is reported to stderr and swallowed so the answer
    path is never broken by observability. Building the record itself (the coded diagnosis) always
    succeeds, so the returned record always carries a valid state code.
    """
    record = record_from_investigation(investigation, signals, now=now)
    try:
        write_record(record, path)
    except OSError as e:  # pragma: no cover — filesystem failure is environmental, not logic
        print(f"[abstain_ledger] failed to write record: {type(e).__name__}: {e}", file=sys.stderr)
    return record
