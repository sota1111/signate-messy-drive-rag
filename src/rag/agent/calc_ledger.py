"""SOT-2495 — typed calculation ledger for numeric answers (数値回答の型付き計算証跡).

Parent SOT-2460. The commit-side twin of :mod:`src.rag.agent.abstain_ledger`: where the abstain ledger
codes *why an answer was withheld*, this ledger records *how a numeric answer was derived*. Every time
the investigator commits a numeric answer, a per-question record is appended to
``artifacts/calc_ledger.jsonl`` treating the number as **data extraction + typing + program execution**
(PAL/FinQA style) rather than free-text generation:

* a **typed value** — ``raw_text / parsed_value / unit / scale / time_period / entity`` plus its
  ``source_range`` (file + sheet + A1 cell range) and the ``formula`` that produced it;
* the **compute steps** — for every deterministic derivation tool the investigator ran (compute /
  corpus_aggregate / chart / pivot), the input rows, excluded rows, normalization rules, executed code
  and output — the calculation証跡 an independent re-executor (後続 SOT-2501) can replay.

Design invariant — *observer, never decider*
--------------------------------------------
Like the abstain ledger this module is a pure **observer**: it inspects an already-finished
:class:`Investigation` and writes a record. It NEVER influences the commit-vs-abstain decision
(受け入れ条件②: 回答挙動は不変・記録のみ). The investigator collects lightweight per-tool derivation
signals as it dispatches, and only after the answer is fixed does it (when the answer is numeric) build
and write the record here.

Missing-entry detection — *ungrounded numeric answers are structurally visible*
------------------------------------------------------------------------------
:func:`is_numeric_answer` decides whether an answer requires a ledger entry (it carries a numeric
quantity). The investigator routes every such commit through :func:`record_calc`, so a numeric answer
without a record is impossible. Whether that number was actually **computed** is a separate, recorded
fact: a record with no compute step has ``grounded=False`` — the machine-detectable signature of a
台帳無し(暗算) numeric answer (:func:`is_grounded` / :func:`ungrounded`), which is exactly what the
independent re-executor must flag.
"""
from __future__ import annotations

import json
import re
import sys
import threading
import unicodedata
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

from config import settings
from src.rag.tools import contract as _contract

# --------------------------------------------------------------------------- numeric-source tools
# Deterministic derivation tools whose {value, evidence, method} contract carries a computed number and
# its provenance. Only these ground a numeric answer; a number produced without one of them is 暗算.
_NUMERIC_SOURCE_TOOLS = frozenset({"compute", "corpus_aggregate", "read_chart_values", "pptx_pivot"})

# Units recognized immediately after the numeric literal (longest match wins, so 万円 beats 円).
_UNITS: tuple[str, ...] = (
    "万円", "億円", "千円", "円", "ドル", "ユーロ", "ポンド", "元",
    "時間", "日間", "週間", "ヶ月", "カ月", "か月", "年", "日", "分", "秒",
    "人", "件", "歳", "点", "倍", "割", "個", "回", "台", "冊", "本",
    "%", "％", "ポイント", "pt", "km", "kg", "cm", "mm", "m", "g",
)
# Scale words that may sit between the number and its unit (1,234万円 → value 1234, scale 10000, 円).
_SCALE_WORDS: dict[str, int] = {"千": 1000, "万": 10000, "億": 100000000, "兆": 1000000000000}

# A signed integer/decimal literal with optional thousands separators (¥/$ style commas).
_NUM_RE = re.compile(r"[+-]?\d{1,3}(?:,\d{3})+(?:\.\d+)?|[+-]?\d+(?:\.\d+)?")
# A date / period expression, mined from question+answer for the typed value's time_period field.
# Ordered longest-first so a full date wins over its year-month prefix.
_PERIOD_RE = re.compile(
    r"\d{4}年\d{1,2}月\d{1,2}日"          # 2025年10月15日
    r"|\d{4}年\d{1,2}月"                 # 2025年10月
    r"|\d{4}[-/]\d{1,2}[-/]\d{1,2}"      # 2025-08-15 / 2025/8/15
    r"|\d{4}[-/]\d{1,2}"                 # 2025-08
    r"|\d{1,2}月\d{1,2}日"               # 8月15日
    r"|第[1-4]四半期|[FHQ][1-4]"          # 第3四半期 / Q3 / H1
)


class LedgerError(ValueError):
    """Raised when a calc-ledger record fails schema validation."""


# --------------------------------------------------------------------------- typed value parsing
def _nfkc(text: str) -> str:
    return unicodedata.normalize("NFKC", str(text or ""))


def is_numeric_answer(answer: str) -> bool:
    """True iff ``answer`` carries a numeric quantity — i.e. it *requires* a ledger entry.

    Detection is a contains-a-number test (any signed integer/decimal literal) over the NFKC-normalized
    answer, biased toward recording: a column-name answer such as ``"bmi"`` / ``"campaign"`` has no
    digits and is excluded, while every derived amount / count / ratio / 年 answer is included. Errs
    toward True so a numeric answer is never silently left off the ledger (受け入れ条件①).
    """
    a = _nfkc(answer).strip()
    if not a:
        return False
    return _NUM_RE.search(a) is not None


def _to_number(literal: str) -> int | float:
    """Parse a matched numeric literal (commas stripped) into an int or float."""
    clean = literal.replace(",", "")
    if "." in clean:
        return float(clean)
    return int(clean)


def _unit_and_scale(tail: str) -> tuple[str | None, int]:
    """From the text right after the number, resolve an optional scale word and unit.

    ``1,234万円`` → tail ``万円`` → (unit ``円``, scale ``10000``); ``958`` → (None, 1);
    ``22,000円の増加`` → (``円``, 1). A leading scale word is consumed first, then the longest unit token.
    """
    scale = 1
    rest = tail
    if rest[:1] in _SCALE_WORDS:
        scale = _SCALE_WORDS[rest[0]]
        rest = rest[1:]
    unit: str | None = None
    for u in _UNITS:
        if rest.startswith(u):
            unit = u
            break
    # A bare 万円/億円/千円 unit already implies the scale; normalize it to (円, scale).
    if unit in ("千円", "万円", "億円"):
        scale = _SCALE_WORDS[unit[0]]
        unit = "円"
    return unit, scale


def _period_of(*texts: str) -> str | None:
    for t in texts:
        m = _PERIOD_RE.search(_nfkc(t))
        if m:
            return m.group(0)
    return None


@dataclass(frozen=True)
class SourceRange:
    """Where a value came from: a corpus file, its sheet, and the A1 cell range touched."""

    file: str | None = None
    sheet: str | None = None
    cell: str | None = None        # A1-style range, e.g. "B1:D45"
    project: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {"file": self.file, "sheet": self.sheet, "cell": self.cell, "project": self.project}


@dataclass(frozen=True)
class TypedValue:
    """A numeric answer typed as data (PAL/FinQA): the raw text, its parsed value, and its provenance."""

    raw_text: str
    parsed_value: int | float | None
    unit: str | None
    scale: int                      # multiplier a 万/億 scale word implies (1 when none)
    time_period: str | None
    entity: str | None
    source_range: SourceRange
    formula: str | None

    def to_dict(self) -> dict[str, Any]:
        return {
            "raw_text": self.raw_text,
            "parsed_value": self.parsed_value,
            "unit": self.unit,
            "scale": self.scale,
            "time_period": self.time_period,
            "entity": self.entity,
            "source_range": self.source_range.to_dict(),
            "formula": self.formula,
        }


@dataclass(frozen=True)
class ComputeStep:
    """One deterministic derivation execution the investigator ran, as replayable計算証跡."""

    tool: str
    file: str | None = None
    sheet: str | None = None
    range: str | None = None                 # A1 range the code touched
    columns_used: tuple[str, ...] = ()
    input_rows: int | None = None            # rows the code saw
    excluded_rows: int | None = None         # rows dropped in normalization (e.g. leading blanks)
    normalization: tuple[str, ...] = ()      # normalization rules applied to reach input rows
    code: str | None = None                  # the executed expression / formula
    output: Any = None                       # the tool's returned value

    def to_dict(self) -> dict[str, Any]:
        return {
            "tool": self.tool,
            "file": self.file,
            "sheet": self.sheet,
            "range": self.range,
            "columns_used": list(self.columns_used),
            "input_rows": self.input_rows,
            "excluded_rows": self.excluded_rows,
            "normalization": list(self.normalization),
            "code": self.code,
            "output": self.output,
        }


def _step_from_contract(tool: str, resp: Mapping[str, Any]) -> ComputeStep:
    """Build a :class:`ComputeStep` from a numeric-source tool's {value, evidence, method} contract."""
    ev = dict(resp.get("evidence") or {})
    method = dict(resp.get("method") or {})
    trace = dict(method.get("trace") or {})
    cols = ev.get("columns_used") or ()
    return ComputeStep(
        tool=tool,
        file=ev.get("file") or ev.get("source"),
        sheet=ev.get("sheet"),
        range=ev.get("range") or ev.get("cell"),
        columns_used=tuple(str(c) for c in cols),
        input_rows=trace.get("input_rows", ev.get("rows")),
        excluded_rows=trace.get("excluded_rows"),
        normalization=tuple(str(n) for n in (trace.get("normalization") or ())),
        code=method.get("code"),
        output=resp.get("value"),
    )


# --------------------------------------------------------------------------- signals (observer)
@dataclass
class CalcSignals:
    """Side-effect-free tally the investigator fills while dispatching derivation tools.

    Every field is derived only by *observing* tool responses (never altering them), so populating a
    :class:`CalcSignals` cannot change the answer path. ``question`` is set once from the investigation.
    """

    question: str = ""
    steps: list[ComputeStep] = field(default_factory=list)
    entity: str | None = None       # last non-empty project/entity a derivation tool reported

    def observe(self, tool_name: str, resp: Any) -> None:
        """Fold one dispatched tool response into the compute trail. Pure bookkeeping."""
        if tool_name not in _NUMERIC_SOURCE_TOOLS:
            return
        if not (isinstance(resp, Mapping) and _contract.is_contract(resp)):
            return
        if resp.get("value") in (None, "", [], {}):
            return
        step = _step_from_contract(tool_name, resp)
        self.steps.append(step)
        proj = (resp.get("evidence") or {}).get("project")
        if proj:
            self.entity = str(proj)

    def typed_value(self, answer: str) -> TypedValue:
        """Type the committed ``answer`` string against the observed compute trail."""
        raw = _nfkc(answer).strip()
        m = _NUM_RE.search(raw)
        parsed: int | float | None = None
        unit: str | None = None
        scale = 1
        if m:
            parsed = _to_number(m.group(0))
            unit, scale = _unit_and_scale(raw[m.end():])
        last = self.steps[-1] if self.steps else None
        source = SourceRange(
            file=last.file if last else None,
            sheet=last.sheet if last else None,
            cell=last.range if last else None,
            project=self.entity,
        )
        return TypedValue(
            raw_text=raw,
            parsed_value=parsed,
            unit=unit,
            scale=scale,
            time_period=_period_of(self.question, raw),
            entity=self.entity,
            source_range=source,
            formula=last.code if last else None,
        )


# --------------------------------------------------------------------------- record + schema
@dataclass(frozen=True)
class CalcRecord:
    """One per-question calculation証跡 line (serialized to JSONL).

    Most records contain a literal number.  A ``numeric`` question contract may instead return the
    label selected by a calculation (for example correlation ``idxmax() -> \"bmi\"``); ``contract``
    keeps that replayable derived answer distinguishable from an ordinary text lookup.
    """

    question: str
    answer: str
    typed_value: TypedValue
    compute_steps: tuple[ComputeStep, ...]
    grounded: bool                       # True iff ≥1 compute step backs the number (not 暗算)
    explored_paths: tuple[str, ...]      # the tool calls actually tried, in order
    model: str
    iterations: int
    elapsed_s: float
    cost_usd: float
    confidence: float
    recorded_at: str
    contract: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "recorded_at": self.recorded_at,
            "question": self.question,
            "answer": self.answer,
            "typed_value": self.typed_value.to_dict(),
            "compute_steps": [s.to_dict() for s in self.compute_steps],
            "grounded": self.grounded,
            "explored_paths": list(self.explored_paths),
            "model": self.model,
            "iterations": self.iterations,
            "elapsed_s": self.elapsed_s,
            "cost_usd": self.cost_usd,
            "confidence": self.confidence,
            "contract": self.contract,
        }


def is_grounded(record: Mapping[str, Any]) -> bool:
    """True iff a ledger record's number is backed by ≥1 recorded compute step."""
    return bool(record.get("compute_steps"))


def ungrounded(records: Sequence[Mapping[str, Any]]) -> list[Mapping[str, Any]]:
    """The numeric-answer records that carry NO compute step — 暗算 answers to flag (欠落検出)."""
    return [r for r in records if not is_grounded(r)]


_REQUIRED_FIELDS = (
    "recorded_at", "question", "answer", "typed_value", "compute_steps",
    "grounded", "explored_paths", "model",
)
_TYPED_FIELDS = ("raw_text", "parsed_value", "unit", "scale", "source_range", "formula")


def validate(record: Mapping[str, Any]) -> None:
    """Raise :class:`LedgerError` unless ``record`` is a well-formed calc-ledger dict."""
    if not isinstance(record, Mapping):
        raise LedgerError(f"record must be a mapping, got {type(record).__name__}")
    missing = [k for k in _REQUIRED_FIELDS if k not in record]
    if missing:
        raise LedgerError(f"record missing required fields: {missing}")
    tv = record["typed_value"]
    if not isinstance(tv, Mapping):
        raise LedgerError("record 'typed_value' must be a mapping")
    tv_missing = [k for k in _TYPED_FIELDS if k not in tv]
    if tv_missing:
        raise LedgerError(f"typed_value missing fields: {tv_missing}")
    if not isinstance(tv.get("source_range"), Mapping):
        raise LedgerError("typed_value 'source_range' must be a mapping")
    if not isinstance(record["compute_steps"], (list, tuple)):
        raise LedgerError("record 'compute_steps' must be a list")
    if not isinstance(record["explored_paths"], (list, tuple)):
        raise LedgerError("record 'explored_paths' must be a list")
    if bool(record["grounded"]) != bool(record["compute_steps"]):
        raise LedgerError("record 'grounded' must equal (compute_steps is non-empty)")


def _utcnow() -> str:
    return datetime.now(timezone.utc).isoformat()


def record_from_investigation(investigation: Any, signals: CalcSignals, *,
                              now: Callable[[], str] | None = None) -> CalcRecord:
    """Build the typed :class:`CalcRecord` for a finished, numeric-answering ``investigation``.

    Duck-typed on the investigator's ``Investigation`` (``question``, ``answer``, ``iterations``,
    ``tool_calls``, ``usage``, ``model``, ``elapsed_s``) so this module never imports the investigator.
    """
    answer = investigation.answer
    signals.question = signals.question or investigation.question
    typed = signals.typed_value(answer.answer)
    steps = tuple(signals.steps)
    return CalcRecord(
        question=investigation.question,
        answer=answer.answer,
        typed_value=typed,
        compute_steps=steps,
        grounded=bool(steps),
        explored_paths=tuple(investigation.tool_calls),
        model=investigation.model,
        iterations=investigation.iterations,
        elapsed_s=round(float(investigation.elapsed_s), 3),
        cost_usd=round(investigation.usage.cost_usd(investigation.model), 6),
        confidence=answer.confidence,
        recorded_at=(now or _utcnow)(),
        contract=getattr(investigation, "contract", None),
    )


_WRITE_LOCK = threading.Lock()


def default_path() -> Path:
    """The production ledger location: ``artifacts/calc_ledger.jsonl`` (gitignored)."""
    return settings.ARTIFACTS_DIR / "calc_ledger.jsonl"


def write_record(record: CalcRecord, path: str | Path | None = None) -> Path:
    """Append ``record`` as one JSON line to the ledger; return the path written.

    Thread-safe (run.py answers gold-100 with a ThreadPoolExecutor): a process-level lock serializes
    the single-line append so concurrent numeric commits never interleave a line.
    """
    p = Path(path) if path is not None else default_path()
    p.parent.mkdir(parents=True, exist_ok=True)
    line = json.dumps(record.to_dict(), ensure_ascii=False)
    with _WRITE_LOCK:
        with p.open("a", encoding="utf-8") as fh:
            fh.write(line + "\n")
    return p


def record_calc(investigation: Any, signals: CalcSignals,
                path: str | Path | None = None, *,
                now: Callable[[], str] | None = None) -> CalcRecord:
    """Build and persist a calc record in one call; return the record.

    Best-effort *persistence*: a filesystem error is reported to stderr and swallowed so the answer path
    is never broken by observability. Building the record itself always succeeds.
    """
    record = record_from_investigation(investigation, signals, now=now)
    try:
        write_record(record, path)
    except OSError as e:  # pragma: no cover — filesystem failure is environmental, not logic
        print(f"[calc_ledger] failed to write record: {type(e).__name__}: {e}", file=sys.stderr)
    return record
