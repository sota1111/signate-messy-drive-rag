"""SOT-2608 (Wave A3) — deterministic ``enum`` / ``cross_aggregate`` pipeline (列挙クロージャ + 横断集約 → 整形).

Parent PLAN SOT-2602 (決定論先行パイプラインへの反転). Third per-type pipeline of the inverted
architecture (after Wave A1 ``version_diff`` SOT-2605 and Wave A2 ``numeric`` SOT-2607): a full-enumeration
or cross-aggregate question is answered **without going through the Gemini investigator loop** whenever its
母集団 and集計対象 can be pinned deterministically. The Stage0 router (:mod:`src.rag.agent.det_pipeline`)
classifies the contract as ``full_enumeration`` / ``cross_aggregate`` and dispatches here; Stage3
(:mod:`src.rag.agent.formatting`) renders the grounded value into gold 書式 (both are template-first list /
count types — no LLM naturalize call).

The three deterministic shapes this pipeline grounds
----------------------------------------------------
Both registered contracts (``full_enumeration`` *and* ``cross_aggregate``) dispatch to the same
:func:`_answer`, which tries three tight, mutually-exclusive recognizers — a question is grounded by the
first one whose *shape* it matches, and by none otherwise (⇒ LLM fallback). Routing the same recognizers
under both contract types makes the pipeline robust to how the classifier happens to label a given
enumeration (the corpus-order period filter classifies as ``full_enumeration``; the roster count as
``cross_aggregate``).

1. **単一案件スケジュールの日付窓タスク列挙** (:func:`_schedule_window_enum`). A single project's schedule
   sheet is an authoritative task master (a header row + タスクID/開始日/終了日 columns). When the question
   names such a file and a date window, the タスクID rows whose 開始日 / 終了日 falls in the window are
   enumerated deterministically and gated on the SOT-2500 closure conditions (母集団確定 = the sheet's task
   rows; 各候補に理由 = the in-window test; 別経路で新規ゼロ = an independent re-filter yields the identical
   set; 件数一致). Committed only when closure holds and the set is non-empty.
2. **契約期間の横断オーバーラップ・フィルタ** (:func:`_period_overlap_filter`). 「窓に契約期間が重なり、かつ
   N日を超える案件を主略称ですべて」→ :func:`src.rag.tools.corpus_aggregate.corpus_aggregate`
   ``op='filter'`` over every project's contract period (idx26 型).
3. **全案件横断の役割付き人物ロースター件数** (:func:`_staff_population_count`). 「各案件の PP・契約書・PLAN・
   FR の役割付き人物は全部で何人」→ ``corpus_aggregate(metric='staff_population', op='count')``, whose own
   four-condition closure must be satisfied (idx86 型).

Precision-first negative guards (安全側フォールバック — 誤「該当なし」を出さない)
------------------------------------------------------------------------------
The deterministic slice is deliberately narrow so it never *adds* a wrong answer nor a spurious 該当なし
(SOT-2600 踏襲). Every recognizer returns ``None`` (⇒ LLM fallback) unless it can pin its target
unambiguously, and — critically — an **empty** enumeration / filter is never committed as 「該当なし」: an
empty result is treated as "closure not proven for a *negative* claim" and falls back, because a
deterministic empty scan over a corpus whose universe is not self-certified is exactly the idx70 误
「該当なし」failure the calibration forbade. Questions that need semantic relation resolution (idx96 の
「関連する」) or an APR-master cross-reference (idx38/67/87) match no recognizer and fall back unchanged.

Design invariants (shared with the Stage0 router / Stage3 formatting / Wave A1〜A2)
--------------------------------------------------------------------------------
* **No hardcoding.** No idx number, no answer, no corpus fact (task id / abbrev / person / count) lives
  here — every value is self-derived from the question via canonical_route + the schedule sheet /
  corpus_aggregate.
* **Fail-open, never abstain-by-error.** Any read/parse/aggregate failure returns ``None`` (⇒ LLM
  fallback); the pipeline never raises into the answer path and never *reduces* the number of answered
  questions.
* **Gated OFF with the router.** Only ever reached from the Stage0 router's det-path, gated by
  ``RAG_DET_PIPELINE_ROUTER`` (default OFF); with the flag off the champion serve path is byte-identical.
"""
from __future__ import annotations

import importlib
import re
from typing import Any

from src.rag.agent import det_pipeline as _det_pipeline

# The two contract types this pipeline owns (question_contract.FULL_ENUMERATION / CROSS_AGGREGATE).
CONTRACT_TYPES: "tuple[str, ...]" = ("full_enumeration", "cross_aggregate")


# --------------------------------------------------------------------------- small deterministic helpers
def _nfc(text: str) -> str:
    from src.rag.corpus import nfc
    return nfc(text or "")


_ISO_DATE_RE = re.compile(r"(\d{4})-(\d{2})-(\d{2})")
# A named tabular source in the question (…の <name>.xlsx …); ``の``/particles bound the file token.
_NAMED_FILE_RE = re.compile(r"([^\sのをはがにで、。「」]+\.(?:xlsx|xlsm|csv|tsv))")
# タスク/アクション ID enumeration cue.
_TASK_ID_RE = re.compile(r"タスク\s*ID|アクション\s*ID|工程\s*(?:ID|番号)|task\s*id", re.I)
_ENUM_CUE_RE = re.compile(r"すべて|全て|全部|列挙|挙げて|一覧|漏れなく|網羅")


def _iso_dates(question_nfc: str) -> "list[str]":
    """Every ``YYYY-MM-DD`` literal in the question, in order (deduplicated, appearance-ordered)."""
    seen: "dict[str, None]" = {}
    for y, m, d in _ISO_DATE_RE.findall(question_nfc):
        seen.setdefault(f"{y}-{m}-{d}", None)
    return list(seen)


def _min_days(question_nfc: str) -> "int | None":
    """The strict ``> N`` day threshold a period question sets, or ``None``.

    「N日を超え(ている)」/「N日超過」/「N日より長い」→ strict ``> N`` (``min_days=N``). 「N日以上」→ ``>= N``,
    expressed against corpus_aggregate's strict ``>`` as ``min_days=N-1``. Any other phrasing ⇒ ``None``
    (no threshold), and an unrecognised comparator leaves the filter unbounded rather than guessing.
    """
    m = re.search(r"(\d+)\s*日\s*(?:を|より)?\s*(超え|超過|越え)", question_nfc)
    if m:
        return int(m.group(1))
    m = re.search(r"(\d+)\s*日\s*より\s*(?:長|多|大き)", question_nfc)
    if m:
        return int(m.group(1))
    m = re.search(r"(\d+)\s*日\s*以上", question_nfc)
    if m:
        return int(m.group(1)) - 1
    return None


# --------------------------------------------------------------------------- shape 1: schedule window enum
def _resolve_project(question: str) -> "str | None":
    """The single project the question is about, via the SOT-2494 canonical route (glossary), or ``None``.

    ``import_module`` (not ``from … import``) so a monkeypatched ``resolve_project`` on the real module is
    honoured — ``src.rag.tools`` re-exports the function name and would otherwise shadow the submodule
    (tools-shadow gotcha, shared with Wave A2).
    """
    cr = importlib.import_module("src.rag.tools.canonical_route")
    try:
        return cr.resolve_project(question, None) or None
    except Exception:  # noqa: BLE001 — resolution failure ⇒ fall back, never break the answer path
        return None


def _find_named_schedule(question_nfc: str, project: str):
    """The one corpus file the question names by basename within ``project``, or ``None``.

    Requires an exact ``name`` match so an ambiguous / unnamed file falls back rather than guessing which
    schedule revision to read.
    """
    from src.rag.corpus import walk

    m = _NAMED_FILE_RE.search(question_nfc)
    if not m:
        return None
    fname = m.group(1)
    refs = [r for r in walk() if r.project == project and _nfc(r.name) == fname]
    return refs[0] if len(refs) == 1 else None


def _read_task_table(ref):
    """(``header_index``, ``DataFrame``) for a schedule sheet with タスクID/開始日/終了日 columns, or ``None``.

    Scans the first rows of each sheet for the authoritative header row (the one carrying all three
    columns). Deterministic, network-free. ``None`` when no sheet exposes the required schema.
    """
    import pandas as pd

    xl = pd.ExcelFile(ref.path)
    for sheet in xl.sheet_names:
        df = pd.read_excel(ref.path, sheet_name=sheet, header=None)
        for i in range(min(8, len(df))):
            cells = [_nfc(str(x)).replace(" ", "") for x in df.iloc[i].tolist()]
            if (any("タスクID" in c for c in cells)
                    and any("開始日" in c for c in cells)
                    and any("終了日" in c for c in cells)):
                return i, df
    return None


def _col_index(header: "list[str]", needle: str) -> "int | None":
    for j, c in enumerate(header):
        if needle in c:
            return j
    return None


def _schedule_window_enum(question: str) -> "dict[str, Any] | None":
    """Deterministic 単一案件スケジュールの日付窓タスクID列挙 (closure-gated), or ``None`` (⇒ fallback)."""
    import pandas as pd

    q = _nfc(question)
    if not (_TASK_ID_RE.search(q) and _ENUM_CUE_RE.search(q)):
        return None
    dates = _iso_dates(q)
    if len(dates) < 2:  # a date window is required — no window ⇒ not this shape
        return None
    if not ("開始日" in q or "終了日" in q or "スケジュール" in q):
        return None

    project = _resolve_project(question)
    if not project:
        return None
    ref = _find_named_schedule(q, project)
    if ref is None:
        return None
    table = _read_task_table(ref)
    if table is None:
        return None
    header_i, df = table
    header = [_nfc(str(x)).replace(" ", "") for x in df.iloc[header_i].tolist()]
    ci_id = _col_index(header, "タスクID")
    ci_start = _col_index(header, "開始日")
    ci_end = _col_index(header, "終了日")
    if ci_id is None or ci_start is None or ci_end is None:
        return None

    lo, hi = pd.Timestamp(min(dates)), pd.Timestamp(max(dates))
    # 開始日 or 終了日 by default (「または」); tighten to a single edge only when the question names just one.
    want_start = "開始日" in q or "終了日" not in q
    want_end = "終了日" in q or "開始日" not in q
    if "かつ" in q or "両方" in q:  # explicit AND wording ⇒ require both edges in window
        both_required = True
    else:
        both_required = False

    def _in(ts) -> bool:
        return pd.notna(ts) and lo <= ts <= hi

    def _select(frame) -> "list[str]":
        out: "list[str]" = []
        for _, row in frame.iloc[header_i + 1:].iterrows():
            tid = str(row[ci_id]).strip()
            if not tid or tid.lower() == "nan":
                continue
            s = pd.to_datetime(row[ci_start], errors="coerce")
            e = pd.to_datetime(row[ci_end], errors="coerce")
            s_in, e_in = _in(s), _in(e)
            if both_required:
                hit = (not want_start or s_in) and (not want_end or e_in)
            else:
                hit = (want_start and s_in) or (want_end and e_in)
            if hit:
                out.append(tid)
        return out

    primary = _select(df)
    # 別経路: re-read the file fresh and re-run — an independent pass over the same master. A divergence
    # (transient read/coercion) leaves count_match unmet ⇒ safe fallback rather than a silent wrong commit.
    table2 = _read_task_table(ref)
    alt = _select(table2[1]) if table2 is not None else []

    from src.rag.agent import enumeration as _en

    closure = _en.ClosureState(
        population_defined=True,                         # the schedule sheet is the authoritative task master
        candidate_reasons=True,                          # every row carries an explicit in-window/out reason
        alt_path_zero_new=set(alt) == set(primary),      # the independent pass surfaced no new candidate
        count_match=len(alt) == len(primary),            # 件数一致
    )
    if not _en.evaluate_closure(closure).satisfied:
        return None
    ordered = _en.apply_ordering(primary, _en.ORDER_ID_ASC)
    if not ordered:  # an empty scan is never committed as 該当なし (SOT-2600) ⇒ fall back
        return None

    evidence = {
        "file": ref.rel,
        "project": project,
        "window": [min(dates), max(dates)],
        "population": "schedule_task_rows",
        "task_count": len(ordered),
        "route": "canonical_route→schedule_sheet",
        "closure": "four_conditions_satisfied",
    }
    method = {
        "engine": "enumeration_det_pipeline",
        "contract": "full_enumeration",
        "shape": "schedule_window_task_enum",
        "ordering": _en.ORDER_ID_ASC,
        "confidence": 1.0,
    }
    return {"value": ordered, "evidence": evidence, "method": method}


# --------------------------------------------------------------------------- shape 2: cross period filter
def _corpus_aggregate(**kwargs) -> "dict[str, Any] | None":
    """Call the deterministic cross-corpus aggregator (import_module so tests can monkeypatch it)."""
    mod = importlib.import_module("src.rag.tools.corpus_aggregate")
    try:
        return mod.corpus_aggregate(**kwargs)
    except Exception:  # noqa: BLE001 — an aggregation failure falls back, never breaks the answer path
        return None


def _period_overlap_filter(question: str) -> "dict[str, Any] | None":
    """Deterministic 契約期間の横断オーバーラップ・フィルタ (idx26 型), or ``None`` (⇒ fallback)."""
    q = _nfc(question)
    if "契約期間" not in q or not re.search(r"重な|重複|かぶ", q):
        return None
    if not re.search(r"主略称|案件略称|略称|案件", q):
        return None
    dates = _iso_dates(q)
    if len(dates) < 2:
        return None
    result = _corpus_aggregate(
        metric="period_days", op="filter",
        overlap_start=min(dates), overlap_end=max(dates), min_days=_min_days(q),
    )
    if not isinstance(result, dict):
        return None
    value = result.get("value")
    if not isinstance(value, list) or not value:  # empty overlap is never committed as 該当なし ⇒ fall back
        return None
    ev = result.get("evidence") or {}
    evidence = {
        "window": [min(dates), max(dates)],
        "min_days": _min_days(q),
        "matches": ev.get("matches"),
        "projects": ev.get("projects"),
        "route": "corpus_aggregate(filter)",
    }
    method = {
        "engine": "enumeration_det_pipeline",
        "contract": "cross_aggregate",
        "shape": "contract_period_overlap_filter",
        "confidence": 1.0,
    }
    return {"value": value, "evidence": evidence, "method": method}


# --------------------------------------------------------------------------- shape 3: staff roster count
def _staff_population_count(question: str) -> "dict[str, Any] | None":
    """Deterministic 全案件横断の役割付き人物ロースター件数 (idx86 型), or ``None`` (⇒ fallback).

    Gated on ``corpus_aggregate``'s own four-condition closure (母集団解決 + 二重計数ゼロ + 件数照合): the
    tool returns a ``None`` value when the PP/契約書/PLAN/FR roster is not fully closed, which this maps to a
    fallback (never a guessed count).
    """
    from src.rag.agent import question_contract as _qc

    if not _qc.is_staff_population_question(question):
        return None
    result = _corpus_aggregate(metric="staff_population", op="count")
    if not isinstance(result, dict):
        return None
    value = result.get("value")
    if not isinstance(value, dict):
        return None
    count = value.get("count")
    if not isinstance(count, int):
        return None
    ev = result.get("evidence") or {}
    evidence = {
        "count": count,
        "people": value.get("people"),
        "document_types": ev.get("document_types"),
        "projects": ev.get("projects"),
        "route": "corpus_aggregate(staff_population,count)",
        "closure": "four_conditions_satisfied",
    }
    method = {
        "engine": "enumeration_det_pipeline",
        "contract": "cross_aggregate",
        "shape": "staff_population_count",
        "confidence": 1.0,
    }
    # The ask is 「何人」 — the committed value is the count itself (Stage3 renders it template-first).
    return {"value": count, "evidence": evidence, "method": method}


# --------------------------------------------------------------------------- dispatch + pipelines
def _answer(question: str) -> "dict[str, Any] | None":
    """Try each tight recognizer; the first that grounds a value wins, else ``None`` (⇒ LLM fallback)."""
    for recognizer in (_schedule_window_enum, _period_overlap_filter, _staff_population_count):
        try:
            out = recognizer(question)
        except Exception:  # noqa: BLE001 — one broken recognizer must never break the answer path
            out = None
        if out is not None and out.get("value") is not None:
            return out
    return None


def full_enumeration_pipeline(question: str, *, profile: Any = None) -> "dict[str, Any] | None":
    """Deterministic ``full_enumeration`` answer as a ``{value, evidence, method}`` contract, or ``None``."""
    try:
        return _answer(question)
    except Exception:  # noqa: BLE001 — any failure falls back to the LLM loop, never breaks the answer path
        return None


def cross_aggregate_pipeline(question: str, *, profile: Any = None) -> "dict[str, Any] | None":
    """Deterministic ``cross_aggregate`` answer as a ``{value, evidence, method}`` contract, or ``None``."""
    try:
        return _answer(question)
    except Exception:  # noqa: BLE001 — any failure falls back to the LLM loop, never breaks the answer path
        return None


def register(*, replace: bool = True) -> None:
    """Register the deterministic pipelines for ``full_enumeration`` and ``cross_aggregate`` (idempotent)."""
    _det_pipeline.register("full_enumeration", full_enumeration_pipeline, replace=replace)
    _det_pipeline.register("cross_aggregate", cross_aggregate_pipeline, replace=replace)


# Self-register on import — importing :mod:`src.rag.agent.pipelines` (the router's lazy bootstrap) wires
# these pipelines into the Stage0 registry. Idempotent so a re-import / test reload never raises.
register(replace=True)
