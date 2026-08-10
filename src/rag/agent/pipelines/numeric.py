"""SOT-2607 (Wave A2) — deterministic ``numeric`` pipeline (canonical → compute → exec_verifier → 整形).

Parent PLAN SOT-2602 (決定論先行パイプラインへの反転). Second per-type pipeline of the inverted
architecture (after Wave A1 ``version_diff``, SOT-2605): a ``numeric`` / derived-calculation question is
answered **without going through the Gemini investigator loop** whenever its計算対象 can be pinned
deterministically. The Stage0 router (:mod:`src.rag.agent.det_pipeline`) classifies the contract as
``numeric`` and dispatches here; Stage3 (:mod:`src.rag.agent.formatting`) renders the grounded value into
gold 書式 (numeric is a template-first type — no LLM naturalize call).

Pipeline (canonical → compute → 独立再計算 → 照合)
-----------------------------------------------
1. **Stage1 — 対象データファイルを確定.** Reuse the SOT-2494 canonical route
   (:mod:`src.rag.tools.canonical_route`): resolve the question's *single* project via the glossary and
   discover that project's canonical ``train`` table — bypassing chunk retrieval. No single project ⇒
   the route declines and the pipeline falls back.
2. **Stage2 — 対象定義を確定して compute で値導出.** Deterministically pin the計算対象 from the question:
   **exactly one** recognized aggregate operation (平均/合計/最大/最小/中央値/標準偏差/分散/件数) over
   **exactly one** column whose name appears literally in the question and matches the table's columns.
   The value is derived by :mod:`src.rag.tools.compute_sandbox` (restricted pandas — 実出力厳守,
   ハードコード禁止).
3. **独立再計算 + 照合.** A second, independently-formulated compute (an explicit ``dropna`` re-expression
   that re-reads the file and re-runs) produces a re-derived record; the two are reconciled by
   :func:`src.rag.agent.exec_verifier.compare_execution` on 値/件数/対象列/単位 — which also runs the
   SOT-2508 numeric contract validation (量の定義/単位/丸め) on both. The answer is committed **only** on
   :data:`~src.rag.agent.exec_verifier.EXEC_MATCH`; any conflict / undecidable / contract violation, or an
   ambiguous / unrecognized / non-tabular target, returns ``None`` so the router falls back to the LLM
   loop (確証不能なら安全側フォールバック — 回答数を減らさない・wrong を増やさない).

Precision-first negative guards (安全側フォールバック)
--------------------------------------------------
The deterministic slice is deliberately narrow so it never *adds* a wrong answer. It declines (⇒ LLM
fallback) whenever the target is not a plain single-column aggregate: a 割合/比率 (population-scoped
denominator — SOT-2508 territory), a unit-bearing quantity (円/件/人… — scale can't be pinned safely),
or a question whose cue words mark it as a highlight-geometry / regression / version-diff / 差額 ask
(idx6/47/63/97 型 — those stay on their existing champion path). The 対象列 is required to be *uniquely*
named, so an ambiguous or unnamed column also falls back.

Design invariants (shared with the Stage0 router / Stage3 formatting / Wave A1)
------------------------------------------------------------------------------
* **No hardcoding.** No idx number, no answer, no corpus fact (column/password/scale) lives here — every
  value is self-derived from the question via canonical_route + compute.
* **Fail-open, never abstain-by-error.** Any read/parse/compute failure returns ``None`` (⇒ LLM fallback);
  the pipeline never raises into the answer path and never *reduces* the number of answered questions.
* **Gated OFF with the router.** Only ever reached from the Stage0 router's det-path, gated by
  ``RAG_DET_PIPELINE_ROUTER`` (default OFF); with the flag off the champion serve path is byte-identical.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

from src.rag.agent import det_pipeline as _det_pipeline

# The contract type this pipeline owns (question_contract.NUMERIC).
CONTRACT_TYPE = "numeric"


# --------------------------------------------------------------------------- aggregate vocabulary
@dataclass(frozen=True)
class _Aggregate:
    """One recognized single-column reducer: how to detect it and how to build its two compute exprs.

    ``primary`` / ``rederive`` are ``column -> pandas expression`` builders. ``primary`` is the direct
    reducer; ``rederive`` is an independently-formulated equivalent (explicit ``dropna`` re-expression)
    that re-reads the file and re-runs — an uncorrelated code path over the *same* target, so a transient
    read/coercion divergence surfaces as a conflict (→ safe fallback) rather than a silent wrong commit.
    """

    name: str
    keywords: tuple[str, ...]
    primary: Callable[[str], str]
    rederive: Callable[[str], str]


def _col(column: str) -> str:
    """A pandas column selector literal for ``column`` (single-quoted, embedded quotes escaped)."""
    return "df[" + repr(column) + "]"


# Ordered; each aggregate's keywords are matched case-insensitively for ascii, exactly for Japanese.
# Only plain single-column reducers live here — 相関(列選択)/回帰/差分 are intentionally excluded so the
# pipeline stays precision-first (それらは従来 LLM ループが担当).
_AGGREGATES: tuple[_Aggregate, ...] = (
    _Aggregate("mean", ("平均値", "平均", "mean", "average"),
               lambda c: f"{_col(c)}.mean()", lambda c: f"{_col(c)}.dropna().mean()"),
    _Aggregate("median", ("中央値", "median"),
               lambda c: f"{_col(c)}.median()", lambda c: f"{_col(c)}.dropna().median()"),
    _Aggregate("sum", ("合計", "総和", "総計", "総額", "sum"),
               lambda c: f"{_col(c)}.sum()", lambda c: f"{_col(c)}.dropna().sum()"),
    _Aggregate("max", ("最大値", "最大", "max"),
               lambda c: f"{_col(c)}.max()", lambda c: f"{_col(c)}.dropna().max()"),
    _Aggregate("min", ("最小値", "最小", "min"),
               lambda c: f"{_col(c)}.min()", lambda c: f"{_col(c)}.dropna().min()"),
    _Aggregate("std", ("標準偏差", "std"),
               lambda c: f"{_col(c)}.std()", lambda c: f"{_col(c)}.dropna().std()"),
    _Aggregate("var", ("分散", "variance"),
               lambda c: f"{_col(c)}.var()", lambda c: f"{_col(c)}.dropna().var()"),
    _Aggregate("count", ("件数", "行数", "レコード数", "データ数", "何件", "何行"),
               lambda c: f"int({_col(c)}.count())", lambda c: f"len({_col(c)}.dropna())"),
)

# Cue words that mark a question as NOT a plain single-column aggregate — those stay on the champion LLM
# path (idx6/47/63/97 型: 差額 / 建設年逆引き / 回帰予測 / 黄色ハイライト幾何 / 版差分). A negative guard so
# an aggregate keyword incidentally present in such a question can never trigger a wrong deterministic commit.
_EXCLUSION_CUES: tuple[str, ...] = (
    "ハイライト", "黄色", "色", "セルの色", "太字", "下線",
    "回帰", "予測", "係数", "モデル", "推論",
    "差額", "差分", "の差", "絶対値",
    "version", "旧版", "新版", "old", "更新前", "更新後", "版",
    "相関", "correlation",
)


# --------------------------------------------------------------------------- deterministic target pinning
def _nfc(text: str) -> str:
    from src.rag.corpus import nfc
    return nfc(text or "")


def _detect_aggregate(question_nfc: str) -> "_Aggregate | None":
    """The single recognized aggregate a question asks for, or ``None`` (0 / >1 ⇒ ambiguous ⇒ fallback)."""
    ql = question_nfc.lower()
    hits = [
        agg for agg in _AGGREGATES
        if any((kw.lower() in ql) if kw.isascii() else (kw in question_nfc) for kw in agg.keywords)
    ]
    return hits[0] if len(hits) == 1 else None


def _select_column(question_nfc: str, columns: "list[str]") -> "str | None":
    """The single table column named literally in the question, or ``None`` (0 / ambiguous ⇒ fallback).

    A column qualifies when its NFC name (≥2 chars, to avoid an incidental 1-char collision) is a
    substring of the NFC question. Requiring **exactly one** such column keeps the target unambiguous —
    a question that names two candidate columns (or none) falls back to the LLM loop.
    """
    named = [c for c in columns if len(_nfc(c)) >= 2 and _nfc(c) in question_nfc]
    # Deduplicate on the normalized name so two columns sharing a display label don't spuriously ambiguate.
    unique = {_nfc(c): c for c in named}
    if len(unique) != 1:
        return None
    return next(iter(unique.values()))


def _resolve_primary_table(question: str):
    """The project's canonical ``train`` table record (dict with ``rel``/``project``), or ``None``.

    Stage1: reuse the canonical route (glossary project resolution → canonical file discovery, chunk
    検索を迂回). Returns the first *tabular* (csv/xlsx) record only — a non-tabular primary or no single
    project ⇒ ``None`` (fallback).
    """
    # Resolve the *module* via importlib: the ``src.rag.tools`` package re-exports ``canonical_route`` as
    # the *function* (shadowing the submodule attribute), so ``import src.rag.tools.canonical_route`` binds
    # the function, not the module — ``import_module`` returns the real module from ``sys.modules`` and lets
    # ``resolve_project`` / ``canonical_route`` resolve (and be monkeypatched) at call time (tools-shadow gotcha).
    import importlib
    _cr = importlib.import_module("src.rag.tools.canonical_route")

    project = _cr.resolve_project(question, None)
    if not project:
        return None
    result = _cr.canonical_route(question=question, project=project, kind="train")
    records = result.get("value") if isinstance(result, dict) else None
    if not isinstance(records, list):
        return None
    for rec in records:
        if isinstance(rec, dict) and str(rec.get("ext", "")).lower() in {"csv", "xlsx", "xlsm"}:
            return {"rel": rec.get("rel"), "project": rec.get("project") or project}
    return None


# --------------------------------------------------------------------------- compute + ledger records
def _compute(rel: str, expr: str, project: "str | None"):
    """Run one restricted-pandas expression against ``rel`` and return its contract, or ``None`` on error."""
    import src.rag.tools.compute_sandbox as _cs

    try:
        return _cs.run(rel, expr, project=project)
    except Exception:  # noqa: BLE001 — a compute/read failure must fall back, not break the answer path
        return None


def _answer_string(value: Any, decimal_places: "int | None") -> "str | None":
    """The committed answer surface for ``value`` — honoring a requested 小数第N位 rounding, else plain.

    Returns ``None`` when the value is not a finite number (so the pipeline falls back rather than
    committing a non-numeric or NaN aggregate). Rounding is a pure display transform of the same value —
    unit/ratio obligations are declined upstream, so only decimal_places is honored here.
    """
    if isinstance(value, bool) or value is None:
        return None
    if not isinstance(value, (int, float)):
        return None
    fv = float(value)
    if fv != fv or fv in (float("inf"), float("-inf")):  # NaN / inf ⇒ not a committable number
        return None
    if decimal_places is not None:
        return f"{fv:.{decimal_places}f}"
    if isinstance(value, float) and value.is_integer():
        return str(int(value))
    return str(value)


def _ledger_record(question: str, result: dict, answer: str) -> dict:
    """A SOT-2495 calc-ledger-shaped record built from one compute contract (for exec_verifier照合)."""
    ev = result.get("evidence") or {}
    method = result.get("method") or {}
    return {
        "question": question,
        "answer": answer,
        "typed_value": {
            "raw_text": answer,
            "parsed_value": result.get("value"),
            "unit": None,
            "scale": None,
            "source_range": ev.get("range"),
            "formula": method.get("code"),
        },
        "compute_steps": [{
            "code": method.get("code"),
            "columns_used": list(ev.get("columns_used") or []),
            "input_rows": ev.get("rows"),
            "output": result.get("value"),
        }],
        "contract": CONTRACT_TYPE,
    }


# --------------------------------------------------------------------------- pipeline
def pipeline(question: str, *, profile: Any = None) -> "dict[str, Any] | None":
    """Deterministic ``numeric`` answer as a ``{value, evidence, method}`` contract, or ``None``.

    ``None`` ⇒ the計算対象 could not be pinned deterministically, or the independent re-execution did not
    confirm the value ⇒ the router falls back to the LLM loop. Never raises into the answer path.
    """
    try:
        return _pipeline(question)
    except Exception:  # noqa: BLE001 — any failure falls back to the LLM loop, never breaks the answer path
        return None


def _pipeline(question: str) -> "dict[str, Any] | None":
    from src.rag.agent import exec_verifier as _ev
    from src.rag.agent import question_contract as _qc

    q = _nfc(question)
    if not q:
        return None

    # Precision-first negative guards: decline anything that is not a plain single-column aggregate.
    if any(cue.lower() in q.lower() if cue.isascii() else cue in q for cue in _EXCLUSION_CUES):
        return None
    req = _qc.numeric_requirements(question)
    if req.ratio or req.unit:  # 割合/比率 (population denominator) or a unit-bearing quantity ⇒ fallback
        return None

    aggregate = _detect_aggregate(q)
    if aggregate is None:
        return None

    table = _resolve_primary_table(question)
    if table is None or not table.get("rel"):
        return None
    rel, project = table["rel"], table.get("project")

    columns_result = _compute(rel, "df.columns.tolist()", project)
    if columns_result is None or not isinstance(columns_result.get("value"), list):
        return None
    columns = [str(c) for c in columns_result["value"]]

    column = _select_column(q, columns)
    if column is None:
        return None

    primary = _compute(rel, aggregate.primary(column), project)
    if primary is None:
        return None
    answer = _answer_string(primary.get("value"), req.decimal_places)
    if answer is None:
        return None

    rederived = _compute(rel, aggregate.rederive(column), project)
    if rederived is None:
        return None
    rederived_answer = _answer_string(rederived.get("value"), req.decimal_places)
    if rederived_answer is None:
        return None

    # 独立再計算との構造照合 (値/件数/対象列/単位 + SOT-2508 契約検証). Commit ONLY on a full match; any
    # conflict / undecidable / contract violation ⇒ None ⇒ LLM fallback (確証不能なら安全側).
    committed_record = _ledger_record(question, primary, answer)
    rederived_record = _ledger_record(question, rederived, rederived_answer)
    comparison = _ev.compare_execution(committed_record, rederived_record, correct=False)
    if comparison.category != _ev.EXEC_MATCH:
        return None

    pev = primary.get("evidence") or {}
    evidence = {
        "file": rel,
        "project": project,
        "aggregate": aggregate.name,
        "column": column,
        "sheet": pev.get("sheet"),
        "rows": pev.get("rows"),
        "columns_used": list(pev.get("columns_used") or []),
        "range": pev.get("range"),
        "route": "canonical_route→compute",
    }
    method = {
        "engine": "numeric_det_pipeline",
        "contract": CONTRACT_TYPE,
        "code": aggregate.primary(column),
        "rederive_code": aggregate.rederive(column),
        "exec_verify": comparison.category,
        "exec_reason": comparison.reason,
        "confidence": 1.0,
    }
    # ``answer`` (a string) already honors any 小数第N位 rounding; a plain aggregate stays numeric so
    # Stage3 renders it template-first (numeric is a _TEMPLATE_CONTRACTS type — no LLM naturalize call).
    value: Any = answer if req.decimal_places is not None else primary.get("value")
    return {"value": value, "evidence": evidence, "method": method}


def register(*, replace: bool = True) -> None:
    """Register this pipeline for the ``numeric`` contract (idempotent: ``replace=True`` by default)."""
    _det_pipeline.register(CONTRACT_TYPE, pipeline, replace=replace)


# Self-register on import — importing :mod:`src.rag.agent.pipelines` (the router's lazy bootstrap) wires
# this pipeline into the Stage0 registry. Idempotent so a re-import / test reload never raises.
register(replace=True)
