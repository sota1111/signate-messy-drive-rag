"""SOT-2501 — execution verifier for numeric answers (計算台帳の独立再実行照合).

Parent SOT-2460. The heterogeneous verifier (:mod:`~src.rag.agent.verifier`, SOT-2470) reconciles two
answers as *text* — value / enumeration / rounding / abstain. That catches many correlated errors, but a
whole class of high-confidence numeric mistakes is invisible to a text comparison because the two paths
happen to land on the *same wrong shape of number*: a correlation computed over the **wrong column pair**
(age×loan instead of the asked age×bmi — gold idx4/28), a histogram bucketed over the **wrong input set**
(idx10), or an over-inclusive selection. Two independently-run agents that both mis-identify the same
input set will *agree on a wrong number*, and text reconciliation will wave it through.

This module replaces「文章のもっともらしさ評価」with「集合と計算の再実行照合」for numeric answers. It takes
the committed answer's typed calculation証跡 — the SOT-2495 :mod:`~src.rag.agent.calc_ledger` record
(``typed_value`` + ``compute_steps``: which file/columns, how many input rows, the executed code, the
output) — and runs an **independent re-execution** whose errors are uncorrelated with the investigator's:

* **別プロファイル / 独立ツール経路** — a fresh :class:`~src.rag.tools.profile.CorpusProfile`, so no
  self-discovered secret or cached state leaks in from the investigator's run;
* **質問からの独立な入力集合の再特定** — the re-executor is *blind to the committed answer* and re-derives
  the input set (target file, columns, row filter) from the question alone, then recomputes;
* **構造照合 (not text)** — :func:`compare_execution` reconciles the two calculation証跡 on four axes drawn
  from the acceptance criteria: **値 (value)**, **件数 (input rows)**, **単位 (unit)** and — the axis a text
  comparison cannot see — **対象列 (columns_used)**. A column-set mismatch is the direct signature of a
  相関ペア取り違え; a row-count mismatch is the direct signature of an input-set inclusion/exclusion error.

A mismatch is reported as :data:`EXEC_CONFLICT` (``EVIDENCE_CONFLICT``) with the specific入力集合の差 diagnosed;
when the two calculations cannot be reconciled the verdict is *decision-unable*, and the precision-first
policy (Incorrect=−1 / Missing=0) is to **abstain** rather than commit an unverifiable number. A committed
numeric answer with **no** compute step (暗算 — :func:`~src.rag.agent.calc_ledger.is_grounded` is False) is
:data:`EXEC_UNGROUNDED`: there is nothing to replay, so it too cannot be confirmed.

Design mirrors the investigator/verifier/tiebreak stack: the reconciliation core
(:func:`compare_execution`) is a **pure** function over two ledger-shaped dicts (no Vertex, trivially
testable), and the driver :func:`verify_execution` takes an injectable ``rederive`` callable so offline
tests script the independent re-derivation; :func:`rederive_calc` / :func:`verify_question` wire the live
Gemini re-execution. This module is purely advisory — it returns a verdict; the commit/abstain call stays
in :mod:`~src.rag.agent.gate`, and there it is **default-OFF** until real-LB / 関門2 confirmation.
"""
from __future__ import annotations

import json
import re
import tempfile
import unicodedata
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

from config import settings
from src.rag.agent.calc_ledger import is_grounded, is_numeric_answer
from src.rag.agent.question_contract import validate_numeric_answer
from src.rag.agent.investigator import (
    DEFAULT_MAX_TURNS,
    DEFAULT_TIMEOUT_S,
    AgentTool,
    Model,
    build_tools,
    investigate,
    is_abstain,
)
from src.rag.agent.verifier import CATEGORY_ABSTAIN, compare_answers
from src.rag.tools.profile import CorpusProfile

ABSTAIN = settings.ABSTAIN

# The re-executor deliberately re-derives the input set from scratch; the *hard* tier is used because the
# whole point is a rigorous independent re-computation (columns / rows / value), not a cheap sanity ping.
DEFAULT_EXEC_MODEL = settings.GEN_MODEL_HARD  # gemini-2.5-pro (fresh profile ⇒ still uncorrelated)

EXEC_VERIFIER_SYSTEM_PROMPT = (
    "あなたは計算回答の独立実行検証エージェントです。別の調査エージェントが計算由来の答えを出していますが、"
    "その回答・その計算過程は一切与えられません。あなたは質問だけを手掛かりに、**入力集合(対象ファイル・"
    "対象列・対象行の絞り込み)を一から自力で再特定**し、計算を**再実行**して数値を再導出します。目的は、"
    "値そのものだけでなく『どの集合をどの列で数えた/計算したか』を独立に確かめ、相関ペアの取り違えや"
    "対象行の包含/除外の誤りを炙り出すことです。\n"
    "厳守事項:\n"
    "1. 暗算・記憶・創作で答えない。必ず compute 等のツールで根拠となる値を再計算してから答える。\n"
    "2. パスワード・略称・書式規則などのコーパス固有事実は与えられていない。ツール(ファイル探索/grep/"
    "Office抽出/復号/pandas計算/チャート読取/画像説明)で自力発見する。\n"
    "3. 数値計算(平均・合計・件数・相関など)は必ず compute ツールで行い、列名や絞り込み値が不明なときは"
    "`df.columns.tolist()` や `df['列'].unique().tolist()` で先に確認し、**質問が指す対象列を取り違えない**"
    "よう厳密に対応付けてから式を組む。\n"
    "4. 証拠の優先順位は『computeによる生データ再計算・notebookの実行出力 > markdown/散文の主張』とする。"
    "矛盾時は canonical_route で元データへ直行し、compute の再計算値だけを採用する。notebook質問では"
    " canonical_route(question=..., kind='train', expr='...') を使い、矛盾した散文を evidence に含めない。\n"
    "5. 十分な根拠が得られたら、最終回答は必ず submit_answer ツールを1回だけ呼んで返す。answer=回答本文"
    "(計算値または計算で選ばれた列名、金額は原文表記)、confidence=0.0〜1.0、evidence=根拠、"
    "method=導出手順(使った列・行数を明記)。\n"
    f"6. あらゆる手段を尽くしても根拠が該当しない/再導出できない場合に限り answer=「{ABSTAIN}」・"
    "confidence=0.0 で submit_answer する(存在しない値を捏造しない)。"
)

# --------------------------------------------------------------------------- verdict categories
EXEC_MATCH = "exec_match"            # 独立再実行が値・件数・単位・対象列すべてで一致 → 数値を確認
EXEC_CONFLICT = "evidence_conflict"  # 入力集合/計算の相違を検出(値/件数/単位/対象列) → 誤commit阻止
EXEC_UNGROUNDED = "exec_ungrounded"  # 台帳に compute 証跡が無い(暗算) → 再実行対象が存在せず確認不能
EXEC_UNDECIDABLE = "exec_undecidable"  # 独立再実行が値を再導出できず(棄権/失敗) → 決着不能
# SOT-2547 — 大外し(桁/符号)を、同一対象を独立に再計算した確定値で矯正する(棄権化でなく置換).
EXEC_CORRECTED = "exec_corrected"    # 台帳が大外し・独立再実行が同一対象で確定値 → 再計算値を採用(match化)

# The conflict/ungrounded/undecidable outcomes are the ones the precision-first gate must not commit on.
_ABSTAIN_CATEGORIES = frozenset({EXEC_CONFLICT, EXEC_UNGROUNDED, EXEC_UNDECIDABLE})


# --------------------------------------------------------------------------- gross-miss thresholds (SOT-2547)
def _float_env(key: str, default: float) -> float:
    import os

    try:
        return float(os.getenv(key, str(default)))
    except (TypeError, ValueError):
        return default


# Order-of-magnitude divergence between committed and re-derived value that counts as a 大外し (idx97 型:
# 18948 vs 272 ≈ 70x). Only such a *gross* miss — never a small/rounding-scale difference — is confident
# enough to replace the committed number with the independent re-derivation.
GROSS_MISS_RATIO = _float_env("EXEC_GROSS_MISS_RATIO", 10.0)
# Minimum self-reported confidence of the independent re-execution before its value may *replace* the
# committed answer. A low-confidence re-derivation cannot correct — it can only refuse (→ 棄権).
GROSS_MISS_MIN_CONFIDENCE = _float_env("EXEC_CORRECT_MIN_CONFIDENCE", 0.7)

_GROSS_NUM_RE = re.compile(r"[+-]?\d{1,3}(?:,\d{3})+(?:\.\d+)?|[+-]?\d+(?:\.\d+)?")


def _leading_number(answer: str) -> float | None:
    """Parse the first signed numeric literal in an answer (commas stripped), or ``None``."""
    m = _GROSS_NUM_RE.search(unicodedata.normalize("NFKC", str(answer or "")))
    if not m:
        return None
    try:
        return float(m.group(0).replace(",", ""))
    except ValueError:
        return None


def _confidence_of(record: Mapping[str, Any] | None) -> float:
    if not isinstance(record, Mapping):
        return 0.0
    try:
        return float(record.get("confidence", 0.0) or 0.0)
    except (TypeError, ValueError):
        return 0.0


def _gross_miss_kind(committed_answer: str, rederived_answer: str) -> str | None:
    """Classify a *gross* numeric divergence between two answers, or ``None`` for a small/near miss.

    Returns ``"符号"`` when the two values carry opposite signs (idx63 型: 回帰予測 −30.78 vs +0.15) or
    ``"桁"`` when their magnitudes differ by ≥ :data:`GROSS_MISS_RATIO` (idx97 型: 差の別計算 18948 vs 272).
    A rounding-scale or otherwise modest difference returns ``None`` — those stay ambiguous (→ 棄権),
    never corrected. Non-numeric / unparseable answers also return ``None`` (the target/definition axis,
    not value magnitude, governs those)."""
    a, b = _leading_number(committed_answer), _leading_number(rederived_answer)
    if a is None or b is None:
        return None
    # 符号: opposite-signed values (treating 0 as sign-agnostic) with a real gap.
    if a * b < 0 and abs(a - b) > 1e-9:
        return "符号(正負反転)"
    hi, lo = max(abs(a), abs(b)), min(abs(a), abs(b))
    if lo <= 1e-12:
        # one side is (near-)zero: a gross miss only if the other side is not itself ~0.
        return "桁(一方が0近傍)" if hi > 1e-6 else None
    if hi / lo >= GROSS_MISS_RATIO:
        return f"桁(比率≈{hi / lo:.0f}倍)"
    return None


# --------------------------------------------------------------------------- ledger-record accessors
def _nfkc_lower(text: Any) -> str:
    return unicodedata.normalize("NFKC", str(text)).strip().lower()


def _answer_of(record: Mapping[str, Any]) -> str:
    return str(record.get("answer", ""))


def _compute_steps(record: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    steps = record.get("compute_steps") or ()
    return [s for s in steps if isinstance(s, Mapping)]


def _columns_used(record: Mapping[str, Any]) -> set[str]:
    """Union of the (normalized) columns every compute step in the record touched — the 対象列 axis."""
    cols: set[str] = set()
    for step in _compute_steps(record):
        for c in step.get("columns_used") or ():
            norm = _nfkc_lower(c)
            if norm:
                cols.add(norm)
    return cols


def _input_rows(record: Mapping[str, Any]) -> int | None:
    """Rows the last compute step saw — the 件数/入力集合 size axis (aligns with typed_value's source)."""
    steps = _compute_steps(record)
    for step in reversed(steps):
        rows = step.get("input_rows")
        if isinstance(rows, (int, float)) and not isinstance(rows, bool):
            return int(rows)
    return None


def _unit_of(record: Mapping[str, Any]) -> str | None:
    tv = record.get("typed_value")
    if isinstance(tv, Mapping):
        u = tv.get("unit")
        return _nfkc_lower(u) if u else None
    return None


# --------------------------------------------------------------------------- comparison result type
@dataclass(frozen=True)
class ExecComparison:
    """The reconciliation of a committed numeric calculation証跡 with an independent re-execution."""

    category: str            # EXEC_* — which axis decided the outcome
    match: bool              # True iff the re-execution confirms the committed number
    reason: str              # human-readable 判定根拠 (auditable)
    conflicts: tuple[str, ...] = ()  # per-axis入力集合の差 (empty on a match)
    corrected_answer: str = ""       # SOT-2547 — the re-derived value to adopt on EXEC_CORRECTED

    @property
    def should_abstain(self) -> bool:
        """Precision-first: a conflict / ungrounded / undecidable verdict must not commit (−1 avoidance)."""
        return self.category in _ABSTAIN_CATEGORIES

    @property
    def should_correct(self) -> bool:
        """SOT-2547 — the committed number was a大外し and a confident同一対象 re-derivation replaces it."""
        return self.category == EXEC_CORRECTED and bool(self.corrected_answer)

    def to_dict(self) -> dict[str, Any]:
        return {
            "category": self.category,
            "match": self.match,
            "reason": self.reason,
            "conflicts": list(self.conflicts),
            "corrected_answer": self.corrected_answer,
        }


def _columns_compatible(c_cols: set[str], r_cols: set[str]) -> bool:
    """True when committed and re-derivation computed the *same* target (対象列一致 / 一方が未記録).

    A gross-miss may only be *corrected* (re-derivation adopted) when both sides addressed the same
    target — same columns, or one side did not record its columns. When the target columns genuinely
    diverge the disagreement is a対象取り違え whose correct side is ambiguous, so it must 棄権 (idx47/70
    型), never silently adopt one target over the other."""
    return not c_cols or not r_cols or c_cols == r_cols


def compare_execution(committed: Mapping[str, Any], rederived: Mapping[str, Any] | None, *,
                      correct: bool = False) -> ExecComparison:
    """Reconcile a committed numeric ledger record with an independent re-execution's ledger record.

    ``committed`` and ``rederived`` are both SOT-2495 calc-ledger-shaped dicts (``answer`` / ``typed_value``
    / ``compute_steps``). The precedence, all drawn from the acceptance criteria:

    1. **暗算 (:data:`EXEC_UNGROUNDED`)** — the committed number has no compute step, so there is nothing to
       replay: it cannot be confirmed.
    2. **決着不能 (:data:`EXEC_UNDECIDABLE`)** — the independent re-execution abstained or produced no compute
       step, so it cannot confirm or refute.
    3. **対象列 → 件数 → 値 → 単位 (:data:`EXEC_CONFLICT`)** — any structural mismatch. Column-set difference is
       reported first (直接的な相関ペア取り違えの証拠), then input-row difference (入力集合の包含/除外),
       then the value itself (via the tested :func:`~src.rag.agent.verifier.compare_answers` rounding axis),
       then the unit. Every detected axis is collected into ``conflicts`` for the audit line.
    4. **一致 (:data:`EXEC_MATCH`)** — none of the above: the re-execution confirms the number.

    **大外し矯正 (:data:`EXEC_CORRECTED`, SOT-2547).** When ``correct`` is True and the deciding conflict is a
    *value* mismatch that is a大外し (:func:`_gross_miss_kind` — 符号反転 idx63 型 / 桁違い idx97 型) while
    the two addressed the **same target** (:func:`_columns_compatible`) and the independent re-derivation is
    grounded, contract-clean and confident, the committed number is *replaced* by the re-derivation instead
    of abstaining — a deterministic-backed matching correction. Default ``correct=False`` keeps the pure
    EXEC_CONFLICT→棄権 behavior byte-identical (対象取り違え idx47/70 型 stays 棄権, 確証不能につき安全側)."""
    # 1) 暗算: a committed number with no calculation証跡 has nothing to re-run.
    if not is_grounded(committed):
        return ExecComparison(
            EXEC_UNGROUNDED, False,
            "台帳に compute 証跡が無い暗算数値 — 独立再実行で照合できず確認不能",
        )

    # SOT-2508 — verify the *requested quantity* before comparing two numeric values.  Two executions
    # can agree while both divide by the wrong population; the deterministic question contract catches
    # that correlated error from ``XのうちYの割合`` wording, and also enforces the requested unit/rounding.
    committed_contract = validate_numeric_answer(
        str(committed.get("question", "")), _answer_of(committed), _compute_steps(committed))
    if not committed_contract.passed:
        conflicts = tuple(committed_contract.issues)
        return ExecComparison(
            EXEC_CONFLICT, False,
            "計算台帳が質問の量の定義/単位/丸め契約に違反(EVIDENCE_CONFLICT) — "
            + " / ".join(conflicts),
            conflicts=conflicts,
        )

    # 2) 決着不能: the independent re-execution could not produce a comparable number.
    if rederived is None or is_abstain(_answer_of(rederived)) or not is_grounded(rederived):
        detail = "再実行が得られず" if rederived is None else (
            "再実行が棄権" if is_abstain(_answer_of(rederived)) else "再実行に compute 証跡が無く")
        return ExecComparison(
            EXEC_UNDECIDABLE, False,
            f"独立再実行が値を再導出できず({detail}) — 決着不能",
        )

    rederived_contract = validate_numeric_answer(
        str(committed.get("question", "")), _answer_of(rederived), _compute_steps(rederived))
    if not rederived_contract.passed:
        conflicts = tuple("再実行: " + issue for issue in rederived_contract.issues)
        return ExecComparison(
            EXEC_CONFLICT, False,
            "独立再実行が質問の量の定義/単位/丸め契約に違反(EVIDENCE_CONFLICT) — "
            + " / ".join(conflicts),
            conflicts=conflicts,
        )

    conflicts: list[str] = []
    primary: str | None = None
    value_gross_miss: str | None = None

    # 3a) 対象列 (columns_used): the direct signature of a 相関ペア取り違え / wrong-column derivation.
    c_cols, r_cols = _columns_used(committed), _columns_used(rederived)
    if c_cols and r_cols and c_cols != r_cols:
        extra = sorted(c_cols - r_cols)    # columns the committed calc used but the re-execution did not
        missing = sorted(r_cols - c_cols)  # columns the re-execution used but the committed calc did not
        parts = []
        if extra:
            parts.append(f"台帳のみ: {', '.join(extra)}")
        if missing:
            parts.append(f"再実行のみ: {', '.join(missing)}")
        conflicts.append("対象列の相違(相関ペア/対象列取り違え) — " + " / ".join(parts))
        primary = primary or EXEC_CONFLICT

    # 3b) 件数 (input rows): the direct signature of an input-set inclusion/exclusion difference.
    c_rows, r_rows = _input_rows(committed), _input_rows(rederived)
    if c_rows is not None and r_rows is not None and c_rows != r_rows:
        conflicts.append(f"入力集合の差(行の包含/除外): 台帳={c_rows}行 vs 再実行={r_rows}行")
        primary = primary or EXEC_CONFLICT

    # 3c) 値 (value): reuse the tested numeric-rounding / value reconciliation from the verifier.
    value_cmp = compare_answers(_answer_of(committed), _answer_of(rederived))
    if not value_cmp.agree and value_cmp.category != CATEGORY_ABSTAIN:
        conflicts.append(f"値の相違: {value_cmp.reason}")
        primary = primary or EXEC_CONFLICT
        value_gross_miss = _gross_miss_kind(_answer_of(committed), _answer_of(rederived))

    # 3d) 単位 (unit): a value that matches numerically but differs in unit is still a mismatch.
    c_unit, r_unit = _unit_of(committed), _unit_of(rederived)
    if c_unit and r_unit and c_unit != r_unit:
        conflicts.append(f"単位の相違: 台帳={c_unit} vs 再実行={r_unit}")
        primary = primary or EXEC_CONFLICT

    if primary == EXEC_CONFLICT:
        # 3e) 大外し矯正 (SOT-2547): a value 大外し over the *same target* with a confident, grounded,
        # contract-clean independent re-derivation is corrected — the re-derivation replaces the committed
        # number (idx63 符号 / idx97 桁). 対象列が食い違う取り違え(idx47/70)は same-target 条件で除外され
        # 従来どおり棄権(確証不能につき安全側)。value 以外の食い違い(単位/丸めのみ)も矯正しない。
        rederived_answer = _answer_of(rederived)
        if (correct and value_gross_miss and _columns_compatible(c_cols, r_cols)
                and _confidence_of(rederived) >= GROSS_MISS_MIN_CONFIDENCE
                and not is_abstain(rederived_answer)):
            return ExecComparison(
                EXEC_CORRECTED, True,
                f"台帳が大外し[{value_gross_miss}]・同一対象の独立再実行が確定値 → 再計算値を採用"
                f"(台帳『{_answer_of(committed)}』→ 採用『{rederived_answer}』): " + " / ".join(conflicts),
                conflicts=tuple(conflicts),
                corrected_answer=rederived_answer,
            )
        return ExecComparison(
            EXEC_CONFLICT, False,
            "独立再実行と不一致(EVIDENCE_CONFLICT) — " + " / ".join(conflicts),
            conflicts=tuple(conflicts),
        )

    # 4) 一致: value, rows, columns and unit all reconcile.
    return ExecComparison(
        EXEC_MATCH, True,
        f"独立再実行が数値を確認(値/件数/単位/対象列 一致): {value_cmp.reason}",
    )


# --------------------------------------------------------------------------- verdict type
@dataclass(frozen=True)
class ExecVerdict:
    """An execution-verifier verdict for one committed numeric answer."""

    question: str
    category: str
    match: bool
    should_abstain: bool
    reason: str
    committed_answer: str
    rederived_answer: str
    conflicts: tuple[str, ...] = field(default_factory=tuple)
    corrected_answer: str = ""       # SOT-2547 — value to adopt when the verdict is EXEC_CORRECTED

    @property
    def should_correct(self) -> bool:
        """SOT-2547 — the committed 大外し should be replaced by :attr:`corrected_answer`."""
        return self.category == EXEC_CORRECTED and bool(self.corrected_answer)

    def to_dict(self) -> dict[str, Any]:
        return {
            "question": self.question,
            "category": self.category,
            "match": self.match,
            "should_abstain": self.should_abstain,
            "should_correct": self.should_correct,
            "reason": self.reason,
            "committed_answer": self.committed_answer,
            "rederived_answer": self.rederived_answer,
            "corrected_answer": self.corrected_answer,
            "conflicts": list(self.conflicts),
        }


def _build_verdict(committed: Mapping[str, Any], rederived: Mapping[str, Any] | None,
                   cmp: ExecComparison) -> ExecVerdict:
    return ExecVerdict(
        question=str(committed.get("question", "")),
        category=cmp.category,
        match=cmp.match,
        should_abstain=cmp.should_abstain,
        reason=cmp.reason,
        committed_answer=_answer_of(committed),
        rederived_answer=("" if rederived is None else _answer_of(rederived)),
        conflicts=cmp.conflicts,
        corrected_answer=cmp.corrected_answer,
    )


# --------------------------------------------------------------------------- driver (pure over rederive)
def verify_execution(committed: Mapping[str, Any], *,
                     rederive: Callable[[str], Mapping[str, Any] | None],
                     correct: bool = False) -> ExecVerdict:
    """Independently re-execute the committed numeric answer's calculation and reconcile the two.

    ``committed`` is a SOT-2495 calc-ledger record. ``rederive`` maps the *question* to an independently
    re-executed calc-ledger record (or ``None`` when the re-execution could not run) — inject a scripted
    record in offline tests, or :func:`rederive_calc` for the live heterogeneous Gemini re-execution. The
    re-executor never sees the committed answer, so its errors stay uncorrelated. When ``correct`` is True
    a同一対象 大外し is replaced by the re-derivation (:data:`EXEC_CORRECTED`, SOT-2547) instead of abstaining.
    """
    question = str(committed.get("question", ""))
    rederived = rederive(question)
    cmp = compare_execution(committed, rederived, correct=correct)
    return _build_verdict(committed, rederived, cmp)


# --------------------------------------------------------------------------- live re-execution
def _record_shell(question: str, answer: str) -> dict[str, Any]:
    """A minimal ledger-shaped dict for a re-execution that produced no calc record (棄権/暗算)."""
    return {"question": question, "answer": answer, "typed_value": {}, "compute_steps": [],
            "contract": None}


def rederive_calc(question: str, *,
                  model: Model | None = None,
                  model_factory: Callable[[str, Sequence[AgentTool]], Model] | None = None,
                  profile_factory: Callable[[], CorpusProfile] | None = None,
                  max_turns: int = DEFAULT_MAX_TURNS,
                  timeout_s: float = DEFAULT_TIMEOUT_S) -> dict[str, Any]:
    """Run one independent re-execution of ``question`` and return its calc-ledger-shaped record.

    Reuses the fully-tested SOT-2495 calc-ledger wiring: the investigation is run with a fresh profile /
    own tools (tool-path independence) and its numeric/derived-calculation commit is recorded to a
    throwaway JSONL, which we read back as the re-derived record. A re-execution that abstains or produces
    no calculation record yields a shell whose ``answer`` drives the :data:`EXEC_UNDECIDABLE` branch.
    """
    prof = (profile_factory or CorpusProfile)()
    tools = build_tools(prof)
    if model is None:
        factory = model_factory or exec_model_factory
        model = factory(question, tools)
    # The contract is part of the calculation semantics.  In particular, a correlation argmax returns
    # a column label (``bmi``), not a numeric literal, but still needs a replayable calc record.
    try:
        from src.rag.agent import question_contract as qc

        contract = qc.classify(question).contract
    except Exception:  # noqa: BLE001 — classification is advisory; preserve the old numeric path
        contract = None
    with tempfile.TemporaryDirectory() as td:
        path = Path(td) / "rederive.jsonl"
        investigation = investigate(model, question, tools, max_turns=max_turns,
                                    timeout_s=timeout_s, calc_ledger=path, contract=contract)
        answer = investigation.answer.answer
        if path.exists():
            lines = [ln for ln in path.read_text(encoding="utf-8").splitlines() if ln.strip()]
            if lines:
                return json.loads(lines[-1])
    return _record_shell(question, answer)


def exec_model_factory(question: str, tools: Sequence[AgentTool], *,
                       model: str | None = None) -> Model:
    """Fresh live Gemini conversation for the re-executor: the *hard* tier + the exec-verifier prompt."""
    from src.rag.agent.investigator import GeminiModel, to_genai_tools

    return GeminiModel(question, to_genai_tools(tools),
                       model=model or DEFAULT_EXEC_MODEL, system=EXEC_VERIFIER_SYSTEM_PROMPT)


def verify_question(committed: Mapping[str, Any], *, model: str | None = None,
                    profile: CorpusProfile | None = None,
                    max_turns: int = DEFAULT_MAX_TURNS,
                    timeout_s: float = DEFAULT_TIMEOUT_S,
                    correct: bool = False) -> ExecVerdict:
    """Convenience live entry point: independently re-execute a committed numeric record's calculation.

    Only meaningful for a numeric, grounded committed record; callers gate on :func:`is_numeric_answer`.
    When ``correct`` is True a同一対象 大外し is corrected to the re-derivation (SOT-2547) instead of 棄権.
    """
    def _rederive(q: str) -> Mapping[str, Any]:
        return rederive_calc(
            q, model_factory=lambda qq, t: exec_model_factory(qq, t, model=model),
            profile_factory=(lambda: profile) if profile is not None else None,
            max_turns=max_turns, timeout_s=timeout_s)

    return verify_execution(committed, rederive=_rederive, correct=correct)


def is_verifiable_record(record: Mapping[str, Any] | None) -> bool:
    """True for a literal number or a value selected by the numeric/derived-calculation contract."""
    return record is not None and (
        is_numeric_answer(_answer_of(record)) or record.get("contract") == "numeric")


def is_numeric_record(record: Mapping[str, Any] | None) -> bool:
    """Backward-compatible name for :func:`is_verifiable_record` (SOT-2501 public API)."""
    return is_verifiable_record(record)
