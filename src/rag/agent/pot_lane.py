"""SOT-2586 — NUMERIC route の PoT 強制ハードレーン(Evidence Binder→制限AST→Decimal実行→独立検算).

親 SOT-2568 ディープリサーチコメント(実装順 4/7, P1)。最弱型 ``derived_calculation`` (gold100 で
32問中 match 6・wrong 3、idx76 の 79,200→73,260 誤算に代表される) を、**operand binding / formula /
execution の三層分離**で立て直す。狙いは「operand が届かない(A)」と「計算を間違える(B)」を切り分けて
両方に対策すること — その切り分けが可能なように、各層を*個別に*判定できる形にする。

パイプライン(SOT-2584 の typed router / Evidence Packet の上に載る NUMERIC route の primary lane):

    Evidence Binder      operand を value / unit / source(doc:sheet!cell) 付きで束縛。
                         ``operand_sources_complete`` を evidence slot として要求する。
    Condition Interpreter what-if / 条件文を IR 化(predicate / predicate_truth / base_quantity /
                         adjustments[kind, rate, order])。条件分岐の選択ミス(idx76 型)を明示検証可能に。
    Formula Builder      自由 Python 生成を廃止し、**許可演算限定 AST/DSL** のみ受け付ける
                         (ADD/SUB/MUL/DIV/SUM/MEAN/RATIO/PERCENT_CHANGE/WEIGHTED_SUM/ROUND/MIN/MAX)。
    Hard Executor        :mod:`decimal` の高精度 exact arithmetic で Bound AST を評価する。
    Independent Verifier 同じ Bound AST を(SymPy があれば)``sympy.Rational`` に、無ければ
                         :class:`fractions.Fraction` に翻訳して**独立に**再計算し、実行と突き合わせる。

設計不変条件(姉妹 opt-in — evidence_packet / document_registry — と同じ):

* **純粋・決定論・ネットワーク非依存.** すべて構造化入力(束縛済み operand + 演算限定 spec)上の全域写像。
  LLM 呼び出しも I/O も無いのでオフラインで再現・計測できる。
* **eval 系を通さない.** LLM が出す式は制限 DSL の *構造(dict)* としてのみ受け取り、検証済み Bound AST を
  経由して評価する。文字列を ``eval`` / ``sympy.parse_expr`` へ直接渡すことは一切しない
  (security: 任意コード実行経路を作らない)。SymPy 不在環境では Fraction fallback で degrade する。
* **回答を注入しない.** レーンは operand の value/unit/source と演算だけを扱い、値を推測しない。
"""
from __future__ import annotations

import math
import os
from dataclasses import dataclass
from decimal import Context, Decimal, InvalidOperation, localcontext
from fractions import Fraction
from typing import Any, Mapping, Sequence

from src.rag.agent import failure_taxonomy as _tax

_ON = {"1", "true", "yes", "on"}

# High, fixed precision for the Decimal hard executor, applied via a *local* context only (never the
# process-global one) so importing this module — even when the lane is OFF — cannot perturb any other
# code's Decimal arithmetic. Division of large 見込金額 stays exact enough here that the independent
# Fraction verifier can confirm equality after the question's own ROUND.
_EXEC_CONTEXT = Context(prec=50)


def enabled() -> bool:
    """True when the NUMERIC route should dispatch through the PoT forced lane (default OFF — opt-in).

    Gated by ``RAG_POT_HARD_LANE`` exactly like the sibling opt-ins (RAG_EVIDENCE_PACKET / RAG_FONT_EMPHASIS
    / RAG_FORMAT_EVENTS): default-OFF means the serve path exposes the identical tool set and injects the
    identical prompt, so the champion answer path is byte-identical.
    """
    return os.getenv("RAG_POT_HARD_LANE", "0").strip().lower() in _ON


# --------------------------------------------------------------------------- optional SymPy backend
try:  # pragma: no cover - import guard
    import sympy as _sympy  # type: ignore

    _HAVE_SYMPY = True
except Exception:  # pragma: no cover - sympy is optional; Fraction is the always-available backbone
    _sympy = None  # type: ignore
    _HAVE_SYMPY = False


def have_sympy() -> bool:
    """Whether the SymPy Rational cross-check backend is importable (else Fraction-only)."""
    return _HAVE_SYMPY


# --------------------------------------------------------------------------- allowed operations (DSL)
# The ONLY operations the restricted formula DSL admits (research §Formula Builder). Any op outside this
# set makes the formula layer fail — the model cannot smuggle free Python through the lane.
ADD, SUB, MUL, DIV = "ADD", "SUB", "MUL", "DIV"
SUM, MEAN = "SUM", "MEAN"
RATIO, PERCENT_CHANGE, WEIGHTED_SUM = "RATIO", "PERCENT_CHANGE", "WEIGHTED_SUM"
ROUND, MIN, MAX = "ROUND", "MIN", "MAX"

ALLOWED_OPS: frozenset[str] = frozenset({
    ADD, SUB, MUL, DIV, SUM, MEAN, RATIO, PERCENT_CHANGE, WEIGHTED_SUM, ROUND, MIN, MAX,
})

# Fixed arity (None = variadic, ≥1 arg). ROUND takes 1 value arg + an ``ndigits`` attribute.
_ARITY: dict[str, int | None] = {
    ADD: 2, SUB: 2, MUL: 2, DIV: 2,
    RATIO: 2, PERCENT_CHANGE: 2,
    SUM: None, MEAN: None, MIN: None, MAX: None, WEIGHTED_SUM: None,
    ROUND: 1,
}


class FormulaError(ValueError):
    """Raised when a formula spec violates the restricted DSL (unknown op, bad arity, unbound ref)."""


# --------------------------------------------------------------------------- layer 1: Evidence Binder
@dataclass(frozen=True)
class BoundOperand:
    """One operand bound with its value, unit and source — the atom the binder produces.

    ``source`` is the provenance string the research asks for (``doc:sheet!cell`` / ``file:page`` …).
    ``sourced`` (source non-empty) is what ``operand_sources_complete`` aggregates: a value with no
    source is an ungrounded 暗算 operand and must not silently satisfy the operand layer.
    """

    name: str
    value: Fraction
    unit: str = ""
    source: str = ""

    @property
    def sourced(self) -> bool:
        return bool(self.source.strip())

    @property
    def has_unit(self) -> bool:
        return bool(self.unit.strip())

    def to_dict(self) -> dict[str, Any]:
        return {"name": self.name, "value": _fraction_to_jsonable(self.value),
                "unit": self.unit, "source": self.source}


def _to_fraction(value: Any) -> Fraction:
    """Exact-parse a numeric operand into a :class:`Fraction` (rejects NaN/inf and junk).

    Accepts int / Fraction / Decimal / a numeric string (``"73,260"`` / ``"1,100,000円"`` tolerated) and
    float (via its exact Decimal expansion is *not* used — a float literal like ``0.1`` is parsed through
    ``str`` so ``Fraction("0.1") == 1/10``, matching how a human reads the sheet value)."""
    if isinstance(value, Fraction):
        return value
    if isinstance(value, bool):  # bool is an int subclass — never a numeric operand
        raise FormulaError(f"operand value must be numeric, got bool {value!r}")
    if isinstance(value, int):
        return Fraction(value)
    if isinstance(value, Decimal):
        if not value.is_finite():
            raise FormulaError(f"operand value is not finite: {value!r}")
        return Fraction(value)
    if isinstance(value, float):
        if not math.isfinite(value):
            raise FormulaError(f"operand value is not finite: {value!r}")
        return Fraction(str(value))
    if isinstance(value, str):
        cleaned = value.strip().replace(",", "").replace("，", "")
        for suffix in ("円", "%", "％", "人", "件", "時間", "個", "社"):
            if cleaned.endswith(suffix):
                cleaned = cleaned[: -len(suffix)]
        cleaned = cleaned.strip()
        if not cleaned:
            raise FormulaError(f"empty numeric operand: {value!r}")
        try:
            return Fraction(cleaned)
        except (ValueError, ZeroDivisionError):
            try:
                return Fraction(Decimal(cleaned))
            except (InvalidOperation, ValueError) as exc:
                raise FormulaError(f"not a number: {value!r}") from exc
    raise FormulaError(f"unsupported operand value type: {type(value).__name__}")


def bind_operand(name: str, value: Any, *, unit: str = "", source: str = "") -> BoundOperand:
    """Bind one operand (value exact-parsed to :class:`Fraction`) — the binder's per-operand entry."""
    if not str(name).strip():
        raise FormulaError("operand name must be non-empty")
    return BoundOperand(name=str(name), value=_to_fraction(value), unit=str(unit or ""),
                        source=str(source or ""))


@dataclass(frozen=True)
class Binding:
    """The full operand binding for one candidate — a name→:class:`BoundOperand` map with completeness."""

    operands: tuple[BoundOperand, ...]

    @property
    def by_name(self) -> dict[str, BoundOperand]:
        return {o.name: o for o in self.operands}

    @property
    def operand_sources_complete(self) -> bool:
        """True iff every bound operand carries a non-empty source (the operand-layer evidence slot)."""
        return bool(self.operands) and all(o.sourced for o in self.operands)

    @property
    def units_complete(self) -> bool:
        return bool(self.operands) and all(o.has_unit for o in self.operands)

    def source_signature(self) -> tuple[str, ...]:
        """The sorted operand sources — the ``operand source`` axis of N-sample agreement."""
        return tuple(sorted(o.source for o in self.operands))

    def to_dict(self) -> dict[str, Any]:
        return {"operands": [o.to_dict() for o in self.operands],
                "operand_sources_complete": self.operand_sources_complete,
                "units_complete": self.units_complete}


def build_binding(operands: Sequence[Mapping[str, Any] | BoundOperand]) -> Binding:
    """Build a :class:`Binding` from raw operand dicts ``{name,value,unit?,source?}`` (or BoundOperands)."""
    bound: list[BoundOperand] = []
    seen: set[str] = set()
    for raw in operands:
        if isinstance(raw, BoundOperand):
            op = raw
        else:
            op = bind_operand(raw.get("name", ""), raw.get("value"),
                              unit=raw.get("unit", ""), source=raw.get("source", ""))
        if op.name in seen:
            raise FormulaError(f"duplicate operand name: {op.name!r}")
        seen.add(op.name)
        bound.append(op)
    return Binding(operands=tuple(bound))


# --------------------------------------------------------------------------- layer 2: Condition IR
@dataclass(frozen=True)
class Adjustment:
    """One ordered adjustment applied to a base quantity under a branch (kind/rate/order)."""

    kind: str            # e.g. "discount" / "surcharge" / "tax" — free label, recorded for audit
    rate: Fraction       # the multiplicative/additive rate (e.g. 1/10 for 10%, exact)
    order: int = 0       # application order (ascending); ties keep declaration order

    def to_dict(self) -> dict[str, Any]:
        return {"kind": self.kind, "rate": _fraction_to_jsonable(self.rate), "order": self.order}


@dataclass(frozen=True)
class ConditionIR:
    """A what-if / conditional question's interpretation, made explicitly checkable (idx76 型対策).

    A conditional numeric question selects *which* base quantity / adjustment set applies from the truth
    of a predicate ("税抜なら〜/税込なら〜", "3分の2に減額した場合"). The router/formula builder emit that
    choice as this IR so the *branch interpretation* is a first-class, comparable field of the answer —
    a wrong branch (idx76: applied the 減額 to the wrong base) is caught by the formula layer and shows up
    as a branch-signature disagreement in N-sample majority, instead of hiding inside a single number.
    """

    predicate: str = ""                 # human-readable condition, verbatim from the question
    predicate_truth: bool | None = None  # the branch actually taken (None = unconditional)
    base_quantity: str = ""             # operand name chosen as the base under this branch
    adjustments: tuple[Adjustment, ...] = ()

    @property
    def is_conditional(self) -> bool:
        return bool(self.predicate) or self.predicate_truth is not None or bool(self.adjustments)

    def branch_signature(self) -> str:
        """A stable string identifying the chosen branch — the ``branch interpretation`` agreement axis."""
        adj = ";".join(f"{a.kind}={a.rate}@{a.order}" for a in sorted(self.adjustments, key=_adj_key))
        return f"pred={self.predicate_truth}|base={self.base_quantity}|adj=[{adj}]"

    def to_dict(self) -> dict[str, Any]:
        return {"predicate": self.predicate, "predicate_truth": self.predicate_truth,
                "base_quantity": self.base_quantity,
                "adjustments": [a.to_dict() for a in self.adjustments],
                "is_conditional": self.is_conditional,
                "branch_signature": self.branch_signature()}


def _adj_key(a: Adjustment) -> tuple[int, str]:
    return (a.order, a.kind)


def build_condition(spec: Mapping[str, Any] | None) -> ConditionIR:
    """Interpret a raw condition spec into a validated :class:`ConditionIR` (empty → unconditional)."""
    if not spec:
        return ConditionIR()
    adjustments: list[Adjustment] = []
    for raw in spec.get("adjustments", ()) or ():
        adjustments.append(Adjustment(
            kind=str(raw.get("kind", "")),
            rate=_to_fraction(raw.get("rate", 0)),
            order=int(raw.get("order", 0)),
        ))
    truth = spec.get("predicate_truth")
    if truth is not None and not isinstance(truth, bool):
        raise FormulaError("predicate_truth must be a boolean or null")
    return ConditionIR(
        predicate=str(spec.get("predicate", "")),
        predicate_truth=truth,
        base_quantity=str(spec.get("base_quantity", "")),
        adjustments=tuple(adjustments),
    )


# --------------------------------------------------------------------------- layer 3: restricted AST
@dataclass(frozen=True)
class Node:
    """One restricted-DSL formula node: an op over child nodes, an operand ref, or a constant.

    Exactly one of ``op`` / ``ref`` / ``const`` is set. ``ndigits`` is meaningful only for ROUND.
    Built by :func:`build_formula` from a plain dict spec — never from a code string — so there is no
    path from the model's output to ``eval``.
    """

    op: str = ""
    args: tuple["Node", ...] = ()
    ref: str = ""
    const: Fraction | None = None
    ndigits: int = 0

    def to_dict(self) -> dict[str, Any]:
        if self.ref:
            return {"ref": self.ref}
        if self.const is not None:
            return {"const": _fraction_to_jsonable(self.const)}
        out: dict[str, Any] = {"op": self.op, "args": [a.to_dict() for a in self.args]}
        if self.op == ROUND:
            out["ndigits"] = self.ndigits
        return out


def build_formula(spec: Mapping[str, Any]) -> Node:
    """Parse a formula dict spec into a validated restricted AST (raises :class:`FormulaError`).

    Grammar (each node is a dict)::

        {"ref": "<operand name>"}                          # operand leaf
        {"const": <number>}                                # constant leaf
        {"op": "<ALLOWED_OP>", "args": [<node>, ...]}      # operation
        {"op": "ROUND", "args": [<node>], "ndigits": <int>}

    Anything else — an unknown op, wrong arity, a non-dict node, a code string — is rejected here, so the
    formula layer fails loudly instead of the model smuggling free Python through the lane.
    """
    if not isinstance(spec, Mapping):
        raise FormulaError(f"formula node must be an object, got {type(spec).__name__}")
    if "ref" in spec:
        name = str(spec["ref"]).strip()
        if not name:
            raise FormulaError("empty operand ref")
        return Node(ref=name)
    if "const" in spec:
        return Node(const=_to_fraction(spec["const"]))
    if "op" not in spec:
        raise FormulaError("formula node needs one of: op / ref / const")
    op = str(spec["op"]).strip().upper()
    if op not in ALLOWED_OPS:
        raise FormulaError(f"operation {op!r} is not in the allowed DSL {sorted(ALLOWED_OPS)}")
    raw_args = spec.get("args", ())
    if not isinstance(raw_args, (list, tuple)) or not raw_args:
        raise FormulaError(f"{op} needs a non-empty args list")
    args = tuple(build_formula(a) for a in raw_args)
    arity = _ARITY[op]
    if arity is not None and len(args) != arity:
        raise FormulaError(f"{op} takes exactly {arity} args, got {len(args)}")
    ndigits = 0
    if op == ROUND:
        ndigits = int(spec.get("ndigits", 0))
    return Node(op=op, args=args, ndigits=ndigits)


def formula_refs(node: Node) -> set[str]:
    """Every operand name the formula references (for the operand-layer completeness check)."""
    if node.ref:
        return {node.ref}
    out: set[str] = set()
    for a in node.args:
        out |= formula_refs(a)
    return out


# --------------------------------------------------------------------------- layer 4: Hard Executor
def _fraction_to_decimal(value: Fraction) -> Decimal:
    return Decimal(value.numerator) / Decimal(value.denominator)


def _round_half_up(value: Decimal, ndigits: int) -> Decimal:
    from decimal import ROUND_HALF_UP

    quant = Decimal(1).scaleb(-ndigits)
    return value.quantize(quant, rounding=ROUND_HALF_UP)


def _eval_decimal(node: Node, env: Mapping[str, BoundOperand]) -> Decimal:
    """Hard executor: evaluate ``node`` over ``env`` using Decimal arithmetic in a *local* 50-digit context.

    The whole evaluation runs inside :func:`decimal.localcontext` bound to :data:`_EXEC_CONTEXT`, so the
    process-global Decimal context is never mutated (byte-identical OFF guarantee)."""
    with localcontext(_EXEC_CONTEXT):
        return _eval_decimal_inner(node, env)


def _eval_decimal_inner(node: Node, env: Mapping[str, BoundOperand]) -> Decimal:
    if node.ref:
        op = env.get(node.ref)
        if op is None:
            raise FormulaError(f"unbound operand ref: {node.ref!r}")
        return _fraction_to_decimal(op.value)
    if node.const is not None:
        return _fraction_to_decimal(node.const)
    vals = [_eval_decimal_inner(a, env) for a in node.args]
    return _apply_decimal(node, vals)


def _apply_decimal(node: Node, vals: list[Decimal]) -> Decimal:
    op = node.op
    if op == ADD:
        return vals[0] + vals[1]
    if op == SUB:
        return vals[0] - vals[1]
    if op == MUL:
        return vals[0] * vals[1]
    if op == DIV:
        if vals[1] == 0:
            raise FormulaError("division by zero")
        return vals[0] / vals[1]
    if op == SUM:
        return sum(vals, Decimal(0))
    if op == MEAN:
        return sum(vals, Decimal(0)) / Decimal(len(vals))
    if op == RATIO:
        if vals[1] == 0:
            raise FormulaError("ratio by zero")
        return vals[0] / vals[1]
    if op == PERCENT_CHANGE:
        # (new - old) / old * 100 — args are [old, new]
        if vals[0] == 0:
            raise FormulaError("percent change on zero base")
        return (vals[1] - vals[0]) / vals[0] * Decimal(100)
    if op == WEIGHTED_SUM:
        # args alternate value, weight, value, weight, …
        if len(vals) % 2 != 0:
            raise FormulaError("WEIGHTED_SUM needs an even number of args (value,weight pairs)")
        total = Decimal(0)
        for i in range(0, len(vals), 2):
            total += vals[i] * vals[i + 1]
        return total
    if op == ROUND:
        return _round_half_up(vals[0], node.ndigits)
    if op == MIN:
        return min(vals)
    if op == MAX:
        return max(vals)
    raise FormulaError(f"unreachable op in executor: {op!r}")  # pragma: no cover


# --------------------------------------------------------------------------- layer 5: Independent Verifier
def _eval_fraction(node: Node, env: Mapping[str, BoundOperand]) -> Fraction:
    """Independent verifier: evaluate ``node`` over ``env`` using exact :class:`Fraction` arithmetic.

    This is the always-available exact backbone. When SymPy is present, :func:`_eval_sympy` re-derives the
    same value via ``sympy.Rational`` as a redundant, independent cross-check (research: "同じ Bound AST を
    SymPy Rational へ翻訳して独立検算")."""
    if node.ref:
        op = env.get(node.ref)
        if op is None:
            raise FormulaError(f"unbound operand ref: {node.ref!r}")
        return op.value
    if node.const is not None:
        return node.const
    vals = [_eval_fraction(a, env) for a in node.args]
    return _apply_fraction(node, vals)


def _apply_fraction(node: Node, vals: list[Fraction]) -> Fraction:
    op = node.op
    if op == ADD:
        return vals[0] + vals[1]
    if op == SUB:
        return vals[0] - vals[1]
    if op == MUL:
        return vals[0] * vals[1]
    if op == DIV:
        if vals[1] == 0:
            raise FormulaError("division by zero")
        return vals[0] / vals[1]
    if op == SUM:
        return sum(vals, Fraction(0))
    if op == MEAN:
        return sum(vals, Fraction(0)) / Fraction(len(vals))
    if op == RATIO:
        if vals[1] == 0:
            raise FormulaError("ratio by zero")
        return vals[0] / vals[1]
    if op == PERCENT_CHANGE:
        if vals[0] == 0:
            raise FormulaError("percent change on zero base")
        return (vals[1] - vals[0]) / vals[0] * Fraction(100)
    if op == WEIGHTED_SUM:
        if len(vals) % 2 != 0:
            raise FormulaError("WEIGHTED_SUM needs an even number of args (value,weight pairs)")
        total = Fraction(0)
        for i in range(0, len(vals), 2):
            total += vals[i] * vals[i + 1]
        return total
    if op == ROUND:
        return _round_fraction_half_up(vals[0], node.ndigits)
    if op == MIN:
        return min(vals)
    if op == MAX:
        return max(vals)
    raise FormulaError(f"unreachable op in verifier: {op!r}")  # pragma: no cover


def _round_fraction_half_up(value: Fraction, ndigits: int) -> Fraction:
    """Exact round-half-up matching Decimal's ROUND_HALF_UP: ties go *away from zero* (-2.5 → -3)."""
    scale = Fraction(10) ** ndigits
    scaled = value * scale
    sign = -1 if scaled < 0 else 1
    absval = abs(scaled)
    floor = absval.numerator // absval.denominator
    remainder = absval - floor
    rounded = floor + 1 if remainder >= Fraction(1, 2) else floor
    return sign * Fraction(rounded) / scale


def _eval_sympy(node: Node, env: Mapping[str, BoundOperand]) -> Fraction:
    """Re-derive the node value via ``sympy.Rational`` (independent backend) → :class:`Fraction`.

    Builds the SymPy expression *structurally from the validated AST* — never by parsing a string — so no
    ``parse_expr`` / ``eval`` path exists. Returns a :class:`Fraction` for direct comparison."""
    expr = _to_sympy_expr(node, env)
    rat = _sympy.nsimplify(expr, rational=True) if not expr.is_Rational else expr  # type: ignore
    rat = _sympy.Rational(rat)  # type: ignore
    return Fraction(int(rat.p), int(rat.q))  # type: ignore


def _to_sympy_expr(node: Node, env: Mapping[str, BoundOperand]):  # pragma: no cover - exercised when sympy present
    R = _sympy.Rational  # type: ignore
    if node.ref:
        op = env.get(node.ref)
        if op is None:
            raise FormulaError(f"unbound operand ref: {node.ref!r}")
        return R(op.value.numerator, op.value.denominator)
    if node.const is not None:
        return R(node.const.numerator, node.const.denominator)
    args = [_to_sympy_expr(a, env) for a in node.args]
    op = node.op
    if op == ADD:
        return args[0] + args[1]
    if op == SUB:
        return args[0] - args[1]
    if op == MUL:
        return args[0] * args[1]
    if op in (DIV, RATIO):
        return args[0] / args[1]
    if op == SUM:
        return sum(args, R(0))
    if op == MEAN:
        return sum(args, R(0)) / R(len(args))
    if op == PERCENT_CHANGE:
        return (args[1] - args[0]) / args[0] * R(100)
    if op == WEIGHTED_SUM:
        total = R(0)
        for i in range(0, len(args), 2):
            total += args[i] * args[i + 1]
        return total
    if op == ROUND:
        # Exact rational round-half-up mirrors the Fraction path (SymPy has no half-up rounding builtin).
        val = _eval_fraction(node, env)
        return R(val.numerator, val.denominator)
    if op == MIN:
        return _sympy.Min(*args)  # type: ignore
    if op == MAX:
        return _sympy.Max(*args)  # type: ignore
    raise FormulaError(f"unreachable op in sympy verifier: {op!r}")  # pragma: no cover


# --------------------------------------------------------------------------- three-layer verdict
LAYER_OPERAND = "operand"
LAYER_FORMULA = "formula"
LAYER_EXECUTION = "execution"

# Exact comparison is the goal; a tiny tolerance only absorbs Decimal's finite expansion of a
# non-terminating DIV (the Fraction verifier is exact). Both are compared as Fractions.
_EXEC_EPSILON = Fraction(1, 10 ** 12)


@dataclass(frozen=True)
class LayerVerdict:
    """Per-layer pass/fail with a code (aligned to :mod:`failure_taxonomy`) and a human note."""

    layer: str
    ok: bool
    code: str = ""
    note: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {"layer": self.layer, "ok": self.ok, "code": self.code, "note": self.note}


@dataclass(frozen=True)
class CandidateResult:
    """One PoT candidate fully evaluated: three-layer verdicts + the exact value + agreement signatures."""

    binding: Binding
    formula: Node
    condition: ConditionIR
    value: Fraction | None
    unit: str
    operand_verdict: LayerVerdict
    formula_verdict: LayerVerdict
    execution_verdict: LayerVerdict

    @property
    def all_layers_pass(self) -> bool:
        return (self.operand_verdict.ok and self.formula_verdict.ok
                and self.execution_verdict.ok and self.value is not None)

    def answer_signature(self) -> str:
        """Normalized answer value — the ``answer`` axis of N-sample agreement (exact Fraction string)."""
        return "∅" if self.value is None else f"{self.value.numerator}/{self.value.denominator}"

    def full_signature(self) -> tuple[str, tuple[str, ...], str, str]:
        """(answer, operand-source, unit, branch) — the 4-tuple N-sample majority must agree on."""
        return (self.answer_signature(), self.binding.source_signature(),
                self.unit, self.condition.branch_signature())

    def to_dict(self) -> dict[str, Any]:
        return {
            "value": _fraction_to_jsonable(self.value) if self.value is not None else None,
            "unit": self.unit,
            "binding": self.binding.to_dict(),
            "formula": self.formula.to_dict() if self.formula is not None else None,
            "condition": self.condition.to_dict(),
            "verdicts": {
                LAYER_OPERAND: self.operand_verdict.to_dict(),
                LAYER_FORMULA: self.formula_verdict.to_dict(),
                LAYER_EXECUTION: self.execution_verdict.to_dict(),
            },
            "all_layers_pass": self.all_layers_pass,
            "answer_signature": self.answer_signature(),
        }


def evaluate_candidate(spec: Mapping[str, Any], *, result_unit: str = "",
                       require_units: bool = False) -> CandidateResult:
    """Run one candidate through all five layers and return its three-layer :class:`CandidateResult`.

    ``spec`` is the restricted-DSL candidate the formula builder emits::

        {"operands": [{"name","value","unit","source"}, ...],
         "formula":  <restricted formula node>,
         "condition": {<optional ConditionIR spec>},
         "result_unit": "円"}

    Each layer is judged *independently* (research: 三層で個別判定) so an A-type failure (operand not
    bound / not sourced) is never conflated with a B-type failure (formula/execution mismatch):

    * **operand** — every referenced operand is bound, has a source (``operand_sources_complete``), a
      finite value, and (when ``require_units``) a unit. Fails ``EVIDENCE_INCOMPLETE``.
    * **formula**  — the spec parsed into a valid restricted AST (allowed ops / arity / resolvable refs)
      and, when conditional, the branch interpretation is internally consistent. Fails ``EXECUTION_DISAGREEMENT``.
    * **execution** — the Decimal hard executor and the independent Fraction(+SymPy) verifier agree on the
      value. Fails ``EXECUTION_DISAGREEMENT``.
    """
    result_unit = str(result_unit or spec.get("result_unit", "") or "")
    condition = build_condition(spec.get("condition"))

    # ---- formula layer (parse first: an unparseable formula can still let us judge operands) ----
    formula: Node | None = None
    formula_verdict: LayerVerdict
    try:
        formula = build_formula(spec.get("formula", {}))
        branch_ok, branch_note = _branch_consistent(formula, condition)
        if branch_ok:
            formula_verdict = LayerVerdict(LAYER_FORMULA, True, note="制限AST妥当・分岐整合")
        else:
            formula_verdict = LayerVerdict(LAYER_FORMULA, False, _tax.EXECUTION_DISAGREEMENT, branch_note)
    except FormulaError as exc:
        formula_verdict = LayerVerdict(LAYER_FORMULA, False, _tax.EXECUTION_DISAGREEMENT, str(exc))

    # ---- operand layer ----
    binding = build_binding(spec.get("operands", ()))
    operand_verdict = _judge_operands(binding, formula, require_units=require_units)

    # ---- execution layer ----
    value: Fraction | None = None
    if formula is not None and formula_verdict.ok and operand_verdict.ok:
        execution_verdict, value = _judge_execution(formula, binding)
    elif formula is None or not formula_verdict.ok:
        execution_verdict = LayerVerdict(LAYER_EXECUTION, False, _tax.EXECUTION_DISAGREEMENT,
                                         "式が無効なため実行不能")
    else:
        execution_verdict = LayerVerdict(LAYER_EXECUTION, False, _tax.EVIDENCE_INCOMPLETE,
                                         "operand 不足のため実行不能")

    return CandidateResult(
        binding=binding, formula=formula if formula is not None else Node(),
        condition=condition, value=value, unit=result_unit,
        operand_verdict=operand_verdict, formula_verdict=formula_verdict,
        execution_verdict=execution_verdict,
    )


def _judge_operands(binding: Binding, formula: Node | None, *, require_units: bool) -> LayerVerdict:
    if not binding.operands:
        return LayerVerdict(LAYER_OPERAND, False, _tax.EVIDENCE_INCOMPLETE, "operand が1つも束縛されていない")
    if not binding.operand_sources_complete:
        missing = [o.name for o in binding.operands if not o.sourced]
        return LayerVerdict(LAYER_OPERAND, False, _tax.EVIDENCE_INCOMPLETE,
                            f"source 欠落 operand: {', '.join(missing)}")
    if formula is not None:
        refs = formula_refs(formula)
        unbound = sorted(refs - set(binding.by_name))
        if unbound:
            return LayerVerdict(LAYER_OPERAND, False, _tax.EVIDENCE_INCOMPLETE,
                                f"式が参照する未束縛 operand: {', '.join(unbound)}")
    if require_units and not binding.units_complete:
        missing = [o.name for o in binding.operands if not o.has_unit]
        return LayerVerdict(LAYER_OPERAND, False, _tax.EVIDENCE_INCOMPLETE,
                            f"unit 欠落 operand: {', '.join(missing)}")
    return LayerVerdict(LAYER_OPERAND, True, note="operand_sources_complete")


def _judge_execution(formula: Node, binding: Binding) -> tuple[LayerVerdict, Fraction | None]:
    env = binding.by_name
    try:
        decimal_val = _eval_decimal(formula, env)
        fraction_val = _eval_fraction(formula, env)
    except FormulaError as exc:
        return LayerVerdict(LAYER_EXECUTION, False, _tax.EXECUTION_DISAGREEMENT, str(exc)), None
    # Decimal (hard executor) vs exact Fraction (independent verifier): agree within epsilon.
    decimal_as_fraction = Fraction(decimal_val)
    if abs(decimal_as_fraction - fraction_val) > _EXEC_EPSILON:
        return (LayerVerdict(LAYER_EXECUTION, False, _tax.EXECUTION_DISAGREEMENT,
                             f"実行={decimal_val} vs 検算={float(fraction_val)} 不一致"), None)
    # Redundant SymPy Rational cross-check when available — must also agree.
    if _HAVE_SYMPY:
        try:
            sympy_val = _eval_sympy(formula, env)
        except Exception as exc:  # noqa: BLE001 - sympy path is a cross-check; never crash the lane
            return (LayerVerdict(LAYER_EXECUTION, False, _tax.EXECUTION_DISAGREEMENT,
                                 f"SymPy検算失敗: {exc}"), None)
        if abs(sympy_val - fraction_val) > _EXEC_EPSILON:
            return (LayerVerdict(LAYER_EXECUTION, False, _tax.EXECUTION_DISAGREEMENT,
                                 f"SymPy={float(sympy_val)} vs Fraction={float(fraction_val)} 不一致"), None)
    backend = "sympy+fraction+decimal" if _HAVE_SYMPY else "fraction+decimal"
    return LayerVerdict(LAYER_EXECUTION, True, note=f"三系一致 ({backend})"), fraction_val


def _branch_consistent(formula: Node, condition: ConditionIR) -> tuple[bool, str]:
    """Whether a conditional formula's branch interpretation is internally consistent (idx76 型対策).

    For a conditional question the chosen ``base_quantity`` must actually be referenced by the formula —
    otherwise the branch selection and the arithmetic disagree (the idx76 failure: the 減額 was applied to
    a quantity the formula never used). Unconditional formulas are always consistent."""
    if not condition.is_conditional:
        return True, ""
    if condition.base_quantity:
        refs = formula_refs(formula)
        if condition.base_quantity not in refs:
            return (False,
                    f"分岐の base_quantity『{condition.base_quantity}』を式が参照していない"
                    f"(参照: {', '.join(sorted(refs)) or 'なし'})")
    if condition.predicate and condition.predicate_truth is None:
        return False, "条件文があるが predicate_truth が未確定(分岐選択が未定)"
    return True, ""


# --------------------------------------------------------------------------- layer 6: N-sample majority
COMMIT = "COMMIT"
NEED_MORE = "NEED_MORE"
ABSTAIN = "ABSTAIN"


@dataclass(frozen=True)
class MajorityDecision:
    """The outcome of N-sample majority over candidates (research §N-sample)."""

    status: str                      # COMMIT | NEED_MORE | ABSTAIN
    value: Fraction | None
    unit: str
    agreement: int                   # size of the winning agreeing group
    total: int                       # candidates considered
    signature: tuple[str, tuple[str, ...], str, str] | None
    code: str = ""                   # failure taxonomy code when not COMMIT
    note: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "status": self.status,
            "value": _fraction_to_jsonable(self.value) if self.value is not None else None,
            "unit": self.unit,
            "agreement": self.agreement,
            "total": self.total,
            "signature": ({"answer": self.signature[0], "operand_source": list(self.signature[1]),
                           "unit": self.signature[2], "branch": self.signature[3]}
                          if self.signature else None),
            "code": self.code,
            "note": self.note,
        }


def nsample_majority(candidates: Sequence[CandidateResult], *, simple: bool = False,
                     min_multi_agreement: int = 2) -> MajorityDecision:
    """Aggregate PoT candidates by full-signature agreement (research §N-sample).

    Only candidates whose all three layers pass are eligible to vote. Agreement requires the full
    4-tuple ``(answer, operand source, unit, branch interpretation)`` to match — not the answer alone —
    so two candidates that reach the same number via different operands / branches do **not** count as
    agreeing (the idx76 lesson: a right-looking number from the wrong branch must not win).

    * ``simple`` (simple derivation, 1 candidate): commit iff that single candidate passes all layers and
      its execution was independently verified — no vote needed.
    * multi-hop / conditional (≥3 candidates): the largest fully-agreeing group must reach
      ``min_multi_agreement``; otherwise the caller should draw 2 more candidates (``NEED_MORE``) and, if
      still short, abstain (``ABSTAIN``) rather than commit a plurality-of-one.
    """
    total = len(candidates)
    eligible = [c for c in candidates if c.all_layers_pass]
    if not eligible:
        return MajorityDecision(ABSTAIN, None, "", 0, total, None,
                                code=_dominant_failure_code(candidates),
                                note="全候補が三層のいずれかで不合格")
    if simple:
        c = eligible[0]
        return MajorityDecision(COMMIT, c.value, c.unit, 1, total, c.full_signature(),
                                note="simple: 単一候補が三層合格・独立検算済み")
    groups: dict[tuple[str, tuple[str, ...], str, str], list[CandidateResult]] = {}
    for c in eligible:
        groups.setdefault(c.full_signature(), []).append(c)
    best_sig, best = max(groups.items(), key=lambda kv: len(kv[1]))
    agree = len(best)
    if agree >= min_multi_agreement:
        winner = best[0]
        return MajorityDecision(COMMIT, winner.value, winner.unit, agree, total, best_sig,
                                note=f"majority: {agree}/{total} が answer∧source∧unit∧branch で一致")
    # Only a single-candidate plurality: ask for more before ever committing (or abstain when exhausted).
    if total < 5:
        return MajorityDecision(NEED_MORE, None, "", agree, total, None,
                                code=_tax.EXECUTION_DISAGREEMENT,
                                note="多数決の一致が閾値未満。追加候補を2つ生成して再判定する")
    return MajorityDecision(ABSTAIN, None, "", agree, total, None,
                            code=_tax.EXECUTION_DISAGREEMENT,
                            note="追加生成後も一致せず。誤答を出さず棄権する")


def _dominant_failure_code(candidates: Sequence[CandidateResult]) -> str:
    """The taxonomy code to attribute an all-fail candidate set to (operand-layer failures dominate)."""
    if not candidates:
        return _tax.EVIDENCE_INCOMPLETE
    if any(not c.operand_verdict.ok for c in candidates):
        return _tax.EVIDENCE_INCOMPLETE
    return _tax.EXECUTION_DISAGREEMENT


# --------------------------------------------------------------------------- forced-lane convenience
@dataclass(frozen=True)
class PoTResult:
    """The full forced-lane result: per-candidate three-layer verdicts + the N-sample decision."""

    decision: MajorityDecision
    candidates: tuple[CandidateResult, ...]

    def to_dict(self) -> dict[str, Any]:
        return {"decision": self.decision.to_dict(),
                "candidates": [c.to_dict() for c in self.candidates]}


def run_forced_lane(candidate_specs: Sequence[Mapping[str, Any]], *, simple: bool | None = None,
                    require_units: bool = False, min_multi_agreement: int = 2) -> PoTResult:
    """The NUMERIC route's forced lane: evaluate every candidate spec, then N-sample majority.

    This is the single entry the router/investigator dispatches a NUMERIC question through: a question's
    LLM-emitted candidate(s) (operands + restricted formula + optional condition) are each run
    binder→AST→Decimal→independent-verifier, judged per layer, and combined by full-signature majority.
    ``simple`` defaults to True for a single candidate and False otherwise.
    """
    if simple is None:
        simple = len(candidate_specs) <= 1
    candidates = tuple(
        evaluate_candidate(spec, require_units=require_units) for spec in candidate_specs)
    decision = nsample_majority(candidates, simple=simple, min_multi_agreement=min_multi_agreement)
    return PoTResult(decision=decision, candidates=candidates)


# --------------------------------------------------------------------------- rendering helpers
def _fraction_to_jsonable(value: Fraction) -> Any:
    """Render a Fraction as an int when integral, else a float — for JSON-safe traces."""
    if value.denominator == 1:
        return value.numerator
    return float(value)


def render_value(value: Fraction | None, unit: str = "") -> str | None:
    """Render a lane result value for an answer string (integral → ``12,345``; else a trimmed decimal)."""
    if value is None:
        return None
    if value.denominator == 1:
        body = f"{value.numerator:,}"
    else:
        body = f"{float(value)}"
    return f"{body}{unit}" if unit else body


# --------------------------------------------------------------------------- agent tool wiring
TOOL_NAME = "verify_formula"

TOOL_DESCRIPTION = (
    "数値導出(NUMERIC)専用の強制計算レーン。暗算・自由 Python を禁止し、operand を value/unit/source 付きで"
    "束縛→許可演算限定 AST(ADD/SUB/MUL/DIV/SUM/MEAN/RATIO/PERCENT_CHANGE/WEIGHTED_SUM/ROUND/MIN/MAX)で式を"
    "組む→Decimal 厳密実行→独立検算(SymPy/有理数)で突き合わせ、operand/formula/execution の三層を個別判定"
    "して返す。数値の答えは必ずこのツールで計算してから submit_answer すること。\n"
    "candidates=候補の配列。各候補は {operands:[{name,value,unit,source}...], formula:<式ノード>, "
    "condition?:{predicate,predicate_truth,base_quantity,adjustments?}, result_unit}。式ノードは "
    "{ref:操作名} / {const:数} / {op:演算, args:[...節点...], ndigits?:桁(ROUND時)}。"
    "単純計算は候補1つ、multi-hop/条件付きは候補3つを別々の束縛/式で作ると多数決(answer∧source∧unit∧branch"
    "の全一致)で確定する。source は doc:sheet!cell 形式で必ず埋める(欠落は operand 層で不合格)。")

# JSON-schema (loose) for the tool parameters — the loop passes them straight through to run_forced_lane.
_NODE_SCHEMA: dict[str, Any] = {"type": "object"}
_OPERAND_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "name": {"type": "string"},
        "value": {"type": ["number", "string"]},
        "unit": {"type": "string"},
        "source": {"type": "string"},
    },
    "required": ["name", "value", "source"],
}
_CANDIDATE_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "operands": {"type": "array", "items": _OPERAND_SCHEMA},
        "formula": _NODE_SCHEMA,
        "condition": {"type": "object"},
        "result_unit": {"type": "string"},
    },
    "required": ["operands", "formula"],
}
TOOL_PARAMETERS: dict[str, Any] = {
    "type": "object",
    "properties": {
        "candidates": {"type": "array", "items": _CANDIDATE_SCHEMA},
        "simple": {"type": "boolean"},
        "require_units": {"type": "boolean"},
    },
    "required": ["candidates"],
}

# SOT-2616 — the augmented schema exposed *only* when operand prefill is on: an operand may be selected
# from the injected catalog by ``select`` (id) instead of a literal ``value``, and the call carries the
# ``catalog`` array so the lane binds value/unit/source from the enumerated cell verbatim. The base schema
# above is untouched, so the default-OFF tool definition stays byte-identical.
_OPERAND_SCHEMA_SELECT: dict[str, Any] = {
    "type": "object",
    "properties": {
        "name": {"type": "string"},
        "select": {"type": "string", "description": "operand 候補カタログの id (op1…)"},
        "value": {"type": ["number", "string"]},
        "unit": {"type": "string"},
        "source": {"type": "string"},
    },
    "required": ["name"],
}
_CANDIDATE_SCHEMA_SELECT: dict[str, Any] = {
    "type": "object",
    "properties": {
        "operands": {"type": "array", "items": _OPERAND_SCHEMA_SELECT},
        "formula": _NODE_SCHEMA,
        "condition": {"type": "object"},
        "result_unit": {"type": "string"},
    },
    "required": ["operands", "formula"],
}
TOOL_PARAMETERS_PREFILL: dict[str, Any] = {
    "type": "object",
    "properties": {
        "candidates": {"type": "array", "items": _CANDIDATE_SCHEMA_SELECT},
        "catalog": {"type": "array", "items": {"type": "object"},
                    "description": "注入された operand 候補カタログ(そのまま渡す)"},
        "simple": {"type": "boolean"},
        "require_units": {"type": "boolean"},
    },
    "required": ["candidates"],
}


# --------------------------------------------------------------------------- SOT-2616 operand handoff
class CatalogError(FormulaError):
    """Raised when a candidate selects an operand id absent from the prefilled catalog."""


def resolve_operand_selections(candidates: Sequence[Mapping[str, Any]],
                               catalog: Sequence[Mapping[str, Any]] | None
                               ) -> list[dict[str, Any]]:
    """Bind ``{"name", "select": <id>}`` operands from a prefilled candidate catalog (SOT-2616).

    The operand-prefill layer (:mod:`src.rag.agent.operand_prefill`) injects a deterministic catalog of
    numeric candidates; the LLM *selects* by ``id`` instead of re-typing values. This resolves each
    operand's ``select`` into the catalog entry's value/unit/source (the operand keeps its role ``name``),
    so the value the lane binds is the enumerated cell's value **verbatim** — the model cannot mistype or
    invent it. Operands with an explicit ``value`` and no ``select`` pass through unchanged (mixed
    selection + literal is allowed), so this is a pure superset of the existing tool contract. Returns a
    new candidate list (inputs are not mutated). A ``select`` referencing an unknown id raises
    :class:`CatalogError`; an empty/absent catalog with no ``select`` operands is a no-op.
    """
    by_id: dict[str, Mapping[str, Any]] = {}
    for entry in catalog or ():
        cid = str(entry.get("id", "")).strip()
        if cid:
            by_id[cid] = entry
    resolved: list[dict[str, Any]] = []
    for cand in candidates:
        cand = dict(cand)
        ops_out: list[dict[str, Any]] = []
        for op in cand.get("operands", ()) or ():
            op = dict(op)
            sel = op.pop("select", None)
            if sel is not None and op.get("value") in (None, ""):
                entry = by_id.get(str(sel).strip())
                if entry is None:
                    raise CatalogError(
                        f"未知の operand 候補 id: {sel!r}（catalog の id から選んでください）")
                op["value"] = entry.get("value")
                if not op.get("unit") and entry.get("unit"):
                    op["unit"] = entry.get("unit")
                if not op.get("source") and entry.get("source"):
                    op["source"] = entry.get("source")
                if not str(op.get("name", "")).strip():
                    op["name"] = str(entry.get("label") or sel)
            ops_out.append(op)
        cand["operands"] = ops_out
        resolved.append(cand)
    return resolved


def verify_formula(candidates: Sequence[Mapping[str, Any]] | None = None, *,
                   simple: bool | None = None, require_units: bool = False,
                   min_multi_agreement: int = 2,
                   catalog: Sequence[Mapping[str, Any]] | None = None) -> dict[str, Any]:
    """Agent-callable entry: run the forced lane over ``candidates`` and return a JSON-safe verdict.

    Errors are returned as ``{"error": ...}`` (never raised) so a malformed candidate spec surfaces to the
    model as feedback — the loop must not crash on one bad tool call. On a COMMIT it also carries the
    rendered ``value`` string so the model can submit it verbatim (no re-derivation, no 暗算).

    ``catalog`` (SOT-2616) is the prefilled operand-candidate catalog; when supplied, operands may be
    selected by ``{"name", "select": <id>}`` and are bound to the catalog entry's value/unit/source
    before evaluation. Absent/empty catalog keeps the original literal-operand contract byte-for-byte."""
    if not candidates:
        return {"error": "candidates が空です。少なくとも1つの {operands, formula} 候補を渡してください。"}
    try:
        prepared = resolve_operand_selections(list(candidates), catalog) if catalog else list(candidates)
    except CatalogError as exc:
        return {"error": f"operand 選択 不正: {exc}"}
    try:
        result = run_forced_lane(prepared, simple=simple, require_units=require_units,
                                 min_multi_agreement=min_multi_agreement)
    except FormulaError as exc:
        return {"error": f"formula spec 不正: {exc}"}
    out = result.to_dict()
    dec = result.decision
    out["value"] = render_value(dec.value, dec.unit) if dec.status == COMMIT else None
    out["status"] = dec.status
    return out


def numeric_lane_directive() -> str:
    """The extra pre-inject directive appended to a NUMERIC Evidence Packet when the lane is enabled.

    Tells the agent that the NUMERIC route routes its arithmetic through :data:`TOOL_NAME`
    (binder→制限AST→Decimal→独立検算) rather than 暗算, and adopts the lane's verified value with high
    confidence on success. On a lane failure it does NOT force an abstain (SOT-2598): the agent degrades
    to the normal free-form answer path (still subject to the usual commit/confidence gate, still barred
    from fabricating a number) instead of dropping an evidence-supported correct answer — 原則
    「wrong→abstain は許すが match→abstain は許さない」. Injects no corpus fact and no answer — only the
    lane protocol."""
    return (
        "【NUMERIC 強制計算レーン(PoT)】この質問は数値導出型です。答えの数値はまず必ず "
        f"`{TOOL_NAME}` ツールで計算してください(暗算・自由コード禁止)。operand を "
        "value/unit/source(doc:sheet!cell) 付きで束縛し、許可演算限定の式(AST)で組み、三層(operand/"
        "formula/execution)がすべて合格して独立検算が一致した値は、高信頼で submit_answer に採用してください。"
        "条件付き(税込/税抜・◯分の◯に減額 等)は condition の predicate_truth と base_quantity を明示し、"
        "multi-hop/条件付きでは候補を3つ作って多数決(answer∧source∧unit∧branch の全一致)で確定します。"
        "三層のいずれかが不合格、または検算が一致しない場合でも即棄権にはしないでください: 数値の創作"
        "(暗算・当て推量)は禁止のままですが、原文(セル値・本文の記載値)から直接根拠づけられる回答があれば、"
        "通常の回答経路(submit_answer)でその根拠に基づく最善の回答を返してください(その回答は通常どおり "
        "commit/confidence ゲートで判定されます)。exec_verifier 等の独立照合が併用されているときはその照合結果を"
        "尊重してください。原文から根拠づけられる値が本当に無いときにのみ棄権してください。強制レーンで束縛"
        "しきれなかったというだけの理由で、原文から支持できる正答を棄権に落とさないこと。")
