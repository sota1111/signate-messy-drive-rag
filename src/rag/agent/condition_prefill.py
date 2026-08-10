"""SOT-2621 — pre-loop BranchCondition IR construction for what-if NUMERIC questions.

Phase-0 diagnostics (``docs/ai/budget32_trace_classification.md``) showed the NUMERIC route's
``BUDGET_EXHAUSTED`` losses concentrate on **what-if / 条件分岐型** derivations: the branch structure of a
question ("契約単価が現状より2,000円高く…の場合", "〜の場合は7.5%減額", "〜を除いて") is never made explicit,
so the loop pushes the condition into ``compute`` as guess-and-check (idx76: 18 ターン中 17 回 search で
operand ゼロ; idx47: compute×7) and derivation succeeds only 41%.

The :class:`~src.rag.agent.pot_lane.ConditionIR` IR (SOT-2586, predicate / predicate_truth /
base_quantity / adjustments[kind, rate, order]) that makes a branch *explicitly checkable* already exists —
but it is only ever built *inside* the answer loop, past the point where search has already burned the
budget. This module builds the IR **skeleton before the loop**, deterministically, from the question text,
and injects it into the Evidence Packet so the loop's LLM job shrinks from 自由探索 to *filling the blanks*
(選ぶ: base_quantity は operand 候補から / 確定する: predicate_truth と各調整の値) before handing off to the
PoT lane (制限AST→Decimal→独立検算).

What this module does
---------------------
Given a question, it detects three deterministic condition shapes and emits a ``ConditionIR`` **spec**
(a plain dict the PoT lane's :func:`~src.rag.agent.pot_lane.build_condition` / ``ConditionIR.from_spec``
already consume):

  * **相対増減 (assumption change)** — "契約単価が現状より2,000円高く" / "実績工数が11.2時間少なかった" →
    an additive ``delta`` adjustment (operand hint + signed delta + unit).
  * **分岐率 (branch rate)** — "7.5%減額" / "10%割引" / "5%加算" → a multiplicative ``rate`` adjustment.
  * **除外 (exclusion)** — "〜を除いて" / "〜を除く" → an ``exclusion`` adjustment naming the excluded set.

Design invariants (mirroring the sibling opt-ins — evidence_packet / operand_prefill / pot_lane):
  * **Opt-in, byte-identical OFF.**  :func:`enabled` (``RAG_CONDITION_PREIR``) gates the whole thing; the
    default-OFF serve path never builds a spec, injects nothing, and is byte-identical.
  * **No firing-condition relaxation.**  A question with no *detectable* adjustment builds **no** spec and
    follows the ordinary path (the SOT-2601 net12 failure axis — 発火条件の緩和 — is deliberately avoided).
    The IR is only ever *added* as a hint; it never removes a lane or forces an interpretation.
  * **Deterministic & answer-free.**  Detection is regex over the question text only. The skeleton carries
    the branch *structure* (which operand, what kind of adjustment, in what order) — the LLM still selects
    base_quantity from the operand candidates and confirms predicate_truth; the arithmetic stays in the
    PoT lane. The module never invents an operand value and never chooses the answer.
"""
from __future__ import annotations

import os
import re
from dataclasses import dataclass, field
from typing import Any

_ON = {"1", "true", "yes", "on"}


def enabled() -> bool:
    """True when the NUMERIC serve path should pre-build a BranchCondition IR (default OFF — opt-in)."""
    return os.getenv("RAG_CONDITION_PREIR", "0").strip().lower() in _ON


# --------------------------------------------------------------------------- number / unit lexicon
# Half/full-width digits, thousands separators (,/，) and a decimal point — the exact literal is kept
# verbatim as the delta/rate *hint* (the lane re-parses the operand's real value; this is guidance only).
_NUM = r"[0-9０-９][0-9０-９,，]*(?:[.．][0-9０-９]+)?"
# Units a numeric operand carries in this corpus. Longer units first so "万円" wins over "円".
_UNIT = r"万円|円|時間|分|時|日|ヶ月|カ月|箇月|か月|月|年|人日|人月|人|件|個|回|台|%|％|ポイント|pt"

# Direction verbs/adjectives for a relative assumption change, folded to increase / decrease.
_INC_WORDS = ("高く", "高かっ", "高い", "多く", "多かっ", "多い", "増え", "増加", "上がっ", "上乗せ", "加え")
_DEC_WORDS = ("安く", "安かっ", "安い", "低く", "低かっ", "低い", "少なく", "少なかっ", "少ない",
              "減っ", "減少", "下がっ", "減らし", "引い")
_DIR_ALT = "|".join(re.escape(w) for w in (_INC_WORDS + _DEC_WORDS))

# 相対増減: "<subject>(…より(も)) <num><unit> <dir>". The leading 「…より(も)」 is optional so a bare
# comparative clause ("実績工数が11.2時間少なかった") is caught too; the subject is recovered separately
# (see _operand_hint). Delta scanning is gated on a what-if context (see :func:`_has_whatif_context`) so a
# stray comparative in an ordinary NUMERIC question does not fire.
_REL_DELTA = re.compile(
    rf"(?:より\s*(?:も)?\s*)?(?P<num>{_NUM})\s*(?P<unit>{_UNIT})?\s*(?P<dir>{_DIR_ALT})")

# 分岐率: "<num>%(減額|割引|…|加算|増額|…)". Discount vs surcharge folds into the adjustment kind.
_DISCOUNT_WORDS = ("減額", "割引", "値引き", "値引", "割り引き", "引き下げ", "オフ", "ディスカウント", "ダウン")
_SURCHARGE_WORDS = ("加算", "増額", "上乗せ", "割増", "割り増し", "引き上げ", "アップ")
_RATE_ALT = "|".join(re.escape(w) for w in (_DISCOUNT_WORDS + _SURCHARGE_WORDS))
_BRANCH_RATE = re.compile(rf"(?P<num>{_NUM})\s*[%％]\s*(?:の)?\s*(?P<kind>{_RATE_ALT})")

# 除外: "<target>を除いて/除き/除く/除外し". Target is the noun phrase in the same 、-segment.
_EXCLUSION = re.compile(r"(?P<target>[^、。「」]{1,40}?)\s*を\s*除(?:いて|き|く|外し)")

# Branch marker words that end a what-if predicate clause.
_BRANCH_MARKERS = ("場合", "だったら", "としたら", "とすると", "とした場合", "なら", "ならば")


def _has_whatif_context(question: str) -> bool:
    """Whether the question carries a genuine what-if/仮定 signal that licenses relative-delta detection.

    A bare comparative ("最も多い" / "…より大きい") in an ordinary NUMERIC question must not be read as an
    assumption change, so additive delta scanning only runs when the text also contains a branch marker
    (「場合」「だったら」…) or an explicit 「より(も)」/「仮に」 hypothetical anchor. The percentage-rate and
    exclusion detectors carry their own explicit anchors and are not gated by this.
    """
    if "より" in question or "仮に" in question or "もし" in question:
        return True
    return any(mk in question for mk in _BRANCH_MARKERS)


def _to_halfwidth(s: str) -> str:
    """Fold full-width digits / separators / point to ASCII so a delta hint is a clean numeric literal."""
    trans = {ord("，"): ",", ord("．"): ".", ord("　"): " "}
    for i in range(10):
        trans[ord("０") + i] = str(i)
    return s.translate(trans)


def _num_literal(raw: str) -> str:
    """The raw match folded to ASCII with thousands separators stripped (e.g. '2,000' → '2000')."""
    return _to_halfwidth(raw).replace(",", "").strip()


# --------------------------------------------------------------------------- adjustment / spec model
@dataclass(frozen=True)
class DetectedAdjustment:
    """One deterministically-detected adjustment in a what-if question (a ConditionIR adjustment skeleton).

    ``kind`` mirrors the PoT lane's free ``Adjustment.kind`` label. ``delta``/``rate`` are the *hint*
    literals (strings, verbatim from the text); the lane re-binds the real operand value — these only tell
    the LLM which knob the branch turns. Exactly the additive branch carries ``delta`` (+ ``unit``); the
    multiplicative branch carries ``rate``; the exclusion carries only its ``operand_hint`` (the set name).
    """

    kind: str                       # "delta" | "discount" | "surcharge" | "exclusion"
    order: int
    operand_hint: str = ""          # the subject the adjustment acts on (a hint for base/operand selection)
    delta: str = ""                 # signed additive literal, e.g. "+2000" / "-11.2" (delta branch)
    rate: str = ""                  # fractional literal, e.g. "0.075" (rate branch, = pct/100)
    unit: str = ""
    direction: str = ""             # "increase" | "decrease" | "" — the folded sense of the change
    span: tuple[int, int] = (0, 0)  # match span in the question (used only to build the predicate)

    def to_ir_adjustment(self) -> dict[str, Any]:
        """The adjustment entry for the ConditionIR spec (``build_condition`` reads kind/rate/order).

        Extra keys (delta/unit/direction/operand_hint) are carried for the directive + audit and are
        ignored by :func:`~src.rag.agent.pot_lane.build_condition`, so the spec stays a valid IR skeleton.
        ``rate`` defaults to 0 for a not-yet-quantified additive/exclusion adjustment (a blank the LLM
        fills), keeping the skeleton conservative.
        """
        entry: dict[str, Any] = {
            "kind": self.kind,
            "rate": self.rate if self.rate else 0,
            "order": self.order,
            "operand_hint": self.operand_hint,
        }
        if self.delta:
            entry["delta"] = self.delta
        if self.unit:
            entry["unit"] = self.unit
        if self.direction:
            entry["direction"] = self.direction
        return entry


@dataclass(frozen=True)
class ConditionSpec:
    """The pre-loop BranchCondition IR skeleton for one what-if question (empty ⇒ nothing detected)."""

    condition_type: str                          # "assumption_change" | "branch_rate" | "exclusion" | "mixed"
    predicate: str
    adjustments: tuple[DetectedAdjustment, ...] = ()
    predicate_truth: bool | None = None          # left None: the LLM confirms which branch is taken
    base_quantity: str = ""                      # left "": the LLM selects from operand candidates

    @property
    def is_fired(self) -> bool:
        return bool(self.adjustments)

    def to_ir_spec(self) -> dict[str, Any]:
        """The ConditionIR spec dict injected into the Evidence Packet and consumed by the PoT lane."""
        return {
            "condition_type": self.condition_type,
            "predicate": self.predicate,
            "predicate_truth": self.predicate_truth,
            "base_quantity": self.base_quantity,
            "adjustments": [a.to_ir_adjustment() for a in self.adjustments],
        }


# --------------------------------------------------------------------------- detection helpers
def _operand_hint(question: str, num_start: int) -> str:
    """Recover the subject a relative delta acts on — the noun heading the delta's 、-segment.

    idx76 「契約単価が現状よりも2,000円高く」→ 「契約単価」; 「実績工数が11.2時間少なかった」→ 「実績工数」.
    Take the current 、-segment up to the number, drop the 「…現状より(も)」 reference tail and the leading
    binding particle, then keep the trailing noun phrase.
    """
    seg_start = max((question.rfind(sep, 0, num_start) for sep in ("、", "。", "，", "「", "」")),
                    default=-1) + 1
    pre = question[seg_start:num_start]
    # drop the "…現状より(も)" / "…実際より" reference tail that separates the subject from the delta.
    pre = re.sub(r"(現状|実際|通常|標準|平均|従来|当初|元)?\s*より\s*(?:も|は)?\s*$", "", pre)
    pre = pre.strip("　 \t")
    # if a topic/subject particle heads the noun ("契約単価が" / "工数は"), keep the phrase before it.
    m = re.search(r"([^はがをにでと、。」」]{1,24})[はがを]\s*$", pre)
    if m and m.group(1).strip():
        return m.group(1).strip()
    # otherwise drop a trailing binding particle so a bare "契約単価が" → "契約単価".
    return re.sub(r"[はがをのにでとも]\s*$", "", pre).strip("　 \t")


def _predicate(question: str, adjustments: list[DetectedAdjustment]) -> str:
    """The verbatim what-if clause: from the first adjustment's segment through the branch marker.

    Falls back to the span covering all adjustments when no 「場合/だったら/…」 marker follows them.
    """
    if not adjustments:
        return ""
    first = min(a.span[0] for a in adjustments)
    last = max(a.span[1] for a in adjustments)
    seg_start = max((question.rfind(sep, 0, first) for sep in ("、", "。", "「", "」")), default=-1) + 1
    # extend to include a trailing branch marker within a short window after the last adjustment.
    end = last
    tail = question[last:last + 40]
    marker_end = -1
    for mk in _BRANCH_MARKERS:
        pos = tail.find(mk)
        if pos != -1:
            marker_end = max(marker_end, last + pos + len(mk))
    if marker_end != -1:
        end = marker_end
    return question[seg_start:end].strip("、。　 \t")


# --------------------------------------------------------------------------- public: detect / build
def detect(question: str) -> ConditionSpec | None:
    """Deterministically extract a BranchCondition IR skeleton from ``question`` (None ⇒ no condition).

    Returns ``None`` unless at least one adjustment (relative delta / branch rate / exclusion) is found —
    a bare 「〜の場合」 with no quantifiable knob deliberately does **not** fire, so the ordinary loop keeps
    handling it (no firing-condition relaxation). The returned spec's ``predicate_truth`` and
    ``base_quantity`` are intentionally left blank — those are the LLM's to fill.
    """
    if not question:
        return None
    q = question
    adjustments: list[DetectedAdjustment] = []
    order = 0

    # 相対増減 (additive) — the idx76 flagship shape. Only when a genuine what-if context licenses it.
    if _has_whatif_context(q):
        for m in _REL_DELTA.finditer(q):
            direction = "increase" if m.group("dir") in _INC_WORDS else "decrease"
            sign = "+" if direction == "increase" else "-"
            adjustments.append(DetectedAdjustment(
                kind="delta", order=order,
                operand_hint=_operand_hint(q, m.start("num")),
                delta=f"{sign}{_num_literal(m.group('num'))}",
                unit=m.group("unit") or "",
                direction=direction,
                span=(m.start(), m.end()),
            ))
            order += 1

    # 分岐率 (multiplicative) — "7.5%減額" / "10%割引" / "5%加算".
    for m in _BRANCH_RATE.finditer(q):
        word = m.group("kind")
        is_discount = word in _DISCOUNT_WORDS
        pct = _num_literal(m.group("num"))
        try:
            rate = f"{float(pct) / 100:g}"
        except ValueError:
            rate = ""
        adjustments.append(DetectedAdjustment(
            kind="discount" if is_discount else "surcharge", order=order,
            operand_hint=_operand_hint(q, m.start()),
            rate=rate,
            direction="decrease" if is_discount else "increase",
            span=(m.start(), m.end()),
        ))
        order += 1

    # 除外 — "〜を除いて/除く".
    for m in _EXCLUSION.finditer(q):
        target = m.group("target").strip("　 \t")
        if not target:
            continue
        adjustments.append(DetectedAdjustment(
            kind="exclusion", order=order,
            operand_hint=target,
            span=(m.start("target"), m.end()),
        ))
        order += 1

    if not adjustments:
        return None

    adjustments.sort(key=lambda a: a.span[0])
    for i, a in enumerate(adjustments):
        adjustments[i] = DetectedAdjustment(**{**a.__dict__, "order": i})

    kinds = {a.kind for a in adjustments}
    if kinds <= {"delta"}:
        ctype = "assumption_change"
    elif kinds <= {"discount", "surcharge"}:
        ctype = "branch_rate"
    elif kinds <= {"exclusion"}:
        ctype = "exclusion"
    else:
        ctype = "mixed"

    return ConditionSpec(
        condition_type=ctype,
        predicate=_predicate(q, adjustments),
        adjustments=tuple(adjustments),
    )


def build_condition_ir(question: str) -> dict[str, Any] | None:
    """The ConditionIR spec dict for the Evidence Packet (``None`` when no condition is detected)."""
    spec = detect(question)
    return spec.to_ir_spec() if spec is not None else None


# --------------------------------------------------------------------------- directive rendering
_KIND_JA = {"delta": "増減", "discount": "減額(割引)", "surcharge": "加算(割増)", "exclusion": "除外"}


def _adjustment_line(a: dict[str, Any]) -> str:
    order = a.get("order", 0)
    hint = a.get("operand_hint") or "(operand未特定)"
    kind = a.get("kind", "")
    label = _KIND_JA.get(kind, kind)
    if kind == "delta":
        delta = a.get("delta", "")
        unit = a.get("unit", "")
        sense = "増加" if a.get("direction") == "increase" else "減少"
        return f"  - [order{order}] 『{hint}』を {delta}{unit} {sense}（kind={kind}）"
    if kind in ("discount", "surcharge"):
        rate = a.get("rate", "")
        return f"  - [order{order}] 『{hint}』に rate={rate}（{label}, kind={kind}）"
    if kind == "exclusion":
        return f"  - [order{order}] 『{hint}』を集計/計算から除外（kind={kind}）"
    return f"  - [order{order}] 『{hint}』（kind={kind}）"


def condition_directive(ir_spec: dict[str, Any] | None) -> str:
    """The pre-inject block that hands the loop a ready branch skeleton to *fill in*, not re-derive.

    Lists the detected predicate + ordered adjustment column and instructs the agent to (1) select
    ``base_quantity`` from the operand candidates, (2) confirm ``predicate_truth``, (3) fill each
    adjustment's value, then pass the ``condition`` through to ``verify_formula`` / the PoT lane — instead
    of re-interpreting the condition with more search. Injects no answer: the skeleton is *structure*, and
    every filled value is still checked by the PoT lane's branch-consistency + execution layers.
    """
    if not ir_spec or not ir_spec.get("adjustments"):
        return ""
    predicate = ir_spec.get("predicate", "")
    lines = "\n".join(_adjustment_line(a) for a in ir_spec["adjustments"])
    return (
        "【条件文の事前IR骨格（決定論抽出・SOT-2621）】この数値問いは what-if / 条件分岐を含みます。以下の"
        "分岐骨格を前提に、探索で条件を再解釈せず、空欄だけを埋めてください:\n"
        f"  predicate: {predicate}\n"
        f"  調整列(order順):\n{lines}\n"
        "埋める空欄: (1) base_quantity は operand 候補から『選ぶ』 / (2) predicate_truth（この分岐を取るか）"
        "を確定 / (3) 各調整の base/rate/delta の値を operand 候補で確定。"
        "`verify_formula` には `condition` として {predicate, predicate_truth, base_quantity, "
        "adjustments:[{kind,rate,order}]} を渡し、base_quantity を式が必ず参照するようにしてください。"
        "骨格に無い調整を勝手に足さない・条件を作り直さないでください。骨格が問いに合致しないときは"
        "無理に使わず、通常経路で最善の回答を返してください。")
