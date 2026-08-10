"""SOT-2584 — Evidence Packet: deterministic pre-answer document resolution + slot plan + budget.

Parent SOT-2568 deep-research comment (実装順 2/7, P0). The typed router (:mod:`query_router`) decides
*what kind* of answer a question needs; this module turns that decision, plus the SOT-2583 document
registry, into a single **Evidence Packet** JSON that is *pre-injected* into the generation agent before
its first turn::

    {
      "route": "NUMERIC",
      "resolved_documents": [{"doc_id": ..., "path": ..., "resolution_method": "alias",
                              "resolution_score": 0.9, "case_id": ...}],
      "required_evidence": ["operands", "unit", "operation", "rounding"],
      "evidence": {},                 # slots already filled deterministically (empty in this landing)
      "missing": ["operands", "unit", "operation", "rounding"],
      "tool_budget_remaining": 1      # free-exploration calls allowed for the MISSING slots only
    }

The packet inverts the loop's default: instead of "検索→考える→また検索" until BUDGET_EXHAUSTED, the agent
starts already knowing (a) which documents are the target — resolved deterministically by the registry,
not chunk-similarity — and (b) exactly which evidence slots remain, and is told to fill *only* those,
using the route's deterministic **primary lane** first and at most ``tool_budget_remaining`` free
exploration calls.

Design invariants (mirroring the sibling opt-ins — document_registry, first_move, unanswerable_fallback):
  * **Opt-in, byte-identical OFF.**  :func:`enabled` (``RAG_EVIDENCE_PACKET``) gates the whole thing; the
    default-OFF serve path never builds a packet and is byte-identical.
  * **No corpus fact / no answer injected.**  The packet carries document *identity* + *slot names* +
    the route's tool budget — never a value, never an answer.  The committed answer is still the model's,
    subject to every existing commit guard.
  * **Deterministic & network-free.**  Resolution is the registry's structural chain; slot planning is
    the router's static per-route table.  No LLM call to build the packet.
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from typing import Any

from src.rag.agent import condition_prefill as _condition_prefill
from src.rag.agent import early_abstain as _early
from src.rag.agent import operand_prefill as _operand_prefill
from src.rag.agent import query_router as _router
from src.rag.index import document_registry as _dr

_ON = {"1", "true", "yes", "on"}


def enabled() -> bool:
    """True when the serve path should build & pre-inject an Evidence Packet (default OFF — opt-in)."""
    return os.getenv("RAG_EVIDENCE_PACKET", "0").strip().lower() in _ON


# --------------------------------------------------------------------------- packet model
@dataclass(frozen=True)
class ResolvedDocument:
    """One document the registry resolved for the question, with its resolution provenance."""

    doc_id: str
    path: str
    resolution_method: str
    resolution_score: float
    case_id: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "doc_id": self.doc_id,
            "path": self.path,
            "resolution_method": self.resolution_method,
            "resolution_score": round(float(self.resolution_score), 4),
            "case_id": self.case_id,
        }


@dataclass(frozen=True)
class EvidencePacket:
    """The pre-injected Evidence Packet for one question (research JSON shape)."""

    route: str
    resolved_documents: tuple[ResolvedDocument, ...]
    required_evidence: tuple[str, ...]
    evidence: dict[str, Any] = field(default_factory=dict)
    missing: tuple[str, ...] = ()
    tool_budget_remaining: int = 1
    primary_lane: str | None = None
    contract: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "route": self.route,
            "resolved_documents": [d.to_dict() for d in self.resolved_documents],
            "required_evidence": list(self.required_evidence),
            "evidence": dict(self.evidence),
            "missing": list(self.missing),
            "tool_budget_remaining": self.tool_budget_remaining,
            "primary_lane": self.primary_lane,
            "contract": self.contract,
        }

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=False, sort_keys=True)


# --------------------------------------------------------------------------- packet construction
def _resolve_documents(question: str, *, project: str | None,
                       limit: int) -> tuple[ResolvedDocument, ...]:
    """Resolve the target documents via the registry's deterministic chain (empty when unavailable).

    Fail-open: when the registry artifact is absent, the resolver is ``None`` and no documents are
    resolved — the packet then carries an empty ``resolved_documents`` and the agent falls back to its
    ordinary retrieval, so a missing registry never breaks the serve path.
    """
    resolver = _dr.get_resolver()
    if resolver is None:
        return ()
    rows = resolver.resolve(question, project=project, limit=limit)
    resolver_by_id = getattr(resolver, "_by_id", {})
    out: list[ResolvedDocument] = []
    for r in rows:
        row = resolver_by_id.get(r.doc_id, {})
        out.append(ResolvedDocument(
            doc_id=r.doc_id,
            path=str(row.get("full_path", r.doc_id)),
            resolution_method=r.tier,
            resolution_score=r.score,
            case_id=r.case_id,
        ))
    return tuple(out)


def build_packet(question: str, *, project: str | None = None,
                 decision: "_router.RouteDecision | None" = None,
                 max_documents: int = 3) -> EvidencePacket:
    """Build the Evidence Packet for one question (route → slots → resolved docs → budget).

    Deterministic and network-free.  ``decision`` may be passed to avoid re-typing the question.  In this
    landing the ``evidence`` map starts empty (all slots ``missing``): document *resolution* is filled
    deterministically by the registry, and the per-slot hard-lane pre-population is the scope of the
    downstream numeric/enum/format issues (SOT-2585/2586/2587) which this router promotes to primary
    lanes.  The packet already pre-injects the resolved documents, the required slots and the typed budget
    so the agent stops burning turns re-discovering the corpus.
    """
    dec = decision if decision is not None else _router.classify_route(question)
    docs = _resolve_documents(question, project=project, limit=max_documents)
    required = dec.evidence_slots
    evidence: dict[str, Any] = {}
    # SOT-2616 — NUMERIC operand candidate prefill. Behind ``RAG_OPERAND_PREFILL`` (default OFF ⇒ evidence
    # stays ``{}`` and the packet JSON is byte-identical). When on for a NUMERIC route, enumerate numeric
    # operand candidates from the just-resolved documents' offline assets and record them as a catalog in
    # the ``operand_candidates`` evidence key. The required slots (incl. "operands") deliberately stay in
    # ``missing``: the catalog is *candidates*, and the LLM still selects which is base/rate/条件 — this
    # only adds pre-listed provenance, it never fills the answer. Empty catalog → nothing added (no
    # degradation). Guarded so a prefill hiccup never changes the packet's core fields.
    if _operand_prefill.enabled() and dec.route == _router.NUMERIC:
        try:
            catalog = _operand_prefill.build_catalog(
                question, [d.doc_id for d in docs], project=project)
            if catalog:
                evidence["operand_candidates"] = catalog
        except Exception:  # noqa: BLE001 — additive; never break the packet
            pass
    # SOT-2621 — NUMERIC what-if condition IR prefill. Behind ``RAG_CONDITION_PREIR`` (default OFF ⇒
    # evidence stays untouched and the packet JSON is byte-identical). When on for a NUMERIC route, detect
    # a what-if / 条件分岐 skeleton (predicate + ordered adjustments) deterministically from the question
    # and record it under the ``condition_ir`` evidence key. Only *fires* when an adjustment is detected
    # (no firing-condition relaxation): a question with no branch structure injects nothing → no
    # degradation. The skeleton is a hint (base_quantity / predicate_truth left blank for the LLM); it is
    # not a required slot, so "operands"/etc. stay in ``missing``. Guarded so a detection hiccup never
    # changes the packet's core fields.
    if _condition_prefill.enabled() and dec.route == _router.NUMERIC:
        try:
            ir_spec = _condition_prefill.build_condition_ir(question)
            if ir_spec:
                evidence["condition_ir"] = ir_spec
        except Exception:  # noqa: BLE001 — additive; never break the packet
            pass
    missing = tuple(slot for slot in required if slot not in evidence)
    return EvidencePacket(
        route=dec.route,
        resolved_documents=docs,
        required_evidence=required,
        evidence=evidence,
        missing=missing,
        tool_budget_remaining=dec.budget.tool_budget_remaining,
        primary_lane=dec.primary_lane,
        contract=dec.contract,
    )


# --------------------------------------------------------------------------- pre-inject directive
def packet_directive(packet: EvidencePacket) -> str:
    """The user-message that pre-injects the packet into the generation agent before its first turn.

    Carries the resolved documents (so the agent starts at the target, not chunk search), the required
    evidence slots and which are still missing, the deterministic primary lane to use first, and the
    free-exploration budget — capping free search to the *missing* slots only (research: "自由探索は
    missing slot のみ最大1〜2回").  Injects no corpus fact and no answer.
    """
    docs = packet.resolved_documents
    if docs:
        doc_lines = "\n".join(
            f"  - {d.path}（doc_id={d.doc_id} / 解決={d.resolution_method} score={d.resolution_score:.2f}"
            + (f" / 案件={d.case_id}" if d.case_id else "") + "）"
            for d in docs[:3])
        doc_block = f"回答ループ前に対象文書を決定論的に解決済み（chunk 検索は省略可）:\n{doc_lines}\n"
    else:
        doc_block = ("対象文書はレジストリで一意に解決できませんでした。まず find_files / canonical_route で"
                     "対象を特定してください。\n")

    lane = packet.primary_lane
    lane_block = (
        f"この質問型（{packet.route}）の primary lane は決定論ツール『{lane}』です。まずこれを必ず呼び、"
        "自由な chunk 検索より優先してください。\n"
        if lane else
        f"この質問型（{packet.route}）は retrieval レーンです。対象文書の該当箇所を読み取ってください。\n")

    missing = "、".join(packet.missing) if packet.missing else "（なし）"
    budget = packet.tool_budget_remaining
    slot_block = (
        f"回答確定に必要な証拠スロット: {', '.join(packet.required_evidence)}\n"
        f"未充足スロット(missing): {missing}\n"
        f"自由探索の残予算(tool_budget_remaining)= {budget}。未充足スロット『だけ』を、primary lane→"
        f"（不足時のみ最大{budget}回の追加探索）の順に埋めてください。全スロットが埋まったら余分な探索を"
        "せず submit_answer で確定してください。\n"
        "予算切れ(BUDGET_EXHAUSTED)は『答えられない』ではなく制御上の失敗です。予算内でスロットを埋め、"
        "根拠が本当に存在しない場合のみ棄権してください。")
    # SOT-2599 — relax the *原則追加探索なし* budget contract to *安価な決定論探索は許す* (free chunk search
    # is still not grown). Byte-identical when the recalibration is OFF: the clause is only appended then.
    if _early.enabled():
        slot_block = f"{slot_block}\n{_early.packet_relaxation_clause()}"
    # SOT-2616 — when the NUMERIC operand prefill recorded a candidate catalog, render it as a selection
    # directive so the agent picks operands from the list (→ verify_formula catalog/select handoff) instead
    # of grep-spinning. Absent catalog (default OFF / empty) appends nothing → byte-identical.
    catalog = packet.evidence.get("operand_candidates") if packet.evidence else None
    operand_block = _operand_prefill.candidates_directive(catalog) if catalog else ""
    # SOT-2621 — when the NUMERIC condition prefill recorded a branch IR skeleton, render it as a
    # fill-in-the-blanks directive so the agent uses the pre-built branch structure instead of
    # re-interpreting the condition through more search. Absent skeleton (default OFF / not fired) appends
    # nothing → byte-identical.
    ir_spec = packet.evidence.get("condition_ir") if packet.evidence else None
    condition_block = _condition_prefill.condition_directive(ir_spec) if ir_spec else ""
    packet_json = packet.to_json()
    directive = (
        "【Evidence Packet（回答ループ前・決定論フェーズ）】\n"
        + doc_block + lane_block + slot_block)
    if operand_block:
        directive = f"{directive}\n{operand_block}"
    if condition_block:
        directive = f"{directive}\n{condition_block}"
    return f"{directive}\nEvidence Packet(JSON): {packet_json}"


def build_directive(question: str, *, project: str | None = None,
                    decision: "_router.RouteDecision | None" = None) -> "tuple[EvidencePacket, str]":
    """Convenience: build the packet and its pre-inject directive together."""
    packet = build_packet(question, project=project, decision=decision)
    return packet, packet_directive(packet)
