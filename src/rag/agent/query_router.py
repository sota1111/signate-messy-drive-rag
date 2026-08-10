"""SOT-2584 — typed query router (7 routes) + per-route evidence slots & budget contracts.

Parent SOT-2568 deep-research comment (実装順 2/7, P0). The research inverts the current
"検索→考える→また検索" free exploration into a **回答ループ前の決定論フェーズ**: the question is first
typed, its documents are resolved (SOT-2583 registry), the required evidence is planned as *slots*, and
only then is the generation agent run — ideally once, free-exploring only the still-*missing* slots.

This module is the **typing + planning** half of that phase. It is a thin, deterministic *routing view*
on top of the mature 9-contract classifier (:mod:`src.rag.agent.question_contract`) — it does not
re-implement classification. It folds the 9 contracts (plus a new EXISTENCE detector and the existing
pivot detector) into the research's **6+1 routes**::

    LOOKUP  ENUM  NUMERIC  VERSION_DIFF  FORMAT  PIVOT      (6 typed hard-lane routes)
    EXISTENCE                                                (+1: 存在/不存在 with exhaustive coverage)

and attaches, per route, the **evidence slots** that must be filled before generation
(research §"探索律速への対策" table) and a **budget contract** (per-type turn/fallback budget, replacing
the flat 12-turn limit — research §Budget-Aware). It also names the deterministic **primary lane** the
router should dispatch to (promoting compute/enumeration/diffpair/pivotcond from "tools the agent might
stumble onto" to "lanes the router always calls").

Pure & deterministic: no LLM call, no network, no I/O — every function is a total map over the question
text and the classifier output, so the route/slots/budget are reproducible and the accuracy is
measurable offline (:func:`route_agreement`).
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Callable, Sequence

from src.rag import pivotcond as _pivotcond
from src.rag.agent import question_contract as _qc
from src.rag.corpus import nfc

# --------------------------------------------------------------------------- the 6+1 routes
LOOKUP = "LOOKUP"
ENUM = "ENUM"
NUMERIC = "NUMERIC"
VERSION_DIFF = "VERSION_DIFF"
FORMAT = "FORMAT"
PIVOT = "PIVOT"
EXISTENCE = "EXISTENCE"

ROUTES: tuple[str, ...] = (LOOKUP, ENUM, NUMERIC, VERSION_DIFF, FORMAT, PIVOT, EXISTENCE)

ROUTE_LABELS: dict[str, str] = {
    LOOKUP: "単純検索", ENUM: "完全列挙", NUMERIC: "数値導出", VERSION_DIFF: "版差分",
    FORMAT: "書式判定", PIVOT: "ピボット", EXISTENCE: "存在判定",
}

# --------------------------------------------------------------------------- contract → base route
# The 9 routing contracts fold into the 6+1 routes. spatial/chart_read/multi_hop are specialised
# lookups (they read one value from one resolved target); cross_aggregate is a numeric aggregation.
_CONTRACT_ROUTE: dict[str, str] = {
    _qc.SIMPLE_LOOKUP: LOOKUP,
    _qc.MULTI_HOP: LOOKUP,
    _qc.SPATIAL: LOOKUP,
    _qc.CHART_READ: LOOKUP,
    _qc.CROSS_AGGREGATE: NUMERIC,
    _qc.NUMERIC: NUMERIC,
    _qc.FULL_ENUMERATION: ENUM,
    _qc.VERSION_DIFF: VERSION_DIFF,
    _qc.FORMAT_CHECK: FORMAT,
}

# The deterministic primary lane per contract — the tool the router should dispatch to (a hard,
# corpus-fact-free lane) instead of letting the agent wander through chunk search. ``None`` = the route
# has no deterministic compute/extract lane (a genuine retrieval lookup / existence scan).
_CONTRACT_PRIMARY_LANE: dict[str, str | None] = {
    _qc.SIMPLE_LOOKUP: None,
    _qc.MULTI_HOP: None,
    _qc.SPATIAL: "seating_lookup",
    _qc.CHART_READ: "read_chart_values",
    _qc.CROSS_AGGREGATE: "corpus_aggregate",
    _qc.NUMERIC: "compute",
    _qc.FULL_ENUMERATION: "corpus_aggregate",
    _qc.VERSION_DIFF: "version_diff",
    _qc.FORMAT_CHECK: "highlight_extract",
}

# Deterministic (hard) lanes — a primary lane in this set counts toward hard_lane_invocation_rate.
_HARD_LANES: frozenset[str] = frozenset({
    "compute", "corpus_aggregate", "version_diff", "highlight_extract", "font_emphasis",
    "read_chart_values", "seating_lookup", "pptx_pivot",
})

# --------------------------------------------------------------------------- per-route evidence slots
# The evidence a route must fill before the generation agent may commit (research table).  These are the
# *slot names*, not corpus facts — the packet builder records which are still missing.
ROUTE_SLOTS: dict[str, tuple[str, ...]] = {
    LOOKUP: ("canonical_doc", "answer_span"),
    NUMERIC: ("operands", "unit", "operation", "rounding"),
    ENUM: ("universe", "filter_predicate", "scan_completion"),
    VERSION_DIFF: ("version_pair", "aligned_block", "substantive_change"),
    FORMAT: ("target_cell_or_run", "effective_style", "style_provenance"),
    PIVOT: ("pivot_identity", "field_item_value"),
    EXISTENCE: ("exhaustive_search_coverage", "parser_support"),
}

# --------------------------------------------------------------------------- per-route budget contract
@dataclass(frozen=True)
class BudgetContract:
    """The per-route turn/fallback budget (replaces the flat 12-turn cap).

    ``retrieval_calls`` = the deterministic/primary-lane invocations the router does up front;
    ``fallback_calls`` = the free-exploration calls the generation agent is allowed for *missing* slots
    only (the research's "自由探索は missing slot のみ最大1〜2回").  ``tool_budget_remaining`` is exactly
    ``fallback_calls`` — how much free exploration the packet grants the agent.
    """

    retrieval_calls: int
    fallback_calls: int
    hard_compute: bool = False   # the route runs a deterministic hard lane (compute/scan/diff/extract)

    @property
    def tool_budget_remaining(self) -> int:
        return self.fallback_calls


# Research §"探索律速への対策" budget contracts, per route.
ROUTE_BUDGET: dict[str, BudgetContract] = {
    LOOKUP: BudgetContract(retrieval_calls=1, fallback_calls=1),
    NUMERIC: BudgetContract(retrieval_calls=1, fallback_calls=1, hard_compute=True),
    ENUM: BudgetContract(retrieval_calls=1, fallback_calls=0, hard_compute=True),
    VERSION_DIFF: BudgetContract(retrieval_calls=1, fallback_calls=1, hard_compute=True),
    FORMAT: BudgetContract(retrieval_calls=1, fallback_calls=1, hard_compute=True),
    PIVOT: BudgetContract(retrieval_calls=1, fallback_calls=1, hard_compute=True),
    EXISTENCE: BudgetContract(retrieval_calls=1, fallback_calls=0, hard_compute=True),
}

# --------------------------------------------------------------------------- EXISTENCE detection
# A question asking *whether* something exists / is present / is stated (not its value).  Its answer is a
# yes/no + (if present) the located record, and its failure mode is idx16-style "見えていない=存在しない":
# so its slots are exhaustive coverage + parser support, and a 0-fallback exhaustive-scan budget.
_EXISTENCE_RE = re.compile(
    r"(?:は)?(?:存在|該当)(?:する|します|しますか|するか|は)?(?:でしょうか|か|？|\?)?|"
    r"あります(?:か|でしょうか)|ありますか|ある(?:か|でしょうか|のか)|"
    r"含まれ(?:て|る|ている)(?:いますか|ますか|るか)|"
    r"記載(?:は)?(?:あります|ありますか|されて|されていますか)|"
    r"設定(?:されて)?(?:いますか|ありますか)|有無")
# But a "list ALL that exist" question is an enumeration, and a "how many exist" is numeric — the
# existence route is only the yes/no/located-record shape, so enumeration/quantity cues suppress it.
_EXISTENCE_SUPPRESS_RE = re.compile(r"すべて|全て|全部|一覧|列挙|挙げて|何(?:件|人|個|種|社|つ)|いくつ")


def is_existence_question(question: str) -> bool:
    """Whether a question asks *whether* something exists/is stated (yes/no), not its value or a list."""
    q = nfc(question or "")
    return bool(_EXISTENCE_RE.search(q)) and not _EXISTENCE_SUPPRESS_RE.search(q)


# --------------------------------------------------------------------------- the routing decision
@dataclass(frozen=True)
class RouteDecision:
    """The typed route for one question plus its evidence slots, budget contract and primary lane."""

    route: str
    label: str
    contract: str                     # underlying 9-contract classification (preserved for fine routing)
    archetype: str
    evidence_slots: tuple[str, ...]
    budget: BudgetContract
    primary_lane: str | None          # the deterministic hard lane to dispatch to, or None (retrieval)
    method: str                       # "deterministic" | "flash"
    confidence: float
    evidence: str = ""                # short why-note

    @property
    def has_hard_lane(self) -> bool:
        """True when the route dispatches to a deterministic compute/extract/scan lane."""
        return self.budget.hard_compute or (self.primary_lane in _HARD_LANES)

    def to_dict(self) -> dict[str, object]:
        return {
            "route": self.route,
            "label": self.label,
            "contract": self.contract,
            "archetype": self.archetype,
            "evidence_slots": list(self.evidence_slots),
            "budget": {
                "retrieval_calls": self.budget.retrieval_calls,
                "fallback_calls": self.budget.fallback_calls,
                "hard_compute": self.budget.hard_compute,
                "tool_budget_remaining": self.budget.tool_budget_remaining,
            },
            "primary_lane": self.primary_lane,
            "has_hard_lane": self.has_hard_lane,
            "method": self.method,
            "confidence": self.confidence,
            "evidence": self.evidence,
        }


def route_for_contract(contract: str, question: str) -> str:
    """Fold a 9-contract label + question cues into one of the 6+1 routes.

    Precedence: PIVOT (a pivot-condition question) → EXISTENCE (a yes/no existence question on a lookup
    base) → the contract's base route.  Pivot wins over its ``format_check``/``cross_aggregate`` base
    because a pivot has its own deterministic lane; existence refines only a lookup base (a "list all"
    or "how many" question keeps ENUM/NUMERIC).
    """
    base = _CONTRACT_ROUTE.get(contract, LOOKUP)
    if _pivotcond.is_pivot_condition_question(question):
        return PIVOT
    if base == LOOKUP and is_existence_question(question):
        return EXISTENCE
    return base


def classify_route(question: str, *,
                   contract: "_qc.QuestionContract | None" = None,
                   flash: Callable[[str], str | None] | None = None) -> RouteDecision:
    """Type ``question`` into its :class:`RouteDecision` (slots + budget + primary lane).

    Reuses the deterministic-first 9-contract classifier (pass ``contract`` to avoid re-classifying, or
    ``flash`` to enable the arbiter on ambiguity — omit both for a fully offline, network-free typing).
    """
    qc = contract if contract is not None else _qc.classify(question, flash=flash)
    route = route_for_contract(qc.contract, question)
    # The primary lane is a pivot/existence override's own lane, else the contract's.
    if route == PIVOT:
        primary_lane: str | None = "pptx_pivot"
    elif route == EXISTENCE:
        primary_lane = "file_grep"        # exhaustive manifest/corpus scan (coverage), not a value lane
    else:
        primary_lane = _CONTRACT_PRIMARY_LANE.get(qc.contract)
    return RouteDecision(
        route=route,
        label=ROUTE_LABELS[route],
        contract=qc.contract,
        archetype=qc.archetype,
        evidence_slots=ROUTE_SLOTS[route],
        budget=ROUTE_BUDGET[route],
        primary_lane=primary_lane,
        method=qc.method,
        confidence=qc.confidence,
        evidence=f"contract={qc.contract}; {qc.evidence}",
    )


# --------------------------------------------------------------------------- offline route accuracy
def route_from_archetype(archetype: str) -> str:
    """The route implied by a gold *archetype* label (via the contract backbone) — the accuracy target."""
    base = _qc._base_contract((archetype or "unknown").strip() or "unknown")
    return _CONTRACT_ROUTE.get(base, LOOKUP)


# Routes that are *refinements* of a coarser gold route (a legitimate specificity gain, not an error):
# PIVOT/FORMAT/EXISTENCE all refine a generic LOOKUP/NUMERIC/ENUM base the archetype cannot express.
_REFINEMENT_ROUTES: frozenset[str] = frozenset({PIVOT, FORMAT, EXISTENCE})
# The coarse routes each refinement may legitimately specialise.
_REFINES: dict[str, frozenset[str]] = {
    PIVOT: frozenset({LOOKUP, NUMERIC, FORMAT}),
    FORMAT: frozenset({LOOKUP}),
    EXISTENCE: frozenset({LOOKUP, ENUM}),
}


@dataclass(frozen=True)
class RouteAgreement:
    """Result of scoring the router against gold archetype labels (offline diagnostic)."""

    total: int
    agree: int
    rate: float
    distribution: dict[str, int]
    hard_lane_rate: float
    refinements: tuple[dict[str, str], ...]
    mismatches: tuple[dict[str, str], ...]

    def to_dict(self) -> dict[str, object]:
        return {
            "total": self.total, "agree": self.agree, "rate": self.rate,
            "distribution": self.distribution, "hard_lane_rate": self.hard_lane_rate,
            "refinements": list(self.refinements), "mismatches": list(self.mismatches),
        }


def _route_consistent(predicted: str, gold_route: str) -> bool:
    """True when ``predicted`` matches ``gold_route`` or legitimately refines it."""
    if predicted == gold_route:
        return True
    if predicted in _REFINEMENT_ROUTES and gold_route in _REFINES.get(predicted, frozenset()):
        return True
    return False


def route_agreement(rows: Sequence[dict[str, str]], *,
                    flash: Callable[[str], str | None] | None = None) -> RouteAgreement:
    """Score the router vs. the gold ``archetype`` column (route_accuracy) + record diagnostics.

    Each row must carry ``question`` and ``archetype``. A row agrees when the predicted route matches the
    archetype-implied route or legitimately refines it (PIVOT/FORMAT/EXISTENCE over a coarse base). Also
    returns the route distribution and hard_lane_invocation_rate — the diagnostic metrics SOT-2584 asks
    to record on gold100.
    """
    agree = 0
    distribution = {r: 0 for r in ROUTES}
    hard_lane = 0
    refinements: list[dict[str, str]] = []
    mismatches: list[dict[str, str]] = []
    for row in rows:
        q = row.get("question", "")
        gold = (row.get("archetype", "") or "unknown").strip() or "unknown"
        decision = classify_route(q, flash=flash)
        distribution[decision.route] += 1
        if decision.has_hard_lane:
            hard_lane += 1
        gold_route = route_from_archetype(gold)
        if _route_consistent(decision.route, gold_route):
            agree += 1
            if decision.route in _REFINEMENT_ROUTES and decision.route != gold_route:
                refinements.append({"question": q, "gold_archetype": gold,
                                    "gold_route": gold_route, "route": decision.route})
        else:
            mismatches.append({"question": q, "gold_archetype": gold,
                               "gold_route": gold_route, "route": decision.route})
    total = len(rows)
    return RouteAgreement(
        total=total, agree=agree, rate=(agree / total if total else 1.0),
        distribution=distribution,
        hard_lane_rate=(hard_lane / total if total else 0.0),
        refinements=tuple(refinements), mismatches=tuple(mismatches),
    )
