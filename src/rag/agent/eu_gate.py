"""SOT-2589 — expected-utility three-tier abstain gate (HARD / SOFT / ABSTAIN).

Parent SOT-2568 deep-research comment (実装順 7/7, P2 — the *final* re-calibration). The historical
:mod:`src.rag.agent.gate` is a **confidence-threshold** gate: it commits a 合議一致 answer only when the
model's self-reported ``confidence`` clears :data:`~src.rag.agent.gate.GATE_COMMIT_CONFIDENCE` (0.7). That
conflates *verbal* confidence with actual precision, and the gold-100 loss analysis showed two problems:

* **Verbal confidence is not faithful to precision** (research: RiskEval "Are LLM Decisions Faithful to
  Verbal Confidence?" arXiv:2601.07767 / SABER "Trust or Abstain?" arXiv:2605.18792). A high self-report
  does not imply the answer is right, and a low one does not imply it is wrong.
* **BUDGET_EXHAUSTED was being treated as an abstain criterion** even though it is an *operational* control
  failure (the agent ran out of turns *searching*), not an *epistemic* "the corpus does not contain the
  answer" (see :mod:`src.rag.agent.failure_taxonomy`).

This module re-frames the decision as an **expected-utility** one, aligned exactly to the official metric
(Perfect +1 / Acceptable +0.5 / Missing 0 / Incorrect −1). For a committed answer with grade probabilities
``P(P) / P(A) / P(I)`` the expected score is::

    U = P(P) + 0.5·P(A) − P(I)

and **committing beats abstaining (Missing = 0) exactly when U > 0**. The gate estimates ``P(P)/P(A)/P(I)``
from a signal bundle in which the **hard evidence signals** (the deterministic assets SOT-2583〜2588 emit:
canonical-doc resolution, parser capability, evidence-slot completeness, universe scan, version-pair
resolution, operand/execution agreement, source-conflict absence) are the **primary** input, retrieval
quality is secondary, and **verbal confidence is demoted to a small auxiliary weight**.

Three tiers (research comment §"三段 gate"):

* **HARD ACCEPT** — a deterministic lane produced the answer, its required evidence slots are complete, and
  the answer verifier agrees. Highest precision; commit unconditionally (U is computed for logging only).
* **SOFT ACCEPT** — no hard blocker and the estimated ``U > 0``. Commit on positive expected utility.
* **ABSTAIN** — a *hard* epistemic blocker (true corpus absence / parser unsupported / an unresolved
  *required* evidence slot / conflicting evidence / execution disagreement), OR ``U ≤ 0``. ``わかりません``.

**BUDGET_EXHAUSTED never appears in the decision** — it is an operational error, not an abstain reason
(the whole point of SOT-2583/2584). It is carried on the bundle for diagnostics only and is *ignored* by
:func:`decide`.

Design invariants (mirroring the sibling opt-ins — document_registry / evidence_packet / pot_lane /
diff_align):

* **Opt-in, byte-identical OFF.** :func:`enabled` (``RAG_EU_GATE``) gates the whole thing; the default-OFF
  serve path never constructs a bundle nor calls :func:`decide`, so the answer path is byte-identical.
* **Pure & network-free.** :func:`decide` / :func:`expected_utility` are pure functions over a
  :class:`GateSignals` bundle — trivially testable, no Vertex, no I/O. The probability mapping is an
  interpretable, hand-set model, **not** a fitted one: local proxy ↔ real LB is ρ=−0.09, so calibration is
  used only to *diagnose* whether an imagined failure mode was removed, never to claim a promotion.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any

_ON = {"1", "true", "yes", "on"}


def enabled() -> bool:
    """True when the serve path should apply the expected-utility gate (default OFF — opt-in)."""
    return os.getenv("RAG_EU_GATE", "0").strip().lower() in _ON


def _env_float(key: str, default: float) -> float:
    """Read a float calibration knob from ``key``; fail-open to ``default`` on any parse error."""
    raw = os.getenv(key)
    if raw is None or not raw.strip():
        return default
    try:
        return float(raw.strip())
    except (TypeError, ValueError):
        return default


def commit_threshold() -> float:
    """SOT-2635 — the expected-utility commit bar τ (commit iff ``U > τ``).

    Env-tunable via ``RAG_EU_GATE_TAU`` (default **0.0** — the original ``U > 0`` decision, byte-identical
    to SOT-2589). A *positive* τ makes the SOFT-ACCEPT tier stricter (commit only clearly-EV-positive
    answers → more precision, fewer commits); a *negative* τ is used only as a **telemetry probe** (commit
    everything, record ``U`` per question) to calibrate τ offline without over-abstaining. This is the sole
    calibration knob the paired answer-increasing path is tuned against — the probability *weights* below
    stay the hand-set (non-fitted) model, since local proxy ↔ real LB is ρ=−0.09."""
    return _env_float("RAG_EU_GATE_TAU", 0.0)


def base_prior() -> float:
    """SOT-2635 — the base correctness prior (env-tunable via ``RAG_EU_GATE_BASE``, default 0.30).

    Deliberately low so an answered-yet-unsupported bundle sits below the commit bar — committing must be
    *earned* by positive evidence, not defaulted into. Exposed for calibration only; the default keeps the
    SOT-2589 pure-function behaviour byte-identical."""
    return _env_float("RAG_EU_GATE_BASE", _C_BASE)


# --------------------------------------------------------------------------- tiers
HARD_ACCEPT = "HARD_ACCEPT"
SOFT_ACCEPT = "SOFT_ACCEPT"
ABSTAIN = "ABSTAIN"

# The official scoring the utility is aligned to (Perfect / Acceptable / Missing / Incorrect).
SCORE_PERFECT = 1.0
SCORE_ACCEPTABLE = 0.5
SCORE_MISSING = 0.0
SCORE_INCORRECT = -1.0


def _clamp01(x: float) -> float:
    try:
        v = float(x)
    except (TypeError, ValueError):
        return 0.0
    if v < 0.0:
        return 0.0
    if v > 1.0:
        return 1.0
    return v


# --------------------------------------------------------------------------- signal bundle
@dataclass(frozen=True)
class GateSignals:
    """The typed input to the expected-utility gate for one question.

    Signals are grouped exactly as the research comment prescribes. **Hard evidence signals** are the
    primary input; retrieval signals are secondary; the model signals (verbal confidence especially) are
    *auxiliary*. The ten hard evidence signals split by role:

    * **Positive-evidence** signals (``canonical_doc_resolved`` / ``evidence_slots_complete`` /
      ``self_consistency_agrees`` / ``operand_sources_complete`` / ``universe_exhaustively_scanned`` /
      ``version_pair_resolved``) **default False** = "not obtained" and *raise* the correctness estimate
      only when the caller confirms them. A signal that is simply *not applicable* to a question type
      (e.g. ``version_pair_resolved`` for a plain lookup) stays False and contributes nothing — it never
      inflates the estimate. This is what keeps a **bare** bundle below the commit bar.
    * **Blocker** signals (``parser_capable`` / ``execution_engines_agree`` / ``source_conflict_absent`` /
      ``explicit_file_constraint_satisfied``) **default True** = "no problem". They do *not* add to the
      correctness estimate; they force a hard ABSTAIN when in their bad state (a violated file constraint
      instead applies a penalty). Set them False only with positive evidence of the failure.
    """

    # --- hard evidence signals — POSITIVE EVIDENCE (default False = not obtained; raise correctness) ---
    canonical_doc_resolved: bool = False           # SOT-2583 registry resolved the target document
    evidence_slots_complete: bool = False          # SOT-2584 evidence packet slots all filled
    self_consistency_agrees: bool = False          # N-sample / investigator↔verifier self-consistency
    operand_sources_complete: bool = False         # SOT-2586 every operand bound to a source (numeric)
    universe_exhaustively_scanned: bool = False    # SOT-2587 enum universe fully scanned (enum only)
    version_pair_resolved: bool = False            # SOT-2588 version pair resolved (version_diff only)
    # --- hard evidence signals — BLOCKERS (default True = no problem; gate ABSTAIN, do not reward) ---
    parser_capable: bool = True                    # the extractor can read the required evidence
    execution_engines_agree: bool = True           # SOT-2586 Decimal ↔ SymPy re-execution agree (numeric)
    source_conflict_absent: bool = True            # no contradicting sources for the answer
    explicit_file_constraint_satisfied: bool = True  # a named-file constraint in the question is honoured

    # --- retrieval signals (SECONDARY) ---
    top1_score: float = 0.0                        # retriever top-1 similarity
    top1_top2_margin: float = 0.0                  # separation between the top two candidates
    dense_sparse_agreement: float = 0.0            # dense ↔ sparse retriever agreement
    source_diversity: float = 0.0                  # independent sources supporting the answer

    # --- model signals (AUXILIARY; verbal confidence DEMOTED) ---
    answer_verifier_agrees: bool = False           # the heterogeneous answer verifier confirmed
    verbal_confidence: float = 0.0                 # the model's self-reported confidence (demoted)

    # --- hard epistemic blockers (set True only with positive evidence of the failure) ---
    corpus_absent: bool = False                    # exhaustively scanned; the answer truly is not present
    required_evidence_unresolved: bool = False     # a *required* evidence slot is categorically missing

    # --- lane / operational context ---
    deterministic_lane: bool = False               # the answer came from a deterministic primary lane
    budget_exhausted: bool = False                 # OPERATIONAL — never an abstain criterion (diag only)

    def to_dict(self) -> dict[str, Any]:
        return {
            "canonical_doc_resolved": self.canonical_doc_resolved,
            "explicit_file_constraint_satisfied": self.explicit_file_constraint_satisfied,
            "parser_capable": self.parser_capable,
            "evidence_slots_complete": self.evidence_slots_complete,
            "universe_exhaustively_scanned": self.universe_exhaustively_scanned,
            "version_pair_resolved": self.version_pair_resolved,
            "operand_sources_complete": self.operand_sources_complete,
            "execution_engines_agree": self.execution_engines_agree,
            "self_consistency_agrees": self.self_consistency_agrees,
            "source_conflict_absent": self.source_conflict_absent,
            "top1_score": round(float(self.top1_score), 4),
            "top1_top2_margin": round(float(self.top1_top2_margin), 4),
            "dense_sparse_agreement": round(float(self.dense_sparse_agreement), 4),
            "source_diversity": round(float(self.source_diversity), 4),
            "answer_verifier_agrees": self.answer_verifier_agrees,
            "verbal_confidence": round(float(self.verbal_confidence), 4),
            "corpus_absent": self.corpus_absent,
            "required_evidence_unresolved": self.required_evidence_unresolved,
            "deterministic_lane": self.deterministic_lane,
            "budget_exhausted": self.budget_exhausted,
        }


# --------------------------------------------------------------------------- probability model
# Interpretable, hand-set weights — NOT fitted (local proxy ↔ LB ρ=−0.09; a fitted model would overfit
# noise). Hard *positive-evidence* signals dominate; retrieval is secondary; verbal confidence is a small
# auxiliary term. The base prior is deliberately low so an answered-yet-unsupported question (a bare
# bundle) sits *below* the U>0 bar — committing must be *earned* by positive evidence, not defaulted into.
_C_BASE = 0.30

# hard positive-evidence contributions (only added when the caller confirms the signal).
_W_CANONICAL_DOC = 0.15        # target document resolved deterministically (SOT-2583)
_W_EVIDENCE_SLOTS = 0.15       # every required evidence slot filled (SOT-2584)
_W_SELF_CONSISTENCY = 0.08     # N-sample / investigator↔verifier self-consistency
_W_OPERAND_SOURCES = 0.05      # every numeric operand bound to a source (SOT-2586)
_W_UNIVERSE_SCANNED = 0.05     # enum universe exhaustively scanned (SOT-2587)
_W_VERSION_PAIR = 0.05         # version pair resolved (SOT-2588)

# answer verifier — a model signal but a strong, independent one.
_W_VERIFIER = 0.12

# a *violated* explicit file constraint is positive evidence the wrong document was used → penalty.
_P_FILE_CONSTRAINT_VIOLATED = 0.20

# retrieval contributions (secondary; scaled by the [0,1] signal magnitude).
_W_TOP1 = 0.10
_W_MARGIN = 0.08
_W_DENSE_SPARSE = 0.05
_W_DIVERSITY = 0.03

# verbal confidence — DEMOTED: a small auxiliary weight (≤0.05), far below the hard-evidence mass.
_W_VERBAL_CONFIDENCE = 0.05


def correctness_prob(s: GateSignals) -> float:
    """Estimate ``P(answer is correct)`` ∈ [0, 1] from the signal bundle.

    Only **positive-evidence** signals raise the estimate (each defaults False, so an unconfirmed signal
    adds nothing); the blocker signals do not add here (they gate a hard ABSTAIN in :func:`decide`), except
    a *violated* file constraint which subtracts a penalty. Retrieval is secondary; verbal confidence is a
    small auxiliary term (``_W_VERBAL_CONFIDENCE`` ≤ 0.05) — never the primary driver (faithfulness caveat).
    """
    c = base_prior()
    # hard positive evidence (primary mass)
    if s.canonical_doc_resolved:
        c += _W_CANONICAL_DOC
    if s.evidence_slots_complete:
        c += _W_EVIDENCE_SLOTS
    if s.self_consistency_agrees:
        c += _W_SELF_CONSISTENCY
    if s.operand_sources_complete:
        c += _W_OPERAND_SOURCES
    if s.universe_exhaustively_scanned:
        c += _W_UNIVERSE_SCANNED
    if s.version_pair_resolved:
        c += _W_VERSION_PAIR
    if s.answer_verifier_agrees:
        c += _W_VERIFIER
    if not s.explicit_file_constraint_satisfied:
        c -= _P_FILE_CONSTRAINT_VIOLATED
    # retrieval (secondary)
    c += _W_TOP1 * _clamp01(s.top1_score)
    c += _W_MARGIN * _clamp01(s.top1_top2_margin)
    c += _W_DENSE_SPARSE * _clamp01(s.dense_sparse_agreement)
    c += _W_DIVERSITY * _clamp01(s.source_diversity)
    # verbal confidence — auxiliary, small weight (never the primary driver)
    c += _W_VERBAL_CONFIDENCE * _clamp01(s.verbal_confidence)
    return _clamp01(c)


def perfect_share(s: GateSignals) -> float:
    """Among correct answers, the share graded **Perfect** (vs merely **Acceptable**) ∈ [0, 1].

    A deterministic lane with a confirming verifier and complete evidence slots produces exact,
    fully-graded answers (Perfect); a soft retrieval answer is likelier to be only Acceptable. This only
    shifts the Perfect↔Acceptable split within the correct mass; it does not change ``correctness_prob``.
    """
    ps = 0.5
    if s.deterministic_lane:
        ps += 0.30
    if s.answer_verifier_agrees:
        ps += 0.10
    if s.evidence_slots_complete:
        ps += 0.10
    return _clamp01(ps)


def grade_probabilities(s: GateSignals) -> tuple[float, float, float]:
    """``(P(Perfect), P(Acceptable), P(Incorrect))`` for committing this answer.

    ``P(correct) = correctness_prob``; the correct mass is split into Perfect/Acceptable by
    :func:`perfect_share`; the remainder is Incorrect. (Committing yields a graded answer, so the three sum
    to 1 — Missing only arises from *abstaining*, which is the U=0 alternative this is compared against.)
    """
    c = correctness_prob(s)
    ps = perfect_share(s)
    p_perfect = c * ps
    p_acceptable = c * (1.0 - ps)
    p_incorrect = 1.0 - c
    return p_perfect, p_acceptable, p_incorrect


def expected_utility(s: GateSignals) -> float:
    """The expected score of **committing**: ``U = P(P) + 0.5·P(A) − P(I)`` (Missing = 0 to abstain)."""
    p_perfect, p_acceptable, p_incorrect = grade_probabilities(s)
    return (SCORE_PERFECT * p_perfect
            + SCORE_ACCEPTABLE * p_acceptable
            + SCORE_INCORRECT * p_incorrect)


# --------------------------------------------------------------------------- hard blockers
def hard_abstain_reason(s: GateSignals) -> str | None:
    """The epistemic blocker that forces ABSTAIN regardless of ``U``, or ``None`` when none applies.

    These are the categorical impossibilities from the research comment's ABSTAIN definition — *true*
    corpus absence, an unsupported parser, an unresolved *required* evidence slot, conflicting evidence, or
    an execution disagreement. **BUDGET_EXHAUSTED is deliberately not here**: it is operational, not
    epistemic, so a budget-exhausted question with otherwise-positive utility is *not* forced to abstain.
    """
    if s.corpus_absent:
        return "true_corpus_absence"
    if not s.parser_capable:
        return "parser_unsupported"
    if s.required_evidence_unresolved:
        return "unresolved_required_evidence"
    if not s.source_conflict_absent:
        return "conflicting_evidence"
    if not s.execution_engines_agree:
        return "execution_disagreement"
    return None


# --------------------------------------------------------------------------- decision
@dataclass(frozen=True)
class EUDecision:
    """The commit/abstain call for one question under the expected-utility three-tier gate."""

    commit: bool
    tier: str                 # HARD_ACCEPT | SOFT_ACCEPT | ABSTAIN
    utility: float            # U = P(P)+0.5P(A)−P(I) (computed even when a hard blocker abstains)
    p_perfect: float
    p_acceptable: float
    p_incorrect: float
    correctness: float
    reason: str

    @property
    def abstained(self) -> bool:
        return not self.commit

    def to_dict(self) -> dict[str, Any]:
        return {
            "commit": self.commit,
            "tier": self.tier,
            "utility": round(self.utility, 4),
            "p_perfect": round(self.p_perfect, 4),
            "p_acceptable": round(self.p_acceptable, 4),
            "p_incorrect": round(self.p_incorrect, 4),
            "correctness": round(self.correctness, 4),
            "reason": self.reason,
        }


def decide(s: GateSignals) -> EUDecision:
    """Apply the expected-utility three-tier gate to a signal bundle.

    1. **Hard blocker** (:func:`hard_abstain_reason`) → ABSTAIN, irrespective of ``U``.
    2. **HARD ACCEPT** — deterministic lane ∧ evidence-slots complete ∧ verifier agrees → commit.
    3. **SOFT ACCEPT** — ``U > 0`` → commit on positive expected utility.
    4. Otherwise ABSTAIN (``U ≤ 0``).

    ``budget_exhausted`` is never consulted: an operational failure does not by itself abstain.
    """
    p_perfect, p_acceptable, p_incorrect = grade_probabilities(s)
    u = (SCORE_PERFECT * p_perfect + SCORE_ACCEPTABLE * p_acceptable
         + SCORE_INCORRECT * p_incorrect)
    correctness = p_perfect + p_acceptable

    def _mk(commit: bool, tier: str, reason: str) -> EUDecision:
        return EUDecision(
            commit=commit, tier=tier, utility=u, p_perfect=p_perfect,
            p_acceptable=p_acceptable, p_incorrect=p_incorrect,
            correctness=correctness, reason=reason)

    tau = commit_threshold()

    blocker = hard_abstain_reason(s)
    if blocker is not None:
        return _mk(False, ABSTAIN, f"hard-abstain: {blocker} (U={u:.3f}, epistemic blocker)")

    if s.deterministic_lane and s.evidence_slots_complete and s.answer_verifier_agrees:
        return _mk(True, HARD_ACCEPT,
                   f"HARD ACCEPT: deterministic lane + evidence complete + verifier agree (U={u:.3f})")

    # SOT-2635 — commit bar is the calibrated τ (``RAG_EU_GATE_TAU``, default 0.0 ⇒ the original U>0).
    if u > tau:
        return _mk(True, SOFT_ACCEPT, f"SOFT ACCEPT: expected utility U={u:.3f} > τ={tau:.3f}")

    return _mk(False, ABSTAIN,
               f"ABSTAIN: expected utility U={u:.3f} ≤ τ={tau:.3f} (commit is EV-negative)")
