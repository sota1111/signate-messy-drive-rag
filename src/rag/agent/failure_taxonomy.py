"""SOT-2584 — failure taxonomy separating operational control failure from epistemic abstention.

Parent SOT-2568 deep-research comment (実装順 2/7, P0). The gold-100 loss analysis found the dominant
abstain cause is ``BUDGET_EXHAUSTED`` (27/54) — the agent ran out of turns *searching for where the
evidence lives*, not because the answer does not exist. Treating that as a terminal abstain category
conflates a **control-plane failure** with a genuine *epistemic* "the corpus does not contain the
answer". This module gives the abstain reasons a single typed taxonomy in which ``BUDGET_EXHAUSTED`` is
explicitly an **operational error**, disjoint from the epistemic reasons, so diagnostics can measure the
recoverable share separately.

The taxonomy (research comment §"探索律速への対策")::

    CORPUS_ABSENT           # 全対象を探索したが本当に存在しない (epistemic)
    DOC_RESOLUTION_FAILED   # 対象ファイルを一意に確定できなかった
    NOT_RETRIEVED           # ファイルは分かるが必要 evidence を取得できない
    PARSER_CAPABILITY_MISS  # evidence は存在するが extractor が扱えない (cf/comment/theme color)
    EVIDENCE_INCOMPLETE     # 一部の operand / version / row が不足
    EXECUTION_DISAGREEMENT  # 数式/検算結果が一致しない
    BUDGET_EXHAUSTED        # 制御上の失敗。epistemic 棄権と分離する

This module is **pure and additive** — importing it and mapping a code never touches the answer path.
It maps the existing :mod:`src.rag.agent.abstain_ledger` state codes and the document-registry hard
constraint reason onto the taxonomy, and exposes :func:`is_operational` so a diagnostic can split
``BUDGET_EXHAUSTED`` (recoverable, control failure) from the epistemic remainder.
"""
from __future__ import annotations

from typing import Any, Mapping

# --------------------------------------------------------------------------- the taxonomy
CORPUS_ABSENT = "CORPUS_ABSENT"
DOC_RESOLUTION_FAILED = "DOC_RESOLUTION_FAILED"
NOT_RETRIEVED = "NOT_RETRIEVED"
PARSER_CAPABILITY_MISS = "PARSER_CAPABILITY_MISS"
EVIDENCE_INCOMPLETE = "EVIDENCE_INCOMPLETE"
EXECUTION_DISAGREEMENT = "EXECUTION_DISAGREEMENT"
BUDGET_EXHAUSTED = "BUDGET_EXHAUSTED"

TAXONOMY: tuple[str, ...] = (
    CORPUS_ABSENT, DOC_RESOLUTION_FAILED, NOT_RETRIEVED, PARSER_CAPABILITY_MISS,
    EVIDENCE_INCOMPLETE, EXECUTION_DISAGREEMENT, BUDGET_EXHAUSTED,
)

# Human-readable JP note per code (for reports / diagnostics).
TAXONOMY_LABELS: dict[str, str] = {
    CORPUS_ABSENT: "コーパス不在(真の不存在)",
    DOC_RESOLUTION_FAILED: "文書解決失敗(対象を一意に確定不能)",
    NOT_RETRIEVED: "証拠未取得(ファイルは既知だが値を取れない)",
    PARSER_CAPABILITY_MISS: "抽出器能力不足(条件付き書式/コメント/テーマ色)",
    EVIDENCE_INCOMPLETE: "証拠不完全(operand/version/row の一部欠落)",
    EXECUTION_DISAGREEMENT: "実行不一致(式/検算の結果が不一致)",
    BUDGET_EXHAUSTED: "予算枯渇(制御上の失敗・epistemic 棄権ではない)",
}

# The ONLY operational (control-plane) failure — everything else is an epistemic abstain reason.  This
# is the split the research comment insists on: ``BUDGET_EXHAUSTED`` must not be counted as "we correctly
# decided not to answer" — it is recoverable by reaching the evidence faster (SOT-2583/2584's whole point).
_OPERATIONAL: frozenset[str] = frozenset({BUDGET_EXHAUSTED})

# --------------------------------------------------------------------------- source-code → taxonomy map
# Abstain-ledger state codes (:mod:`src.rag.agent.abstain_ledger`).
_LEDGER_STATE_MAP: dict[str, str] = {
    "NOT_RETRIEVED": NOT_RETRIEVED,
    "RETRIEVED_NOT_PARSED": PARSER_CAPABILITY_MISS,
    "PARSED_AMBIGUOUS": DOC_RESOLUTION_FAILED,
    "EVIDENCE_CONFLICT": EVIDENCE_INCOMPLETE,
    "EVIDENCE_INCOMPLETE": EVIDENCE_INCOMPLETE,
    "UNANSWERABLE": CORPUS_ABSENT,
    "BUDGET_EXHAUSTED": BUDGET_EXHAUSTED,
    "SPIN_CUTOFF": BUDGET_EXHAUSTED,   # dead-end spin cut short = a control failure, not epistemic
}

# The investigator ``stop_reason`` (when no ledger code is available) — a hard boundary abstain.
_STOP_REASON_MAP: dict[str, str] = {
    "max_turns": BUDGET_EXHAUSTED,
    "timeout": BUDGET_EXHAUSTED,
    "model_error": BUDGET_EXHAUSTED,
}

# The document-registry explicit-file hard constraint (SOT-2583) resolves before the loop.
_REGISTRY_REASON = "DOC_NOT_FOUND_AFTER_EXHAUSTIVE_MANIFEST_SCAN"


def is_operational(code: str) -> bool:
    """True when ``code`` is a control-plane failure (recoverable), not an epistemic abstain reason."""
    return code in _OPERATIONAL


def is_epistemic(code: str) -> bool:
    """True when ``code`` is a genuine "cannot/should not answer" reason (the disjoint complement)."""
    return code in TAXONOMY and code not in _OPERATIONAL


def from_ledger_state(state: str | None) -> str | None:
    """Map an abstain-ledger state code onto the taxonomy, or ``None`` if unrecognised."""
    if not state:
        return None
    return _LEDGER_STATE_MAP.get(str(state).strip().upper())


def from_stop_reason(stop_reason: str | None) -> str | None:
    """Map an investigator ``stop_reason`` onto the taxonomy, or ``None`` if unrecognised.

    The registry hard-constraint reason (a named file absent after an exhaustive scan) is a
    :data:`DOC_RESOLUTION_FAILED`, not a budget failure.
    """
    if not stop_reason:
        return None
    reason = str(stop_reason).strip()
    if reason == _REGISTRY_REASON:
        return DOC_RESOLUTION_FAILED
    return _STOP_REASON_MAP.get(reason)


def classify(*, ledger_state: str | None = None, stop_reason: str | None = None) -> str | None:
    """Best-effort taxonomy code for one abstain, ledger state first then ``stop_reason``.

    The ledger state is the richer signal (it reflects the observed tool outcomes), so it wins when
    present; the ``stop_reason`` is the fallback for a boundary abstain that never reached the ledger.
    Returns ``None`` when neither maps (e.g. a committed answer — not an abstain).
    """
    return from_ledger_state(ledger_state) or from_stop_reason(stop_reason)


def classify_investigation(investigation: Any,
                           ledger_state: str | None = None) -> str | None:
    """Taxonomy code for an :class:`~src.rag.agent.investigator.Investigation`-like object.

    Accepts the ``ledger_state`` explicitly (the caller reads it from the abstain ledger) and falls back
    to the investigation's ``stop_reason``. A committed (non-abstain) answer returns ``None``.
    """
    stop_reason = getattr(investigation, "stop_reason", None)
    if stop_reason is None and isinstance(investigation, Mapping):
        stop_reason = investigation.get("stop_reason")
    return classify(ledger_state=ledger_state, stop_reason=stop_reason)


def tally(codes: "list[str]") -> dict[str, Any]:
    """Aggregate a list of taxonomy codes into operational/epistemic totals (for diagnostics)."""
    per_code = {code: 0 for code in TAXONOMY}
    for c in codes:
        if c in per_code:
            per_code[c] += 1
    operational = sum(v for k, v in per_code.items() if is_operational(k))
    epistemic = sum(v for k, v in per_code.items() if is_epistemic(k))
    return {
        "total": len(codes),
        "operational": operational,          # = BUDGET_EXHAUSTED share (recoverable)
        "epistemic": epistemic,
        "per_code": per_code,
    }
