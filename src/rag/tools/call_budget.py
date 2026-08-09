"""Per-call wall-clock budget propagation for long-running scan tools (SOT-2563).

The investigator's ``timeout_s`` is only checked **between** model turns
(``src.rag.agent.investigator`` — ``if clock() - start > timeout_s``), so a single synchronous
full-corpus ``file_grep`` scan cannot be interrupted mid-call and can overrun the nominal 180/240s
budget by 1.5–3.6× (280–658s) before the between-turn check trips — melting the whole question
budget in one call and leaving nothing for an alternate route (see
``docs/ai/abstain_wrong_root_cause_SOT-2550.md`` §3(A)).

This module is the runtime-side guard. The investigate loop publishes the *remaining* wall-clock
budget for the tool call it is about to dispatch via :func:`remaining_budget`; a scan tool derives a
per-call deadline from it (:func:`scan_deadline`) and cooperatively cancels its scan when the
deadline is reached, returning early ("not found") so the model can switch to a cheaper route
instead of blowing the budget in one call.

Precision-first, non-destructive default
-----------------------------------------
The per-call cap :func:`max_scan_seconds` (env ``RAG_FILE_GREP_MAX_SCAN_S``, default **180s**) sits
above the observed maximum *legitimate* file_grep completion (match questions: median 22s, max 125s
— SOT-2550 ledger), so a completing/match answer is never cut; only a runaway call that would
otherwise time out anyway is bounded. The deadline is additionally clamped to the propagated
remaining budget, so the scan never runs past the question's overall ``timeout_s`` either. With the
cap disabled (``RAG_FILE_GREP_MAX_SCAN_S<=0``) *and* no budget propagated (a direct call / unit
test), the scan is unbounded and the champion serve path is byte-identical.
"""
from __future__ import annotations

import contextlib
import os
import time
from contextvars import ContextVar
from typing import Callable, Iterator

# Remaining per-question wall-clock budget (seconds) the investigator has left when it dispatches a
# tool call. ``None`` ⇒ not propagated (direct call / test); the tool then uses only the env cap.
_REMAINING_S: ContextVar["float | None"] = ContextVar("rag_call_remaining_s", default=None)

_DEFAULT_MAX_SCAN_S = 180.0
_DEFAULT_RESERVE_S = 30.0


def max_scan_seconds(default: float = _DEFAULT_MAX_SCAN_S) -> float:
    """Per-call wall-clock cap (seconds) for a full-corpus scan (env ``RAG_FILE_GREP_MAX_SCAN_S``).

    Default 180s is above the observed max legitimate file_grep completion (125s), so the cap never
    cuts a completing answer — only runaway calls (280–658s) that would time out anyway. ``<=0``
    disables the cap (unbounded scan). An unparseable value falls back to ``default``.
    """
    raw = os.getenv("RAG_FILE_GREP_MAX_SCAN_S")
    if raw is None or not raw.strip():
        return default
    try:
        return float(raw.strip())
    except (TypeError, ValueError):
        return default


def reserve_seconds(default: float = _DEFAULT_RESERVE_S) -> float:
    """Question budget kept for route switching after a cancelled scan.

    ``RAG_FILE_GREP_RESERVE_S`` is applied only when the investigator propagated a remaining
    question budget.  Direct calls still receive the full per-call cap.  Keeping a small reserve is
    essential: clamping a scan to *all* remaining time bounds the runaway, but leaves no model turn
    in which to consume ``deadline_hit`` and select the alternate route promised by the contract.
    """
    raw = os.getenv("RAG_FILE_GREP_RESERVE_S")
    if raw is None or not raw.strip():
        return default
    try:
        return max(0.0, float(raw.strip()))
    except (TypeError, ValueError):
        return default


def current_remaining() -> "float | None":
    """The remaining wall-clock budget published for the current tool call, or ``None``."""
    return _REMAINING_S.get()


@contextlib.contextmanager
def remaining_budget(seconds: "float | None") -> Iterator[None]:
    """Publish the remaining wall-clock budget (seconds) for tool calls made inside this block.

    A non-positive value is clamped to 0.0 (deadline = now ⇒ an already-exhausted budget makes the
    scan return immediately rather than run a full pass it has no time for).
    """
    value = seconds if (seconds is None or seconds > 0) else 0.0
    token = _REMAINING_S.set(value)
    try:
        yield
    finally:
        _REMAINING_S.reset(token)


def scan_deadline(clock: Callable[[], float] = time.monotonic,
                  *, max_scan_s: "float | None" = None) -> "float | None":
    """Absolute (``clock``-domain) deadline for a scan starting *now*, or ``None`` if unbounded.

    Combines the env per-call cap with any propagated remaining budget: the scan is bounded by the
    **smaller** of the two, so it neither overruns the per-call cap nor the question's overall
    ``timeout_s``. Returns ``None`` only when the cap is disabled *and* no budget was propagated.
    """
    cap = max_scan_seconds() if max_scan_s is None else max_scan_s
    remaining = _REMAINING_S.get()
    budget: "float | None" = None
    if cap is not None and cap > 0:
        budget = float(cap)
    if remaining is not None:
        # Leave time for the model to observe deadline_hit and execute a cheaper route.  Without
        # this reserve the former min(cap, remaining) merely changed a 300–658s overrun into a
        # ~180s timeout/abstain, as the SOT-2563 human-review focused run demonstrated.
        route_budget = max(0.0, float(remaining) - reserve_seconds())
        budget = route_budget if budget is None else min(budget, route_budget)
    if budget is None:
        return None
    return clock() + max(0.0, budget)
