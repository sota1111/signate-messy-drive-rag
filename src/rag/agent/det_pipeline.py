"""SOT-2603 (Stage0) — deterministic pipeline dispatch: the entry-gate registry.

Parent PLAN SOT-2602 (決定論先行パイプラインへの反転). This module promotes the contract
classifier (SOT-2493) from an *in-loop hint* into a deterministic **router (入口ゲート)**:
when enabled, a question whose contract type has a registered deterministic pipeline is
answered by that pipeline **without going through the LLM loop**, and every other question —
an unregistered contract type, or a pipeline that cannot ground a ``{value, evidence, method}``
result — falls back to the existing investigator LLM loop unchanged (回答数を減らさない).

Design invariants
-----------------
* **Thin & dependency-light.** This module owns only the ``contract type → pipeline`` registry
  and the enable flag. It imports nothing from :mod:`src.rag.agent.investigator`, so importing
  it never drags in the generation stack and there is no circular import — the per-type
  pipelines (Wave A1〜B2) register into it from their own modules, and the investigator calls
  :func:`resolve` at its entry gate.
* **Empty at Stage0 ⇒ champion byte-identical.** Stage0 ships the registry EMPTY. With the flag
  on and nothing registered, :func:`resolve` returns ``None`` for every question, so every
  question still routes to the LLM loop — the champion serve path is byte-identical. The flag
  (:func:`enabled`, ``RAG_DET_PIPELINE_ROUTER``) is **default OFF** on top of that, so even a
  future non-empty registry is dormant until explicitly enabled.
* **Fallback, never abstain.** A registered pipeline that returns ``None`` / a null ``value`` /
  a malformed result — or that raises — is treated as "resolved nothing", and :func:`resolve`
  returns ``None`` so the caller falls back to the LLM loop. The router only ever *adds* a
  deterministic short-circuit; it never reduces the number of questions that get an answer.
* **No corpus fact / no hardcoding.** This layer holds no answers and no corpus facts; the
  registered pipelines self-derive their value from the question via the deterministic tools.
"""
from __future__ import annotations

import os
from typing import Any, Callable, Mapping

from src.rag.tools import contract as _contract

# A deterministic pipeline maps a question (and the live corpus profile) to a
# ``{value, evidence, method}`` contract mapping, or ``None`` when it does not apply / cannot
# ground an answer. A returned contract whose ``value`` is ``None`` is likewise treated as
# "resolved nothing" and routes to the LLM fallback.
Pipeline = Callable[..., "Mapping[str, Any] | None"]

# The single source of truth for ``contract type → deterministic pipeline``. Ships EMPTY at
# Stage0 (see the module docstring); Wave A1〜B2 register their per-type pipelines here.
_REGISTRY: "dict[str, Pipeline]" = {}

# One-time lazy discovery of the per-type pipelines. The pipelines self-register on import (Wave A1〜B2
# live in :mod:`src.rag.agent.pipelines`); importing that package here — rather than at investigator
# import time — keeps this module dependency-light and the import graph acyclic. Only ever triggered on
# the enabled router path, so the flag-OFF serve path takes on no extra work.
_BOOTSTRAPPED = False


def _bootstrap_pipelines() -> None:
    """Import the pipelines package once so its per-type pipelines register (fail-open)."""
    global _BOOTSTRAPPED
    if _BOOTSTRAPPED:
        return
    _BOOTSTRAPPED = True
    try:
        import src.rag.agent.pipelines  # noqa: F401 — import side effect: per-type registration
    except Exception:  # noqa: BLE001 — a broken pipeline package must never break the answer path
        pass


def _env_flag(key: str, default: bool) -> bool:
    raw = os.getenv(key)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def enabled() -> bool:
    """Whether the deterministic router (entry gate) is active.

    **Default OFF** (``RAG_DET_PIPELINE_ROUTER``) so the champion serve path stays
    byte-identical: when off, :func:`resolve` short-circuits to ``None`` and every question
    flows through the unchanged LLM loop.
    """
    return _env_flag("RAG_DET_PIPELINE_ROUTER", False)


# Per-Wave sub-gates layered UNDER the master router (SOT-2618). The Wave B pipelines register the
# same as A1〜A4, but SOT-2613's consolidated gold100 split their fates: B1 (document_extract) was a
# local PASS (match 13→17 / wrong 2→1) while B2 (fact_lookup) regressed (match 18→16 / wrong 2→3), so
# "Wave A + B" as a whole was rejected. These flags let B1 and B2 be turned on independently so the
# adopted composition is **Wave A + B1** (B1 default ON, B2 default OFF) without dragging B2's
# regression back in — while B2 stays recoverable (``RAG_DET_PIPELINE_B2=1``) for a future
# re-measurement. A contract NOT listed here has no sub-gate and is always active when the router is on
# (Wave A1〜A4 are unaffected). The master router flag still governs everything: with the router OFF the
# whole layer is dormant and the serve path is byte-identical regardless of these values.
_WAVE_FLAGS: "dict[str, tuple[str, bool]]" = {
    "format_check": ("RAG_DET_PIPELINE_B1", True),    # Wave B1 (document_extract, SOT-2611) — default ON
    "simple_lookup": ("RAG_DET_PIPELINE_B2", False),  # Wave B2 (fact_lookup, SOT-2612) — default OFF
}


def wave_enabled(contract: "str | None") -> bool:
    """Whether ``contract``'s per-Wave sub-gate is on (independent of the master router).

    Contracts without a sub-gate (everything except Wave B1/B2) return ``True`` — they are governed
    solely by the master router. Wave B1/B2 read their own ``RAG_DET_PIPELINE_B1`` / ``…_B2`` flag
    (defaults: B1 ON, B2 OFF), so the router-on default composition is **Wave A + B1**.
    """
    spec = _WAVE_FLAGS.get(contract or "")
    if spec is None:
        return True
    key, default = spec
    return _env_flag(key, default)


def register(contract_type: str, pipeline: Pipeline, *, replace: bool = False) -> None:
    """Register the deterministic ``pipeline`` for a contract type (Wave A1〜B2 wiring).

    Raises :class:`ValueError` when ``contract_type`` already has a pipeline unless ``replace``
    is set, so a double-wire is caught loudly rather than silently shadowed.
    """
    if not contract_type:
        raise ValueError("contract_type must be a non-empty string")
    if not callable(pipeline):
        raise TypeError("pipeline must be callable")
    if contract_type in _REGISTRY and not replace:
        raise ValueError(f"a pipeline is already registered for contract {contract_type!r}")
    _REGISTRY[contract_type] = pipeline


def unregister(contract_type: str) -> None:
    """Remove a registered pipeline (test / rollback helper); a no-op when absent."""
    _REGISTRY.pop(contract_type, None)


def registered_contracts() -> "frozenset[str]":
    """The contract types that currently have a deterministic pipeline registered."""
    _bootstrap_pipelines()
    return frozenset(_REGISTRY)


def resolve(question: str, contract: "str | None", *, profile: Any = None,
            force: bool = False) -> "dict[str, Any] | None":
    """Run the registered deterministic pipeline for ``contract`` — the entry gate.

    Returns the pipeline's normalized ``{value, evidence, method}`` contract dict **only when**
    all of these hold: the router is enabled (or ``force`` is set — for direct unit testing), a
    pipeline is registered for ``contract``, and that pipeline produced a valid contract with a
    non-``None`` ``value``. In every other case — router off, no ``contract`` classified, no
    pipeline registered, or a pipeline that returned ``None`` / a null value / a malformed
    result / raised — it returns ``None`` to signal the caller "fall back to the LLM loop"
    (回答数を減らさない). It never raises into the answer path.
    """
    if not (force or enabled()):
        return None
    if not contract:
        return None
    # Per-Wave sub-gate (SOT-2618): a Wave B1/B2 contract whose flag is off falls back to the LLM loop
    # even with the master router on. ``force`` bypasses it so direct unit/integration tests exercise a
    # pipeline's grounding regardless of the env flags (回答数を減らさない).
    if not force and not wave_enabled(contract):
        return None
    _bootstrap_pipelines()
    pipeline = _REGISTRY.get(contract)
    if pipeline is None:
        return None
    try:
        result = pipeline(question, profile=profile)
    except Exception:  # noqa: BLE001 — a broken pipeline must never break the answer path
        return None
    if result is None or not _contract.is_contract(result):
        return None
    normalized = _contract.ensure_contract(result)
    if normalized.get("value") is None:
        return None
    return normalized
