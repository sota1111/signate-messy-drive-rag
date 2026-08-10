"""Per-contract-type deterministic pipelines (反転アーキ Wave A1〜B2, PLAN SOT-2602).

Importing this package registers every per-type pipeline into the Stage0 router registry
(:mod:`src.rag.agent.det_pipeline`). The router lazily imports this package the first time it
resolves (``det_pipeline._bootstrap_pipelines``), so a registered pipeline is discovered without
the investigator having to import each module by hand.

Each submodule owns exactly one ``contract type`` and self-registers on import via
``det_pipeline.register`` — this package only lists which submodules to load. Adding a new Wave
pipeline is a one-line import here plus the new module.
"""
from __future__ import annotations

# Wave A1 — version_diff (SOT-2605). Importing the module runs its module-level ``register()``.
from . import version_diff as version_diff  # noqa: F401

__all__ = ["version_diff"]
