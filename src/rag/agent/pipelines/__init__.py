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

# Wave A2 — numeric / derived_calculation (SOT-2607). Importing the module runs its ``register()``.
from . import numeric as numeric  # noqa: F401

# Wave A3 — enum / cross_aggregate (SOT-2608). Importing the module runs its ``register()`` for both the
# ``full_enumeration`` and ``cross_aggregate`` contracts.
from . import enumeration as enumeration  # noqa: F401

# Wave A4 — chart_read / spatial (SOT-2609). Imported AFTER ``enumeration`` so its ``register()`` can
# capture A3's ``full_enumeration`` pipeline and compose the seating列挙 recognizer in front of it
# (座席列挙 idx44) without dropping A3's behavior.
from . import chart_spatial as chart_spatial  # noqa: F401

# Wave B2 — simple_lookup / fact_lookup (SOT-2612). Deterministic schedule-table / report-page reads;
# everything else it cannot pin structurally falls back to the LLM loop.
from . import fact_lookup as fact_lookup  # noqa: F401

# Wave B1 — document_extract / format_check (SOT-2611). Deterministic highlighted-pivot-cell extraction
# conditions + aggregation (real-cell outline, embedded-EMF outline, EMF 2D cross-tab); anything it cannot
# ground structurally falls back to the LLM loop.
from . import document_extract as document_extract  # noqa: F401

__all__ = ["version_diff", "numeric", "enumeration", "chart_spatial", "fact_lookup", "document_extract"]
