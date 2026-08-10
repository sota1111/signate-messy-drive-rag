#!/usr/bin/env python3
"""SOT-2616 — focused offline verification of NUMERIC operand candidate prefill.

The phase-0 diagnostics (``docs/ai/budget32_trace_classification.md``) showed the NUMERIC
``BUDGET_EXHAUSTED`` losses are an *operand delivery* deficit (derivation ok 41%, idx76 spends 17/18
turns searching without binding a single operand). This script measures — **offline, network-free, no
LLM** — whether the prefill (``RAG_OPERAND_PREFILL``) actually surfaces the operand candidates the loop
was grep-spinning for, on the focused set from the trace classification.

For each focused question it:
  1. resolves the target documents deterministically (document registry),
  2. enumerates operand candidates from the built offline assets (structure store highlights + the
     resolved spreadsheets' normalized rows),
  3. simulates the *bound-operand handoff*: selecting the top candidate(s) through
     ``pot_lane.resolve_operand_selections`` + the operand layer, and reports whether the operand layer
     would pass (``operand_sources_complete``) — i.e. the ``operand_binding`` first layer that
     ``scripts/measure_pot_lane.py`` aggregates.

The headline metric is ``operand_candidates_available`` / ``operand_binding_reachable``: the share of the
focused set for which the loop no longer has to *discover* operands because they are pre-listed with
provenance. This is the offline proxy for the "operand_binding 改善・compute 連発の減少" acceptance check;
the full three-layer accuracy still needs a live ``RAG_POT_HARD_LANE=1`` trace (see measure_pot_lane.py).

Usage::

    RAG_DOCUMENT_REGISTRY=1 RAG_OPERAND_PREFILL=1 .venv/bin/python scripts/verify_operand_prefill.py

Writes ``artifacts/operand_prefill_focused.json`` and prints a per-question + summary table.
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# The prefill + registry are opt-in; default them ON for this verification run (a caller may still
# override). Set before importing the modules that read the flags at call time.
os.environ.setdefault("RAG_DOCUMENT_REGISTRY", "1")
os.environ.setdefault("RAG_OPERAND_PREFILL", "1")

from config import settings  # noqa: E402
from src.rag.agent import evidence_packet as ep  # noqa: E402
from src.rag.agent import operand_prefill as opf  # noqa: E402
from src.rag.agent import pot_lane as pl  # noqa: E402
from src.rag.agent import query_router as qr  # noqa: E402

# The numeric-system BUDGET indices called out in the trace classification (§起票への含意 #3 / 検証内容).
FOCUSED = [76, 47, 50, 99, 8, 28, 57, 63]


def _resolve_project(question: str) -> str | None:
    try:
        from src.rag.tools.canonical_route import resolve_project
        return resolve_project(question, None)
    except Exception:
        return None


def _binding_reachable(catalog: list[dict]) -> tuple[bool, dict | None]:
    """Simulate the handoff: pick the top candidate as a lone operand and run the operand layer.

    Uses ``pot_lane.resolve_operand_selections`` (the real handoff) + ``evaluate_candidate`` so the
    reported pass/fail is exactly the lane's operand verdict, not a re-implementation.
    """
    if not catalog:
        return False, None
    top = catalog[0]
    candidate = {"operands": [{"name": "x", "select": top["id"]}], "formula": {"ref": "x"}}
    prepared = pl.resolve_operand_selections([candidate], catalog)
    result = pl.evaluate_candidate(prepared[0])
    return result.operand_verdict.ok, result.operand_verdict.to_dict()


def main() -> None:
    review = pd.read_csv(settings.ARTIFACTS_DIR / "gold_100_review.csv", encoding="utf-8-sig")
    by_index = {int(r["index"]): r for _, r in review.iterrows()}

    rows: list[dict] = []
    for idx in FOCUSED:
        r = by_index.get(idx)
        if r is None:
            rows.append({"index": idx, "error": "index absent from gold_100_review.csv"})
            continue
        question = str(r["question"])
        project = _resolve_project(question)
        decision = qr.classify_route(question)
        packet = ep.build_packet(question, project=project, decision=decision)
        catalog = list(packet.evidence.get("operand_candidates") or [])
        reachable, verdict = _binding_reachable(catalog)
        rows.append({
            "index": idx,
            "archetype": str(r.get("archetype", "")),
            "route": decision.route,
            "resolved_docs": [d.doc_id for d in packet.resolved_documents],
            "n_candidates": len(catalog),
            "top_candidates": [
                {"id": c["id"], "value": c.get("value"), "label": c.get("label"),
                 "source": c.get("source"), "origin": c.get("origin")}
                for c in catalog[:5]
            ],
            "operand_binding_reachable": reachable,
            "operand_verdict": verdict,
        })

    n = len([x for x in rows if "error" not in x])
    with_cands = sum(1 for x in rows if x.get("n_candidates", 0) > 0)
    reachable = sum(1 for x in rows if x.get("operand_binding_reachable"))
    summary = {
        "prefill_enabled": opf.enabled(),
        "registry_enabled": os.getenv("RAG_DOCUMENT_REGISTRY"),
        "focused_n": n,
        "operand_candidates_available": with_cands,
        "operand_candidates_rate": round(with_cands / n, 4) if n else 0.0,
        "operand_binding_reachable": reachable,
        "operand_binding_reachable_rate": round(reachable / n, 4) if n else 0.0,
    }
    out = settings.ARTIFACTS_DIR / "operand_prefill_focused.json"
    out.write_text(json.dumps({"summary": summary, "detail": rows}, ensure_ascii=False, indent=2)
                   + "\n", encoding="utf-8")

    print("=== SOT-2616 operand prefill — focused verification ===")
    for x in rows:
        if "error" in x:
            print(f"idx {x['index']:>3}: {x['error']}")
            continue
        print(f"idx {x['index']:>3} [{x['archetype']}/{x['route']}]: "
              f"docs={len(x['resolved_docs'])} candidates={x['n_candidates']} "
              f"binding_reachable={x['operand_binding_reachable']}")
        for c in x["top_candidates"][:3]:
            print(f"        - {c['id']}: {c['value']}  «{c['label']}»  [{c['source']}]")
    print("--- summary ---")
    print(f"operand_candidates_available: {with_cands}/{n} "
          f"({summary['operand_candidates_rate']})")
    print(f"operand_binding_reachable:    {reachable}/{n} "
          f"({summary['operand_binding_reachable_rate']})")
    print(f"-> {out}")


if __name__ == "__main__":
    main()
