#!/usr/bin/env python3
"""SOT-2584 — offline diagnostics for the typed query router + failure taxonomy over gold100.

The internal harness's role is a *故障箇所を分離する診断器*, not an LB predictor (local proxy ↔ real
LB ρ=-0.09). This script records the SOT-2584 routing/diagnostic metrics **offline** (no Gemini/LLM)
from the last gold-100 review (``artifacts/gold_100_review.csv``, produced by ``scoring.gold_offline``):

  * ``route_accuracy``           — router route vs. gold-archetype-implied route (refinements allowed)
  * ``route_distribution``       — how the 100 questions type across the 6+1 routes
  * ``hard_lane_invocation_rate``— fraction routed to a deterministic primary lane (compute/enum/diff/…)
  * ``failure_taxonomy``         — the abstains re-coded onto the taxonomy that **separates
                                   BUDGET_EXHAUSTED (operational) from epistemic abstains**

    .venv/bin/python scripts/measure_query_router.py

Writes ``artifacts/query_router_diagnostics.json`` (machine) and prints a summary. Deterministic and
network-free — the router types from question text and the taxonomy maps the recorded ``state_code``.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from config import settings
from src.rag.agent import failure_taxonomy as ft
from src.rag.agent import query_router as qr


def _load_review() -> pd.DataFrame:
    path = settings.ARTIFACTS_DIR / "gold_100_review.csv"
    # utf-8-sig strips the BOM the review CSV header carries ("﻿index").
    return pd.read_csv(path, encoding="utf-8-sig")


def _rows(df: pd.DataFrame) -> list[dict[str, str]]:
    return [{"question": str(r.get("question", "")), "archetype": str(r.get("archetype", "") or "")}
            for _, r in df.iterrows()]


def _taxonomy_tally(df: pd.DataFrame) -> dict:
    """Re-code every abstain's recorded ``state_code`` onto the failure taxonomy and tally it."""
    codes: list[str] = []
    unmapped: list[str] = []
    for _, r in df.iterrows():
        status = str(r.get("status", "") or "").strip().lower()
        state = str(r.get("state_code", "") or "").strip()
        # Only abstains carry a taxonomy code; a matched/wrong committed answer is not a failure code.
        if status not in ("abstain", "missing", "abstained"):
            continue
        code = ft.from_ledger_state(state)
        if code is None:
            unmapped.append(state or "(empty)")
            continue
        codes.append(code)
    tally = ft.tally(codes)
    tally["unmapped_states"] = unmapped
    return tally


def main() -> None:
    df = _load_review()
    rows = _rows(df)
    report = qr.route_agreement(rows)
    taxonomy = _taxonomy_tally(df)

    result = {
        "n_questions": len(rows),
        "route_accuracy": round(report.rate, 4),
        "route_distribution": report.distribution,
        "hard_lane_invocation_rate": round(report.hard_lane_rate, 4),
        "route_refinements": list(report.refinements),
        "route_mismatches": list(report.mismatches),
        "failure_taxonomy": {
            "total_abstains": taxonomy["total"],
            "operational_budget_exhausted": taxonomy["operational"],
            "epistemic": taxonomy["epistemic"],
            "per_code": taxonomy["per_code"],
            "unmapped_states": taxonomy["unmapped_states"],
        },
    }
    out = settings.ARTIFACTS_DIR / "query_router_diagnostics.json"
    out.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    print(f"n_questions: {result['n_questions']}")
    print(f"route_accuracy: {result['route_accuracy']}")
    print(f"route_distribution: {result['route_distribution']}")
    print(f"hard_lane_invocation_rate: {result['hard_lane_invocation_rate']}")
    print(f"failure_taxonomy: operational(BUDGET_EXHAUSTED)="
          f"{result['failure_taxonomy']['operational_budget_exhausted']} "
          f"epistemic={result['failure_taxonomy']['epistemic']} "
          f"per_code={result['failure_taxonomy']['per_code']}")
    print(f"-> {out}")


if __name__ == "__main__":
    main()
