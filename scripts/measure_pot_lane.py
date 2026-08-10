#!/usr/bin/env python3
"""SOT-2586 — three-layer diagnostics for the NUMERIC PoT forced lane over gold100.

The internal harness is a *故障箇所を分離する診断器*, not an LB predictor (local proxy ↔ real LB
ρ=-0.09). This script records the SOT-2586 three-layer metrics for ``derived_calculation`` — the weakest
gold100 archetype — so an A-type failure (operand が届かない) is separated from a B-type failure (計算を
間違える), rather than judged only on the final-answer match.

Two layers of measurement, both offline / network-free:

  1. **Route coverage (always).** Over the ``derived_calculation`` questions in the last gold-100 review
     (``artifacts/gold_100_review.csv``, produced by ``scoring.gold_offline``) it records how many type to
     the NUMERIC route and dispatch to the ``compute`` hard lane — i.e. how many are *eligible* for the
     forced PoT lane.

  2. **Three-layer accuracy (when traces are supplied via ``--details``).** A ``.details.jsonl`` from a
     gold run with ``RAG_POT_HARD_LANE=1`` may carry, per question, a ``pot_lane`` verdict dict (the shape
     of :meth:`src.rag.agent.pot_lane.PoTResult.to_dict`). This script aggregates them into:
       * ``operand_binding_accuracy``    — share whose operand layer passed (operand_sources_complete)
       * ``formula_accuracy``            — share whose formula layer passed (制限AST妥当・分岐整合)
       * ``execution_accuracy``          — share whose execution layer passed (実行↔独立検算一致)
       * ``verifier_disagreement_rate``  — share where execution disagreed (EXECUTION_DISAGREEMENT)

Usage::

    .venv/bin/python scripts/measure_pot_lane.py
    .venv/bin/python scripts/measure_pot_lane.py --details artifacts/predictions_test.details.jsonl

Writes ``artifacts/pot_lane_diagnostics.json`` (machine) and prints a summary. Deterministic: the router
types from question text; the three-layer accuracies read only recorded lane verdicts (no re-generation).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from config import settings
from src.rag.agent import failure_taxonomy as ft
from src.rag.agent import pot_lane as pl
from src.rag.agent import query_router as qr

DERIVED = "derived_calculation"


def _load_review() -> pd.DataFrame:
    path = settings.ARTIFACTS_DIR / "gold_100_review.csv"
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path, encoding="utf-8-sig")


def _route_coverage(df: pd.DataFrame) -> dict:
    """How the derived_calculation questions type/route — the forced-lane *eligibility* view (offline)."""
    numeric = compute_lane = total = 0
    off_route: list[dict[str, str]] = []
    for _, r in df.iterrows():
        if str(r.get("archetype", "") or "").strip() != DERIVED:
            continue
        total += 1
        q = str(r.get("question", ""))
        decision = qr.classify_route(q)
        if decision.route == qr.NUMERIC:
            numeric += 1
        else:
            off_route.append({"question": q, "route": decision.route})
        if decision.primary_lane == "compute" or decision.has_hard_lane:
            compute_lane += 1
    return {
        "derived_calculation_n": total,
        "routed_numeric": numeric,
        "numeric_route_rate": round(numeric / total, 4) if total else 0.0,
        "hard_lane_eligible": compute_lane,
        "hard_lane_rate": round(compute_lane / total, 4) if total else 0.0,
        "off_route": off_route,
    }


def _iter_pot_verdicts(details: Path):
    """Yield each recorded ``pot_lane`` verdict dict from a ``.details.jsonl`` (empty when none present)."""
    if not details.exists():
        return
    for line in details.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            rec = json.loads(line)
        except json.JSONDecodeError:
            continue
        verdict = rec.get("pot_lane")
        if isinstance(verdict, dict) and verdict.get("candidates"):
            yield rec.get("index"), verdict


def _three_layer(details: Path | None) -> dict:
    """Aggregate the recorded per-question lane verdicts into the three-layer accuracies."""
    if details is None:
        return {"traces": 0, "note": "no --details supplied — three-layer accuracies need a lane trace"}
    n = op_ok = fm_ok = ex_ok = disagree = committed = 0
    for _index, verdict in _iter_pot_verdicts(details):
        n += 1
        # A question's lane verdict = its *chosen* (committed / first) candidate's three layers.
        cands = verdict.get("candidates") or []
        chosen = cands[0]
        vd = chosen.get("verdicts", {})
        op_ok += int(bool(vd.get(pl.LAYER_OPERAND, {}).get("ok")))
        fm_ok += int(bool(vd.get(pl.LAYER_FORMULA, {}).get("ok")))
        ex_ok += int(bool(vd.get(pl.LAYER_EXECUTION, {}).get("ok")))
        if vd.get(pl.LAYER_EXECUTION, {}).get("code") == ft.EXECUTION_DISAGREEMENT:
            disagree += 1
        if (verdict.get("decision") or {}).get("status") == pl.COMMIT:
            committed += 1
    if not n:
        return {"traces": 0, "note": "details file carried no pot_lane verdicts"}
    return {
        "traces": n,
        "operand_binding_accuracy": round(op_ok / n, 4),
        "formula_accuracy": round(fm_ok / n, 4),
        "execution_accuracy": round(ex_ok / n, 4),
        "verifier_disagreement_rate": round(disagree / n, 4),
        "commit_rate": round(committed / n, 4),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="SOT-2586 PoT lane three-layer diagnostics over gold100")
    ap.add_argument("--details", type=Path, default=None,
                    help="a .details.jsonl carrying per-question pot_lane verdicts (optional)")
    args = ap.parse_args()

    df = _load_review()
    coverage = _route_coverage(df) if not df.empty else {"derived_calculation_n": 0,
                                                          "note": "gold_100_review.csv absent"}
    three_layer = _three_layer(args.details)

    result = {
        "lane_enabled_env": pl.enabled(),
        "sympy_backend": pl.have_sympy(),
        "allowed_ops": sorted(pl.ALLOWED_OPS),
        "route_coverage": coverage,
        "three_layer": three_layer,
    }
    out = settings.ARTIFACTS_DIR / "pot_lane_diagnostics.json"
    out.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    print(f"derived_calculation_n: {coverage.get('derived_calculation_n')}")
    print(f"numeric_route_rate:    {coverage.get('numeric_route_rate')}")
    print(f"hard_lane_rate:        {coverage.get('hard_lane_rate')}")
    print(f"sympy_backend:         {result['sympy_backend']}")
    if three_layer.get("traces"):
        print(f"three_layer(n={three_layer['traces']}): "
              f"operand={three_layer['operand_binding_accuracy']} "
              f"formula={three_layer['formula_accuracy']} "
              f"execution={three_layer['execution_accuracy']} "
              f"verifier_disagreement={three_layer['verifier_disagreement_rate']}")
    else:
        print(f"three_layer: {three_layer.get('note')}")
    print(f"-> {out}")


if __name__ == "__main__":
    main()
