#!/usr/bin/env python3
"""SOT-2589 — expected-utility three-tier gate diagnostics over gold100.

The internal harness is a *故障箇所を分離する診断器*, not an LB predictor (local proxy ↔ real LB
ρ=-0.09). This script re-frames the abstain decision as an expected-utility one and records the four
diagnostics the issue asks for, so a *coverage* change can be weighed against an *incorrect-risk* change
in the metric's own units (Perfect +1 / Acceptable +0.5 / Missing 0 / Incorrect −1) — instead of judging
only the raw match count.

Four blocks (all offline / network-free — no LLM, no re-generation):

  * **champion baseline** — coverage / incorrect_rate_on_answered / expected_score straight from the last
    gold-100 review CSV (the current confidence-gate state: coverage 46%, 84.8% correct-when-answered).
  * **EU-gate counterfactual on the answered pool** — apply :func:`eu_gate.decide` to the 46 answered
    questions and report how many it keeps vs newly abstains, how many of the 7 WRONG it *catches*, and
    the resulting expected_score. This is the decision-relevant number: does U drop the wrong answers
    while keeping the right ones?
  * **risk-coverage curve** — order the answered questions by U (descending = least risky first) and, at
    each coverage level, report the incorrect-rate on the committed prefix and its expected_score. A gate
    whose U is informative concentrates the WRONGs in the low-U tail.
  * **abstain taxonomy split** — of the 54 abstains, how many are **operational** (BUDGET_EXHAUSTED,
    which the EU gate deliberately does NOT treat as an abstain criterion) vs **epistemic** (corpus
    absent / parser / unresolved evidence). The operational share is the coverage-expansion headroom the
    old BUDGET_EXHAUSTED=abstain rule was leaving on the table.

Signals are derived per question from what is genuinely available offline: the SOT-2583 document registry
(canonical-doc resolution), the SOT-2584 typed router (deterministic lane), the SOT-2588 differ (version
pair), the recorded verbal confidence, and — for the *abstained* rows — the recorded ``state_code`` mapped
through :mod:`src.rag.agent.failure_taxonomy` (corpus-absent / parser / budget). Runtime-only signals that
an offline pass cannot reconstruct are handled conservatively and documented inline: a row the champion
*committed* is credited with consensus agreement (``answer_verifier_agrees`` / ``self_consistency_agrees``
were true for it to commit) — applied equally to MATCH and WRONG rows, so it is not correctness leakage.

Usage::

    RAG_EU_GATE=1 PYTHONPATH=. .venv/bin/python scripts/measure_eu_gate.py

Writes ``artifacts/eu_gate_diagnostics.json`` (machine) and prints a summary. Deterministic.
"""
from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from config import settings
from src.rag.agent import eu_gate
from src.rag.agent import failure_taxonomy as ft
from src.rag.agent.eu_gate import GateSignals, decide

ANSWERED = ("MATCH", "WRONG")

# Approximate the metric grade of an answered row from the CSV status. The review CSV records MATCH /
# WRONG only (it does not split Perfect vs Acceptable), so a MATCH is scored as +1 (Perfect) — a mild
# upper bound noted in the output; a WRONG is −1.
_SCORE = {"MATCH": eu_gate.SCORE_PERFECT, "WRONG": eu_gate.SCORE_INCORRECT}


def _load_rows() -> list[dict[str, str]]:
    path = settings.ARTIFACTS_DIR / "gold_100_review.csv"
    if not path.exists():
        return []
    with path.open(encoding="utf-8-sig", newline="") as fh:
        return [
            {
                "index": str(r.get("index", "")),
                "question": str(r.get("question", "")),
                "status": str(r.get("status", "")).strip().upper(),
                "archetype": str(r.get("archetype", "")).strip(),
                "state_code": str(r.get("state_code", "")).strip().upper(),
                "confidence": str(r.get("confidence", "")).strip(),
            }
            for r in csv.DictReader(fh)
        ]


def _float(x: str, default: float = 0.0) -> float:
    try:
        return float(x)
    except (TypeError, ValueError):
        return default


def _registry_resolves(question: str) -> bool:
    try:
        from src.rag.index import document_registry as dr

        resolver = dr.get_resolver()
        return bool(resolver and resolver.resolve(question, project=None, limit=1))
    except Exception:  # noqa: BLE001 — advisory offline signal
        return False


def _deterministic_lane(question: str) -> bool:
    """True when the typed router routes this question to a deterministic primary lane / hard type."""
    try:
        from src.rag.agent import query_router as qr

        dec = qr.classify_route(question)
        if getattr(dec, "primary_lane", None):
            return True
        return dec.route in {"NUMERIC", "ENUM", "VERSION_DIFF"}
    except Exception:  # noqa: BLE001
        return False


def _version_pair_resolved(question: str, archetype: str) -> bool:
    if archetype != "version_diff":
        return False
    try:
        from src.rag import diffpair

        return diffpair._resolve_pair_for_render(question) is not None
    except Exception:  # noqa: BLE001
        return False


def signals_for_row(row: dict[str, str]) -> GateSignals:
    """Derive an offline :class:`GateSignals` bundle for one gold row (see module docstring for scope)."""
    q = row["question"]
    answered = row["status"] in ANSWERED
    canonical = _registry_resolves(q)
    det_lane = _deterministic_lane(q)

    # Abstained rows carry a recorded state_code → map onto the failure taxonomy to set the epistemic
    # blockers (and to recognise the operational BUDGET_EXHAUSTED, which is NOT a blocker).
    tax = ft.from_ledger_state(row["state_code"]) if row["state_code"] else None
    corpus_absent = tax == ft.CORPUS_ABSENT
    parser_capable = tax != ft.PARSER_CAPABILITY_MISS
    required_unresolved = tax in {ft.EVIDENCE_INCOMPLETE, ft.NOT_RETRIEVED, ft.DOC_RESOLUTION_FAILED}
    budget_exhausted = tax is not None and ft.is_operational(tax)

    return GateSignals(
        canonical_doc_resolved=canonical,
        # A row the champion committed reached consensus agreement to do so — credited equally to MATCH
        # and WRONG (not correctness leakage). Evidence slots are proxied by doc resolution.
        evidence_slots_complete=answered and canonical,
        self_consistency_agrees=answered,
        answer_verifier_agrees=answered,
        version_pair_resolved=_version_pair_resolved(q, row["archetype"]),
        deterministic_lane=det_lane,
        verbal_confidence=_float(row["confidence"]),
        parser_capable=parser_capable,
        corpus_absent=corpus_absent,
        required_evidence_unresolved=required_unresolved,
        budget_exhausted=budget_exhausted,
    )


def _risk_coverage_curve(committed: list[dict]) -> list[dict]:
    """Points on the risk-coverage curve: order answered questions by U desc, sweep the commit prefix."""
    ordered = sorted(committed, key=lambda r: r["utility"], reverse=True)
    n_total = 100  # coverage / expected_score are always over the full gold100 denominator
    points: list[dict] = []
    cum_wrong = cum_score = 0
    for k, rec in enumerate(ordered, start=1):
        if rec["status"] == "WRONG":
            cum_wrong += 1
        cum_score += _SCORE.get(rec["status"], 0.0)
        points.append({
            "coverage": round(k / n_total, 4),
            "committed": k,
            "incorrect_rate_on_answered": round(cum_wrong / k, 4),
            "expected_score": round(cum_score / n_total, 4),
            "utility_threshold": round(rec["utility"], 4),
        })
    return points


def main() -> None:
    rows = _load_rows()
    n = len(rows)
    answered = [r for r in rows if r["status"] in ANSWERED]
    abstained = [r for r in rows if r["status"] == "ABSTAIN"]
    n_answered = len(answered)
    n_wrong = sum(1 for r in answered if r["status"] == "WRONG")
    n_match = sum(1 for r in answered if r["status"] == "MATCH")

    champion = {
        "coverage": round(n_answered / n, 4) if n else 0.0,
        "answered": n_answered,
        "match": n_match,
        "wrong": n_wrong,
        "incorrect_rate_on_answered": round(n_wrong / n_answered, 4) if n_answered else 0.0,
        "expected_score": round((n_match * eu_gate.SCORE_PERFECT + n_wrong * eu_gate.SCORE_INCORRECT) / n, 4)
        if n else 0.0,
    }

    # --- EU gate over the answered pool -------------------------------------------------------------
    committed: list[dict] = []
    kept = caught_wrong = kept_match = 0
    per_question: list[dict] = []
    for r in answered:
        s = signals_for_row(r)
        d = decide(s)
        rec = {"index": r["index"], "status": r["status"], "archetype": r["archetype"],
               "utility": d.utility, "tier": d.tier, "gate_commit": d.commit,
               "confidence": _float(r["confidence"])}
        per_question.append(rec)
        if d.commit:
            kept += 1
            committed.append(rec)
            if r["status"] == "MATCH":
                kept_match += 1
        elif r["status"] == "WRONG":
            caught_wrong += 1

    gate_wrong = sum(1 for r in committed if r["status"] == "WRONG")
    gate_expected = (kept_match * eu_gate.SCORE_PERFECT + gate_wrong * eu_gate.SCORE_INCORRECT) / n if n else 0.0
    mean_u_match = (sum(r["utility"] for r in per_question if r["status"] == "MATCH") / n_match) if n_match else 0.0
    mean_u_wrong = (sum(r["utility"] for r in per_question if r["status"] == "WRONG") / n_wrong) if n_wrong else 0.0

    eu_pool = {
        "kept": kept,
        "newly_abstained": n_answered - kept,
        "kept_match": kept_match,
        "committed_wrong": gate_wrong,
        "caught_wrong": caught_wrong,
        "coverage": round(kept / n, 4) if n else 0.0,
        "incorrect_rate_on_answered": round(gate_wrong / kept, 4) if kept else 0.0,
        "expected_score": round(gate_expected, 4),
        "mean_utility_match": round(mean_u_match, 4),
        "mean_utility_wrong": round(mean_u_wrong, 4),
        "utility_discriminates": mean_u_match > mean_u_wrong,
    }

    # --- abstain taxonomy split ---------------------------------------------------------------------
    codes = [c for c in (ft.from_ledger_state(r["state_code"]) for r in abstained) if c]
    tally = ft.tally(codes)
    abstain_split = {
        "total": len(abstained),
        "operational_budget_exhausted": tally["operational"],   # EU gate does NOT abstain on these
        "epistemic": tally["epistemic"],
        "per_code": tally["per_code"],
        "coverage_expansion_headroom": tally["operational"],    # recoverable if the evidence is reached
    }

    result = {
        "lane_enabled_env": eu_gate.enabled(),
        "n": n,
        "champion_baseline": champion,
        "eu_gate_on_answered": eu_pool,
        "abstain_taxonomy": abstain_split,
        "risk_coverage_curve": _risk_coverage_curve(committed),
        "per_question": sorted(per_question, key=lambda r: r["utility"], reverse=True),
        "notes": (
            "MATCH scored as Perfect(+1) — CSV does not split Perfect/Acceptable, mild upper bound. "
            "Offline signals; runtime verifier/self-consistency proxied by 'champion committed' (equal "
            "to MATCH & WRONG). Diagnostic only — promotion is real-LB gated (ρ=-0.09)."),
    }

    out = settings.ARTIFACTS_DIR / "eu_gate_diagnostics.json"
    out.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    print(f"lane_enabled_env:              {result['lane_enabled_env']}")
    print(f"n:                             {n}")
    print("-- champion baseline (confidence gate) --")
    print(f"  coverage:                    {champion['coverage']}  (answered {n_answered})")
    print(f"  incorrect_rate_on_answered:  {champion['incorrect_rate_on_answered']}")
    print(f"  expected_score:              {champion['expected_score']}")
    print("-- EU gate on answered pool --")
    print(f"  kept / newly_abstained:      {kept} / {n_answered - kept}")
    print(f"  caught_wrong / committed_wrong: {caught_wrong} / {gate_wrong}")
    print(f"  expected_score:              {eu_pool['expected_score']}")
    print(f"  mean U (match/wrong):        {eu_pool['mean_utility_match']} / {eu_pool['mean_utility_wrong']} "
          f"(discriminates={eu_pool['utility_discriminates']})")
    print("-- abstain taxonomy --")
    print(f"  operational(BUDGET):         {abstain_split['operational_budget_exhausted']}  "
          f"(EU gate excludes these from abstain)")
    print(f"  epistemic:                   {abstain_split['epistemic']}  {abstain_split['per_code']}")
    print(f"-> {out}")


if __name__ == "__main__":
    main()
