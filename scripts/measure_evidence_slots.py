#!/usr/bin/env python3
"""SOT-2615 — offline diagnostics for per-route *evidence slot* completion over gold100.

Sibling of ``measure_query_router.py`` (route_accuracy / failure taxonomy) and
``measure_document_registry.py`` (canonical_doc_recall). Those measure *which route* a question gets and
*whether the target document is resolved*; this one measures the next stage — **which evidence slots a
route needs actually got filled** — so that ``BUDGET_EXHAUSTED`` waste can be attributed to a *missing
slot* rather than just "ran out of turns". It is the effect-measurement器 for operand-prefill and the
other SOT-2602 cycle-2 waste-removal axes.

The evidence slots per route are the static research table in ``query_router.ROUTE_SLOTS`` (e.g. NUMERIC
→ operands / unit / operation / rounding). The Evidence Packet (:mod:`evidence_packet`) records which
slots are *missing* at serve time — but only when ``RAG_EVIDENCE_PACKET`` is ON. The SOT-2613 gold100
run had the packet OFF, so slot fills are **not** recorded directly. This harness therefore *approximates*
per-slot completion from the recorded trace (task 指示の fallback):

  * committed answers (MATCH/WRONG) ⇒ every required slot counts as *filled* — the agent could not have
    committed without satisfying its evidence obligation;
  * abstains ⇒ each slot is mapped to its acquisition **stage** (retrieval / extraction / derivation) and
    that slot is *filled* iff the stage succeeded (``*_ok > 0``) in the abstain_ledger signals, *unfilled*
    iff the stage was attempted but never succeeded, *not_reached* iff the stage was never attempted;
  * an abstain with **no** ledger signals ⇒ ``not_measurable`` (no forced estimate).

If a details record ever carries a real per-slot ``evidence`` map (packet ON), those slots are read
*directly* instead of approximated — the harness is forward-compatible with the packet-ON serve path.

Inputs (all under ``artifacts/``; overridable via CLI):
  * ``predictions_test_investigator.details.jsonl`` — per-index contract / tool_calls / stop_reason
  * ``abstain_ledger.jsonl``                       — per-question retrieval/extraction/derivation signals
  * ``gold_100_review.csv``                        — per-index verdict (MATCH/WRONG/ABSTAIN) + state_code

Usage::

    .venv/bin/python scripts/measure_evidence_slots.py

Writes ``artifacts/evidence_slot_diagnostics.json`` (machine) and prints a summary. Deterministic and
network-free (no Gemini/LLM). Diagnostic only — touches nothing on the serve path (scripts/ 配下に閉じる).
"""
from __future__ import annotations

import argparse
import json
import sys
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from config import settings
from src.rag.agent import query_router as qr

# --------------------------------------------------------------------------- slot → acquisition stage
# Each evidence slot is filled at one of three trace stages; the abstain_ledger records a success/attempt
# counter per stage. This mapping is what lets us approximate slot completion when the Evidence Packet
# was OFF, and it deliberately mirrors the budget32 trace classification (A=search-only / B=extract /
# C=compute) so the two diagnostics stay consistent.
STAGE_RETRIEVAL = "retrieval"
STAGE_EXTRACTION = "extraction"
STAGE_DERIVATION = "derivation"

SLOT_STAGE: dict[str, dict[str, str]] = {
    "LOOKUP": {"canonical_doc": STAGE_RETRIEVAL, "answer_span": STAGE_EXTRACTION},
    "NUMERIC": {"operands": STAGE_EXTRACTION, "unit": STAGE_EXTRACTION,
                "operation": STAGE_DERIVATION, "rounding": STAGE_DERIVATION},
    "ENUM": {"universe": STAGE_RETRIEVAL, "filter_predicate": STAGE_EXTRACTION,
             "scan_completion": STAGE_DERIVATION},
    "VERSION_DIFF": {"version_pair": STAGE_RETRIEVAL, "aligned_block": STAGE_EXTRACTION,
                     "substantive_change": STAGE_DERIVATION},
    "FORMAT": {"target_cell_or_run": STAGE_RETRIEVAL, "effective_style": STAGE_EXTRACTION,
               "style_provenance": STAGE_EXTRACTION},
    "PIVOT": {"pivot_identity": STAGE_RETRIEVAL, "field_item_value": STAGE_DERIVATION},
    "EXISTENCE": {"exhaustive_search_coverage": STAGE_RETRIEVAL, "parser_support": STAGE_EXTRACTION},
}

# tool-name → coarse class, mirroring docs/ai/budget32_trace_classification.md (search / extract / compute).
_TOOL_CLASS: dict[str, str] = {
    "file_grep": "search", "find_files": "search", "canonical_route": "search",
    "read_office": "extract", "read_chart_values": "extract", "highlight_extract": "extract",
    "font_emphasis": "extract", "seating_lookup": "extract", "read_pdf": "extract",
    "compute": "compute", "version_diff": "compute", "corpus_aggregate": "compute",
    "pptx_pivot": "compute",
}

_FILLED, _UNFILLED, _NOT_REACHED, _NOT_MEASURABLE = "filled", "unfilled", "not_reached", "not_measurable"


def _norm(text: str) -> str:
    """Normalize a question string for cross-artifact matching (NFC + whitespace-collapsed)."""
    return " ".join(unicodedata.normalize("NFC", str(text or "")).split())


# --------------------------------------------------------------------------- artifact loaders
def _load_details(path: Path) -> dict[int, dict]:
    by_index: dict[int, dict] = {}
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            if "index" in rec:
                by_index[int(rec["index"])] = rec
    return by_index


def _load_ledger_latest(path: Path) -> dict[str, dict]:
    """Newest abstain_ledger record per (normalized) question — the ledger is append-only across runs."""
    latest: dict[str, dict] = {}
    latest_ts: dict[str, str] = {}
    if not path.exists():
        return latest
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            q = _norm(rec.get("question", ""))
            if not q:
                continue
            ts = str(rec.get("recorded_at", ""))
            if q not in latest or ts >= latest_ts.get(q, ""):
                latest[q] = rec
                latest_ts[q] = ts
    return latest


def _load_review(path: Path) -> dict[int, dict]:
    """gold review CSV → per-index {question, verdict, state_code} (optional enrichment)."""
    by_index: dict[int, dict] = {}
    if not path.exists():
        return by_index
    df = pd.read_csv(path, encoding="utf-8-sig")
    for _, r in df.iterrows():
        try:
            idx = int(r["index"])
        except (KeyError, ValueError, TypeError):
            continue
        by_index[idx] = {
            "question": str(r.get("question", "") or ""),
            "verdict": str(r.get("status", "") or "").strip().upper(),
            "state_code": str(r.get("state_code", "") or "").strip(),
        }
    return by_index


# --------------------------------------------------------------------------- slot completion logic
def _route_for(rec: dict, question: str) -> str:
    """Route whose ROUTE_SLOTS apply: the executed contract's route, else classify the question text."""
    contract = str(rec.get("contract", "") or "").strip()
    route = qr._CONTRACT_ROUTE.get(contract)
    if route:
        return route
    try:
        return qr.classify_route(question).route
    except Exception:
        return "LOOKUP"


def _stage_signals(ledger_rec: dict | None) -> dict[str, tuple[int, int]] | None:
    """Aggregate (attempts, ok) per stage from an abstain_ledger record's missing[].signals.

    Returns None when no signals are present at all (⇒ not_measurable)."""
    if not ledger_rec:
        return None
    agg = {STAGE_RETRIEVAL: [0, 0], STAGE_EXTRACTION: [0, 0], STAGE_DERIVATION: [0, 0]}
    seen = False
    for miss in ledger_rec.get("missing", []) or []:
        sig = (miss or {}).get("signals") or {}
        for stage, (a_key, ok_key) in (
            (STAGE_RETRIEVAL, ("retrieval_attempts", "retrieval_ok")),
            (STAGE_EXTRACTION, ("extraction_attempts", "extraction_ok")),
            (STAGE_DERIVATION, ("derivation_attempts", "derivation_ok")),
        ):
            if a_key in sig or ok_key in sig:
                seen = True
                agg[stage][0] += int(sig.get(a_key, 0) or 0)
                agg[stage][1] += int(sig.get(ok_key, 0) or 0)
    if not seen:
        return None
    return {k: (v[0], v[1]) for k, v in agg.items()}


def _direct_slot_map(rec: dict, slots: tuple[str, ...]) -> dict[str, str] | None:
    """If the details record carries a real per-slot evidence map (packet ON), read fills directly."""
    ev = rec.get("evidence")
    if not isinstance(ev, dict):
        return None
    filled = ev.get("evidence") if isinstance(ev.get("evidence"), dict) else ev
    if not isinstance(filled, dict) or not any(s in filled for s in slots):
        return None
    return {s: (_FILLED if filled.get(s) not in (None, "", [], {}) else _UNFILLED) for s in slots}


def _slot_states(route: str, verdict: str, ledger_rec: dict | None,
                 rec: dict) -> tuple[dict[str, str], str]:
    """Per-slot state map + the measurement mode used for this question."""
    slots = qr.ROUTE_SLOTS.get(route, ())
    if not slots:
        return {}, "no_slots"

    direct = _direct_slot_map(rec, slots)
    if direct is not None:
        return direct, "direct_packet"

    # A committed answer means the evidence obligation was met → every required slot is filled.
    if verdict in ("MATCH", "WRONG"):
        return {s: _FILLED for s in slots}, "committed"

    signals = _stage_signals(ledger_rec)
    if signals is None:
        return {s: _NOT_MEASURABLE for s in slots}, _NOT_MEASURABLE

    stage_map = SLOT_STAGE.get(route, {})
    out: dict[str, str] = {}
    for s in slots:
        stage = stage_map.get(s, STAGE_EXTRACTION)
        attempts, ok = signals.get(stage, (0, 0))
        if ok > 0:
            out[s] = _FILLED
        elif attempts > 0:
            out[s] = _UNFILLED
        else:
            out[s] = _NOT_REACHED
    return out, "approx_from_trace"


def _stage_class_signals(ledger_rec: dict | None, rec: dict) -> str:
    """A/B/C stage class from the abstain_ledger stage signals (C=derivation, B=extraction, A=search).

    Falls back to the tool_calls categories when the ledger has no stage signals."""
    signals = _stage_signals(ledger_rec)
    if signals is not None:
        if signals[STAGE_DERIVATION][0] > 0:
            return "C"
        if signals[STAGE_EXTRACTION][0] > 0:
            return "B"
        return "A"
    return _stage_class_toolcalls(rec)


def _stage_class_toolcalls(rec: dict) -> str:
    """A/B/C stage class from details tool_calls — the exact method of budget32_trace_classification.md
    (C=any compute-class tool, B=any extract-class tool, A=search-only)."""
    classes = {_TOOL_CLASS.get(t) for t in (rec.get("tool_calls") or [])}
    if "compute" in classes:
        return "C"
    if "extract" in classes:
        return "B"
    return "A"


# --------------------------------------------------------------------------- aggregation
def _completion_rate(counts: Counter) -> float | None:
    """filled / measurable(=filled+unfilled+not_reached); None when nothing measurable."""
    measurable = counts[_FILLED] + counts[_UNFILLED] + counts[_NOT_REACHED]
    if measurable == 0:
        return None
    return round(counts[_FILLED] / measurable, 4)


def build_report(details: dict[int, dict], ledger: dict[str, dict],
                 review: dict[int, dict]) -> dict:
    per_route_slot: dict[str, dict[str, Counter]] = defaultdict(lambda: defaultdict(Counter))
    per_route_n: Counter = Counter()
    verdict_slot: dict[str, Counter] = defaultdict(Counter)
    verdict_n: Counter = Counter()
    mode_counts: Counter = Counter()

    budget_unfilled: Counter = Counter()   # slot -> times unfilled/not_reached among BUDGET_EXHAUSTED
    budget_by_route: Counter = Counter()
    budget_stage_class: Counter = Counter()
    budget_stage_class_tc: Counter = Counter()
    budget_not_measurable = 0
    budget_n = 0

    indexes = sorted(set(details) | set(review))
    for idx in indexes:
        rec = details.get(idx, {})
        meta = review.get(idx, {})
        question = meta.get("question") or rec.get("question") or ""
        verdict = meta.get("verdict", "")
        state_code = meta.get("state_code", "")
        if not verdict:
            # No review row: infer committed vs abstain from the trace.
            verdict = "ABSTAIN" if _norm(question) in ledger else "COMMITTED"

        route = _route_for(rec, question)
        ledger_rec = ledger.get(_norm(question))
        states, mode = _slot_states(route, verdict, ledger_rec, rec)
        mode_counts[mode] += 1
        per_route_n[route] += 1
        verdict_n[verdict] += 1
        for slot, st in states.items():
            per_route_slot[route][slot][st] += 1
            verdict_slot[verdict][st] += 1

        if state_code == "BUDGET_EXHAUSTED":
            budget_n += 1
            budget_by_route[route] += 1
            budget_stage_class[_stage_class_signals(ledger_rec, rec)] += 1
            budget_stage_class_tc[_stage_class_toolcalls(rec)] += 1
            if mode == _NOT_MEASURABLE:
                budget_not_measurable += 1
            for slot, st in states.items():
                if st in (_UNFILLED, _NOT_REACHED):
                    budget_unfilled[f"{route}:{slot}"] += 1

    per_route = {}
    for route, slot_counts in sorted(per_route_slot.items()):
        slots_out = {}
        for slot, counts in slot_counts.items():
            slots_out[slot] = {
                "completion_rate": _completion_rate(counts),
                "counts": dict(counts),
            }
        per_route[route] = {
            "label": qr.ROUTE_LABELS.get(route, route),
            "n_questions": per_route_n[route],
            "slots": slots_out,
        }

    verdict_cross = {}
    for verdict, counts in sorted(verdict_slot.items()):
        verdict_cross[verdict] = {
            "n_questions": verdict_n[verdict],
            "slot_completion_rate": _completion_rate(counts),
            "counts": dict(counts),
        }

    return {
        "per_route": per_route,
        "verdict_cross": verdict_cross,
        "budget_exhausted": {
            "n": budget_n,
            "by_route": dict(budget_by_route.most_common()),
            "unfilled_slot_distribution": dict(budget_unfilled.most_common()),
            "stage_class_distribution": dict(sorted(budget_stage_class.items())),
            "stage_class_distribution_toolcalls": dict(sorted(budget_stage_class_tc.items())),
            "not_measurable": budget_not_measurable,
        },
        "measurement_modes": dict(mode_counts.most_common()),
        "n_questions": len(indexes),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ad = settings.ARTIFACTS_DIR
    ap.add_argument("--details", type=Path, default=ad / "predictions_test_investigator.details.jsonl")
    ap.add_argument("--ledger", type=Path, default=ad / "abstain_ledger.jsonl")
    ap.add_argument("--review", type=Path, default=ad / "gold_100_review.csv")
    ap.add_argument("--out", type=Path, default=ad / "evidence_slot_diagnostics.json")
    args = ap.parse_args()

    if not args.details.exists():
        raise SystemExit(f"details artifact not found: {args.details}")

    details = _load_details(args.details)
    ledger = _load_ledger_latest(args.ledger)
    review = _load_review(args.review)

    report = build_report(details, ledger, review)
    report["sources"] = {
        "details": str(args.details),
        "ledger": str(args.ledger) if args.ledger.exists() else None,
        "review": str(args.review) if args.review.exists() else None,
    }
    report["note"] = (
        "Slot completion is approximated from the recorded trace when the Evidence Packet was OFF "
        "(RAG_EVIDENCE_PACKET); committed answers count all slots filled, abstains map each slot to its "
        "retrieval/extraction/derivation stage signal, and abstains with no signals are not_measurable. "
        "No serve-path code is touched (diagnostic only)."
    )

    args.out.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    be = report["budget_exhausted"]
    print(f"n_questions: {report['n_questions']}")
    print(f"measurement_modes: {report['measurement_modes']}")
    print(f"verdict_cross: " + ", ".join(
        f"{v}(n={d['n_questions']} slot_completion={d['slot_completion_rate']})"
        for v, d in report["verdict_cross"].items()))
    print(f"BUDGET_EXHAUSTED n={be['n']} by_route={be['by_route']} "
          f"stage_class={be['stage_class_distribution']} not_measurable={be['not_measurable']}")
    print(f"BUDGET_EXHAUSTED top unfilled slots: "
          f"{dict(list(be['unfilled_slot_distribution'].items())[:8])}")
    print(f"-> {args.out}")


if __name__ == "__main__":
    main()
