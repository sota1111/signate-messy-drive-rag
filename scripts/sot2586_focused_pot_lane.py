"""SOT-2586 focused three-layer measurement for the NUMERIC PoT forced lane.

Runs the ``derived_calculation`` gold100 archetype (the weakest type, 32 questions) through the
*production* investigator with ``RAG_POT_HARD_LANE=1`` so the ``verify_formula`` forced lane
(binder→制限AST→Decimal→独立検算→N-sample majority) is exercised live, then records the SOT-2586
three-layer metrics — operand_binding_accuracy / formula_accuracy / execution_accuracy /
verifier_disagreement_rate — separating an A-type failure (operand が届かない) from a B-type failure
(計算を間違える), rather than judging only the final-answer match (Acceptance #4).

The details.jsonl written here carries, per question, the full ``Investigation.to_dict()`` — which now
includes the retained ``pot_lane`` verdict (SOT-2586 propagation fix) — so
``scripts/measure_pot_lane.py --details <this>.details.jsonl`` aggregates the three layers.

    RAG_POT_HARD_LANE=1 .venv/bin/python scripts/sot2586_focused_pot_lane.py

Env knobs: ``SOT2586_INDICES`` (comma list to override), ``SOT2586_WORKERS`` (default 4),
``SOT2586_OUTPUT_TAG`` (default sot2586_focused_pot_lane). Gemini-only production path — no
corpus-specific facts injected. Only the derived_calculation archetype is run, NOT all of gold100.
"""
from __future__ import annotations

import json
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

# The lane's ``POT_HARD_LANE`` module constant is read at import time — force the flag on BEFORE any
# investigator import so the forced lane is both registered (tool) and directive-injected (NUMERIC route).
os.environ.setdefault("RAG_POT_HARD_LANE", "1")

import pandas as pd  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from config import settings  # noqa: E402
from scoring import gold_offline  # noqa: E402
from src.rag import llm  # noqa: E402
from src.rag.agent import investigator  # noqa: E402
from src.rag.agent import pot_lane as pl  # noqa: E402

# The 32 derived_calculation indices from the last gold-100 review (artifacts/gold_100_review.csv).
DERIVED = [4, 5, 6, 8, 10, 17, 25, 27, 28, 30, 35, 37, 40, 41, 47, 50, 53, 56, 57, 63,
           64, 65, 68, 72, 76, 83, 86, 90, 91, 92, 97, 99]
_requested = os.environ.get("SOT2586_INDICES", "").strip()
INDICES = ([int(p) for p in _requested.split(",") if p.strip()] if _requested else DERIVED)

WORKERS = int(os.environ.get("SOT2586_WORKERS", "4"))
OUTPUT_TAG = os.environ.get("SOT2586_OUTPUT_TAG", "sot2586_focused_pot_lane")
MATCH = {"Perfect", "Acceptable"}


def _inject_timeout() -> None:
    """Rebuild the genai client with a bounded HTTP timeout (genai default None can hang)."""
    from google import genai
    from google.genai import types

    llm._client = genai.Client(
        vertexai=True, project=settings.GCP_PROJECT_ID, location=settings.VERTEX_LOCATION,
        http_options=types.HttpOptions(timeout=180_000))


def _run_one(idx: int, q: str) -> tuple[int, dict]:
    t0 = time.monotonic()
    try:
        # keep the FULL to_dict() so the retained pot_lane verdict survives into details.jsonl
        rec = investigator.answer_question(q).to_dict()
        err = None
    except Exception as exc:  # noqa: BLE001 — record, do not abort the batch
        rec = {"answer": settings.ABSTAIN, "stop_reason": "harness_error",
               "iterations": 0, "tool_calls": [], "elapsed_s": 0.0}
        err = f"{type(exc).__name__}: {exc}"
    wall = time.monotonic() - t0
    rec["index"] = idx
    rec["question"] = q
    rec["wall_s"] = round(wall, 1)
    if err:
        rec["error"] = err
    return idx, rec


def _lane_layers(rec: dict) -> dict | None:
    """Extract the chosen candidate's three-layer verdict booleans for a per-question print."""
    verdict = rec.get("pot_lane")
    if not isinstance(verdict, dict) or not verdict.get("candidates"):
        return None
    vd = (verdict["candidates"][0] or {}).get("verdicts", {})
    return {
        "operand": bool(vd.get(pl.LAYER_OPERAND, {}).get("ok")),
        "formula": bool(vd.get(pl.LAYER_FORMULA, {}).get("ok")),
        "execution": bool(vd.get(pl.LAYER_EXECUTION, {}).get("ok")),
        "status": (verdict.get("decision") or {}).get("status"),
    }


def main() -> int:
    _inject_timeout()
    print(f"[sot2586] RAG_POT_HARD_LANE={os.environ.get('RAG_POT_HARD_LANE')} "
          f"lane.enabled={pl.enabled()} sympy={pl.have_sympy()} workers={WORKERS} "
          f"n={len(INDICES)} (derived_calculation)", flush=True)

    qbyidx = {int(r["index"]): str(r["question"])
              for _, r in pd.read_csv(settings.QUESTIONS_TEST).iterrows()}
    gold = gold_offline.load_gold(settings.ARTIFACTS_DIR / "predictions_test_v3_final.csv")

    results: dict[int, dict] = {}
    with ThreadPoolExecutor(max_workers=WORKERS) as ex:
        futs = {ex.submit(_run_one, i, qbyidx[i]): i for i in INDICES}
        for n, fut in enumerate(as_completed(futs), 1):
            idx, rec = fut.result()
            rec["gold"] = gold.get(idx, "")
            results[idx] = rec
            layers = _lane_layers(rec)
            print(f"[{n}/{len(INDICES)}] idx={idx} stop={rec.get('stop_reason')} "
                  f"elapsed={rec.get('elapsed_s')}s lane={layers}\n"
                  f"   ans: {str(rec.get('answer'))[:80]}\n"
                  f"   gold: {str(rec.get('gold'))[:80]}", flush=True)

    ordered = [i for i in INDICES if i in results]

    # Final-answer accuracy (same offline CRAG judge the scoring stack uses).
    pairs = [(results[i].get("answer", ""), results[i].get("gold", "")) for i in ordered]
    verdicts = gold_offline.default_judge(pairs)
    for i, verdict in zip(ordered, verdicts):
        results[i]["verdict"] = verdict

    def _bucket(i: int) -> str:
        v = results[i].get("verdict")
        a = str(results[i].get("answer", "")).strip()
        if v in MATCH:
            return "match"
        if a == "" or a == settings.ABSTAIN or v == "Missing":
            return "abstain"
        return "wrong"

    for i in ordered:
        results[i]["bucket"] = _bucket(i)

    # Write the details.jsonl FIRST (carries pot_lane), then aggregate the three layers off it.
    details = settings.ARTIFACTS_DIR / f"{OUTPUT_TAG}.details.jsonl"
    with details.open("w", encoding="utf-8") as fh:
        for i in ordered:
            fh.write(json.dumps(results[i], ensure_ascii=False) + "\n")

    # Three-layer aggregation over the retained lane verdicts.
    lane_recs = [_lane_layers(results[i]) for i in ordered]
    lane_recs = [r for r in lane_recs if r is not None]
    n_lane = len(lane_recs)
    three_layer = {"traces": n_lane}
    if n_lane:
        three_layer.update({
            "operand_binding_accuracy": round(sum(r["operand"] for r in lane_recs) / n_lane, 4),
            "formula_accuracy": round(sum(r["formula"] for r in lane_recs) / n_lane, 4),
            "execution_accuracy": round(sum(r["execution"] for r in lane_recs) / n_lane, 4),
            "commit_rate": round(sum(r["status"] == pl.COMMIT for r in lane_recs) / n_lane, 4),
        })

    n_match = sum(1 for i in ordered if results[i]["bucket"] == "match")
    n_abstain = sum(1 for i in ordered if results[i]["bucket"] == "abstain")
    n_wrong = sum(1 for i in ordered if results[i]["bucket"] == "wrong")

    print("\n==== SOT-2586 derived_calculation three-layer measurement ====", flush=True)
    for i in ordered:
        r = results[i]
        print(f"idx {i:>3} [{r['bucket']:>7}] verdict={str(r.get('verdict')):<10} "
              f"lane={_lane_layers(r)} :: {str(r.get('answer'))[:50]}", flush=True)
    print(f"\nTOTAL n={len(ordered)}  match={n_match}  abstain={n_abstain}  wrong={n_wrong}", flush=True)
    print(f"lane traces={n_lane}  three_layer={three_layer}", flush=True)

    out = settings.ARTIFACTS_DIR / f"{OUTPUT_TAG}.json"
    out.write_text(json.dumps({
        "indices": ordered, "n": len(ordered),
        "match": n_match, "abstain": n_abstain, "wrong": n_wrong,
        "lane_enabled": pl.enabled(), "sympy_backend": pl.have_sympy(),
        "three_layer": three_layer, "results": results,
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nwrote {out}\nwrote {details}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
