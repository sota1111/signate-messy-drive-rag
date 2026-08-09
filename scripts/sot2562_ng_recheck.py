"""SOT-2562 (review=human follow-up) — re-check the 6 NG (wrong) gold-100 items.

The last integrated gold-100 (SOT-2550, 2026-08-09 04:15) left wrong={4,9,14,49,83,92}.
Main has since merged SOT-2563 (file_grep deadline) and SOT-2564 (highlight / font-emphasis
extraction). This harness re-runs *just those 6 items* through the production investigator with the
A1-E candidate flag set (+ the newer format/candidate flags) and scores them with the same offline
CRAG judge the scoring stack uses, so we can see which NG items now pass, which are still wrong, and
their failure mode — before deciding on per-idx re-fixes.

gold100 is NOT run here. Gemini-only production path; no corpus facts injected.

    .venv/bin/python scripts/sot2562_ng_recheck.py
"""
from __future__ import annotations

import json
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from config import settings
from scoring import gold_offline
from src.rag import llm
from src.rag.agent import investigator

NG = [int(x) for x in os.environ.get("NG_INDICES", "4,9,14,49,83,92").split(",") if x.strip()]
WORKERS = int(os.environ.get("NG_WORKERS", "3"))
OUTPUT_TAG = os.environ.get("NG_OUTPUT_TAG", "sot2562_ng_recheck")
MATCH = {"Perfect", "Acceptable"}


def _inject_timeout() -> None:
    from google import genai
    from google.genai import types

    llm._client = genai.Client(
        vertexai=True, project=settings.GCP_PROJECT_ID, location=settings.VERTEX_LOCATION,
        http_options=types.HttpOptions(timeout=180_000))


def _run_one(idx: int, q: str) -> tuple[int, dict]:
    t0 = time.monotonic()
    try:
        inv = investigator.answer_question(q).to_dict()
        err = None
    except Exception as exc:  # noqa: BLE001
        inv = {"answer": settings.ABSTAIN, "stop_reason": "harness_error",
               "iterations": 0, "tool_calls": [], "elapsed_s": 0.0}
        err = f"{type(exc).__name__}: {exc}"
    wall = time.monotonic() - t0
    return idx, {
        "index": idx, "question": q,
        "answer": inv.get("answer", ""), "method": inv.get("method", ""),
        "evidence": str(inv.get("evidence", ""))[:600],
        "iterations": inv.get("iterations", 0),
        "tool_calls": inv.get("tool_calls", []),
        "stop_reason": inv.get("stop_reason", ""),
        "elapsed_s": round(inv.get("elapsed_s", wall), 1),
        "wall_s": round(wall, 1), "error": err,
    }


def main() -> int:
    _inject_timeout()
    print(f"[ng] n={len(NG)} idx={NG} workers={WORKERS}", flush=True)
    for k in sorted(os.environ):
        if k.startswith("RAG_") or k.startswith("GATE_"):
            print(f"  {k}={os.environ[k]}", flush=True)

    qbyidx = {int(r["index"]): str(r["question"])
              for _, r in pd.read_csv(settings.QUESTIONS_TEST).iterrows()}
    gold = gold_offline.load_gold(settings.ARTIFACTS_DIR / "predictions_test_v3_final.csv")

    results: dict[int, dict] = {}
    with ThreadPoolExecutor(max_workers=WORKERS) as ex:
        futs = {ex.submit(_run_one, i, qbyidx[i]): i for i in NG}
        for n, fut in enumerate(as_completed(futs), 1):
            idx, rec = fut.result()
            rec["gold"] = gold.get(idx, "")
            results[idx] = rec
            print(f"[{n}/{len(NG)}] idx={idx} stop={rec['stop_reason']} "
                  f"elapsed={rec['elapsed_s']}s iters={rec['iterations']} tools={rec['tool_calls']}\n"
                  f"   ans : {str(rec['answer'])[:100]}\n   gold: {str(rec['gold'])[:100]}", flush=True)

    ordered = [i for i in NG if i in results]
    pairs = [(results[i]["answer"], results[i]["gold"]) for i in ordered]
    verdicts = gold_offline.default_judge(pairs)
    for i, verdict in zip(ordered, verdicts):
        results[i]["verdict"] = verdict

    def _bucket(i: int) -> str:
        v = results[i]["verdict"]
        a = str(results[i]["answer"]).strip()
        if v in MATCH:
            return "match"
        if a == "" or a == settings.ABSTAIN or v == "Missing":
            return "abstain"
        return "wrong"

    for i in ordered:
        results[i]["bucket"] = _bucket(i)

    n_match = sum(1 for i in ordered if results[i]["bucket"] == "match")
    n_abstain = sum(1 for i in ordered if results[i]["bucket"] == "abstain")
    n_wrong = sum(1 for i in ordered if results[i]["bucket"] == "wrong")

    print("\n==== SOT-2562 NG re-check ====", flush=True)
    for i in ordered:
        r = results[i]
        print(f"idx {i:>3} [{r['bucket']:>7}] verdict={r['verdict']:<10} stop={r['stop_reason']:<12} "
              f"elapsed={r['elapsed_s']:>6}s\n    ans : {str(r['answer'])[:90]}\n    gold: {str(r['gold'])[:90]}",
              flush=True)
    print(f"\nTOTAL n={len(ordered)}  match={n_match}  abstain={n_abstain}  wrong={n_wrong}", flush=True)

    out = settings.ARTIFACTS_DIR / f"{OUTPUT_TAG}.json"
    out.write_text(json.dumps({
        "indices": ordered, "n": len(ordered),
        "match": n_match, "abstain": n_abstain, "wrong": n_wrong,
        "flags": {k: os.environ[k] for k in sorted(os.environ)
                  if k.startswith("RAG_") or k.startswith("GATE_")},
        "results": results,
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nwrote {out}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
