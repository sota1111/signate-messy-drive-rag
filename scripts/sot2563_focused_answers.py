"""SOT-2563 focused answer verification (review=human follow-up).

Runs the timeout-abstain idx group (the NG items PR#101's per-call file_grep deadline was meant to
recover) through the *production* investigator with the merged default guard
(``RAG_FILE_GREP_MAX_SCAN_S=180``), captures per-question wall-clock + stop_reason, then scores the
answers against gold with the same offline CRAG (codex) judge the scoring stack uses.

Acceptance signals:
  (2a) single-question elapsed is bounded (no 280-658s runaway; stop_reason no longer ``timeout`` on
       the single-file_grep pattern-A group);
  (2b) the timeout abstains are recovered -> ideally ``Perfect|Acceptable`` (match), at minimum the
       ``timeout`` stop_reason is gone.

Gemini-only production path — no corpus-specific facts injected. gold100 is NOT run here.

    RAG_FILE_GREP_MAX_SCAN_S=180 .venv/bin/python scripts/sot2563_focused_answers.py
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

# Pattern A (single slow file_grep, iter<=2) + Pattern B (spin, iter>=10) from
# docs/ai/abstain_wrong_root_cause_SOT-2550.md §3.
PATTERN_A = [23, 38, 48, 55, 67, 76, 37, 41, 43, 50, 53, 70, 98]
PATTERN_B = [40, 73, 63, 27, 28, 39]
_requested = os.environ.get("SOT2563_INDICES", "").strip()
INDICES = ([int(part) for part in _requested.split(",") if part.strip()]
           if _requested else PATTERN_A + PATTERN_B)

WORKERS = int(os.environ.get("SOT2563_WORKERS", "6"))
OUTPUT_TAG = os.environ.get("SOT2563_OUTPUT_TAG", "sot2563_focused_answers")
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
        inv = investigator.answer_question(q).to_dict()
        err = None
    except Exception as exc:  # noqa: BLE001 — record, do not abort the batch
        inv = {"answer": settings.ABSTAIN, "stop_reason": "harness_error",
               "iterations": 0, "tool_calls": [], "elapsed_s": 0.0}
        err = f"{type(exc).__name__}: {exc}"
    wall = time.monotonic() - t0
    return idx, {
        "index": idx,
        "question": q,
        "answer": inv.get("answer", ""),
        "method": inv.get("method", ""),
        "evidence": inv.get("evidence", ""),
        "iterations": inv.get("iterations", 0),
        "tool_calls": inv.get("tool_calls", []),
        "stop_reason": inv.get("stop_reason", ""),
        "elapsed_s": round(inv.get("elapsed_s", wall), 1),
        "wall_s": round(wall, 1),
        "error": err,
    }


def main() -> int:
    _inject_timeout()
    cap = os.environ.get("RAG_FILE_GREP_MAX_SCAN_S", "(default 180)")
    print(f"[sot2563] RAG_FILE_GREP_MAX_SCAN_S={cap} workers={WORKERS} "
          f"n={len(INDICES)} A={len(PATTERN_A)} B={len(PATTERN_B)}", flush=True)

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
            print(f"[{n}/{len(INDICES)}] idx={idx} stop={rec['stop_reason']} "
                  f"elapsed={rec['elapsed_s']}s iters={rec['iterations']} "
                  f"tools={rec['tool_calls']}\n   ans: {str(rec['answer'])[:80]}\n"
                  f"   gold: {str(rec['gold'])[:80]}", flush=True)

    # Judge with the same offline CRAG (codex) judge.
    ordered = [i for i in INDICES if i in results]
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
    n_timeout = sum(1 for i in ordered if results[i]["stop_reason"] == "timeout")
    max_elapsed = max((results[i]["elapsed_s"] for i in ordered), default=0.0)

    print("\n==== SOT-2563 focused answer verification ====", flush=True)
    for i in ordered:
        r = results[i]
        print(f"idx {i:>3} [{r['bucket']:>7}] verdict={r['verdict']:<10} "
              f"stop={r['stop_reason']:<12} elapsed={r['elapsed_s']:>6}s :: {str(r['answer'])[:60]}",
              flush=True)
    print(f"\nTOTAL n={len(ordered)}  match={n_match}  abstain={n_abstain}  wrong={n_wrong}",
          flush=True)
    print(f"timeout stop_reason remaining={n_timeout}  max_elapsed={max_elapsed}s", flush=True)

    out = settings.ARTIFACTS_DIR / f"{OUTPUT_TAG}.json"
    out.write_text(json.dumps({
        "indices": ordered, "pattern_a": PATTERN_A, "pattern_b": PATTERN_B,
        "n": len(ordered), "match": n_match, "abstain": n_abstain, "wrong": n_wrong,
        "timeout_remaining": n_timeout, "max_elapsed_s": max_elapsed,
        "file_grep_max_scan_s": cap, "results": results,
        "file_grep_reserve_s": os.environ.get("RAG_FILE_GREP_RESERVE_S", "(default 30)"),
        "evidence_index": os.environ.get("RAG_EVIDENCE_INDEX", "0"),
        "index_candidates": os.environ.get("RAG_FILE_GREP_INDEX_CANDIDATES", "0"),
    }, ensure_ascii=False, indent=2), encoding="utf-8")

    details = settings.ARTIFACTS_DIR / f"{OUTPUT_TAG}.details.jsonl"
    with details.open("w", encoding="utf-8") as fh:
        for i in ordered:
            fh.write(json.dumps({**results[i]}, ensure_ascii=False) + "\n")
    print(f"\nwrote {out}\nwrote {details}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
