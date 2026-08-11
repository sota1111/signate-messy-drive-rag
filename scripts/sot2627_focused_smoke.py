"""SOT-2627 — focused claude-mcp vs Gemini reachability smoke (dev-only, not a gold100 run).

Runs a small set of representative questions through the claude-mcp investigator backend and records a
details.jsonl compatible row per question (so scoring.gold_offline can score it). Prints a compact
per-question reachability line (stop_reason / iterations / tool_calls / elapsed / answer preview).
"""
from __future__ import annotations

import csv
import json
import os
import sys
import time
from pathlib import Path

from config import settings
from src.rag.agent import investigator


def _questions() -> dict[int, str]:
    p = settings.QUESTIONS_TEST
    return {int(r["index"]): str(r["question"])
            for r in csv.DictReader(open(p, encoding="utf-8-sig"))}


def main() -> int:
    idxs = [int(x) for x in sys.argv[1:]] or [24]
    backend = os.getenv("RAG_INVESTIGATOR_BACKEND", "gemini")
    tag = "claude_mcp" if backend == "claude-mcp" else backend
    out = settings.ARTIFACTS_DIR / f"sot2627_focused_{tag}.details.jsonl"
    qs = _questions()
    rows = []
    for i in idxs:
        q = qs[i]
        t0 = time.monotonic()
        inv = investigator.answer_question(q, timeout_s=float(os.getenv("SMOKE_TIMEOUT_S", "240")))
        d = inv.to_dict()
        d = {"index": i, **d}
        rows.append(d)
        print(f"[idx {i}] stop={d['stop_reason']} iters={d['iterations']} "
              f"tools={d['tool_calls']} elapsed={d['elapsed_s']}s model={d['model']} "
              f"cost={d['cost_usd']}\n         Q: {q[:80]}\n         A: {d['answer'][:120]}",
              flush=True)
        with out.open("w", encoding="utf-8") as fh:
            for r in rows:
                fh.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"\nwrote {out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
