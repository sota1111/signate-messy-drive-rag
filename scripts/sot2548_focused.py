"""SOT-2548 focused verification — run idx78/idx80 through the production investigator with the
extraction-failure-fix flags ON, then score against gold with the same offline judge.

    RAG_XLSX_EMBEDDED_IMAGE=1 python scripts/sot2548_focused.py

idx80 needs the embedded-picture reader (RAG_XLSX_EMBEDDED_IMAGE); idx78 exercises the widened
regulation-completeness trigger (no flag). Prints, per index: the committed answer, the gold, and the
judge verdict. match = Perfect|Acceptable. No corpus-specific facts are injected — Gemini-only path.
A 180s request timeout is injected into the google-genai client so a half-open Vertex connection cannot
hang the run (repo answer logic unchanged).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from config import settings
from scoring import gold_offline
from src.rag import llm
from src.rag.agent import investigator

INDICES = [78, 80]


def _inject_timeout() -> None:
    """Rebuild the genai client with a bounded HTTP timeout (genai default is None → can hang)."""
    from google import genai
    from google.genai import types

    llm._client = genai.Client(
        vertexai=True, project=settings.GCP_PROJECT_ID, location=settings.VERTEX_LOCATION,
        http_options=types.HttpOptions(timeout=180_000))


def main() -> int:
    _inject_timeout()
    qbyidx = {int(r["index"]): str(r["question"])
              for _, r in pd.read_csv(settings.QUESTIONS_TEST).iterrows()}
    gold = gold_offline.load_gold(settings.ARTIFACTS_DIR / "predictions_test_v3_final.csv")

    results = {}
    for idx in INDICES:
        q = qbyidx[idx]
        print(f"\n=== idx {idx} :: {q}", flush=True)
        inv = investigator.answer_question(q).to_dict()
        ans = inv.get("answer", "")
        results[idx] = {"index": idx, "question": q, "answer": ans,
                        "gold": gold.get(idx, ""),
                        "method": inv.get("method", ""),
                        "tool_calls": inv.get("tool_calls", []),
                        "stop_reason": inv.get("stop_reason", "")}
        print(f"  answer: {ans}", flush=True)
        print(f"  gold  : {gold.get(idx,'')}", flush=True)
        print(f"  tools : {inv.get('tool_calls',[])} stop={inv.get('stop_reason','')}", flush=True)

    pairs = [(results[i]["answer"], results[i]["gold"]) for i in INDICES]
    verdicts = gold_offline.default_judge(pairs)
    match = {"Perfect", "Acceptable"}
    n_match = 0
    for i, verdict in zip(INDICES, verdicts):
        results[i]["verdict"] = verdict
        ok = verdict in match
        n_match += int(ok)
        print(f"\nidx {i}: verdict={verdict} {'MATCH' if ok else 'NOT-MATCH'}", flush=True)

    out = settings.ARTIFACTS_DIR / "sot2548_focused.json"
    out.write_text(json.dumps({"results": results, "n_match": n_match,
                               "n_total": len(INDICES)}, ensure_ascii=False, indent=2),
                   encoding="utf-8")
    print(f"\nFOCUSED RESULT: {n_match}/{len(INDICES)} match  -> {out}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
