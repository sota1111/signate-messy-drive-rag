"""SOT-2620 focused OFF/ON verification for the per-route search-call cap (RAG_SEARCH_CAP).

Runs the phase-0 search-heavy focused indices (search≥12 in the BUDGET_EXHAUSTED-32 trace) through the
*production* investigator twice — cap OFF then cap ON — in the SAME process, and reports per question:

    search_off / search_on   : # of search-style tool calls (file_grep/find_files) in tool_calls
    cap_hits                  : Investigation.search_cap_hits (over-cap calls intercepted)
    bucket_off / bucket_on    : match|abstain|wrong (offline CRAG judge vs gold)

then the aggregate the acceptance asks for: total search calls (must drop ON), and wrong count
(must NOT increase). Gemini-only production path (the investigator function-calling is Gemini-fixed);
no corpus-specific facts injected. Only the focused indices are run, NOT all of gold100 (that is the
SOT-2622 integration measurement this issue blocks).

    .venv/bin/python scripts/sot2620_focused_search_cap.py
    SOT2620_INDICES=76,50 SOT2620_WORKERS=4 .venv/bin/python scripts/sot2620_focused_search_cap.py
"""
from __future__ import annotations

import json
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import pandas as pd  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from config import settings  # noqa: E402
from scoring import gold_offline  # noqa: E402
from src.rag import llm  # noqa: E402
from src.rag.agent import investigator  # noqa: E402

# The search≥12 focused indices from docs/ai/budget32_trace_classification.md (issue 検証内容).
FOCUSED = [76, 50, 99, 67, 38, 87, 32, 63, 83, 98]
_requested = os.environ.get("SOT2620_INDICES", "").strip()
INDICES = ([int(p) for p in _requested.split(",") if p.strip()] if _requested else FOCUSED)
WORKERS = int(os.environ.get("SOT2620_WORKERS", "4"))
SEARCH_TOOLS = investigator._SEARCH_TOOLS
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
        rec = investigator.answer_question(q).to_dict()
        err = None
    except Exception as exc:  # noqa: BLE001 — record, do not abort the batch
        rec = {"answer": settings.ABSTAIN, "stop_reason": "harness_error",
               "iterations": 0, "tool_calls": []}
        err = f"{type(exc).__name__}: {exc}"
    rec["index"] = idx
    rec["question"] = q
    rec["wall_s"] = round(time.monotonic() - t0, 1)
    if err:
        rec["error"] = err
    return idx, rec


def _search_attempts(rec: dict) -> int:
    """Search-tool entries in tool_calls (includes over-cap attempts, which are recorded then withheld)."""
    return sum(1 for c in rec.get("tool_calls", []) if c in SEARCH_TOOLS)


def _search_count(rec: dict) -> int:
    """Actually-*dispatched* search calls = attempts minus the intercepted (over-cap) ones.

    ``tool_calls.append`` runs before the cap/pivot guards, so a withheld call is still recorded in
    tool_calls; subtracting ``search_cap_hits`` gives the real number of search executions — the metric
    the acceptance ("search呼び数が上限内") is about."""
    return _search_attempts(rec) - int(rec.get("search_cap_hits", 0))


def _run_phase(label: str, indices: list[int], qbyidx: dict[int, str]) -> dict[int, dict]:
    print(f"\n=== phase {label} (RAG_SEARCH_CAP={os.environ.get('RAG_SEARCH_CAP')}) ===", flush=True)
    results: dict[int, dict] = {}
    with ThreadPoolExecutor(max_workers=WORKERS) as ex:
        futs = {ex.submit(_run_one, i, qbyidx[i]): i for i in indices}
        for n, fut in enumerate(as_completed(futs), 1):
            idx, rec = fut.result()
            results[idx] = rec
            print(f"[{label} {n}/{len(indices)}] idx={idx} stop={rec.get('stop_reason')} "
                  f"search={_search_count(rec)} cap_hits={rec.get('search_cap_hits', 0)} "
                  f"wall={rec.get('wall_s')}s ans={str(rec.get('answer'))[:48]}", flush=True)
    return results


def _bucket(rec: dict, gold: str, verdict: str) -> str:
    a = str(rec.get("answer", "")).strip()
    if verdict in MATCH:
        return "match"
    if a == "" or a == settings.ABSTAIN or verdict == "Missing":
        return "abstain"
    return "wrong"


def main() -> int:
    _inject_timeout()
    qbyidx = {int(r["index"]): str(r["question"])
              for _, r in pd.read_csv(settings.QUESTIONS_TEST).iterrows()}
    gold = gold_offline.load_gold(settings.ARTIFACTS_DIR / "predictions_test_v3_final.csv")
    indices = [i for i in INDICES if i in qbyidx]

    # Phase OFF must set the flag off BEFORE the module reads it. answer_question reads the module-level
    # SEARCH_CAP constant, so flip it directly on the imported module for a clean in-process A/B.
    os.environ["RAG_SEARCH_CAP"] = "0"
    investigator.SEARCH_CAP = False
    off = _run_phase("OFF", indices, qbyidx)

    os.environ["RAG_SEARCH_CAP"] = "1"
    investigator.SEARCH_CAP = True
    on = _run_phase("ON", indices, qbyidx)

    # Judge both phases against gold with the same offline CRAG judge.
    def _verdicts(res: dict[int, dict]) -> dict[int, str]:
        pairs = [(res[i].get("answer", ""), gold.get(i, "")) for i in indices]
        vs = gold_offline.default_judge(pairs)
        return {i: v for i, v in zip(indices, vs)}

    voff, von = _verdicts(off), _verdicts(on)

    print("\n=== per-question ===", flush=True)
    print(f"{'idx':>4} {'s_off':>5} {'s_on':>5} {'hits':>4}  {'off':<7} {'on':<7}", flush=True)
    tot_off = tot_on = wrong_off = wrong_on = 0
    for i in indices:
        so, sn = _search_count(off[i]), _search_count(on[i])
        bo = _bucket(off[i], gold.get(i, ""), voff[i])
        bn = _bucket(on[i], gold.get(i, ""), von[i])
        tot_off += so
        tot_on += sn
        wrong_off += (bo == "wrong")
        wrong_on += (bn == "wrong")
        print(f"{i:>4} {so:>5} {sn:>5} {on[i].get('search_cap_hits', 0):>4}  {bo:<7} {bn:<7}",
              flush=True)

    def _dist(res, v):
        b = [_bucket(res[i], gold.get(i, ""), v[i]) for i in indices]
        return {k: b.count(k) for k in ("match", "abstain", "wrong")}

    print("\n=== aggregate ===", flush=True)
    print(f"search calls: OFF={tot_off}  ON={tot_on}  (Δ={tot_on - tot_off})", flush=True)
    print(f"buckets OFF: {_dist(off, voff)}", flush=True)
    print(f"buckets ON : {_dist(on, von)}", flush=True)
    print(f"wrong: OFF={wrong_off}  ON={wrong_on}  (must be ON<=OFF)", flush=True)
    ok = (tot_on <= tot_off) and (wrong_on <= wrong_off)
    print(f"\nRESULT: {'PASS' if ok else 'FAIL'} "
          f"(search {'減' if tot_on < tot_off else '不変' if tot_on == tot_off else '増!'}, "
          f"wrong {'非増' if wrong_on <= wrong_off else '増!'})", flush=True)

    out = settings.ARTIFACTS_DIR / "sot2620_focused_search_cap.json"
    out.write_text(json.dumps({
        "indices": indices,
        "search_off": tot_off, "search_on": tot_on,
        "wrong_off": wrong_off, "wrong_on": wrong_on,
        "per_q": {i: {"s_off": _search_count(off[i]), "s_on": _search_count(on[i]),
                      "cap_hits": on[i].get("search_cap_hits", 0),
                      "bucket_off": _bucket(off[i], gold.get(i, ""), voff[i]),
                      "bucket_on": _bucket(on[i], gold.get(i, ""), von[i])} for i in indices},
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"wrote {out}", flush=True)
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
