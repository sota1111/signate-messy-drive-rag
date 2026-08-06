#!/usr/bin/env python3
"""SOT-2484 — independent re-derivation of all 100 test answers for gold v4.

Runs the heterogeneous verifier (:mod:`src.rag.agent.verifier`) over every test question with the
v3 gold answer supplied as the investigator answer, so :func:`compare_answers` reports where the
independent path (default: ``gemini-2.5-pro``) agrees / disagrees with v3. Each verdict is appended to
``artifacts/v4_verify_100.jsonl`` (resumable: indices already present are skipped), so a crash / rate
limit never loses completed work and never double-writes a result.

Usage:
    python scripts/run_v4_verify.py [--indices 4,10,28] [--limit N] [--workers 2] \
        [--model gemini-2.5-pro] [--out artifacts/v4_verify_100.jsonl] \
        [--answers artifacts/predictions_test_v3_final.csv]
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.rag.run import load_questions  # noqa: E402
from src.rag.agent.verifier import verify_question  # noqa: E402


def load_answers(path: Path) -> dict[int, str]:
    """Load a headerless index,answer CSV (the v3 gold) into an index→answer map."""
    out: dict[int, str] = {}
    with path.open(newline="", encoding="utf-8") as fh:
        for row in csv.reader(fh):
            if not row:
                continue
            out[int(row[0])] = row[1] if len(row) > 1 else ""
    return out


def done_indices(out: Path) -> set[int]:
    if not out.exists():
        return set()
    seen: set[int] = set()
    for line in out.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            seen.add(int(json.loads(line)["index"]))
        except Exception:
            pass
    return seen


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--answers", type=Path, default=ROOT / "artifacts/predictions_test_v3_final.csv")
    ap.add_argument("--out", type=Path, default=ROOT / "artifacts/v4_verify_100.jsonl")
    ap.add_argument("--model", default="gemini-2.5-pro")
    ap.add_argument("--indices", default=None, help="comma-separated subset of indices to (re)run")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--force", action="store_true", help="re-run even if index already in out")
    args = ap.parse_args(argv)

    questions = dict(load_questions("test"))
    answers = load_answers(args.answers)
    seen = set() if args.force else done_indices(args.out)

    if args.indices:
        want = [int(x) for x in args.indices.split(",") if x.strip() != ""]
    else:
        want = sorted(questions)
        if args.limit:
            want = want[: args.limit]
    todo = [i for i in want if i not in seen]
    print(f"[v4-verify] model={args.model} total_want={len(want)} already_done={len(want)-len(todo)} "
          f"todo={len(todo)} out={args.out}", flush=True)

    out_fh = args.out.open("a", encoding="utf-8")

    def work(idx: int) -> dict:
        t0 = time.time()
        v = verify_question(questions[idx], answers.get(idx, ""), model=args.model)
        d = v.to_dict()
        d["index"] = idx
        d["v3_answer"] = answers.get(idx, "")
        d["elapsed_s"] = round(time.time() - t0, 1)
        return d

    n_done = 0
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futs = {ex.submit(work, i): i for i in todo}
        for fut in as_completed(futs):
            idx = futs[fut]
            try:
                d = fut.result()
            except Exception as e:  # never let one failure kill the batch
                d = {"index": idx, "error": f"{type(e).__name__}: {e}", "agree": None,
                     "v3_answer": answers.get(idx, "")}
            out_fh.write(json.dumps(d, ensure_ascii=False) + "\n")
            out_fh.flush()
            n_done += 1
            flag = "ERR" if d.get("error") else ("DISAGREE" if d.get("agree") is False else "agree")
            print(f"[v4-verify] {n_done}/{len(todo)} idx={idx} {flag} "
                  f"({d.get('elapsed_s','?')}s)", flush=True)

    out_fh.close()
    print(f"[v4-verify] DONE wrote {n_done} verdicts to {args.out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
