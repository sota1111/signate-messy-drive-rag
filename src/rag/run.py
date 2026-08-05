"""Answer a question split and write predictions.csv (+ a detailed run log).

    python -m src.rag.run --split valid          # answers data/questions/questions_valid.csv
    python -m src.rag.run --split test            # answers questions_test.csv (100, submission)
    python -m src.rag.run --split valid --limit 5 # smoke test

predictions.csv is headerless `index,answer` (the official validator format). Answers are
sanitised to a single line so the validator's line-based delimiter check is satisfied.
"""
from __future__ import annotations

import argparse
import csv
import json
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import pandas as pd

from config import settings
from src.rag import generate


def _sanitize(answer: str) -> str:
    a = re.sub(r"\s+", " ", (answer or "").replace("\r", " ").replace("\n", " ")).strip()
    return a or settings.ABSTAIN


def load_questions(split: str) -> list[tuple[int, str]]:
    path = settings.QUESTIONS_VALID if split == "valid" else settings.QUESTIONS_TEST
    df = pd.read_csv(path)
    col = "question"
    return [(int(r["index"]), str(r[col])) for _, r in df.iterrows()]


def run(split: str, out: Path, limit: int | None, workers: int, hard: bool,
        gen: str = "gemini") -> None:
    questions = load_questions(split)
    if limit:
        questions = questions[:limit]
    results: dict[int, dict] = {}

    if gen == "opus":
        from src.rag import opus_gen

        def work(idx: int, q: str) -> tuple[int, dict]:
            return idx, opus_gen.answer_question(q)
    else:
        def work(idx: int, q: str) -> tuple[int, dict]:
            return idx, generate.answer_question(q, hard=hard)

    with ThreadPoolExecutor(max_workers=workers) as ex:
        futs = [ex.submit(work, idx, q) for idx, q in questions]
        for n, fut in enumerate(as_completed(futs), 1):
            idx, res = fut.result()
            results[idx] = res
            print(f"[{n}/{len(questions)}] idx={idx} conf={res['confidence']} "
                  f"imgs={res['used_images']} :: {res['answer'][:50]}")

    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f, quoting=csv.QUOTE_MINIMAL)
        for idx, _q in questions:
            w.writerow([idx, _sanitize(results[idx]["answer"])])
    # detailed log for analysis / gate scoring
    log = out.with_suffix(".details.jsonl")
    with open(log, "w", encoding="utf-8") as f:
        for idx, q in questions:
            rec = {"index": idx, **results[idx]}
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    n_abstain = sum(1 for idx, _ in questions if results[idx]["answer"] == settings.ABSTAIN)
    print(f"\nwrote {out}  ({len(questions)} answers, {n_abstain} abstained)  log={log}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", choices=["valid", "test"], default="valid")
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--hard", action="store_true", help="use the stronger model for all questions")
    ap.add_argument("--gen", choices=["gemini", "opus"], default="gemini",
                    help="answer backend: gemini (Vertex) or opus (Claude CLI, SOT-2457)")
    args = ap.parse_args()
    out = args.out or (settings.ARTIFACTS_DIR / f"predictions_{args.split}.csv")
    run(args.split, out, args.limit, args.workers, args.hard, gen=args.gen)
