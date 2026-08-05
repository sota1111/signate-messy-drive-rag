"""Answer a question split and write predictions.csv (+ a detailed run log).

    python -m src.rag.run --split valid          # answers data/questions/questions_valid.csv
    python -m src.rag.run --split test            # answers questions_test.csv (100, submission)
    python -m src.rag.run --split valid --limit 5 # smoke test

predictions.csv is headerless `index,answer` (the official validator format). Answers are
sanitised to a single line so the validator's line-based delimiter check is satisfied.

Answer backend (``--gen`` / ``gen=``, SOT-2469)
-----------------------------------------------
The **production** answer path is the Gemini tool-driven **investigation agent**
(:mod:`src.rag.agent.investigator`): each question is answered by planning + iterating over the
generic Step1 tools (file discovery / grep / Office extraction / decryption / pandas compute /
chart & vision reads) and reading the *real files*, with **no** corpus-specific fact injected —
so it is Gemini-only and Claude-independent. Backends:

* ``investigator`` — production default. Gemini function-calling agent (real-file tools).
* ``resolve``      — the full agent chain investigator → heterogeneous verifier → third-judge
  tie-break (:mod:`src.rag.agent.tiebreak`); still Gemini-only, higher-precision/slower.
* ``gemini``       — **legacy**, experimental only: text-only retrieval+LLM (:mod:`src.rag.generate`).
* ``opus``         — **legacy**, experimental only, **non-production**: Claude Opus CLI
  (:mod:`src.rag.opus_gen`, SOT-2457). Kept for backward-compatible experiments; NOT used by the
  production path (the SOT-2460 Gemini-only mandate excludes Claude from production).
"""
from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Callable

import pandas as pd

from config import settings
from src.rag import generate

# Answer backends. The production default is the Gemini investigation agent; the text-only backends
# (``gemini``/``opus``) are retained for backward-compatible experiments only (SOT-2469).
GEN_CHOICES = ("investigator", "resolve", "gated", "gemini", "opus")
DEFAULT_GEN = "investigator"
# ``opus`` (Claude) is explicitly excluded from the production answer path — Gemini-only (SOT-2460).
EXPERIMENTAL_GEN = ("gemini", "opus")


def _sanitize(answer: str) -> str:
    a = re.sub(r"\s+", " ", (answer or "").replace("\r", " ").replace("\n", " ")).strip()
    return a or settings.ABSTAIN


def load_questions(split: str) -> list[tuple[int, str]]:
    path = settings.QUESTIONS_VALID if split == "valid" else settings.QUESTIONS_TEST
    df = pd.read_csv(path)
    col = "question"
    return [(int(r["index"]), str(r[col])) for _, r in df.iterrows()]


def _agent_result(d: dict) -> dict:
    """Normalise an agent backend's ``to_dict()`` to the run-log schema (adds ``used_images``).

    The investigation/resolution agents read files through tools rather than attaching raster
    images, so ``used_images`` is 0; every other key (answer/confidence/evidence/method/…) is kept
    for the details log.
    """
    d.setdefault("used_images", 0)
    return d


def make_worker(gen: str, hard: bool) -> Callable[[int, str], tuple[int, dict]]:
    """Build the per-question answer function for backend ``gen`` (see module docstring)."""
    if gen == "investigator":
        from src.rag.agent import investigator

        def work(idx: int, q: str) -> tuple[int, dict]:
            return idx, _agent_result(investigator.answer_question(q).to_dict())
    elif gen == "resolve":
        from src.rag.agent import tiebreak

        def work(idx: int, q: str) -> tuple[int, dict]:
            return idx, _agent_result(tiebreak.resolve_question(q).to_dict())
    elif gen == "gated":
        # Full 合議 + precision-first confidence gate (SOT-2473): commit only 合議一致・高確信, else 棄権.
        from src.rag.agent import gate

        def work(idx: int, q: str) -> tuple[int, dict]:
            return idx, _agent_result(gate.gate_question(q).to_dict())
    elif gen == "opus":
        # Legacy Claude backend — excluded from the production path, experiments only (SOT-2469).
        from src.rag import opus_gen

        def work(idx: int, q: str) -> tuple[int, dict]:
            return idx, opus_gen.answer_question(q)
    elif gen == "gemini":
        # Legacy text-only retrieval+LLM backend — experiments only (SOT-2469).
        def work(idx: int, q: str) -> tuple[int, dict]:
            return idx, generate.answer_question(q, hard=hard)
    else:
        raise ValueError(f"unknown answer backend gen={gen!r}; choose one of {GEN_CHOICES}")
    return work


def run(split: str, out: Path, limit: int | None, workers: int, hard: bool,
        gen: str = DEFAULT_GEN) -> None:
    if gen in EXPERIMENTAL_GEN:
        print(f"[run] WARNING: gen={gen!r} is an experimental/legacy backend, NOT the production "
              f"answer path (production is Gemini-only: gen='investigator').", file=sys.stderr)
    questions = load_questions(split)
    if limit:
        questions = questions[:limit]
    results: dict[int, dict] = {}
    work = make_worker(gen, hard)

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
    ap.add_argument("--gen", choices=list(GEN_CHOICES), default=DEFAULT_GEN,
                    help="answer backend (SOT-2469): investigator (Gemini tool-agent, production "
                         "default) | resolve (investigator→verifier→tiebreak, full Gemini chain) | "
                         "gated (resolve + precision-first confidence gate, SOT-2473: commit only "
                         "合議一致・高確信, else 棄権) | gemini (legacy text-only RAG, experimental) | "
                         "opus (legacy Claude CLI, experimental/non-production)")
    args = ap.parse_args()
    out = args.out or (settings.ARTIFACTS_DIR / f"predictions_{args.split}.csv")
    run(args.split, out, args.limit, args.workers, args.hard, gen=args.gen)
