"""Self-improvement loop: RAG → deterministic scoring → per-archetype trust map.

Runs the real RAG (`src.rag.generate`) over the synthetic benchmark (`scoring.synth`), scores each
answer with the noise-free deterministic scorer (`scoring.deterministic`), and aggregates
**committed precision** and **coverage** per archetype. Archetypes whose committed precision clears a
threshold (with enough samples) are written as *trusted* to ``config/archetype_trust.json``; the
generator (`src.rag.generate`) then answers only trusted-or-unknown archetypes and abstains on the
measured-unreliable ones.

    python -m scoring.selfimprove                       # build synth + run full RAG + write trust map
    python -m scoring.selfimprove --limit-per-archetype 2   # cheap smoke (2 items/archetype)
    python -m scoring.selfimprove --self-test           # offline: only validate the scorer, no LLM
    python -m scoring.selfimprove --preds artifacts/synth_preds.jsonl  # score a cached RAG run

committed precision = (Perfect + Acceptable) / committed ,  committed = answered (non-abstain).
coverage           = committed / n .   trust ⇔ precision ≥ threshold AND committed ≥ min_committed.
"""
from __future__ import annotations

import argparse
import collections
import itertools
import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from config import settings
from scoring import deterministic, synth

TRUST_PATH = Path(settings.REPO_ROOT) / "config" / "archetype_trust.json"
DEFAULT_THRESHOLD = 0.80
DEFAULT_MIN_COMMITTED = 5


# ---------------------------------------------------------------------------------------------
def self_test() -> bool:
    """Offline sanity: every synthetic truth must score Perfect against itself, and the rubric
    rounding / abstention invariants must hold. Returns True on success (no LLM calls)."""
    items = synth.build()
    ok = True
    bad: list[str] = []
    for it in items:
        v = deterministic.score(it.truth, it.truth, it.kind)
        if v != "Perfect":
            ok = False
            bad.append(f"{it.id} [{it.kind}] truth={it.truth!r} → {v}")
    # invariants
    checks = [
        deterministic.score("わかりません", "42", "numeric") == "Missing",
        deterministic.score("", "x", "string") == "Missing",
        deterministic.score("5775000", "5,775,000", "numeric") == "Perfect",
        deterministic.score("0.72243", "0.7224", "numeric") == "Acceptable",  # equal at GT precision
        deterministic.score("A、B", "B、A", "set") == "Perfect",
        deterministic.score("cat", "dog", "string") == "Incorrect",
    ]
    print(f"self-test: {len(items)} synthetic truths, "
          f"{'ALL Perfect' if ok else f'{len(bad)} FAILED'}")
    for b in bad[:10]:
        print("   MISMATCH", b)
    inv_ok = all(checks)
    print(f"self-test invariants: {'PASS' if inv_ok else 'FAIL'} {checks}")
    return ok and inv_ok


# ---------------------------------------------------------------------------------------------
def _limit_per_archetype(items: list[synth.SynthItem], n: int) -> list[synth.SynthItem]:
    out: list[synth.SynthItem] = []
    for _arch, grp in itertools.groupby(
            sorted(items, key=lambda x: x.archetype), key=lambda x: x.archetype):
        out.extend(list(grp)[:n])
    return out


def run_rag(items: list[synth.SynthItem], workers: int, hard: bool) -> dict[str, dict]:
    """Answer every synthetic question with the real RAG; returns id -> result dict."""
    from src.rag import generate

    results: dict[str, dict] = {}

    def work(it: synth.SynthItem) -> tuple[str, dict]:
        # bypass the trust gate here — this loop is what *measures* trust in the first place.
        return it.id, generate.answer_question(it.question, hard=hard, apply_trust_gate=False)

    with ThreadPoolExecutor(max_workers=workers) as ex:
        futs = [ex.submit(work, it) for it in items]
        for n, fut in enumerate(as_completed(futs), 1):
            _id, res = fut.result()
            results[_id] = res
            print(f"[{n}/{len(items)}] {_id[:40]} :: {str(res.get('answer'))[:40]}")
    return results


def score_results(items: list[synth.SynthItem], preds: dict[str, dict]) -> list[dict]:
    rows: list[dict] = []
    for it in items:
        pred = (preds.get(it.id) or {}).get("answer", settings.ABSTAIN)
        verdict = deterministic.score(str(pred), it.truth, it.kind)
        rows.append({"id": it.id, "archetype": it.archetype, "kind": it.kind,
                     "pred": pred, "truth": it.truth, "verdict": verdict,
                     "points": deterministic.POINTS[verdict]})
    return rows


def aggregate(rows: list[dict], threshold: float, min_committed: int) -> dict[str, dict]:
    trust: dict[str, dict] = {}
    by_arch: dict[str, list[dict]] = collections.defaultdict(list)
    for r in rows:
        by_arch[r["archetype"]].append(r)
    for arch, rs in sorted(by_arch.items()):
        n = len(rs)
        vc = collections.Counter(r["verdict"] for r in rs)
        committed = n - vc["Missing"]
        good = vc["Perfect"] + vc["Acceptable"]
        precision = (good / committed) if committed else 0.0
        coverage = committed / n if n else 0.0
        mean_score = sum(r["points"] for r in rs) / n if n else 0.0
        trusted = bool(committed >= min_committed and precision >= threshold)
        trust[arch] = {
            "trust": trusted, "precision": round(precision, 4),
            "coverage": round(coverage, 4), "mean_score": round(mean_score, 4),
            "n": n, "committed": committed,
            "verdicts": {k: vc[k] for k in ("Perfect", "Acceptable", "Missing", "Incorrect")},
        }
    return trust


def write_trust(trust: dict[str, dict], threshold: float, min_committed: int) -> None:
    TRUST_PATH.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "_meta": {
            "generator": "scoring.selfimprove",
            "threshold": threshold, "min_committed": min_committed,
        },
        "archetypes": trust,
    }
    TRUST_PATH.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def report(trust: dict[str, dict], rows: list[dict]) -> None:
    overall = sum(r["points"] for r in rows) / len(rows) if rows else 0.0
    print("\n==================== self-improvement: archetype精度 ====================")
    print(f"{'archetype':22} {'n':>3} {'commit':>6} {'prec':>6} {'cov':>6} {'mean':>7}  trust")
    print("-" * 70)
    for arch in sorted(trust):
        t = trust[arch]
        print(f"{arch:22} {t['n']:>3} {t['committed']:>6} {t['precision']:>6.2f} "
              f"{t['coverage']:>6.2f} {t['mean_score']:>7.3f}  {'✓' if t['trust'] else '·'}")
    print("-" * 70)
    trusted = [a for a, t in trust.items() if t["trust"]]
    print(f"overall mean score: {overall:+.4f}   trusted archetypes: {trusted}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--self-test", action="store_true", help="offline scorer validation only (no LLM)")
    ap.add_argument("--limit-per-archetype", type=int, default=None)
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--hard", action="store_true")
    ap.add_argument("--threshold", type=float, default=DEFAULT_THRESHOLD)
    ap.add_argument("--min-committed", type=int, default=DEFAULT_MIN_COMMITTED)
    ap.add_argument("--preds", type=Path, default=None,
                    help="score a cached RAG run (jsonl of {id, answer}) instead of calling the LLM")
    args = ap.parse_args()

    if not self_test():
        print("SELF-TEST FAILED — deterministic scorer disagrees with its own ground truth; aborting.")
        return 1
    if args.self_test:
        return 0

    items = synth.build()
    synth.write(items)
    if args.limit_per_archetype:
        items = _limit_per_archetype(items, args.limit_per_archetype)

    if args.preds and args.preds.exists():
        preds: dict[str, dict] = {}
        with open(args.preds, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    o = json.loads(line)
                    preds[o["id"]] = o
    else:
        preds = run_rag(items, workers=args.workers, hard=args.hard)
        cache = settings.ARTIFACTS_DIR / "synth_preds.jsonl"
        with open(cache, "w", encoding="utf-8") as f:
            for _id, res in preds.items():
                f.write(json.dumps({"id": _id, **res}, ensure_ascii=False) + "\n")
        print(f"cached RAG answers → {cache}")

    rows = score_results(items, preds)
    trust = aggregate(rows, args.threshold, args.min_committed)
    write_trust(trust, args.threshold, args.min_committed)
    report(trust, rows)
    scored = settings.ARTIFACTS_DIR / "synth_scored.csv"
    import csv as _csv
    with open(scored, "w", encoding="utf-8", newline="") as f:
        w = _csv.DictWriter(f, fieldnames=["id", "archetype", "kind", "verdict", "points", "pred", "truth"])
        w.writeheader()
        w.writerows(rows)
    print(f"wrote {TRUST_PATH}\nwrote {scored}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
