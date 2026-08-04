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
import datetime as _dt
import itertools
import json
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from config import settings
from scoring import deterministic, synth
from src.rag.corpus import nfc

TRUST_PATH = Path(settings.REPO_ROOT) / "config" / "archetype_trust.json"
HISTORY_PATH = settings.ARTIFACTS_DIR / "holdout_history.jsonl"
DEFAULT_THRESHOLD = 0.80
DEFAULT_MIN_COMMITTED = 5
# Per-company synthetic samples are scarce, so the hold-out slice needs a smaller commit floor than
# the dev slice for an archetype to be judged at all.
DEFAULT_HOLDOUT_MIN_COMMITTED = 3

# Companies sealed out of development and used ONLY as the generalization hold-out. Trust for a
# question archetype is decided on this unseen slice — never on the dev/valid slice — so an archetype
# that merely overfits the visible projects cannot earn trust. Override with SEAL_COMPANIES
# (comma-separated company folder names). Glossary (社内管理) and cross-document (横断) items are not
# owned by a single project and therefore never fall into the hold-out slice.
_DEFAULT_SEALED = (
    "株式会社青葉バイオメディカル機器",
    "医療法人社団 蒼泉会 ひがし丘総合病院",
    "青葉与信マネジメント株式会社",
)


def sealed_companies() -> set[str]:
    raw = os.getenv("SEAL_COMPANIES")
    names = raw.split(",") if raw else list(_DEFAULT_SEALED)
    return {nfc(s).strip() for s in names if s and s.strip()}


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
                     "company": it.company, "pred": pred, "truth": it.truth, "verdict": verdict,
                     "points": deterministic.POINTS[verdict]})
    return rows


def split_rows(rows: list[dict]) -> tuple[list[dict], list[dict]]:
    """Partition scored rows into (dev, hold-out) by sealed company."""
    sealed = sealed_companies()
    dev, hold = [], []
    for r in rows:
        (hold if nfc(str(r.get("company", ""))) in sealed else dev).append(r)
    return dev, hold


def _overall_mean(rows: list[dict]) -> float:
    return sum(r["points"] for r in rows) / len(rows) if rows else 0.0


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


def decide_trust(dev_agg: dict[str, dict], hold_agg: dict[str, dict],
                 threshold: float, min_committed: int, holdout_min: int) -> dict[str, dict]:
    """Merge dev + hold-out per-archetype stats into the trust map, judging trust on the hold-out.

    - ``holdout_validated`` (gates hard-module *direct commit* in generate.py) is True ONLY when the
      hold-out slice has enough committed samples AND clears the precision threshold on the *unseen*
      projects. An archetype proven only on dev/valid never earns this.
    - ``trust`` (drives the additive abstain gate) is set False only with positive evidence of
      unreliability on the judging split (hold-out when sufficient, else dev). Insufficient data
      leaves ``trust`` True so the gate never abstains an unmeasured archetype (additive-safe)."""
    out: dict[str, dict] = {}
    for arch in sorted(set(dev_agg) | set(hold_agg)):
        d = dev_agg.get(arch)
        h = hold_agg.get(arch)
        holdout_validated = bool(h and h["committed"] >= holdout_min and h["precision"] >= threshold)
        if h and h["committed"] >= holdout_min:
            trust, basis = h["precision"] >= threshold, "holdout"
        elif d and d["committed"] >= min_committed:
            trust, basis = d["precision"] >= threshold, "dev"
        else:
            trust, basis = True, "insufficient"
        out[arch] = {
            "trust": trust,
            "holdout_validated": holdout_validated,
            "trust_basis": basis,
            "holdout": h,
            "dev": d,
        }
    return out


def write_trust(trust: dict[str, dict], threshold: float, min_committed: int,
                holdout_min: int) -> None:
    TRUST_PATH.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "_meta": {
            "generator": "scoring.selfimprove",
            "threshold": threshold, "min_committed": min_committed,
            "holdout_min_committed": holdout_min,
            "sealed_companies": sorted(sealed_companies()),
            "trust_judged_on": "holdout",
        },
        "archetypes": trust,
    }
    TRUST_PATH.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def append_history(label: str, dev_rows: list[dict], hold_rows: list[dict],
                   decided: dict[str, dict], when: str | None = None) -> dict:
    """Append this run's dev/hold-out means (overall + per archetype) to the history ledger, so
    overfit_check can compare consecutive runs (valid gain vs hold-out gain)."""
    rec = {
        "recordedAt": when or _dt.datetime.now().isoformat(timespec="seconds"),
        "label": label,
        "dev_mean": round(_overall_mean(dev_rows), 4),
        "holdout_mean": round(_overall_mean(hold_rows), 4),
        "archetypes": {
            a: {"dev_mean": (decided[a]["dev"] or {}).get("mean_score"),
                "holdout_mean": (decided[a]["holdout"] or {}).get("mean_score")}
            for a in decided
        },
    }
    HISTORY_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(HISTORY_PATH, "a", encoding="utf-8") as f:
        f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    return rec


def report(decided: dict[str, dict], dev_rows: list[dict], hold_rows: list[dict]) -> None:
    dev_mean, hold_mean = _overall_mean(dev_rows), _overall_mean(hold_rows)
    print("\n============== self-improvement: dev vs hold-out archetype精度 ==============")
    print(f"sealed (hold-out) companies: {sorted(sealed_companies())}")
    print(f"{'archetype':20} | {'devN':>4} {'dCom':>5} {'dPrc':>5} {'dMean':>6} "
          f"| {'hoN':>4} {'hCom':>5} {'hPrc':>5} {'hMean':>6} | basis      commit")
    print("-" * 96)
    for arch in sorted(decided):
        e = decided[arch]
        d, h = e["dev"], e["holdout"]
        dcell = (f"{d['n']:>4} {d['committed']:>5} {d['precision']:>5.2f} {d['mean_score']:>6.3f}"
                 if d else f"{'-':>4} {'-':>5} {'-':>5} {'-':>6}")
        hcell = (f"{h['n']:>4} {h['committed']:>5} {h['precision']:>5.2f} {h['mean_score']:>6.3f}"
                 if h else f"{'-':>4} {'-':>5} {'-':>5} {'-':>6}")
        print(f"{arch:20} | {dcell} | {hcell} | {e['trust_basis']:10} "
              f"{'commit✓' if e['holdout_validated'] else 'advisory'}")
    print("-" * 96)
    committable = [a for a, e in decided.items() if e["holdout_validated"]]
    print(f"overall mean — dev: {dev_mean:+.4f}   hold-out: {hold_mean:+.4f}")
    print(f"hold-out-validated (direct-commit) archetypes: {committable}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--self-test", action="store_true", help="offline scorer validation only (no LLM)")
    ap.add_argument("--limit-per-archetype", type=int, default=None)
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--hard", action="store_true")
    ap.add_argument("--threshold", type=float, default=DEFAULT_THRESHOLD)
    ap.add_argument("--min-committed", type=int, default=DEFAULT_MIN_COMMITTED)
    ap.add_argument("--holdout-min-committed", type=int, default=DEFAULT_HOLDOUT_MIN_COMMITTED)
    ap.add_argument("--label", default="run", help="label recorded in the hold-out history ledger")
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
    dev_rows, hold_rows = split_rows(rows)
    dev_agg = aggregate(dev_rows, args.threshold, args.min_committed)
    hold_agg = aggregate(hold_rows, args.threshold, args.holdout_min_committed)
    decided = decide_trust(dev_agg, hold_agg, args.threshold, args.min_committed,
                           args.holdout_min_committed)
    write_trust(decided, args.threshold, args.min_committed, args.holdout_min_committed)
    report(decided, dev_rows, hold_rows)
    rec = append_history(args.label, dev_rows, hold_rows, decided)
    print(f"appended hold-out history → {HISTORY_PATH} "
          f"(dev {rec['dev_mean']:+.4f} / hold-out {rec['holdout_mean']:+.4f})")
    scored = settings.ARTIFACTS_DIR / "synth_scored.csv"
    import csv as _csv
    with open(scored, "w", encoding="utf-8", newline="") as f:
        w = _csv.DictWriter(f, fieldnames=["id", "archetype", "kind", "company", "verdict",
                                           "points", "pred", "truth"])
        w.writeheader()
        w.writerows(rows)
    print(f"wrote {TRUST_PATH}\nwrote {scored}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
