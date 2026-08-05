"""関門1 — SIGNATE教師データ (valid 30問, 公式GT) を客観採点する.

official GT = data/questions/valid_txt.csv (headerless index,answer).
Runs the RAG over the valid split if predictions are missing, then scores with the local
CRAG judge (codex batch judge on the official grader's model family; SOT-2457).

Answer backend (``--gen`` / ``gen=``, SOT-2475)
-----------------------------------------------
valid30 is solved through the **production Gemini investigation agent** (``investigator``), the same
backend the real submission uses (``src.rag.run.make_worker``, SOT-2469). ``gen`` therefore defaults
to ``investigator`` — the legacy text-only ``gemini`` path is retained only for experiment/regression
comparison. The backend choices are shared verbatim with ``src.rag.run`` so gate1 can never diverge
from the submission pipeline.

    python -m scoring.gate1                # solve valid with the investigator agent + score
    python -m scoring.gate1 --gen resolve  # full investigator→verifier→tiebreak chain
    python -m scoring.gate1 --gen gemini    # legacy text-only path (regression comparison)
    python -m scoring.gate1 --preds path   # score an existing predictions.csv
"""
from __future__ import annotations

import argparse
import collections
from pathlib import Path

import pandas as pd

from config import settings
from scoring import crag


def _read_headerless(path: Path) -> dict[int, str]:
    df = pd.read_csv(path, header=None, index_col=0)
    return {int(i): ("" if pd.isna(v) else str(v)) for i, v in df[1].items()}


def main(preds: Path | None, run_first: bool, gen: str = "investigator", workers: int = 8) -> None:
    gt = _read_headerless(settings.VALID_GROUND_TRUTH)
    preds_path = preds or (settings.ARTIFACTS_DIR / "predictions_valid.csv")

    if run_first and not preds_path.exists():
        from src.rag import run as runner
        print(f"no predictions found — solving valid split with the {gen} agent path...")
        runner.run("valid", preds_path, limit=None, workers=workers, hard=False, gen=gen)

    pred_map = _read_headerless(preds_path)
    idxs = sorted(gt)
    pairs = [(pred_map.get(i, settings.ABSTAIN), gt[i]) for i in idxs]
    score, results = crag.score_pairs(pairs)

    dist = collections.Counter(r["judged"] for r in results)
    print("\n==================== 関門1 (valid 30) ====================")
    backend = crag.resolve_backend()
    print(f"JUDGE backend: {backend}"
          + (f" ({settings.JUDGE_MODEL})" if backend != "codex" else " (codex exec)"))
    print(f"SCORE (mean): {score:+.4f}   [Perfect+1 / Acceptable+0.5 / Missing0 / Incorrect-1]")
    print("verdicts:", dict(dist))
    print("-" * 58)
    for i, r in zip(idxs, results):
        mark = {"Perfect": "✓", "Acceptable": "△", "Missing": "·", "Incorrect": "✗"}.get(
            r["judged"], "?")
        print(f" {mark} [{i:2}] {r['judged']:11} pred={r['pred'][:34]!r:36} gt={r['truth'][:30]!r}")

    out = settings.ARTIFACTS_DIR / "gate1_scoring.csv"
    pd.DataFrame([{"index": i, "judged": r["judged"], "points": r["points"],
                   "pred": r["pred"], "gt": r["truth"]} for i, r in zip(idxs, results)]).to_csv(
        out, index=False)
    print(f"\nsaved {out}")


if __name__ == "__main__":
    from src.rag.run import DEFAULT_GEN, GEN_CHOICES

    ap = argparse.ArgumentParser()
    ap.add_argument("--preds", type=Path, default=None)
    ap.add_argument("--no-run", action="store_true", help="do not run RAG; score existing preds")
    ap.add_argument("--gen", choices=list(GEN_CHOICES), default=DEFAULT_GEN,
                    help="answer backend when predictions are missing (SOT-2475): investigator "
                         "(production, default) | resolve | gated | gemini/opus (legacy)")
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args()
    main(args.preds, run_first=not args.no_run, gen=args.gen, workers=args.workers)
