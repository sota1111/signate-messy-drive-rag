"""関門1 — SIGNATE教師データ (valid 30問, 公式GT) を客観採点する.

official GT = data/questions/valid_txt.csv (headerless index,answer).
Runs the RAG over the valid split if predictions are missing, then scores with the local
CRAG judge (Gemini proxy of the official gpt-5.2 rubric).

    python -m scoring.gate1                # run RAG on valid + score
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


def main(preds: Path | None, run_first: bool) -> None:
    gt = _read_headerless(settings.VALID_GROUND_TRUTH)
    preds_path = preds or (settings.ARTIFACTS_DIR / "predictions_valid.csv")

    if run_first and not preds_path.exists():
        from src.rag import run as runner
        print("no predictions found — running RAG on valid split...")
        runner.run("valid", preds_path, limit=None, workers=8, hard=False)

    pred_map = _read_headerless(preds_path)
    idxs = sorted(gt)
    pairs = [(pred_map.get(i, settings.ABSTAIN), gt[i]) for i in idxs]
    score, results = crag.score_pairs(pairs)

    dist = collections.Counter(r["judged"] for r in results)
    print("\n==================== 関門1 (valid 30) ====================")
    print(f"JUDGE backend: {settings.JUDGE_BACKEND} ({settings.JUDGE_MODEL})")
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
    ap = argparse.ArgumentParser()
    ap.add_argument("--preds", type=Path, default=None)
    ap.add_argument("--no-run", action="store_true", help="do not run RAG; score existing preds")
    args = ap.parse_args()
    main(args.preds, run_first=not args.no_run)
