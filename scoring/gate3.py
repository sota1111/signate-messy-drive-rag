"""関門3 — 完全な未知 (test 100問). Build predictions and submit to SIGNATE.

The test ground truth is hidden; this gate produces the official submission and (optionally)
submits it. The real score comes back from the SIGNATE leaderboard.

    python -m scoring.gate3                    # build predictions_test.csv + zip
    python -m scoring.gate3 --submit --memo "baseline hybrid RAG"
"""
from __future__ import annotations

import argparse
import subprocess
import zipfile
from pathlib import Path

from config import settings


def build_submission(preds: Path, run_first: bool, hard: bool) -> Path:
    if run_first and not preds.exists():
        from src.rag import run as runner
        runner.run("test", preds, limit=None, workers=8, hard=hard)
    # SIGNATE requires the csv to be named exactly predictions.csv, zipped.
    submit_csv = settings.ARTIFACTS_DIR / "predictions.csv"
    submit_csv.write_bytes(preds.read_bytes())
    zip_path = settings.ARTIFACTS_DIR / "submission.zip"
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as z:
        z.write(submit_csv, arcname="predictions.csv")
    print(f"built submission: {zip_path}")
    return zip_path


def submit(zip_path: Path, memo: str) -> None:
    # 関門3 (real submission) is human-gated (SOT-2457): the submission budget may only be
    # spent with explicit permission, recorded by setting this env for the one run.
    import os
    if os.getenv("SIGNATE_SUBMIT_ALLOWED", "").strip().lower() not in ("1", "true", "yes", "on"):
        raise SystemExit(
            "関門3 submission is blocked: human permission required (SOT-2457). "
            "Set SIGNATE_SUBMIT_ALLOWED=1 for this run only after explicit approval.")
    cmd = ["signate", "submit", "--task_key", settings.SIGNATE_TASK_KEY,
           "--path", str(zip_path), "--memo", memo]
    print("running:", " ".join(cmd))
    subprocess.run(cmd, check=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--preds", type=Path, default=None)
    ap.add_argument("--no-run", action="store_true")
    ap.add_argument("--hard", action="store_true", help="use stronger model for all questions")
    ap.add_argument("--submit", action="store_true")
    ap.add_argument("--memo", default="hybrid RAG submission")
    args = ap.parse_args()

    preds = args.preds or (settings.ARTIFACTS_DIR / "predictions_test.csv")
    zip_path = build_submission(preds, run_first=not args.no_run, hard=args.hard)
    if args.submit:
        submit(zip_path, args.memo)
    else:
        print("dry run — pass --submit to upload to SIGNATE.")
