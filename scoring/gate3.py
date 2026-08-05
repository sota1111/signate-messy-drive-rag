"""関門3 — offline ゴールド100問採点を前段化した guarded 実提出 (SOT-2477).

test100 の正解は隠されているため、実提出はSIGNATEリーダーボードでしか採点できず、かつ日次上限
5・human許可 (``SIGNATE_SUBMIT_ALLOWED=1``) の希少資源である。そこで本ゲートは実提出の**前段**に
「ゴールド100問オフライン採点」(SOT-2472 :mod:`scoring.gold_offline`) を挟み、候補提出物をまず
オフラインのゴールド (``artifacts/predictions_test_v3_final.csv``) に照合して一致率を測る。

    予測生成 → ゴールド照合(offline) → 一致率がしきい値以上 かつ human-gate成立 → zip提出

つまり実提出には二重ゲートがかかる:
  1. **オフライン合格ゲート** — ゴールド一致率 ≥ しきい値 (既定 = 手動基準 26/30 ≈ 86.7%、
     ``GATE3_GOLD_THRESHOLD`` / ``--threshold`` で上書き可)。未満なら zip を作らず提出しない。
  2. **human-gate** — ``SIGNATE_SUBMIT_ALLOWED=1`` (SOT-2457)。実提出予算の消費は明示許可時のみ。

本番の解答経路は Gemini-only: ``runner.run`` は既定で ``investigator`` バックエンド (Gemini
tool駆動の調査エージェント) を使うため、提出物生成に Claude は関与しない (SOT-2469)。

    python -m scoring.gate3                    # 生成 → offlineゴールド採点 → dry run (提出しない)
    python -m scoring.gate3 --no-run           # 既存 predictions を offline採点のみ
    python -m scoring.gate3 --submit --memo "baseline hybrid RAG"   # 二重ゲート通過時のみ実提出
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import zipfile
from pathlib import Path

from config import settings

# Offline gold match-rate required before a real submission is allowed. Default = the manual human
# baseline (26/30 on the real gate, mirrored from scoring.gold_offline). Override per-run via the
# GATE3_GOLD_THRESHOLD env var or --threshold.
from scoring.gold_offline import MANUAL_MATCH_RATE as _MANUAL_MATCH_RATE

DEFAULT_GOLD_THRESHOLD = _MANUAL_MATCH_RATE


def gold_threshold() -> float:
    """The offline gold match-rate submission threshold (env GATE3_GOLD_THRESHOLD or default)."""
    raw = os.getenv("GATE3_GOLD_THRESHOLD", "").strip()
    if not raw:
        return DEFAULT_GOLD_THRESHOLD
    try:
        return float(raw)
    except ValueError:
        raise SystemExit(f"GATE3_GOLD_THRESHOLD is not a number: {raw!r}")


# --------------------------------------------------------------------------- predictions + offline
def ensure_predictions(preds: Path, run_first: bool, hard: bool, gen: str | None = None) -> Path:
    """Produce ``preds`` with the production answer chain if it is missing (and ``run_first``)."""
    if run_first and not preds.exists():
        from src.rag import run as runner
        kwargs = {"gen": gen} if gen else {}
        runner.run("test", preds, limit=None, workers=8, hard=hard, **kwargs)
    return preds


def _answers_path(preds: Path) -> Path:
    """Prefer the sibling ``.details.jsonl`` (carries cost/method) for scoring; fall back to csv."""
    details = preds.with_suffix(".details.jsonl")
    return details if details.exists() else preds


def score_offline(preds: Path, gold: Path, judge=None):
    """Score ``preds`` against the offline gold set and return a :class:`gold_offline.Report`.

    ``judge`` is injectable (default = the shared CRAG judge) so the gating logic is testable
    without any network call."""
    from scoring import gold_offline

    gold_map = gold_offline.load_gold(gold)
    questions = gold_offline._questions_by_index()
    predictions = gold_offline.load_predictions(_answers_path(preds))
    kwargs = {} if judge is None else {"judge": judge}
    return gold_offline.evaluate(predictions, gold_map, questions, **kwargs)


def offline_gate(report, threshold: float) -> bool:
    """True when the offline gold match rate clears ``threshold`` (关门3 submission precondition)."""
    return report.match_rate >= threshold


# --------------------------------------------------------------------------- submission build/upload
def build_zip(preds: Path) -> Path:
    """Package ``preds`` into the official SIGNATE ``submission.zip`` (predictions.csv inside)."""
    # SIGNATE requires the csv to be named exactly predictions.csv, zipped.
    submit_csv = settings.ARTIFACTS_DIR / "predictions.csv"
    submit_csv.write_bytes(preds.read_bytes())
    zip_path = settings.ARTIFACTS_DIR / "submission.zip"
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as z:
        z.write(submit_csv, arcname="predictions.csv")
    print(f"built submission: {zip_path}")
    return zip_path


def build_submission(preds: Path, run_first: bool, hard: bool) -> Path:
    """Backward-compatible: ensure predictions exist, then package the submission zip."""
    ensure_predictions(preds, run_first, hard)
    return build_zip(preds)


def submit(zip_path: Path, memo: str) -> None:
    # 関門3 (real submission) is human-gated (SOT-2457): the submission budget may only be
    # spent with explicit permission, recorded by setting this env for the one run.
    if os.getenv("SIGNATE_SUBMIT_ALLOWED", "").strip().lower() not in ("1", "true", "yes", "on"):
        raise SystemExit(
            "関門3 submission is blocked: human permission required (SOT-2457). "
            "Set SIGNATE_SUBMIT_ALLOWED=1 for this run only after explicit approval.")
    cmd = ["signate", "submit", "--task_key", settings.SIGNATE_TASK_KEY,
           "--path", str(zip_path), "--memo", memo]
    print("running:", " ".join(cmd))
    subprocess.run(cmd, check=True)


# --------------------------------------------------------------------------- CLI
def main(argv: list[str] | None = None) -> int:
    from src.rag.run import DEFAULT_GEN, GEN_CHOICES

    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--preds", type=Path, default=None,
                    help="candidate test predictions (default artifacts/predictions_test.csv)")
    ap.add_argument("--gold", type=Path,
                    default=settings.ARTIFACTS_DIR / "predictions_test_v3_final.csv",
                    help="offline gold answers (headerless index,answer)")
    ap.add_argument("--gen", choices=list(GEN_CHOICES), default=DEFAULT_GEN,
                    help="answer backend when predictions are missing (SOT-2469, default investigator)")
    ap.add_argument("--no-run", action="store_true",
                    help="do not run the chain; score/submit existing predictions")
    ap.add_argument("--hard", action="store_true", help="use stronger model for all questions")
    ap.add_argument("--threshold", type=float, default=None,
                    help="offline gold match-rate threshold (default GATE3_GOLD_THRESHOLD or "
                         f"{DEFAULT_GOLD_THRESHOLD:.4f})")
    ap.add_argument("--out", type=Path, default=None,
                    help="also write the offline JSON report to this path")
    ap.add_argument("--submit", action="store_true",
                    help="upload to SIGNATE — only if BOTH the offline gate and the human gate pass")
    ap.add_argument("--memo", default="hybrid RAG submission")
    args = ap.parse_args(argv)

    preds = args.preds or (settings.ARTIFACTS_DIR / "predictions_test.csv")
    ensure_predictions(preds, run_first=not args.no_run, hard=args.hard, gen=args.gen)

    report = score_offline(preds, args.gold)
    threshold = args.threshold if args.threshold is not None else gold_threshold()
    passed = offline_gate(report, threshold)

    print(report.render())
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(report.to_dict(), ensure_ascii=False, indent=2),
                            encoding="utf-8")
        print(f"wrote {args.out}")
    print(f"\noffline gold gate: match {report.match_rate:.1%} vs threshold {threshold:.1%} "
          f"-> {'PASS' if passed else 'BELOW'}")

    if not args.submit:
        print("dry run — pass --submit to (gate-permitting) upload to SIGNATE.")
        return 0

    if not passed:
        raise SystemExit(
            f"関門3 blocked: offline gold match {report.match_rate:.1%} < threshold "
            f"{threshold:.1%}; not submitting.")
    zip_path = build_zip(preds)
    submit(zip_path, args.memo)  # additionally enforces the SIGNATE_SUBMIT_ALLOWED human gate
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
