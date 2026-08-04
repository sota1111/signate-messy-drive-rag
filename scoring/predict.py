"""Predict a submission's real Public score before uploading it."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from config import settings
from scoring import calibrate
from src.rag.archetype import classify


def read_answers(path: Path) -> list[dict]:
    if path.suffix.lower() == ".jsonl":
        return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    with path.open(encoding="utf-8-sig", newline="") as fh:
        raw = list(csv.reader(fh))
    if not raw:
        return []
    if {c.strip().lower() for c in raw[0]} & {"answer", "prediction", "question"}:
        return [dict(zip(raw[0], row)) for row in raw[1:]]
    # Official submissions are headerless index,answer. Reattach the question so the same
    # archetype classifier used by the RAG can characterize the committed answers.
    questions = {}
    if settings.QUESTIONS_TEST.exists():
        with settings.QUESTIONS_TEST.open(encoding="utf-8-sig", newline="") as fh:
            questions = {str(r["index"]): r["question"] for r in csv.DictReader(fh)}
    return [{"index": r[0], "answer": r[1] if len(r) > 1 else "",
             "question": questions.get(r[0], "")} for r in raw]


def summarize_answers(rows: list[dict], abstain: str = settings.ABSTAIN) -> dict[str, int]:
    counts: dict[str, int] = {}
    abstentions = {"", abstain, "該当なし", "不明", "N/A"}
    for row in rows:
        answer = str(row.get("answer", row.get("prediction", ""))).strip()
        if answer in abstentions:
            continue
        arch = str(row.get("archetype") or classify(str(row.get("question", ""))))
        counts[arch] = counts.get(arch, 0) + 1
    return counts


def predict(model: dict, local_score: float, counts: dict[str, int]) -> dict:
    total = max(1, sum(counts.values()))
    estimate = model["intercept"] + model["local_score_coefficient"] * local_score
    contributions = {}
    for arch, count in sorted(counts.items()):
        effect = model["archetype_effects"].get(arch, 0.0)
        contributions[arch] = effect * count / total
        estimate += contributions[arch]
    half = float(model["interval_half_width"])
    return {"estimated_real_public_score": estimate,
            "confidence_interval_95": [max(-1.0, estimate-half), min(1.0, estimate+half)],
            "local_score": local_score, "committed": sum(counts.values()),
            "archetype_committed": counts, "archetype_contributions": contributions,
            "calibration_n": model["n_submissions"],
            "proxy_real_spearman": model["proxy_real_spearman"]}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--answers", type=Path,
                    default=settings.ARTIFACTS_DIR / "predictions_test.csv")
    ap.add_argument("--local-score", type=float, default=None,
                    help="gate1/hold-out proxy score (default: latest ledger value)")
    ap.add_argument("--ledger", type=Path, default=calibrate.LEDGER_PATH)
    ap.add_argument("--model", type=Path, default=calibrate.MODEL_PATH)
    args = ap.parse_args()
    ledger = calibrate.load_ledger(args.ledger)
    model = calibrate.calibrate(ledger)
    calibrate.save_model(model, args.model)
    local_score = args.local_score if args.local_score is not None else float(ledger[-1]["local_score"])
    if args.answers.exists():
        counts = summarize_answers(read_answers(args.answers))
        source = str(args.answers)
    else:
        counts = dict(ledger[-1]["archetype_committed"])
        source = f"latest ledger mix ({ledger[-1]['submission']}); pass --answers for a new set"
    result = predict(model, local_score, counts)
    result["answers_source"] = source
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
