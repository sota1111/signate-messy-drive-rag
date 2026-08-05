"""Submission ledger — the append-only record that ties every SIGNATE submission's
ex-ante prediction to its realised Public score.

Each row is one submission with a complete, self-contained record:

    date, submission, config, commit,
    local_score, archetype_committed,          # inputs to the predictor
    predicted, ci_95, prediction_basis,        # what we forecast before uploading
    real_public_score, absolute_error,         # the realised outcome and the miss
    notes

`predicted`/`real_public_score`/`absolute_error` may be null while a submission is pending
(prediction recorded, score not yet returned by SIGNATE). `record_prediction` writes the
ex-ante half from a `scoring.predict` output; `set_actual` fills in the realised half and the
error once the Public score is known. This is the machine-recording path every future
submission runs through, so the ledger stays complete without hand edits.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from config import settings

LEDGER_PATH = Path(settings.REPO_ROOT) / "scoring" / "ledger.jsonl"

FIELDS = ("date", "submission", "config", "commit", "local_score", "archetype_committed",
          "predicted", "ci_95", "prediction_basis", "real_public_score", "absolute_error",
          "notes")


def load(path: Path = LEDGER_PATH) -> list[dict]:
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


def _write(rows: list[dict], path: Path = LEDGER_PATH) -> None:
    path.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows),
                    encoding="utf-8")


def _round(value, ndigits: int = 4):
    return None if value is None else round(float(value), ndigits)


def record_prediction(submission: str, prediction: dict, *, date: str, config: str = "",
                      commit: str = "", notes: str = "", real_public_score: float | None = None,
                      basis: str = "ex-ante", path: Path = LEDGER_PATH) -> dict:
    """Append (or upsert) a submission row from a `scoring.predict` output dict.

    `prediction` is the JSON that `scoring.predict` prints: it carries
    `estimated_real_public_score`, `confidence_interval_95`, `local_score` and
    `archetype_committed`. If `real_public_score` is known already it is stored and the
    absolute error is computed; otherwise both stay null until `set_actual`.
    """
    predicted = _round(prediction.get("estimated_real_public_score"))
    ci = prediction.get("confidence_interval_95")
    ci = [_round(ci[0]), _round(ci[1])] if ci else None
    row = {
        "date": date,
        "submission": submission,
        "config": config,
        "commit": commit,
        "local_score": prediction.get("local_score"),
        "archetype_committed": dict(prediction.get("archetype_committed", {})),
        "predicted": predicted,
        "ci_95": ci,
        "prediction_basis": basis,
        "real_public_score": _round(real_public_score),
        "absolute_error": (None if real_public_score is None or predicted is None
                           else _round(abs(predicted - real_public_score))),
        "notes": notes,
    }
    rows = load(path) if path.exists() else []
    for i, existing in enumerate(rows):
        if existing.get("submission") == submission:
            rows[i] = row
            break
    else:
        rows.append(row)
    _write(rows, path)
    return row


def set_actual(submission: str, real_public_score: float, path: Path = LEDGER_PATH) -> dict:
    """Fill in a pending submission's realised Public score and its absolute error."""
    rows = load(path)
    for row in rows:
        if row.get("submission") == submission:
            row["real_public_score"] = _round(real_public_score)
            predicted = row.get("predicted")
            row["absolute_error"] = (None if predicted is None
                                     else _round(abs(float(predicted) - real_public_score)))
            _write(rows, path)
            return row
    raise KeyError(f"submission {submission!r} not found in {path}")


def _main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    sub = ap.add_subparsers(dest="cmd", required=True)

    rec = sub.add_parser("record", help="run scoring.predict and append the prediction")
    rec.add_argument("--submission", required=True)
    rec.add_argument("--date", required=True)
    rec.add_argument("--config", default="")
    rec.add_argument("--commit", default="")
    rec.add_argument("--notes", default="")
    rec.add_argument("--answers", type=Path, default=None)
    rec.add_argument("--local-score", type=float, default=None)
    rec.add_argument("--real", type=float, default=None, help="realised Public score if known")

    act = sub.add_parser("actual", help="fill in a realised Public score")
    act.add_argument("--submission", required=True)
    act.add_argument("--real", type=float, required=True)

    args = ap.parse_args()
    if args.cmd == "actual":
        row = set_actual(args.submission, args.real)
        print(json.dumps(row, ensure_ascii=False))
        return 0

    from scoring import calibrate, predict
    ledger = calibrate.load_ledger()
    model = calibrate.calibrate(ledger)
    local_score = (args.local_score if args.local_score is not None
                   else float(ledger[-1]["local_score"]))
    answers = args.answers or (settings.ARTIFACTS_DIR / "predictions_test.csv")
    counts = (predict.summarize_answers(predict.read_answers(answers)) if answers.exists()
              else dict(ledger[-1]["archetype_committed"]))
    prediction = predict.predict(model, local_score, counts)
    row = record_prediction(args.submission, prediction, date=args.date, config=args.config,
                            commit=args.commit, notes=args.notes, real_public_score=args.real)
    print(json.dumps(row, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())
