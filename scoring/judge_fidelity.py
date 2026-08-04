"""Measure agreement on a hand-labelled judge regression set."""
from __future__ import annotations

import argparse
import json

from scoring import crag

CASES = [
    ("Recall", "Recall", "Perfect"),
    ("100円（税込）", "100円（税抜）", "Incorrect"),
    ("20", "20日", "Perfect"),
    ("12.5", "12.47", "Acceptable"),
    ("A、B、C", "C、A、B", "Perfect"),
    ("A、B", "A、B、C", "Incorrect"),
    ("わかりません", "Recall", "Missing"),
    ("Precision", "Recall", "Incorrect"),
]

# Observations recorded in SOT-2427 before this change.  Keeping these separate
# from CASES avoids presenting unmeasured LLM outputs as a full baseline.
RECORDED_GEMINI = [
    ("Recall", "Recall", "Perfect", "Missing"),
    ("100円（税込）", "100円（税抜）", "Incorrect", "Perfect"),
]


def evaluate(judge_fn) -> dict:
    rows = []
    for pred, truth, expected in CASES:
        actual = judge_fn(pred, truth)
        rows.append({"pred": pred, "truth": truth, "expected": expected,
                     "actual": actual, "match": actual == expected})
    matched = sum(row["match"] for row in rows)
    return {"matched": matched, "total": len(rows),
            "accuracy": matched / len(rows), "rows": rows}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--compare-llm", action="store_true")
    args = parser.parse_args()
    baseline_matches = sum(actual == expected for _, _, expected, actual in RECORDED_GEMINI)
    hybrid_on_baseline = sum(crag.judge(pred, truth) == expected
                             for pred, truth, expected, _ in RECORDED_GEMINI)
    report = {
        "hand_labelled_hybrid": evaluate(crag.judge),
        "recorded_regression_comparison": {
            "cases": len(RECORDED_GEMINI),
            "gemini_only_accuracy": baseline_matches / len(RECORDED_GEMINI),
            "hybrid_accuracy": hybrid_on_baseline / len(RECORDED_GEMINI),
        },
    }
    if args.compare_llm:
        backend = crag._judge_openai if crag.settings.JUDGE_BACKEND.lower() == "openai" else crag._judge_gemini
        report["llm_only"] = evaluate(lambda p, t: crag._majority(backend, p, t, 3))
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
