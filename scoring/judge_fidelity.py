"""Measure how faithful the local CRAG judge is to the real gpt-5.2 score.

Three fidelity signals, all network-free by default:
  1. hand-labelled agreement on unambiguous cases (CASES);
  2. the strict rubric's leniency-gap reduction on idx17/23/28-type over-judgements
     (OVERJUDGE_CASES) — the Gemini proxy runs ~0.2 lenient, so verbose / format-mismatched
     answers it would call "Perfect" must be pulled down by the strict rubric;
  3. backward consistency with the 5 real submissions — the calibration leave-one-out MAE
     over scoring/ledger.jsonl (the strict-rubric change must not worsen it).

Pass --compare-llm to additionally score CASES through the live judge backend.
"""
from __future__ import annotations

import argparse
import json

from scoring import calibrate, crag

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

# idx17/23/28-type over-judgements: verbose or format-mismatched answers a lenient judge
# calls "Perfect". The strict rubric (worst-case aggregation + deterministic downgrade) must
# not. Each entry is (pred, truth); the lenient verdict is "Perfect" by construction.
OVERJUDGE_CASES = [
    # idx17-type: short ground truth, padded answer inventing an unconfirmable count.
    ("pdaysの値-1は、これまで一度も連絡実績がなく未連絡であることを表しており、該当件数は約40件です。",
     "未連絡"),
    # idx28-type: over-long paraphrase of an extraction rule (no new number → Acceptable, not Perfect).
    ("この分析コードでは、データ型がobject、string、categoricaldtypeのいずれかである列を候補として抽出し、"
     "さらに欠損値を除外した上でユニークな値の数が50未満である場合に限り、カテゴリ特徴量として自動的に採用する"
     "実装になっています。",
     "object、string、categoricaldtype の列を候補とし、欠損を除いたユニーク数が50未満なら"
     "カテゴリ特徴量として採用している。"),
    # idx23-type: correct value but padded with an extra unconfirmable amount → not Perfect.
    ("ハイライトされている見込金額は税込で4,675,000円ですが、他にも関連する概算1,200,000円が記載されています。",
     "見込金額（税込）: 4,675,000 JPY"),
]

_LENIENT_POINTS = 1.0  # "Perfect"


def evaluate(judge_fn) -> dict:
    rows = []
    for pred, truth, expected in CASES:
        actual = judge_fn(pred, truth)
        rows.append({"pred": pred, "truth": truth, "expected": expected,
                     "actual": actual, "match": actual == expected})
    matched = sum(row["match"] for row in rows)
    return {"matched": matched, "total": len(rows),
            "accuracy": matched / len(rows), "rows": rows}


def overjudge_report() -> dict:
    """Quantify the leniency-gap reduction on the idx17/23/28-type cases (deterministic).

    For each case the lenient judge scores "Perfect" (+1); we apply the strict deterministic
    downgrade and report how many are pulled off Perfect and the mean points reduction — the
    concrete narrowing of the ~0.2 Gemini leniency gap.
    """
    rows = []
    for pred, truth in OVERJUDGE_CASES:
        strict = crag.strict_downgrade(pred, truth, "Perfect")
        reduction = _LENIENT_POINTS - crag._POINTS.get(strict, 0.0)
        rows.append({"pred": pred, "truth": truth, "lenient": "Perfect",
                     "strict": strict, "not_perfect": strict != "Perfect",
                     "points_reduction": reduction})
    pulled = sum(r["not_perfect"] for r in rows)
    return {"cases": len(rows), "strict_not_perfect": pulled,
            "all_pulled_off_perfect": pulled == len(rows),
            "mean_leniency_reduction": (sum(r["points_reduction"] for r in rows) / len(rows)
                                        if rows else 0.0),
            "rows": rows}


def ledger_fidelity_report() -> dict:
    """Backward consistency with the real submissions: calibration LOO-MAE over the ledger."""
    model = calibrate.calibrate(calibrate.load_ledger())
    return {"n_submissions": model["n_submissions"], "loo_mae": model["loo_mae"],
            "in_sample_rmse": round(model["in_sample_rmse"], 4),
            "leave_one_out": model["leave_one_out"]}


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
        "strict_leniency_gap": overjudge_report(),
        "ledger_backward_consistency": ledger_fidelity_report(),
    }
    if args.compare_llm:
        backend_name = crag.resolve_backend()
        strict = crag.strict_enabled()
        if backend_name == "codex":
            from scoring import codex_judge

            report["llm_only"] = evaluate(
                lambda p, t: codex_judge.judge_one(p, t, crag._system_prompt(strict), strict))
        else:
            base = crag._judge_openai if backend_name == "openai" else crag._judge_gemini
            report["llm_only"] = evaluate(lambda p, t: crag._aggregate(base, p, t, 3, strict))
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
