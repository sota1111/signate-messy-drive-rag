"""Calibrate local proxy scores to SIGNATE Public scores.

The submission ledger is intentionally append-only.  Four observations are not enough for a
high-dimensional model, so archetype effects use ridge shrinkage towards zero and prediction
intervals include both residual error and small-sample uncertainty.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Iterable

import numpy as np

from config import settings

LEDGER_PATH = Path(settings.REPO_ROOT) / "scoring" / "ledger.jsonl"
MODEL_PATH = settings.ARTIFACTS_DIR / "real_score_calibration.json"


def load_ledger(path: Path = LEDGER_PATH, scored_only: bool = True) -> list[dict]:
    """Load submission rows. By default only rows with a realised ``real_public_score`` are
    returned (pending forward-recorded predictions have it ``null`` and are skipped for
    calibration); pass ``scored_only=False`` to get the full ledger."""
    rows = []
    for line_no, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        row = json.loads(line)
        for key in ("submission", "local_score", "real_public_score", "archetype_committed"):
            if key not in row:
                raise ValueError(f"{path}:{line_no}: missing {key}")
        if not isinstance(row["archetype_committed"], dict):
            raise ValueError(f"{path}:{line_no}: archetype_committed must be an object")
        if scored_only and row["real_public_score"] is None:
            continue  # prediction recorded but SIGNATE score not yet returned
        rows.append(row)
    if len(rows) < 2:
        raise ValueError("at least two scored submissions are required")
    return rows


def spearman(xs: Iterable[float], ys: Iterable[float]) -> float | None:
    """Spearman rho with average ranks for ties (None when either side is constant)."""
    def ranks(values: list[float]) -> np.ndarray:
        order = sorted(range(len(values)), key=values.__getitem__)
        out = np.empty(len(values), dtype=float)
        i = 0
        while i < len(order):
            j = i + 1
            while j < len(order) and values[order[j]] == values[order[i]]:
                j += 1
            out[order[i:j]] = (i + j - 1) / 2 + 1
            i = j
        return out

    a, b = list(xs), list(ys)
    if len(a) != len(b) or len(a) < 2:
        return None
    ra, rb = ranks(a), ranks(b)
    if np.std(ra) == 0 or np.std(rb) == 0:
        return None
    return float(np.corrcoef(ra, rb)[0, 1])


def _features(rows: list[dict], archetypes: list[str]) -> np.ndarray:
    matrix = []
    for row in rows:
        counts = row["archetype_committed"]
        total = max(1, sum(max(0, int(v)) for v in counts.values()))
        matrix.append([1.0, float(row["local_score"])] +
                      [max(0, int(counts.get(a, 0))) / total for a in archetypes])
    return np.asarray(matrix, dtype=float)


def _fit(x: np.ndarray, y: np.ndarray, ridge: float) -> np.ndarray:
    penalty = np.eye(x.shape[1]) * ridge
    penalty[0, 0] = 0.0  # never shrink the intercept
    # Archetype proportions are deliberately shrunk more aggressively than the proxy slope.
    if x.shape[1] > 2:
        penalty[2:, 2:] *= 4.0
    return np.linalg.pinv(x.T @ x + penalty) @ x.T @ y


def calibrate(rows: list[dict], ridge: float = 1.0) -> dict:
    archetypes = sorted({a for r in rows for a in r["archetype_committed"]})
    x = _features(rows, archetypes)
    y = np.asarray([float(r["real_public_score"]) for r in rows])
    coef = _fit(x, y, ridge)
    fitted = x @ coef
    residuals = y - fitted
    rmse = float(math.sqrt(np.mean(residuals ** 2)))

    loo = []
    for i, row in enumerate(rows):
        train = np.arange(len(rows)) != i
        pred = float(x[i] @ _fit(x[train], y[train], ridge))
        loo.append({"submission": row["submission"], "predicted": round(pred, 4),
                    "actual": round(float(y[i]), 4), "absolute_error": round(abs(pred-y[i]), 4)})
    # With very small n, a floor avoids a falsely precise interval after an exact-ish fit.
    interval_half_width = max(0.05, 1.96 * max(rmse, float(np.std(residuals, ddof=1))))
    backtest = [{"submission": r["submission"], "predicted": round(float(p), 4),
                 "actual": round(float(a), 4), "absolute_error": round(abs(float(p-a)), 4)}
                for r, p, a in zip(rows, fitted, y)]
    reference_local = float(np.mean([r["local_score"] for r in rows]))
    priors = {}
    for archetype, effect in zip(archetypes, coef[2:]):
        expected = max(-1.0, min(1.0, float(coef[0] + coef[1] * reference_local + effect)))
        # The official rubric also permits Acceptable/Missing. They cannot be identified from the
        # aggregate leaderboard score, so expose the identifiable two-outcome equivalent and label it.
        perfect = (expected + 1.0) / 2.0
        priors[archetype] = {"expected_points": expected, "perfect_rate_equivalent": perfect,
                             "incorrect_rate_equivalent": 1.0-perfect,
                             "basis": "ridge-shrunk aggregate net score"}
    return {
        "schema_version": 1,
        "n_submissions": len(rows),
        "ridge": ridge,
        "intercept": float(coef[0]),
        "local_score_coefficient": float(coef[1]),
        "archetype_effects": {a: float(v) for a, v in zip(archetypes, coef[2:])},
        "archetype_real_reliability_priors": priors,
        "interval_half_width": interval_half_width,
        "proxy_real_spearman": spearman([r["local_score"] for r in rows], y.tolist()),
        "calibrated_real_spearman": spearman(fitted.tolist(), y.tolist()),
        "in_sample_rmse": rmse,
        "loo_mae": round(float(np.mean([r["absolute_error"] for r in loo])), 4) if loo else None,
        "backtest": backtest,
        "leave_one_out": loo,
    }


def save_model(model: dict, path: Path = MODEL_PATH) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(model, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


# ===========================================================================================
# Abstention-threshold calibration on gold EV (SOT-2474)
# -------------------------------------------------------------------------------------------
# A *separate* calibration from the ledger→real-public-score model above: here we calibrate the
# commit/棄権 **confidence threshold** that the consensus gate (:mod:`src.rag.agent.gate`) applies,
# by sweeping it on the gold-100 set and picking the point that maximises the official expected
# value (EV).
#
# The official rubric scores a served answer:  Perfect +1 / Acceptable +0.5 / Missing 0 /
# Incorrect −1.  Committing an answer of would-be verdict ``v`` at threshold ``t`` therefore yields
# ``POINTS[v]`` when its (合議) confidence ``c ≥ t`` and ``0`` (棄権) otherwise.  EV(t) is the sum of
# those per-item contributions; the calibrated threshold is ``argmax_t EV(t)``.  Because Incorrect
# costs −1, the profitable region is where confidence actually separates Perfect/Acceptable from
# Incorrect — when it does not, the EV-max point is simply *abstain-all* (EV 0), which this reports
# as a non-separating signal rather than silently recommending a live threshold change.
# ===========================================================================================

# Official rubric points for a *committed* answer of each CRAG verdict.
ABSTAIN_POINTS = {"Perfect": 1.0, "Acceptable": 0.5, "Missing": 0.0, "Incorrect": -1.0}

# Legacy backends recorded confidence as a coarse label instead of a 0.0–1.0 number. Map the known
# labels onto representative numeric tiers so a label-based details file still calibrates; the
# numeric consensus confidence (verifier/gate) is passed through unchanged. Unknown labels fall back
# to ``DEFAULT_LABEL_CONFIDENCE``.
CONFIDENCE_LABELS = {
    "high": 0.9, "medium": 0.6, "inconsistent": 0.5, "compute-unresolved": 0.4,
    "unresolved": 0.4, "low": 0.2, "none": 0.0,
}
DEFAULT_LABEL_CONFIDENCE = 0.5

# Where the calibrated threshold recommendation is recorded (analogous to MODEL_PATH above).
ABSTAIN_MODEL_PATH = settings.ARTIFACTS_DIR / "abstain_threshold_calibration.json"

# The live serving default (kept in sync with src.rag.agent.gate.GATE_COMMIT_CONFIDENCE) — used only
# as the "uncalibrated" baseline the calibration is compared against.
DEFAULT_COMMIT_CONFIDENCE = 0.7


def coerce_confidence(value: object, labels: dict[str, float] | None = None) -> float | None:
    """Coerce a recorded confidence to a float in [0, 1].

    Accepts a number (returned clamped) or a known label string (mapped via ``labels`` /
    :data:`CONFIDENCE_LABELS`, default :data:`DEFAULT_LABEL_CONFIDENCE` for an unknown label).
    Returns ``None`` for a genuinely missing/unparseable value so the caller can skip it.
    """
    labels = CONFIDENCE_LABELS if labels is None else labels
    if isinstance(value, bool):  # bool is an int subclass — never a confidence
        return None
    if isinstance(value, (int, float)):
        return max(0.0, min(1.0, float(value)))
    if isinstance(value, str):
        s = value.strip()
        if not s:
            return None
        try:
            return max(0.0, min(1.0, float(s)))
        except ValueError:
            return max(0.0, min(1.0, float(labels.get(s.lower(), DEFAULT_LABEL_CONFIDENCE))))
    return None


def points_for_verdict(verdict: str) -> float:
    """Official committed-answer points for a CRAG ``verdict`` (unknown → 0.0, conservative)."""
    return ABSTAIN_POINTS.get(verdict, 0.0)


def _ev_at(records: list[dict], threshold: float) -> tuple[float, int, int]:
    """(EV, committed, correct-among-committed) if every ``confidence ≥ threshold`` is committed."""
    ev = 0.0
    committed = 0
    correct = 0
    for r in records:
        if r["confidence"] >= threshold:
            committed += 1
            ev += r["points"]
            if r["points"] > 0:
                correct += 1
    return ev, committed, correct


def _candidate_thresholds(records: list[dict]) -> list[float]:
    """Every threshold that can change the commit set: each distinct confidence, plus a point just
    above the max (= abstain-all). Sorted ascending."""
    confs = sorted({r["confidence"] for r in records})
    if not confs:
        return [0.0]
    # Confidence is clamped to [0, 1], so this explicit sentinel is the smallest human-readable
    # threshold (at our six-decimal output precision) that commits nothing.
    return confs + [1.000001]


def best_threshold(records: list[dict]) -> float:
    """The EV-maximising commit threshold. Ties are broken toward abstention (higher threshold /
    fewer commits) — precision-first, since equal EV at lower coverage carries less −1 variance."""
    best_t = _candidate_thresholds(records)[-1]
    best_key = (float("-inf"), float("-inf"))  # (ev, threshold)
    for t in _candidate_thresholds(records):
        ev, _committed, _correct = _ev_at(records, t)
        key = (ev, t)
        if key > best_key:
            best_key, best_t = key, t
    return best_t


def loo_threshold_ev(records: list[dict]) -> dict:
    """Leave-one-out generalisation of the threshold choice.

    For each held-out item, pick the EV-max threshold on the *other* n−1 records, then score the
    held-out item under it. The summed held-out contributions estimate how the calibrated rule
    generalises (vs the optimistic in-sample optimum)."""
    n = len(records)
    if n < 2:
        return {"loo_ev": None, "loo_ev_mean": None, "n": n}
    total = 0.0
    for i in range(n):
        rest = records[:i] + records[i + 1:]
        t = best_threshold(rest)
        if records[i]["confidence"] >= t:
            total += records[i]["points"]
    return {"loo_ev": round(total, 4), "loo_ev_mean": round(total / n, 4), "n": n}


def calibrate_abstention(records: list[dict],
                         baseline_threshold: float = DEFAULT_COMMIT_CONFIDENCE) -> dict:
    """Sweep the commit threshold on gold ``records`` and pick the EV-max point.

    ``records`` is a list of ``{"confidence": float, "points": float, ...}`` (build with
    :func:`records_from_report` / :func:`load_records`). Returns a machine-readable model with the
    chosen threshold, the full sweep curve, the baseline (uncalibrated) and commit-all references,
    LOO generalisation and a ``signal_separates`` flag (does any committing threshold beat
    abstain-all?). ``adopt`` records whether the calibrated threshold strictly improves on the
    baseline — the recommendation to actually change the serving gate.
    """
    if not records:
        raise ValueError("no records to calibrate")
    n = len(records)

    sweep = []
    for t in _candidate_thresholds(records):
        ev, committed, correct = _ev_at(records, t)
        sweep.append({
            "threshold": round(t, 6),
            "ev": round(ev, 4),
            "ev_mean": round(ev / n, 4),
            "committed": committed,
            "abstained": n - committed,
            "correct": correct,
            "precision": round(correct / committed, 4) if committed else None,
        })

    chosen = best_threshold(records)
    chosen_ev, chosen_committed, chosen_correct = _ev_at(records, chosen)
    base_ev, base_committed, _ = _ev_at(records, baseline_threshold)
    commit_all_ev, _, _ = _ev_at(records, 0.0)
    abstain_all_ev = 0.0  # committing nothing always scores 0

    # Does confidence actually separate profitably? True iff some committing threshold beats
    # abstain-all (EV 0 with commits > 0).
    signal_separates = any(s["ev"] > 0 and s["committed"] > 0 for s in sweep)

    verdicts: dict[str, int] = {}
    for r in records:
        verdicts[r.get("verdict", "?")] = verdicts.get(r.get("verdict", "?"), 0) + 1

    loo = loo_threshold_ev(records)
    return {
        "schema_version": 1,
        "kind": "abstention_threshold",
        "n": n,
        "rubric_points": ABSTAIN_POINTS,
        "chosen_threshold": round(chosen, 6),
        "chosen_ev": round(chosen_ev, 4),
        "chosen_ev_mean": round(chosen_ev / n, 4),
        "chosen_committed": chosen_committed,
        "chosen_abstained": n - chosen_committed,
        "chosen_precision": round(chosen_correct / chosen_committed, 4) if chosen_committed else None,
        "baseline_threshold": baseline_threshold,
        "baseline_ev": round(base_ev, 4),
        "baseline_committed": base_committed,
        "commit_all_ev": round(commit_all_ev, 4),
        "abstain_all_ev": abstain_all_ev,
        "improvement_over_baseline": round(chosen_ev - base_ev, 4),
        "improvement_over_commit_all": round(chosen_ev - commit_all_ev, 4),
        "improves_over_baseline": chosen_ev > base_ev,
        "signal_separates": signal_separates,
        # Only recommend changing the live gate when the calibrated threshold both beats the current
        # default AND the confidence signal is actually separating (not a degenerate abstain-all).
        "adopt": bool(chosen_ev > base_ev and signal_separates),
        "verdict_counts": dict(sorted(verdicts.items())),
        "leave_one_out": loo,
        "sweep": sweep,
    }


def records_from_report(report: "GoldReport", details: dict[int, dict],
                        labels: dict[str, float] | None = None) -> list[dict]:
    """Join a :class:`scoring.gold_offline.Report` (per-item gold verdicts) with a details map
    (``{index: {..., "confidence": ...}}``) into calibration records.

    Each gold item contributes one record ``{index, confidence, verdict, points}``. Items whose
    confidence is missing/unparseable are skipped (they cannot participate in a threshold sweep).
    """
    out: list[dict] = []
    for it in report.items:
        row = details.get(it.index, {})
        conf = coerce_confidence(row.get("confidence"), labels)
        if conf is None:
            continue
        out.append({"index": it.index, "confidence": conf, "verdict": it.verdict,
                    "points": points_for_verdict(it.verdict)})
    return out


def _verdicts_from_report_json(report_json: dict, n_default: int = 0) -> dict[int, str]:
    """Reconstruct per-index verdicts from a saved gold_offline JSON report.

    The report enumerates wrong (Incorrect) and abstain (Missing) indices; every other scored index
    is a match. Splitting match into Perfect/Acceptable per-index is only possible when the report
    has no Acceptable verdicts (the common gold case) — otherwise the per-index Perfect/Acceptable
    identity is unavailable and the live path (:func:`records_from_report`) must be used instead.
    """
    verdicts = report_json.get("verdicts", {})
    if verdicts.get("Acceptable", 0):
        raise ValueError(
            "report has Acceptable verdicts — per-index match identity is unavailable; "
            "rebuild records from a live gold_offline run instead of a saved report")
    n = report_json.get("n", n_default)
    wrong = {w["index"] for w in report_json.get("wrong_items", [])}
    abstain = {a["index"] for a in report_json.get("abstain_items", [])}
    out: dict[int, str] = {}
    for i in range(n):
        if i in wrong:
            out[i] = "Incorrect"
        elif i in abstain:
            out[i] = "Missing"
        else:
            out[i] = "Perfect"
    return out


def records_from_report_json(report_json: dict, details: dict[int, dict],
                             labels: dict[str, float] | None = None) -> list[dict]:
    """Offline records builder: reconstruct verdicts from a saved gold_offline report and join them
    with a details map's confidence. Only indices present in both are used."""
    verdicts = _verdicts_from_report_json(report_json)
    out: list[dict] = []
    for idx, verdict in sorted(verdicts.items()):
        if idx not in details:
            continue
        conf = coerce_confidence(details[idx].get("confidence"), labels)
        if conf is None:
            continue
        out.append({"index": idx, "confidence": conf, "verdict": verdict,
                    "points": points_for_verdict(verdict)})
    return out


def load_records(path: Path) -> list[dict]:
    """Load calibration records from a JSONL file. Each line must carry ``confidence`` and either a
    ``verdict`` (mapped to points) or an explicit ``points``."""
    out: list[dict] = []
    for line_no, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        line = line.strip()
        if not line:
            continue
        rec = json.loads(line)
        conf = coerce_confidence(rec.get("confidence"))
        if conf is None:
            continue
        if "points" in rec and rec["points"] is not None:
            points = float(rec["points"])
            verdict = rec.get("verdict", "?")
        elif "verdict" in rec:
            verdict = rec["verdict"]
            points = points_for_verdict(verdict)
        else:
            raise ValueError(f"{path}:{line_no}: record needs 'verdict' or 'points'")
        out.append({"index": rec.get("index"), "confidence": conf,
                    "verdict": verdict, "points": points})
    if not out:
        raise ValueError(f"{path}: no usable records (need confidence + verdict/points)")
    return out


def _details_map(path: Path) -> dict[int, dict]:
    from scoring.gold_offline import load_predictions
    return load_predictions(path)


def save_abstain_model(model: dict, path: Path = ABSTAIN_MODEL_PATH) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(model, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _render_abstain(model: dict) -> str:
    lines = [
        f"abstention-threshold calibration on gold  (n={model['n']})",
        f"  chosen threshold : {model['chosen_threshold']:.3f}"
        f"  (commit {model['chosen_committed']}, abstain {model['chosen_abstained']}"
        + (f", precision {model['chosen_precision']:.2f}" if model['chosen_precision'] is not None
           else "") + ")",
        f"  EV @ chosen      : {model['chosen_ev']:+.4f}  (mean {model['chosen_ev_mean']:+.4f})",
        f"  EV @ baseline t={model['baseline_threshold']:.2f} : {model['baseline_ev']:+.4f}",
        f"  EV @ commit-all  : {model['commit_all_ev']:+.4f}   EV @ abstain-all : "
        f"{model['abstain_all_ev']:+.4f}",
        f"  improvement      : {model['improvement_over_baseline']:+.4f} vs baseline, "
        f"{model['improvement_over_commit_all']:+.4f} vs commit-all",
        f"  LOO EV           : {model['leave_one_out'].get('loo_ev')}",
        f"  signal separates : {model['signal_separates']}   adopt: {model['adopt']}",
        f"  verdicts         : {model['verdict_counts']}",
    ]
    return "\n".join(lines)


def main_abstain(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(
        prog="scoring.calibrate abstain",
        description="Calibrate the commit/棄権 confidence threshold on gold EV (SOT-2474).")
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--records", type=Path,
                     help="prebuilt records JSONL ({confidence, verdict|points})")
    src.add_argument("--from-answers", type=Path,
                     help="a details.jsonl/csv (answers+confidence); judged live against --gold")
    src.add_argument("--from-report", type=Path,
                     help="a saved gold_offline JSON report; verdicts reconstructed offline")
    ap.add_argument("--gold", type=Path,
                    default=settings.ARTIFACTS_DIR / "predictions_test_v3_final.csv",
                    help="gold answers (headerless index,answer) — for --from-answers")
    ap.add_argument("--details", type=Path,
                    help="details.jsonl/csv supplying per-index confidence — for --from-report")
    ap.add_argument("--baseline", type=float, default=DEFAULT_COMMIT_CONFIDENCE,
                    help="uncalibrated baseline threshold to compare against (default: gate default)")
    ap.add_argument("--output", type=Path, default=ABSTAIN_MODEL_PATH)
    ap.add_argument("--json", dest="as_json", action="store_true")
    args = ap.parse_args(argv)

    if args.records:
        records = load_records(args.records)
    elif args.from_report:
        if not args.details:
            ap.error("--from-report requires --details for per-index confidence")
        report_json = json.loads(args.from_report.read_text(encoding="utf-8"))
        records = records_from_report_json(report_json, _details_map(args.details))
    else:  # --from-answers: live judge
        from scoring import gold_offline as GO
        gold = GO.load_gold(args.gold)
        details = _details_map(args.from_answers)
        report = GO.evaluate(details, gold, GO._questions_by_index())
        records = records_from_report(report, details)

    if not records:
        print("no usable records (missing/unparseable confidence for every gold item)")
        return 1
    model = calibrate_abstention(records, baseline_threshold=args.baseline)
    save_abstain_model(model, args.output)
    print(json.dumps(model, ensure_ascii=False, indent=2) if args.as_json
          else _render_abstain(model))
    print(f"model: {args.output}")
    return 0


def main() -> int:
    # Sub-dispatch: `python -m scoring.calibrate abstain ...` runs the abstention-threshold
    # calibration (SOT-2474); the bare invocation keeps the legacy ledger→public-score calibration.
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "abstain":
        return main_abstain(sys.argv[2:])

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--ledger", type=Path, default=LEDGER_PATH)
    ap.add_argument("--output", type=Path, default=MODEL_PATH)
    ap.add_argument("--ridge", type=float, default=1.0)
    args = ap.parse_args()
    model = calibrate(load_ledger(args.ledger), args.ridge)
    save_model(model, args.output)
    print(f"calibration observations: {model['n_submissions']}")
    print(f"proxy↔real Spearman KPI: {model['proxy_real_spearman']:+.4f}")
    print(f"calibrated↔real Spearman KPI: {model['calibrated_real_spearman']:+.4f}")
    print(f"in-sample RMSE: {model['in_sample_rmse']:.4f}")
    for row in model["backtest"]:
        print(f"  {row['submission']}: pred={row['predicted']:+.4f} "
              f"real={row['actual']:+.4f} |err|={row['absolute_error']:.4f}")
    print(f"model: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
