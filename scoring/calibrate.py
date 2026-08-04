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


def load_ledger(path: Path = LEDGER_PATH) -> list[dict]:
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
        "backtest": backtest,
        "leave_one_out": loo,
    }


def save_model(model: dict, path: Path = MODEL_PATH) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(model, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def main() -> int:
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
