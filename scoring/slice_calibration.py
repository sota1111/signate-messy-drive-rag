"""SOT-2503 — per-contract-slice EV calibration of the answer gate.

Parent SOT-2460 Step10. The consensus gate (:mod:`src.rag.agent.gate`) applies **one global**
commit/棄権 confidence threshold to every question. But the residual risk is *not* uniform across
question kinds: an easy 単純検索 (simple_lookup) answer of high measured precision is smothered by the
same abstain-leaning bar that (correctly) guards a dangerous グラフ読取 / 書式判定 slice. SOT-2483 already
established that a **global** commit-all relaxation is EV-negative; this Issue is the *different* axis —
relax the gate **only for the contract-type slices whose measured EV is positive**, on the new baseline
that lands after the capability merges (routing / obligation loop / closure / exec-verify).

What "slice" means
------------------
Every gold item is labelled with its :mod:`~src.rag.agent.question_contract` (the nine contracts:
``simple_lookup`` / ``multi_hop`` / ``cross_aggregate`` / ``full_enumeration`` / ``format_check`` /
``chart_read`` / ``spatial`` / ``version_diff`` / ``numeric``). A *slice* is all calibration records
sharing a contract. Per slice we sweep the commit threshold exactly like the global calibration
(:func:`scoring.calibrate.best_threshold`, SOT-2474) and decide — on **measured** EV, never on a
"abstained ⇒ safe to commit" fallacy — whether to relax that slice's bar.

Adoption rule (precision-first, fail-safe)
------------------------------------------
Under the official rubric (Perfect +1 / Acceptable +0.5 / Missing 0 / Incorrect −1) a committed answer
of would-be verdict ``v`` scores ``POINTS[v]``; abstaining scores 0. A slice's relaxed threshold is
adopted only when **all** hold — otherwise the slice keeps the current global bar (no change):

1. **Genuine relaxation** — the EV-max threshold is strictly *below* the global baseline (it commits
   strictly more; a tightening is the already-shipped abstain-leaning default, out of scope here).
2. **Measured EV>0** — the relaxed commit set has positive absolute EV *and* beats the baseline commit
   set on this slice (受け入れ条件①「緩和は EV>0 実測スライスのみ」).
3. **WRONG 非増加** — relaxing introduces **zero** additional Incorrect on the calibration set
   (``wrong@relaxed ≤ wrong@baseline``), so the global WRONG total cannot rise (受け入れ条件①後半).
4. **Robust to judge noise** — the CRAG judge flips ≈1/30 of verdicts; requiring the EV to stay
   positive after conservatively flipping ``ceil(noise·committed)`` correct commits to Incorrect
   (each a −2 swing) guards a slice whose positive EV is only an artefact of judge fluctuation
   (実装内容「判定ゆらぎ≈1/30 を考慮」).

The output is a machine-readable model: per-slice sweeps plus an ``adopted_thresholds`` map
``{contract: relaxed_threshold}`` that the gate consumes. It is **fail-safe by construction**: an empty
map (the default — no calibration file, ``GATE_SLICE_CALIBRATE`` OFF) leaves the gate byte-identical to
today. Actually changing the live gate requires wiring the file *and* the env toggle, gated behind
関門2 非劣化 (SOT-2478) / 実LB confirmation exactly like the other opt-in capability layers.

The calibration core (:func:`calibrate_slices`) is a **pure** function over records — trivially
testable with constructed dicts, no network / LLM. The CLI wires a saved gold report + details.
"""
from __future__ import annotations

import argparse
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path

from config import settings
from scoring import calibrate as _cal

# Default CRAG judge flip rate used to stress the EV estimate (≈1/30 — one flip per gold-30 gate).
DEFAULT_JUDGE_NOISE = 1.0 / 30.0
# The live serving default commit threshold (mirrors src.rag.agent.gate.GATE_COMMIT_CONFIDENCE).
BASELINE_THRESHOLD = _cal.DEFAULT_COMMIT_CONFIDENCE
SLICE_MODEL_PATH = settings.ARTIFACTS_DIR / "slice_gate_calibration.json"


def _slice_stats(records: list[dict], threshold: float) -> dict:
    """Commit every ``confidence ≥ threshold`` record in a slice → ``{committed, correct, wrong, ev}``.

    ``correct`` = committed items scoring >0 (Perfect/Acceptable), ``wrong`` = committed Incorrect
    (points <0); Missing (0) is neither. ``ev`` is the summed official points of the commit set (the
    marginal gain over abstaining, since an abstention scores 0)."""
    committed = correct = wrong = 0
    ev = 0.0
    for r in records:
        if r["confidence"] >= threshold:
            committed += 1
            ev += r["points"]
            if r["points"] > 0:
                correct += 1
            elif r["points"] < 0:
                wrong += 1
    return {"committed": committed, "correct": correct, "wrong": wrong, "ev": round(ev, 4)}


@dataclass(frozen=True)
class SliceCalibration:
    """Per-contract calibration verdict: the relaxed threshold and whether it is safe to adopt."""

    contract: str
    n: int
    baseline_threshold: float
    chosen_threshold: float
    baseline: dict            # commit stats at the global baseline threshold
    relaxed: dict             # commit stats at the EV-max (relaxed) threshold
    added_committed: int      # extra commits the relaxation makes vs baseline
    added_correct: int
    added_wrong: int
    judge_noise: float
    robust_ev: float          # relaxed EV after a conservative judge-flip stress
    adopt: bool
    reason: str

    def to_dict(self) -> dict:
        return asdict(self)


def calibrate_slice(contract: str, records: list[dict], *,
                    baseline_threshold: float = BASELINE_THRESHOLD,
                    judge_noise: float = DEFAULT_JUDGE_NOISE) -> SliceCalibration:
    """Calibrate one contract slice — sweep the commit threshold and apply the adoption rule.

    See the module docstring for the four-condition adoption rule (relaxation / EV>0 / WRONG非増加 /
    judge-noise robustness). A slice that fails any condition returns ``adopt=False`` with the global
    baseline as its ``chosen_threshold`` (no change)."""
    n = len(records)
    base = _slice_stats(records, baseline_threshold)
    chosen = _cal.best_threshold(records) if records else baseline_threshold
    relaxed = _slice_stats(records, chosen)

    added_committed = relaxed["committed"] - base["committed"]
    added_correct = relaxed["correct"] - base["correct"]
    added_wrong = relaxed["wrong"] - base["wrong"]

    # Conservative judge-noise stress: assume ceil(noise·committed) of the relaxed commit's correct
    # answers were actually Incorrect (each a +1→−1 = −2 EV swing).
    flips = math.ceil(judge_noise * relaxed["committed"])
    robust_ev = round(relaxed["ev"] - 2.0 * flips, 4)

    is_relaxation = chosen < baseline_threshold
    ev_positive = relaxed["ev"] > 0 and relaxed["ev"] > base["ev"]
    wrong_nonincreasing = relaxed["wrong"] <= base["wrong"]
    robust = robust_ev > 0

    adopt = bool(is_relaxation and ev_positive and wrong_nonincreasing and robust and added_committed > 0)
    if not records:
        reason = "スライスにレコード無し → 変更なし"
    elif not is_relaxation:
        reason = f"EV最大しきい値 {chosen:.3f} ≥ baseline {baseline_threshold:.3f} → 緩和対象外(変更なし)"
    elif not ev_positive:
        reason = f"緩和後EV {relaxed['ev']:+.3f} が非正 or baseline {base['ev']:+.3f} 以下 → 棄却"
    elif not wrong_nonincreasing:
        reason = f"緩和で誤答増加(wrong {base['wrong']}→{relaxed['wrong']}) → 棄却"
    elif not robust:
        reason = f"判定ゆらぎ考慮後EV {robust_ev:+.3f} が非正(flips={flips}) → 棄却"
    else:
        reason = (f"緩和採用: しきい値 {baseline_threshold:.3f}→{chosen:.3f}, "
                  f"EV {base['ev']:+.3f}→{relaxed['ev']:+.3f}, 追加commit +{added_committed}"
                  f"(正{added_correct:+d}/誤{added_wrong:+d}), robustEV {robust_ev:+.3f}")

    return SliceCalibration(
        contract=contract, n=n, baseline_threshold=baseline_threshold,
        chosen_threshold=chosen if adopt else baseline_threshold,
        baseline=base, relaxed=relaxed, added_committed=added_committed,
        added_correct=added_correct, added_wrong=added_wrong, judge_noise=judge_noise,
        robust_ev=robust_ev, adopt=adopt, reason=reason,
    )


def calibrate_slices(records: list[dict], *,
                     baseline_threshold: float = BASELINE_THRESHOLD,
                     judge_noise: float = DEFAULT_JUDGE_NOISE) -> dict:
    """Group ``records`` by contract and calibrate each slice → the machine-readable slice model.

    ``records`` is a list of ``{contract, confidence, verdict, points, ...}`` (build with
    :func:`slice_records_from_report` / :func:`records_from_report_json`). Returns the per-slice
    sweeps, the adopted ``{contract: relaxed_threshold}`` map (only EV>0 / WRONG-safe slices), and the
    global WRONG/EV totals under the baseline vs the relaxed gate — with a ``wrong_nonincreasing`` flag
    that is the machine-checkable form of 受け入れ条件①."""
    by_contract: dict[str, list[dict]] = {}
    for r in records:
        by_contract.setdefault(str(r.get("contract") or "unknown"), []).append(r)

    slices = [calibrate_slice(c, by_contract[c], baseline_threshold=baseline_threshold,
                              judge_noise=judge_noise)
              for c in sorted(by_contract)]

    adopted = {s.contract: s.chosen_threshold for s in slices if s.adopt}

    # Global WRONG/EV totals: baseline gate everywhere vs the relaxed threshold on adopted slices.
    wrong_baseline = ev_baseline = 0
    wrong_relaxed = 0
    ev_relaxed = 0.0
    for s in slices:
        wrong_baseline += s.baseline["wrong"]
        ev_baseline += s.baseline["ev"]
        thr = adopted.get(s.contract, baseline_threshold)
        eff = _slice_stats(by_contract[s.contract], thr)
        wrong_relaxed += eff["wrong"]
        ev_relaxed += eff["ev"]

    return {
        "schema_version": 1,
        "kind": "slice_gate_calibration",
        "issue": "SOT-2503",
        "n": len(records),
        "baseline_threshold": baseline_threshold,
        "judge_noise": judge_noise,
        "n_slices": len(slices),
        "n_adopted": len(adopted),
        "adopted_thresholds": adopted,
        "wrong_total_baseline": wrong_baseline,
        "wrong_total_relaxed": wrong_relaxed,
        "wrong_nonincreasing": wrong_relaxed <= wrong_baseline,
        "ev_total_baseline": round(ev_baseline, 4),
        "ev_total_relaxed": round(ev_relaxed, 4),
        "ev_gain": round(ev_relaxed - ev_baseline, 4),
        "slices": [s.to_dict() for s in slices],
    }


def to_gate_thresholds(model: dict) -> dict[str, float]:
    """The ``{contract: relaxed_threshold}`` map the gate consumes (empty ⇒ gate unchanged)."""
    thresholds = model.get("adopted_thresholds", {}) if isinstance(model, dict) else {}
    return {str(k): float(v) for k, v in thresholds.items()}


# --------------------------------------------------------------------------- record builders
def _contract_of(question: str) -> str:
    """Deterministic (network-free) contract label for a question (flash arbiter never consulted)."""
    from src.rag.agent import question_contract as qc

    q = (question or "").strip()
    return qc.classify(q).contract if q else qc.SIMPLE_LOOKUP


def slice_records_from_report(report, details: dict[int, dict],
                              labels: dict[str, float] | None = None) -> list[dict]:
    """Join a live :class:`scoring.gold_offline.Report` with a details map into slice records.

    Each scored gold item contributes ``{index, contract, confidence, verdict, points}``; the contract
    is classified deterministically from the item's question. Items whose confidence is
    missing/unparseable are skipped (they cannot participate in a threshold sweep)."""
    out: list[dict] = []
    for it in report.items:
        row = details.get(it.index, {})
        conf = _cal.coerce_confidence(row.get("confidence"), labels)
        if conf is None:
            continue
        out.append({"index": it.index, "contract": _contract_of(it.question),
                    "confidence": conf, "verdict": it.verdict,
                    "points": _cal.points_for_verdict(it.verdict)})
    return out


def records_from_report_json(report_json: dict, details: dict[int, dict],
                             labels: dict[str, float] | None = None) -> list[dict]:
    """Offline builder: reconstruct verdicts from a saved gold_offline report (:func:`scoring.calibrate
    .records_from_report_json`) and attach each item's contract from the details' question text."""
    base = _cal.records_from_report_json(report_json, details, labels)
    for r in base:
        q = str(details.get(r["index"], {}).get("question", "") or "")
        r["contract"] = _contract_of(q)
    return base


def save_model(model: dict, path: Path = SLICE_MODEL_PATH) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(model, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def render(model: dict) -> str:
    lines = [
        f"per-contract slice gate calibration  (n={model['n']}, "
        f"baseline t={model['baseline_threshold']:.2f}, noise≈{model['judge_noise']:.3f})",
        f"  adopted slices : {model['n_adopted']}/{model['n_slices']}   "
        f"WRONG {model['wrong_total_baseline']}→{model['wrong_total_relaxed']} "
        f"({'非増加 OK' if model['wrong_nonincreasing'] else '増加 NG'})   "
        f"EV {model['ev_total_baseline']:+.3f}→{model['ev_total_relaxed']:+.3f} "
        f"({model['ev_gain']:+.3f})",
        "  by slice:",
    ]
    for s in model["slices"]:
        flag = "ADOPT" if s["adopt"] else "keep "
        lines.append(
            f"    [{flag}] {s['contract']:<16} n={s['n']:>2}  "
            f"t {s['baseline_threshold']:.2f}→{s['chosen_threshold']:.2f}  "
            f"EV {s['baseline']['ev']:+.2f}→{s['relaxed']['ev']:+.2f}  "
            f"wrong {s['baseline']['wrong']}→{s['relaxed']['wrong']}")
        lines.append(f"             {s['reason']}")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        prog="scoring.slice_calibration",
        description="Per-contract-slice EV calibration of the answer gate (SOT-2503).")
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--records", type=Path,
                     help="prebuilt slice records JSONL ({contract, confidence, verdict|points})")
    src.add_argument("--from-report", type=Path,
                     help="a saved gold_offline JSON report; verdicts reconstructed offline")
    ap.add_argument("--details", type=Path,
                    help="details.jsonl/csv supplying per-index confidence + question (for --from-report)")
    ap.add_argument("--baseline", type=float, default=BASELINE_THRESHOLD,
                    help="uncalibrated baseline commit threshold (default: gate default)")
    ap.add_argument("--judge-noise", type=float, default=DEFAULT_JUDGE_NOISE,
                    help="CRAG judge flip rate used to stress EV (default ≈1/30)")
    ap.add_argument("--output", type=Path, default=SLICE_MODEL_PATH)
    ap.add_argument("--json", dest="as_json", action="store_true")
    args = ap.parse_args(argv)

    if args.records:
        records = []
        for line in args.records.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            conf = _cal.coerce_confidence(rec.get("confidence"))
            if conf is None:
                continue
            points = (float(rec["points"]) if rec.get("points") is not None
                      else _cal.points_for_verdict(rec.get("verdict", "?")))
            records.append({"index": rec.get("index"),
                            "contract": str(rec.get("contract") or "unknown"),
                            "confidence": conf, "verdict": rec.get("verdict", "?"),
                            "points": points})
    else:
        if not args.details:
            ap.error("--from-report requires --details for per-index confidence + question")
        from scoring.calibrate import _details_map
        report_json = json.loads(args.from_report.read_text(encoding="utf-8"))
        records = records_from_report_json(report_json, _details_map(args.details))

    if not records:
        print("no usable slice records (missing confidence/contract for every gold item)")
        return 1
    model = calibrate_slices(records, baseline_threshold=args.baseline,
                             judge_noise=args.judge_noise)
    save_model(model, args.output)
    print(json.dumps(model, ensure_ascii=False, indent=2) if args.as_json else render(model))
    print(f"model: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
