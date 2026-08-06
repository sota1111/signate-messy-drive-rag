"""SOT-2483 — gold-100 abstention reason breakdown + recoverable-commit analysis.

The gold-100 offline run (``gen='resolve'``: investigator → heterogeneous verifier → third-judge
tie-break, :mod:`src.rag.agent.tiebreak`) abstains on ~82/100 items. That over-abstention is *not*
uniform, and the parent Issue's premise — "根拠が強いのに棄権 (strong evidence yet abstained)" — is
testable: this module classifies every abstention by *why* the consensus declined to commit, and
separates the genuinely unrecoverable (both agents independently found no answer) from the
*recoverable* ones (at least one agent proposed a value a relaxed gate could commit).

Reason taxonomy (deterministic — read straight off the resolve ``details.jsonl``, no LLM):

- ``both_abstain``          — investigator AND verifier both abstained → 真の無回答. NOT recoverable.
- ``one_side_abstain``      — exactly one side proposed a value, the other abstained; tie-break was
                              invoked but the consensus stayed 棄権. Recoverable (candidate = the
                              proposing side, promoted by the third judge when it adjudicated).
- ``verifier_disagreement`` — both sides proposed *different* values; tie-break could not reconcile
                              them. Recoverable (candidate = the third judge's adjudicated answer).
- ``enumeration_mismatch``  — an enumeration whose element sets differed between the agents.
                              Recoverable (candidate = the adjudicated / larger set).
- ``other``                 — any residual outcome (defensive; recoverable iff a value survives).

**Recovery EV.** Committing a recovered candidate scores the official rubric (Perfect +1 /
Acceptable +0.5 / Missing 0 / Incorrect −1) against gold. Because the recoverable pool is dominated
by the *disagreement* region — precisely where committing is EV-negative — whether recovery improves
the expected value is an EMPIRICAL question. :func:`recovery_records` judges the candidates (via the
shared :mod:`scoring.crag` judge, the same grader the rest of the stack uses) so the calibration in
:mod:`scoring.calibrate` chooses a recovery threshold on *measured* EV, never on the fallacy that
"abstained ⇒ safe to commit".

The classification core is a **pure** function over one resolve record (:func:`classify_record`),
trivially testable with constructed dicts; the CLI wires the live details/gold/judge end-to-end.
"""
from __future__ import annotations

import argparse
import csv
import json
from dataclasses import asdict, dataclass
from pathlib import Path

from config import settings

ABSTAIN = settings.ABSTAIN

# Reason labels (stable identifiers used in the JSON report and by tests).
BOTH_ABSTAIN = "both_abstain"
ONE_SIDE_ABSTAIN = "one_side_abstain"
VERIFIER_DISAGREEMENT = "verifier_disagreement"
ENUMERATION_MISMATCH = "enumeration_mismatch"
OTHER = "other"

ALL_REASONS = (BOTH_ABSTAIN, ONE_SIDE_ABSTAIN, VERIFIER_DISAGREEMENT,
               ENUMERATION_MISMATCH, OTHER)


def _val(text: object) -> str | None:
    """A proposed *value*, or ``None`` when the text is empty / the ABSTAIN sentinel."""
    s = str(text or "").strip()
    return None if s == "" or s == ABSTAIN else s


def is_abstained(record: dict) -> bool:
    """True when the served resolve answer is an abstention (empty / ABSTAIN sentinel)."""
    return _val(record.get("answer")) is None


@dataclass(frozen=True)
class Abstention:
    """One classified abstained gold item.

    ``recoverable`` marks that a committable ``candidate`` value exists (from the judge's
    adjudication or the single proposing side); ``confidence`` is the recovery-time confidence used
    for the EV sweep (the third judge's confidence when it adjudicated, else the surviving side's).
    """

    index: int
    reason: str
    recoverable: bool
    candidate: str | None
    candidate_source: str        # "judge" | "verifier" | "investigator" | ""
    confidence: float
    status: str
    category: str
    archetype: str

    def to_dict(self) -> dict:
        return asdict(self)


def _category(record: dict) -> str:
    verdict = record.get("verdict")
    if isinstance(verdict, dict):
        return str(verdict.get("category") or "")
    return ""


def _verifier_confidence(record: dict) -> float | None:
    verdict = record.get("verdict")
    if isinstance(verdict, dict):
        c = verdict.get("verifier_confidence")
        if isinstance(c, (int, float)) and not isinstance(c, bool):
            return float(c)
    return None


def classify_record(record: dict, *, default_confidence: float = 0.5) -> Abstention | None:
    """Classify one resolve ``details.jsonl`` record. Returns ``None`` for a committed item.

    Pure over the record: reads the served answer, the per-agent proposals
    (``investigator_answer`` / ``verifier_answer``), the third judge's ``judge_answer`` /
    ``judge_confidence`` and the resolution ``status`` / ``verdict.category``. No network / LLM.
    """
    if not is_abstained(record):
        return None

    inv = _val(record.get("investigator_answer"))
    ver = _val(record.get("verifier_answer"))
    jud = _val(record.get("judge_answer"))
    status = str(record.get("status") or "")
    category = _category(record)
    archetype = str(record.get("archetype") or "")

    # The committable candidate: prefer the third judge's adjudicated value (it saw both sides), else
    # the single side that proposed a value. When both sides abstained there is nothing to commit.
    jconf = record.get("judge_confidence")
    jconf = float(jconf) if isinstance(jconf, (int, float)) and not isinstance(jconf, bool) else None

    candidate: str | None = None
    source = ""
    confidence = 0.0
    if jud is not None and (jconf is None or jconf > 0.0):
        candidate, source = jud, "judge"
        confidence = jconf if jconf is not None else default_confidence
    elif ver is not None and inv is None:
        candidate, source = ver, "verifier"
        confidence = _verifier_confidence(record) or default_confidence
    elif inv is not None and ver is None:
        candidate, source = inv, "investigator"
        confidence = default_confidence
    elif ver is not None and inv is not None:
        # Both proposed but the judge did not leave a usable value — fall back to the verifier
        # (heterogeneous, independent re-derivation) as the representative candidate.
        candidate, source = ver, "verifier"
        confidence = _verifier_confidence(record) or default_confidence

    # Reason: describe *why* the consensus abstained, independent of candidate selection.
    if inv is None and ver is None:
        reason = BOTH_ABSTAIN
    elif inv is None or ver is None:
        reason = ONE_SIDE_ABSTAIN
    elif category == "enumeration":
        reason = ENUMERATION_MISMATCH
    elif record.get("agree") is False:
        reason = VERIFIER_DISAGREEMENT
    else:
        reason = OTHER

    recoverable = candidate is not None
    return Abstention(
        index=int(record.get("index", -1)), reason=reason, recoverable=recoverable,
        candidate=candidate, candidate_source=source, confidence=float(confidence),
        status=status, category=category, archetype=archetype,
    )


def classify_all(records: list[dict]) -> list[Abstention]:
    """Classify every abstained record (committed items are skipped)."""
    out = []
    for r in records:
        cls = classify_record(r)
        if cls is not None:
            out.append(cls)
    return out


def breakdown(records: list[dict]) -> dict:
    """Deterministic abstention-reason breakdown over resolve ``details`` records.

    Returns counts by reason, the recoverable / unrecoverable split, per-reason recoverable counts,
    a per-archetype view and the full recoverable-candidate list (index/source/confidence). This is
    the ``artifacts/gold_abstain_breakdown.json`` payload — no judge / gold needed.
    """
    n_total = len(records)
    n_committed = sum(1 for r in records if not is_abstained(r))
    classified = classify_all(records)

    by_reason = {r: 0 for r in ALL_REASONS}
    recoverable_by_reason = {r: 0 for r in ALL_REASONS}
    by_archetype: dict[str, dict] = {}
    for a in classified:
        by_reason[a.reason] = by_reason.get(a.reason, 0) + 1
        if a.recoverable:
            recoverable_by_reason[a.reason] = recoverable_by_reason.get(a.reason, 0) + 1
        b = by_archetype.setdefault(a.archetype or "unknown",
                                    {"abstained": 0, "recoverable": 0})
        b["abstained"] += 1
        b["recoverable"] += int(a.recoverable)

    recoverable = [a for a in classified if a.recoverable]
    return {
        "schema_version": 1,
        "kind": "abstain_breakdown",
        "issue": "SOT-2483",
        "n_total": n_total,
        "n_committed": n_committed,
        "n_abstained": len(classified),
        "n_recoverable": len(recoverable),
        "n_unrecoverable": len(classified) - len(recoverable),
        "by_reason": dict(sorted(by_reason.items())),
        "recoverable_by_reason": dict(sorted(recoverable_by_reason.items())),
        "by_archetype": dict(sorted(by_archetype.items())),
        "candidate_source_counts": _source_counts(recoverable),
        "recoverable_items": [a.to_dict() for a in sorted(recoverable, key=lambda x: x.index)],
    }


def _source_counts(items: list[Abstention]) -> dict[str, int]:
    out: dict[str, int] = {}
    for a in items:
        out[a.candidate_source] = out.get(a.candidate_source, 0) + 1
    return dict(sorted(out.items()))


# --------------------------------------------------------------------------- recovery EV (needs gold)
def recovery_records(records: list[dict], gold: dict[int, str], judge=None) -> list[dict]:
    """Judge each recoverable candidate against gold → calibration records for EV analysis.

    Each returned record is ``{index, confidence, verdict, points, candidate_source}`` — the schema
    :func:`scoring.calibrate.calibrate_abstention` consumes. Candidates whose gold answer is missing
    are skipped. ``judge`` defaults to the shared CRAG judge (GPT-family, SOT-2457); pass a stub in
    tests to stay network-free.
    """
    from scoring.calibrate import points_for_verdict

    if judge is None:
        from scoring.gold_offline import default_judge as judge  # noqa: N806

    recoverable = [a for a in classify_all(records)
                   if a.recoverable and a.index in gold and a.candidate is not None]
    if not recoverable:
        return []
    pairs = [(a.candidate, gold[a.index]) for a in recoverable]
    verdicts = judge(pairs)
    if len(verdicts) != len(recoverable):
        raise ValueError(f"judge returned {len(verdicts)} verdicts for {len(recoverable)} pairs")
    out = []
    for a, v in zip(recoverable, verdicts):
        out.append({"index": a.index, "confidence": a.confidence, "verdict": v,
                    "points": points_for_verdict(v), "candidate_source": a.candidate_source})
    return out


def recovery_ev(recovery_recs: list[dict]) -> dict:
    """EV of committing recovered candidates, plus the calibrated recovery threshold (SOT-2474).

    Reports commit-all-recoverable EV (the maximally-aggressive recovery), the per-verdict split and
    the EV-max confidence threshold from :func:`scoring.calibrate.calibrate_abstention`. All figures
    are the *marginal* gain over abstaining (each recovered item currently scores 0), so a negative
    ``commit_all_ev`` means blanket recovery would *hurt* — precisely the precision-first guard.
    """
    from scoring.calibrate import calibrate_abstention

    verdict_counts: dict[str, int] = {}
    for r in recovery_recs:
        verdict_counts[r["verdict"]] = verdict_counts.get(r["verdict"], 0) + 1
    commit_all_ev = round(sum(r["points"] for r in recovery_recs), 4)
    calib = calibrate_abstention(recovery_recs) if recovery_recs else None
    return {
        "n_recoverable_judged": len(recovery_recs),
        "commit_all_recovery_ev": commit_all_ev,
        "recovery_verdict_counts": dict(sorted(verdict_counts.items())),
        # The EV-optimal recovery: commit only recovered candidates whose confidence clears the
        # calibrated threshold. ``adopt`` (calibrate) is True only if this beats abstaining AND the
        # confidence signal actually separates.
        "calibrated_recovery": calib,
    }


# --------------------------------------------------------------------------- IO helpers / CLI
def load_details(path: Path) -> list[dict]:
    """Load a resolve ``details.jsonl`` into an ordered list of records."""
    out = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            out.append(json.loads(line))
    return out


def load_gold(path: Path) -> dict[int, str]:
    gold: dict[int, str] = {}
    with path.open(encoding="utf-8-sig", newline="") as fh:
        for row in csv.reader(fh):
            if row:
                gold[int(row[0])] = row[1] if len(row) > 1 else ""
    return gold


def render(report: dict) -> str:
    lines = [
        f"gold abstention breakdown  (n={report['n_total']}, "
        f"committed={report['n_committed']}, abstained={report['n_abstained']})",
        f"  recoverable   : {report['n_recoverable']:>3}  "
        f"(a value was proposed a relaxed gate could commit)",
        f"  unrecoverable : {report['n_unrecoverable']:>3}  (真の無回答 — both sides abstained)",
        "  by reason:",
    ]
    for reason, count in report["by_reason"].items():
        rec = report["recoverable_by_reason"].get(reason, 0)
        lines.append(f"    {reason:<22} {count:>3}   (recoverable {rec})")
    if "recovery" in report:
        rv = report["recovery"]
        lines += [
            "  recovery EV (judged vs gold):",
            f"    candidates judged     : {rv['n_recoverable_judged']}",
            f"    commit-all recovery EV: {rv['commit_all_recovery_ev']:+.4f}  "
            f"(marginal over abstaining)",
            f"    verdicts              : {rv['recovery_verdict_counts']}",
        ]
        calib = rv.get("calibrated_recovery")
        if calib:
            lines.append(
                f"    calibrated recovery   : threshold {calib['chosen_threshold']:.3f}, "
                f"EV {calib['chosen_ev']:+.4f}, adopt={calib['adopt']}, "
                f"signal_separates={calib['signal_separates']}")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        prog="scoring.abstain_breakdown",
        description="Classify gold abstentions by reason and quantify recoverable commits (SOT-2483).")
    ap.add_argument("--details", type=Path,
                    default=settings.ARTIFACTS_DIR / "predictions_test_resolve.details.jsonl",
                    help="resolve details.jsonl (per-item investigator/verifier/judge answers)")
    ap.add_argument("--gold", type=Path,
                    default=settings.ARTIFACTS_DIR / "predictions_test_v3_final.csv",
                    help="gold answers (headerless index,answer) — required for --recovery-ev")
    ap.add_argument("--recovery-ev", action="store_true",
                    help="also judge recoverable candidates against gold and report recovery EV "
                         "(uses the shared CRAG judge; costs LLM calls)")
    ap.add_argument("--out", type=Path,
                    default=settings.ARTIFACTS_DIR / "gold_abstain_breakdown.json",
                    help="write the JSON breakdown report here")
    ap.add_argument("--json", dest="as_json", action="store_true",
                    help="print the full JSON report instead of the human summary")
    args = ap.parse_args(argv)

    records = load_details(args.details)
    report = breakdown(records)

    if args.recovery_ev:
        gold = load_gold(args.gold)
        recs = recovery_records(records, gold)
        report["recovery"] = recovery_ev(recs)
        report["recovery"]["records"] = recs

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2) if args.as_json else render(report))
    print(f"report: {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
