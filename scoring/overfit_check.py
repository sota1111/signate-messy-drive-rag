"""Overfit detector — flag changes that improved the dev/valid slice but NOT the hold-out slice.

The #4 regression (valid30 proxy climbed to +0.5833 while the real Public30 submission *dropped* to
−0.1) is the failure mode this guards against: a change that looks like an improvement on the visible
dev set but fails to generalize. We measure both slices every self-improvement run
(`scoring.selfimprove` appends to ``artifacts/holdout_history.jsonl``) and compare consecutive runs:

    dev_gain     = curr.dev_mean     - prev.dev_mean
    holdout_gain = curr.holdout_mean - prev.holdout_mean

A change is *overfit-suspected* (adoption should be BLOCKED) when the dev slice improved materially
but the hold-out slice did not keep up — concretely when ``dev_gain > EPS`` and
``holdout_gain < dev_gain - MARGIN`` (a hold-out that stalls or drops relative to a rising dev). A
hold-out that actually fell (``holdout_gain < 0``) while dev rose is the strongest signal.

    python -m scoring.overfit_check                 # compare the last two history entries
    python -m scoring.overfit_check --dev-gain 0.5833 --holdout-gain -0.1   # ad-hoc / the #4 state

Exit code: 0 = OK (safe to adopt), 3 = overfit suspected (block adoption), 2 = insufficient history.
"""
from __future__ import annotations

import argparse
import json
from dataclasses import dataclass

from config import settings

HISTORY_PATH = settings.ARTIFACTS_DIR / "holdout_history.jsonl"

# A dev gain smaller than EPS is treated as noise (no adoption to judge). MARGIN is how far the
# hold-out gain may trail the dev gain before we call it overfitting.
EPS = 0.02
MARGIN = 0.05


@dataclass
class Verdict:
    overfit: bool
    dev_gain: float
    holdout_gain: float
    reasons: list[str]

    @property
    def label(self) -> str:
        return "OVERFIT_SUSPECTED" if self.overfit else "OK"


def assess(dev_gain: float, holdout_gain: float, *, eps: float = EPS,
           margin: float = MARGIN) -> Verdict:
    """Pure decision: did dev improve while the hold-out failed to keep up?"""
    reasons: list[str] = []
    overfit = False
    if dev_gain > eps and holdout_gain < dev_gain - margin:
        overfit = True
        reasons.append(
            f"dev improved {dev_gain:+.4f} but hold-out only {holdout_gain:+.4f} "
            f"(gap {dev_gain - holdout_gain:+.4f} exceeds margin {margin})")
        if holdout_gain < 0:
            reasons.append(f"hold-out generalization REGRESSED ({holdout_gain:+.4f}) while dev rose")
    elif dev_gain > eps:
        reasons.append(f"dev {dev_gain:+.4f} and hold-out {holdout_gain:+.4f} move together — healthy")
    else:
        reasons.append(f"no material dev gain ({dev_gain:+.4f}) — nothing to adopt/judge")
    return Verdict(overfit, round(dev_gain, 4), round(holdout_gain, 4), reasons)


# =========================== two-axis adoption gate (SOT-2447 / A2) ============================
# The single sealed hold-out axis let the #5 regression through (dev 0.87 / hold-out 0.98 proxy but
# real −0.1333): the hold-out is scored with the same synthetic phrasings the RAG was tuned against,
# so a change can climb it while failing on the real test100 wording. Adoption is now gated on TWO
# independent generalization axes — the sealed hold-out AND the real-style transcription bench
# (`scoring.realstyle`, test100-style phrasing) — and a change is BLOCKED unless BOTH keep up with the
# dev gain. valid30/dev is NOT an adoption axis (it is the burnt-out dev set, isolated here).

# A generalization axis that drops at least this far in absolute terms blocks adoption outright, even
# if dev did not move — a change that regresses transfer is never worth adopting.
ABS_REGRESSION_FLOOR = 0.05


@dataclass
class TwoAxisVerdict:
    block: bool
    dev_gain: float
    holdout_gain: float
    realstyle_gain: float
    reasons: list[str]

    @property
    def label(self) -> str:
        return "ADOPTION_BLOCKED" if self.block else "OK_TO_ADOPT"


def assess_two_axis(dev_gain: float, holdout_gain: float, realstyle_gain: float, *,
                    eps: float = EPS, margin: float = MARGIN,
                    floor: float = ABS_REGRESSION_FLOOR) -> TwoAxisVerdict:
    """Block adoption unless BOTH generalization axes keep up with dev.

    An axis fails (→ block) when it is overfit-suspected relative to the dev gain (``assess``) OR it
    regressed by more than ``floor`` in absolute terms. This is what catches #5: the sealed hold-out
    rose (passes single-axis) but the real-style axis fell, so the two-axis gate still blocks."""
    reasons: list[str] = []
    block = False
    for name, gain in (("sealed hold-out", holdout_gain), ("real-style bench", realstyle_gain)):
        v = assess(dev_gain, gain, eps=eps, margin=margin)
        if v.overfit:
            block = True
            reasons.append(f"{name}: overfit — {v.reasons[0]}")
        elif gain < -floor:
            block = True
            reasons.append(f"{name}: regressed {gain:+.4f} (below −{floor}) — a change that drops a "
                           "generalization axis is not adopted")
    if not block:
        reasons.append(f"both generalization axes kept up with dev (hold-out {holdout_gain:+.4f}, "
                       f"real-style {realstyle_gain:+.4f}) — safe to adopt")
    return TwoAxisVerdict(block, round(dev_gain, 4), round(holdout_gain, 4),
                          round(realstyle_gain, 4), reasons)


def adoption_axes(prev: dict, curr: dict) -> tuple[float, float, float]:
    """Return (dev_gain, holdout_gain, realstyle_gain) between two history records.

    valid30/dev is reported as ``dev_gain`` for context ONLY — it never enters the block decision, so
    a change that improves only the burnt-out valid30/dev slice cannot earn adoption (valid30 isolation)."""
    dev_gain = curr.get("dev_mean", 0.0) - prev.get("dev_mean", 0.0)
    holdout_gain = curr.get("holdout_mean", 0.0) - prev.get("holdout_mean", 0.0)
    realstyle_gain = curr.get("realstyle_mean", 0.0) - prev.get("realstyle_mean", 0.0)
    return dev_gain, holdout_gain, realstyle_gain


def assess_last_two_axes(history: list[dict]) -> TwoAxisVerdict | None:
    """Two-axis adoption verdict for the last two history records (None if <2 or no real-style axis)."""
    if len(history) < 2:
        return None
    prev, curr = history[-2], history[-1]
    if "realstyle_mean" not in curr or "realstyle_mean" not in prev:
        return None
    return assess_two_axis(*adoption_axes(prev, curr))


def load_history(path=HISTORY_PATH) -> list[dict]:
    if not path.exists():
        return []
    out: list[dict] = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                out.append(json.loads(line))
    return out


def assess_last_two(history: list[dict]) -> Verdict | None:
    if len(history) < 2:
        return None
    prev, curr = history[-2], history[-1]
    return assess(curr.get("dev_mean", 0.0) - prev.get("dev_mean", 0.0),
                  curr.get("holdout_mean", 0.0) - prev.get("holdout_mean", 0.0))


def per_archetype_flags(history: list[dict]) -> list[dict]:
    """Per-archetype overfit flags between the last two runs (only where both means are present)."""
    if len(history) < 2:
        return []
    prev, curr = history[-2], history[-1]
    pa, ca = prev.get("archetypes", {}), curr.get("archetypes", {})
    flags: list[dict] = []
    for arch in sorted(set(pa) | set(ca)):
        pd_, cd_ = pa.get(arch, {}), ca.get(arch, {})
        if None in (pd_.get("dev_mean"), cd_.get("dev_mean"),
                    pd_.get("holdout_mean"), cd_.get("holdout_mean")):
            continue
        v = assess(cd_["dev_mean"] - pd_["dev_mean"], cd_["holdout_mean"] - pd_["holdout_mean"])
        if v.overfit:
            flags.append({"archetype": arch, "dev_gain": v.dev_gain,
                          "holdout_gain": v.holdout_gain})
    return flags


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dev-gain", type=float, default=None)
    ap.add_argument("--holdout-gain", type=float, default=None)
    ap.add_argument("--realstyle-gain", type=float, default=None,
                    help="real-style bench gain — enables the two-axis adoption gate")
    args = ap.parse_args()

    # ---- ad-hoc mode ----
    if args.dev_gain is not None and args.holdout_gain is not None:
        if args.realstyle_gain is not None:
            return _report_two_axis(assess_two_axis(
                args.dev_gain, args.holdout_gain, args.realstyle_gain))
        v = assess(args.dev_gain, args.holdout_gain)
    else:
        history = load_history()
        # ---- two-axis gate when the real-style axis is recorded (the current adoption gate) ----
        tv = assess_last_two_axes(history)
        if tv is not None:
            return _report_two_axis(tv)
        v = assess_last_two(history)
        if v is None:
            print(f"insufficient history in {HISTORY_PATH} (need ≥2 runs) — cannot assess overfit.")
            return 2
        flags = per_archetype_flags(history)
        if flags:
            print("per-archetype overfit flags:")
            for fl in flags:
                print(f"  - {fl['archetype']:20} dev {fl['dev_gain']:+.4f} / "
                      f"hold-out {fl['holdout_gain']:+.4f}")

    print(f"\noverfit check: {v.label}")
    print(f"  dev gain      : {v.dev_gain:+.4f}")
    print(f"  hold-out gain : {v.holdout_gain:+.4f}")
    for r in v.reasons:
        print(f"  · {r}")
    if v.overfit:
        print("\n→ ADOPTION BLOCKED: the change did not generalize to the hold-out slice.")
        return 3
    print("\n→ OK to adopt (or no adoption to judge).")
    return 0


def _report_two_axis(tv: TwoAxisVerdict) -> int:
    print(f"\ntwo-axis adoption gate: {tv.label}")
    print(f"  dev gain        : {tv.dev_gain:+.4f}   (valid30/dev — context only, NOT an adoption axis)")
    print(f"  hold-out gain   : {tv.holdout_gain:+.4f}   (sealed-project axis)")
    print(f"  real-style gain : {tv.realstyle_gain:+.4f}   (test100-style transfer axis)")
    for r in tv.reasons:
        print(f"  · {r}")
    if tv.block:
        print("\n→ ADOPTION BLOCKED: a change must clear BOTH the sealed hold-out AND the real-style "
              "bench.")
        return 3
    print("\n→ OK to adopt: both generalization axes kept up with dev.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
