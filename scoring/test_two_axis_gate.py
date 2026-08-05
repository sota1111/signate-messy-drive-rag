"""Two-axis adoption gate + #4/#5 back-detection (SOT-2447 / A2). Offline, no LLM / network.

The raison d'être of the real-style bench: the single sealed-hold-out gate let the #5 regression
through (dev 0.87 / hold-out 0.98 proxy but real −0.1333). These tests prove the two-axis gate
(sealed hold-out × real-style bench) BLOCKS the historical #4 and #5 changes, while a single-axis
gate would have adopted #5.

    .venv/bin/python -m pytest scoring/test_two_axis_gate.py -q
"""
from __future__ import annotations

import json
from pathlib import Path

from config import settings
from scoring import overfit_check as OC

# --- historical adoption snapshots, sourced from scoring/ledger.jsonl + docs/status.md -----------
# dev/holdout are the local proxies recorded at each commit; realstyle is proxied by the real Public
# score (the real-style bench is BUILT to track the real test100 style — see scoring.realstyle).
#   #3 (0ce7f73): best floor, real −0.0333, local 0.10, no independent hold-out yet
#   #4 (28c0867): valid30 proxy +0.5833 but real −0.10                (1st regression)
#   #5 (3dad0fc): dev 0.87 / hold-out 0.98 proxy but real −0.1333     (2nd regression, hold-out passed)
SNAP_3 = {"dev_mean": 0.10, "holdout_mean": 0.10, "realstyle_mean": -0.0333, "label": "#3-floor"}
SNAP_4 = {"dev_mean": 0.5833, "holdout_mean": 0.5833, "realstyle_mean": -0.10, "label": "#4"}
SNAP_5 = {"dev_mean": 0.87, "holdout_mean": 0.98, "realstyle_mean": -0.1333, "label": "#5"}


def _real_scores() -> dict[str, float]:
    path = Path(settings.REPO_ROOT) / "scoring" / "ledger.jsonl"
    out = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            o = json.loads(line)
            out[o["submission"]] = o["real_public_score"]
    return out


# ------------------------------- back-test: #4 and #5 are blocked ------------------------------
def test_realstyle_axis_is_anchored_to_the_real_public_scores():
    """The snapshots' real-style values ARE the recorded real Public scores (honest linkage)."""
    real = _real_scores()
    assert SNAP_3["realstyle_mean"] == real["#3"]
    assert SNAP_4["realstyle_mean"] == real["#4"]
    assert SNAP_5["realstyle_mean"] == real["#5"]
    # both regressions really did drop below the #3 floor.
    assert real["#4"] < real["#3"] and real["#5"] < real["#3"]


def test_two_axis_blocks_state_4():
    v = OC.assess_two_axis(*OC.adoption_axes(SNAP_3, SNAP_4))
    assert v.block is True and v.label == "ADOPTION_BLOCKED"
    assert any("real-style" in r for r in v.reasons)


def test_two_axis_blocks_state_5_that_single_axis_lets_through():
    dev_gain, holdout_gain, realstyle_gain = OC.adoption_axes(SNAP_3, SNAP_5)
    # 1) the sealed hold-out ROSE (0.10 → 0.98): the single-axis gate does NOT block #5 — the gap.
    single = OC.assess(dev_gain, holdout_gain)
    assert single.overfit is False, "single sealed-hold-out axis would have adopted #5 (the bug)"
    # 2) the two-axis gate blocks #5 because the real-style axis regressed.
    two = OC.assess_two_axis(dev_gain, holdout_gain, realstyle_gain)
    assert two.block is True
    assert any("real-style" in r for r in two.reasons)


def test_two_axis_adopts_a_genuinely_transferring_change():
    """Positive control: both generalization axes rise with dev → adoption is allowed."""
    good_curr = {"dev_mean": 0.30, "holdout_mean": 0.28, "realstyle_mean": 0.20}
    v = OC.assess_two_axis(*OC.adoption_axes(SNAP_3, good_curr))
    assert v.block is False and v.label == "OK_TO_ADOPT"


# ------------------------------- valid30 isolation --------------------------------------------
def test_dev_only_improvement_is_not_adopted():
    """A change that lifts ONLY valid30/dev while both generalization axes stall is BLOCKED —
    valid30 can never grant adoption (it is not an adoption axis)."""
    dev_only = {"dev_mean": 0.90, "holdout_mean": 0.10, "realstyle_mean": -0.0333}  # axes unchanged
    dev_gain, holdout_gain, realstyle_gain = OC.adoption_axes(SNAP_3, dev_only)
    assert dev_gain > 0.5 and abs(holdout_gain) < 1e-9 and abs(realstyle_gain) < 1e-9
    v = OC.assess_two_axis(dev_gain, holdout_gain, realstyle_gain)
    assert v.block is True  # dev soared but neither generalization axis moved → overfit


def test_adoption_axes_excludes_dev_from_the_decision():
    """adoption_axes surfaces dev_gain for context, but flat axes with a huge dev gain still block."""
    dev_gain, holdout_gain, realstyle_gain = OC.adoption_axes(
        SNAP_3, {"dev_mean": 5.0, "holdout_mean": 0.10, "realstyle_mean": -0.0333})
    assert holdout_gain == 0.0 and realstyle_gain == 0.0
    assert OC.assess_two_axis(dev_gain, holdout_gain, realstyle_gain).block is True


# ------------------------------- history plumbing ---------------------------------------------
def test_assess_last_two_axes_uses_realstyle_when_present():
    history = [SNAP_3, SNAP_5]
    v = OC.assess_last_two_axes(history)
    assert v is not None and v.block is True


def test_assess_last_two_axes_absent_without_realstyle_axis():
    # records lacking realstyle_mean → no two-axis verdict (falls back to single-axis overfit_check)
    history = [{"dev_mean": 0.1, "holdout_mean": 0.1}, {"dev_mean": 0.3, "holdout_mean": 0.2}]
    assert OC.assess_last_two_axes(history) is None


def test_regression_floor_blocks_even_without_dev_gain():
    """A change that drops a generalization axis is blocked even when dev did not move."""
    v = OC.assess_two_axis(dev_gain=0.0, holdout_gain=0.0, realstyle_gain=-0.20)
    assert v.block is True
