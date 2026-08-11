"""SOT-2635 — offline τ selection from the focused-gate EU-gate probe telemetry.

Reads the probe focused-gate JSON (RAG_EU_GATE=1 at a permissive τ so nothing was flipped and every
committed answer carries its expected utility U in ``interventions.eu_gate.utility``) and, because a flip
is a pure function of (U, τ), computes the post-flip focused net at every candidate τ WITHOUT re-running:

    flipped(idx, τ)  = committed(idx) AND U(idx) ≤ τ         # 倒される to 棄権 → Missing (score 0)
    verdict(idx, τ)  = Missing if flipped else the probe verdict
    focused_net(τ)   = Σ_target score(verdict)               # P=+1 A=+0.5 M=0 I=−1
    sentinel_reg(τ)  = #sentinels that were MATCH but U ≤ τ  # a sentinel the gate would wrongly倒す

Picks the τ that maximises focused_net subject to sentinel_reg == 0 (never倒す a champion-correct answer).

    .venv/bin/python scripts/sot2635_analyze.py artifacts/focused_gate_sot2635_eu_taum9.json \
        [artifacts/focused_gate_sot2634_baseline.json]
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

MATCH = {"Perfect", "Acceptable"}
SCORE = {"Perfect": 1.0, "Acceptable": 0.5, "Missing": 0.0, "Incorrect": -1.0}


def _u(row: dict):
    iv = (row.get("interventions") or {}).get("eu_gate") or {}
    return iv.get("utility"), iv


def _score(v: str) -> float:
    return SCORE.get(v, 0.0)


def main(argv: list[str]) -> int:
    probe = json.loads(Path(argv[1]).read_text(encoding="utf-8"))
    baseline = json.loads(Path(argv[2]).read_text(encoding="utf-8")) if len(argv) > 2 else None

    trows = probe["target"]["rows"]
    srows = probe["sentinel"]["rows"]

    print(f"== probe {probe['label']} official={probe['official']} gate={probe['gate']} ==")
    base_v = {}
    if baseline:
        base_v = {r["index"]: r["verdict"] for r in baseline["target"]["rows"]}

    print("\n-- target rows (idx | verdict | route | U | commit | flipped | baseline) --")
    committed = []  # (idx, U, verdict)
    for r in sorted(trows, key=lambda x: x["index"]):
        u, iv = _u(r)
        us = f"{u:+.3f}" if isinstance(u, (int, float)) else "  —  "
        bv = base_v.get(r["index"], "-")
        print(f" idx={r['index']:<3} {r['verdict']:<10} {r['route']:<13} U={us} "
              f"commit={iv.get('commit')} flip={iv.get('flipped')} base={bv}")
        if isinstance(u, (int, float)) and not iv.get("already_abstain", False):
            committed.append((r["index"], float(u), r["verdict"]))

    print("\n-- sentinel rows (idx | verdict | U | commit | already_abstain) --")
    sent_committed = []
    for r in sorted(srows, key=lambda x: x["index"]):
        u, iv = _u(r)
        us = f"{u:+.3f}" if isinstance(u, (int, float)) else "  —  "
        print(f" idx={r['index']:<3} {r['verdict']:<10} U={us} commit={iv.get('commit')} "
              f"already_abstain={iv.get('already_abstain')}")
        if isinstance(u, (int, float)) and not iv.get("already_abstain", False):
            sent_committed.append((r["index"], float(u), r["verdict"]))

    base_net = sum(_score(r["verdict"]) for r in trows)
    print(f"\nfocused target net @ probe (no flip): {base_net:+.2f}")

    # candidate τ = midpoints between sorted committed U (plus 0.0 default), evaluate net + sentinel reg.
    us = sorted({round(u, 4) for _, u, _ in committed} | {round(u, 4) for _, u, _ in sent_committed})
    cands = [0.0] + us + [round((a + b) / 2, 4) for a, b in zip(us, us[1:])]
    cands = sorted(set(cands))
    print("\n-- τ sweep (τ | focused_net | wrong_flipped | correct_flipped | sentinel_reg) --")
    best = None
    for tau in cands:
        net = 0.0
        wrong_flipped = correct_flipped = 0
        for idx, u, v in committed:
            if u <= tau:  # flipped → Missing
                net += 0.0
                if v == "Incorrect":
                    wrong_flipped += 1
                elif v in MATCH:
                    correct_flipped += 1
            else:
                net += _score(v)
        # Missing (non-committed) target idx contribute 0 either way.
        sent_reg = sum(1 for _, u, v in sent_committed if u <= tau and v in MATCH)
        flag = ""
        if sent_reg == 0 and (best is None or net > best[1]):
            best = (tau, net, wrong_flipped, correct_flipped)
            flag = "  <= best (sent_reg=0)"
        print(f" τ={tau:+.4f}  net={net:+.2f}  wrong_flip={wrong_flipped} "
              f"correct_flip={correct_flipped} sent_reg={sent_reg}{flag}")

    if best:
        print(f"\n==> recommended τ={best[0]:+.4f}: focused net {base_net:+.2f} → {best[1]:+.2f} "
              f"(wrong倒し {best[2]}, correct誤倒し {best[3]}, sentinel regression 0)")
    else:
        print("\n==> no τ improves focused net without a sentinel regression "
              "(U does not separate wrong from correct on this path).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
