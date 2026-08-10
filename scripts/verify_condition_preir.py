#!/usr/bin/env python3
"""SOT-2621 — focused offline verification of pre-loop BranchCondition IR construction.

The phase-0 diagnostics (``docs/ai/budget32_trace_classification.md``) showed the NUMERIC
``BUDGET_EXHAUSTED`` losses concentrate on **what-if / 条件分岐型** derivations whose branch structure is
never made explicit before the loop (idx76: 18 ターン中 17 回 search で operand ゼロ). This script measures —
**offline, network-free, no LLM** — whether the pre-loop condition prefill (``RAG_CONDITION_PREIR``)
actually builds a correct branch skeleton for the what-if questions **and does not over-fire** on the
ordinary derived questions in the focused set (wrong 非増加 の担保).

For each focused question it:
  1. classifies the route (only NUMERIC gets a condition IR),
  2. runs the deterministic detector (:func:`condition_prefill.detect`),
  3. materializes the skeleton through the PoT lane receiver (``ConditionIR.from_spec``) and reports the
     branch signature — the same ``branch interpretation`` axis the PoT N-sample majority agrees on.

The headline metrics are ``whatif_fire_rate`` (share of the *genuine* what-if subset that now arrives at
the loop with a pre-built branch skeleton) and ``overfire`` (any non-condition question that fired — must
be 0). This is the offline proxy for the "分岐解釈の改善・wrong非増加" acceptance check; the full three-layer
accuracy still needs a live ``RAG_POT_HARD_LANE=1`` trace.

Usage::

    RAG_CONDITION_PREIR=1 .venv/bin/python scripts/verify_condition_preir.py

Writes ``artifacts/condition_preir_focused.json`` and prints a per-question + summary table.
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

os.environ.setdefault("RAG_CONDITION_PREIR", "1")

from src.rag.agent import condition_prefill as cp  # noqa: E402
from src.rag.agent import pot_lane as pl  # noqa: E402
from src.rag.agent import query_router as qr  # noqa: E402

# The focused set from the issue (検証内容: idx 76, 47, 57, 6, 27). ``whatif`` marks the questions that
# genuinely contain a what-if / 条件分岐 the pre-loop IR is meant to fire on; the rest are ordinary derived
# questions that must NOT fire (else the branch skeleton would inject noise = wrong risk).
FOCUSED = [
    (76, True, "AOMINEの契約条件において、契約単価が現状よりも2,000円高く、実績工数が11.2時間少なかった場合、"
               "税込請求金額は、実際の税込請求金額と比べていくら変動しますか。"),
    (47, False, "青嶺不動産アセットマネジメントのtrain.xlsxにおいて、黄色ハイライトセルは予測と実際の誤差を"
                "計算していますが、その予測値の対象となっている不動産の建設年を算出してください。"),
    (57, False, "青葉のTXにて算出された回帰係数を用いて全データの予測値を計算し、正解データに対する F1 スコアが"
                "最大となるように閾値を設定したときの F1 スコアを答えてください。小数第5位まで求めてください。"),
    (6, False, "蒼泉会 ひがし丘総合病院案件において、提案時の税込み見込み金額と最終請求金額の差額は"
               "いくらですか。"),
    (27, False, "恒一会 かえで総合病院の提案書において、スコープ対象外としている項目はいくつありますか。"),
]


def main() -> int:
    rows = []
    for idx, whatif, q in FOCUSED:
        dec = qr.classify_route(q)
        ir = cp.build_condition_ir(q) if dec.route == qr.NUMERIC else None
        fired = ir is not None
        branch_sig = ""
        adjustments = []
        if fired:
            cond = pl.ConditionIR.from_spec(ir)
            branch_sig = cond.branch_signature()
            adjustments = [
                {"kind": a["kind"], "operand_hint": a.get("operand_hint", ""),
                 "delta": a.get("delta", ""), "rate": a.get("rate", 0), "order": a["order"]}
                for a in ir["adjustments"]
            ]
        rows.append({
            "index": idx, "route": dec.route, "expected_whatif": whatif, "fired": fired,
            "condition_type": ir["condition_type"] if fired else None,
            "predicate": ir["predicate"] if fired else "",
            "adjustments": adjustments, "branch_signature": branch_sig,
        })

    whatif_total = sum(1 for r in rows if r["expected_whatif"])
    whatif_fired = sum(1 for r in rows if r["expected_whatif"] and r["fired"])
    overfire = [r["index"] for r in rows if (not r["expected_whatif"]) and r["fired"]]
    summary = {
        "whatif_total": whatif_total,
        "whatif_fired": whatif_fired,
        "whatif_fire_rate": (whatif_fired / whatif_total) if whatif_total else 0.0,
        "overfire_indices": overfire,
        "overfire": len(overfire),
        "pass": whatif_fired == whatif_total and not overfire,
    }

    print(f"{'idx':>4} {'route':<9} {'whatif':<7} {'fired':<6} {'type':<18} adjustments")
    for r in rows:
        print(f"{r['index']:>4} {r['route']:<9} {str(r['expected_whatif']):<7} {str(r['fired']):<6} "
              f"{str(r['condition_type'] or ''):<18} "
              + "; ".join(f"{a['operand_hint']}{a['delta'] or ('@'+str(a['rate']))}" for a in r["adjustments"]))
    print("\nsummary:", json.dumps(summary, ensure_ascii=False))

    out = ROOT / "artifacts" / "condition_preir_focused.json"
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps({"rows": rows, "summary": summary}, ensure_ascii=False, indent=2))
    print("wrote", out)
    return 0 if summary["pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
