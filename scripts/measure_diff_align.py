#!/usr/bin/env python3
"""SOT-2588 — version-diff block-alignment + edit-intent diagnostics over gold100.

The internal harness is a *故障箇所を分離する診断器*, not an LB predictor (local proxy ↔ real LB
ρ=-0.09). This script records the three SOT-2588 diagnostics for the ``version_diff`` archetype so a
pair-resolution failure is separated from an alignment failure, and both from a substantive-ranking
failure — instead of judging only the final-answer match.

Three metrics (all offline / network-free; the differ is a total function over the two Office files):

  * ``version_pair_accuracy``          — share of gold ``version_diff`` questions whose version *pair*
                                         resolves (via the filename rules OR, on the align lane, the
                                         SOT-2583 registry version-family fallback).
  * ``alignment_accuracy``             — of the resolved pairs, share that produce a non-empty aligned
                                         atomic-change list (block alignment + ADD/DELETE/MODIFY/MOVE).
  * ``substantive_change_precision``   — of the questions that render an answer, share whose *top*
                                         ranked candidate is classified SUBSTANTIVE (not boilerplate /
                                         layout / move) — i.e. the pipeline surfaced a real change first.

It also runs a focused idx1 / idx14 check (does the gold substantive change overlap the top candidate?),
and reports how many pairs the registry version-family fallback recovered over the filename rules alone.

Usage::

    RAG_DIFF_ALIGN=1 PYTHONPATH=. .venv/bin/python scripts/measure_diff_align.py

Writes ``artifacts/diff_align_diagnostics.json`` (machine) and prints a summary. Deterministic: reads the
gold100 review CSV for the ``version_diff`` questions + gold answers and re-runs only the deterministic
differ (no LLM, no re-generation).
"""
from __future__ import annotations

import csv
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from config import settings
from src.rag import diffpair

ARCHETYPE = "version_diff"


def _load_version_diff_rows() -> list[dict[str, str]]:
    """The gold ``version_diff`` questions (index / question / gold) from the last gold100 review CSV."""
    path = settings.ARTIFACTS_DIR / "gold_100_review.csv"
    if not path.exists():
        return []
    rows: list[dict[str, str]] = []
    with path.open(encoding="utf-8-sig", newline="") as fh:
        for r in csv.DictReader(fh):
            if str(r.get("archetype", "") or "").strip() != ARCHETYPE:
                continue
            rows.append({
                "index": str(r.get("index", "")),
                "question": str(r.get("question", "")),
                "gold": str(r.get("gold_v3", "") or r.get("gold", "")),
                "status": str(r.get("status", "")),
            })
    return rows


_TOKEN_RE = re.compile(r"[A-Za-z][A-Za-z0-9_]{2,}|[一-龥ぁ-んァ-ヶー]{2,}|\d[\d,.]*")


def _gold_overlap(gold: str, text: str) -> float:
    """Share of the gold answer's salient tokens present in ``text`` (rough content-recall proxy)."""
    g = {diffpair._norm(t) for t in _TOKEN_RE.findall(diffpair.nfc(gold))}
    g = {t for t in g if t}
    if not g:
        return 0.0
    hay = diffpair._norm(text)
    hit = sum(1 for t in g if t in hay)
    return round(hit / len(g), 3)


def main() -> None:
    rows = _load_version_diff_rows()
    per_question: list[dict] = []
    resolved = aligned = rendered = subst_top = 0
    recovered_by_registry = 0

    for row in rows:
        q = row["question"]
        filename_pair = diffpair.resolve_pair(q)
        pair = diffpair._resolve_pair_for_render(q)
        if pair is not None and filename_pair is None:
            recovered_by_registry += 1
        ranked = diffpair.rank_changes(pair) if pair is not None else None
        try:
            answer = diffpair.answer_question(q)
        except Exception:
            answer = None

        top = ranked[0] if ranked else None
        rec = {
            "index": row["index"],
            "gold_status": row["status"],
            "pair_resolved": pair is not None,
            "pair_basis": pair.basis if pair is not None else None,
            "registry_recovered": pair is not None and filename_pair is None,
            "aligned_changes": len(ranked) if ranked else 0,
            "top_intent": top.intent if top else None,
            "top_score": round(top.score, 3) if top else None,
            "rendered": (answer or "")[:200],
            "gold_overlap_top": _gold_overlap(row["gold"], top.change.render()) if top else 0.0,
            "gold_overlap_answer": _gold_overlap(row["gold"], answer or ""),
        }
        per_question.append(rec)
        if pair is not None:
            resolved += 1
        if ranked:
            aligned += 1
        if answer:
            rendered += 1
            if top and top.intent == diffpair.SUBSTANTIVE:
                subst_top += 1

    n = len(rows)
    result = {
        "lane_enabled_env": diffpair.align_enabled(),
        "n_version_diff": n,
        "version_pair_accuracy": round(resolved / n, 4) if n else 0.0,
        "alignment_accuracy": round(aligned / resolved, 4) if resolved else 0.0,
        "substantive_change_precision": round(subst_top / rendered, 4) if rendered else 0.0,
        "registry_family_recovered_pairs": recovered_by_registry,
        "per_question": per_question,
    }
    out = settings.ARTIFACTS_DIR / "diff_align_diagnostics.json"
    out.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    print(f"lane_enabled_env:              {result['lane_enabled_env']}")
    print(f"n_version_diff:                {n}")
    print(f"version_pair_accuracy:         {result['version_pair_accuracy']}")
    print(f"alignment_accuracy:            {result['alignment_accuracy']}")
    print(f"substantive_change_precision:  {result['substantive_change_precision']}")
    print(f"registry_family_recovered:     {recovered_by_registry}")
    for rec in per_question:
        print(f"  idx{rec['index']:>3} status={rec['gold_status']:<8} basis={str(rec['pair_basis']):<16} "
              f"top={str(rec['top_intent']):<12} gold_overlap(top/ans)={rec['gold_overlap_top']}/{rec['gold_overlap_answer']}")
    print(f"-> {out}")


if __name__ == "__main__":
    main()
