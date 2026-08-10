"""SOT-2587 — offline diagnostic metrics for the symbolic ENUM scan lane over gold-100 enum_set.

The ``enum_set`` archetype scores MATCH 2 / 9 on gold-100. Live A/B needs a Gemini answer pass; this
runner instead measures the *deterministic* lane properties the SOT-2587 verification asks to record —
independent of the model — so the mechanism can be evaluated offline and reproducibly:

    * ``universe_resolution_accuracy``  — did the registry-resolved *applicable* universe contain the
                                          gold target document(s)? (an enumeration cannot be complete if
                                          its target file is out of scope)
    * ``exhaustive_scan_completion``    — mean documents_scanned / documents_applicable (coverage the scan
                                          could actually achieve; the rest is unsupported = encrypted /
                                          image-only)
    * ``set_precision`` / ``set_recall`` — the scan's matched element set vs. the gold enumeration, using
                                          the same element-containment normalization as the gold set judge.

Gold enumeration answers are read at run time from ``artifacts/gold_100_review.csv`` (``gold_v3``); the
per-question **target-document labels** (document identities, not answers) are the small fixture below —
mirroring :func:`src.rag.index.document_registry.measure_recall`'s ``(question, project, target)`` cases.

    .venv/bin/python -m scoring.enum_diagnostics                     # writes artifacts/enum_scan_gold100.json
    .venv/bin/python -m scoring.enum_diagnostics --out /tmp/x.json   # custom output path
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

from config import settings
from src.rag.agent import enum_scan as es
from src.rag.agent import enumeration as _enum

ENUM_INDICES = (19, 26, 32, 38, 44, 45, 67, 73, 87)

# Per-question fixture: the target-document universe (registry doc_id substrings that must resolve into
# the applicable scope), plus the population kind / optional predicate the enumeration scans. Document
# identities only — the gold *answers* are loaded from the gold CSV at run time (no answer embedded here).
_P = "プロジェクト"
_AO = f"{_P}/株式会社青嶺不動産アセットマネジメント"
_KY = f"{_P}/京橋信用ソリューションズ株式会社"
_KO = f"{_P}/医療法人社団 恒一会 かえで総合病院"

FIXTURE: dict[int, dict[str, Any]] = {
    19: {"kind": _enum.TASK,
         "universe": (f"{_AO}/02.計画/スケジュール_r2.xlsx",)},
    26: {"kind": _enum.PROJECT, "universe": ()},               # cross-corpus 契約期間 aggregation
    32: {"kind": _enum.GENERIC, "project": "青嶺",
         "universe": (f"{_AO}/04.分析/analysis_outputs/metrics.json",
                      f"{_AO}/04.分析/analysis_project/src/modeling.py")},
    38: {"kind": _enum.PROJECT, "universe": ()},               # cross-corpus APR-M3 + sum
    44: {"kind": _enum.SEAT, "universe": ("社内管理/座席表.pptx",)},
    45: {"kind": _enum.TASK,
         "universe": (f"{_KY}/05.会議/会議録/会議録_2025-10-29.pdf",
                      f"{_KY}/05.会議/会議録/会議録_2025-11-11.pdf")},
    67: {"kind": _enum.PROJECT, "universe": ()},               # cross-corpus APR-M2 proposal≠FR
    73: {"kind": _enum.GENERIC, "project": "恒一会",
         "universe": (f"{_KO}/00.提案/提案書.pptx",)},
    87: {"kind": _enum.PROJECT, "universe": ()},               # cross-corpus APR-M1 + sample≥10000
}


def _load_gold_enum() -> dict[int, dict[str, str]]:
    """Load ``{index: {question, gold}}`` for the enum_set rows from the gold review CSV."""
    path = settings.ARTIFACTS_DIR / "gold_100_review.csv"
    out: dict[int, dict[str, str]] = {}
    if not path.exists():
        return out
    with path.open(encoding="utf-8-sig", newline="") as fh:
        for row in csv.DictReader(fh):
            try:
                idx = int(row["index"])
            except (KeyError, ValueError):
                continue
            if idx in ENUM_INDICES:
                out[idx] = {"question": row.get("question", ""),
                            "gold": row.get("gold_v3", "")}
    return out


def build_cases() -> list[es.EnumCase]:
    """Join the gold review rows with the target-document fixture into labelled EnumCases."""
    gold = _load_gold_enum()
    cases: list[es.EnumCase] = []
    for idx in ENUM_INDICES:
        g = gold.get(idx, {})
        fx = FIXTURE.get(idx, {})
        cases.append(es.EnumCase(
            index=idx,
            question=g.get("question", ""),
            project=fx.get("project"),
            gold_universe=tuple(fx.get("universe", ())),
            gold_set=tuple(t for t in (g.get("gold", "") or "").replace("／", "、").split("、") if t.strip()),
            predicate=fx.get("predicate"),
            entry_types=tuple(fx.get("entry_types", ())),
            population_kind=fx.get("kind"),
        ))
    return cases


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", type=Path,
                    default=settings.ARTIFACTS_DIR / "enum_scan_gold100.json")
    args = ap.parse_args(argv)

    cases = build_cases()
    report = es.measure_enum(cases)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"enum_scan gold-100 diagnostics (n={report['n_cases']}, "
          f"universe-graded={report['graded_universe_cases']}, set-graded={report['graded_set_cases']})")
    print(f"  universe_resolution_accuracy : {report['universe_resolution_accuracy']}")
    print(f"  exhaustive_scan_completion   : {report['exhaustive_scan_completion']}")
    print(f"  set_precision / set_recall   : {report['set_precision']} / {report['set_recall']}")
    print(f"  -> {args.out}")
    for d in report["detail"]:
        print(f"    idx{d['index']:>2}  scope={d['scope']:<13} "
              f"univ_ok={d['universe_resolution_ok']}  "
              f"scan={d['universe']['documents_scanned']}/{d['universe']['documents_applicable']} "
              f"complete={d['complete']}  matched={d['matched_count']}  "
              f"tp/fp/fn={d['set_tp']}/{d['set_fp']}/{d['set_fn']}  {d['state_code']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
