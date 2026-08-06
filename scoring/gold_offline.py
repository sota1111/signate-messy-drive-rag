"""Gold-100 offline runner + match-rate report (SOT-2472).

Establishes the offline regression baseline for the Gemini answer path. ``--gen`` defaults to the
production **investigator** single pass (SOT-2490); the heavier 合議 chain (investigator →
heterogeneous verifier → third-judge tie-break; ``src.rag.run`` with ``gen='resolve'``) stays
available as an opt-in. It runs — or reads — the 100 test answers, matches them against the gold
answers (``artifacts/predictions_test_v3_final.csv``) with the SAME local CRAG judge the rest of
the scoring stack uses, and reports the match rate, abstain rate, per-archetype breakdown,
wrong / abstain lists and cost.

    # score an already-produced prediction set (a details.jsonl carries cost/method; a csv works too):
    python -m scoring.gold_offline --answers artifacts/predictions_test.details.jsonl

    # run the answer path over the 100 test questions first, then score it. ``--gen`` defaults to
    # ``investigator`` (SOT-2490: the production single pass); pass ``--gen resolve`` for the heavier
    # 合議 chain (investigator → verifier → tie-break) when higher precision is worth the cost:
    python -m scoring.gold_offline --run --workers 8               # investigator (default)
    python -m scoring.gold_offline --run --gen resolve --workers 8 # opt-in 合議

    # confirm the "wrong → abstain" behaviour by diffing a baseline set against the primary set:
    python -m scoring.gold_offline --answers artifacts/predictions_test.details.jsonl \
        --baseline-answers artifacts/predictions_test_opus.details.jsonl

Match semantics reuse ``scoring.crag`` (GPT-family codex judge by default, SOT-2457):

  * **match**   = the judge returns ``Perfect`` or ``Acceptable``
  * **abstain** = the answer is the :data:`config.settings.ABSTAIN` sentinel / empty, or the judge
    returns ``Missing`` ("わかりません", 空回答). Under the SIGNATE rubric an abstention scores 0,
    an ``Incorrect`` scores −1, so the pipeline is designed to *drop a would-be wrong answer to an
    abstention* — ``--baseline-answers`` quantifies exactly that conversion.
  * **wrong**   = the judge returns ``Incorrect``.

Note "該当なし" / "ありません" (a data-does-not-exist conclusion) is a REAL answer, not an
abstention, per the official rubric — only the explicit ABSTAIN sentinel / empty string counts as
an abstention here.
"""
from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Iterable

from config import settings
from src.rag.archetype import classify

# CRAG verdict buckets.
MATCH_VERDICTS = ("Perfect", "Acceptable")
ABSTAIN_VERDICT = "Missing"
WRONG_VERDICT = "Incorrect"

# Manual human baseline: the operator answered 26/30 on the real gate, so the offline runner must
# reproduce at least that match rate on the gold-100 set (acceptance criterion #1).
MANUAL_MATCH_RATE = 26 / 30

# A judge maps [(prediction, ground_truth), ...] -> [verdict, ...] in order. The default calls the
# shared CRAG judge; tests inject a deterministic stub so the report logic stays network-free.
Judge = Callable[[list[tuple[str, str]]], list[str]]


def default_judge(pairs: list[tuple[str, str]]) -> list[str]:
    """Judge pairs with the shared CRAG stack (deterministic pre-pass + codex batch)."""
    from scoring import crag

    _mean, results = crag.score_pairs(pairs)
    return [r["judged"] for r in results]


def load_gold(path: Path) -> dict[int, str]:
    """Load the headerless ``index,answer`` gold file into ``{index: gold_answer}``."""
    gold: dict[int, str] = {}
    with path.open(encoding="utf-8-sig", newline="") as fh:
        for row in csv.reader(fh):
            if not row:
                continue
            gold[int(row[0])] = row[1] if len(row) > 1 else ""
    return gold


def _questions_by_index() -> dict[int, str]:
    if not settings.QUESTIONS_TEST.exists():
        return {}
    with settings.QUESTIONS_TEST.open(encoding="utf-8-sig", newline="") as fh:
        return {int(r["index"]): str(r["question"]) for r in csv.DictReader(fh)}


def load_predictions(path: Path) -> dict[int, dict]:
    """Load a prediction set into ``{index: row}``.

    Accepts a ``.details.jsonl`` (one JSON object per line — keeps ``answer`` / ``question`` /
    ``cost_usd`` / ``method`` …) or a csv (``index,answer[,question]`` with or without a header,
    including the official headerless submission format)."""
    rows: dict[int, dict] = {}
    if path.suffix.lower() == ".jsonl":
        for line in path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            rows[int(rec["index"])] = rec
        return rows
    with path.open(encoding="utf-8-sig", newline="") as fh:
        raw = list(csv.reader(fh))
    if not raw:
        return rows
    header = {c.strip().lower() for c in raw[0]}
    if header & {"answer", "prediction", "question", "index"}:
        keys = [c.strip().lower() for c in raw[0]]
        for row in raw[1:]:
            rec = dict(zip(keys, row))
            rows[int(rec["index"])] = rec
    else:  # official headerless index,answer
        for row in raw:
            if not row:
                continue
            rows[int(row[0])] = {"index": row[0], "answer": row[1] if len(row) > 1 else ""}
    return rows


def _answer(row: dict) -> str:
    return str(row.get("answer", row.get("prediction", "")) or "").strip()


def _is_sentinel_abstain(answer: str) -> bool:
    return answer == "" or answer == settings.ABSTAIN


@dataclass
class Item:
    index: int
    question: str
    answer: str
    gold: str
    archetype: str
    verdict: str
    cost_usd: float = 0.0

    @property
    def is_match(self) -> bool:
        return self.verdict in MATCH_VERDICTS

    @property
    def is_abstain(self) -> bool:
        return self.verdict == ABSTAIN_VERDICT

    @property
    def is_wrong(self) -> bool:
        return self.verdict == WRONG_VERDICT


@dataclass
class Report:
    items: list[Item] = field(default_factory=list)
    baseline_conversion: dict | None = None

    @property
    def n(self) -> int:
        return len(self.items)

    def _rate(self, count: int) -> float:
        return count / self.n if self.n else 0.0

    @property
    def match_rate(self) -> float:
        """Gold match rate over scored items (0.0 when empty). Used by 関門3 gating (SOT-2477)."""
        return self._rate(sum(1 for it in self.items if it.is_match))

    def to_dict(self) -> dict:
        matches = [it for it in self.items if it.is_match]
        abstains = [it for it in self.items if it.is_abstain]
        wrongs = [it for it in self.items if it.is_wrong]
        verdicts: dict[str, int] = {}
        for it in self.items:
            verdicts[it.verdict] = verdicts.get(it.verdict, 0) + 1

        by_type: dict[str, dict] = {}
        for it in self.items:
            b = by_type.setdefault(
                it.archetype, {"n": 0, "match": 0, "abstain": 0, "wrong": 0})
            b["n"] += 1
            b["match"] += int(it.is_match)
            b["abstain"] += int(it.is_abstain)
            b["wrong"] += int(it.is_wrong)
        for b in by_type.values():
            b["match_rate"] = round(b["match"] / b["n"], 4) if b["n"] else 0.0

        non_match = len(abstains) + len(wrongs)
        match_rate = self._rate(len(matches))
        out = {
            "n": self.n,
            "match": {"count": len(matches), "rate": round(match_rate, 4)},
            "abstain": {"count": len(abstains), "rate": round(self._rate(len(abstains)), 4)},
            "wrong": {"count": len(wrongs), "rate": round(self._rate(len(wrongs)), 4)},
            "verdicts": verdicts,
            "by_type": dict(sorted(by_type.items())),
            # Among non-matches, how many were safely dropped to an abstention vs left Incorrect —
            # the "誤答を棄権へ落とす" health signal (acceptance criterion #2). Higher is safer.
            "wrong_to_abstain": {
                "non_match": non_match,
                "abstained": len(abstains),
                "incorrect": len(wrongs),
                "abstain_share": round(len(abstains) / non_match, 4) if non_match else 0.0,
            },
            "cost_usd": round(sum(it.cost_usd for it in self.items), 6),
            "baseline": {
                "manual_match_rate": round(MANUAL_MATCH_RATE, 4),
                "meets": match_rate >= MANUAL_MATCH_RATE,
            },
            "wrong_items": [
                {"index": it.index, "archetype": it.archetype, "question": it.question,
                 "answer": it.answer, "gold": it.gold}
                for it in sorted(wrongs, key=lambda x: x.index)
            ],
            "abstain_items": [
                {"index": it.index, "archetype": it.archetype, "question": it.question,
                 "gold": it.gold}
                for it in sorted(abstains, key=lambda x: x.index)
            ],
        }
        if self.baseline_conversion is not None:
            out["conversion"] = self.baseline_conversion
        return out

    def render(self) -> str:
        d = self.to_dict()
        lines = [
            f"gold-offline report  (n={d['n']})",
            f"  match   : {d['match']['count']:>3}  ({d['match']['rate']:.1%})",
            f"  abstain : {d['abstain']['count']:>3}  ({d['abstain']['rate']:.1%})",
            f"  wrong   : {d['wrong']['count']:>3}  ({d['wrong']['rate']:.1%})",
            f"  cost    : ${d['cost_usd']:.4f}",
            f"  baseline: manual {d['baseline']['manual_match_rate']:.1%} -> "
            f"{'MEETS' if d['baseline']['meets'] else 'BELOW'}",
            f"  drop-to-abstain: {d['wrong_to_abstain']['abstained']}/"
            f"{d['wrong_to_abstain']['non_match']} of non-matches abstained "
            f"({d['wrong_to_abstain']['abstain_share']:.1%}), "
            f"{d['wrong_to_abstain']['incorrect']} still incorrect",
            "  by type:",
        ]
        for arch, b in d["by_type"].items():
            lines.append(
                f"    {arch:<18} n={b['n']:>2}  match={b['match']:>2} "
                f"({b['match_rate']:.0%})  abstain={b['abstain']:>2}  wrong={b['wrong']:>2}")
        if "conversion" in d:
            c = d["conversion"]
            lines += [
                "  baseline -> primary conversion:",
                f"    wrong->abstain : {c['wrong_to_abstain']:>3}  (safe drop)",
                f"    wrong->match   : {c['wrong_to_match']:>3}  (fixed)",
                f"    abstain->match : {c['abstain_to_match']:>3}  (recovered)",
                f"    abstain->wrong : {c['abstain_to_wrong']:>3}  (regression!)",
                f"    match->wrong   : {c['match_to_wrong']:>3}  (regression!)",
                f"    match->abstain : {c['match_to_abstain']:>3}  (over-cautious)",
            ]
        return "\n".join(lines)


def _bucket(verdict: str) -> str:
    if verdict in MATCH_VERDICTS:
        return "match"
    if verdict == ABSTAIN_VERDICT:
        return "abstain"
    if verdict == WRONG_VERDICT:
        return "wrong"
    return "error"


def evaluate(predictions: dict[int, dict], gold: dict[int, str],
             questions: dict[int, str] | None = None,
             judge: Judge = default_judge,
             indices: Iterable[int] | None = None) -> Report:
    """Match ``predictions`` against ``gold`` and build a :class:`Report`.

    Only indices present in BOTH gold and predictions are scored (or the given ``indices`` subset).
    Sentinel abstentions (empty / ABSTAIN) are bucketed as ``Missing`` without spending a judge
    call; every other answer is sent to ``judge`` against its gold truth."""
    questions = questions or {}
    idxs = sorted(set(gold) & set(predictions))
    if indices is not None:
        wanted = set(indices)
        idxs = [i for i in idxs if i in wanted]

    # Pre-classify sentinel abstentions; judge only the rest (cost + determinism).
    verdicts: dict[int, str] = {}
    to_judge: list[int] = []
    for i in idxs:
        ans = _answer(predictions[i])
        if _is_sentinel_abstain(ans):
            verdicts[i] = ABSTAIN_VERDICT
        else:
            to_judge.append(i)
    if to_judge:
        pairs = [(_answer(predictions[i]), gold[i]) for i in to_judge]
        judged = judge(pairs)
        if len(judged) != len(to_judge):
            raise ValueError(
                f"judge returned {len(judged)} verdicts for {len(to_judge)} pairs")
        for i, v in zip(to_judge, judged):
            verdicts[i] = v

    items: list[Item] = []
    for i in idxs:
        row = predictions[i]
        q = str(row.get("question") or questions.get(i, ""))
        try:
            cost = float(row.get("cost_usd", 0.0) or 0.0)
        except (TypeError, ValueError):
            cost = 0.0
        items.append(Item(index=i, question=q, answer=_answer(row), gold=gold[i],
                          archetype=classify(q) if q else "unknown",
                          verdict=verdicts[i], cost_usd=cost))
    return Report(items=items)


def compare_conversion(baseline: Report, primary: Report) -> dict:
    """Count how each shared question's bucket moved from ``baseline`` to ``primary``.

    ``wrong->abstain`` is the intended safety behaviour (a would-be −1 becomes a 0); ``*->wrong``
    transitions are regressions. Only indices scored in both reports are compared."""
    b = {it.index: _bucket(it.verdict) for it in baseline.items}
    p = {it.index: _bucket(it.verdict) for it in primary.items}
    shared = sorted(set(b) & set(p))
    keys = ["wrong_to_abstain", "wrong_to_match", "abstain_to_match",
            "abstain_to_wrong", "match_to_wrong", "match_to_abstain",
            "wrong_to_wrong", "match_to_match", "abstain_to_abstain"]
    counts = {k: 0 for k in keys}
    for i in shared:
        key = f"{b[i]}_to_{p[i]}"
        if key in counts:
            counts[key] += 1
    counts["shared"] = len(shared)
    return counts


def run_pipeline(gen: str, workers: int, limit: int | None) -> Path:
    """Run the integrated answer chain over the test split and return its details.jsonl path."""
    from src.rag import run as runner

    out = settings.ARTIFACTS_DIR / f"predictions_test_{gen}.csv"
    runner.run("test", out, limit=limit, workers=workers, hard=False, gen=gen)
    return out.with_suffix(".details.jsonl")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--gold", type=Path,
                    default=settings.ARTIFACTS_DIR / "predictions_test_v3_final.csv",
                    help="gold answers (headerless index,answer)")
    ap.add_argument("--answers", type=Path,
                    default=settings.ARTIFACTS_DIR / "predictions_test.details.jsonl",
                    help="prediction set to score (.details.jsonl or .csv)")
    ap.add_argument("--baseline-answers", type=Path, default=None,
                    help="optional baseline set to diff against --answers (wrong->abstain check)")
    ap.add_argument("--run", action="store_true",
                    help="run the answer path over test first, then score its output")
    ap.add_argument("--gen", default="investigator",
                    help="answer backend when --run (default: investigator = production single "
                         "pass; use 'resolve' for the opt-in 合議 investigator→verifier→tie-break chain)")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--limit", type=int, default=None,
                    help="score only the first N gold indices (smoke)")
    ap.add_argument("--json", dest="as_json", action="store_true",
                    help="print the full JSON report instead of the human summary")
    ap.add_argument("--out", type=Path, default=None,
                    help="also write the JSON report to this path")
    args = ap.parse_args(argv)

    if args.run:
        args.answers = run_pipeline(args.gen, args.workers, args.limit)

    gold = load_gold(args.gold)
    questions = _questions_by_index()
    predictions = load_predictions(args.answers)
    indices = sorted(gold)[: args.limit] if args.limit else None
    report = evaluate(predictions, gold, questions, indices=indices)

    if args.baseline_answers:
        base_report = evaluate(load_predictions(args.baseline_answers), gold, questions,
                               indices=indices)
        report.baseline_conversion = compare_conversion(base_report, report)

    payload = report.to_dict()
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False, indent=2) if args.as_json else report.render())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
