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
    """Judge pairs with the deterministic pre-pass, then a stable 3-vote majority."""
    from scoring import crag

    _mean, results = crag.score_pairs(pairs, votes=3, majority=True)
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
    # SOT-2629 — per-question intervention/guard firing telemetry copied from the prediction row's
    # ``interventions`` field (empty when the row predates the telemetry or no flag was active). Enables the
    # 介入発火×verdict cross-tab below without re-running anything.
    interventions: dict = field(default_factory=dict)

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
    # SOT-2497: per-index abstain diagnosis joined from SOT-2492's abstain ledger by question text.
    # ``{index: {"state_code", "kind", "obligation"}}`` for abstains found in the ledger; empty when
    # no ledger exists (fail-open) or for abstains never routed through the investigator.
    abstain_codes: dict[int, dict] = field(default_factory=dict)

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
            # SOT-2497: 原因別の棄権集計（メトリクスのみ・設問/正答を含まない）。改善Issueの効果を
            # 「棄権総数」でなく NOT_RETRIEVED / EVIDENCE_INCOMPLETE / BUDGET_EXHAUSTED … の増減で測れる。
            "abstain_state_codes": _abstain_state_block(self.items, self.abstain_codes),
            # SOT-2629 — 介入発火×verdict クロス集計. Per active intervention flag, how often it FIRED across
            # match / abstain / wrong items — the metric cycle2's report could not compute because the
            # firings lived only in the abstain ledger (adversarial-review hole H6). Empty when no prediction
            # row carried intervention telemetry (fail-open on legacy details.jsonl).
            "interventions": _intervention_block(self.items),
            # SOT-2660 — fallback依存率: among CORRECT answers, the share that needed a 生ファイル系 tool
            # (raw_file_access.used). The KPI to drive toward 0 as the DB経路 (precomputed stores) widens —
            # NOT forced to 0 (raw-file fallback is a measured safety net). Also reports the DB_ONLY diagnostic
            # coverage (matches with zero raw access) and the not-yet-DB-ized idx list. Empty when no row
            # carried raw_file_access telemetry (legacy details.jsonl → fail-open).
            "fallback_dependency": _fallback_dependency_block(self.items),
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
        asc = d["abstain_state_codes"]
        lines.append(
            f"  abstain by state-code: coded={asc['coded']}/{asc['n_abstain']} "
            f"(uncoded={asc['uncoded']})")
        for code, cnt in asc["by_code"].items():
            if cnt:
                lines.append(f"    {code:<20} {cnt:>3}")
        iv = d.get("interventions") or {}
        if iv:
            lines.append("  interventions (fired / on)  by verdict:")
            for key, entry in iv.items():
                t = entry["total"]
                lines.append(
                    f"    {key:<18} on={t['on']:>2} fired={t['fired']:>2} "
                    f"({t['fire_rate']:.0%})  "
                    f"match={entry['match']['fired']}/{entry['match']['on']} "
                    f"abstain={entry['abstain']['fired']}/{entry['abstain']['on']} "
                    f"wrong={entry['wrong']['fired']}/{entry['wrong']['on']}")
        fb = d.get("fallback_dependency") or {}
        if fb.get("measured"):
            tag = " [DB_ONLY diagnostic]" if fb.get("db_only") else ""
            lines.append(
                f"  fallback依存率{tag}: {fb['match_raw_dependent']}/{fb['match_total']} correct "
                f"needed raw files ({fb['fallback_rate']:.1%}); "
                f"DB-only coverage={fb['db_only_coverage']}"
                + (f", blocked={fb['blocked_total']}" if fb.get("db_only") else ""))
            if fb.get("raw_dependent_match_idx"):
                lines.append(
                    f"    DB化未達idx (correct-but-raw-dependent): {fb['raw_dependent_match_idx']}")
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


# --------------------------------------------------------------------------- SOT-2497: abstain codes
# Join SOT-2492's abstain ledger (``artifacts/abstain_ledger.jsonl`` — one coded record per abstain,
# keyed by the question text) onto the gold-100 items so a run's 棄権 can be split by *cause*
# (NOT_RETRIEVED / RETRIEVED_NOT_PARSED / … / BUDGET_EXHAUSTED) rather than collapsed into one number.
# The ledger is append-only, so the LAST record for a question wins (most recent run). No ledger /
# no record for a question → that abstain is "uncoded" (fail-open; never raises).

def load_abstain_ledger(path: Path | None = None) -> dict[str, dict]:
    """Load the abstain ledger into ``{question: latest_record}`` (last line for a question wins).

    ``path`` defaults to :func:`src.rag.agent.abstain_ledger.default_path`
    (``artifacts/abstain_ledger.jsonl``). A missing file returns ``{}``; malformed lines are skipped —
    diagnostics must never break scoring."""
    if path is None:
        from src.rag.agent import abstain_ledger as _al
        path = _al.default_path()
    path = Path(path)
    out: dict[str, dict] = {}
    if not path.exists():
        return out
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            rec = json.loads(line)
        except json.JSONDecodeError:
            continue
        q = str(rec.get("question", "") or "").strip()
        if q:
            out[q] = rec
    return out


def _obligation_summary(rec: dict) -> dict:
    """Extract the concise ``{state_code, kind, obligation}`` from a ledger record.

    ``obligation`` uses the structured ``missing[0].detail`` (the canonical per-code template — short,
    no model free-text), not the verbose ``evidence_obligation``, so the review表 stays a summary."""
    code = str(rec.get("state_code", "") or "")
    missing = rec.get("missing") or []
    kind = detail = ""
    if isinstance(missing, (list, tuple)) and missing and isinstance(missing[0], dict):
        kind = str(missing[0].get("kind", "") or "")
        detail = str(missing[0].get("detail", "") or "")
    if not detail:
        detail = str(rec.get("evidence_obligation", "") or "")
    return {"state_code": code, "kind": kind, "obligation": detail}


def resolve_abstain_codes(report: "Report", ledger_by_q: dict[str, dict]) -> dict[int, dict]:
    """Map each abstaining item's index to its ledger diagnosis (matched by question text)."""
    out: dict[int, dict] = {}
    for it in report.items:
        if not it.is_abstain:
            continue
        rec = ledger_by_q.get((it.question or "").strip())
        if rec is not None:
            out[it.index] = _obligation_summary(rec)
    return out


def attach_abstain_state_codes(report: "Report", path: Path | None = None) -> "Report":
    """Populate ``report.abstain_codes`` from the abstain ledger (best-effort; in place). Returns it."""
    report.abstain_codes = resolve_abstain_codes(report, load_abstain_ledger(path))
    return report


def _intervention_fired(val) -> bool:
    """Whether a recorded intervention actually *fired* for this question (SOT-2629).

    ``val`` is the details ``interventions[<flag>]`` value: an int firing count (spin_pivot /
    search_cap_hits), or a small dict (operand_prefill ``injected``, condition_preir ``built``, eu_gate
    ``enabled`` …). A flag key present with a 0/false value = the flag was ON but idle this question."""
    if isinstance(val, bool):
        return val
    if isinstance(val, (int, float)):
        return val > 0
    if isinstance(val, dict):
        # ``used`` first: raw_file_access (SOT-2660) fired iff the answer touched a raw file, regardless of
        # the always-present db_only/tools/blocked companions (a DB_ONLY run has db_only=True everywhere).
        for k in ("used", "injected", "built", "enabled", "hit", "hits", "fired"):
            if k in val:
                return bool(val[k])
        return any(bool(v) for v in val.values())
    return bool(val)


def _intervention_block(items: list[Item]) -> dict:
    """介入発火×verdict クロス集計 (SOT-2629).

    For every intervention flag that appears in any prediction row's ``interventions``, count — split by
    match / abstain / wrong verdict — how many items had the flag ON and how many of those actually FIRED,
    with the fire rate. This is the answered-case attribution the cycle2 report could not produce because
    the firings lived only in the abstain ledger (adversarial-review hole H6). Empty when no row carried
    intervention telemetry (legacy details.jsonl → fail-open)."""
    _VERDS = ("match", "abstain", "wrong")

    def _verdict_of(it: Item) -> str:
        return "match" if it.is_match else ("abstain" if it.is_abstain else "wrong")

    buckets: dict[str, dict[str, dict[str, int]]] = {}
    for it in items:
        vk = _verdict_of(it)
        for key, val in (it.interventions or {}).items():
            b = buckets.setdefault(key, {v: {"on": 0, "fired": 0} for v in _VERDS})
            b[vk]["on"] += 1
            b[vk]["fired"] += int(_intervention_fired(val))

    out: dict[str, dict] = {}
    for key in sorted(buckets):
        vb = buckets[key]
        entry: dict[str, dict] = {}
        for v in _VERDS:
            on, fired = vb[v]["on"], vb[v]["fired"]
            entry[v] = {"on": on, "fired": fired,
                        "fire_rate": round(fired / on, 4) if on else 0.0}
        total_on = sum(vb[v]["on"] for v in _VERDS)
        total_fired = sum(vb[v]["fired"] for v in _VERDS)
        entry["total"] = {"on": total_on, "fired": total_fired,
                          "fire_rate": round(total_fired / total_on, 4) if total_on else 0.0}
        out[key] = entry
    return out


def _fallback_dependency_block(items: list[Item]) -> dict:
    """fallback依存率 (生ファイル依存) — SOT-2660.

    From each row's ``interventions['raw_file_access']`` (``{used, tools, blocked, db_only}``), compute:

    * ``fallback_rate`` — share of CORRECT (match) answers that needed a 生ファイル系 tool. The KPI to push
      toward 0 by widening the DB経路; 0 is never forced (the fallback is a measured ~30-answer safety net).
    * ``db_only_coverage`` — matches solved with zero raw access (the true DB-path coverage).
    * ``raw_dependent_match_idx`` — the correct-answer indices that still needed raw files (the DB-ization
      backlog / next store-expansion priorities).
    * ``db_only`` — whether this run was a RAG_DB_ONLY diagnostic; ``blocked_total`` — refused raw calls.

    Empty (``{"measured": False}``) when no row carried the telemetry (legacy details.jsonl → fail-open)."""
    rows = [it for it in items if isinstance(it.interventions, dict)
            and isinstance(it.interventions.get("raw_file_access"), dict)]
    if not rows:
        return {"measured": False}

    def _rfa(it: Item) -> dict:
        return it.interventions["raw_file_access"]

    matches = [it for it in rows if it.is_match]
    match_raw = [it for it in matches if _rfa(it).get("used")]
    match_db_only = [it for it in matches if not _rfa(it).get("used")]
    any_db_only = any(_rfa(it).get("db_only") for it in rows)
    blocked_total = sum(sum((_rfa(it).get("blocked") or {}).values()) for it in rows)
    # cross-run raw-file tool usage tally (どのツールが依存の主因か)
    tool_tally: dict[str, int] = {}
    for it in rows:
        for name, cnt in (_rfa(it).get("tools") or {}).items():
            tool_tally[name] = tool_tally.get(name, 0) + int(cnt)
    return {
        "measured": True,
        "db_only": any_db_only,
        "scored_rows": len(rows),
        "match_total": len(matches),
        "match_raw_dependent": len(match_raw),
        "fallback_rate": round(len(match_raw) / len(matches), 4) if matches else 0.0,
        "db_only_coverage": len(match_db_only),
        "raw_dependent_match_idx": sorted(it.index for it in match_raw),
        "raw_file_tool_calls": dict(sorted(tool_tally.items(), key=lambda kv: (-kv[1], kv[0]))),
        "blocked_total": blocked_total,
    }


def _abstain_state_block(items: list[Item], abstain_codes: dict[int, dict]) -> dict:
    """Aggregate abstains by state code and by state-code × archetype — METRICS ONLY (no text).

    ``by_code`` always lists all registered codes (zeros included) so history diffs are stable and an
    improvement Issue can read a per-cause increase/decrease straight off the entry."""
    from src.rag.agent.abstain_ledger import STATE_CODES

    abstains = [it for it in items if it.is_abstain]
    by_code = {c: 0 for c in STATE_CODES}
    by_code_archetype: dict[str, dict[str, int]] = {}
    coded = 0
    for it in abstains:
        info = abstain_codes.get(it.index)
        code = info.get("state_code") if info else None
        if code in by_code:
            by_code[code] += 1
            coded += 1
            arch = by_code_archetype.setdefault(code, {})
            arch[it.archetype] = arch.get(it.archetype, 0) + 1
    return {
        "n_abstain": len(abstains),
        "coded": coded,
        "uncoded": len(abstains) - coded,
        "by_code": by_code,
        "by_code_archetype": {c: dict(sorted(v.items()))
                              for c, v in sorted(by_code_archetype.items())},
    }


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
        _iv = row.get("interventions")
        items.append(Item(index=i, question=q, answer=_answer(row), gold=gold[i],
                          archetype=classify(q) if q else "unknown",
                          verdict=verdicts[i], cost_usd=cost,
                          interventions=dict(_iv) if isinstance(_iv, dict) else {}))
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
    ap.add_argument("--no-review", action="store_true",
                    help="skip regenerating artifacts/gold_100_review.md/.csv + history (SOT-2491)")
    ap.add_argument("--official", dest="official", action="store_true", default=True,
                    help="mark this run official in the history jsonl (default; flash-3.6 stack)")
    ap.add_argument("--no-official", dest="official", action="store_false",
                    help="mark this run official:false — dev-only calibration/exploration runs "
                         "(e.g. Sonnet / claude-mcp) that must NOT back official non-regression "
                         "claims (SOT-2625/2628); keeps them machine-distinguishable in history")
    ap.add_argument("--abstain-ledger", type=Path, default=None,
                    help="abstain ledger to join for per-code棄権集計 (SOT-2497; default "
                         "artifacts/abstain_ledger.jsonl)")
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

    # SOT-2497: join SOT-2492's abstain ledger by question so the report/history/review carry
    # per-cause棄権集計. Best-effort — a missing ledger just leaves every abstain uncoded.
    attach_abstain_state_codes(report, args.abstain_ledger)

    payload = report.to_dict()
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    # SOT-2491: 実行毎に gold-100 レビュー表 (md/csv) を再生成し、履歴 jsonl へ追記する。
    # details.jsonl があれば reason/confidence を補完する。ベストエフォート（採点結果は既に確定）。
    if not args.no_review:
        try:
            from scoring import gold_review
            details = args.answers if args.answers.suffix.lower() == ".jsonl" else None
            gen = args.gen if args.run else "answers"
            entry = gold_review.generate(report, details=details, gen=gen,
                                         official=args.official)
            print(f"[gold_review] wrote {gold_review.DEFAULT_MD} / {gold_review.DEFAULT_CSV}; "
                  f"history += {entry['recordedAt']} (official={entry['official']})")
        except Exception as exc:  # pragma: no cover - レビュー生成失敗で採点を落とさない
            print(f"[gold_review] skipped ({exc!r})")

    print(json.dumps(payload, ensure_ascii=False, indent=2) if args.as_json else report.render())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
