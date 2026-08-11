#!/usr/bin/env python3
"""SOT-2638 — offline *model divergence* diagnostics across the two serve backends.

Sibling of the other ``scripts/measure_*.py`` offline harnesses (route accuracy, evidence-slot
completion, EU gate, PoT lane). Those measure one backend; **this one compares two** — the flash 公式
gold100 run and the Sonnet dev gold100 run — and emits the per-idx *verdict divergence* table plus the
convergence指標 that cycle4 ("Gemini でも Sonnet でも同じ正答率") lacked a standing meter for.

Why this exists (SOT-2602 cycle4 5/6, convergence KPI): the divergence list is *itself* the mutual-port
work-list — G1-G3 recovered 6 problems from an 11-problem divergence set. Measuring it every run means
"収束作業が自動供給される" — each backend's win becomes the other's TODO.

What it does (pure offline aggregation — **no LLM re-run, serve path 不変更**):
  1. Reconstruct each backend's per-idx verdict (MATCH / ABSTAIN / WRONG) from its frozen gold100 score
     JSON (``wrong_items`` → WRONG, ``abstain_items`` → ABSTAIN, everything else → MATCH). MATCH means
     verdict ∈ {Perfect, Acceptable}; 非MATCH means {Missing(abstain), Incorrect(wrong)}.
  2. A divergence idx is one where the two backends disagree on *match status*
     (``flash_is_match != sonnet_is_match``).
  3. Split the divergence set into the three actionable buckets the收束 loop needs:
       * **commit_precision** ("wrong 側乖離"): the *failing* backend committed a WRONG answer. This is
         what a Commit Gate should catch — a precision gap, not a reachability gap.
       * **reachability** ("到達性乖離"): the failing backend ABSTAINed while the other reached a MATCH.
         This is what a *trace port* fixes — the winning backend's procedure is portable.
       * **judge_noise** ("judge 揺らぎ疑い"): the divergence is not a real capability gap but a scorer
         inconsistency. See below — these are excluded from the port candidate lists.
  4. **Deterministic-direct verdict-agreement check.** A deterministic-direct answer (the flash details
     record carries a ``det_pipeline:*`` tool_call) is model-invariant *by construction* — both backends
     route the same contract to the same deterministic pipeline and get the same value. So on a
     deterministic-direct idx the two verdicts *should* agree; a disagreement means either the judgment
     system has a bug or the judge is noisy — this harness **warns**. Among those, an idx where *both*
     sides committed (MATCH/WRONG, neither ABSTAIN) but the verdicts still differ cannot be a real gap (a
     deterministic value can't be simultaneously right and wrong) → it is classified **judge_noise**. An
     idx with one ABSTAIN side is a genuine reachability difference on a deterministic contract, not
     noise. (This is exactly how idx74 — ``det_pipeline:version_diff``, flash WRONG vs Sonnet MATCH — is
     recovered as a judge false-positive without needing Sonnet's answer string, which is absent from the
     frozen artifacts; it matches the SOT-2630 dossier finding "Sonnet も実質 わかりません＝judge 偽陽性".)
  5. **Judge-noise / answer-string cross-check.** As an additional, string-based signal (issue item 3:
     "同一回答文字列で verdict が異なる"), when *both* backends' answer strings are recoverable the harness
     also flags: (a) normalized-equal answer with differing verdict → judge_noise; (b) the MATCH-side
     answer is an abstain phrase (わかりません / 不明 / 該当なし / …) or confidence 0.0 → judge_noise
     (a committed-looking MATCH that is really an abstain). Any idx classified judge_noise is removed from
     the mutual-port candidate lists (回答文字列一致×verdict不一致は移植候補にしない).

Inputs (all under ``artifacts/``; overridable via CLI):
  * ``gold100_sot2610_waveA.json``        — flash 公式 gold100 score (SOT-2610 champion)
  * ``gold100_sonnet_dev.json``           — Sonnet dev gold100 score (SOT-2628)
  * ``predictions_test_investigator.details.jsonl`` — flash per-idx details: ``contract`` / ``tool_calls``
    (``det_pipeline:*`` ⇒ deterministic-direct) / ``answer`` / ``confidence`` / ``question``
  * ``gold100_sonnet_dev_resume.jsonl``   — Sonnet answer cache (optional; per-question, used only to
    recover Sonnet answer strings for the string-based cross-check — never for verdicts)

Writes ``artifacts/model_divergence.json`` (machine) + ``artifacts/model_divergence.md`` (summary) and
prints a summary. Deterministic and network-free. Diagnostic only — touches nothing on the serve path.

Usage::

    .venv/bin/python scripts/measure_model_divergence.py
"""
from __future__ import annotations

import argparse
import ast
import json
import sys
import unicodedata
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ARTIFACTS = ROOT / "artifacts"

# Answer strings that mean "the backend actually abstained" even if the scorer counted it as a MATCH.
ABSTAIN_PHRASES = ("わかりません", "分かりません", "不明", "回答できません", "該当なし", "特定できません", "n/a", "na", "")


# --------------------------------------------------------------------------- loading / verdict rebuild
def _load_json(path: Path) -> dict:
    with path.open(encoding="utf-8") as fh:
        return json.load(fh)


def _norm(s: object) -> str:
    """NFKC + strip whitespace, for robust question / answer equality across the two runs."""
    return unicodedata.normalize("NFKC", str(s)).replace(" ", "").replace("　", "").strip()


def rebuild_verdicts(score: dict, n: int = 100) -> dict[int, str]:
    """Per-idx verdict from a gold100 score JSON. wrong_items → WRONG, abstain_items → ABSTAIN, else MATCH."""
    wrong = {int(it["index"]) for it in score.get("wrong_items", [])}
    abstain = {int(it["index"]) for it in score.get("abstain_items", [])}
    total = int(score.get("n", n))
    verdicts: dict[int, str] = {}
    for i in range(total):
        verdicts[i] = "WRONG" if i in wrong else ("ABSTAIN" if i in abstain else "MATCH")
    return verdicts


def items_index(score: dict) -> dict[int, dict]:
    """idx → {archetype, gold, answer, question} for the wrong+abstain items a score JSON lists."""
    out: dict[int, dict] = {}
    for it in score.get("wrong_items", []) + score.get("abstain_items", []):
        out[int(it["index"])] = it
    return out


def load_flash_details(path: Path) -> dict[int, dict]:
    if not path.exists():
        return {}
    out: dict[int, dict] = {}
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            out[int(r["index"])] = r
    return out


def load_sonnet_answers(path: Path) -> dict[str, dict]:
    """normalized-question → sonnet record (answer/confidence/contract) from the resume cache.

    The resume file is a per-question cache: each line has ``question`` + ``record`` (a stringified dict).
    Used only to recover Sonnet answer *strings* for the string-based judge-noise cross-check; verdicts
    always come from the score JSON.
    """
    if not path.exists():
        return {}
    out: dict[str, dict] = {}
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            rec = row.get("record")
            if isinstance(rec, str):
                try:
                    rec = ast.literal_eval(rec)
                except (ValueError, SyntaxError):
                    rec = None
            if not isinstance(rec, dict):
                continue
            q = row.get("question") or rec.get("question")
            if q is None:
                continue
            out[_norm(q)] = rec
    return out


# --------------------------------------------------------------------------- helpers
def is_deterministic_direct(flash_rec: dict | None) -> bool:
    """A flash details record whose tool_calls include a ``det_pipeline:*`` call answered deterministically."""
    if not flash_rec:
        return False
    return "det_pipeline" in str(flash_rec.get("tool_calls", ""))


def is_abstain_phrase(answer: object, confidence: object = None) -> bool:
    a = _norm(answer).lower()
    if a in {_norm(p).lower() for p in ABSTAIN_PHRASES}:
        return True
    if any(a.startswith(_norm(p).lower()) for p in ABSTAIN_PHRASES if p):
        return True
    try:
        if confidence is not None and float(confidence) == 0.0:
            return True
    except (TypeError, ValueError):
        pass
    return False


def flash_answer(idx: int, flash_details: dict, flash_items: dict) -> tuple[str | None, object]:
    """Flash answer string for the string-based judge-noise cross-check.

    Sourced ONLY from the frozen score JSON's committed(WRONG)/ABSTAIN items — never from the details
    file. The details file may be a *different* flash run than the score JSON (it is overwritten by every
    gold100 invocation, flash or Sonnet, sharing the same path), so its answer strings are not guaranteed
    to correspond to the SOT-2610 verdicts and must not be trusted for per-idx equality. A flash MATCH
    answer is therefore not recoverable in frozen form (score JSON lists only wrong/abstain items) — that
    is fine: the deterministic-direct rule, not string comparison, is what catches the judge_noise case.
    """
    it = flash_items.get(idx)
    if it is not None and it.get("answer") is not None:
        return str(it["answer"]), None
    return None, None


def sonnet_answer(idx: int, question: str | None, sonnet_resume: dict, sonnet_items: dict) -> tuple[str | None, object]:
    it = sonnet_items.get(idx)  # WRONG/ABSTAIN items carry the answer directly
    if it is not None and it.get("answer") is not None:
        return str(it["answer"]), None
    if question is not None:
        rec = sonnet_resume.get(_norm(question))
        if rec is not None and rec.get("answer") is not None:
            return str(rec["answer"]), rec.get("confidence")
    return None, None


# --------------------------------------------------------------------------- core
def build_report(flash_score, sonnet_score, flash_details, sonnet_resume) -> dict:
    fv = rebuild_verdicts(flash_score)
    sv = rebuild_verdicts(sonnet_score)
    flash_items = items_index(flash_score)
    sonnet_items = items_index(sonnet_score)
    n = min(len(fv), len(sv))

    def question_for(idx: int) -> str | None:
        r = flash_details.get(idx)
        if r is not None and r.get("question"):
            return str(r["question"])
        for src in (flash_items, sonnet_items):
            if idx in src and src[idx].get("question"):
                return str(src[idx]["question"])
        return None

    def contract_for(idx: int) -> str | None:
        r = flash_details.get(idx)
        if r is not None and r.get("contract"):
            return str(r["contract"])
        for src in (flash_items, sonnet_items):
            if idx in src and src[idx].get("archetype"):
                return str(src[idx]["archetype"])
        return None

    rows: list[dict] = []
    det_direct_idxs: list[int] = []
    det_disagreements: list[dict] = []

    for i in range(n):
        f, s = fv[i], sv[i]
        f_match, s_match = (f == "MATCH"), (s == "MATCH")
        det = is_deterministic_direct(flash_details.get(i))
        if det:
            det_direct_idxs.append(i)

        # deterministic-direct verdict-agreement warning (over ALL det-direct idx, divergent or not)
        if det and f != s:
            det_disagreements.append({"index": i, "flash_verdict": f, "sonnet_verdict": s,
                                      "both_committed": ("ABSTAIN" not in (f, s))})

        if f_match == s_match:
            continue  # not a divergence

        # ---- this idx is a divergence: classify direction + bucket + judge-noise
        direction = "flash->sonnet" if f_match else "sonnet->flash"  # winner -> loser (port target)
        winner, loser = (f, s) if f_match else (s, f)  # winner is always MATCH; loser is the failing verdict

        q = question_for(i)
        f_ans, f_conf = flash_answer(i, flash_details, flash_items)
        s_ans, s_conf = sonnet_answer(i, q, sonnet_resume, sonnet_items)
        match_ans, match_conf = (f_ans, f_conf) if f_match else (s_ans, s_conf)

        # judge-noise signals -----------------------------------------------------------------
        noise_reasons: list[str] = []
        # (4) deterministic-direct + both sides committed but verdict differs -> can't be a real gap
        if det and "ABSTAIN" not in (f, s):
            noise_reasons.append("det_direct_committed_verdict_split")
        # (5a) both answer strings recovered and normalized-equal, verdict differs
        if f_ans is not None and s_ans is not None and _norm(f_ans) == _norm(s_ans):
            noise_reasons.append("identical_answer_verdict_split")
        # (5b) the MATCH side's recovered answer is really an abstain phrase / conf 0.0
        if match_ans is not None and is_abstain_phrase(match_ans, match_conf):
            noise_reasons.append("match_side_is_abstain")

        if noise_reasons:
            bucket = "judge_noise"
        elif loser == "WRONG":
            bucket = "commit_precision"   # wrong 側乖離: Commit Gate territory
        else:
            bucket = "reachability"        # ABSTAIN↔MATCH: trace-port territory

        rows.append({
            "index": i,
            "question": q,
            "contract": contract_for(i),
            "flash_verdict": f,
            "sonnet_verdict": s,
            "deterministic_direct": det,
            "direction": direction,
            "bucket": bucket,
            "judge_noise_reasons": noise_reasons,
            "match_side_answer_recovered": match_ans is not None,
            "flash_answer": f_ans,
            "sonnet_answer": s_ans,
        })

    # ---- aggregates -----------------------------------------------------------------------------
    div_idxs = [r["index"] for r in rows]
    by_bucket: dict[str, list[int]] = {"commit_precision": [], "reachability": [], "judge_noise": []}
    for r in rows:
        by_bucket[r["bucket"]].append(r["index"])

    flash_to_sonnet = [r["index"] for r in rows if r["direction"] == "flash->sonnet"]
    sonnet_to_flash = [r["index"] for r in rows if r["direction"] == "sonnet->flash"]

    # mutual-port candidate lists exclude judge_noise
    port_to_sonnet = [r["index"] for r in rows
                      if r["direction"] == "flash->sonnet" and r["bucket"] != "judge_noise"]
    port_to_flash = [r["index"] for r in rows
                     if r["direction"] == "sonnet->flash" and r["bucket"] != "judge_noise"]

    det_verdict_check = {
        "n_deterministic_direct": len(det_direct_idxs),
        "deterministic_direct_idxs": det_direct_idxs,
        "disagreements": det_disagreements,
        "warning": (
            f"{len(det_disagreements)} deterministic-direct idx have disagreeing verdicts across backends "
            "(a deterministic value is model-invariant; disagreement ⇒ judgment-system bug or judge noise)"
            if det_disagreements else "all deterministic-direct idx agree across backends"
        ),
    }

    flash_counts = {v: sum(1 for x in fv.values() if x == v) for v in ("MATCH", "ABSTAIN", "WRONG")}
    sonnet_counts = {v: sum(1 for x in sv.values() if x == v) for v in ("MATCH", "ABSTAIN", "WRONG")}

    return {
        "n_questions": n,
        "flash_verdict_counts": flash_counts,
        "sonnet_verdict_counts": sonnet_counts,
        "divergence_total": len(div_idxs),
        "divergence_idxs": div_idxs,
        "by_direction": {
            "flash_match_sonnet_not": {"n": len(flash_to_sonnet), "idxs": flash_to_sonnet},
            "sonnet_match_flash_not": {"n": len(sonnet_to_flash), "idxs": sonnet_to_flash},
        },
        "by_bucket": {
            "commit_precision": {"n": len(by_bucket["commit_precision"]), "idxs": by_bucket["commit_precision"],
                                 "meaning": "wrong 側乖離 — failing backend committed WRONG; Commit Gate should catch"},
            "reachability": {"n": len(by_bucket["reachability"]), "idxs": by_bucket["reachability"],
                             "meaning": "到達性乖離 — failing backend ABSTAINed; trace port applies"},
            "judge_noise": {"n": len(by_bucket["judge_noise"]), "idxs": by_bucket["judge_noise"],
                            "meaning": "judge 揺らぎ疑い — excluded from port candidates"},
        },
        "port_candidates": {
            "to_sonnet": {"n": len(port_to_sonnet), "idxs": port_to_sonnet,
                          "meaning": "flash succeeded, Sonnet did not — port flash's win to Sonnet"},
            "to_flash": {"n": len(port_to_flash), "idxs": port_to_flash,
                         "meaning": "Sonnet succeeded, flash did not — port Sonnet's win to flash"},
        },
        "deterministic_direct_verdict_check": det_verdict_check,
        "divergence_table": rows,
    }


# --------------------------------------------------------------------------- markdown
def render_md(report: dict, sources: dict) -> str:
    L: list[str] = []
    L.append("# Model Divergence Diagnostics (SOT-2638)\n")
    L.append(f"- flash verdicts: {report['flash_verdict_counts']}")
    L.append(f"- sonnet verdicts: {report['sonnet_verdict_counts']}")
    L.append(f"- **divergence total: {report['divergence_total']}**")
    bd = report["by_direction"]
    L.append(f"  - flash MATCH → Sonnet 非MATCH: **{bd['flash_match_sonnet_not']['n']}** "
             f"{bd['flash_match_sonnet_not']['idxs']}")
    L.append(f"  - Sonnet MATCH → flash 非MATCH: **{bd['sonnet_match_flash_not']['n']}** "
             f"{bd['sonnet_match_flash_not']['idxs']}")
    bk = report["by_bucket"]
    L.append("")
    L.append("## Divergence buckets")
    for name in ("commit_precision", "reachability", "judge_noise"):
        b = bk[name]
        L.append(f"- **{name}** (n={b['n']}) {b['idxs']} — {b['meaning']}")
    dc = report["deterministic_direct_verdict_check"]
    L.append("")
    L.append("## Deterministic-direct verdict-agreement check")
    L.append(f"- deterministic-direct idx (n={dc['n_deterministic_direct']}): {dc['deterministic_direct_idxs']}")
    L.append(f"- {dc['warning']}")
    for d in dc["disagreements"]:
        note = "both committed → judge_noise" if d["both_committed"] else "one side ABSTAIN → reachability, not noise"
        L.append(f"  - idx{d['index']}: flash={d['flash_verdict']} sonnet={d['sonnet_verdict']} ({note})")
    pc = report["port_candidates"]
    L.append("")
    L.append("## Mutual-port candidate lists (judge_noise excluded)")
    L.append(f"- → Sonnet (flash won): n={pc['to_sonnet']['n']} {pc['to_sonnet']['idxs']}")
    L.append(f"- → flash (Sonnet won): n={pc['to_flash']['n']} {pc['to_flash']['idxs']}")
    L.append("")
    L.append("## Per-idx divergence table")
    L.append("| idx | contract | flash | sonnet | det | direction | bucket | judge_noise_reasons |")
    L.append("| --- | --- | --- | --- | --- | --- | --- | --- |")
    for r in report["divergence_table"]:
        L.append(f"| {r['index']} | {r['contract']} | {r['flash_verdict']} | {r['sonnet_verdict']} | "
                 f"{'Y' if r['deterministic_direct'] else ''} | {r['direction']} | {r['bucket']} | "
                 f"{','.join(r['judge_noise_reasons'])} |")
    L.append("")
    L.append("## Sources")
    for k, v in sources.items():
        L.append(f"- {k}: `{v}`")
    L.append("")
    L.append("_Offline aggregation only; no LLM re-run; serve path unchanged._")
    return "\n".join(L) + "\n"


# --------------------------------------------------------------------------- main
def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--flash-score", type=Path, default=ARTIFACTS / "gold100_sot2610_waveA.json")
    ap.add_argument("--sonnet-score", type=Path, default=ARTIFACTS / "gold100_sonnet_dev.json")
    ap.add_argument("--flash-details", type=Path, default=ARTIFACTS / "predictions_test_investigator.details.jsonl")
    ap.add_argument("--sonnet-resume", type=Path, default=ARTIFACTS / "gold100_sonnet_dev_resume.jsonl")
    ap.add_argument("--out", type=Path, default=ARTIFACTS / "model_divergence.json")
    ap.add_argument("--out-md", type=Path, default=ARTIFACTS / "model_divergence.md")
    args = ap.parse_args()

    for label, p in (("flash-score", args.flash_score), ("sonnet-score", args.sonnet_score)):
        if not p.exists():
            raise SystemExit(f"required {label} artifact not found: {p}")

    flash_score = _load_json(args.flash_score)
    sonnet_score = _load_json(args.sonnet_score)
    flash_details = load_flash_details(args.flash_details)
    sonnet_resume = load_sonnet_answers(args.sonnet_resume)

    report = build_report(flash_score, sonnet_score, flash_details, sonnet_resume)
    sources = {
        "flash_score": str(args.flash_score),
        "sonnet_score": str(args.sonnet_score),
        "flash_details": str(args.flash_details) if args.flash_details.exists() else None,
        "sonnet_resume": str(args.sonnet_resume) if args.sonnet_resume.exists() else None,
    }
    report["sources"] = sources
    report["note"] = (
        "Offline divergence aggregation across the flash 公式 and Sonnet dev gold100 score artifacts. "
        "Verdicts are reconstructed from each run's frozen score JSON; deterministic-direct answers are "
        "read from the flash details det_pipeline tool_calls. No LLM is re-run and no serve-path code is "
        "touched (diagnostic only)."
    )

    args.out.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    args.out_md.write_text(render_md(report, sources), encoding="utf-8")

    print(f"n_questions: {report['n_questions']}")
    print(f"flash: {report['flash_verdict_counts']}  sonnet: {report['sonnet_verdict_counts']}")
    print(f"divergence_total: {report['divergence_total']}  idxs: {report['divergence_idxs']}")
    bd = report["by_direction"]
    print(f"  flash->sonnet (flash MATCH, sonnet not): n={bd['flash_match_sonnet_not']['n']} "
          f"{bd['flash_match_sonnet_not']['idxs']}")
    print(f"  sonnet->flash (sonnet MATCH, flash not): n={bd['sonnet_match_flash_not']['n']} "
          f"{bd['sonnet_match_flash_not']['idxs']}")
    bk = report["by_bucket"]
    print(f"  buckets: commit_precision={bk['commit_precision']['n']}{bk['commit_precision']['idxs']} "
          f"reachability={bk['reachability']['n']}{bk['reachability']['idxs']} "
          f"judge_noise={bk['judge_noise']['n']}{bk['judge_noise']['idxs']}")
    dc = report["deterministic_direct_verdict_check"]
    print(f"  det-direct verdict check: {dc['warning']}")
    print(f"-> {args.out}")
    print(f"-> {args.out_md}")


if __name__ == "__main__":
    main()
