"""SOT-2623 — common focused verification gate with a MANDATORY existing-MATCH sentinel set.

Every child issue's focused verification MUST go through this runner. It runs the child's
``--target`` idx AND a fixed 10-question **sentinel set** of existing champion MATCH answers
(``scripts/focused_sentinels.json``) in the SAME production path, judges both with the SAME codex
judge (``scoring.gold_offline.default_judge``) against the SAME gold
(``artifacts/predictions_test_v3_final.csv``), and FAILS THE GATE (exit != 0) if any sentinel drops
from MATCH — i.e. it is the regression detector the cycle-2 focused sets never had (adversarial review
H1; see the header of ``scripts/focused_sentinels.json``).

WHY sentinels ARE the tripwire: a child's ``--target`` idx are its abstain-recovery candidates, which
may legitimately still be non-MATCH — so the gate does NOT fail on a target verdict. It fails ONLY on a
sentinel regression, because every sentinel is an answer champion already gets right, so a sentinel
dropping from MATCH means the change under test broke an existing-correct answer.

The runner touches NO serve-path code: it only *reads* the production ``investigator.answer_question``
(the same entry the gold_offline run uses) and writes ``artifacts/focused_gate_<label>.json`` plus one
``docs/ai/experiment_ledger.jsonl`` line with the target/sentinel breakdown separated. No gold values
are hardcoded — sentinels' expected verdict (MATCH) is champion's own recorded verdict, re-checked live.

Model guard (SOT-2625 — adversarial-review hole H2): an OFFICIAL non-regression verdict is only valid
on the exact production stack — provider=gemini + GEN_MODEL/GEN_MODEL_HARD/VISION_MODEL=gemini-3.6-flash
+ VERTEX_LOCATION=global (flash 3.6 is global-only; no pro tier there). If a LIVE run's resolved config
does not match, official mode is REFUSED (exit 3, no GATE verdict) because non-regression on Sonnet /
unit tests does NOT transfer to flash 3.6. ``--dev`` runs a NON-official Sonnet (claude-cli) exploration
instead: the result is stamped ``official:false`` in the JSON and the ledger and CANNOT be a
non-regression basis. Every runner-written ledger line carries ``model`` / ``provider`` / ``official``.

Usage (champion / production env is set by the CALLER, exactly like scripts/sot2610_gold100.sh):

    .venv/bin/python scripts/run_focused_gate.py --label myaxis --target "4,47,99"   # OFFICIAL (flash 3.6)
    .venv/bin/python scripts/run_focused_gate.py --label selftest            # sentinels only (self-consistency)
    LLM_PROVIDER=claude-cli .venv/bin/python scripts/run_focused_gate.py --label devprobe --dev  # DEV Sonnet

Testing / no-Gemini modes (bypass live generation, judge still real codex — fully-injected => no smoke):

    # dry-run FAIL: inject answers, corrupt one sentinel -> gate must exit != 0
    .venv/bin/python scripts/run_focused_gate.py --label dryrun \
        --inject-answers artifacts/focused_gate_selftest.answers.json --no-ledger

Exit codes: 0 = GATE PASS (no sentinel regression); 2 = GATE FAIL (>=1 sentinel non-MATCH);
3 = usage/precondition error (missing gold / sentinels / codex judge / OFFICIAL model-guard reject).
"""
from __future__ import annotations

import argparse
import datetime as _dt
import json
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import pandas as pd  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from config import settings  # noqa: E402
from scoring import gold_offline  # noqa: E402
from src.rag import llm  # noqa: E402
from src.rag.agent import investigator  # noqa: E402

MATCH = {"Perfect", "Acceptable"}
REPO = Path(__file__).resolve().parents[1]
DEFAULT_SENTINELS = REPO / "scripts" / "focused_sentinels.json"
DEFAULT_GOLD = settings.ARTIFACTS_DIR / "predictions_test_v3_final.csv"
LEDGER = REPO / "docs" / "ai" / "experiment_ledger.jsonl"

# --- Official model guard (SOT-2625) -------------------------------------------------------------
# The adversarial review of cycle-2 (SOT-2622, net28) found hole H2: focused "no-regression" was
# claimed on Sonnet (claude-cli) / unit tests, but non-regression on Sonnet DOES NOT transfer to the
# production model gemini-3.6-flash (focused-PASS idx 47/99/8 later regressed under flash 3.6). So an
# OFFICIAL non-regression verdict is only valid when produced by the exact production stack:
#   provider=gemini + GEN_MODEL/GEN_MODEL_HARD/VISION_MODEL=gemini-3.6-flash + VERTEX_LOCATION=global
# (gemini-3.6-flash is global-only; no pro tier on global, so HARD/VISION must also be flash 3.6 —
# see the model-policy memo of 2026-08-10). Any other configuration is DEV-only (Sonnet exploration)
# and is mechanically stamped official:false so it can never be cited as a non-regression basis.
OFFICIAL_PROVIDER = "gemini"
OFFICIAL_MODEL = "gemini-3.6-flash"
OFFICIAL_LOCATION = "global"
LEDGER_REQUIRED_FIELDS = ("model", "provider", "official")


def _model_env() -> dict:
    """Resolve the EFFECTIVE text/vision model configuration (env overrides win, exactly as the
    serve path resolves it at call time), so the guard and the ledger record what will really run —
    not the static settings defaults."""
    provider = (os.getenv("LLM_PROVIDER")
                or getattr(settings, "LLM_PROVIDER", OFFICIAL_PROVIDER)
                or OFFICIAL_PROVIDER).strip().lower()
    gen_model = (os.getenv("GEN_MODEL") or settings.GEN_MODEL or "").strip()
    hard_model = (os.getenv("GEN_MODEL_HARD") or settings.GEN_MODEL_HARD or "").strip()
    vision_model = (os.getenv("VISION_MODEL") or settings.VISION_MODEL or "").strip()
    location = (os.getenv("VERTEX_LOCATION") or settings.VERTEX_LOCATION or "").strip()
    info = {"provider": provider, "gen_model": gen_model, "hard_model": hard_model,
            "vision_model": vision_model, "location": location}
    # The single string recorded as ledger `model`: the production Gemini model for an official run,
    # else "<provider>:<text-model>" for a dev run (e.g. claude-cli:sonnet) so the record is honest.
    if provider == "claude-cli":
        info["model"] = f"claude-cli:{(os.getenv('CLAUDE_CLI_MODEL') or getattr(settings, 'CLAUDE_CLI_MODEL', 'sonnet') or 'sonnet').strip()}"
    else:
        info["model"] = gen_model or provider
    return info


def _official_guard_reasons(info: dict) -> list[str]:
    """Return the list of reasons the config is NOT an official flash-3.6 stack (empty == official)."""
    reasons: list[str] = []
    if info["provider"] != OFFICIAL_PROVIDER:
        reasons.append(f"LLM_PROVIDER={info['provider']!r} (official requires {OFFICIAL_PROVIDER!r})")
    if info["gen_model"] != OFFICIAL_MODEL:
        reasons.append(f"GEN_MODEL={info['gen_model']!r} (official requires {OFFICIAL_MODEL!r})")
    if info["hard_model"] != OFFICIAL_MODEL:
        reasons.append(f"GEN_MODEL_HARD={info['hard_model']!r} (official requires {OFFICIAL_MODEL!r})")
    if info["vision_model"] != OFFICIAL_MODEL:
        reasons.append(f"VISION_MODEL={info['vision_model']!r} (official requires {OFFICIAL_MODEL!r})")
    if info["location"] != OFFICIAL_LOCATION:
        reasons.append(f"VERTEX_LOCATION={info['location']!r} (official requires {OFFICIAL_LOCATION!r})")
    return reasons


def _smoke(info: dict) -> tuple[bool, str]:
    """One-shot 'reply OK' probe on the RESOLVED text path so a mis-resolved/unreachable model (e.g.
    flash 3.6 404 off `global`) is caught BEFORE spending the batch. Returns (ok, detail)."""
    try:
        resp = llm.generate("Reply with exactly: OK", max_output_tokens=8, temperature=0.0)
        return True, str(resp).strip()[:40]
    except Exception as exc:  # noqa: BLE001
        return False, f"{type(exc).__name__}: {exc}"


def _require_ledger_fields(entry: dict) -> None:
    """Refuse to write a ledger entry missing any required provenance field (SOT-2625 criterion 3)."""
    missing = [k for k in LEDGER_REQUIRED_FIELDS if entry.get(k) in (None, "")]
    if missing:
        raise SystemExit(f"[error] ledger entry missing required field(s) {missing}; refusing to write")


def _inject_timeout() -> None:
    """Rebuild the genai client with a bounded HTTP timeout (genai default None can hang); SOT-2568."""
    try:
        from google import genai
        from google.genai import types

        llm._client = genai.Client(
            vertexai=True, project=settings.GCP_PROJECT_ID, location=settings.VERTEX_LOCATION,
            http_options=types.HttpOptions(timeout=180_000))
    except Exception as exc:  # noqa: BLE001 — best-effort; a live run still works with the default client
        print(f"[warn] genai timeout injection skipped: {exc}", file=sys.stderr)


def _parse_indices(spec: str) -> list[int]:
    return [int(p) for p in (spec or "").replace(" ", "").split(",") if p.strip()]


def _load_sentinels(path: Path) -> list[dict]:
    data = json.loads(path.read_text(encoding="utf-8"))
    sents = data.get("sentinels")
    if not sents:
        raise SystemExit(f"[error] no 'sentinels' in {path}")
    return sents


def _route_of(rec: dict) -> str:
    """Observed answer path: 'deterministic' when the router answered without the LLM loop, else 'llm'
    ('injected' for a bypassed answer supplied via --inject-answers, which has no real route)."""
    if rec.get("stop_reason") == "injected":
        return "injected"
    if str(rec.get("model")) == "deterministic":
        return "deterministic"
    if int(rec.get("iterations", 0) or 0) == 0 and rec.get("stop_reason") not in {"model_error", "timeout"}:
        return "deterministic"
    return "llm"


def _run_one(idx: int, q: str, injected: dict[int, str] | None) -> tuple[int, dict]:
    if injected is not None and idx in injected:
        return idx, {"index": idx, "question": q, "answer": injected[idx],
                     "stop_reason": "injected", "model": "injected", "iterations": 0,
                     "tool_calls": [], "wall_s": 0.0}
    t0 = time.monotonic()
    try:
        rec = investigator.answer_question(q).to_dict()
    except Exception as exc:  # noqa: BLE001 — record, do not abort the batch
        rec = {"answer": settings.ABSTAIN, "stop_reason": "harness_error", "model": "error",
               "iterations": 0, "tool_calls": [], "error": f"{type(exc).__name__}: {exc}"}
    rec["index"] = idx
    rec["question"] = q
    rec["wall_s"] = round(time.monotonic() - t0, 1)
    return idx, rec


def _run(indices: list[int], qbyidx: dict[int, str], workers: int,
         injected: dict[int, str] | None) -> dict[int, dict]:
    results: dict[int, dict] = {}
    with ThreadPoolExecutor(max_workers=workers) as ex:
        futs = {ex.submit(_run_one, i, qbyidx[i], injected): i for i in indices}
        for n, fut in enumerate(as_completed(futs), 1):
            idx, rec = fut.result()
            results[idx] = rec
            print(f"[{n}/{len(indices)}] idx={idx} route={_route_of(rec)} "
                  f"stop={rec.get('stop_reason')} wall={rec.get('wall_s')}s "
                  f"ans={str(rec.get('answer'))[:56]}", flush=True)
    return results


def _judge(indices: list[int], recs: dict[int, dict], gold: dict[int, str]) -> dict[int, str]:
    """Verdict per index via the SAME codex judge gold_offline uses (sentinel abstains handled there)."""
    preds = {i: {"answer": recs[i].get("answer", ""), "question": recs[i].get("question", "")}
             for i in indices if i in gold}
    report = gold_offline.evaluate(preds, gold, indices=indices, judge=gold_offline.default_judge)
    return {it.index: it.verdict for it in report.items}


def _row(idx: int, recs: dict[int, dict], verdicts: dict[int, str]) -> dict:
    rec = recs.get(idx, {})
    verdict = verdicts.get(idx, "MISSING")
    return {"index": idx, "verdict": verdict, "is_match": verdict in MATCH,
            "route": _route_of(rec), "stop_reason": rec.get("stop_reason"),
            "answer": str(rec.get("answer", ""))[:200]}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--label", required=True, help="short label for output/ledger (the axis under test)")
    ap.add_argument("--target", default="", help='child target idx, e.g. "4,47,99" (may be empty)')
    ap.add_argument("--sentinels", type=Path, default=DEFAULT_SENTINELS)
    ap.add_argument("--gold", type=Path, default=DEFAULT_GOLD)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--out", type=Path, default=None,
                    help="output json (default artifacts/focused_gate_<label>.json)")
    ap.add_argument("--inject-answers", type=Path, default=None,
                    help="json {index: answer} to bypass live generation for those idx (testing)")
    ap.add_argument("--no-ledger", action="store_true", help="skip the experiment_ledger append")
    ap.add_argument("--issue", default="SOT-2623", help="issue id recorded in the ledger entry")
    ap.add_argument("--dev", action="store_true",
                    help="DEV mode: allow a non-flash-3.6 stack (e.g. Sonnet/claude-cli). Result is "
                         "stamped official:false and CANNOT be cited as a non-regression basis.")
    ap.add_argument("--no-smoke", action="store_true",
                    help="skip the pre-run model smoke probe (for offline/inject testing)")
    args = ap.parse_args(argv)

    if not args.gold.exists():
        print(f"[error] gold not found: {args.gold}", file=sys.stderr)
        return 3
    try:
        from scoring import codex_judge
        if not codex_judge.available():
            print("[error] codex judge unavailable (JUDGE backend is codex, no fallback; SOT-2457).",
                  file=sys.stderr)
            return 3
    except Exception as exc:  # noqa: BLE001
        print(f"[error] codex judge preflight failed: {exc}", file=sys.stderr)
        return 3

    sentinels = _load_sentinels(args.sentinels)
    sentinel_idx = [int(s["index"]) for s in sentinels]
    target_idx = _parse_indices(args.target)
    injected = None
    if args.inject_answers:
        injected = {int(k): v for k, v in
                    json.loads(args.inject_answers.read_text(encoding="utf-8")).items()}

    qbyidx = {int(r["index"]): str(r["question"])
              for _, r in pd.read_csv(settings.QUESTIONS_TEST).iterrows()}
    gold = gold_offline.load_gold(args.gold)

    # Union (dedup, target first for reporting), but only idx that exist in the question set.
    order: list[int] = []
    for i in target_idx + sentinel_idx:
        if i in qbyidx and i not in order:
            order.append(i)
    missing = [i for i in target_idx + sentinel_idx if i not in qbyidx]
    if missing:
        print(f"[warn] idx not in question set, skipped: {missing}", file=sys.stderr)

    live = injected is None or any(i not in injected for i in order)

    # --- Model guard (SOT-2625): resolve the EFFECTIVE model config and gate official verdicts ------
    model_info = _model_env()
    guard_reasons = _official_guard_reasons(model_info)
    # A verdict is OFFICIAL only when: not --dev, real generation happens (not fully injected), and the
    # stack matches flash-3.6/global/gemini. Injected answers are never a real model measurement.
    official = (not args.dev) and live and not guard_reasons
    print("=== effective model resolution (env overrides included) ===")
    print(f"  LLM_PROVIDER={model_info['provider']} GEN_MODEL={model_info['gen_model']} "
          f"GEN_MODEL_HARD={model_info['hard_model']} VISION_MODEL={model_info['vision_model']} "
          f"VERTEX_LOCATION={model_info['location']}", flush=True)
    if guard_reasons and not args.dev and live:
        print("[error] OFFICIAL model guard FAILED — refusing to emit a non-regression verdict.",
              file=sys.stderr)
        for r in guard_reasons:
            print(f"  - {r}", file=sys.stderr)
        print("  Official focused verdicts require the production flash-3.6 stack "
              f"(provider={OFFICIAL_PROVIDER}, GEN_MODEL/HARD/VISION={OFFICIAL_MODEL}, "
              f"VERTEX_LOCATION={OFFICIAL_LOCATION}). Re-run with the champion env, or pass --dev to "
              "run a NON-official Sonnet exploration (result cannot be a non-regression basis).",
              file=sys.stderr)
        return 3
    if not official:
        why = "--dev" if args.dev else ("injected answers" if not live else "; ".join(guard_reasons))
        print(f"[dev] NON-OFFICIAL run ({why}); result stamped official:false — NOT a non-regression "
              "basis (model mismatch). See SOT-2625.", flush=True)

    if live:
        _inject_timeout()
        if not args.no_smoke:
            ok, detail = _smoke(model_info)
            print(f"[smoke] model={model_info['model']} provider={model_info['provider']} "
                  f"reachable={ok} resp={detail!r}", flush=True)
            if not ok and official:
                print("[error] official smoke failed — the flash-3.6 stack is not reachable "
                      "(e.g. 404 off 'global'); refusing to emit an official verdict.", file=sys.stderr)
                return 3

    print(f"=== focused gate '{args.label}' start {_dt.datetime.now(_dt.timezone.utc).replace(tzinfo=None).isoformat()}Z ===")
    print(f"target={target_idx} sentinels={sentinel_idx} official={official} "
          f"mode={'inject' if injected is not None else 'live'}", flush=True)

    recs = _run(order, qbyidx, args.workers, injected)
    verdicts = _judge(order, recs, gold)

    target_rows = [_row(i, recs, verdicts) for i in target_idx if i in qbyidx]
    sentinel_rows = []
    for s in sentinels:
        i = int(s["index"])
        r = _row(i, recs, verdicts)
        r["archetype"] = s.get("archetype")
        r["expected_route"] = s.get("route")
        r["regressed"] = (i in qbyidx) and not r["is_match"]  # existing-MATCH dropped from MATCH
        sentinel_rows.append(r)

    regressions = [r for r in sentinel_rows if r["regressed"]]
    gate = "FAIL" if regressions else "PASS"
    sentinel_match = sum(1 for r in sentinel_rows if r["is_match"])

    out = args.out or (settings.ARTIFACTS_DIR / f"focused_gate_{args.label}.json")
    payload = {
        "label": args.label,
        "recordedAt": _dt.datetime.now(_dt.timezone.utc).replace(microsecond=0, tzinfo=None).isoformat() + "Z",
        "mode": "inject" if injected is not None else "live",
        "gate": gate,
        "official": official,
        "official_note": (None if official else
                          "NON-OFFICIAL (model mismatch / dev / injected) — cannot be cited as a "
                          "non-regression basis; official verdicts require the flash-3.6 stack (SOT-2625)."),
        "model": model_info["model"],
        "provider": model_info["provider"],
        "model_env": model_info,
        "champion": "SOT-2610 Wave A net40",
        "gold": str(args.gold),
        "target": {"indices": target_idx, "rows": target_rows},
        "sentinel": {"indices": sentinel_idx, "match": sentinel_match,
                     "total": len(sentinel_rows), "rows": sentinel_rows,
                     "regressions": [r["index"] for r in regressions]},
    }
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"\n=== GATE {gate} (official={official}) ===")
    if not official:
        print("  ⚠ NON-OFFICIAL result — model mismatch/dev/injected; NOT a non-regression basis.")
    print(f"sentinels: {sentinel_match}/{len(sentinel_rows)} MATCH; "
          f"regressions={[r['index'] for r in regressions]}")
    print(f"target verdicts: {[(r['index'], r['verdict']) for r in target_rows]}")
    print(f"wrote {out}")

    if not args.no_ledger:
        entry = {
            "recordedAt": payload["recordedAt"],
            "axis": f"focused gate run [{args.label}] (mandatory existing-MATCH sentinel set, SOT-2623)",
            "result": "pass" if gate == "PASS" else "fail",
            "cycle": "SOT-2623",
            "gate": gate,
            # Required provenance (SOT-2625): every runner-written ledger line carries the resolved
            # model/provider and whether the verdict is official (flash-3.6) or dev-only.
            "official": official,
            "model": model_info["model"],
            "provider": model_info["provider"],
            "target": {"indices": target_idx,
                       "verdicts": {str(r["index"]): r["verdict"] for r in target_rows}},
            "sentinel": {"indices": sentinel_idx,
                         "match": sentinel_match, "total": len(sentinel_rows),
                         "verdicts": {str(r["index"]): r["verdict"] for r in sentinel_rows},
                         "regressions": [r["index"] for r in regressions]},
            "evidence": f"champion=SOT-2610 Wave A net40; gate={gate}; "
                        f"sentinel {sentinel_match}/{len(sentinel_rows)} MATCH; out={out.name}",
            "issue": args.issue,
        }
        _require_ledger_fields(entry)
        with LEDGER.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(entry, ensure_ascii=False) + "\n")
        print(f"appended ledger: {LEDGER}")

    return 0 if gate == "PASS" else 2


if __name__ == "__main__":
    raise SystemExit(main())
