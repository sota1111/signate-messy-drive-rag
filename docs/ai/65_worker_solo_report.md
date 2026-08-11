# Worker Report (solo) — SOT-2648

## Summary

Completed Sonnet gold improvement cycle 1 in the private, `official:false` lane. Added a default-OFF
`RAG_FORBID_GEMINI` guard that raises before any cached or new genai client can be used, a reproducible
serial Sonnet focused/full-cycle runner, a cross-type 10-question Sonnet sentinel set, and the Sonnet
cycle history ledger.

The final focused gate passed with 10/10 sentinels and zero regressions. The fact layer recovered idx63
and idx87 from abstain to correct answers; idx38 and idx57 remained safely abstained. The required full
Sonnet dev gold100 completed under the guard at $0.0000 Gemini cost, but scored 39 match / 28 abstain /
33 wrong (net 6), below cycle-0 net 18. The axis is therefore recorded as rejected for promotion; no
official Flash/champion or submission asset was changed.

## Changed Files

- `src/rag/llm.py` — default-OFF `RAG_FORBID_GEMINI` guard on all genai client access.
- `tests/test_llm_provider.py` — cached/new client blocking, truthy parsing, text-only Claude routing,
  and no-silent-fallback coverage.
- `scripts/sonnet_gold_cycle1.sh` — guarded serial/resumable Sonnet dev gold100 runner.
- `scripts/sonnet_gold_cycle1_focused.sh` — guarded focused target + sentinel runner.
- `scripts/sonnet_sentinels.json` — 10-question, cross-type Sonnet sentinel set; wording-flaky idx16/71
  replaced with two-sample-stable verbatim extraction idx3/81 after the first live gate.
- `docs/ai/sonnet_gold_history.jsonl` — cycle 0 baseline and cycle 1 result/handoff.
- `docs/ai/experiment_ledger.jsonl` — failed calibration gate, passing final focused gate, and rejected
  full-cycle axis recorded.
- `docs/gold_offline_history.jsonl`, `artifacts/gold_100_review.{md,csv}` — private dev measurement record.

## Commands Run

- `.venv/bin/pytest -q tests/test_llm_provider.py tests/test_focused_gate_model_guard.py` — 26 passed.
- `.venv/bin/pytest -q` — 1,619 passed, 17 existing openpyxl warnings (599.56s).
- `bash scripts/sonnet_gold_cycle1_focused.sh` — final PASS, 10/10 sentinels, regressions `[]`;
  idx63=`Acceptable`, idx87=`Perfect`, idx38/57=`Missing`.
- `bash scripts/sonnet_gold_cycle1.sh` — completed 100/100, `official:false`, match39 / abstain28 /
  wrong33 / net6 / cost `$0.0000`.

## Acceptance Criteria

- [x] Gemini cost $0 confirmed: full run reports `$0.0000`; `RAG_FORBID_GEMINI=1` guarded every genai
  client access throughout answer execution.
- [x] Focused improvement plus zero sentinel regression: idx63/87 recovered; final sentinel gate 10/10.
- [x] Ledger and next-cycle handoff recorded in both Sonnet history and experiment ledger.
- [x] Official lane untouched: all measurements are stamped `official:false`; no Flash champion/LB asset
  or SIGNATE submission was changed.

## Risks

- Full-run net6 regressed from cycle0 net18 despite focused deterministic recovery. The configuration is
  not promoted; repeated-sample re-anchoring and generic verbosity control are the next priorities.
- idx38/57 remain abstentions; extending their question-independent evidence coverage is preferred over
  forcing answers.

## Linear Report: POSTED
## Acceptance: PASS
## Next Action: READY_FOR_REVIEW
