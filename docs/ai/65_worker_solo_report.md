# Worker Report (solo) — SOT-2651

## Summary

Completed Sonnet local gold100 improvement cycle 4 after all five child issues merged. The integrated
focused gate passed with 20/24 target questions matching, four safe abstentions, and all 10 established
Sonnet sentinels preserved. The one required resumed full dev gold100 then completed at 57 match,
22 abstain, 21 wrong (net 36), versus the last measured cycle-2 result of 44/30/26 (net 18).

The evaluation was strictly `official:false`, used `RAG_INVESTIGATOR_BACKEND=claude-mcp` with Sonnet
and one worker, and recorded `$0.0000` Gemini cost. The official Flash champion and submission assets
were not changed.

## Changed Files

- `scripts/sonnet_gold_cycle4.sh` — reproducible serial, resumable cycle-4 Sonnet gold100 runner with
  all five child stores enabled and serve-time Gemini forbidden.
- `scripts/sonnet_gold_cycle4_integrated_focused.sh` — integrated target-plus-sentinel focused gate.
- `docs/ai/sonnet_cycle_analysis/cycle4.md` — exhaustive pre-decomposition abstain/wrong classification.
- `docs/ai/sonnet_gold_history.jsonl` — measured cycle-4 history entry and next-cycle handoff.
- `docs/ai/experiment_ledger.jsonl` — integrated focused and full gold100 evidence.
- `artifacts/gold_100_review.{md,csv}`, `docs/gold_offline_history.jsonl`, `scoring/ledger.jsonl` —
  generated non-official gold100 review and scoring history.

## Commands Run

- `bash scripts/sonnet_gold_cycle4_integrated_focused.sh` — PASS; targets 20/24 MATCH, four Missing,
  wrong 0 for the changed-path gate; sentinels 10/10 MATCH, regressions 0; `official:false`.
- `bash scripts/sonnet_gold_cycle4.sh` — PASS; 100/100 completed using resume cache; 57 match,
  22 abstain, 21 wrong, net 36; Gemini cost `$0.0000`; `official:false`.
- `.venv/bin/python scripts/check_flag_manifest.py scripts/sonnet_gold_cycle4.sh` — PASS; no unknown flags.
- `.venv/bin/python scripts/check_flag_manifest.py scripts/sonnet_gold_cycle4_integrated_focused.sh` — PASS; no unknown flags.
- JSONL parse check for both experiment/history ledgers — PASS.
- `.venv/bin/python -m pytest -q` — 1,836 passed, 17 warnings in 605.16s.

## Acceptance Criteria

- [x] Gemini cost is mechanically reported as `$0.0000`; answer execution used claude-mcp/Sonnet only
  with `RAG_FORBID_GEMINI=1`.
- [x] Integrated focused improvement passed for the child target set, with Sonnet sentinels 10/10 and
  zero sentinel regressions.
- [x] Cycle 4 is appended to both ledgers with measured full-run metrics and next-cycle handoff.

## Risks

- This is a single resumed dev measurement, so the large net gain is promising but not an official
  leaderboard claim; future cycles should retain deterministic focused evidence as the attribution unit.
- Remaining 22 abstentions are still dominated by `BUDGET_EXHAUSTED` (16), while version-diff questions
  remain the weakest class (1/6 MATCH, four wrong).
- The full run remains below the historical manual 86.7% reference and does not alter the official lane.

## Linear Report: POSTED
## Acceptance: PASS
## Next Action: READY_FOR_REVIEW
