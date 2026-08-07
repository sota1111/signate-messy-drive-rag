# SOT-2511 Final Report

## Summary

Implemented generic regulation-answer completeness obligations and deterministic PPTX Gantt week-grid
extraction. The Gantt path calibrates native week-header x coordinates and maps bar `left`/`width` by
positive half-open interval overlap; the real idx51 source resolves to weeks 6–8 in every live cycle.
Regulation answers that deny a special rule are now required to include the fallback rule's rate and tax
treatment, billing unit and rounding, settlement cycle, and cap before `submit_answer` can commit.

The required focused gate did not pass within the maximum three cycles. The third cycle exposed an
additional negative wording ("特別な精算方法は規定されていません") that bypassed the guard. That parser
case is fixed and unit-tested after cycle 3, but the issue forbids a fourth live cycle, so the fix remains
live-unverified and no PR was created.

## Improvement Cycles

| Cycle | idx51 | idx78 | Focused result |
| --- | --- | --- | --- |
| 1 | Perfect (第6–8週) | Incorrect (explicit tax addition missing) | match 1 / wrong 1 |
| 2 | Acceptable (第6–8週) | Missing (safe abstain) | match 1 / abstain 1 |
| 3 | Perfect (第6–8週) | Incorrect (monthly/tax/cap details missing via negative-wording bypass) | match 1 / wrong 1 |

## Changed Files

- `src/rag/agent/question_contract.py` — question-specific regulation/Gantt completion contracts and
  negative-rule answer validation.
- `src/rag/agent/obligations.py` — fallback regulation and Gantt geometry/conflict obligations.
- `src/rag/agent/routing.py`, `src/rag/agent/investigator.py` — deterministic-first routes and incomplete
  regulation submit rejection.
- `src/rag/extract/office.py` — native shape week-grid calibration, boundary mapping, and ambiguity output.
- `tests/test_question_contract.py`, `tests/test_obligations.py`, `tests/test_routing.py`,
  `tests/test_investigator.py`, `tests/test_pptx_gantt.py` — offline regression and boundary coverage.
- `docs/ai/experiment_ledger.jsonl` — `regulation-completeness-gantt-grid` attribution.

## Verification

- Focused unit tests: 104 passed.
- Full pytest: 748 passed, 7 non-fatal openpyxl WMF warnings.
- Python compile check (`src`, `scoring`, `tests`, `backend`): PASS.
- `git diff --check`: PASS.
- npm lint/typecheck/test/e2e: N/A (Python repository; no `package.json`).
- Full gold-offline: not run because the prerequisite focused gate failed.
- Local implementation commit: `613fc54`; not pushed.

## Acceptance

- [ ] idx78 and idx51 are both match in one focused run within three cycles (idx51 passed all cycles;
  idx78 did not).
- [ ] Full gold-offline non-regression (not run after focused failure).
- [x] Gantt week-grid unit tests pass, including exact boundary and boundary ±epsilon behavior.
- [x] Ambiguous competing bars fail closed instead of selecting a week by vision.
- [x] Experiment ledger attribution is recorded; no corpus-specific answer value was hard-coded.

## GitHub

No push, PR, or merge was performed because the mandatory acceptance/PR gate failed. The feature branch
and local commit are retained for a later authorized debug cycle.

## Linear Report: POSTED

## Acceptance: FAIL

## Next Action: BLOCKED
