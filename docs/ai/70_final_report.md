# SOT-2511 Final Report

## Summary

Implemented complete fallback-rule answers and deterministic PPTX Gantt week-grid reading. Negative
special-rule answers now require rate/tax treatment, billing unit/rounding, settlement cycle, and cap.
Native Gantt shapes are mapped to calibrated half-open week cells, with ambiguous geometry failing
closed. When the authoritative file and requested activity/rule are unique, a generic deterministic
path returns the extracted result without relying on model phrasing or vision.

## Improvement Cycles

| Cycle | Result | Decision |
| --- | --- | --- |
| 1–3 | idx51 remained match; idx78 incomplete/wrong or safely abstained | Added negative-wording guard; initial run stopped at the original cap |
| 4 | Both Missing due Gemini role error / timeout | Normalize model history and separate Gantt from numeric-chart routing |
| 5 | idx51 Missing; idx78 incomplete free text | Add fail-closed deterministic paths and apply completeness guard to free text |
| 6 | idx51 Perfect; idx78 semantic match judged Incorrect on surface form | Normalize the generic regulation answer template |
| 7 | idx51 Perfect; idx78 Perfect | Focused gate PASS (2/2 match, 0 wrong) |
| 8 | Full-run-only idx4 regression restored to `bmi` via mandatory compute evidence | Reconciled non-regression gate PASS |

## Changed Files

- `src/rag/agent/question_contract.py`, `obligations.py`, `routing.py` — regulation/Gantt completion
  contracts, evidence obligations, routing, and numeric compute-evidence requirement.
- `src/rag/agent/investigator.py` — complete-answer guards, plain-user retry directives, normalized Gemini
  history, bounded timeout, and generic fail-closed deterministic resolution.
- `src/rag/extract/office.py` — week-header calibration and bar-to-week overlap extraction.
- `tests/test_question_contract.py`, `test_obligations.py`, `test_routing.py`, `test_investigator.py`,
  `test_pptx_gantt.py` — contract, transport, real-corpus, geometry, boundary, and ambiguity regressions.
- `docs/ai/experiment_ledger.jsonl` — inconclusive and promoted experiment records.

## Verification

- Focused idx51/78 cycle 7: 2 match / 0 wrong / 0 abstain; both `Perfect`.
- Full gold-offline reconciled: match 23 / wrong 6 / abstain 71; baseline match→wrong 0.
- Full pytest: 762 passed, 7 non-fatal openpyxl WMF warnings.
- Focused suite after final fix: 113 passed.
- Python compile check (`src`, `scoring`, `tests`, `backend`): PASS.
- `git diff --check`: PASS.
- npm lint/typecheck/test/e2e: N/A (Python repository; no `package.json`).

## Acceptance

- [x] idx78 and idx51 are both focused matches.
- [x] Full gold-offline meets match≥18, wrong≤13, and baseline match→wrong=0.
- [x] Week-grid boundary and ambiguity tests pass.
- [x] Ledger attribution is promoted and no corpus answer value is hard-coded.

## Linear Report: POSTED

## Acceptance: PASS

## Next Action: READY_FOR_REVIEW
