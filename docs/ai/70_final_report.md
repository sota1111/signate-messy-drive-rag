# SOT-2505 Final Report

## Summary

Added a conservative deterministic pre-pass for structured answer equivalence before the CRAG LLM
judge. It normalizes assignment notation such as `=` / `は` / `です`, recognizes ranked key/value
facts, and accepts gold containment only when all structured facts are present with no contradictory
value, extra number, missing identifier, or negation mismatch. Gold-offline unresolved pairs now use
three-vote majority aggregation while retaining the strict judging prompt and deterministic downgrade.

The full-suite gate exposed a pre-existing malformed calibration path for diagnostic ledger rows.
Those rows are now explicitly marked/handled as diagnostics: excluded from model fitting and retained
by full-ledger reads.

## Improvement Cycles

| Cycle | Input | Result | Decision |
| --- | --- | --- | --- |
| 1 | Existing `artifacts/predictions_test_investigator.csv`; no answer regeneration | 20 match / 69 abstain / 11 wrong | PASS; stop after first of at most three cycles |

Baseline was 18 match / 69 abstain / 13 wrong. idx61 changed to `Perfect`, idx62 to `Acceptable`;
both are therefore `MATCH`. All previous 18 matches remain matches, and the other 11 previous wrong
indices remain wrong: 0, 4, 7, 10, 30, 44, 51, 52, 78, 86, 89.

## Changed Files

- `scoring/crag.py` — deterministic structured equivalence/containment and configurable majority path.
- `scoring/gold_offline.py` — three-vote majority for unresolved gold-offline pairs.
- `scoring/calibrate.py`, `scoring/ledger.jsonl` — exclude annotated diagnostic summaries from calibration.
- `scoring/test_crag.py`, `scoring/test_gold_offline.py`, `scoring/test_codex_judge.py`, `scoring/test_ledger_fidelity.py` — regression coverage.
- `README.md` — document the gold-offline judging behavior.

## Verification

- Python compile check: PASS (`src`, `scoring`, `tests`, `backend`).
- Focused judge/gold/ledger tests: 65 passed.
- Ledger fidelity tests: 6 passed.
- Full test suite: 724 passed, 7 non-fatal openpyxl WMF warnings.
- npm lint/typecheck/test/e2e: N/A (Python repository; no `package.json`).
- Live offline re-score: PASS, cycle 1, existing answers only (`--answers`, no `--run`).

## Acceptance

- [x] idx61/62 are matches without changing or regenerating answers.
- [x] Existing 18 matches do not regress; no other previous wrong answer becomes a match.
- [x] Focused tests are green and the result was achieved within one of three allowed cycles.
- [x] No question-specific answer was hard-coded; all rules operate on general structured facts.

## Acceptance: PASS

## Next Action: READY_FOR_REVIEW
