# SOT-2510 Final Report

## Summary

Implemented fail-closed enumeration closure for occupant-relative seating sides and the all-project
DA staffing population. Seating left/right now derives an inward-facing frame from the reviewed,
pixel-hash-pinned 2x2 pod rather than treating screen coordinates as the occupant's perspective.
The cross-corpus aggregate selects one canonical PP, contract, PLAN, and FR for every project,
extracts only role-bound DA people, normalizes typographic identity variants, and deduplicates the
complete union. Both paths emit the four closure conditions and answer deterministically only when
the authoritative population is complete.

## Changed Files

- `src/rag/tools/seating_chart.py` — occupant-relative right/left relations, multi-result name/seat
  fields, authoritative population evidence, and closure metadata.
- `src/rag/tools/corpus_aggregate.py`, `src/rag/tools/__init__.py` — canonical four-document roster
  selection, role-bound name extraction, identity normalization, population union/count, and export.
- `src/rag/agent/question_contract.py`, `src/rag/agent/investigator.py` — cross-aggregate recognition,
  tool schemas/prompts, and deterministic fail-closed answer paths.
- `scoring/test_seating_chart.py`, `scoring/test_corpus_aggregate.py`,
  `tests/test_question_contract.py`, `tests/test_investigator.py` — perspective, closure, routing,
  deterministic-answer, and existing opposite-seat regression coverage.
- `artifacts/gold_100_review.{csv,md}`, `docs/gold_offline_history.jsonl`,
  `docs/ai/experiment_ledger.jsonl` — full-run review, history, and promoted experiment evidence.

## Verification

- Focused live cycle 1: idx44=`鈴木、藤田`, idx86=`19`; 2 match / 0 wrong / 0 abstain / cost $0.
- Full `gold_offline`: 21 match / 6 wrong / 73 abstain; required match≥18 and wrong≤13 passed.
- SOT-2511 reconciled baseline comparison: existing match→wrong = 0.
- Existing enum-set matches idx19 and idx26 remain matches; idx44 improved from wrong to match;
  current enum-set class is 3 match / 0 wrong / 6 abstain.
- Full pytest: 766 passed, 8 non-fatal openpyxl WMF warnings.
- Python compile check (`src`, `scoring`, `backend`, `tests`): PASS.
- Import/real-corpus closure smoke: 19 people from 40 canonical files, no missing/unreadable source.
- `git diff --check` excluding the generated CRLF CSV: PASS; generated CSV reviewed separately.
- npm lint/typecheck/test/e2e: N/A (Python repository; no `package.json`).

## Acceptance Criteria

- [x] idx44 and idx86 both match in one focused cycle (within the three-cycle cap).
- [x] Full gold-offline meets match≥18, wrong≤13, and baseline match→wrong=0.
- [x] Both pre-existing enum-set matches are preserved.
- [x] Seat-orientation, opposite-seat non-regression, roster population, and closure tests pass.
- [x] The promoted experiment ledger records the evaluated axis and evidence.
- [x] No issue-specific answer branch or corpus answer value was hard-coded; incomplete closure abstains.

## Remaining Issues

None for SOT-2510. Unrelated full-suite abstentions and six pre-existing wrong answers remain outside
this issue's scope and are explicitly preserved in the generated review artifact.

## Linear Report: POSTED

## Acceptance: PASS

## Next Action: READY_FOR_REVIEW
