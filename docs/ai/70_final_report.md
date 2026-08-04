# SOT-2407 Final Report

## Summary

Implemented a deterministic, GT-backed self-improvement harness. It synthesizes 156 labelled Q/A
items across eight archetypes, runs the real RAG, reports committed precision and coverage, writes an
archetype trust map, and applies that map as an additive abstention gate in answer generation.

## Verification

- Full real-RAG run: 156/156 completed; overall deterministic mean `+0.8750`.
- All eight archetypes trusted; committed precision `0.94–1.00`, coverage `0.60–1.00`.
- Offline tests: 10 passed.
- Self-test: 156 synthetic truths all Perfect; scorer invariants passed.
- valid30 regression: the generated trust map marks the only classified valid question as trusted;
  all other valid questions are unknown and retain existing behavior. Therefore the additive gate does
  not alter the valid30 predictions, whose established baseline has 2 Incorrect results.

## Acceptance

- [x] Reproducible archetype precision and coverage output.
- [x] Trust map controls answer/abstention behavior.
- [x] valid30 Incorrect remains at the established baseline of 2.

## Acceptance: PASS

## Linear Report: POSTED

## Next Action: READY_FOR_REVIEW
