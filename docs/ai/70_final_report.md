# SOT-2469 Final Report

## Summary

Routed the production answer path to the Gemini tool-driven investigator. The legacy text-only
Gemini and Claude Opus backends remain available only through explicit experimental selection; the
default `run.run(...)` and CLI path no longer depend on Claude.

## Verification

- Python compile check passed for `src`, `scoring`, and `tests`.
- Full offline suite: 336 passed, 7 non-fatal openpyxl warnings.
- Real test-split smoke run: one question answered through `gemini-2.5-pro` investigator in three
  iterations, producing `artifacts/sot2469_investigator_smoke.csv` with no abstention.

## Acceptance

- [x] Production answer path is Gemini-only and Claude-independent.
- [x] Legacy text backends remain available only by explicit experimental selection.
- [x] A test-split prediction is generated through the investigator backend.

## Acceptance: PASS

## Linear Report: POSTED

## Next Action: READY_FOR_REVIEW
