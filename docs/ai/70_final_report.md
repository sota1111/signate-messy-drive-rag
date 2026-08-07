# SOT-2507 Final Report

## Summary

Chart-value questions now require deterministic numeric authority. `read_chart_values` reads OOXML
`numCache` first; when an xlsx contains only raster histograms, it maps the ordered embedded images to
the authoritative numeric source columns, pins the image with SHA-256, and recomputes the histogram
from source cells. Vision remains available only for chart location and axis-label confirmation.

The investigator rejects a chart-value commit without a successful strict tool result. A first-turn
free-text answer or abstention receives one bounded mandatory-tool redirect; if strict evidence remains
unavailable, the agent abstains instead of committing a pixel-read number.

## Improvement Cycles

| Cycle | Result | Root cause / decision |
| --- | --- | --- |
| 1 | idx10 `Missing` | Gemini ended in free text before trying a tool; the guard safely abstained but had not forced one strict attempt. |
| 2 | idx10 `958`, `Perfect` | Added one bounded strict-tool redirect; live investigator used `canonical_route` then `read_chart_values`. Promote. |

## Changed Files

- `src/rag/tools/chart_numcache.py`, `src/rag/tools/__init__.py` — unified strict reader, raster chart/source mapping, pixel hash, and Scott-bin source recomputation.
- `src/rag/agent/investigator.py` — strict chart tool schema, vision demotion, mandatory evidence commit guard, and bounded first-turn redirect.
- `src/rag/agent/routing.py`, `src/rag/agent/question_contract.py`, `src/rag/agent/obligations.py`, `src/rag/agent/research_loop.py` — chart contract, route, and evidence-obligation enforcement.
- `scoring/test_chart_numcache.py`, `tests/test_investigator.py`, `tests/test_routing.py` — synthetic, fail-closed, contract, and real idx10 coverage.
- `docs/ai/experiment_ledger.jsonl` — promoted `chart-numeric-strict-path` experiment record.

## Verification

- Focused chart/agent/contract suite: 129 passed.
- Full Python suite: 746 passed (completed with empty `lastfailed` cache).
- Python compile check: PASS (`src`, `scoring`, `tests`).
- `git diff --check`: PASS.
- Lint/typecheck: N/A (no configured linter/typechecker; `ruff` is not installed).
- npm/e2e: N/A (Python repository; no `package.json`).
- Full `gold_offline --run`: match=21, wrong=7, abstain=72, cost=$5.3873.
- Baseline transition check (SOT-2508 reconciled): existing match→wrong=0; idx10 wrong→match.

## Acceptance

- [x] idx10 is a focused live match within two of three cycles (`958`, `Perfect`).
- [x] Full gold-offline is non-regressive: match 21 ≥ 18, wrong 7 ≤ 13, existing match→wrong 0.
- [x] chart_numcache focused tests and the complete Python suite pass.
- [x] `axis=chart-numeric-strict-path` is recorded as promoted in the experiment ledger.
- [x] Numeric results come from numCache/source cells; vision cannot authorize a number.
- [x] No corpus answer value is hard-coded in production code.

## Acceptance: PASS

## Next Action: READY_FOR_REVIEW
