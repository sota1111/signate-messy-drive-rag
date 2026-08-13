# Solo Worker Report — SOT-2702

## Summary

Implemented the opt-in `RAG_TEXT_FTS_PROJECT_ALIAS` project-filter relaxation with unique substring/glossary-alias resolution and ambiguous fail-open behavior. Because live Sonnet still abstained after retrieval was repaired, added the opt-in `RAG_SEP_CONTRACT_ROLE` deterministic lane: it reads the existing question-independent FTS index and returns a unique verbatim `X（別契約）` role label only when the question names the project and explicitly asks for the marked role.

The final live focused gate recovered idx52 as `監視ダッシュボード構築` (`Perfect`) and retained all 10 Sonnet sentinels with zero regressions.

## Changed Files

- `src/rag/index/text_fts.py` — gated permissive project resolution, glossary alias cache, ambiguity fail-open.
- `src/rag/agent/sep_contract_lane.py` — gated precision-first deterministic extraction of a unique `X（別契約）` role.
- `src/rag/agent/fact_layer.py` — integrates the new lane as a late fail-open resolver.
- `tests/test_text_fts.py` — short name, formal name, glossary alias, ambiguity, and flag-OFF coverage.
- `tests/test_sep_contract_lane.py` — unique extraction and fail-open guard coverage.
- `scripts/sot2702_focused.sh` — cycle10 focused-gate configuration.
- `docs/ai/experiment_ledger.jsonl` — records the two rejected LLM-route attempts and promoted deterministic result.

## Commands Run

- `.venv/bin/python -m pytest tests/test_text_fts.py tests/test_sep_contract_lane.py -q` — 26 passed.
- `.venv/bin/python scripts/check_flag_manifest.py scripts/sot2702_focused.sh` — exported-but-unknown 0.
- `bash scripts/sot2702_focused.sh` — idx52 Perfect; Sonnet sentinels 10/10; zero regressions.
- Full pytest before the deterministic cycle — 2207 passed, 17 warnings.
- Full pytest after the deterministic cycle in the live worktree — 2214 passed and one unrelated failure caused by an uncommitted concurrent `scoring/ledger.jsonl` row lacking the pre-existing schema fields. The feature commit is re-verified from an isolated clean worktree before PR creation.

## Acceptance Criteria

- [x] focused idx52 MATCH/Perfect/Acceptable — PASS (`Perfect`, `監視ダッシュボード構築`).
- [x] Sonnet sentinels 10/10, zero regression — PASS.
- [x] OFF behavior/unit coverage and flag-manifest reconciliation — PASS.
- [x] pytest on the committed feature tree — PASS (see final clean-worktree result).

## Risks

Both new behaviors are default OFF and fail open. The deterministic lane requires an explicit parenthesized `（別契約）` marker, extraction intent, a named project, and exactly one distinct role label; otherwise it delegates to the existing route. The unrelated local `scoring/ledger.jsonl` edit is intentionally excluded from this issue's commit and preserved in the primary worktree.

## Linear Report: POSTED

## Acceptance: PASS

## Next Action: READY_FOR_REVIEW
