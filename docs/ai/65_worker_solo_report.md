# Worker Report (solo) — SOT-2661

## Summary

Implemented the default-OFF `RAG_PLAN_FANOUT` Sonnet/claude-mcp flow: one planning/fan-out search,
evidence-packet synthesis, at most two focused supplements, then grounded submit or abstention. The MCP
budget is five non-terminal tool calls plus the uncapped submit (six total), and per-stage counts are
recorded under `interventions.plan_fanout`.

The final required `--dev` Sonnet focused gate passed with all 10 existing-MATCH sentinels preserved,
zero target wrong-answer increase, and 4.4 mean non-submit tool calls across the ten live LLM cases.

## Changed Files

- `src/rag/agent/investigator.py` — default-OFF three-stage prompt, compact compute/output discipline,
  and stage-shaped budget helper.
- `src/rag/llm_providers/claude_mcp.py` — budget wiring and plan/fan-out/supplement telemetry.
- `tests/test_claude_mcp.py` — OFF invariance, budget, prompt contract, cap wiring, and telemetry tests.
- `scripts/sonnet_plan_fanout_focused.sh` — reproducible Sonnet focused gate.
- `docs/ai/experiment_ledger.jsonl` — two rejected prompt variants and the promoted final gate result.

## Commands Run

- `.venv/bin/python -m pytest -q tests/test_claude_mcp.py tests/test_investigator.py tests/test_unified_search.py` — 133 passed.
- `.venv/bin/python scripts/check_flag_manifest.py scripts/sonnet_plan_fanout_focused.sh` — pass; zero unknown flags.
- `.venv/bin/python -m pytest -q` — 1,816 passed; one unrelated dirty-artifact failure because the
  focused review CSV contains 15 rows while the legacy test requires at least 30. The same test passed
  from a clean detached `main` worktree (1 passed).
- `bash scripts/sonnet_plan_fanout_focused.sh` — final PASS (`official:false`): sentinels 10/10,
  regressions `[]`; targets idx34/76/98 all Missing (wrong count 0 vs baseline 1); mean live LLM tool
  calls 4.4 (≤6). Two preceding variants failed 9/10 and were recorded/rejected before the precision
  prompt was finalized.

## Acceptance Criteria

- [x] Three-stage flow is feature-gated and live telemetry shows mean 4.4 tool calls, down from the
  stated 12–18 baseline and below the ≤6 KPI.
- [x] Final focused gate preserved 10/10 MATCH sentinels; target wrong count decreased from one to zero.
- [x] OFF path preserves the legacy prompt and budget; ON records budget, first tool, search calls,
  supplements, extra searches, and tool turns under `interventions.plan_fanout`.

## Risks

- The focused gate is intentionally `official:false`; the running Sonnet cycle owns full-gold integration.
- All three BUDGET targets safely abstained; this flow controls turn cost and precision but did not recover
  a new target answer in this focused sample.
- Generated review/history/submission artifacts already dirty before this issue remain excluded from the commit.

## Acceptance: PASS
## Next Action: READY_FOR_REVIEW
