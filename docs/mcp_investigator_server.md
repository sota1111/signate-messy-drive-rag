# Investigator MCP server (dev-only) — SOT-2626

A stdio [Model-Context-Protocol](https://modelcontextprotocol.io) server that exposes the investigator's
**read-only** tool set to a flat-rate `claude -p --mcp-config` session. This is the dev substrate for
SOT-2627 (running the investigator's tool loop on flat-rate Sonnet to avoid Gemini metering); the
production answer path (Gemini-only, SOT-2460) is **untouched** and official measurement stays on
flash 3.6 (SOT-2625).

## What it exposes

Every tool returned by `src.rag.agent.investigator.build_tools(CorpusProfile())` — the *single source of
truth* that also feeds the live Gemini `to_genai_tools` — is published verbatim as an MCP tool, with the
tool's own JSON-Schema `parameters` as the MCP `inputSchema`. No second schema is maintained. With the
champion default flags this is 15 tools: `find_files`, `file_grep`, `read_office`, `decrypt`, `compute`,
`canonical_route`, `read_chart_values`, `caption_image`, `pdf_emphasis`, `pptx_pivot`,
`highlight_extract`, `version_diff`, `seating_lookup`, `corpus_aggregate`, and the terminal
`submit_answer`. (Flag-gated tools such as `font_emphasis`/`format_events`/`pot_verify`/`enum_scan`
appear automatically when their env flags are on, since the list comes from `build_tools`.)

**Read-only guarantee.** Only `build_tools` is exposed — corpus read + restricted-pandas compute only;
there is no filesystem-write or shell tool in that set. `build_server` additionally *asserts* no exposed
tool name looks like a write/shell capability, so a future write tool cannot leak in silently.

## Files

- `src/rag/mcp/server.py` — the server (self-contained JSON-RPC 2.0 stdio transport; **no `mcp` SDK
  dependency**). Run as `python -m src.rag.mcp.server`.
- `scripts/mcp_investigator_server.sh` — launcher: resolves the repo root, the project `.venv` Python,
  and `PYTHONPATH=/tmp/genai_patch:<repo>` (matching the gold100 harness). stdout is pure protocol; all
  diagnostics go to stderr.
- `scripts/mcp_investigator.config.json` — `--mcp-config` template for the claude CLI.

## Connectivity check (claude CLI)

Run from the repo root (the launcher resolves `.venv`/`PYTHONPATH` itself):

```bash
# List the exposed tools
printf '%s' '利用可能なMCPツール mcp__investigator__* を列挙して' \
  | claude -p --model sonnet --mcp-config scripts/mcp_investigator.config.json

# Call one representative tool and check the result matches the investigator's own dispatch
printf '%s' 'mcp__investigator__find_files を query=train ext=xlsx で1回呼び、先頭のファイルパスを出力して' \
  | claude -p --model sonnet \
      --mcp-config scripts/mcp_investigator.config.json \
      --allowedTools "mcp__investigator__find_files"
```

The MCP tools are surfaced to claude as `mcp__investigator__<tool>`. A one-question manual investigation
works by allowing the tools you expect it to use (or all `mcp__investigator__*`) and letting it take a
few tool round-trips before it emits its answer.

### Raw stdio smoke (no claude, free)

The server speaks newline-delimited JSON-RPC — the same protocol claude drives:

```bash
printf '%s\n' \
  '{"jsonrpc":"2.0","id":1,"method":"initialize","params":{"protocolVersion":"2025-06-18"}}' \
  '{"jsonrpc":"2.0","method":"notifications/initialized"}' \
  '{"jsonrpc":"2.0","id":2,"method":"tools/list","params":{}}' \
  '{"jsonrpc":"2.0","id":3,"method":"tools/call","params":{"name":"find_files","arguments":{"query":"train","ext":"xlsx"}}}' \
  | bash scripts/mcp_investigator_server.sh
```

## Tool-call log (for SOT-2627 details reconstruction)

Set `RAG_MCP_TOOL_LOG` (the config template points it at `artifacts/mcp_tool_calls.jsonl`) to append one
`{ts, tool, args, elapsed_ms, ok}` record per `tools/call`. This feeds the later details.jsonl-compatible
reconstruction of a Sonnet tool loop. Best-effort — a logging failure never breaks a tool call.

## SOT-2627 — driving the whole loop from the investigator (`RAG_INVESTIGATOR_BACKEND=claude-mcp`)

SOT-2627 wires this server into the investigator itself. With `RAG_INVESTIGATOR_BACKEND=claude-mcp`
(default `gemini`), `investigator.answer_question` delegates one question's **entire** tool loop (tool
round-trips → final answer) to a flat-rate `claude -p --mcp-config` session over this server, instead of
the live Gemini function-calling loop. Every deterministic pre-stage (document_registry / det_pipeline /
`deterministic_*` shortcuts) runs first and unchanged — only the model-driven loop moves to Sonnet, so
the `gemini` default is byte-identical.

- **Selection**: `RAG_INVESTIGATOR_BACKEND=claude-mcp` (dev-only; production stays Gemini-only per
  SOT-2460, official measurement stays flash 3.6 per SOT-2625). Provider: `src/rag/llm_providers/claude_mcp.py`.
- **Model**: `RAG_CLAUDE_MCP_MODEL` (default `sonnet`); reported as `model="sonnet(claude-mcp)"` with
  `cost_usd=0` (flat rate has zero marginal cost).
- **Budget**: `claude` has no `--max-turns`, so the per-question budget is a subprocess wall-clock
  timeout (`timeout_s`) plus a server-side non-terminal tool-call cap `RAG_MCP_MAX_TOOL_CALLS`
  (= `max_turns`, passed through a generated per-question mcp-config; **0/unset = disabled**, so the
  SOT-2626 behaviour above is byte-identical). On the cap, the server returns a *finalize* directive
  steering the model to `submit_answer`; `submit_answer` itself is never capped.
- **Usage-limit interruption / resume**: on a detected flat-rate usage limit the run trips a
  process-wide latch so every remaining question short-circuits to an abstain immediately (no more
  `claude` calls hammer the shared account limit). Answered questions are persisted to a resume sidecar
  (`RAG_CLAUDE_MCP_RESUME`, default `<ARTIFACTS_DIR>/claude_mcp_resume.jsonl`; `0`/`off` disables) so a
  later re-run continues where it stopped.
- **WARNING**: the flat-rate limit is **shared account-wide with the autonomous workers** — run this
  only when workers are idle, at parallelism 1 (the investigator run default for this backend).

Focused reachability smoke (dev; not a gold100 run):

```bash
RAG_INVESTIGATOR_BACKEND=claude-mcp PYTHONPATH=/tmp/genai_patch:. \
  .venv/bin/python scripts/sot2627_focused_smoke.py 1 24 56 96
```

### Focused comparison (2026-08-11)

The SOT-2627 development smoke completed the four representative indices above on both backends. The
existing `scoring.gold_offline` reader accepted both details files without conversion.

| Backend | Completed | Match | Abstain | Elapsed by index (1 / 24 / 56 / 96) | Investigator turns |
| --- | ---: | ---: | ---: | --- | --- |
| `sonnet(claude-mcp)` | 4/4 | 2/4 | 0/4 | 21.0s / 43.7s / 211.5s / 157.5s | 4 / 12 / 19 / 7 |
| `gemini-2.5-pro` | 4/4 | 1/4 | 1/4 | 63.2s / 74.7s / 112.1s / 68.5s | 5 / 16 / 9 / 4 |

This is a reachability smoke, not a model-quality conclusion or an official regression run. The raw
details/score artifacts are intentionally untracked; the next issue performs the single full gold100
measurement. The Sonnet run reported zero marginal cost and did not trip the usage-limit latch.

## Handled MCP methods

`initialize` (echoes the client's `protocolVersion`, else `2024-11-05`), `notifications/initialized`,
`ping`, `tools/list`, `tools/call`. Unknown methods return JSON-RPC error `-32601`.
