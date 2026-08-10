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

## Handled MCP methods

`initialize` (echoes the client's `protocolVersion`, else `2024-11-05`), `notifications/initialized`,
`ping`, `tools/list`, `tools/call`. Unknown methods return JSON-RPC error `-32601`.
