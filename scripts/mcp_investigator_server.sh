#!/usr/bin/env bash
# SOT-2626 — launch the investigator stdio MCP server for a flat-rate `claude -p --mcp-config` session.
#
# Resolves the repo root from this script's location, activates the project's .venv Python, and sets the
# PYTHONPATH (including /tmp/genai_patch, matching the gold100 harness) so the same deterministic corpus
# tools the live loop uses are importable. All diagnostics go to stderr; stdout is pure MCP protocol.
#
# Usage (normally invoked by claude via --mcp-config, not by hand):
#   scripts/mcp_investigator_server.sh
# Env:
#   RAG_MCP_TOOL_LOG  optional jsonl path to record {ts,tool,args,elapsed_ms,ok} per tools/call.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

PY="${REPO_ROOT}/.venv/bin/python"
if [[ ! -x "${PY}" ]]; then
  echo "[mcp-investigator] .venv python not found at ${PY}" >&2
  exit 1
fi

export PYTHONPATH="/tmp/genai_patch:${REPO_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
exec "${PY}" -m src.rag.mcp.server
