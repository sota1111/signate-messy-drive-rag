#!/usr/bin/env bash
# SOT-2628 — first full DEV gold100 over the flat-rate Sonnet tool loop (claude-mcp backend).
#
# Configuration = champion "Wave A + B1" (identical to scripts/sot_cycle2_gold100.sh's champion base,
# net40) with the investigator loop moved OFF Gemini and onto flat-rate Sonnet via
# RAG_INVESTIGATOR_BACKEND=claude-mcp (SOT-2627). This makes the two runs apples-to-apples: the SAME
# deterministic-first pipeline (Wave A router + B1 document_extract), differing ONLY in which model
# drives the investigator tool loop for the questions that actually reach it. That isolates a pure
# model-reachability diff (Sonnet vs gemini-3.6-flash) on the same evidence tooling.
#
# ***OFFICIAL:FALSE*** — this is a DEV measurement only. The official net / non-regression judgment is
# pinned to gemini-3.6-flash (SOT-2625 model guard). This run MUST NOT back a champion promotion or a
# non-regression claim. `--no-official` records it as official:false in docs/gold_offline_history.jsonl
# so it is machine-distinguishable from the official flash-3.6 history.
#
# COST: the Sonnet tool loop is $0 (flat-rate plan; Usage.cost_usd zeroes any "(claude-mcp)" model).
# Any residual cost_usd in the report is Gemini used OUTSIDE the investigator loop (vision fallback /
# deterministic naturalization) at gemini-3.6-flash — report that number and its cause.
#
# USAGE-LIMIT: the flat-rate limit is shared account-wide with the autonomous workers. Run at
# parallelism 1, ideally when workers are idle. On a detected limit the claude-mcp backend abstains
# that question immediately (it does NOT hammer the limit) and persists every ANSWERED question to the
# resume sidecar below — so simply re-running this script continues where it stopped (may span days).
set -euo pipefail
cd /workspaces/signate-messy-drive-rag

# --- flag-manifest preflight (SOT-2624): abort if this script exports a RAG_* no source reads ---
# Catches the cycle-2 H3 accident class (typo / retired-flag export) before the expensive run.
.venv/bin/python scripts/check_flag_manifest.py "$0" || exit 1

# Residual-Gemini model for anything OUTSIDE the investigator loop (vision fallback / naturalization).
# Kept at the official flash-3.6 stack so the non-Sonnet parts match the champion run exactly.
export VERTEX_LOCATION=global
export GEN_MODEL=gemini-3.6-flash
export GEN_MODEL_HARD=gemini-3.6-flash
export VISION_MODEL=gemini-3.6-flash

# genai per-request timeout injection (SOT-2568 method)
export PYTHONPATH=/tmp/genai_patch:.

# --- champion answer-increase / error-type flags (r1, net32 base) ---
export RAG_FIRST_MOVE_ROUTING=1 RAG_SPIN_DETECTION=1 RAG_ADAPTIVE_BUDGET=1 RAG_EVIDENCE_CACHE=1
export RAG_BUDGET_BOUNDARY_RESEARCH=1 RAG_UNANSWERABLE_FALLBACK=1 RAG_PDF_OCR=1 RAG_SHARE_CORPUS_PROFILE=1
export RAG_CANONICAL_MANIFEST=1 RAG_EVIDENCE_INDEX=1 RAG_STRUCTURE_STORE=1
export RAG_GRANULARITY_NORMALIZATION=1 RAG_XLSX_EMBEDDED_IMAGE=1 RAG_CONFLICT_RESOLUTION=1 GATE_EXEC_CORRECT=1
export RAG_NUMERIC_FEATURE_CORR=1 RAG_RELEVANCE_STRICT=1 RAG_HIGHLIGHT_EXTRA=1 RAG_FONT_EMPHASIS=1
export RAG_FILE_GREP_INDEX_CANDIDATES=1
export RAG_FORMAT_EVENTS=1
# --- Wave A champion deterministic router with B1-only (SOT-2618 single-gate config = net40) ---
export RAG_DET_PIPELINE_ROUTER=1
export RAG_DET_PIPELINE_B1=1   # Wave B1 document_extract (default ON)
export RAG_DET_PIPELINE_B2=0   # Wave B2 fact_lookup stays OFF

# --- dev-only: drive the investigator tool loop on flat-rate Sonnet via claude CLI + MCP (SOT-2627) ---
export RAG_INVESTIGATOR_BACKEND=claude-mcp
export RAG_CLAUDE_MCP_MODEL=sonnet
export RAG_CLAUDE_MCP_RESUME=artifacts/gold100_sonnet_dev_resume.jsonl   # resume sidecar (interrupt/resume)
export RAG_MCP_TOOL_LOG=artifacts/gold100_sonnet_dev_tool_calls.jsonl    # per tools/call trace

echo "=== SOT-2628 Sonnet dev gold100 (official:false) start $(date -u +%FT%TZ) ==="
echo "BACKEND=$RAG_INVESTIGATOR_BACKEND MODEL=$RAG_CLAUDE_MCP_MODEL ROUTER=$RAG_DET_PIPELINE_ROUTER B1=$RAG_DET_PIPELINE_B1 B2=$RAG_DET_PIPELINE_B2"
echo "GEN_MODEL(residual gemini)=$GEN_MODEL RESUME=$RAG_CLAUDE_MCP_RESUME"
# Parallelism 1 (shared-limit protection). --no-official ⇒ history official:false. judge=codex (default).
.venv/bin/python -m scoring.gold_offline --run --workers 1 --no-official \
  --out artifacts/gold100_sonnet_dev.json
echo "=== SOT-2628 Sonnet dev gold100 done $(date -u +%FT%TZ) exit=$? ==="
