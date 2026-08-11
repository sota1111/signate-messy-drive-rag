#!/usr/bin/env bash
# SOT-2648 — Sonnet gold100 improvement cycle 1 (official:false lane).
#
# Base env = SOT-2628 Sonnet dev gold100 (champion Wave A + B1, investigator on flat-rate Sonnet via
# RAG_INVESTIGATOR_BACKEND=claude-mcp) with THREE cycle-1 levers on top:
#   1. RAG_FACT_LAYER=1   — SOT-2647 precomputed fact layer (det lanes + case_filter/id_lookup/
#                           metric_lookup/diff_lookup tools). Targets the Sonnet abstains that are
#                           hard-core coverage-satisfied: idx38/87 (case_master), idx57/63
#                           (derived_metrics), plus diff_lookup for B-class version idx1/14/95.
#   2. RAG_FORBID_GEMINI=1 — no-Gemini guard (SOT-2648): any genai client access during this run
#                           raises GeminiForbiddenError instead of silently billing. This machine-
#                           enforces the "Gemini cost $0" acceptance criterion.
#   3. LLM_PROVIDER=claude-cli — tool-free text generate() (Stage3 naturalization / rerank fallback)
#                           goes to flat-rate Sonnet instead of Gemini, so the guard cannot kill a
#                           deterministic-lane answer whose naturalization fires.
#
# ***OFFICIAL:FALSE*** — DEV measurement only (--no-official). Never backs a champion promotion or
# non-regression claim. The official lane stays gemini-3.6-flash (SOT-2625 model guard), untouched.
#
# USAGE-LIMIT: flat-rate Sonnet is shared account-wide. Parallelism 1; on a detected limit the
# claude-mcp backend abstains the question and persists answered ones to the resume sidecar — simply
# re-run this script to continue where it stopped.
set -euo pipefail
cd /workspaces/signate-messy-drive-rag

# --- flag-manifest preflight (SOT-2624): abort if this script exports a RAG_* no source reads ---
.venv/bin/python scripts/check_flag_manifest.py "$0" || exit 1

# Residual model env kept at the official stack for config parity; with RAG_FORBID_GEMINI=1 these
# paths raise instead of billing (vision was 0-use in the SOT-2628 baseline).
export VERTEX_LOCATION=global
export GEN_MODEL=gemini-3.6-flash
export GEN_MODEL_HARD=gemini-3.6-flash
export VISION_MODEL=gemini-3.6-flash

# genai per-request timeout injection (SOT-2568 method)
export PYTHONPATH=/tmp/genai_patch:.

# --- champion answer-increase / error-type flags (r1, net32 base) — verbatim from SOT-2628 ---
export RAG_FIRST_MOVE_ROUTING=1 RAG_SPIN_DETECTION=1 RAG_ADAPTIVE_BUDGET=1 RAG_EVIDENCE_CACHE=1
export RAG_BUDGET_BOUNDARY_RESEARCH=1 RAG_UNANSWERABLE_FALLBACK=1 RAG_PDF_OCR=1 RAG_SHARE_CORPUS_PROFILE=1
export RAG_CANONICAL_MANIFEST=1 RAG_EVIDENCE_INDEX=1 RAG_STRUCTURE_STORE=1
export RAG_GRANULARITY_NORMALIZATION=1 RAG_XLSX_EMBEDDED_IMAGE=1 RAG_CONFLICT_RESOLUTION=1 GATE_EXEC_CORRECT=1
export RAG_NUMERIC_FEATURE_CORR=1 RAG_RELEVANCE_STRICT=1 RAG_HIGHLIGHT_EXTRA=1 RAG_FONT_EMPHASIS=1
export RAG_FILE_GREP_INDEX_CANDIDATES=1
export RAG_FORMAT_EVENTS=1
export RAG_DET_PIPELINE_ROUTER=1
export RAG_DET_PIPELINE_B1=1   # Wave B1 document_extract
export RAG_DET_PIPELINE_B2=0   # Wave B2 fact_lookup stays OFF

# --- cycle-1 levers (see header) ---
export RAG_FACT_LAYER=1
export RAG_FORBID_GEMINI=1
export LLM_PROVIDER=claude-cli
export CLAUDE_CLI_MODEL=sonnet

# --- investigator on flat-rate Sonnet (SOT-2627) ---
export RAG_INVESTIGATOR_BACKEND=claude-mcp
export RAG_CLAUDE_MCP_MODEL=sonnet
export RAG_CLAUDE_MCP_RESUME=artifacts/gold100_sonnet_cycle1_resume.jsonl
export RAG_MCP_TOOL_LOG=artifacts/gold100_sonnet_cycle1_tool_calls.jsonl

echo "=== SOT-2648 Sonnet gold100 cycle1 (official:false) start $(date -u +%FT%TZ) ==="
echo "BACKEND=$RAG_INVESTIGATOR_BACKEND MODEL=$RAG_CLAUDE_MCP_MODEL FACT_LAYER=$RAG_FACT_LAYER FORBID_GEMINI=$RAG_FORBID_GEMINI TEXT_PROVIDER=$LLM_PROVIDER"
echo "RESUME=$RAG_CLAUDE_MCP_RESUME"
.venv/bin/python -m scoring.gold_offline --run --workers 1 --no-official \
  --out artifacts/gold100_sonnet_cycle1.json
echo "=== SOT-2648 Sonnet gold100 cycle1 done $(date -u +%FT%TZ) exit=$? ==="
