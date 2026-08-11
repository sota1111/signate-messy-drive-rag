#!/usr/bin/env bash
# SOT-2650 — Sonnet gold100 improvement cycle 3 (official:false lane).
#
# Base env = scripts/sonnet_gold_cycle2.sh (champion Wave A + B1, investigator on flat-rate Sonnet via
# RAG_INVESTIGATOR_BACKEND=claude-mcp, RAG_FACT_LAYER=1, RAG_FORBID_GEMINI=1) with the cycle-3 levers:
#   1. RAG_OCR_STORE=1 — SOT-2650 persisted build-time OCR of the 18 effectively-image-only PDFs
#      (みなみ野/白峰/東都 会議録・報告資料, 青潮/みなみ野 最終報告, 青葉バイオ 提案書PDF …). The live
#      RAG_PDF_OCR fallback silently no-ops under RAG_FORBID_GEMINI, so this store is the ONLY route to
#      scanned-PDF evidence on the Sonnet lane — targets abstains idx18/28/34/45/50/68/70/93/99 class.
#   2. RAG_FORMAT_STRIP_PAREN=1 — value-preserving 括弧内付加情報 strip at the claude-mcp answer
#      boundary (idx4/8/12/41/52/59/84/87/88 wrong class; fail-closed annotation whitelist).
#   3. In-code (no new flag): case_master proposal/FR amount coverage 10/10 + fact-layer amount-diff
#      enumeration lane (idx67), diff_store notebook lane for the ipynb pair (idx22).
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
# paths raise instead of billing.
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

# --- cycle-2 levers (kept) ---
export RAG_FACT_LAYER=1
export RAG_FORBID_GEMINI=1
export LLM_PROVIDER=claude-cli
export CLAUDE_CLI_MODEL=sonnet

# --- cycle-3 levers (SOT-2650) ---
export RAG_OCR_STORE=1
export RAG_FORMAT_STRIP_PAREN=1

# --- investigator on flat-rate Sonnet (SOT-2627) ---
export RAG_INVESTIGATOR_BACKEND=claude-mcp
export RAG_CLAUDE_MCP_MODEL=sonnet
export RAG_CLAUDE_MCP_RESUME=artifacts/gold100_sonnet_cycle3_resume.jsonl
export RAG_MCP_TOOL_LOG=artifacts/gold100_sonnet_cycle3_tool_calls.jsonl

echo "=== SOT-2650 Sonnet gold100 cycle3 (official:false) start $(date -u +%FT%TZ) ==="
echo "BACKEND=$RAG_INVESTIGATOR_BACKEND MODEL=$RAG_CLAUDE_MCP_MODEL FACT_LAYER=$RAG_FACT_LAYER OCR_STORE=$RAG_OCR_STORE STRIP_PAREN=$RAG_FORMAT_STRIP_PAREN FORBID_GEMINI=$RAG_FORBID_GEMINI"
echo "RESUME=$RAG_CLAUDE_MCP_RESUME"
.venv/bin/python -m scoring.gold_offline --run --workers 1 --no-official \
  --out artifacts/gold100_sonnet_cycle3.json
echo "=== SOT-2650 Sonnet gold100 cycle3 done $(date -u +%FT%TZ) exit=$? ==="
