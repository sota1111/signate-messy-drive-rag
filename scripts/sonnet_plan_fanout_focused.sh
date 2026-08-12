#!/usr/bin/env bash
# SOT-2661 — Sonnet planner -> parallel fan-out -> synthesis focused gate (official:false).
set -euo pipefail
cd /workspaces/signate-messy-drive-rag

.venv/bin/python scripts/check_flag_manifest.py "$0" || exit 1

export VERTEX_LOCATION=global
export GEN_MODEL=gemini-3.6-flash GEN_MODEL_HARD=gemini-3.6-flash VISION_MODEL=gemini-3.6-flash
export PYTHONPATH=/tmp/genai_patch:.
export RAG_FIRST_MOVE_ROUTING=1 RAG_SPIN_DETECTION=1 RAG_ADAPTIVE_BUDGET=1 RAG_EVIDENCE_CACHE=1
export RAG_BUDGET_BOUNDARY_RESEARCH=1 RAG_UNANSWERABLE_FALLBACK=1 RAG_PDF_OCR=1 RAG_SHARE_CORPUS_PROFILE=1
export RAG_CANONICAL_MANIFEST=1 RAG_EVIDENCE_INDEX=1 RAG_STRUCTURE_STORE=1
export RAG_GRANULARITY_NORMALIZATION=1 RAG_XLSX_EMBEDDED_IMAGE=1 RAG_CONFLICT_RESOLUTION=1 GATE_EXEC_CORRECT=1
export RAG_NUMERIC_FEATURE_CORR=1 RAG_RELEVANCE_STRICT=1 RAG_HIGHLIGHT_EXTRA=1 RAG_FONT_EMPHASIS=1
export RAG_FILE_GREP_INDEX_CANDIDATES=1 RAG_FORMAT_EVENTS=1
export RAG_DET_PIPELINE_ROUTER=1 RAG_DET_PIPELINE_B1=1 RAG_DET_PIPELINE_B2=0
export RAG_FACT_LAYER=1 RAG_FORBID_GEMINI=1 RAG_OCR_STORE=1 RAG_FORMAT_STRIP_PAREN=1 RAG_ACTION_ROW_STORE=1
export RAG_TEXT_FTS=1 RAG_ID_MASTER=1 RAG_DIFF_STORE=1 RAG_DISTILL_STORE=1 RAG_UNIFIED_SEARCH=1
export RAG_PLAN_FANOUT=1
export LLM_PROVIDER=claude-cli CLAUDE_CLI_MODEL=sonnet
export RAG_INVESTIGATOR_BACKEND=claude-mcp RAG_CLAUDE_MCP_MODEL=sonnet
export RAG_CLAUDE_MCP_RESUME=artifacts/sonnet_plan_fanout_focused_resume_v3.jsonl
export RAG_MCP_TOOL_LOG=artifacts/sonnet_plan_fanout_tool_calls.jsonl

# Representative BUDGET_EXHAUSTED cases. Existing MATCH non-regression is enforced independently by
# the mandatory 10-question sentinel set (including the complex numeric idx30).
TARGET="${TARGET:-34,76,98}"
.venv/bin/python scripts/run_focused_gate.py \
  --label sot2661_plan_fanout --target "$TARGET" \
  --sentinels scripts/sonnet_sentinels.json --issue SOT-2661 \
  --dev --no-smoke --workers 1
