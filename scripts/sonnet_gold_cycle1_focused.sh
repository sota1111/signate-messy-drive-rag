#!/usr/bin/env bash
# SOT-2648 — focused gate for Sonnet gold100 cycle 1 (official:false lane).
#
# Same env as scripts/sonnet_gold_cycle1.sh (claude-mcp investigator + RAG_FACT_LAYER=1 +
# RAG_FORBID_GEMINI=1 + LLM_PROVIDER=claude-cli), run over the SONNET sentinel set
# (scripts/sonnet_sentinels.json, SOT-2648) + the cycle-1 abstain targets:
#   idx38/87 (enum, case_master) / idx57/63 (derived, derived_metrics) — the four hard-core
#   abstains whose fact-layer coverage SOT-2647 already verified on the flash lane.
# Gate FAILS on any Sonnet sentinel regression. --dev ⇒ official:false (non-flash stack).
# --no-smoke: the smoke probe talks to Gemini directly, which RAG_FORBID_GEMINI forbids.
set -euo pipefail
cd /workspaces/signate-messy-drive-rag

.venv/bin/python scripts/check_flag_manifest.py "$0" || exit 1

export VERTEX_LOCATION=global
export GEN_MODEL=gemini-3.6-flash
export GEN_MODEL_HARD=gemini-3.6-flash
export VISION_MODEL=gemini-3.6-flash
export PYTHONPATH=/tmp/genai_patch:.

# --- champion Wave A + B1 flags, verbatim from SOT-2628/sonnet_gold_cycle1.sh ---
export RAG_FIRST_MOVE_ROUTING=1 RAG_SPIN_DETECTION=1 RAG_ADAPTIVE_BUDGET=1 RAG_EVIDENCE_CACHE=1
export RAG_BUDGET_BOUNDARY_RESEARCH=1 RAG_UNANSWERABLE_FALLBACK=1 RAG_PDF_OCR=1 RAG_SHARE_CORPUS_PROFILE=1
export RAG_CANONICAL_MANIFEST=1 RAG_EVIDENCE_INDEX=1 RAG_STRUCTURE_STORE=1
export RAG_GRANULARITY_NORMALIZATION=1 RAG_XLSX_EMBEDDED_IMAGE=1 RAG_CONFLICT_RESOLUTION=1 GATE_EXEC_CORRECT=1
export RAG_NUMERIC_FEATURE_CORR=1 RAG_RELEVANCE_STRICT=1 RAG_HIGHLIGHT_EXTRA=1 RAG_FONT_EMPHASIS=1
export RAG_FILE_GREP_INDEX_CANDIDATES=1
export RAG_FORMAT_EVENTS=1
export RAG_DET_PIPELINE_ROUTER=1
export RAG_DET_PIPELINE_B1=1
export RAG_DET_PIPELINE_B2=0

# --- cycle-1 levers ---
export RAG_FACT_LAYER=1
export RAG_FORBID_GEMINI=1
export LLM_PROVIDER=claude-cli
export CLAUDE_CLI_MODEL=sonnet

# --- investigator on flat-rate Sonnet (SOT-2627); parallelism 1 = shared-limit protection ---
export RAG_INVESTIGATOR_BACKEND=claude-mcp
export RAG_CLAUDE_MCP_MODEL=sonnet
export RAG_CLAUDE_MCP_RESUME=artifacts/sonnet_cycle1_focused_resume.jsonl
export RAG_MCP_TOOL_LOG=artifacts/sonnet_cycle1_focused_tool_calls.jsonl

TARGET="38,57,63,87"

echo "=== SOT-2648 Sonnet cycle1 focused gate start $(date -u +%FT%TZ) ==="
echo "BACKEND=$RAG_INVESTIGATOR_BACKEND FACT_LAYER=$RAG_FACT_LAYER FORBID_GEMINI=$RAG_FORBID_GEMINI TARGET=$TARGET"
.venv/bin/python scripts/run_focused_gate.py \
  --label sot2648_sonnet_cycle1 \
  --target "$TARGET" \
  --sentinels scripts/sonnet_sentinels.json \
  --issue SOT-2648 \
  --dev --no-smoke \
  --workers 1
echo "=== SOT-2648 Sonnet cycle1 focused gate done $(date -u +%FT%TZ) exit=$? ==="
