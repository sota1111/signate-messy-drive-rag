#!/usr/bin/env bash
# SOT-2659 — focused gate for the unified fan-out search child (Cerebras 型検索基盤 3/5, official:false).
#
# Same champion env as scripts/sonnet_child_text_fts_focused.sh plus the SOT-2659 lever:
#   RAG_UNIFIED_SEARCH=1 — expose the one-call `search` tool: it fans the query out over every
#                          query-addressable retriever (FTS / 蒸留 / id_master 逆引き / 版差分 / registry)
#                          IN PARALLEL, RRF-fuses (Σ weight/(60+rank)), dedups, caps per file, and
#                          re-expands context — compressing the 12〜18 ターンの逐次ツール往復 (BUDGET32 の
#                          主犯) into ONE turn. LLM-free / Gemini-free at serve time.
# The unified tool composes the upstream deterministic retrievers, so their flags are on too:
#   RAG_TEXT_FTS (FTS index), RAG_ID_MASTER / RAG_DIFF_STORE (fact stores), RAG_DISTILL_STORE (optional).
# Targets: BUDGET-exhausted file_grep-spin abstain idx the fan-out is meant to un-stick in one call.
# Gate FAILS on any Sonnet sentinel regression. --dev ⇒ official:false (non-flash stack, cannot be a
# production non-regression basis; the OFF path is byte-identical so production is unaffected regardless).
# Primary KPI is the per-question average turn count (BUDGET32) — read artifacts/*tool_calls.jsonl.
set -euo pipefail
cd /workspaces/signate-messy-drive-rag

.venv/bin/python scripts/check_flag_manifest.py "$0" || exit 1

export VERTEX_LOCATION=global
export GEN_MODEL=gemini-3.6-flash
export GEN_MODEL_HARD=gemini-3.6-flash
export VISION_MODEL=gemini-3.6-flash
export PYTHONPATH=/tmp/genai_patch:.

# --- champion Wave A + B1 flags, verbatim from sonnet_child_text_fts_focused.sh ---
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

# --- cycle-2/3/4 levers (kept) ---
export RAG_FACT_LAYER=1
export RAG_FORBID_GEMINI=1
export LLM_PROVIDER=claude-cli
export CLAUDE_CLI_MODEL=sonnet
export RAG_OCR_STORE=1
export RAG_FORMAT_STRIP_PAREN=1
export RAG_ACTION_ROW_STORE=1

# --- upstream retrievers the unified search composes ---
export RAG_TEXT_FTS=1
export RAG_ID_MASTER=1
export RAG_DIFF_STORE=1
export RAG_DISTILL_STORE=1

# --- SOT-2659 lever: unified fan-out search + RRF fusion + context re-expansion ---
export RAG_UNIFIED_SEARCH=1

# --- investigator on flat-rate Sonnet (SOT-2627); parallelism 1 = shared-limit protection ---
export RAG_INVESTIGATOR_BACKEND=claude-mcp
export RAG_CLAUDE_MCP_MODEL=sonnet
export RAG_MCP_TOOL_LOG=artifacts/sonnet_child_unified_search_tool_calls.jsonl

TARGET="${TARGET:-34,76,93,98,99}"

echo "=== SOT-2659 Sonnet unified_search focused gate start $(date -u +%FT%TZ) ==="
echo "BACKEND=$RAG_INVESTIGATOR_BACKEND FACT_LAYER=$RAG_FACT_LAYER UNIFIED_SEARCH=$RAG_UNIFIED_SEARCH TARGET=$TARGET"
.venv/bin/python scripts/run_focused_gate.py \
  --label sot_child_unified_search \
  --target "$TARGET" \
  --sentinels scripts/sonnet_sentinels.json \
  --issue SOT-2659 \
  --dev --no-smoke \
  --workers 1
echo "=== SOT-2659 Sonnet unified_search focused gate done $(date -u +%FT%TZ) exit=$? ==="
