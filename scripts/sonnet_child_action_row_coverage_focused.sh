#!/usr/bin/env bash
# SOT-2667 — focused gate for the action-row cross-project COVERAGE extension (cycle5 C4, official:false).
#
# Base config = scripts/sonnet_gold_manual_20260812.sh (net48, official:false) — champion Wave A+B1 +
# cycle2/3/4 levers + the 2026-08-12 manual interim stack (FTS / unified search / plan-fanout / bare
# answer). This child extends src/rag/index/action_row_store.py to also read the OCR *space-separated,
# column-wrapped* action-item table (京橋信用ソリューションズ・蒼樹会 みなみ野 等のスキャンPDF、パイプ表
# 無し) so previously-skipped 案件×ID行 become recorded, and adds a deterministic idx45-type lane
# (2会議録間で Open→完了 に転じたアクションIDの列挙) to src/rag/agent/action_row_lane.py.
#
# Targets:
#   idx45 (enum_set: 会議ID M2→M3 で完了したアクションアイテムID全列挙 — deterministic completed-set lane)
#   idx93 (document_extract: 蒼樹会 A10 の内容をそのまま — action_row_lookup が found:true になり原文
#          リージョンを Evidence として供給。束縛できない内容抽出は棄権のまま=無理な回答化はしない)
# Gate FAILS on any Sonnet sentinel regression. --dev ⇒ official:false (non-flash stack).
set -euo pipefail
cd /workspaces/signate-messy-drive-rag

.venv/bin/python scripts/check_flag_manifest.py "$0" || exit 1

# Rebuild the action-row store (LLM-free, question-independent, idempotent) so the OCR-table coverage
# extension is reflected before the gate reads the serve path.
.venv/bin/python -m src.rag.index.action_row_store

export VERTEX_LOCATION=global
export GEN_MODEL=gemini-3.6-flash
export GEN_MODEL_HARD=gemini-3.6-flash
export VISION_MODEL=gemini-3.6-flash
export PYTHONPATH=/tmp/genai_patch:.

# --- champion Wave A + B1 flags ---
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

# --- cycle-2/3 levers ---
export RAG_FACT_LAYER=1
export RAG_FORBID_GEMINI=1
export LLM_PROVIDER=claude-cli
export CLAUDE_CLI_MODEL=sonnet
export RAG_OCR_STORE=1
export RAG_FORMAT_STRIP_PAREN=1

# --- cycle-4 levers (the five children, combined) ---
export RAG_ACTION_ROW_STORE=1
export RAG_VISUAL_STORE=1
export RAG_CASE_FINANCE_STORE=1
export RAG_REPORT_ATTR_STORE=1
export RAG_FORMAT_VALUE_NORM=1

# --- 2026-08-12 manual interim stack ---
export RAG_TEXT_FTS=1
export RAG_UNIFIED_SEARCH=1
export RAG_PLAN_FANOUT=1
export RAG_BARE_ANSWER=1

# --- investigator on flat-rate Sonnet (SOT-2627); parallelism 1 = shared-limit protection ---
export RAG_INVESTIGATOR_BACKEND=claude-mcp
export RAG_CLAUDE_MCP_MODEL=sonnet
export RAG_CLAUDE_MCP_RESUME=artifacts/sot2667_action_row_coverage_r3_resume.jsonl
export RAG_MCP_TOOL_LOG=artifacts/sot2667_action_row_coverage_r3_tool_calls.jsonl

TARGET="${TARGET:-45,93}"

echo "=== SOT-2667 Sonnet action-row coverage focused gate start $(date -u +%FT%TZ) ==="
echo "BACKEND=$RAG_INVESTIGATOR_BACKEND FACT_LAYER=$RAG_FACT_LAYER ACTION_ROW_STORE=$RAG_ACTION_ROW_STORE TARGET=$TARGET"
.venv/bin/python scripts/run_focused_gate.py \
  --label sot2667_action_row_coverage \
  --target "$TARGET" \
  --sentinels scripts/sonnet_sentinels.json \
  --issue SOT-2667 \
  --dev --no-smoke \
  --workers 1
echo "=== SOT-2667 Sonnet action-row coverage focused gate done $(date -u +%FT%TZ) exit=$? ==="
