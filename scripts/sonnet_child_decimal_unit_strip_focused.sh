#!/usr/bin/env bash
# SOT-2682 — focused gate for cycle6 K6 (小数指定問の単位strip書式契約, official:false).
#
# Base = scripts/sonnet_gold_cycle5.sh (net59 実証済み構成: champion Wave A+B1 + cycle-4 stores +
# Cerebras stack + RAG_BARE_ANSWER + cycle5 promoted levers) PLUS the single K6 lever under test:
#   RAG_DECIMAL_UNIT_STRIP=1 — deterministic value-preserving post-format at the claude-mcp backend
#                              boundary. 「小数第N位で答えて」書式指定を持つ問いへの回答末尾に付いた
#                              単位サフィックスだけを落とす (idx79「池田 直哉、7.00時間/タスク」→
#                              gold「池田 直哉、7.00」)。数値トークンは保存。値の変更は一切しない。
#                              全gold100較正: 小数第N位指定問の gold は例外なく裸数値なので fix か no-op。
# 一次対象 (gold=artifacts/predictions_test_v3_final.csv): idx79 (二次観測: 78)。
# Gate FAILS on any Sonnet sentinel regression (the target idx may legitimately still be non-MATCH on a
# single stochastic serve). --dev ⇒ official:false (non-flash stack; NOT a promotion basis). NO resume
# cache (SOT-2664 gotcha: claude-mcp resume key is (model,question), config非依存 → stale replay罠).
#
# STRIP_ON env selects ON (default, RAG_DECIMAL_UNIT_STRIP=1) vs OFF-control (STRIP_ON=0).
set -euo pipefail
cd /workspaces/signate-messy-drive-rag

.venv/bin/python scripts/check_flag_manifest.py "$0" || exit 1

# Rebuild the question-independent stores the base config relies on (LLM-free, idempotent).
.venv/bin/python -m src.rag.index.visual_store
.venv/bin/python -m src.rag.index.action_row_store
.venv/bin/python -m src.rag.index.case_finance_store
.venv/bin/python -c "from src.rag.index import report_attr_store as s; r=s.build()['report']; print('[build] report_attr_store', r)"
HOME="${HOME:-/tmp}" .venv/bin/python -m src.rag.index.heading_page_store

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

# --- cycle-2/3 levers (kept) ---
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

# --- cycle4.5 manual interim (Cerebras stack + bare-answer) ---
export RAG_TEXT_FTS=1
export RAG_UNIFIED_SEARCH=1
export RAG_PLAN_FANOUT=1
export RAG_BARE_ANSWER=1

# --- cycle-5 promoted levers ---
export RAG_FANOUT_FINISHER=1
export RAG_FANOUT_FINISHER_MAX=1
export RAG_VDIFF_SUBJECT=1
export RAG_NONE_BARE=1
export RAG_EXACT_LABEL=1
export RAG_HEADING_PAGE_STORE=1

# --- axis under test (SOT-2682, cycle6 K6) ---
export RAG_DECIMAL_UNIT_STRIP="${STRIP_ON:-1}"

# --- investigator on flat-rate Sonnet; parallelism 1 = shared-limit protection ---
export RAG_INVESTIGATOR_BACKEND=claude-mcp
export RAG_CLAUDE_MCP_MODEL=sonnet
export RAG_MCP_TOOL_LOG=artifacts/sonnet_child_decimal_unit_strip_tool_calls.jsonl
export RAG_CLAUDE_MCP_RESUME=0

TARGET="${TARGET:-79,78}"
LABEL="${LABEL:-sot_child_decimal_unit_strip}"

echo "=== SOT-2682 Sonnet K6 decimal-unit-strip focused gate start $(date -u +%FT%TZ) ==="
echo "BACKEND=$RAG_INVESTIGATOR_BACKEND DECIMAL_UNIT_STRIP=$RAG_DECIMAL_UNIT_STRIP TARGET=$TARGET LABEL=$LABEL"
.venv/bin/python scripts/run_focused_gate.py \
  --label "$LABEL" \
  --target "$TARGET" \
  --sentinels scripts/sonnet_sentinels.json \
  --issue SOT-2682 \
  --dev --no-smoke \
  --workers 1
echo "=== SOT-2682 Sonnet K6 focused gate done $(date -u +%FT%TZ) exit=$? ==="
