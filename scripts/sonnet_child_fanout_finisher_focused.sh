#!/usr/bin/env bash
# SOT-2664 — focused gate for cycle5 C1 (bounded plan-fanout finisher, official:false).
#
# Base = scripts/sonnet_gold_manual_20260812.sh (cycle4 net48 実証済み構成: champion Wave A+B1 + cycle-4
# stores + Cerebras stack + RAG_PLAN_FANOUT + RAG_BARE_ANSWER) PLUS the single C1 lever under test:
#   RAG_FANOUT_FINISHER=1  — when RAG_PLAN_FANOUT's 5-turn stage budget is exhausted but the model is one
#                            already-located targeted raw-evidence read away from an answer, the MCP server
#                            grants up to RAG_FANOUT_FINISHER_MAX (default 3) EXTRA such reads instead of
#                            refusing with budget_exhausted. Only the targeted readers (read_office /
#                            format_events / highlight_extract / pdf_emphasis / compute / …) with a resolved
#                            file target qualify; every exploratory/search/resolve tool keeps hitting
#                            budget_exhausted, so the finisher cannot re-open the wandering exploration that
#                            turns abstains into wrong answers. serve-path gated, default OFF ⇒ OFF byte-id.
#
# 一次対象 (churn: 次ツール+対象ファイル特定済みのまま予算切れ) idx16/49/75/83.
# precision ガード (wrong→abstain 転換組; Incorrect へ逆流しないこと) idx11/24/36/48/77/95/96.
# gold=artifacts/predictions_test_v3_final.csv. Gate FAILS on any Sonnet sentinel regression only (the
# target/guard idx may legitimately still be non-MATCH). --dev ⇒ official:false (non-flash stack; NOT a
# promotion basis). NO resume cache: the serve behaviour changes, so target AND sentinels re-derive live.
set -euo pipefail
cd /workspaces/signate-messy-drive-rag

.venv/bin/python scripts/check_flag_manifest.py "$0" || exit 1

# Rebuild the question-independent stores the base config relies on (LLM-free, idempotent).
.venv/bin/python -m src.rag.index.visual_store
.venv/bin/python -m src.rag.index.action_row_store
.venv/bin/python -m src.rag.index.case_finance_store
.venv/bin/python -c "from src.rag.index import report_attr_store as s; r=s.build()['report']; print('[build] report_attr_store', r)"

export VERTEX_LOCATION=global
export GEN_MODEL=gemini-3.6-flash
export GEN_MODEL_HARD=gemini-3.6-flash
export VISION_MODEL=gemini-3.6-flash
export PYTHONPATH=/tmp/genai_patch:.

# --- champion Wave A + B1 flags, verbatim from scripts/sonnet_gold_manual_20260812.sh ---
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

# --- axis under test (SOT-2664, cycle5 C1) ---
export RAG_FANOUT_FINISHER=1
export RAG_FANOUT_FINISHER_MAX="${RAG_FANOUT_FINISHER_MAX:-3}"

# --- investigator on flat-rate Sonnet; parallelism 1 = shared-limit protection ---
export RAG_INVESTIGATOR_BACKEND=claude-mcp
export RAG_CLAUDE_MCP_MODEL=sonnet
export RAG_MCP_TOOL_LOG=artifacts/sonnet_child_fanout_finisher_tool_calls.jsonl

# 一次 churn 4 + precision ガード 7 (Incorrect へ逆流しないこと)。
TARGET="${TARGET:-16,49,75,83,11,24,36,48,77,95,96}"

echo "=== SOT-2664 Sonnet C1 fanout-finisher focused gate start $(date -u +%FT%TZ) ==="
echo "BACKEND=$RAG_INVESTIGATOR_BACKEND FANOUT_FINISHER=$RAG_FANOUT_FINISHER MAX=$RAG_FANOUT_FINISHER_MAX TARGET=$TARGET"
.venv/bin/python scripts/run_focused_gate.py \
  --label sot_child_fanout_finisher \
  --target "$TARGET" \
  --sentinels scripts/sonnet_sentinels.json \
  --issue SOT-2664 \
  --dev --no-smoke \
  --workers 1
echo "=== SOT-2664 Sonnet C1 fanout-finisher focused gate done $(date -u +%FT%TZ) exit=$? ==="
