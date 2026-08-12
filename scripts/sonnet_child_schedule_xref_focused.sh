#!/usr/bin/env bash
# SOT-2680 — focused gate for cycle6 K4 (スケジュール/ID/体制クロス参照, official:false).
#
# Base = scripts/sonnet_gold_cycle5.sh (net59 実証構成) PLUS the one K4 lever under test:
#   RAG_SCHEDULE_STORE=1 — スケジュール/ID/体制クロス参照ストア(schedule_store)を serve へ配線する決定論
#     直答レーン(+補助ツール schedule_lookup):
#     (idx92) マイルストーンID/タスクID/アクションID の 3 種別 distinct 合計(非マークダウン出典)。
#     (idx96) チェックポイント N → 関連タスクID(CP→直接T ∪ CP→MS→関連T の事前確定チェーン)。
#     (idx94) マイルストーン N × 役職 → タスクID(案件スコープ roster で役職フィルタ; 役割の案件横断流用禁止)。
#     (idx72) 役職 → 担当タスク数。serve gated・既定 OFF ⇒ OFF 時 byte-identical。
# Targets (gold=artifacts/predictions_test_v3_final.csv): 一次 idx92/94/96, 二次 idx34/72.
# Gate FAILS on any Sonnet sentinel regression. --dev ⇒ official:false (non-flash stack; NOT a promotion
# basis). RAG_CLAUDE_MCP_RESUME=0 必須: serve change alters answers, so target AND sentinels re-derive
# live (SOT-2664 resume-replay 罠).
set -euo pipefail
cd /workspaces/signate-messy-drive-rag

.venv/bin/python scripts/check_flag_manifest.py "$0" || exit 1

# Rebuild the question-independent stores the base config + this axis rely on (LLM-free, idempotent).
# schedule_store is (re)built because the new lane reads its id_counts / ms_tasks / checkpoints / roster.
.venv/bin/python -m src.rag.index.schedule_store
.venv/bin/python -m src.rag.index.derived_metrics
.venv/bin/python -m src.rag.index.visual_store
.venv/bin/python -m src.rag.index.action_row_store
.venv/bin/python -m src.rag.index.case_finance_store
.venv/bin/python -c "from src.rag.index import report_attr_store as s; r=s.build()['report']; print('[build] report_attr_store', r)"
.venv/bin/python -c "from src.rag.index import diff_store as s; r=s.build(); print('[build] diff_store', r.get('pairs') if isinstance(r,dict) else r)"
HOME="${HOME:-/tmp}" .venv/bin/python -m src.rag.index.heading_page_store

export VERTEX_LOCATION=global
export GEN_MODEL=gemini-3.6-flash
export GEN_MODEL_HARD=gemini-3.6-flash
export VISION_MODEL=gemini-3.6-flash
export PYTHONPATH=/tmp/genai_patch:.

# --- champion Wave A + B1 flags (verbatim from scripts/sonnet_gold_cycle5.sh) ---
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

# --- cycle-4.5 manual interim (Cerebras stack + bare-answer) ---
export RAG_TEXT_FTS=1
export RAG_UNIFIED_SEARCH=1
export RAG_PLAN_FANOUT=1
export RAG_BARE_ANSWER=1

# --- cycle-5 promoted children (base net59) ---
export RAG_FANOUT_FINISHER=1
export RAG_FANOUT_FINISHER_MAX=1
export RAG_VDIFF_SUBJECT=1
export RAG_NONE_BARE=1
export RAG_EXACT_LABEL=1
export RAG_HEADING_PAGE_STORE=1

# --- axis under test (SOT-2680, cycle6 K4) ---
export RAG_SCHEDULE_STORE=1

# --- investigator on flat-rate Sonnet; parallelism 1 = shared-limit protection ---
export RAG_INVESTIGATOR_BACKEND=claude-mcp
export RAG_CLAUDE_MCP_MODEL=sonnet
# MANDATORY: no resume cache — the serve change alters answers, so re-derive live (SOT-2664).
export RAG_CLAUDE_MCP_RESUME=0
export RAG_MCP_TOOL_LOG=artifacts/sonnet_child_schedule_xref_tool_calls.jsonl

TARGET="${TARGET:-92,94,96,34,72}"

echo "=== SOT-2680 Sonnet K4 schedule-xref focused gate start $(date -u +%FT%TZ) ==="
echo "BACKEND=$RAG_INVESTIGATOR_BACKEND SCHEDULE_STORE=$RAG_SCHEDULE_STORE RESUME=$RAG_CLAUDE_MCP_RESUME TARGET=$TARGET"
.venv/bin/python scripts/run_focused_gate.py \
  --label sot_child_schedule_xref \
  --target "$TARGET" \
  --sentinels scripts/sonnet_sentinels.json \
  --issue SOT-2680 \
  --dev --no-smoke \
  --workers 1
echo "=== SOT-2680 Sonnet K4 schedule-xref focused gate done $(date -u +%FT%TZ) exit=$? ==="
