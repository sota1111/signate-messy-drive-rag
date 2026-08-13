#!/usr/bin/env bash
# SOT-2697 (cycle9) — case_finance 汎用金額差額レーン focused gate (official:false lane).
#
# Base env = the cycle-7/8 integrated config PLUS this child's lever, default OFF ⇒ OFF byte-identical:
#   RAG_CASE_FINANCE_DIFF=1 — 「提案時の税込み見込み金額と最終請求金額の差額」型の単純差額を決定論回答:
#       idx6 = 蒼泉会 ひがし丘総合病院: 見込税込 4,675,000 と確定税込 4,675,000（store 既存オペランド）の
#         差額 → gold「0円」。案件バインドは質問中トークンの白書きではなく store 全レコードの case_id/略称
#         マッチ（質問非依存・全案件対象）。fail-closed（両金額が整数円で存在するときのみ回答）。gold 非依存。
#
# 番兵: scripts/sonnet_sentinels.json (idx58 版)。Gate FAILS on any sentinel regression (10/10)。
# GOLD = artifacts/predictions_test_v3_final.csv (sonnet_sentinels と同一)。gold ハードコードなし。
#
# --dev ⇒ official:false (昇格根拠にしない)。RAG_CLAUDE_MCP_RESUME=0 (本レバーは回答を変えるので
#   sidecar replay は変化を隠す)。Gemini $0 (RAG_FORBID_GEMINI=1)。
set -euo pipefail
cd /workspaces/signate-messy-drive-rag

.venv/bin/python scripts/check_flag_manifest.py "$0" || exit 1

# Rebuild the question-independent stores (LLM-free, idempotent). case_finance store は既に SOHK 行に
# estimate/confirmed 両金額を保持（親調査で機械確認済み）; 本子は serve 側レーン追加のみ。
.venv/bin/python -m src.rag.index.derived_metrics
.venv/bin/python -m src.rag.index.raw_artifact_store
.venv/bin/python -m src.rag.index.doc_reach_store
.venv/bin/python -m src.rag.index.schedule_store
.venv/bin/python -m src.rag.index.visual_store
.venv/bin/python -m src.rag.index.action_row_store
.venv/bin/python -m src.rag.index.case_finance_store
.venv/bin/python -c "from src.rag.index import report_attr_store as s; r=s.build()['report']; print('[build] report_attr_store', r)"
RAG_VDIFF_CLASSIFY=1 .venv/bin/python -c "from src.rag.index import diff_store as s; r=s.build(); print('[build] diff_store (RAG_VDIFF_CLASSIFY=1)', r.get('pairs') if isinstance(r,dict) else r)"
HOME="${HOME:-/tmp}" .venv/bin/python -m src.rag.index.heading_page_store
.venv/bin/python -m src.rag.index.xlsx_formula_trace
RAG_FORMAT_EVENTS=1 .venv/bin/python -m src.rag.index.format_series_store
.venv/bin/python -m src.rag.index.rate_table_store
.venv/bin/python -m src.rag.index.formula_apply_store
RAG_OCR_STORE=1 RAG_IMAGE_OCR_STORE=1 .venv/bin/python -c "from src.rag.index import text_fts as t; r=t.build(); print('[build] text_fts', {k:r.get(k) for k in ('records','docs')})"
RAG_OCR_STORE=1 RAG_IMAGE_OCR_STORE=1 .venv/bin/python -c "from src.rag.index import evidence_index as e; print('[build] evidence_index', e.build_only())"

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

# --- cycle-4 levers ---
export RAG_ACTION_ROW_STORE=1
export RAG_VISUAL_STORE=1
export RAG_CASE_FINANCE_STORE=1
export RAG_REPORT_ATTR_STORE=1
export RAG_FORMAT_VALUE_NORM=1

# --- cycle-4.5 manual interim ---
export RAG_TEXT_FTS=1
export RAG_UNIFIED_SEARCH=1
export RAG_PLAN_FANOUT=1
export RAG_BARE_ANSWER=1

# --- cycle-5 levers ---
export RAG_FANOUT_FINISHER=1
export RAG_FANOUT_FINISHER_MAX=1
export RAG_VDIFF_SUBJECT=1
export RAG_NONE_BARE=1
export RAG_EXACT_LABEL=1
export RAG_HEADING_PAGE_STORE=1

# --- cycle-6 levers ---
export RAG_DOC_REACH_STORE=1
export RAG_RAW_ARTIFACT_STORE=1
export RAG_DERIVED_COVERAGE=1
export RAG_SCHEDULE_STORE=1
export RAG_VDIFF_NORMALIZE=1
export RAG_DECIMAL_UNIT_STRIP=1

# --- cycle-7 levers ---
export RAG_IMAGE_OCR_STORE=1
export RAG_NB_CHART_STORE=1
export RAG_XLSX_FORMULA_TRACE=1
export RAG_XREF_COVERAGE=1
export RAG_CORR_SIGN=1
export RAG_BIN_RANGE_FORMAT=1
export RAG_SPECIAL_PROVISION=1

# --- cycle-8 levers (integrated baseline) ---
export RAG_VDIFF_CLASSIFY=1
export RAG_REPORT_SERIES=1
export RAG_FORMAT_SERIES=1
export RAG_RATE_TABLE=1
export RAG_FORMULA_APPLY=1

# --- cycle-9 lever (this child) ---
export RAG_CASE_FINANCE_DIFF=1

# --- investigator on flat-rate Sonnet; parallelism 1 = shared-limit protection ---
export RAG_INVESTIGATOR_BACKEND=claude-mcp
export RAG_CLAUDE_MCP_MODEL=sonnet
export RAG_MCP_TOOL_LOG=artifacts/sot2697_tool_calls.jsonl
export RAG_CLAUDE_MCP_RESUME=0

TARGET="${TARGET:-6}"
GOLD="${GOLD:-artifacts/predictions_test_v3_final.csv}"

echo "=== SOT-2697 cycle9 case_finance diff focused gate start $(date -u +%FT%TZ) ==="
echo "BACKEND=$RAG_INVESTIGATOR_BACKEND CASE_FINANCE_DIFF=$RAG_CASE_FINANCE_DIFF RESUME=$RAG_CLAUDE_MCP_RESUME TARGET=$TARGET GOLD=$GOLD"
.venv/bin/python scripts/run_focused_gate.py \
  --label sot2697_case_finance_diff \
  --target "$TARGET" \
  --sentinels scripts/sonnet_sentinels.json \
  --gold "$GOLD" \
  --issue SOT-2697 \
  --dev --no-smoke \
  --workers 1
echo "=== SOT-2697 cycle9 case_finance diff focused gate done $(date -u +%FT%TZ) exit=$? ==="
