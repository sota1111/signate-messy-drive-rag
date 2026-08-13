#!/usr/bin/env bash
# SOT-2685 — focused gate for cycle7 K2 (ノートブック描画チャート読み取りストア, official:false).
#
# Base = scripts/sonnet_child_xref_coverage_focused.sh (cycle7 K4 構成) PLUS the one K2 lever:
#   RAG_NB_CHART_STORE=1 — build 時に vision で焼いた notebook チャートの描画属性ストアを serve へ質問非依存に
#     引く決定論直答レーン(+補助ツール nb_chart_lookup):
#     (idx56) 目的変数分析の y軸に実際に印字されている目盛りの最大値(全 vision サンプル一致の時のみ確定 = 1200)。
#     (idx66) 日付分析で件数が最も高い日 = データ側 day 別件数 argmax(権威; チャートは groupby(day).size() の忠実な
#       描画。vision の 31 点線グラフ peak-x 読みは弱いので cross-check のみ) = 20日。
#     serve gated・既定 OFF ⇒ OFF 時 byte-identical。build のみ Gemini 使用(RAG_NB_CHART_STORE_BUILD)、serve は $0。
# Targets (gold=artifacts/predictions_test_v3_final.csv): 一次 idx56/66.
# Gate FAILS on any Sonnet sentinel regression. --dev ⇒ official:false (non-flash stack; NOT a promotion
# basis). RAG_CLAUDE_MCP_RESUME=0 必須: serve change alters answers, re-derive target AND sentinels live.
set -euo pipefail
cd /workspaces/signate-messy-drive-rag

.venv/bin/python scripts/check_flag_manifest.py "$0" || exit 1

# Rebuild the question-independent stores the base config relies on (LLM-free, idempotent).
# notebook_chart_store is NOT rebuilt here (its build needs vision → RAG_FORBID_GEMINI): the lane reads
# the persisted artifact (artifacts/notebook_chart_store.jsonl) baked at build time.
.venv/bin/python -m src.rag.index.derived_metrics
.venv/bin/python -m src.rag.index.visual_store
.venv/bin/python -m src.rag.index.action_row_store
.venv/bin/python -m src.rag.index.case_finance_store
.venv/bin/python -c "from src.rag.index import raw_artifact_store as s; r=s.build(); print('[build] raw_artifact_store', {k:r[k] for k in ('files','cases')})"
.venv/bin/python -c "from src.rag.index import report_attr_store as s; r=s.build()['report']; print('[build] report_attr_store', r)"
.venv/bin/python -c "from src.rag.index import diff_store as s; r=s.build(); print('[build] diff_store', r.get('pairs') if isinstance(r,dict) else r)"
HOME="${HOME:-/tmp}" .venv/bin/python -m src.rag.index.heading_page_store

export VERTEX_LOCATION=global
export GEN_MODEL=gemini-3.6-flash
export GEN_MODEL_HARD=gemini-3.6-flash
export VISION_MODEL=gemini-3.6-flash
export PYTHONPATH=/tmp/genai_patch:.

# --- champion Wave A + B1 flags (verbatim from scripts/sonnet_child_wrong_det_focused.sh) ---
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

# --- cycle-4 levers ---
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

# --- cycle-5 promoted children ---
export RAG_FANOUT_FINISHER=1
export RAG_FANOUT_FINISHER_MAX=1
export RAG_VDIFF_SUBJECT=1
export RAG_NONE_BARE=1
export RAG_EXACT_LABEL=1
export RAG_HEADING_PAGE_STORE=1

# --- cycle-6 promoted children ---
export RAG_DERIVED_COVERAGE=1
export RAG_VDIFF_NORMALIZE=1

# --- cycle-7 K5 promoted (wrong 決定論修正) ---
export RAG_CORR_SIGN=1
export RAG_BIN_RANGE_FORMAT=1
export RAG_SPECIAL_PROVISION=1

# --- cycle-7 K4 promoted (クロス参照カバレッジ) ---
export RAG_XREF_COVERAGE=1

# --- axis under test (SOT-2685, cycle7 K2) ---
export RAG_NB_CHART_STORE=1

# --- investigator on flat-rate Sonnet; parallelism 1 = shared-limit protection ---
export RAG_INVESTIGATOR_BACKEND=claude-mcp
export RAG_CLAUDE_MCP_MODEL=sonnet
# MANDATORY: no resume cache — the serve change alters answers, so re-derive live (SOT-2664).
export RAG_CLAUDE_MCP_RESUME=0
export RAG_MCP_TOOL_LOG=artifacts/sonnet_child_nb_chart_tool_calls.jsonl

TARGET="${TARGET:-56,66}"

echo "=== SOT-2685 Sonnet K2 nb-chart focused gate start $(date -u +%FT%TZ) ==="
echo "BACKEND=$RAG_INVESTIGATOR_BACKEND NB_CHART_STORE=$RAG_NB_CHART_STORE RESUME=$RAG_CLAUDE_MCP_RESUME TARGET=$TARGET"
.venv/bin/python scripts/run_focused_gate.py \
  --label sot_child_nb_chart \
  --target "$TARGET" \
  --sentinels scripts/sonnet_sentinels.json \
  --issue SOT-2685 \
  --dev --no-smoke \
  --workers 1
echo "=== SOT-2685 Sonnet K2 nb-chart focused gate done $(date -u +%FT%TZ) exit=$? ==="
