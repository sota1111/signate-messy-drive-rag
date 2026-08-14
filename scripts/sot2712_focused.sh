#!/usr/bin/env bash
# SOT-2712 (cycle11) — 版ペア差分 direct-commit の対象クラス拡大 focused gate (official:false lane).
#
# Base env = SOT-2706 のフラグ一式（RAG_VDIFF_DIRECT_COMMIT=1）＋本子の新サブフラグ（既定 OFF・分離）:
#   RAG_VDIFF_DC_COLRENAME=1 — 列名変更クラスの direct-commit（idx14: 提案書_v1→_v3 の schema_name_change
#     群 interest_rate 等を old/new 実物から質問非依存に集約し裸形式で逐語コミット）。
#   RAG_VDIFF_DC_NOCHANGE=1  — 実質変更ゼロの「該当なし」verdict の direct-commit（idx9: 最終報告 old→最新は
#     全変更が LAYOUT/SURFACE、SUBSTANTIVE 0 件 ⇒ 裸形式「該当なし」）。
#   いずれもサーブ時のみ（store は再ビルド不要・byte-identical）、fact_layer.resolve 末尾, route=deterministic。
#   両サブフラグ OFF なら SOT-2706 と完全同一（byte-identical）。gold 文言のハードコードなし。
#
# Targets: idx9（該当なし verdict）+ idx14（列名変更クラス）。
# Guards : idx22（白峰 01_eda 記述統計逐語）+ idx95（青嶺 スケジュール 非劣化）。
# 番兵   : scripts/sonnet_sentinels.json。Gate FAILS on any sentinel regression (10/10)。
# GOLD   = artifacts/predictions_test_v3_final.csv。gold ハードコードなし（summary は old/new 実物由来）。
#   --dev ⇒ official:false。RAG_CLAUDE_MCP_RESUME=0（回答が変わるレバーなので sidecar replay は変化を隠す）。
#   Gemini $0（RAG_FORBID_GEMINI=1）。
set -euo pipefail
cd /workspaces/signate-messy-drive-rag

.venv/bin/python scripts/check_flag_manifest.py "$0" || exit 1

# Rebuild the diff store WITH the new flag so office table-deletion records carry the semantic-frame
# summary (idx1). Notebook (idx22) vision output is reused from the prior store (RAG_OCR_STORE_BUILD off).
# RAG_VDIFF_STRUCT/CLASSIFY are kept to preserve SOT-2700's structural repairs. LLM-free, idempotent.
RAG_VDIFF_STRUCT=1 RAG_VDIFF_CLASSIFY=1 RAG_VDIFF_DIRECT_COMMIT=1 RAG_FORBID_GEMINI=1 \
  .venv/bin/python -c "from src.rag.index import diff_store as s; r=s.build(); print('[build] diff_store', r.get('pairs') if isinstance(r,dict) else r)"

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

# --- cycle-8 levers ---
export RAG_ANALYSIS_XREF=1
export RAG_PLAN_COVERAGE=1
export RAG_FORMAT_SERIES=1
export RAG_RATE_TABLE=1
export RAG_FORMULA_APPLY=1
export RAG_VDIFF_CLASSIFY=1
export RAG_REPORT_SERIES=1

# --- cycle-9 levers ---
export RAG_CASE_FINANCE_DIFF=1
export RAG_ANALYSIS_METRICS_ENUM=1
export RAG_ANALYSIS_CONFIG_HYPERPARAM=1
export RAG_STAGED_METRICS=1
export RAG_DERIVED_RANKING=1
export RAG_VDIFF_STRUCT=1

# --- cycle-10 lever ---
export RAG_VDIFF_DIRECT_COMMIT=1

# --- cycle-11 lever (this child): direct-commit の対象クラス拡大（サブフラグ既定 OFF・分離） ---
export RAG_VDIFF_DC_COLRENAME=1
export RAG_VDIFF_DC_NOCHANGE=1

# --- investigator on flat-rate Sonnet; parallelism 1 = shared-limit protection ---
export RAG_INVESTIGATOR_BACKEND=claude-mcp
export RAG_CLAUDE_MCP_MODEL=sonnet
export RAG_MCP_TOOL_LOG=artifacts/sot2712_tool_calls.jsonl
export RAG_CLAUDE_MCP_RESUME=0

TARGET="${TARGET:-9,14,22,95}"
GOLD="${GOLD:-artifacts/predictions_test_v3_final.csv}"

echo "=== SOT-2712 cycle11 vdiff-direct-commit focused gate start $(date -u +%FT%TZ) ==="
echo "BACKEND=$RAG_INVESTIGATOR_BACKEND DIRECT_COMMIT=$RAG_VDIFF_DIRECT_COMMIT RESUME=$RAG_CLAUDE_MCP_RESUME TARGET=$TARGET GOLD=$GOLD"
.venv/bin/python scripts/run_focused_gate.py \
  --label sot2712_vdiff_direct_commit \
  --target "$TARGET" \
  --sentinels scripts/sonnet_sentinels.json \
  --gold "$GOLD" \
  --issue SOT-2712 \
  --dev --no-smoke \
  --workers 1
echo "=== SOT-2712 cycle11 vdiff-direct-commit focused gate done $(date -u +%FT%TZ) exit=$? ==="
