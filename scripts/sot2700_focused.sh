#!/usr/bin/env bash
# SOT-2700 (cycle9) — vdiff 構造欠陥の決定論修正 focused gate (official:false lane).
#
# Base env = the cycle-8 INTEGRATED config (net84 実証済み) PLUS one new lever, default OFF ⇒ OFF
# byte-identical:
#   RAG_VDIFF_STRUCT=1 — diffpair / diff_store の決定論 *構造* 修正 (baked into diff_store at build):
#       * idx95: _xlsx_struct を座標キー → (行ラベル=タスクID, 列ヘッダ) キーイングに。ヘッダ行シフト＋
#         列並べ替えで 369 の偽 modify を量産していたのが、真の変更 (T15 担当者に小林 直樹を追加) ＋
#         22 件の一律 status_transition (未着手→完了、質問側除外可) だけに収束。
#       * idx1: 丸ごと削除された pptx/docx 比較表を、全セル値の列挙ではなく「性能比較表を削除し1行要約に
#         置換」の 1 変更に集約 (中身が NEW に verbatim 再出現＝移動 の表は集約しない＝idx9 該当なし保護)。
#       * idx14: 列名 underscore 化 (_schema_underscore_renames) を advisory 経路にも前置 (従来 rank_changes 内のみ)。
#       * idx22: notebook の記述統計表に追加された目的変数列 (既存 vision 焼込 headers_added を再利用) を
#         決定論的に「目的変数 class の列の統計量が追加 (他列同一)」と属性化 (新規 vision 実行なし)。
#
# 番兵: scripts/sonnet_sentinels.json (idx58 版, SOT-2695)。GATE は番兵回帰 (10/10 必須) でのみ FAIL。
# 対象 idx: 1(表→要約)/14(列名網羅)/22(class統計量)/95(担当者追加) ＋ 回帰ガード 9(該当なし)/16(0.589)。
# --dev ⇒ official:false (flash 非スタック; 昇格根拠にしない)。
#
# GOLD = artifacts/predictions_test_v4_final.csv。gold ハードコードなし・per-idx 分岐なし。
# NO resume cache (RAG_CLAUDE_MCP_RESUME=0): resume key は (model, question) のみで config 非依存
#   (SOT-2664 gotcha)、本レバーは回答を変えるので sidecar replay は変化を隠す。Gemini $0 (RAG_FORBID_GEMINI=1)。
set -euo pipefail
cd /workspaces/signate-messy-drive-rag

.venv/bin/python scripts/check_flag_manifest.py "$0" || exit 1

# Rebuild the question-independent stores (LLM-free, idempotent; cycle-8 integrated union). The
# diff_store MUST be rebuilt with RAG_VDIFF_STRUCT=1 (and RAG_VDIFF_CLASSIFY=1) so the new structural
# repairs are baked into diff_store.jsonl (serve reads the persisted store). image_ocr / nb_chart / ocr
# stores are NOT rebuilt (their builds need vision): serve reads the persisted artifacts baked at build.
.venv/bin/python -m src.rag.index.derived_metrics
.venv/bin/python -m src.rag.index.raw_artifact_store
.venv/bin/python -m src.rag.index.doc_reach_store
.venv/bin/python -m src.rag.index.schedule_store
.venv/bin/python -m src.rag.index.visual_store
.venv/bin/python -m src.rag.index.action_row_store
.venv/bin/python -m src.rag.index.case_finance_store
.venv/bin/python -c "from src.rag.index import report_attr_store as s; r=s.build()['report']; print('[build] report_attr_store', r)"
RAG_VDIFF_STRUCT=1 RAG_VDIFF_CLASSIFY=1 .venv/bin/python -c "from src.rag.index import diff_store as s; r=s.build(); print('[build] diff_store (RAG_VDIFF_STRUCT=1 RAG_VDIFF_CLASSIFY=1)', r.get('pairs') if isinstance(r,dict) else r)"
HOME="${HOME:-/tmp}" .venv/bin/python -m src.rag.index.heading_page_store
.venv/bin/python -m src.rag.index.xlsx_formula_trace
.venv/bin/python -m src.rag.index.rate_table_store
.venv/bin/python -m src.rag.index.plan_coverage_store
.venv/bin/python -m src.rag.index.analysis_xref_store
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

# --- cycle-8 levers ---
export RAG_ANALYSIS_XREF=1
export RAG_PLAN_COVERAGE=1
export RAG_FORMAT_SERIES=1
export RAG_RATE_TABLE=1
export RAG_FORMULA_APPLY=1
export RAG_VDIFF_CLASSIFY=1
export RAG_REPORT_SERIES=1

# --- cycle-9 C? lever (this child) ---
export RAG_VDIFF_STRUCT=1

# --- investigator on flat-rate Sonnet; parallelism 1 = shared-limit protection ---
export RAG_INVESTIGATOR_BACKEND=claude-mcp
export RAG_CLAUDE_MCP_MODEL=sonnet
export RAG_MCP_TOOL_LOG=artifacts/sot2700_tool_calls.jsonl
export RAG_CLAUDE_MCP_RESUME=0

TARGET="${TARGET:-1,9,14,16,22,95}"
GOLD="${GOLD:-artifacts/predictions_test_v4_final.csv}"

echo "=== SOT-2700 cycle9 vdiff-struct focused gate start $(date -u +%FT%TZ) ==="
echo "BACKEND=$RAG_INVESTIGATOR_BACKEND VDIFF_STRUCT=$RAG_VDIFF_STRUCT VDIFF_CLASSIFY=$RAG_VDIFF_CLASSIFY RESUME=$RAG_CLAUDE_MCP_RESUME TARGET=$TARGET GOLD=$GOLD"
.venv/bin/python scripts/run_focused_gate.py \
  --label sot2700_vdiff_struct \
  --target "$TARGET" \
  --sentinels scripts/sonnet_sentinels.json \
  --gold "$GOLD" \
  --issue SOT-2700 \
  --dev --no-smoke \
  --workers 1
echo "=== SOT-2700 cycle9 vdiff-struct focused gate done $(date -u +%FT%TZ) exit=$? ==="
