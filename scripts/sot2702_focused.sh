#!/usr/bin/env bash
# SOT-2702 (cycle10) — text_search の project フィルタを短縮名/用語集エイリアス許容にする focused gate
# (official:false lane).
#
# Base env = scripts/sonnet_gold_cycle9.sh のフラグ一式（net84+ 実証済み）＋本子の新フラグ:
#   RAG_TEXT_FTS_PROJECT_ALIAS=1 — text_fts.search の project フィルタを完全一致から「短縮名の部分一致
#                                  ＋社内用語集の案件略称エイリアス」許容へ緩和する（曖昧時は絞らず fail-open）。
# Target: idx52（蒼樹会みなみ野「別契約」と明記された役割 → gold=監視ダッシュボード構築）。cycle9 実測は
#   serve が `text_search(query='別契約', project='みなみ野')` と短縮名を渡し、完全一致フィルタで全行落ち →
#   no_match → abstain。本フラグで『監視ダッシュボード構築(別契約)』（最終報告.pdf page:8/9, OCR 由来）へ到達。
#
# 番兵: scripts/sonnet_sentinels.json (idx58 版)。Gate FAILS on any sentinel regression (10/10)。
# GOLD = artifacts/predictions_test_v4_final.csv。gold ハードコードなし（フィルタ緩和のみ、新 store 無し）。
#   --dev ⇒ official:false。RAG_CLAUDE_MCP_RESUME=0（回答が変わるレバーなので sidecar replay は変化を隠す）。
#   Gemini $0（RAG_FORBID_GEMINI=1）。OFF 時 byte-identical（serve 経路のフィルタ分岐のみ、既定 OFF）。
set -euo pipefail
cd /workspaces/signate-messy-drive-rag

.venv/bin/python scripts/check_flag_manifest.py "$0" || exit 1

# Rebuild the question-independent stores (LLM-free, idempotent) — this child adds NO new store
# (serve-path filter change only); the union mirrors cycle9's rebuild so the focused run matches base.
.venv/bin/python -m src.rag.index.derived_metrics
.venv/bin/python -m src.rag.index.raw_artifact_store
.venv/bin/python -m src.rag.index.analysis_metrics_enum_store
.venv/bin/python -m src.rag.index.doc_reach_store
.venv/bin/python -m src.rag.index.derived_ranking_store
.venv/bin/python -m src.rag.index.schedule_store
.venv/bin/python -m src.rag.index.visual_store
.venv/bin/python -m src.rag.index.action_row_store
.venv/bin/python -m src.rag.index.case_finance_store
.venv/bin/python -c "from src.rag.index import report_attr_store as s; r=s.build()['report']; print('[build] report_attr_store', r)"
RAG_VDIFF_STRUCT=1 RAG_VDIFF_CLASSIFY=1 .venv/bin/python -c "from src.rag.index import diff_store as s; r=s.build(); print('[build] diff_store', r.get('pairs') if isinstance(r,dict) else r)"
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

# --- cycle-9 levers ---
export RAG_CASE_FINANCE_DIFF=1
export RAG_ANALYSIS_METRICS_ENUM=1
export RAG_ANALYSIS_CONFIG_HYPERPARAM=1
export RAG_STAGED_METRICS=1
export RAG_DERIVED_RANKING=1
export RAG_VDIFF_STRUCT=1

# --- cycle-10 lever (this child) ---
export RAG_TEXT_FTS_PROJECT_ALIAS=1
# LLM-route recovery was insufficient (project フィルタ緩和で証拠は retrieval 到達したが plan_fanout Sonnet が
# 2 回とも わかりません)。決定論昇格: 既存 text_fts を質問非依存で引き ``X（別契約）`` 役割ラベルを逐語直答。
export RAG_SEP_CONTRACT_ROLE=1

# --- investigator on flat-rate Sonnet; parallelism 1 = shared-limit protection ---
export RAG_INVESTIGATOR_BACKEND=claude-mcp
export RAG_CLAUDE_MCP_MODEL=sonnet
export RAG_MCP_TOOL_LOG=artifacts/sot2702_tool_calls.jsonl
export RAG_CLAUDE_MCP_RESUME=0

TARGET="${TARGET:-52}"
GOLD="${GOLD:-artifacts/predictions_test_v4_final.csv}"

echo "=== SOT-2702 cycle10 text-fts project-alias focused gate start $(date -u +%FT%TZ) ==="
echo "BACKEND=$RAG_INVESTIGATOR_BACKEND TEXT_FTS_PROJECT_ALIAS=$RAG_TEXT_FTS_PROJECT_ALIAS RESUME=$RAG_CLAUDE_MCP_RESUME TARGET=$TARGET GOLD=$GOLD"
.venv/bin/python scripts/run_focused_gate.py \
  --label sot2702_text_fts_project_alias \
  --target "$TARGET" \
  --sentinels scripts/sonnet_sentinels.json \
  --gold "$GOLD" \
  --issue SOT-2702 \
  --dev --no-smoke \
  --workers 1
echo "=== SOT-2702 cycle10 text-fts project-alias focused gate done $(date -u +%FT%TZ) exit=$? ==="
