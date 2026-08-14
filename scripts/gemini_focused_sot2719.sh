#!/usr/bin/env bash
# SOT-2719 — backend=gemini FOCUSED verification (official:false/dev; no submit, no champion touch).
# Cheap pre-check on the Gemini flash-3.6 serve path for the idx84「ページ数」bare recovery target + the
# idx12/18/59 page-form regression guard + 10 sentinels,
# scored vs the audited v4 gold, BEFORE any full gold100 run. --no-ledger (dev; recorded by caller in
# docs/ai/gemini_gold100_history.jsonl, not the champion experiment ledger).
set -euo pipefail
cd /workspaces/signate-messy-drive-rag
.venv/bin/python scripts/check_flag_manifest.py "$0" || exit 1
.venv/bin/python -m src.rag.index.derived_metrics
.venv/bin/python -m src.rag.index.raw_artifact_store
.venv/bin/python -m src.rag.index.analysis_metrics_enum_store
.venv/bin/python -m src.rag.index.doc_reach_store
.venv/bin/python -m src.rag.index.derived_ranking_store
.venv/bin/python -m src.rag.index.schedule_store
RAG_HIGHLIGHT_PIVOT_CONTEXT=1 .venv/bin/python -m src.rag.index.visual_store
.venv/bin/python -m src.rag.index.format_facts_store
.venv/bin/python -m src.rag.index.pptx_money_page_store
.venv/bin/python -m src.rag.index.contact_master_store
.venv/bin/python -m src.rag.index.pptx_note_scope_store
.venv/bin/python -m src.rag.index.schedule_plan_store
.venv/bin/python -m src.rag.index.docx_comment_store
.venv/bin/python -m src.rag.index.action_row_store
.venv/bin/python -m src.rag.index.case_finance_store
.venv/bin/python -m src.rag.index.master_join_store
.venv/bin/python -c "from src.rag.index import report_attr_store as s; r=s.build()['report']; print('[build] report_attr_store', r)"
RAG_VDIFF_STRUCT=1 RAG_VDIFF_CLASSIFY=1 RAG_VDIFF_DIRECT_COMMIT=1 .venv/bin/python -c "from src.rag.index import diff_store as s; r=s.build(); print('[build] diff_store (RAG_VDIFF_STRUCT=1 RAG_VDIFF_CLASSIFY=1 RAG_VDIFF_DIRECT_COMMIT=1)', r.get('pairs') if isinstance(r,dict) else r)"
HOME="${HOME:-/tmp}" .venv/bin/python -m src.rag.index.heading_page_store
.venv/bin/python -m src.rag.index.xlsx_formula_trace
.venv/bin/python -m src.rag.index.rate_table_store
.venv/bin/python -m src.rag.index.plan_coverage_store
.venv/bin/python -m src.rag.index.analysis_xref_store
.venv/bin/python -m src.rag.index.formula_apply_store
RAG_OCR_STORE=1 RAG_IMAGE_OCR_STORE=1 .venv/bin/python -c "from src.rag.index import text_fts as t; r=t.build(); print('[build] text_fts', {k:r.get(k) for k in ('records','docs')})"
RAG_OCR_STORE=1 RAG_IMAGE_OCR_STORE=1 .venv/bin/python -c "from src.rag.index import evidence_index as e; print('[build] evidence_index', e.build_only())"
export VERTEX_LOCATION=global GEN_MODEL=gemini-3.6-flash GEN_MODEL_HARD=gemini-3.6-flash VISION_MODEL=gemini-3.6-flash
export PYTHONPATH=/tmp/genai_patch:.
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
export RAG_FACT_LAYER=1
export RAG_OCR_STORE=1
export RAG_FORMAT_STRIP_PAREN=1
export RAG_ACTION_ROW_STORE=1
export RAG_VISUAL_STORE=1
export RAG_CASE_FINANCE_STORE=1
export RAG_REPORT_ATTR_STORE=1
export RAG_FORMAT_VALUE_NORM=1
export RAG_TEXT_FTS=1
export RAG_UNIFIED_SEARCH=1
export RAG_PLAN_FANOUT=1
export RAG_BARE_ANSWER=1
export RAG_FANOUT_FINISHER=1
export RAG_FANOUT_FINISHER_MAX=1
export RAG_VDIFF_SUBJECT=1
export RAG_NONE_BARE=1
export RAG_EXACT_LABEL=1
export RAG_HEADING_PAGE_STORE=1
export RAG_DOC_REACH_STORE=1
export RAG_RAW_ARTIFACT_STORE=1
export RAG_DERIVED_COVERAGE=1
export RAG_SCHEDULE_STORE=1
export RAG_VDIFF_NORMALIZE=1
export RAG_DECIMAL_UNIT_STRIP=1
export RAG_IMAGE_OCR_STORE=1
export RAG_NB_CHART_STORE=1
export RAG_XLSX_FORMULA_TRACE=1
export RAG_XREF_COVERAGE=1
export RAG_CORR_SIGN=1
export RAG_BIN_RANGE_FORMAT=1
export RAG_SPECIAL_PROVISION=1
export RAG_ANALYSIS_XREF=1
export RAG_PLAN_COVERAGE=1
export RAG_FORMAT_SERIES=1
export RAG_RATE_TABLE=1
export RAG_FORMULA_APPLY=1
export RAG_VDIFF_CLASSIFY=1
export RAG_REPORT_SERIES=1
export RAG_CASE_FINANCE_DIFF=1
export RAG_ANALYSIS_METRICS_ENUM=1
export RAG_ANALYSIS_CONFIG_HYPERPARAM=1
export RAG_STAGED_METRICS=1
export RAG_DERIVED_RANKING=1
export RAG_VDIFF_STRUCT=1
export RAG_TEXT_FTS_PROJECT_ALIAS=1
export RAG_SEP_CONTRACT_ROLE=1
export RAG_FORMAT_FACTS=1
export RAG_HIGHLIGHT_PIVOT_CONTEXT=1
export RAG_PPTX_MONEY_PAGE=1
export RAG_VDIFF_DIRECT_COMMIT=1
export RAG_CONTACT_MASTER=1
export RAG_SCHEDULE_PLAN_LOOKUP=1
export RAG_DOCX_COMMENT_ANCHOR=1
export RAG_VDIFF_DC_NOCHANGE=1
export RAG_VDIFF_DC_COLRENAME=1
export RAG_MASTER_JOIN_LOOKUP=1
export RAG_PPTX_NOTE_SCOPE=1
# --- SOT-2718 levers (net96 baseline) ---
export RAG_VDIFF_LOWCONF_ABSTAIN=1
export RAG_CURRENCY_DIFF_UNIT=1
# --- SOT-2719 lever: 「ページ数」型のみ bare 番号へ決定論整形 (idx84「5ページ（スライド6）」→「5」) ---
export RAG_PAGE_COUNT_BARE=1
export RAG_INVESTIGATOR_BACKEND=gemini
OUT="${FOCUSED_OUT:-artifacts/focused_gemini_sot2719.json}"
# target = idx84 (bare recovery) + idx12/18/59 (page-form regression guard: Nページ/ページ目 must hold)
.venv/bin/python scripts/run_focused_gate.py \
  --label sot2719-page-count-bare-idx84-gemini --issue SOT-2719 \
  --target "84,12,18,59" \
  --gold artifacts/predictions_test_v4_final.csv \
  --workers 4 --no-ledger --out "$OUT"
