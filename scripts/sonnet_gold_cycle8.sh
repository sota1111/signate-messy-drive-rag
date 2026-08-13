#!/usr/bin/env bash
# SOT-2689 — Sonnet gold100 improvement cycle 8, integrated config (official:false lane).
#
# Base env = scripts/sonnet_gold_cycle7.sh (net72 実証済み: cycle7 gold100 = 80 match / 12 abstain /
# 8 wrong) PLUS the cycle-8 child levers that PASSED their focused gates (all six children):
#   (C1 SOT-2690) RAG_CASE_FINANCE_STORE 拡張 — 請求シナリオ計算＋特別規定合成の決定論化
#                 (idx23/78; idx98 は RATE 変更日の証拠がコーパスに無く honest abstain 維持)
#   (C2 SOT-2691) RAG_ANALYSIS_XREF=1 — 全精度段階メトリクス・未完事項ID・実装設定 enum の
#                 クロス参照 (idx36/60/73, stretch 53)
#   (C3 SOT-2692) RAG_PLAN_COVERAGE=1 — 暗号化計画 xlsx の工数派生＋提案書週次スケジュール
#                 (idx79/88)
#   (C4 SOT-2693) RAG_FORMAT_SERIES=1 / RAG_RATE_TABLE=1 — 黄×赤時系列上昇率レーン＋
#                 税率表 doc-table 帯別 argmin (idx17/48)
#   (C5 SOT-2694) RAG_FORMULA_APPLY=1 — 全文OCR索引統合＋EMF給与表 FTS 到達＋式適用レーン
#                 (idx68/50; idx8 ガード)
#   (C6 SOT-2695) RAG_VDIFF_CLASSIFY=1 / RAG_REPORT_SERIES=1 — 見出しラベル=cosmetic /
#                 列名underscore化=substantive 前置＋報告系列スコープ (idx9/14/16)
# Composition was verified by scripts/sonnet_gold_cycle8_integrated_focused.sh (sentinel gate,
# idx58 replacing flaky idx21 per SOT-2695) before this gold100 run.
#
# GOLD = artifacts/predictions_test_v4_final.csv (2026-08-13 gold audit).
#
# ***OFFICIAL:FALSE*** — DEV measurement only (--no-official). Never backs a champion promotion or
# non-regression claim. The official lane stays gemini-3.6-flash (SOT-2625 model guard), untouched.
#
# RESUME: FRESH sidecar (below). The claude-mcp resume key is (model, question) only — NOT config
# (SOT-2664 gotcha) — and this cycle changes answers globally (new stores merged into FTS/evidence
# indexes), so replaying the cycle-7 sidecar would mask the change on every LLM lane. The fresh
# sidecar still gives usage-limit continuation: answered questions persist, so re-running this
# script resumes where it stopped.
#
# USAGE-LIMIT: flat-rate Sonnet is shared account-wide. Parallelism 1; on a detected limit the
# claude-mcp backend abstains the question and persists answered ones to the resume sidecar — simply
# re-run this script to continue where it stopped.
set -euo pipefail
cd /workspaces/signate-messy-drive-rag

.venv/bin/python scripts/check_flag_manifest.py "$0" || exit 1

# Rebuild the question-independent stores (LLM-free, idempotent; union of cycle-7 base + all
# cycle-8 child rebuilds; heading_page_store needs a writable HOME for LibreOffice headless).
# image_ocr_store / notebook_chart_store / ocr_store are NOT rebuilt here (their builds need vision
# → RAG_FORBID_GEMINI): serve reads the persisted artifacts baked at build time (Gemini $0 at serve).
.venv/bin/python -m src.rag.index.derived_metrics
.venv/bin/python -m src.rag.index.raw_artifact_store
.venv/bin/python -m src.rag.index.doc_reach_store
.venv/bin/python -m src.rag.index.schedule_store
.venv/bin/python -m src.rag.index.visual_store
.venv/bin/python -m src.rag.index.action_row_store
.venv/bin/python -m src.rag.index.case_finance_store
.venv/bin/python -c "from src.rag.index import report_attr_store as s; r=s.build()['report']; print('[build] report_attr_store', r)"
.venv/bin/python -c "from src.rag.index import diff_store as s; r=s.build(); print('[build] diff_store', r.get('pairs') if isinstance(r,dict) else r)"
HOME="${HOME:-/tmp}" .venv/bin/python -m src.rag.index.heading_page_store
.venv/bin/python -m src.rag.index.xlsx_formula_trace
# --- cycle-8 child stores (LLM-free) ---
.venv/bin/python -m src.rag.index.rate_table_store
.venv/bin/python -m src.rag.index.plan_coverage_store
.venv/bin/python -m src.rag.index.analysis_xref_store
.venv/bin/python -m src.rag.index.formula_apply_store
# Rebuild the lexical FTS + typed evidence indexes WITH the OCR stores enabled so *search* surfaces
# the image-OCR content (SOT-2684/2694 full-text OCR integration). LLM-free (OCR text comes from
# the persisted stores).
RAG_OCR_STORE=1 RAG_IMAGE_OCR_STORE=1 .venv/bin/python -c "from src.rag.index import text_fts as t; r=t.build(); print('[build] text_fts', {k:r.get(k) for k in ('records','docs')})"
RAG_OCR_STORE=1 RAG_IMAGE_OCR_STORE=1 .venv/bin/python -c "from src.rag.index import evidence_index as e; print('[build] evidence_index', e.build_only())"

export VERTEX_LOCATION=global
export GEN_MODEL=gemini-3.6-flash
export GEN_MODEL_HARD=gemini-3.6-flash
export VISION_MODEL=gemini-3.6-flash
export PYTHONPATH=/tmp/genai_patch:.

# --- champion Wave A + B1 flags, verbatim from scripts/sonnet_gold_cycle6.sh ---
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

# --- cycle-5 levers (promoted children only) ---
export RAG_FANOUT_FINISHER=1
export RAG_FANOUT_FINISHER_MAX=1
export RAG_VDIFF_SUBJECT=1
export RAG_NONE_BARE=1
export RAG_EXACT_LABEL=1
export RAG_HEADING_PAGE_STORE=1

# --- cycle-6 levers (FULL net68 base) ---
export RAG_DOC_REACH_STORE=1
export RAG_RAW_ARTIFACT_STORE=1
export RAG_DERIVED_COVERAGE=1
export RAG_SCHEDULE_STORE=1
export RAG_VDIFF_NORMALIZE=1
export RAG_DECIMAL_UNIT_STRIP=1

# --- cycle-7 levers (all five children promoted) ---
export RAG_IMAGE_OCR_STORE=1
export RAG_NB_CHART_STORE=1
export RAG_XLSX_FORMULA_TRACE=1
export RAG_XREF_COVERAGE=1
export RAG_CORR_SIGN=1
export RAG_BIN_RANGE_FORMAT=1
export RAG_SPECIAL_PROVISION=1

# --- cycle-8 levers (all six children passed their focused gates) ---
export RAG_ANALYSIS_XREF=1
export RAG_PLAN_COVERAGE=1
export RAG_FORMAT_SERIES=1
export RAG_RATE_TABLE=1
export RAG_FORMULA_APPLY=1
export RAG_VDIFF_CLASSIFY=1
export RAG_REPORT_SERIES=1

# --- investigator on flat-rate Sonnet; parallelism 1 = shared-limit protection ---
export RAG_INVESTIGATOR_BACKEND=claude-mcp
export RAG_CLAUDE_MCP_MODEL=sonnet
export RAG_CLAUDE_MCP_RESUME=artifacts/gold100_sonnet_cycle8_resume.jsonl
export RAG_MCP_TOOL_LOG=artifacts/gold100_sonnet_cycle8_tool_calls.jsonl

GOLD="${GOLD:-artifacts/predictions_test_v4_final.csv}"

echo "=== SOT-2689 Sonnet gold100 cycle8 (official:false) start $(date -u +%FT%TZ) ==="
echo "BACKEND=$RAG_INVESTIGATOR_BACKEND MODEL=$RAG_CLAUDE_MCP_MODEL RESUME=$RAG_CLAUDE_MCP_RESUME GOLD=$GOLD"
.venv/bin/python -m scoring.gold_offline --run --workers 1 --no-official \
  --gold "$GOLD" \
  --out artifacts/gold100_sonnet_cycle8.json
echo "=== SOT-2689 Sonnet gold100 cycle8 done $(date -u +%FT%TZ) exit=$? ==="
