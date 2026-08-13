#!/usr/bin/env bash
# SOT-2683 — Sonnet gold100 improvement cycle 7, integrated config (official:false lane).
#
# Base env = scripts/sonnet_gold_cycle6.sh (net68 実証済み: cycle6 gold100 = 77 match / 14 abstain /
# 9 wrong) PLUS the cycle-7 child levers that PASSED their focused gates (all five promoted):
#   RAG_IMAGE_OCR_STORE=1    — SOT-2684 K1 build 時 OCR/EMF 画像証拠ストア→extractor 合流＋FTS/evidence
#                              再索引 (idx8 EMF給与表 / idx52「別契約」/ idx68「投資実装係数」)
#   RAG_NB_CHART_STORE=1     — SOT-2685 K2 notebook 描画チャートの build 時 vision 焼き込みストア＋
#                              決定論レーン (idx56 y軸目盛り最大 / idx66 件数最多日)
#   RAG_XLSX_FORMULA_TRACE=1 — SOT-2686 K3 xlsx 数式依存トレース＋記載回帰係数の行適用 (idx47 参照行の
#                              YEAR BUILT / idx83 係数×index=1770 行=0.38317)
#   RAG_XREF_COVERAGE=1      — SOT-2687 K4 用語集ローマ字別名/段階メトリクス/leaderboard 上位2設定差分の
#                              クロス参照レーン (idx34/36/62)
#   RAG_CORR_SIGN=1          — SOT-2688 K5a 相関レーンの符号対応 (idx91「最も負」; idx4 |r|最大は保存)
#   RAG_BIN_RANGE_FORMAT=1   — SOT-2688 K5b ビン範囲の区間記法→チルダ書式 naturalization (idx29)
#   RAG_SPECIAL_PROVISION=1  — SOT-2688 K5c 「特別規定なし＋一般規定内容」合成回答契約 (idx78)
#
# BASE NOTE (config drift, resolved deliberately): the cycle-7 child gates ran on a base missing four
# cycle-6 promoted flags (DOC_REACH/RAW_ARTIFACT/SCHEDULE stores, DECIMAL_UNIT_STRIP) without recorded
# evidence for the drop; those flags carry cycle-6 recoveries (idx99/32,61,62/92,94,96,72/79), so this
# integrated config keeps the FULL net68 base. Composition was verified by
# scripts/sonnet_gold_cycle7_integrated_focused.sh (sentinel gate) before this gold100 run.
#
# GOLD = artifacts/predictions_test_v4_final.csv (2026-08-13 gold audit, SOT-2684: idx8 corrected
# 14,744→17,744ドル; idx7 reworded, value-identical; v4 differs from v3 only at idx7/idx8).
#
# ***OFFICIAL:FALSE*** — DEV measurement only (--no-official). Never backs a champion promotion or
# non-regression claim. The official lane stays gemini-3.6-flash (SOT-2625 model guard), untouched.
#
# RESUME: FRESH sidecar (below). The claude-mcp resume key is (model, question) only — NOT config
# (SOT-2664 gotcha) — and this cycle changes answers globally (K5 levers; extractor-merged OCR text),
# so replaying the cycle-6 sidecar would mask the change on every LLM lane. The fresh sidecar still
# gives usage-limit continuation: answered questions persist, so re-running this script resumes where
# it stopped.
#
# USAGE-LIMIT: flat-rate Sonnet is shared account-wide. Parallelism 1; on a detected limit the
# claude-mcp backend abstains the question and persists answered ones to the resume sidecar — simply
# re-run this script to continue where it stopped.
set -euo pipefail
cd /workspaces/signate-messy-drive-rag

.venv/bin/python scripts/check_flag_manifest.py "$0" || exit 1

# Rebuild the question-independent stores (LLM-free, idempotent; union of cycle-6 base + all cycle-7
# child rebuilds; heading_page_store needs a writable HOME for LibreOffice headless).
# image_ocr_store / notebook_chart_store / ocr_store are NOT rebuilt here (their builds need vision →
# RAG_FORBID_GEMINI): serve reads the persisted artifacts baked at build time (Gemini $0 at serve).
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
# Rebuild the lexical FTS + typed evidence indexes WITH the OCR stores enabled so *search* surfaces
# the image-OCR content (SOT-2684). LLM-free (OCR text comes from the persisted stores).
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

# --- cycle-6 levers (FULL net68 base — see BASE NOTE) ---
export RAG_DOC_REACH_STORE=1
export RAG_RAW_ARTIFACT_STORE=1
export RAG_DERIVED_COVERAGE=1
export RAG_SCHEDULE_STORE=1
export RAG_VDIFF_NORMALIZE=1
export RAG_DECIMAL_UNIT_STRIP=1

# --- cycle-7 levers (all five children promoted their focused gates) ---
export RAG_IMAGE_OCR_STORE=1
export RAG_NB_CHART_STORE=1
export RAG_XLSX_FORMULA_TRACE=1
export RAG_XREF_COVERAGE=1
export RAG_CORR_SIGN=1
export RAG_BIN_RANGE_FORMAT=1
export RAG_SPECIAL_PROVISION=1

# --- investigator on flat-rate Sonnet; parallelism 1 = shared-limit protection ---
export RAG_INVESTIGATOR_BACKEND=claude-mcp
export RAG_CLAUDE_MCP_MODEL=sonnet
export RAG_CLAUDE_MCP_RESUME=artifacts/gold100_sonnet_cycle7_resume.jsonl
export RAG_MCP_TOOL_LOG=artifacts/gold100_sonnet_cycle7_tool_calls.jsonl

GOLD="${GOLD:-artifacts/predictions_test_v4_final.csv}"

echo "=== SOT-2683 Sonnet gold100 cycle7 (official:false) start $(date -u +%FT%TZ) ==="
echo "BACKEND=$RAG_INVESTIGATOR_BACKEND MODEL=$RAG_CLAUDE_MCP_MODEL RESUME=$RAG_CLAUDE_MCP_RESUME GOLD=$GOLD"
.venv/bin/python -m scoring.gold_offline --run --workers 1 --no-official \
  --gold "$GOLD" \
  --out artifacts/gold100_sonnet_cycle7.json
echo "=== SOT-2683 Sonnet gold100 cycle7 done $(date -u +%FT%TZ) exit=$? ==="
