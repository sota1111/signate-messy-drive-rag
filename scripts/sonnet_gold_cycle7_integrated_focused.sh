#!/usr/bin/env bash
# SOT-2683 — cycle-7 INTEGRATED focused gate (official:false lane).
#
# Env = scripts/sonnet_gold_cycle6.sh (net68 実証済み: cycle6 gold100 = 77/14/9) + ALL promoted
# cycle-7 child levers combined:
#   RAG_IMAGE_OCR_STORE=1    — SOT-2684 K1 build 時 OCR/EMF 画像証拠ストア→extractor 合流 (idx8/52 class)
#   RAG_NB_CHART_STORE=1     — SOT-2685 K2 notebook 描画チャート vision 焼き込みストア (idx56/66 class)
#   RAG_XLSX_FORMULA_TRACE=1 — SOT-2686 K3 xlsx 数式依存トレース＋記載回帰係数の行適用 (idx47/83 class)
#   RAG_XREF_COVERAGE=1      — SOT-2687 K4 用語集別名/段階メトリクス/leaderboard 差分クロス参照 (idx34/36/62 class)
#   RAG_CORR_SIGN=1          — SOT-2688 K5a 相関レーン符号対応 (idx91; idx4 保存)
#   RAG_BIN_RANGE_FORMAT=1   — SOT-2688 K5b ビン範囲チルダ書式 naturalization (idx29 class)
#   RAG_SPECIAL_PROVISION=1  — SOT-2688 K5c 特別規定なし＋一般規定合成契約 (idx78 class)
# Purpose: verify the levers COMPOSE before the one full gold100 run — each child gated alone;
# this is the first time they run together.
#
# BASE NOTE (config drift, resolved deliberately): the cycle-7 child gates ran on a base that
# omitted four cycle-6 promoted flags (RAG_DOC_REACH_STORE / RAG_RAW_ARTIFACT_STORE /
# RAG_SCHEDULE_STORE / RAG_DECIMAL_UNIT_STRIP) with no recorded rationale except K1's
# RAG_DOC_REACH_STORE note (its two lookup tools were suspected in an idx21 flap later shown by
# SOT-2686's A/B to be judge noise). The net68-proven base INCLUDES those four flags and they carry
# cycle-6 recoveries (idx99 / idx32,61,62 / idx92,94,96,72 / idx79), so this integrated gate keeps
# the FULL cycle-6 base and lets the sentinel gate arbitrate the composition.
#
# GOLD = artifacts/predictions_test_v4_final.csv (2026-08-13 gold audit, SOT-2684: idx8 corrected
# 14,744→17,744ドル [ML *平均* 143,000 − DE 125,256; the 143,000 is only in the EMF]; idx7 reworded,
# value-identical). v4 differs from v3 only at idx7/idx8 — no sentinel overlap.
#
# Targets (per docs/ai/sonnet_cycle_analysis/cycle7.md §3, all-child primary union + guards):
#   K1 8,50,52,68 / K2 56,66 / K3 47,83 / K4 34,36,62 / K5 91,29,78
#   + K5 lane canaries 4 (corr-lane 保存), 0,85 (RAG_NONE_BARE 非衝突)
#   Union = 17 idx. Child-achieved Perfect to reconfirm: 8,52,56,66,47,83,34,62,91.
# Gate FAILS on any Sonnet sentinel regression. --dev ⇒ official:false (non-flash stack; NOT a
# promotion basis).
#
# NO resume cache (RAG_CLAUDE_MCP_RESUME=0): the resume key is (model, question) only — NOT config
# (SOT-2664 gotcha) — and the K5 levers change answers globally, so any sidecar replay would mask
# the very composition under test. Target AND sentinels re-derive live.
set -euo pipefail
cd /workspaces/signate-messy-drive-rag

.venv/bin/python scripts/check_flag_manifest.py "$0" || exit 1

# Rebuild the question-independent stores (LLM-free, idempotent; union of cycle-6 base + all
# cycle-7 child rebuilds; heading_page_store needs a writable HOME for LibreOffice headless).
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
# Rebuild the lexical FTS + typed evidence indexes WITH the OCR stores enabled so *search* surfaces
# the image-OCR content (SOT-2684: idx8 EMF給与表 / idx52「別契約」/ idx68「投資実装係数」). LLM-free
# (RAG_PDF_OCR unset during build ⇒ no live vision; OCR text comes from the persisted stores).
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

# --- cycle-6 levers (all six children promoted; FULL net68 base — see BASE NOTE) ---
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
export RAG_MCP_TOOL_LOG=artifacts/sonnet_cycle7_integrated_tool_calls.jsonl
export RAG_CLAUDE_MCP_RESUME=0

TARGET="${TARGET:-0,4,8,29,34,36,47,50,52,56,62,66,68,78,83,85,91}"
GOLD="${GOLD:-artifacts/predictions_test_v4_final.csv}"

echo "=== SOT-2683 cycle7 integrated focused gate start $(date -u +%FT%TZ) ==="
echo "BACKEND=$RAG_INVESTIGATOR_BACKEND IMAGE_OCR=$RAG_IMAGE_OCR_STORE NB_CHART=$RAG_NB_CHART_STORE XLSX_TRACE=$RAG_XLSX_FORMULA_TRACE XREF=$RAG_XREF_COVERAGE CORR_SIGN=$RAG_CORR_SIGN BIN_RANGE=$RAG_BIN_RANGE_FORMAT SPECIAL_PROV=$RAG_SPECIAL_PROVISION RESUME=$RAG_CLAUDE_MCP_RESUME TARGET=$TARGET GOLD=$GOLD"
.venv/bin/python scripts/run_focused_gate.py \
  --label sot2683_cycle7_integrated \
  --target "$TARGET" \
  --sentinels scripts/sonnet_sentinels.json \
  --gold "$GOLD" \
  --issue SOT-2683 \
  --dev --no-smoke \
  --workers 1
echo "=== SOT-2683 cycle7 integrated focused gate done $(date -u +%FT%TZ) exit=$? ==="
