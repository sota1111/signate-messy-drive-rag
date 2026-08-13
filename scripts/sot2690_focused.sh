#!/usr/bin/env bash
# SOT-2690 (cycle8 C1) — 契約・請求ファクトレーン focused gate (official:false lane).
#
# Base env = cycle-7 INTEGRATED champion config (net72 実証済み) を **そのまま** 使う。本子は新フラグを
# 足さず、既に champion-ON の RAG_CASE_FINANCE_STORE=1 が gate する case_finance ストア/レーンを拡張する:
#   * store: ひがし丘 6.2条「時間単価は25,000円（税別）」の格助詞入り本文から time_rate_excl_tax を抽出、
#            契約中の『N時間を超える場合』時間閾値条項を special_settlement_provisions に網羅列挙(質問非依存)。
#   * serve (case_finance_lane): 2 レーン追加 (RAG_CASE_FINANCE_STORE で gate・OFF時 byte-identical):
#       idx23 = ACTH155h10m の 30分切上→税込請求 4,276,250 と 見込税込 4,675,000 の差額 398,750円。
#       idx78 = >200h の時間閾値条項が契約に無い→「特別規定なし＋一般規定要点」を gold 同形に決定論合成。
#   * idx98 (TM案件 RATE変更開始日 2025年7月1日) は非決定論 (コーパスに変更イベント/該当日付の記載なし)。
#     gold ハードコード禁止のため本子では未着手 (TARGET に含めて非改善を可視化のみ; 受入は 2/3=idx23/78)。
#
# 番兵: scripts/sonnet_sentinels.json (idx58 版)。Gate FAILS on any sentinel regression (10/10)。
# GOLD = artifacts/predictions_test_v4_final.csv。gold ハードコードなし (idx23=請求シナリオ算術, idx78=
#   閾値条項不在確認後の一般規定機械合成)。--dev ⇒ official:false。RAG_CLAUDE_MCP_RESUME=0 (回答が変わる
#   レバーなので sidecar replay は変化を隠す)。Gemini $0 (RAG_FORBID_GEMINI=1)。
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

TARGET="${TARGET:-23,78,98}"
GOLD="${GOLD:-artifacts/predictions_test_v4_final.csv}"

echo "=== SOT-2690 cycle8 C1 contract-billing focused gate start $(date -u +%FT%TZ) ==="
echo "BACKEND=$RAG_INVESTIGATOR_BACKEND IMAGE_OCR=$RAG_IMAGE_OCR_STORE NB_CHART=$RAG_NB_CHART_STORE XLSX_TRACE=$RAG_XLSX_FORMULA_TRACE XREF=$RAG_XREF_COVERAGE CORR_SIGN=$RAG_CORR_SIGN BIN_RANGE=$RAG_BIN_RANGE_FORMAT SPECIAL_PROV=$RAG_SPECIAL_PROVISION RESUME=$RAG_CLAUDE_MCP_RESUME TARGET=$TARGET GOLD=$GOLD"
.venv/bin/python scripts/run_focused_gate.py \
  --label sot2690_contract_billing_fact_lane \
  --target "$TARGET" \
  --sentinels scripts/sonnet_sentinels.json \
  --gold "$GOLD" \
  --issue SOT-2690 \
  --dev --no-smoke \
  --workers 1
echo "=== SOT-2690 cycle8 C1 contract-billing focused gate done $(date -u +%FT%TZ) exit=$? ==="
