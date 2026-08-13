#!/usr/bin/env bash
# SOT-2693 (cycle8 C4) — 書式系列・文書内数値表リーチ focused gate (official:false lane).
#
# Base env = the cycle-7/8 integrated config PLUS the two cycle-8 C4 levers, both default OFF ⇒ OFF
# byte-identical:
#   RAG_FORMAT_SERIES=1 — 書式付き数値系列の上昇率レーン (format_series_store を build 時に焼く):
#       各案件の 会議録(MM)/報告資料(RP) 系列を日付順に並べ、各文書の 黄色ハイライト×赤字 数値を系列化。
#       serve は fact_layer.resolve の決定論直答で「最初→最後の上昇率 (last-first)/first*100」を返す。
#       idx17 = 青葉与信マネジメント(AYM) の会議録(MM=Meeting Minutes, 社内用語集) 系列
#       (2025-04-29=0.589 → 2025-05-27=0.602 ⇒ 2.21)。04-09 は黄×赤 数値なし=対象外。
#   RAG_RATE_TABLE=1    — 税率表 帯別差分レーン (rate_table_store を build 時に焼く):
#       全 PDF の「価格帯×現行率×新率」表を検出し帯別 |新-現行| ＋ argmin/argmax を事前計算。
#       idx48 = 青嶺 NY 不動産調査 PDF のマンション税表で 絶対増加が最小の帯 (100万ドル超-500万ドル以下)。
#
# 番兵: scripts/sonnet_sentinels.json (idx58 版, SOT-2695)。Gate FAILS on any sentinel regression (10/10)。
# 非番兵 idx16 (黄×赤の別問=中間報告資料スコープ, SOT-2695) を TARGET に含めて非悪化を確認する
#   (同一書式面を触るため; target verdict は gate を落とさない)。
#
# GOLD = artifacts/predictions_test_v3_final.csv (sonnet_sentinels と同一)。gold ハードコードなし
#   (idx17=会議録PDFの黄×赤数値の決定論抽出＋上昇率, idx48=税率表の帯別 |新-現行| argmin)。
#
# --dev ⇒ official:false (昇格根拠にしない)。RAG_CLAUDE_MCP_RESUME=0 (本レバーは回答を変えるので
#   sidecar replay は変化を隠す)。Gemini $0 (RAG_FORBID_GEMINI=1)。
set -euo pipefail
cd /workspaces/signate-messy-drive-rag

.venv/bin/python scripts/check_flag_manifest.py "$0" || exit 1

# Rebuild the question-independent stores (LLM-free, idempotent). diff_store は RAG_VDIFF_CLASSIFY=1 で
# 焼く (SOT-2695 の idx16 用)。cycle8 C4 の 2 ストアを追加で焼く (format_events/pdfplumber, genai $0)。
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

# --- cycle-8 C6 levers (idx16 の再現に必要) ---
export RAG_VDIFF_CLASSIFY=1
export RAG_REPORT_SERIES=1

# --- cycle-8 C4 levers (this child) ---
export RAG_FORMAT_SERIES=1
export RAG_RATE_TABLE=1

# --- investigator on flat-rate Sonnet; parallelism 1 = shared-limit protection ---
export RAG_INVESTIGATOR_BACKEND=claude-mcp
export RAG_CLAUDE_MCP_MODEL=sonnet
export RAG_MCP_TOOL_LOG=artifacts/sot2693_tool_calls.jsonl
export RAG_CLAUDE_MCP_RESUME=0

TARGET="${TARGET:-17,48,16}"
GOLD="${GOLD:-artifacts/predictions_test_v3_final.csv}"

echo "=== SOT-2693 cycle8 C4 focused gate start $(date -u +%FT%TZ) ==="
echo "BACKEND=$RAG_INVESTIGATOR_BACKEND FORMAT_SERIES=$RAG_FORMAT_SERIES RATE_TABLE=$RAG_RATE_TABLE RESUME=$RAG_CLAUDE_MCP_RESUME TARGET=$TARGET GOLD=$GOLD"
.venv/bin/python scripts/run_focused_gate.py \
  --label sot2693_format_series_rate_table \
  --target "$TARGET" \
  --sentinels scripts/sonnet_sentinels.json \
  --gold "$GOLD" \
  --issue SOT-2693 \
  --dev --no-smoke \
  --workers 1
echo "=== SOT-2693 cycle8 C4 focused gate done $(date -u +%FT%TZ) exit=$? ==="
