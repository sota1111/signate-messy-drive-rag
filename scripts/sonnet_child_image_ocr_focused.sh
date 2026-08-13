#!/usr/bin/env bash
# SOT-2684 — focused gate for cycle7 K1 (画像ロック証拠の build 時 OCR/vision 焼き込みストア, official:false).
#
# Base = scripts/sonnet_child_nb_chart_focused.sh (cycle7 K2 構成) PLUS the two K1 levers:
#   RAG_DOC_REACH_STORE=1 — 既存の文書テーブル/全文チャンク lookup ツール(doc_table_lookup /
#     doc_fulltext_search, SOT-2677)。K1 はこの検索対象へ画像由来チャンクを合流させるので reach host として ON。
#   RAG_IMAGE_OCR_STORE=1 — build 時に焼いた画像OCRストア(artifacts/image_ocr_store.jsonl)を serve の
#     doc-reach lookup へ質問非依存に合流させる:
#     (idx8) 東都 データサイエンティスト調査.docx の EMF 埋め込み給与表(EMF テキストレコードを決定論抽出=
#       vision 不要)→ ML エンジニア中央値 140,000 − データエンジニア 125,256 = 14,744。
#     (idx52) みなみ野 最終報告.pdf の画像ページ(テキストレイヤ空)を vision OCR →「別契約」到達。
#     (idx68) 東都 未来予測.pdf の「投資実装係数」計算式ページ(画像)を vision OCR → 式・数値到達。
#     serve gated・既定 OFF ⇒ OFF 時 byte-identical。build のみ Gemini 使用(RAG_IMAGE_OCR_STORE_BUILD)、serve は $0。
# GOLD = predictions_test_v4_final.csv (idx8 corrected 14,744→17,744ドル by the 2026-08-13 gold audit — ML
#   *平均* 143,000 − DE 平均 125,256; that exact 143,000 is ONLY in the EMF, so the image store is what
#   makes idx8 answerable). v4 differs from v3 only at idx7/idx8 — no sentinel overlap, so sentinel gold
#   is identical.
# Targets: 一次 idx8/50/52/68（idx50 は Salary.com 表が本文テキスト側=doc_reach 領域で画像ロックではない;
#   画像から確定できない問いは honest abstain のまま可)。
# Gate FAILS on any Sonnet sentinel regression. --dev ⇒ official:false (non-flash stack; NOT a promotion
# basis). RAG_CLAUDE_MCP_RESUME=0 必須: serve change alters answers, re-derive target AND sentinels live.
set -euo pipefail
cd /workspaces/signate-messy-drive-rag

.venv/bin/python scripts/check_flag_manifest.py "$0" || exit 1

# Rebuild the question-independent stores the base config relies on (LLM-free, idempotent).
# image_ocr_store is NOT rebuilt here (its build needs vision for PDF pages → RAG_FORBID_GEMINI): the serve
# path reads the persisted artifact (artifacts/image_ocr_store.jsonl) baked at build time.
.venv/bin/python -m src.rag.index.derived_metrics
.venv/bin/python -m src.rag.index.visual_store
.venv/bin/python -m src.rag.index.action_row_store
.venv/bin/python -m src.rag.index.case_finance_store
.venv/bin/python -c "from src.rag.index import raw_artifact_store as s; r=s.build(); print('[build] raw_artifact_store', {k:r[k] for k in ('files','cases')})"
.venv/bin/python -c "from src.rag.index import report_attr_store as s; r=s.build()['report']; print('[build] report_attr_store', r)"
.venv/bin/python -c "from src.rag.index import diff_store as s; r=s.build(); print('[build] diff_store', r.get('pairs') if isinstance(r,dict) else r)"
HOME="${HOME:-/tmp}" .venv/bin/python -m src.rag.index.heading_page_store
# image_ocr_store is NOT rebuilt here (its PDF-page build needs vision → RAG_FORBID_GEMINI): the serve path
# reads the persisted artifact (artifacts/image_ocr_store.jsonl) baked at build time and merges it via the
# extractor, so live read_office/read_pdf/file_grep surface it with no genai call.
# Rebuild the lexical FTS + typed evidence indexes WITH the OCR stores enabled so *search* surfaces the
# image-OCR content (idx8 EMF給与表 / idx52「別契約」/ idx68「投資実装係数」). The champion indexes were built
# without OCR text, so search could not discover these image PDFs / the EMF table before. LLM-free
# (RAG_PDF_OCR unset ⇒ no live vision; the OCR comes from the persisted stores).
RAG_OCR_STORE=1 RAG_IMAGE_OCR_STORE=1 .venv/bin/python -c "from src.rag.index import text_fts as t; r=t.build(); print('[build] text_fts', {k:r.get(k) for k in ('records','docs')})"
RAG_OCR_STORE=1 RAG_IMAGE_OCR_STORE=1 .venv/bin/python -c "from src.rag.index import evidence_index as e; print('[build] evidence_index', e.build_only())"

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

# --- cycle-7 K2 promoted (ノートブック描画チャート) ---
export RAG_NB_CHART_STORE=1

# --- axis under test (SOT-2684, cycle7 K1) — image OCR merges via the EXTRACTOR (no new serve tool),
#     so live read_office/read_pdf/file_grep surface it. NOT enabling RAG_DOC_REACH_STORE: that adds two
#     lookup tools the investigator did not use and which destabilised a sentinel (idx21) in the first run.
export RAG_IMAGE_OCR_STORE=1

# --- investigator on flat-rate Sonnet; parallelism 1 = shared-limit protection ---
export RAG_INVESTIGATOR_BACKEND=claude-mcp
export RAG_CLAUDE_MCP_MODEL=sonnet
# MANDATORY: no resume cache — the serve change alters answers, so re-derive live (SOT-2664).
export RAG_CLAUDE_MCP_RESUME=0
export RAG_MCP_TOOL_LOG=artifacts/sonnet_child_image_ocr_tool_calls.jsonl

TARGET="${TARGET:-8,50,52,68}"

GOLD="${GOLD:-artifacts/predictions_test_v4_final.csv}"

echo "=== SOT-2684 Sonnet K1 image-ocr focused gate start $(date -u +%FT%TZ) ==="
echo "BACKEND=$RAG_INVESTIGATOR_BACKEND IMAGE_OCR=$RAG_IMAGE_OCR_STORE RESUME=$RAG_CLAUDE_MCP_RESUME TARGET=$TARGET GOLD=$GOLD"
.venv/bin/python scripts/run_focused_gate.py \
  --label sot_child_image_ocr \
  --target "$TARGET" \
  --sentinels scripts/sonnet_sentinels.json \
  --gold "$GOLD" \
  --issue SOT-2684 \
  --dev --no-smoke \
  --workers 1
echo "=== SOT-2684 Sonnet K1 image-ocr focused gate done $(date -u +%FT%TZ) exit=$? ==="
