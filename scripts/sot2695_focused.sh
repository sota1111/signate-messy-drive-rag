#!/usr/bin/env bash
# SOT-2695 (cycle8 C6) — vdiff 実質変更分類の決定論修正＋番兵安定化 focused gate (official:false lane).
#
# Base env = the cycle-7 integrated config (net72 proven) PLUS the two cycle-8 C6 levers, both default
# OFF ⇒ OFF byte-identical:
#   RAG_VDIFF_CLASSIFY=1 — diffpair の追加分類規則 (baked into diff_store at build):
#       * 見出しラベル追加・スライド分割・体裁再構成 (numbered heading + 「― …（短期/中期/長期）」) を
#         LAYOUT_METADATA に降格 → idx9 (gold「該当なし」) が SUBSTANTIVE 0 件になる。
#       * データ列名の underscore 表記化 (`a b`→`a_b`, old にのみスペース形) を SUBSTANTIVE 昇格・最上位
#         rank に前置 → idx14 (gold = loan_status/employment_length/application_type/interest_rate)。
#       claude_mcp の system suffix にも対の言い回し契約 (体裁のみ→該当なし／列名変更が最優先) を付与。
#   RAG_REPORT_SERIES=1  — 報告資料の系列スコープ＋最小集合契約 (prompt-only, claude_mcp suffix):
#       各報告資料の自己申告チェックポイント (M01=キックオフ／M02=中間レビュー『中間分析報告』／最終報告)
#       で系列を質問非依存に判定し、『中間報告資料』なら中間該当文書のみから最小集合抽出 → idx16
#       (05.会議/報告資料 の 04-09=M01キックオフ=4,620,000 を除外、04-29=M02中間=0.589 のみ)。
#
# 番兵: scripts/sonnet_sentinels.json の idx21 を idx58 に置換済み (SOT-2695)。idx21 は cycle7 で WRONG に
#   落ちた flaky LLM 番兵 (dev/cycle4/cycle6 は MATCH)。idx58 は fact_lookup の決定論 Wave-A レーンで
#   byte-exact gold (7102)、4 Sonnet サンプル全 MATCH・model 不変。det/llm 3/7 → 4/6。
#
# GOLD = artifacts/predictions_test_v4_final.csv (v4 は v3 と idx7/idx8 のみ差異 — 対象 9/14/16 も番兵も
#   v3/v4 同値・番兵に idx8 なし)。gold ハードコードなし (idx16 の scope は文書自己申告メタから決定論確定)。
#
# Gate FAILS on any Sonnet sentinel regression (10/10 必須)。--dev ⇒ official:false (昇格根拠にしない)。
# NO resume cache (RAG_CLAUDE_MCP_RESUME=0): resume key は (model, question) のみで config 非依存
#   (SOT-2664 gotcha)、本レバーは回答を変えるので sidecar replay は変化を隠す。Gemini $0 (RAG_FORBID_GEMINI=1)。
set -euo pipefail
cd /workspaces/signate-messy-drive-rag

.venv/bin/python scripts/check_flag_manifest.py "$0" || exit 1

# Rebuild the question-independent stores (LLM-free, idempotent). The diff_store MUST be rebuilt with
# RAG_VDIFF_CLASSIFY=1 so the new edit-intent classification is baked into diff_store.jsonl (serve reads
# the persisted store). image_ocr / nb_chart / ocr stores are NOT rebuilt (their builds need vision):
# serve reads the persisted artifacts baked at build time (Gemini $0 at serve).
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
RAG_OCR_STORE=1 RAG_IMAGE_OCR_STORE=1 .venv/bin/python -c "from src.rag.index import text_fts as t; r=t.build(); print('[build] text_fts', {k:r.get(k) for k in ('records','docs')})"
RAG_OCR_STORE=1 RAG_IMAGE_OCR_STORE=1 .venv/bin/python -c "from src.rag.index import evidence_index as e; print('[build] evidence_index', e.build_only())"

export VERTEX_LOCATION=global
export GEN_MODEL=gemini-3.6-flash
export GEN_MODEL_HARD=gemini-3.6-flash
export VISION_MODEL=gemini-3.6-flash
export PYTHONPATH=/tmp/genai_patch:.

# --- champion Wave A + B1 flags, verbatim from scripts/sonnet_gold_cycle7.sh ---
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

# --- cycle-8 C6 levers (this child) ---
export RAG_VDIFF_CLASSIFY=1
export RAG_REPORT_SERIES=1

# --- investigator on flat-rate Sonnet; parallelism 1 = shared-limit protection ---
export RAG_INVESTIGATOR_BACKEND=claude-mcp
export RAG_CLAUDE_MCP_MODEL=sonnet
export RAG_MCP_TOOL_LOG=artifacts/sot2695_tool_calls.jsonl
export RAG_CLAUDE_MCP_RESUME=0

TARGET="${TARGET:-9,14,16}"
GOLD="${GOLD:-artifacts/predictions_test_v4_final.csv}"

echo "=== SOT-2695 cycle8 C6 focused gate start $(date -u +%FT%TZ) ==="
echo "BACKEND=$RAG_INVESTIGATOR_BACKEND VDIFF_CLASSIFY=$RAG_VDIFF_CLASSIFY REPORT_SERIES=$RAG_REPORT_SERIES RESUME=$RAG_CLAUDE_MCP_RESUME TARGET=$TARGET GOLD=$GOLD"
.venv/bin/python scripts/run_focused_gate.py \
  --label sot2695_vdiff_classify_sentinel \
  --target "$TARGET" \
  --sentinels scripts/sonnet_sentinels.json \
  --gold "$GOLD" \
  --issue SOT-2695 \
  --dev --no-smoke \
  --workers 1
echo "=== SOT-2695 cycle8 C6 focused gate done $(date -u +%FT%TZ) exit=$? ==="
