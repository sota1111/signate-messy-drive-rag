#!/usr/bin/env bash
# SOT-2699 (cycle9) — 段階メトリクス全精度＋統計表 rank/ratio focused gate (official:false lane).
#
# Base env = cycle8 integrated config (net84) PLUS the two cycle9 levers, default OFF ⇒ OFF byte-identical:
#   RAG_STAGED_METRICS=1 — 段階メトリクス フル精度 F1 差 (analysis_xref_store.staged_metrics を拡張):
#       05.会議/報告資料(docx/pptx)×全メトリクス名×全フル精度数値を網羅抽出し、interim 段階の
#       フル精度 f1_macro と metrics.json:f1_macro をペア化して |最終−中間| を全精度で焼く。
#       idx36 = 恒一会 かえで総合病院 = |0.8291582445227382 − 0.7329671168078127| = 0.09619112771492555。
#       フル精度が焼けた案件だけ発火（丸め値で近似回答しない, SOT-2687 の教訓 = honest abstain）。
#   RAG_DERIVED_RANKING=1 — 統計表 rank/ratio (derived_ranking_store 新設 + serve レーン):
#       doc_reach_store の全統計表を header 名で数値系列化(分割ランキング表は同一 header 列を統合)し、
#       昇順/降順ソート済み rank-k の値から 序数(最も高い/N番目に低い)×比(何倍)/差 を丸め指定つきで直答。
#       idx99 = 蒼樹会 みなみ野女性医療センター 死亡率 最高(青森 18.2) ÷ 4番目に低い(滋賀 7.3) = 2.49。
#       serve は fact_layer.resolve の決定論後置レーンで直答。OFF ⇒ resolve/tool は None (byte-identical)。
#
# 番兵: scripts/sonnet_sentinels.json (idx58 版, SOT-2695)。Gate FAILS on any sentinel regression (10/10)。
# GOLD = artifacts/predictions_test_v3_final.csv (sonnet_sentinels と同一)。gold ハードコードなし
#   (idx36=05.会議/報告資料 のフル精度 f1_macro 抽出×metrics.json 差、idx99=統計表の rank-k 決定論比)。
#
# --dev ⇒ official:false (昇格根拠にしない)。RAG_CLAUDE_MCP_RESUME=0 (本レバーは回答を変えるので
#   sidecar replay は変化を隠す)。Gemini $0 (RAG_FORBID_GEMINI=1)。
set -euo pipefail
cd /workspaces/signate-messy-drive-rag

.venv/bin/python scripts/check_flag_manifest.py "$0" || exit 1

# Rebuild the question-independent stores (LLM-free, idempotent). cycle9 の derived_ranking_store は
# doc_reach_store の出力に依存するので doc_reach_store の直後に焼く。analysis_xref_store は staged_metrics
# 拡張(SOT-2699)込みで再構築する。
.venv/bin/python -m src.rag.index.derived_metrics
.venv/bin/python -m src.rag.index.raw_artifact_store
.venv/bin/python -m src.rag.index.doc_reach_store
.venv/bin/python -m src.rag.index.derived_ranking_store
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
.venv/bin/python -m src.rag.index.plan_coverage_store
.venv/bin/python -m src.rag.index.analysis_xref_store
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

# --- cycle-8 levers (integrated, net84) ---
export RAG_ANALYSIS_XREF=1
export RAG_PLAN_COVERAGE=1
export RAG_FORMAT_SERIES=1
export RAG_RATE_TABLE=1
export RAG_FORMULA_APPLY=1
export RAG_VDIFF_CLASSIFY=1
export RAG_REPORT_SERIES=1

# --- cycle-9 levers (this child) ---
export RAG_STAGED_METRICS=1
export RAG_DERIVED_RANKING=1

# --- investigator on flat-rate Sonnet; parallelism 1 = shared-limit protection ---
export RAG_INVESTIGATOR_BACKEND=claude-mcp
export RAG_CLAUDE_MCP_MODEL=sonnet
export RAG_MCP_TOOL_LOG=artifacts/sot2699_tool_calls.jsonl
export RAG_CLAUDE_MCP_RESUME=0

TARGET="${TARGET:-36,99}"
GOLD="${GOLD:-artifacts/predictions_test_v3_final.csv}"

echo "=== SOT-2699 cycle9 focused gate start $(date -u +%FT%TZ) ==="
echo "BACKEND=$RAG_INVESTIGATOR_BACKEND STAGED_METRICS=$RAG_STAGED_METRICS DERIVED_RANKING=$RAG_DERIVED_RANKING RESUME=$RAG_CLAUDE_MCP_RESUME TARGET=$TARGET GOLD=$GOLD"
.venv/bin/python scripts/run_focused_gate.py \
  --label sot2699_staged_metrics_derived_ranking \
  --target "$TARGET" \
  --sentinels scripts/sonnet_sentinels.json \
  --gold "$GOLD" \
  --issue SOT-2699 \
  --dev --no-smoke \
  --workers 1
echo "=== SOT-2699 cycle9 focused gate done $(date -u +%FT%TZ) exit=$? ==="
