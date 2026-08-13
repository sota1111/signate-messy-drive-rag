#!/usr/bin/env bash
# SOT-2703 (cycle10) — 複合書式ファクトストア + 決定論 lookup レーン focused gate (official:false lane).
#
# Base env = scripts/sonnet_gold_cycle9.sh のフラグ一式（net84+ 実証済み）＋本子の新フラグ:
#   RAG_FORMAT_FACTS=1 — format_facts_store が build 時に、全 docx の書式付き run を質問非依存で全数
#     {project, rel, doc_kind, locator, text, attrs:{highlight,font_color,bold,underline,italic}} で焼き、
#     serve の決定論 lookup レーン (fact_layer 下位) が「黄色ハイライトかつ赤字」「太字/下線/イタリックの
#     すべて」型質問へ、属性集合∧project∧doc-kind（本文自己同定『中間報告』含む）で一意に絞れた時だけ
#     run テキストを逐語直答する（複数一致/ゼロ一致は従来経路へ defer）。
# Target: idx16（青葉与信マネジメント 中間報告資料 黄ハイライト×赤字 → gold=0.589; 同案件の報告資料は
#   2 本ありどちらも黄×赤を持つため 中間報告 自己同定で 04-29 に一意化）＋ idx71（青嶺不動産アセット
#   マネジメント 会議録 太字∧下線∧イタリック → gold=4,250,000円; 会議録 3 本中 1 本のみ該当）。
#   原ファイルの run テキストのみから決定論導出（gold 参照なし・ハードコードなし）。
#
# 番兵: scripts/sonnet_sentinels.json (idx58 版)。Gate FAILS on any sentinel regression (10/10)。
# GOLD = artifacts/predictions_test_v4_final.csv。 --dev ⇒ official:false。
#   RAG_CLAUDE_MCP_RESUME=0（回答が変わるレバーなので sidecar replay は変化を隠す）。Gemini $0（RAG_FORBID_GEMINI=1）。
set -euo pipefail
cd /workspaces/signate-messy-drive-rag

.venv/bin/python scripts/check_flag_manifest.py "$0" || exit 1

# Rebuild the question-independent stores (LLM-free, idempotent; cycle9 rebuild union) PLUS this child's
# format_facts_store (全 docx の書式付き run を焼く; serve reads the persisted store when RAG_FORMAT_FACTS=1).
.venv/bin/python -m src.rag.index.derived_metrics
.venv/bin/python -m src.rag.index.raw_artifact_store
.venv/bin/python -m src.rag.index.analysis_metrics_enum_store
.venv/bin/python -m src.rag.index.doc_reach_store
.venv/bin/python -m src.rag.index.derived_ranking_store
.venv/bin/python -m src.rag.index.schedule_store
.venv/bin/python -m src.rag.index.visual_store
.venv/bin/python -m src.rag.index.format_facts_store
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
export RAG_FORMAT_FACTS=1

# --- investigator on flat-rate Sonnet; parallelism 1 = shared-limit protection ---
export RAG_INVESTIGATOR_BACKEND=claude-mcp
export RAG_CLAUDE_MCP_MODEL=sonnet
export RAG_MCP_TOOL_LOG=artifacts/sot2703_tool_calls.jsonl
export RAG_CLAUDE_MCP_RESUME=0

TARGET="${TARGET:-16,71}"
GOLD="${GOLD:-artifacts/predictions_test_v4_final.csv}"

echo "=== SOT-2703 cycle10 format-facts focused gate start $(date -u +%FT%TZ) ==="
echo "BACKEND=$RAG_INVESTIGATOR_BACKEND FORMAT_FACTS=$RAG_FORMAT_FACTS RESUME=$RAG_CLAUDE_MCP_RESUME TARGET=$TARGET GOLD=$GOLD"
.venv/bin/python scripts/run_focused_gate.py \
  --label sot2703_format_facts \
  --target "$TARGET" \
  --sentinels scripts/sonnet_sentinels.json \
  --gold "$GOLD" \
  --issue SOT-2703 \
  --dev --no-smoke \
  --workers 1
echo "=== SOT-2703 cycle10 format-facts focused gate done $(date -u +%FT%TZ) exit=$? ==="
