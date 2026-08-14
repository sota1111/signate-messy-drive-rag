#!/usr/bin/env bash
# SOT-2710 (cycle11) — xlsx スケジュール/プラン既存ストア自動発火 決定論レーン focused gate
# (official:false lane).
#
# Base env = cycle9 champion のフラグ一式＋本子の新フラグ:
#   RAG_SCHEDULE_PLAN_LOOKUP=1 — schedule_plan_store が build 時に、案件ごとの 02.計画/スケジュール.xlsx と
#     00.提案/提案書.pptx を質問非依存で走査し、(a) 主シート（改訂優先→先頭）のタスク行 {フェーズNo/開始日/
#     種別/工数}、(b) 種別=バッファ行の工数総和、(c) 提案書ガントの塗り潰しバー割付 phase→week を全数焼く。
#     serve の決定論レーン (fact_layer 下位) が、案件を glossary.company_of で一意束縛でき属性が一意確定する時
#     だけ直答し、既存 visual_store（行ハイライト）/ plan_coverage_store（担当者別タスク数）も自動発火に接続
#     する（曖昧なら従来 LLM 経路へ defer）。gold 値ハードコードなし・全数事前計算のみ。
# Target: idx2（青嶺 スケジュール_r2.xlsx オレンジ行のタスク名列挙）/ idx41（AOBM=青葉バイオメディカル PLAN
#   加藤さん担当タスク数）/ idx75（MINAMINO=みなみ野 提案書ガント モデル構築の実施週）/ idx89（京橋 フェーズNo6
#   最終開始タスク名）/ idx90（青潮 バッファ工数合計）。5 idx を deterministic 経路へ昇格（details の model 確認）。
#
# 番兵: scripts/sonnet_sentinels.json。Gate FAILS on any sentinel regression (10/10)。
# GOLD = artifacts/predictions_test_v4_final.csv。 --dev ⇒ official:false。
#   RAG_CLAUDE_MCP_RESUME=0（回答が変わるレバーなので sidecar replay は変化を隠す）。Gemini $0（RAG_FORBID_GEMINI=1）。
set -euo pipefail
cd /workspaces/signate-messy-drive-rag

.venv/bin/python scripts/check_flag_manifest.py "$0" || exit 1

# Rebuild the question-independent stores (LLM-free, idempotent). contact_master_store is the new build
# this child adds; the rest mirror cycle9's rebuild union so the focused run matches the integrated base.
.venv/bin/python -m src.rag.index.derived_metrics
.venv/bin/python -m src.rag.index.raw_artifact_store
.venv/bin/python -m src.rag.index.analysis_metrics_enum_store
.venv/bin/python -m src.rag.index.doc_reach_store
.venv/bin/python -m src.rag.index.derived_ranking_store
.venv/bin/python -m src.rag.index.schedule_store
.venv/bin/python -m src.rag.index.visual_store
.venv/bin/python -m src.rag.index.action_row_store
.venv/bin/python -m src.rag.index.pptx_note_scope_store
.venv/bin/python -m src.rag.index.schedule_plan_store
.venv/bin/python -m src.rag.index.case_finance_store
.venv/bin/python -m src.rag.index.contact_master_store
.venv/bin/python -m src.rag.index.master_join_store
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

# --- cycle-11 lever (this child) ---
export RAG_SCHEDULE_PLAN_LOOKUP=1

# --- investigator on flat-rate Sonnet; parallelism 1 = shared-limit protection ---
export RAG_INVESTIGATOR_BACKEND=claude-mcp
export RAG_CLAUDE_MCP_MODEL=sonnet
export RAG_MCP_TOOL_LOG=artifacts/sot2710_tool_calls.jsonl
export RAG_CLAUDE_MCP_RESUME=0

TARGET="${TARGET:-2,41,75,89,90}"
GOLD="${GOLD:-artifacts/predictions_test_v4_final.csv}"

echo "=== SOT-2710 cycle11 schedule/plan auto-fire focused gate start $(date -u +%FT%TZ) ==="
echo "BACKEND=$RAG_INVESTIGATOR_BACKEND SCHEDULE_PLAN=$RAG_SCHEDULE_PLAN_LOOKUP RESUME=$RAG_CLAUDE_MCP_RESUME TARGET=$TARGET GOLD=$GOLD"
.venv/bin/python scripts/run_focused_gate.py \
  --label sot2710_schedule_plan \
  --target "$TARGET" \
  --sentinels scripts/sonnet_sentinels.json \
  --gold "$GOLD" \
  --issue SOT-2710 \
  --dev --no-smoke \
  --workers 1
echo "=== SOT-2710 cycle11 schedule/plan auto-fire focused gate done $(date -u +%FT%TZ) exit=$? ==="
