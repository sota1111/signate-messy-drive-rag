#!/usr/bin/env bash
# SOT-2698 (cycle9) — 分析出力メタデータ (metrics.json enum / 適用ハイパラ) 決定論レーン focused gate.
#
# Base env = the cycle-8 INTEGRATED config (net84 実証済み) + cycle-9 の vdiff-struct を含む cycle9 union
# PLUS two new levers, default OFF ⇒ OFF byte-identical:
#   RAG_ANALYSIS_METRICS_ENUM=1 — analysis_metrics_enum_store を build 時に焼く。metrics.json
#       feature_selection.selected_columns から数値交互作用特徴量 (__x__ 列) の部分集合を per-case で確定。
#       idx32 = 青嶺不動産 = BOROUGH__x__BLOCK / BOROUGH__x__LOT / BOROUGH__x__ZIP CODE / BLOCK__x__LOT /
#       BLOCK__x__ZIP CODE / LOT__x__ZIP CODE の 6 列。
#   RAG_ANALYSIS_CONFIG_HYPERPARAM=1 — raw_artifact_store の per-case applied_hyperparams (config.model_params
#       ＋ modeling.py コード上デフォルトのマージ) から、質問が名指ししたパラメタを質問中の出現順で 名前=値。
#       idx61 = 京橋信用ソリューションズ = n_estimators=300、learning_rate=0.1、random_state=42。
#   両レーンとも fact_layer.resolve の決定論後置レーンで直答。OFF ⇒ resolve/tool は None (byte-identical)。
#
# 番兵: scripts/sonnet_sentinels.json (idx58 版, SOT-2695)。GATE は番兵回帰 (10/10 必須) でのみ FAIL。
# 対象 idx: 32 (交互作用特徴量列名) / 61 (適用ハイパラ)。gold ハードコードなし・per-idx 分岐なし
#   (store は全案件×全 metrics.json enum フィールドを質問非依存に網羅抽出。lane は質問が名指しした対象だけ返す)。
# --dev ⇒ official:false (flash 非スタック; 昇格根拠にしない)。
#
# GOLD = artifacts/predictions_test_v4_final.csv (cycle9 gold)。
# NO resume cache (RAG_CLAUDE_MCP_RESUME=0): resume key は (model, question) のみで config 非依存
#   (SOT-2664 gotcha)、本レバーは回答を変えるので sidecar replay は変化を隠す。Gemini $0 (RAG_FORBID_GEMINI=1)。
set -euo pipefail
cd /workspaces/signate-messy-drive-rag

.venv/bin/python scripts/check_flag_manifest.py "$0" || exit 1

# Rebuild the question-independent stores (LLM-free, idempotent; cycle-9 union). The new
# analysis_metrics_enum_store と raw_artifact_store (applied_hyperparams の証拠) を確実に焼く。
.venv/bin/python -m src.rag.index.derived_metrics
.venv/bin/python -m src.rag.index.raw_artifact_store
.venv/bin/python -m src.rag.index.analysis_metrics_enum_store
.venv/bin/python -m src.rag.index.doc_reach_store
.venv/bin/python -m src.rag.index.schedule_store
.venv/bin/python -m src.rag.index.visual_store
.venv/bin/python -m src.rag.index.action_row_store
.venv/bin/python -m src.rag.index.case_finance_store
.venv/bin/python -c "from src.rag.index import report_attr_store as s; r=s.build()['report']; print('[build] report_attr_store', r)"
RAG_VDIFF_STRUCT=1 RAG_VDIFF_CLASSIFY=1 .venv/bin/python -c "from src.rag.index import diff_store as s; r=s.build(); print('[build] diff_store (RAG_VDIFF_STRUCT=1 RAG_VDIFF_CLASSIFY=1)', r.get('pairs') if isinstance(r,dict) else r)"
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

# --- cycle-9 levers (prior children) ---
export RAG_VDIFF_STRUCT=1

# --- cycle-9 this child (SOT-2698) ---
export RAG_ANALYSIS_METRICS_ENUM=1
export RAG_ANALYSIS_CONFIG_HYPERPARAM=1

# --- investigator on flat-rate Sonnet; parallelism 1 = shared-limit protection ---
export RAG_INVESTIGATOR_BACKEND=claude-mcp
export RAG_CLAUDE_MCP_MODEL=sonnet
export RAG_MCP_TOOL_LOG=artifacts/sot2698_tool_calls.jsonl
export RAG_CLAUDE_MCP_RESUME=0

TARGET="${TARGET:-32,61}"
GOLD="${GOLD:-artifacts/predictions_test_v4_final.csv}"

echo "=== SOT-2698 cycle9 analysis-metrics focused gate start $(date -u +%FT%TZ) ==="
echo "BACKEND=$RAG_INVESTIGATOR_BACKEND METRICS_ENUM=$RAG_ANALYSIS_METRICS_ENUM HYPERPARAM=$RAG_ANALYSIS_CONFIG_HYPERPARAM RESUME=$RAG_CLAUDE_MCP_RESUME TARGET=$TARGET GOLD=$GOLD"
.venv/bin/python scripts/run_focused_gate.py \
  --label sot2698_analysis_metrics \
  --target "$TARGET" \
  --sentinels scripts/sonnet_sentinels.json \
  --gold "$GOLD" \
  --issue SOT-2698 \
  --dev --no-smoke \
  --workers 1
echo "=== SOT-2698 cycle9 analysis-metrics focused gate done $(date -u +%FT%TZ) exit=$? ==="
