#!/usr/bin/env bash
# SOT-2686 — focused gate for cycle7 K3 (xlsx 数式依存トレース＋記載回帰係数の行適用, official:false).
#
# Base = scripts/sonnet_child_wrong_det_focused.sh (cycle7 net68 実証構成) PLUS the one K3 lever:
#   RAG_XLSX_FORMULA_TRACE=1 — build 時に焼いた xlsx_formula_trace ストアを serve へ質問非依存に決定論束縛
#     する直答レーン(route=deterministic; 投資者ツールは足さない ⇒ LLM サーフェスは champion と byte-identical):
#     (idx47) 黄色ハイライト数式セル(誤差 (予測−実測)^2)が参照するデータ行を辿り、その行の属性
#             「建設年」→ YEAR BUILT を返す(青嶺 B22→Sheet1 行26118→1899年)。
#     (idx83) 記載回帰係数(切片＋列名付き係数)を index=N 行へ当てはめた予測値(事前計算)を小数指定桁で返す
#             (みなみ野 index=1770→0.38317)。
#     serve gated・既定 OFF ⇒ OFF 時 byte-identical。参照行/係数表/index が一意束縛できない時のみ発火。
# Targets (gold=artifacts/predictions_test_v3_final.csv): 一次 idx47/83。stretch idx17(docx ハイライト×赤字)
#   は「MM」対象ファイル群の定義が曖昧で一意束縛できないため本レーン非対象(honest abstain)。
# Gate FAILS on any Sonnet sentinel regression. --dev ⇒ official:false (non-flash stack; NOT a promotion
# basis). RAG_CLAUDE_MCP_RESUME=0 必須: serve change alters answers, re-derive target AND sentinels live.
set -euo pipefail
cd /workspaces/signate-messy-drive-rag

.venv/bin/python scripts/check_flag_manifest.py "$0" || exit 1

# Rebuild the question-independent stores the base config + this axis rely on (LLM-free, idempotent).
# xlsx_formula_trace is (re)built because the K3 lane reads its highlight_formulas / regressions records.
.venv/bin/python -m src.rag.index.derived_metrics
.venv/bin/python -m src.rag.index.visual_store
.venv/bin/python -m src.rag.index.action_row_store
.venv/bin/python -m src.rag.index.case_finance_store
.venv/bin/python -c "from src.rag.index import report_attr_store as s; r=s.build()['report']; print('[build] report_attr_store', r)"
.venv/bin/python -c "from src.rag.index import diff_store as s; r=s.build(); print('[build] diff_store', r.get('pairs') if isinstance(r,dict) else r)"
HOME="${HOME:-/tmp}" .venv/bin/python -m src.rag.index.heading_page_store
.venv/bin/python -m src.rag.index.xlsx_formula_trace

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

# --- axis under test (SOT-2686, cycle7 K3) ---
export RAG_XLSX_FORMULA_TRACE=1

# --- investigator on flat-rate Sonnet; parallelism 1 = shared-limit protection ---
export RAG_INVESTIGATOR_BACKEND=claude-mcp
export RAG_CLAUDE_MCP_MODEL=sonnet
# MANDATORY: no resume cache — the serve change alters answers, so re-derive live (SOT-2664).
export RAG_CLAUDE_MCP_RESUME=0
export RAG_MCP_TOOL_LOG=artifacts/sonnet_child_xlsx_formula_tool_calls.jsonl

TARGET="${TARGET:-47,83}"

echo "=== SOT-2686 Sonnet K3 xlsx-formula-trace focused gate start $(date -u +%FT%TZ) ==="
echo "BACKEND=$RAG_INVESTIGATOR_BACKEND XLSX_FORMULA_TRACE=$RAG_XLSX_FORMULA_TRACE RESUME=$RAG_CLAUDE_MCP_RESUME TARGET=$TARGET"
.venv/bin/python scripts/run_focused_gate.py \
  --label sot_child_xlsx_formula \
  --target "$TARGET" \
  --sentinels scripts/sonnet_sentinels.json \
  --issue SOT-2686 \
  --dev --no-smoke \
  --workers 1
echo "=== SOT-2686 Sonnet K3 xlsx-formula-trace focused gate done $(date -u +%FT%TZ) exit=$? ==="
