#!/usr/bin/env bash
# SOT-2657 — focused gate for the full-corpus lexical index child (Cerebras 型検索基盤 1/5, official:false).
#
# Same champion env as scripts/sonnet_child_action_row_focused.sh plus the SOT-2657 lever:
#   RAG_TEXT_FTS=1  — consult the build-time SQLite FTS5 + IDF lexical index at serve time and expose
#                     the `text_search` investigator tool (index built by `RAG_TEXT_FTS_BUILD=1
#                     .venv/bin/python -m src.rag.index.text_fts` → artifacts/text_fts.db). Literal/字句
#                     discovery (エラー文字列・フラグ名・ID・固有語) is answered by a millisecond IDF-ranked
#                     index lookup instead of file_grep re-extracting the whole corpus (BUDGET32 spin 対策).
# Targets: the file_grep-spin abstain idx the index is meant to un-stick — the "針" should be a 1発ヒット:
#   idx34 (A08/A09 会議録アクションID), idx76 (増加額の根拠行), idx93 (0値を疑似欠損 前処理), idx98, idx99.
# Gate FAILS on any Sonnet sentinel regression. --dev ⇒ official:false (non-flash stack, cannot be a
# production non-regression basis; the OFF path is byte-identical so production is unaffected regardless).
set -euo pipefail
cd /workspaces/signate-messy-drive-rag

.venv/bin/python scripts/check_flag_manifest.py "$0" || exit 1

export VERTEX_LOCATION=global
export GEN_MODEL=gemini-3.6-flash
export GEN_MODEL_HARD=gemini-3.6-flash
export VISION_MODEL=gemini-3.6-flash
export PYTHONPATH=/tmp/genai_patch:.

# --- champion Wave A + B1 flags, verbatim from sonnet_child_action_row_focused.sh ---
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

# --- cycle-2/3/4 levers (kept) ---
export RAG_FACT_LAYER=1
export RAG_FORBID_GEMINI=1
export LLM_PROVIDER=claude-cli
export CLAUDE_CLI_MODEL=sonnet
export RAG_OCR_STORE=1
export RAG_FORMAT_STRIP_PAREN=1
export RAG_ACTION_ROW_STORE=1

# --- SOT-2657 lever: full-corpus lexical index + text_search tool ---
export RAG_TEXT_FTS=1

# --- investigator on flat-rate Sonnet (SOT-2627); parallelism 1 = shared-limit protection ---
export RAG_INVESTIGATOR_BACKEND=claude-mcp
export RAG_CLAUDE_MCP_MODEL=sonnet
export RAG_MCP_TOOL_LOG=artifacts/sonnet_child_text_fts_tool_calls.jsonl

TARGET="${TARGET:-34,76,93,98,99}"

echo "=== SOT-2657 Sonnet text_fts focused gate start $(date -u +%FT%TZ) ==="
echo "BACKEND=$RAG_INVESTIGATOR_BACKEND FACT_LAYER=$RAG_FACT_LAYER TEXT_FTS=$RAG_TEXT_FTS TARGET=$TARGET"
.venv/bin/python scripts/run_focused_gate.py \
  --label sot_child_text_fts \
  --target "$TARGET" \
  --sentinels scripts/sonnet_sentinels.json \
  --issue SOT-2657 \
  --dev --no-smoke \
  --workers 1
echo "=== SOT-2657 Sonnet text_fts focused gate done $(date -u +%FT%TZ) exit=$? ==="
