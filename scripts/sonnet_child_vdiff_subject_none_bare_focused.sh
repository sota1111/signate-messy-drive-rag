#!/usr/bin/env bash
# SOT-2665 — focused gate for cycle5 C2 (version-diff 候補選択/文書名プレフィクス + 「該当なし」裸形式, official:false).
#
# Base = scripts/sonnet_gold_manual_20260812.sh (cycle4 net48 実証済み構成: champion Wave A+B1 + cycle-4
# stores + Cerebras stack + RAG_BARE_ANSWER) PLUS the two C2 levers under test:
#   RAG_VDIFF_SUBJECT=1  — det version_diff pipeline commits a lone substantive *block insert* and prefixes
#                          the answer with the document name (文書名) recovered from the latest filename
#                          stem (idx0:「提案書スライド6『…』に、…が追記された」— champion LLM was value-perfect
#                          but dropped the「提案書」プレフィクス). serve-path gated, default OFF.
#   RAG_NONE_BARE=1      — bare「該当なし」none-answer contract (prompt-only): a confirmed非存在 answer is
#                          「該当なし」のみ, no 列挙/注記 (idx9 列挙落ち / idx85「なし(全6項目達成)」).
# Targets (gold=artifacts/predictions_test_v3_final.csv): 一次 idx0/9/14/85, 二次 idx1(remove要約)/95/98.
# Gate FAILS on any Sonnet sentinel regression. --dev ⇒ official:false (non-flash stack; NOT a promotion
# basis). NO resume cache: the prompt/serve change alters answers, so target AND sentinels re-derive live.
set -euo pipefail
cd /workspaces/signate-messy-drive-rag

.venv/bin/python scripts/check_flag_manifest.py "$0" || exit 1

# Rebuild the question-independent stores the base config relies on (LLM-free, idempotent). diff_store is
# added here since fact_layer diff_lookup (idx14/95/98 class) reads it.
.venv/bin/python -m src.rag.index.visual_store
.venv/bin/python -m src.rag.index.action_row_store
.venv/bin/python -m src.rag.index.case_finance_store
.venv/bin/python -c "from src.rag.index import report_attr_store as s; r=s.build()['report']; print('[build] report_attr_store', r)"
.venv/bin/python -c "from src.rag.index import diff_store as s; r=s.build(); print('[build] diff_store', r.get('pairs') if isinstance(r,dict) else r)"

export VERTEX_LOCATION=global
export GEN_MODEL=gemini-3.6-flash
export GEN_MODEL_HARD=gemini-3.6-flash
export VISION_MODEL=gemini-3.6-flash
export PYTHONPATH=/tmp/genai_patch:.

# --- champion Wave A + B1 flags, verbatim from scripts/sonnet_gold_manual_20260812.sh ---
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

# --- cycle4.5 manual interim (Cerebras stack + bare-answer) ---
export RAG_TEXT_FTS=1
export RAG_UNIFIED_SEARCH=1
export RAG_PLAN_FANOUT=1
export RAG_BARE_ANSWER=1

# --- axis under test (SOT-2665, cycle5 C2) ---
export RAG_VDIFF_SUBJECT=1
export RAG_NONE_BARE=1

# --- investigator on flat-rate Sonnet; parallelism 1 = shared-limit protection ---
export RAG_INVESTIGATOR_BACKEND=claude-mcp
export RAG_CLAUDE_MCP_MODEL=sonnet
export RAG_MCP_TOOL_LOG=artifacts/sonnet_child_vdiff_subject_none_bare_tool_calls.jsonl

TARGET="${TARGET:-0,1,9,14,85,95,98}"

echo "=== SOT-2665 Sonnet C2 vdiff-subject/none-bare focused gate start $(date -u +%FT%TZ) ==="
echo "BACKEND=$RAG_INVESTIGATOR_BACKEND VDIFF_SUBJECT=$RAG_VDIFF_SUBJECT NONE_BARE=$RAG_NONE_BARE TARGET=$TARGET"
.venv/bin/python scripts/run_focused_gate.py \
  --label sot_child_vdiff_subject_none_bare \
  --target "$TARGET" \
  --sentinels scripts/sonnet_sentinels.json \
  --issue SOT-2665 \
  --dev --no-smoke \
  --workers 1
echo "=== SOT-2665 Sonnet C2 focused gate done $(date -u +%FT%TZ) exit=$? ==="
