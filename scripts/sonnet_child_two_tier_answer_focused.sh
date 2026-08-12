#!/usr/bin/env bash
# SOT-2670 — focused gate for the two-tier answer schema child (上位解法移植 2/4, official:false).
#
# Same champion dev env as scripts/sonnet_child_heading_page_focused.sh, MINUS the cycle-5 heading-page
# lever, PLUS the axis under test:
#   RAG_TWO_TIER_ANSWER=1  — submit_answer advertises full_answer (reasoning-bearing) + bare_answer
#                            (the scored value; labels/titles/units copied verbatim; 確定非存在=「該当なし」).
#                            The scored answer is bare_answer, so a prompt-contract deviation is prevented
#                            structurally, not just by prompt.
# Targets (部分値・冗長・該当なしクラス, gold=artifacts/predictions_test_v3_final.csv):
#   idx21 (役職名の切り詰め: 「部長」← 完全な役職名) / idx62 (ラベル欠落) / idx78 (冗長)
#   idx9, idx85 (「該当なし」型: 内容併記落ち)
# Gate FAILS on any Sonnet sentinel regression. --dev ⇒ official:false (non-flash stack). NO resume cache:
# the schema change alters every LLM answer, so target AND sentinels must be re-derived live.
set -euo pipefail
cd /workspaces/signate-messy-drive-rag

.venv/bin/python scripts/check_flag_manifest.py "$0" || exit 1

export VERTEX_LOCATION=global
export GEN_MODEL=gemini-3.6-flash
export GEN_MODEL_HARD=gemini-3.6-flash
export VISION_MODEL=gemini-3.6-flash
export PYTHONPATH=/tmp/genai_patch:.

# --- champion Wave A + B1 flags, verbatim from sonnet_child_heading_page_focused.sh ---
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

# --- axis under test (SOT-2670) ---
export RAG_TWO_TIER_ANSWER=1

# --- investigator on flat-rate Sonnet; parallelism 1 = shared-limit protection ---
export RAG_INVESTIGATOR_BACKEND=claude-mcp
export RAG_CLAUDE_MCP_MODEL=sonnet
export RAG_MCP_TOOL_LOG=artifacts/sonnet_child_two_tier_answer_tool_calls.jsonl

TARGET="${TARGET:-21,62,78,9,85}"

echo "=== SOT-2670 Sonnet two-tier-answer focused gate start $(date -u +%FT%TZ) ==="
echo "BACKEND=$RAG_INVESTIGATOR_BACKEND FACT_LAYER=$RAG_FACT_LAYER TWO_TIER_ANSWER=$RAG_TWO_TIER_ANSWER TARGET=$TARGET"
.venv/bin/python scripts/run_focused_gate.py \
  --label sot_child_two_tier_answer \
  --target "$TARGET" \
  --sentinels scripts/sonnet_sentinels.json \
  --issue SOT-2670 \
  --dev --no-smoke \
  --workers 1
echo "=== SOT-2670 Sonnet two-tier-answer focused gate done $(date -u +%FT%TZ) exit=$? ==="
