#!/usr/bin/env bash
# SOT-2651 — cycle-4 INTEGRATED focused gate (official:false / Sonnet lane).
#
# Union of the five cycle-4 child levers on top of the cycle-3 champion base, measured together over
# the union of all child target idx + the Sonnet sentinel 10. Child-level evidence (each lever alone,
# sentinels 10/10): SOT-2652 action-row (20/70/93), SOT-2653 visual store (39/65/97/82),
# SOT-2654 case finance (37/40/55/76), SOT-2655 report attr (5/28/64), SOT-2656 value-norm
# (4/8/41/59/88/92). This gate checks the levers do not interfere when combined.
#
# Cerebras-stack flags (SOT-2657 FTS / SOT-2659 unified search / SOT-2661 plan fanout) stay OFF:
# their own focused gates showed zero target-side wins (34/76/93/98 Missing or Incorrect), so there is
# no focused evidence to include them (parent-issue comment 2026-08-12 permits inclusion only on
# focused evidence).
#
# Resume sidecar = artifacts/gold100_sonnet_cycle4_resume.jsonl — the reviewed cycle-4 Sonnet cache
# CURATED for this cycle: all cycle-4 target-idx questions and all non-answered (timeout) records were
# evicted so every changed path measures fresh under this config; the remaining 57 reviewed
# unchanged-lane answers (incl. the LLM sentinels) replay from cache (cycle-3 lesson: cached stable
# sentinels + fresh changed-path probes). Deterministic lanes bypass the cache entirely.
#
# Gate FAILS on any Sonnet sentinel regression. --dev ⇒ official:false (non-flash stack).
set -euo pipefail
cd /workspaces/signate-messy-drive-rag

.venv/bin/python scripts/check_flag_manifest.py "$0" || exit 1

# Rebuild the four cycle-4 stores on disk (LLM-free, question-independent, idempotent) so the serve
# lanes read current artifacts.
.venv/bin/python -m src.rag.index.visual_store
.venv/bin/python -m src.rag.index.action_row_store
.venv/bin/python -m src.rag.index.case_finance_store
.venv/bin/python -c "from src.rag.index import report_attr_store as s; r=s.build()['report']; print('[build] report_attr_store', r)"

export VERTEX_LOCATION=global
export GEN_MODEL=gemini-3.6-flash
export GEN_MODEL_HARD=gemini-3.6-flash
export VISION_MODEL=gemini-3.6-flash
export PYTHONPATH=/tmp/genai_patch:.

# --- champion Wave A + B1 flags, verbatim from sonnet_gold_cycle3_focused.sh ---
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

# --- investigator on flat-rate Sonnet (SOT-2627); parallelism 1 = shared-limit protection ---
export RAG_INVESTIGATOR_BACKEND=claude-mcp
export RAG_CLAUDE_MCP_MODEL=sonnet
export RAG_CLAUDE_MCP_RESUME=artifacts/gold100_sonnet_cycle4_resume.jsonl
export RAG_MCP_TOOL_LOG=artifacts/sonnet_cycle4_integrated_tool_calls.jsonl

TARGET="${TARGET:-4,5,6,8,12,20,28,34,37,39,40,41,55,59,64,65,70,76,82,88,92,93,97,98}"

echo "=== SOT-2651 Sonnet cycle4 integrated focused gate start $(date -u +%FT%TZ) ==="
echo "BACKEND=$RAG_INVESTIGATOR_BACKEND ACTION_ROW=$RAG_ACTION_ROW_STORE VISUAL=$RAG_VISUAL_STORE CASE_FIN=$RAG_CASE_FINANCE_STORE REPORT_ATTR=$RAG_REPORT_ATTR_STORE VALUE_NORM=$RAG_FORMAT_VALUE_NORM TARGET=$TARGET"
.venv/bin/python scripts/run_focused_gate.py \
  --label sot2651_cycle4_integrated \
  --target "$TARGET" \
  --sentinels scripts/sonnet_sentinels.json \
  --issue SOT-2651 \
  --dev --no-smoke \
  --workers 1
echo "=== SOT-2651 Sonnet cycle4 integrated focused gate done $(date -u +%FT%TZ) exit=$? ==="
