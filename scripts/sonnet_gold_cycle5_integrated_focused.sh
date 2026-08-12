#!/usr/bin/env bash
# SOT-2662 — cycle-5 INTEGRATED focused gate (official:false lane).
#
# Env = scripts/sonnet_gold_cycle5.sh (base net48 + ALL promoted cycle-5 child levers combined:
# RAG_FANOUT_FINISHER / RAG_VDIFF_SUBJECT / RAG_NONE_BARE / RAG_EXACT_LABEL / RAG_HEADING_PAGE_STORE +
# SOT-2667 action_row_store coverage rebuild). Purpose: verify the levers COMPOSE before the one
# full gold100 run — each child gated alone; this is the first time they run together.
#
# Targets (per docs/ai/sonnet_cycle_analysis/cycle5.md §5):
#   primary 15 — C1 16,49,75,83 / C2 0,9,14,85 / C3 21,62,78 / C4 45,93 / C5 12,18
#   conversion-guard 7 — 11,24,36,48,77,95,96 (cycle-4.5 wrong→abstain 転換組; must NOT return to
#   Incorrect — checked from the gate JSON verdicts, in addition to the sentinel gate)
# Gate FAILS on any Sonnet sentinel regression. --dev ⇒ official:false (non-flash stack; NOT a
# promotion basis).
#
# NO resume cache (RAG_CLAUDE_MCP_RESUME=0): the resume key is (model, question) only — NOT config
# (SOT-2664 gotcha) — and RAG_EXACT_LABEL changes the system prompt globally, so any sidecar replay
# would mask the very composition under test. Target AND sentinels re-derive live.
set -euo pipefail
cd /workspaces/signate-messy-drive-rag

.venv/bin/python scripts/check_flag_manifest.py "$0" || exit 1

# Rebuild the question-independent stores (LLM-free, idempotent; SOT-2667 expanded action rows,
# SOT-2668 heading→page store needs a writable HOME for LibreOffice headless).
.venv/bin/python -m src.rag.index.visual_store
.venv/bin/python -m src.rag.index.action_row_store
.venv/bin/python -m src.rag.index.case_finance_store
.venv/bin/python -c "from src.rag.index import report_attr_store as s; r=s.build()['report']; print('[build] report_attr_store', r)"
HOME="${HOME:-/tmp}" .venv/bin/python -m src.rag.index.heading_page_store

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

# --- cycle-4.5 manual interim (Cerebras stack + bare-answer) ---
export RAG_TEXT_FTS=1
export RAG_UNIFIED_SEARCH=1
export RAG_PLAN_FANOUT=1
export RAG_BARE_ANSWER=1

# --- cycle-5 levers (promoted children only; SOT-2663 rejected ⇒ stage budget stays default 5) ---
export RAG_FANOUT_FINISHER=1
export RAG_FANOUT_FINISHER_MAX=1
export RAG_VDIFF_SUBJECT=1
export RAG_NONE_BARE=1
export RAG_EXACT_LABEL=1
export RAG_HEADING_PAGE_STORE=1

# --- investigator on flat-rate Sonnet; parallelism 1 = shared-limit protection ---
export RAG_INVESTIGATOR_BACKEND=claude-mcp
export RAG_CLAUDE_MCP_MODEL=sonnet
export RAG_MCP_TOOL_LOG=artifacts/sonnet_cycle5_integrated_tool_calls.jsonl
export RAG_CLAUDE_MCP_RESUME=0

TARGET="${TARGET:-0,9,11,12,14,16,18,21,24,36,45,48,49,62,75,77,78,83,85,93,95,96}"

echo "=== SOT-2662 cycle5 integrated focused gate start $(date -u +%FT%TZ) ==="
echo "BACKEND=$RAG_INVESTIGATOR_BACKEND EXACT_LABEL=$RAG_EXACT_LABEL FINISHER=$RAG_FANOUT_FINISHER TARGET=$TARGET"
.venv/bin/python scripts/run_focused_gate.py \
  --label sot2662_cycle5_integrated \
  --target "$TARGET" \
  --sentinels scripts/sonnet_sentinels.json \
  --issue SOT-2662 \
  --dev --no-smoke \
  --workers 1
echo "=== SOT-2662 cycle5 integrated focused gate done $(date -u +%FT%TZ) exit=$? ==="
