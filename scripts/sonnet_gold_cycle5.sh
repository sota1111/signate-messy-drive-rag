#!/usr/bin/env bash
# SOT-2662 — Sonnet gold100 improvement cycle 5, integrated config (official:false lane).
#
# Base env = scripts/sonnet_gold_manual_20260812.sh (cycle-4.5 net48 実証済み: champion Wave A+B1 +
# cycle-4 stores + Cerebras stack + RAG_BARE_ANSWER) PLUS the cycle-5 child levers that PASSED their
# focused gates (promoted only — SOT-2663's stage-budget bump was REJECTED, so
# RAG_PLAN_FANOUT_MAX_TURNS stays at its default 5):
#   RAG_FANOUT_FINISHER=1    — SOT-2664 bounded plan-fanout finisher (予算切れ後、対象特定済み単一文書
#                              リードのみ +1手; compute除外; idx16/49/75 class)
#   RAG_VDIFF_SUBJECT=1      — SOT-2665 version_diff single-add document-subject prefix (idx0 class)
#   RAG_NONE_BARE=1          — SOT-2665 bare 該当なし contract (idx9/85 class)
#   RAG_EXACT_LABEL=1        — SOT-2666 exact-label transcription + answer-scope contract, prompt-only
#                              (idx21/62/78 class)
#   RAG_HEADING_PAGE_STORE=1 — SOT-2668 見出し→印字ページ locator store + deterministic lane (idx12/18)
#   (SOT-2667 expands action_row_store coverage under the existing RAG_ACTION_ROW_STORE=1 —
#    the rebuild below regenerates the expanded store; idx45/93 class.)
#
# ***OFFICIAL:FALSE*** — DEV measurement only (--no-official). Never backs a champion promotion or
# non-regression claim. The official lane stays gemini-3.6-flash (SOT-2625 model guard), untouched.
#
# RESUME: FRESH sidecar (below). The claude-mcp resume key is (model, question) only — NOT config
# (SOT-2664 gotcha) — and this cycle changes the system prompt globally (RAG_EXACT_LABEL), so replaying
# the cycle-4.5 sidecar would mask the change on every LLM lane. The fresh sidecar still gives
# usage-limit continuation: answered questions persist, so re-running this script resumes where it
# stopped.
#
# USAGE-LIMIT: flat-rate Sonnet is shared account-wide. Parallelism 1; on a detected limit the
# claude-mcp backend abstains the question and persists answered ones to the resume sidecar — simply
# re-run this script to continue where it stopped.
set -euo pipefail
cd /workspaces/signate-messy-drive-rag

.venv/bin/python scripts/check_flag_manifest.py "$0" || exit 1

# Rebuild the question-independent stores (LLM-free, idempotent). action_row_store now includes the
# SOT-2667 coverage expansion; heading_page_store needs a writable HOME for LibreOffice headless.
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
export RAG_CLAUDE_MCP_RESUME=artifacts/gold100_sonnet_cycle5_resume.jsonl
export RAG_MCP_TOOL_LOG=artifacts/gold100_sonnet_cycle5_tool_calls.jsonl

echo "=== SOT-2662 Sonnet gold100 cycle5 (official:false) start $(date -u +%FT%TZ) ==="
echo "BACKEND=$RAG_INVESTIGATOR_BACKEND MODEL=$RAG_CLAUDE_MCP_MODEL RESUME=$RAG_CLAUDE_MCP_RESUME"
.venv/bin/python -m scoring.gold_offline --run --workers 1 --no-official \
  --out artifacts/gold100_sonnet_cycle5.json
echo "=== SOT-2662 Sonnet gold100 cycle5 done $(date -u +%FT%TZ) exit=$? ==="
