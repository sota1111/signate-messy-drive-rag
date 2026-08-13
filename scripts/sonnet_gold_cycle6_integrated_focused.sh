#!/usr/bin/env bash
# SOT-2676 — cycle-6 INTEGRATED focused gate (official:false lane).
#
# Env = scripts/sonnet_gold_cycle5.sh (net59 実証済み) + ALL promoted cycle-6 child levers combined:
#   RAG_DOC_REACH_STORE=1     — SOT-2677 K1 doc table/full-text chunk store (idx8/50/99 class)
#   RAG_RAW_ARTIFACT_STORE=1  — SOT-2678 K2 raw analysis-artifact store (idx32/61/62 class)
#   RAG_DERIVED_COVERAGE=1    — SOT-2679 K3 derived-metrics coverage expansion (idx4/24 class)
#   RAG_SCHEDULE_STORE=1      — SOT-2680 K4 schedule/ID/roster cross-reference (idx92/94/96 class)
#   RAG_VDIFF_NORMALIZE=1     — SOT-2681 K5 version_diff semantic normalization (idx95/1 class)
#   RAG_DECIMAL_UNIT_STRIP=1  — SOT-2682 K6 decimal-spec unit-suffix strip (idx79 class)
# Purpose: verify the levers COMPOSE before the one full gold100 run — each child gated alone;
# this is the first time they run together.
#
# Targets (per docs/ai/sonnet_cycle_analysis/cycle6.md §5):
#   primary 15 — K1 8,50,99 / K2 32,61,62 / K3 4,24 / K4 92,94,96 / K5 1,95,14 / K6 79
#   wrong-guard 7 — 1,9,22,27,78,79,95 (current cycle-5 wrong set; must NOT stay/return Incorrect
#   beyond what each child already accounted for — checked from the gate JSON verdicts, in addition
#   to the sentinel gate). Union = 19 idx.
# Gate FAILS on any Sonnet sentinel regression. --dev ⇒ official:false (non-flash stack; NOT a
# promotion basis).
#
# NO resume cache (RAG_CLAUDE_MCP_RESUME=0): the resume key is (model, question) only — NOT config
# (SOT-2664 gotcha) — and RAG_DECIMAL_UNIT_STRIP/RAG_VDIFF_NORMALIZE change answers globally, so any
# sidecar replay would mask the very composition under test. Target AND sentinels re-derive live.
set -euo pipefail
cd /workspaces/signate-messy-drive-rag

.venv/bin/python scripts/check_flag_manifest.py "$0" || exit 1

# Rebuild the question-independent stores (LLM-free, idempotent; union of base + all cycle-6 child
# rebuilds; heading_page_store needs a writable HOME for LibreOffice headless).
.venv/bin/python -m src.rag.index.raw_artifact_store
.venv/bin/python -m src.rag.index.doc_reach_store
.venv/bin/python -m src.rag.index.schedule_store
.venv/bin/python -m src.rag.index.visual_store
.venv/bin/python -m src.rag.index.action_row_store
.venv/bin/python -m src.rag.index.case_finance_store
.venv/bin/python -c "from src.rag.index import report_attr_store as s; r=s.build()['report']; print('[build] report_attr_store', r)"
.venv/bin/python -c "from src.rag.index import diff_store as s; r=s.build(); print('[build] diff_store', r.get('pairs') if isinstance(r,dict) else r)"
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

# --- cycle-6 levers (all six children promoted their focused gates) ---
export RAG_DOC_REACH_STORE=1
export RAG_RAW_ARTIFACT_STORE=1
export RAG_DERIVED_COVERAGE=1
export RAG_SCHEDULE_STORE=1
export RAG_VDIFF_NORMALIZE=1
export RAG_DECIMAL_UNIT_STRIP=1

# --- investigator on flat-rate Sonnet; parallelism 1 = shared-limit protection ---
export RAG_INVESTIGATOR_BACKEND=claude-mcp
export RAG_CLAUDE_MCP_MODEL=sonnet
export RAG_MCP_TOOL_LOG=artifacts/sonnet_cycle6_integrated_tool_calls.jsonl
export RAG_CLAUDE_MCP_RESUME=0

TARGET="${TARGET:-1,4,8,9,14,22,24,27,32,50,61,62,78,79,92,94,95,96,99}"

echo "=== SOT-2676 cycle6 integrated focused gate start $(date -u +%FT%TZ) ==="
echo "BACKEND=$RAG_INVESTIGATOR_BACKEND DOC_REACH=$RAG_DOC_REACH_STORE RAW_ARTIFACT=$RAG_RAW_ARTIFACT_STORE DERIVED=$RAG_DERIVED_COVERAGE SCHEDULE=$RAG_SCHEDULE_STORE VDIFF_NORM=$RAG_VDIFF_NORMALIZE UNIT_STRIP=$RAG_DECIMAL_UNIT_STRIP TARGET=$TARGET"
.venv/bin/python scripts/run_focused_gate.py \
  --label sot2676_cycle6_integrated \
  --target "$TARGET" \
  --sentinels scripts/sonnet_sentinels.json \
  --issue SOT-2676 \
  --dev --no-smoke \
  --workers 1
echo "=== SOT-2676 cycle6 integrated focused gate done $(date -u +%FT%TZ) exit=$? ==="
