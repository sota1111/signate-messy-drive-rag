#!/usr/bin/env bash
# SOT-2676 — Sonnet gold100 improvement cycle 6, integrated config (official:false lane).
#
# Base env = scripts/sonnet_gold_cycle5.sh (net59 実証済み: champion Wave A+B1 + cycle-4 stores +
# Cerebras stack + RAG_BARE_ANSWER + promoted cycle-5 levers) PLUS the cycle-6 child levers that
# PASSED their focused gates (all six promoted):
#   RAG_DOC_REACH_STORE=1     — SOT-2677 K1 doc table/full-text chunk store + doc_table_lookup /
#                               doc_fulltext_search (idx8/50/99 class; kills read_office truncation)
#   RAG_RAW_ARTIFACT_STORE=1  — SOT-2678 K2 raw .py/.json/.csv artifact store + artifact_grep /
#                               analysis_artifact_lookup + config-applied hyperparams (idx32/61/62)
#   RAG_DERIVED_COVERAGE=1    — SOT-2679 K3 derived-metrics coverage: feature×target correlations,
#                               per-case missing-row counts on the canonical manifest (idx4/24)
#   RAG_SCHEDULE_STORE=1      — SOT-2680 K4 schedule/ID-counts/roster/checkpoint cross-reference
#                               deterministic lane (idx92/94/96/72)
#   RAG_VDIFF_NORMALIZE=1     — SOT-2681 K5 version_diff semantic normalization: list-append→「追加」,
#                               delete+add→置換 collapse (idx95/1 class)
#   RAG_DECIMAL_UNIT_STRIP=1  — SOT-2682 K6 decimal-spec unit-suffix strip at the claude-mcp output
#                               boundary, value-preserving (idx79 class)
#
# ***OFFICIAL:FALSE*** — DEV measurement only (--no-official). Never backs a champion promotion or
# non-regression claim. The official lane stays gemini-3.6-flash (SOT-2625 model guard), untouched.
#
# RESUME: FRESH sidecar (below). The claude-mcp resume key is (model, question) only — NOT config
# (SOT-2664 gotcha) — and this cycle changes answers globally (RAG_DECIMAL_UNIT_STRIP boundary strip,
# RAG_VDIFF_NORMALIZE), so replaying the cycle-5 sidecar would mask the change on every LLM lane. The
# fresh sidecar still gives usage-limit continuation: answered questions persist, so re-running this
# script resumes where it stopped.
#
# USAGE-LIMIT: flat-rate Sonnet is shared account-wide. Parallelism 1; on a detected limit the
# claude-mcp backend abstains the question and persists answered ones to the resume sidecar — simply
# re-run this script to continue where it stopped.
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
export RAG_CLAUDE_MCP_RESUME=artifacts/gold100_sonnet_cycle6_resume.jsonl
export RAG_MCP_TOOL_LOG=artifacts/gold100_sonnet_cycle6_tool_calls.jsonl

echo "=== SOT-2676 Sonnet gold100 cycle6 (official:false) start $(date -u +%FT%TZ) ==="
echo "BACKEND=$RAG_INVESTIGATOR_BACKEND MODEL=$RAG_CLAUDE_MCP_MODEL RESUME=$RAG_CLAUDE_MCP_RESUME"
.venv/bin/python -m scoring.gold_offline --run --workers 1 --no-official \
  --out artifacts/gold100_sonnet_cycle6.json
echo "=== SOT-2676 Sonnet gold100 cycle6 done $(date -u +%FT%TZ) exit=$? ==="
