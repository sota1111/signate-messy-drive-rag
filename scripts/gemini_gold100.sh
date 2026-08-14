#!/usr/bin/env bash
# SOT-2716 — Gemini (flash 3.6) gold100 measurement, backend-non-dependent contract port (official:false).
#
# Purpose: converge the **Gemini answer backend** gold100 toward the Sonnet cycle-11 net96 by firing the
# answer-time contracts that previously lived ONLY in the claude_mcp (Sonnet) lane. This is the clone of
# scripts/sonnet_gold_cycle11.sh with the backend switched to the production Gemini path:
#   * DROP  RAG_FORBID_GEMINI / LLM_PROVIDER=claude-cli / CLAUDE_CLI_MODEL — the serve loop calls Gemini.
#   * DROP  RAG_INVESTIGATOR_BACKEND=claude-mcp (+ RAG_CLAUDE_MCP_*) — use the default gemini tool loop.
#   * KEEP  every store rebuild and every RAG_* store/deterministic/contract flag verbatim, so the ONLY
#           difference vs the Sonnet run is the model that drives the LLM loop.
# GEN_MODEL / VISION_MODEL are already gemini-3.6-flash in the Sonnet script (official model), so they
# are unchanged here.
#
# The ported answer-time contracts (SOT-2716):
#   A. prompt contracts on the shared SYSTEM_PROMPT, Gemini-path only (investigator._answer_contract_suffix):
#      RAG_BARE_ANSWER / RAG_NONE_BARE / RAG_EXACT_LABEL / RAG_VDIFF_NORMALIZE / RAG_SPECIAL_PROVISION /
#      RAG_VDIFF_CLASSIFY / RAG_REPORT_SERIES  (claude_mcp keeps its own suffix ⇒ no Sonnet double-apply).
#   B. value-preserving serve-boundary formats (investigator._apply_serve_formatting + commit_gate):
#      RAG_FORMAT_STRIP_PAREN / RAG_FORMAT_VALUE_NORM / RAG_DECIMAL_UNIT_STRIP / RAG_BIN_RANGE_FORMAT.
# With every flag unset the port is a byte-identical no-op on BOTH backends (verified separately).
#
# GOLD = artifacts/predictions_test_v4_final.csv (2026-08-13 gold audit; idx0 is the only audit-flagged
# candidate error — never hardcode gold values).
#
# ***OFFICIAL:FALSE*** — DEV measurement only (--no-official). Never backs a champion promotion or a
# non-regression claim, and never touches the flash-champion history / LB / SIGNATE CLI.
#
# COST: this IS the production Gemini path, so a full run meters Gemini API (flash 3.6). Keep runs minimal.
#
# BASELINE vs POST-PORT: this script runs the POST-PORT config (contracts ON). To measure the pre-port
# baseline, run the same script with the answer-time contract flags forced OFF via:
#     GEMINI_GOLD_BASELINE=1 bash scripts/gemini_gold100.sh
# which unsets the A/B port flags so the Gemini path behaves as it did before the port (all stores and
# deterministic lanes stay ON, isolating exactly the contract-port delta).
#
# Env knobs: GOLD, GEMINI_GOLD_OUT, GEMINI_GOLD_WORKERS, GEMINI_GOLD_LIMIT (subset smoke), GEMINI_GOLD_BASELINE.
set -euo pipefail
cd /workspaces/signate-messy-drive-rag

.venv/bin/python scripts/check_flag_manifest.py "$0" || exit 1

# Rebuild the question-independent stores (LLM-free, idempotent) — identical set to the Sonnet cycle-11
# script. image_ocr_store / notebook_chart_store / ocr_store are NOT rebuilt (their builds need vision);
# serve reads the persisted artifacts baked at build time.
.venv/bin/python -m src.rag.index.derived_metrics
.venv/bin/python -m src.rag.index.raw_artifact_store
.venv/bin/python -m src.rag.index.analysis_metrics_enum_store
.venv/bin/python -m src.rag.index.doc_reach_store
.venv/bin/python -m src.rag.index.derived_ranking_store
.venv/bin/python -m src.rag.index.schedule_store
RAG_HIGHLIGHT_PIVOT_CONTEXT=1 .venv/bin/python -m src.rag.index.visual_store
.venv/bin/python -m src.rag.index.format_facts_store
.venv/bin/python -m src.rag.index.pptx_money_page_store
.venv/bin/python -m src.rag.index.contact_master_store
.venv/bin/python -m src.rag.index.pptx_note_scope_store
.venv/bin/python -m src.rag.index.schedule_plan_store
.venv/bin/python -m src.rag.index.docx_comment_store
.venv/bin/python -m src.rag.index.action_row_store
.venv/bin/python -m src.rag.index.case_finance_store
.venv/bin/python -m src.rag.index.master_join_store
.venv/bin/python -c "from src.rag.index import report_attr_store as s; r=s.build()['report']; print('[build] report_attr_store', r)"
RAG_VDIFF_STRUCT=1 RAG_VDIFF_CLASSIFY=1 RAG_VDIFF_DIRECT_COMMIT=1 .venv/bin/python -c "from src.rag.index import diff_store as s; r=s.build(); print('[build] diff_store (RAG_VDIFF_STRUCT=1 RAG_VDIFF_CLASSIFY=1 RAG_VDIFF_DIRECT_COMMIT=1)', r.get('pairs') if isinstance(r,dict) else r)"
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

# --- champion Wave A + B1 flags, verbatim from the Sonnet cycle-11 script ---
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

# --- cycle-2/3 levers (kept). NOTE vs Sonnet script: RAG_FORBID_GEMINI / LLM_PROVIDER=claude-cli /
#     CLAUDE_CLI_MODEL are intentionally DROPPED — this is the live Gemini path. ---
export RAG_FACT_LAYER=1
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

# --- cycle-5 levers (promoted children only) ---
export RAG_FANOUT_FINISHER=1
export RAG_FANOUT_FINISHER_MAX=1
export RAG_VDIFF_SUBJECT=1
export RAG_NONE_BARE=1
export RAG_EXACT_LABEL=1
export RAG_HEADING_PAGE_STORE=1

# --- cycle-6 levers (FULL net68 base) ---
export RAG_DOC_REACH_STORE=1
export RAG_RAW_ARTIFACT_STORE=1
export RAG_DERIVED_COVERAGE=1
export RAG_SCHEDULE_STORE=1
export RAG_VDIFF_NORMALIZE=1
export RAG_DECIMAL_UNIT_STRIP=1

# --- cycle-7 levers (all five children promoted) ---
export RAG_IMAGE_OCR_STORE=1
export RAG_NB_CHART_STORE=1
export RAG_XLSX_FORMULA_TRACE=1
export RAG_XREF_COVERAGE=1
export RAG_CORR_SIGN=1
export RAG_BIN_RANGE_FORMAT=1
export RAG_SPECIAL_PROVISION=1

# --- cycle-8 levers (all six children passed their focused gates) ---
export RAG_ANALYSIS_XREF=1
export RAG_PLAN_COVERAGE=1
export RAG_FORMAT_SERIES=1
export RAG_RATE_TABLE=1
export RAG_FORMULA_APPLY=1
export RAG_VDIFF_CLASSIFY=1
export RAG_REPORT_SERIES=1

# --- cycle-9 levers (all four children passed their focused gates) ---
export RAG_CASE_FINANCE_DIFF=1
export RAG_ANALYSIS_METRICS_ENUM=1
export RAG_ANALYSIS_CONFIG_HYPERPARAM=1
export RAG_STAGED_METRICS=1
export RAG_DERIVED_RANKING=1
export RAG_VDIFF_STRUCT=1

# --- cycle-10 levers (all six children passed their focused gates) ---
export RAG_TEXT_FTS_PROJECT_ALIAS=1
export RAG_SEP_CONTRACT_ROLE=1
export RAG_FORMAT_FACTS=1
export RAG_HIGHLIGHT_PIVOT_CONTEXT=1
export RAG_PPTX_MONEY_PAGE=1
export RAG_VDIFF_DIRECT_COMMIT=1
export RAG_CONTACT_MASTER=1

# --- cycle-11 levers (all promoted children; SOT-2709 is a builder fix, no new flag) ---
export RAG_SCHEDULE_PLAN_LOOKUP=1
export RAG_DOCX_COMMENT_ANCHOR=1
export RAG_VDIFF_DC_NOCHANGE=1
export RAG_VDIFF_DC_COLRENAME=1
export RAG_MASTER_JOIN_LOOKUP=1
export RAG_PPTX_NOTE_SCOPE=1

# --- investigator on the live Gemini tool loop (production answer path; parallelizable, not shared-limit) ---
export RAG_INVESTIGATOR_BACKEND=gemini

# --- BASELINE toggle (convenience, PARTIAL): unset only the SOT-2716 port-EXCLUSIVE flags so the Gemini
#     path drops the newly-ported prompt/format contracts. NOTE: RAG_VDIFF_CLASSIFY / RAG_VDIFF_NORMALIZE
#     are intentionally NOT unset here — beyond their claude_mcp/SOT-2716 prompt contract they ALSO carry
#     a pre-existing Gemini serve effect (fact_layer.py / version_diff.py), so unsetting them would
#     over-subtract and misattribute those store/pipeline effects to the port. The AUTHORITATIVE full-port
#     baseline is therefore measured by reverting the port SOURCE (git stash of investigator.py +
#     commit_gate.py) and running this script unchanged — that isolates exactly the SYSTEM_PROMPT +
#     serve-boundary port while leaving every flag's non-prompt effect intact. ---
if [ "${GEMINI_GOLD_BASELINE:-0}" = "1" ]; then
  unset RAG_BARE_ANSWER RAG_NONE_BARE RAG_EXACT_LABEL RAG_SPECIAL_PROVISION RAG_REPORT_SERIES \
        RAG_FORMAT_STRIP_PAREN RAG_FORMAT_VALUE_NORM RAG_DECIMAL_UNIT_STRIP RAG_BIN_RANGE_FORMAT
  MODE=baseline
else
  MODE=postport
fi

GOLD="${GOLD:-artifacts/predictions_test_v4_final.csv}"
WORKERS="${GEMINI_GOLD_WORKERS:-6}"
OUT="${GEMINI_GOLD_OUT:-artifacts/gold100_gemini_${MODE}.json}"
LIMIT_ARG=""
if [ -n "${GEMINI_GOLD_LIMIT:-}" ]; then LIMIT_ARG="--limit ${GEMINI_GOLD_LIMIT}"; fi

echo "=== SOT-2716 Gemini gold100 (${MODE}, official:false) start $(date -u +%FT%TZ) ==="
echo "BACKEND=gemini GEN_MODEL=$GEN_MODEL WORKERS=$WORKERS GOLD=$GOLD OUT=$OUT LIMIT=${GEMINI_GOLD_LIMIT:-none}"
# shellcheck disable=SC2086
.venv/bin/python -m scoring.gold_offline --run --workers "$WORKERS" --no-official \
  --gold "$GOLD" \
  --out "$OUT" $LIMIT_ARG
echo "=== SOT-2716 Gemini gold100 (${MODE}) done $(date -u +%FT%TZ) exit=$? ==="
