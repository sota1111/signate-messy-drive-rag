#!/usr/bin/env bash
# SOT-2651 — Sonnet gold100 improvement cycle 4 (official:false lane).
#
# Base env = scripts/sonnet_gold_cycle3.sh (champion Wave A + B1, investigator on flat-rate Sonnet via
# RAG_INVESTIGATOR_BACKEND=claude-mcp, RAG_FACT_LAYER=1, RAG_FORBID_GEMINI=1, RAG_OCR_STORE=1,
# RAG_FORMAT_STRIP_PAREN=1) with the five cycle-4 child levers on top:
#   RAG_ACTION_ROW_STORE=1  — SOT-2652 scanned-PDF action/task-row store (idx20/70/93 class)
#   RAG_VISUAL_STORE=1      — SOT-2653 xlsx chart/highlight visual-facts store (idx39/65/97/82 class)
#   RAG_CASE_FINANCE_STORE=1 — SOT-2654 case finance/effort derived store (idx37/40/55/76 class)
#   RAG_REPORT_ATTR_STORE=1 — SOT-2655 final-report numeric attribute store (idx5/28/64 class)
#   RAG_FORMAT_VALUE_NORM=1 — SOT-2656 value-preserving answer normalization (idx4/8/41/59/88/92 class)
# Cerebras-stack flags (FTS / unified search / plan fanout) stay OFF — no focused target-side wins.
#
# ***OFFICIAL:FALSE*** — DEV measurement only (--no-official). Never backs a champion promotion or
# non-regression claim. The official lane stays gemini-3.6-flash (SOT-2625 model guard), untouched.
#
# RESUME (SOT-2650 handoff: "run one RESUMED Sonnet dev gold100"): the sidecar below is the reviewed
# cycle-4 Sonnet cache curated for this cycle — all cycle-4 target-idx questions and all timeout
# records evicted (fresh measurement on every changed path), 57 reviewed unchanged-lane answers
# replay. The integrated focused gate (sonnet_gold_cycle4_integrated_focused.sh) shares this sidecar,
# so its fresh target answers replay here identically (same config, same question).
#
# USAGE-LIMIT: flat-rate Sonnet is shared account-wide. Parallelism 1; on a detected limit the
# claude-mcp backend abstains the question and persists answered ones to the resume sidecar — simply
# re-run this script to continue where it stopped.
set -euo pipefail
cd /workspaces/signate-messy-drive-rag

.venv/bin/python scripts/check_flag_manifest.py "$0" || exit 1

# Rebuild the four cycle-4 stores (LLM-free, question-independent, idempotent).
.venv/bin/python -m src.rag.index.visual_store
.venv/bin/python -m src.rag.index.action_row_store
.venv/bin/python -m src.rag.index.case_finance_store
.venv/bin/python -c "from src.rag.index import report_attr_store as s; r=s.build()['report']; print('[build] report_attr_store', r)"

export VERTEX_LOCATION=global
export GEN_MODEL=gemini-3.6-flash
export GEN_MODEL_HARD=gemini-3.6-flash
export VISION_MODEL=gemini-3.6-flash
export PYTHONPATH=/tmp/genai_patch:.

# --- champion Wave A + B1 flags, verbatim from sonnet_gold_cycle3.sh ---
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
# --- 2026-08-12 manual interim (cycle4 net36 base + Cerebras stack + bare-answer contract) ---
export RAG_TEXT_FTS=1          # SOT-2657 全文FTS+IDF (grep枯渇13問対策)
export RAG_UNIFIED_SEARCH=1    # SOT-2659 統合search (RRF融合+文脈復元)
export RAG_PLAN_FANOUT=1       # SOT-2661 planner→並列fan-out→synthesis
export RAG_BARE_ANSWER=1       # 裸回答書式契約 (説明文ラッパー12問対策・プロンプトのみ)

# --- investigator on flat-rate Sonnet; parallelism 1 = shared-limit protection ---
export RAG_INVESTIGATOR_BACKEND=claude-mcp
export RAG_CLAUDE_MCP_MODEL=sonnet
export RAG_CLAUDE_MCP_RESUME=artifacts/gold100_sonnet_manual0812_resume.jsonl
export RAG_MCP_TOOL_LOG=artifacts/gold100_sonnet_manual0812_tool_calls.jsonl

echo "=== SOT-2651 Sonnet gold100 cycle4 (official:false) start $(date -u +%FT%TZ) ==="
echo "BACKEND=$RAG_INVESTIGATOR_BACKEND MODEL=$RAG_CLAUDE_MCP_MODEL RESUME=$RAG_CLAUDE_MCP_RESUME"
.venv/bin/python -m scoring.gold_offline --run --workers 1 --no-official \
  --out artifacts/gold100_sonnet_manual0812.json
echo "=== SOT-2651 Sonnet gold100 cycle4 done $(date -u +%FT%TZ) exit=$? ==="
