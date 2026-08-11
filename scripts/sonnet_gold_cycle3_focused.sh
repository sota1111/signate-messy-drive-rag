#!/usr/bin/env bash
# SOT-2650 — focused gate for Sonnet gold100 cycle 3 (official:false lane).
#
# Same env as scripts/sonnet_gold_cycle2_focused.sh plus the cycle-3 levers:
#   RAG_OCR_STORE=1         — persisted build-time OCR of scanned PDFs (serve reads it with NO genai
#                             call; live RAG_PDF_OCR silently no-ops under RAG_FORBID_GEMINI)
#   RAG_FORMAT_STRIP_PAREN=1 — value-preserving 括弧内付加情報 strip (idx52/84/87/88 wrong class)
# Targets:
#   idx67 (enum, amount-diff composite lane over the 10/10 proposal/FR store coverage)
#   idx22 (version, notebook diff lane — diff_lookup now reaches the ipynb pair)
#   idx87 (enum-lane regression probe + the paren-strip literal target)
#   idx34/93 (MINAMINO scanned-PDF abstains the OCR store should unlock via 会議録 evidence)
# Gate FAILS on any Sonnet sentinel regression. --dev ⇒ official:false (non-flash stack).
set -euo pipefail
cd /workspaces/signate-messy-drive-rag

.venv/bin/python scripts/check_flag_manifest.py "$0" || exit 1

export VERTEX_LOCATION=global
export GEN_MODEL=gemini-3.6-flash
export GEN_MODEL_HARD=gemini-3.6-flash
export VISION_MODEL=gemini-3.6-flash
export PYTHONPATH=/tmp/genai_patch:.

# --- champion Wave A + B1 flags, verbatim from SOT-2628/sonnet_gold_cycle2.sh ---
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

# --- cycle-2 levers (kept) ---
export RAG_FACT_LAYER=1
export RAG_FORBID_GEMINI=1
export LLM_PROVIDER=claude-cli
export CLAUDE_CLI_MODEL=sonnet

# --- cycle-3 levers (SOT-2650) ---
export RAG_OCR_STORE=1
export RAG_FORMAT_STRIP_PAREN=1

# --- investigator on flat-rate Sonnet (SOT-2627); parallelism 1 = shared-limit protection ---
export RAG_INVESTIGATOR_BACKEND=claude-mcp
export RAG_CLAUDE_MCP_MODEL=sonnet
export RAG_CLAUDE_MCP_RESUME=artifacts/sonnet_cycle3_focused_resume.jsonl
export RAG_MCP_TOOL_LOG=artifacts/sonnet_cycle3_focused_tool_calls.jsonl

TARGET="67,22,87,34,93"

echo "=== SOT-2650 Sonnet cycle3 focused gate start $(date -u +%FT%TZ) ==="
echo "BACKEND=$RAG_INVESTIGATOR_BACKEND FACT_LAYER=$RAG_FACT_LAYER OCR_STORE=$RAG_OCR_STORE STRIP_PAREN=$RAG_FORMAT_STRIP_PAREN TARGET=$TARGET"
.venv/bin/python scripts/run_focused_gate.py \
  --label sot2650_sonnet_cycle3 \
  --target "$TARGET" \
  --sentinels scripts/sonnet_sentinels.json \
  --issue SOT-2650 \
  --dev --no-smoke \
  --workers 1
echo "=== SOT-2650 Sonnet cycle3 focused gate done $(date -u +%FT%TZ) exit=$? ==="
