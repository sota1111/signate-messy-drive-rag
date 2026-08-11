#!/usr/bin/env bash
# SOT-2647 (事前計算事実層 5/5) — focused verification of the fact-layer wiring on top of the champion.
# Base env = SOT-2610 Wave A + B1 champion (net40) EXACTLY, plus the one new lever RAG_FACT_LAYER=1
# (4 precomputed stores as det lanes + investigator/MCP tools). Official model = gemini-3.6-flash @ global
# (judge=codex). Target = hard core 16 (idx22,32,38,39,40,45,48,50,55,57,63,67,82,87,97,98) + B-class
# version (idx1,14,95). The gate FAILS only on a sentinel regression; success = 新規MATCH & wrong 非増加.
set -euo pipefail
cd /workspaces/signate-messy-drive-rag

# Official measurement model (08-10): gemini-3.6-flash on global (no pro on global).
export VERTEX_LOCATION=global
export GEN_MODEL=gemini-3.6-flash
export GEN_MODEL_HARD=gemini-3.6-flash
export VISION_MODEL=gemini-3.6-flash

# genai per-request timeout injection (SOT-2568 method); run_focused_gate also injects a bounded client.
export PYTHONPATH=/tmp/genai_patch:.

# --- champion Wave A + B1 flags (SOT-2610 net40), verbatim ---
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

# --- the one new lever under test (this issue) ---
export RAG_FACT_LAYER=1

TARGET="22,32,38,39,40,45,48,50,55,57,63,67,82,87,97,98,1,14,95"

echo "=== SOT-2647 focused gate start $(date -u +%FT%TZ) ==="
echo "GEN_MODEL=$GEN_MODEL VERTEX_LOCATION=$VERTEX_LOCATION RAG_FACT_LAYER=$RAG_FACT_LAYER"
.venv/bin/python scripts/run_focused_gate.py \
  --label sot2647_fact_layer \
  --target "$TARGET" \
  --issue SOT-2647 \
  --workers 4
echo "=== SOT-2647 focused gate done $(date -u +%FT%TZ) exit=$? ==="
