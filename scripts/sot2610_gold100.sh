#!/usr/bin/env bash
# SOT-2610 Wave A consolidated gold100: champion (r1 answer-increase + FORMAT_EVENTS, net32)
# + RAG_DET_PIPELINE_ROUTER=1 (Wave A A1-A4 deterministic pipelines ON).
# Official model = gemini-3.6-flash @ global (judge=codex). Run ONCE.
set -euo pipefail
cd /workspaces/signate-messy-drive-rag

# Official measurement model (08-10): gemini-3.6-flash on global (no pro on global)
export VERTEX_LOCATION=global
export GEN_MODEL=gemini-3.6-flash
export GEN_MODEL_HARD=gemini-3.6-flash
export VISION_MODEL=gemini-3.6-flash

# genai per-request timeout injection (SOT-2568 method)
export PYTHONPATH=/tmp/genai_patch:.

# --- champion answer-increase / error-type flags (r1, net32) ---
export RAG_FIRST_MOVE_ROUTING=1 RAG_SPIN_DETECTION=1 RAG_ADAPTIVE_BUDGET=1 RAG_EVIDENCE_CACHE=1
export RAG_BUDGET_BOUNDARY_RESEARCH=1 RAG_UNANSWERABLE_FALLBACK=1 RAG_PDF_OCR=1 RAG_SHARE_CORPUS_PROFILE=1
export RAG_CANONICAL_MANIFEST=1 RAG_EVIDENCE_INDEX=1 RAG_STRUCTURE_STORE=1
export RAG_GRANULARITY_NORMALIZATION=1 RAG_XLSX_EMBEDDED_IMAGE=1 RAG_CONFLICT_RESOLUTION=1 GATE_EXEC_CORRECT=1
export RAG_NUMERIC_FEATURE_CORR=1 RAG_RELEVANCE_STRICT=1 RAG_HIGHLIGHT_EXTRA=1 RAG_FONT_EMPHASIS=1
export RAG_FILE_GREP_INDEX_CANDIDATES=1
export RAG_FORMAT_EVENTS=1
# --- Wave A deterministic type-specific pipelines (this issue) ---
export RAG_DET_PIPELINE_ROUTER=1

echo "=== SOT-2610 gold100 start $(date -u +%FT%TZ) ==="
echo "GEN_MODEL=$GEN_MODEL VERTEX_LOCATION=$VERTEX_LOCATION RAG_DET_PIPELINE_ROUTER=$RAG_DET_PIPELINE_ROUTER"
.venv/bin/python -m scoring.gold_offline --run --workers 8 \
  --out artifacts/gold100_sot2610_waveA.json
echo "=== SOT-2610 gold100 done $(date -u +%FT%TZ) exit=$? ==="
