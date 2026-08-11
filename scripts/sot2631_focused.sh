#!/usr/bin/env bash
# SOT-2631 (G1, PLAN SOT-2602) — focused gate for the highlight-extraction procedure port.
# Runs the mandatory sentinel set + the G1 targets (idx 15/80/17) plus the same-class EXISTING champion
# MATCH answers idx7/42 (regression checks) on the OFFICIAL flash-3.6 stack, on top of the SOT-2610 Wave A
# champion env, with ONLY the new port flag added (RAG_G1_HIGHLIGHT_PORT=1). This isolates the port's
# effect vs champion. Gate PASSES iff no sentinel drops from MATCH. Run ONCE.
# (judge=codex, gold=predictions_test_v3_final.csv)
set -euo pipefail
cd /workspaces/signate-messy-drive-rag

# Official measurement model (08-10): gemini-3.6-flash on global (no pro on global)
export VERTEX_LOCATION=global
export GEN_MODEL=gemini-3.6-flash
export GEN_MODEL_HARD=gemini-3.6-flash
export VISION_MODEL=gemini-3.6-flash
export PYTHONPATH=/tmp/genai_patch:.

# --- champion (SOT-2610 Wave A net40) env, verbatim ---
export RAG_FIRST_MOVE_ROUTING=1 RAG_SPIN_DETECTION=1 RAG_ADAPTIVE_BUDGET=1 RAG_EVIDENCE_CACHE=1
export RAG_BUDGET_BOUNDARY_RESEARCH=1 RAG_UNANSWERABLE_FALLBACK=1 RAG_PDF_OCR=1 RAG_SHARE_CORPUS_PROFILE=1
export RAG_CANONICAL_MANIFEST=1 RAG_EVIDENCE_INDEX=1 RAG_STRUCTURE_STORE=1
export RAG_GRANULARITY_NORMALIZATION=1 RAG_XLSX_EMBEDDED_IMAGE=1 RAG_CONFLICT_RESOLUTION=1 GATE_EXEC_CORRECT=1
export RAG_NUMERIC_FEATURE_CORR=1 RAG_RELEVANCE_STRICT=1 RAG_HIGHLIGHT_EXTRA=1 RAG_FONT_EMPHASIS=1
export RAG_FILE_GREP_INDEX_CANDIDATES=1
export RAG_FORMAT_EVENTS=1
export RAG_DET_PIPELINE_ROUTER=1

# --- this issue: G1 highlight-extraction procedure port ---
export RAG_G1_HIGHLIGHT_PORT=1

echo "=== SOT-2631 focused gate start $(date -u +%FT%TZ) ==="
echo "GEN_MODEL=$GEN_MODEL VERTEX_LOCATION=$VERTEX_LOCATION RAG_G1_HIGHLIGHT_PORT=$RAG_G1_HIGHLIGHT_PORT"
.venv/bin/python scripts/run_focused_gate.py --label sot2631_g1port \
  --issue SOT-2631 --target "15,80,17,7,42" --workers 8
code=$?
echo "=== SOT-2631 focused gate done $(date -u +%FT%TZ) exit=$code ==="
exit $code
