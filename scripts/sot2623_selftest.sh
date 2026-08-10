#!/usr/bin/env bash
# SOT-2623 self-consistency: run the focused gate sentinel set (10 existing-MATCH idx) under the
# EXACT champion env (SOT-2610 Wave A net40). All 10 must stay MATCH -> GATE PASS (exit 0).
# This is the runner's own regression baseline; it spends Gemini only on the 10 sentinels (no full gold100).
set -uo pipefail
cd /workspaces/signate-messy-drive-rag

# Official measurement model (08-10): gemini-3.6-flash on global (no pro on global); judge=codex.
export VERTEX_LOCATION=global
export GEN_MODEL=gemini-3.6-flash
export GEN_MODEL_HARD=gemini-3.6-flash
export VISION_MODEL=gemini-3.6-flash
export PYTHONPATH=/tmp/genai_patch:.

# --- champion answer-increase / error-type flags (r1) + Wave A deterministic router (net40) ---
export RAG_FIRST_MOVE_ROUTING=1 RAG_SPIN_DETECTION=1 RAG_ADAPTIVE_BUDGET=1 RAG_EVIDENCE_CACHE=1
export RAG_BUDGET_BOUNDARY_RESEARCH=1 RAG_UNANSWERABLE_FALLBACK=1 RAG_PDF_OCR=1 RAG_SHARE_CORPUS_PROFILE=1
export RAG_CANONICAL_MANIFEST=1 RAG_EVIDENCE_INDEX=1 RAG_STRUCTURE_STORE=1
export RAG_GRANULARITY_NORMALIZATION=1 RAG_XLSX_EMBEDDED_IMAGE=1 RAG_CONFLICT_RESOLUTION=1 GATE_EXEC_CORRECT=1
export RAG_NUMERIC_FEATURE_CORR=1 RAG_RELEVANCE_STRICT=1 RAG_HIGHLIGHT_EXTRA=1 RAG_FONT_EMPHASIS=1
export RAG_FILE_GREP_INDEX_CANDIDATES=1 RAG_FORMAT_EVENTS=1
export RAG_DET_PIPELINE_ROUTER=1

echo "=== SOT-2623 selftest start $(date -u +%FT%TZ) ==="
.venv/bin/python scripts/run_focused_gate.py --label selftest --workers 4
code=$?
echo "=== SOT-2623 selftest done $(date -u +%FT%TZ) exit=$code ==="
exit $code
