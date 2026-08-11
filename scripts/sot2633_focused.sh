#!/usr/bin/env bash
# SOT-2633 (G3, PLAN SOT-2602) — flash-side focused verification of the SOT-2619 fixes.
#
# Part (a) of G3 is a MEASUREMENT, not a code change: SOT-2619 (fact_lookup format-equivalence
# normalizer + version_diff idx74 deterministic naturalization, merged 08-10 18:00) landed AFTER the
# champion measurement SOT-2610 Wave A (12:41), so its effect on the OFFICIAL flash-3.6 stack has never
# been measured. This runner replays the CURRENT-main champion serve path (Wave A net40 env, RAG_DET_
# PIPELINE_ROUTER=1 so version_diff naturalization is reached; RAG_ANSWER_NORMALIZE defaults ON so the
# normalize.py gloss-dedup is active) with NO new flag added, and records the live verdict of the five
# SOT-2619 target questions (idx74/84/78/62/75) plus the mandatory existing-MATCH sentinel set. Any of
# these that now MATCH are "recovered for free" (no implementation) and become the baseline for SOT-2636.
#
# There is intentionally NO port flag here: G3 part (b) (idx56 y-axis-tick chart extension) was judged
# NON-FEASIBLE as a deterministic port — ipynb-rendered matplotlib figures carry no numCache, so the
# y-axis max tick can only be read by vision (forbidden as a value source since SOT-2507; see
# src/rag/agent/pipelines/chart_spatial.py lines 33-35). Wave A4's safe abstain is kept unchanged, so
# the serve path is byte-identical and no RAG_G3_CHART_PORT flag is introduced. See
# docs/ai/sot2633_g3_verification.md for the recorded determination.
#
# Gate PASSES iff no sentinel drops from MATCH (run_focused_gate.py, SOT-2623). Run ONCE.
# (judge=codex, gold=predictions_test_v3_final.csv)
set -euo pipefail
cd /workspaces/signate-messy-drive-rag

# Official measurement model (08-10): gemini-3.6-flash on global (no pro on global)
export VERTEX_LOCATION=global
export GEN_MODEL=gemini-3.6-flash
export GEN_MODEL_HARD=gemini-3.6-flash
export VISION_MODEL=gemini-3.6-flash
export PYTHONPATH=/tmp/genai_patch:.

# --- champion (SOT-2610 Wave A net40) env, verbatim — this is current main incl. SOT-2619 (no new flag) ---
export RAG_FIRST_MOVE_ROUTING=1 RAG_SPIN_DETECTION=1 RAG_ADAPTIVE_BUDGET=1 RAG_EVIDENCE_CACHE=1
export RAG_BUDGET_BOUNDARY_RESEARCH=1 RAG_UNANSWERABLE_FALLBACK=1 RAG_PDF_OCR=1 RAG_SHARE_CORPUS_PROFILE=1
export RAG_CANONICAL_MANIFEST=1 RAG_EVIDENCE_INDEX=1 RAG_STRUCTURE_STORE=1
export RAG_GRANULARITY_NORMALIZATION=1 RAG_XLSX_EMBEDDED_IMAGE=1 RAG_CONFLICT_RESOLUTION=1 GATE_EXEC_CORRECT=1
export RAG_NUMERIC_FEATURE_CORR=1 RAG_RELEVANCE_STRICT=1 RAG_HIGHLIGHT_EXTRA=1 RAG_FONT_EMPHASIS=1
export RAG_FILE_GREP_INDEX_CANDIDATES=1
export RAG_FORMAT_EVENTS=1
export RAG_DET_PIPELINE_ROUTER=1

echo "=== SOT-2633 focused gate start $(date -u +%FT%TZ) ==="
echo "GEN_MODEL=$GEN_MODEL VERTEX_LOCATION=$VERTEX_LOCATION (no port flag — current-main SOT-2619 measurement)"
.venv/bin/python scripts/run_focused_gate.py --label sot2633_g3_2619verify \
  --issue SOT-2633 --target "74,84,78,62,75" --workers 8
code=$?
echo "=== SOT-2633 focused gate done $(date -u +%FT%TZ) exit=$code ==="
exit $code
