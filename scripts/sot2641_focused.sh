#!/usr/bin/env bash
# SOT-2641 (PLAN SOT-2602, cycle4 4/6) — focused gate for the model-neutral SYSTEM_PROMPT.
# Runs the mandatory existing-MATCH sentinel set on the OFFICIAL flash-3.6 stack, on top of the
# cycle3 champion env (SOT-2632 net40), with ONLY the new prompt flag added (RAG_NEUTRAL_PROMPT=1).
# The neutral prompt drops the flash-tuned "安易に棄権しない"/"棄権しない" drivers for a model-neutral
# role declaration (explore aggressively; commit verified by the deterministic commit_gate). Since the
# drivers were a brake-release for the CONSERVATIVE flash, the flash-side risk is a RETURN to
# over-abstention — so this gate FAILS iff any sentinel (a champion MATCH) drops from MATCH.
# Mode: BACKEND=flash → OFFICIAL flash 3.6 sentinels; BACKEND=sonnet → --dev Sonnet (claude-cli) probe.
# Run ONCE per backend. (judge=codex, gold=predictions_test_v3_final.csv)
set -euo pipefail
cd /workspaces/signate-messy-drive-rag

BACKEND="${BACKEND:-flash}"

# genai per-request timeout injection (SOT-2568 method) — shared by both backends.
export PYTHONPATH=/tmp/genai_patch:.

# --- cycle3 champion (SOT-2632 net40) env, verbatim ---
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
export RAG_G2_LOOKUP_PORT=1
export RAG_DERIVED_FORMAT_CONTRACTS=1

# --- this issue: model-neutral system prompt ---
# Honor a caller-preset value so the SAME champion env can produce the ON run and the mandatory OFF
# baseline (SOT-2634: a single flash-3.6 sentinel drop is only attributable against an OFF baseline on
# the same sentinels — the serve path is non-deterministic, SOT-2623). Default ON.
export RAG_NEUTRAL_PROMPT="${RAG_NEUTRAL_PROMPT:-1}"

if [[ "$BACKEND" == "sonnet" ]]; then
  export LLM_PROVIDER=claude-cli
  echo "=== SOT-2641 focused gate (Sonnet dev) start $(date -u +%FT%TZ) ==="
  echo "RAG_NEUTRAL_PROMPT=$RAG_NEUTRAL_PROMPT LLM_PROVIDER=$LLM_PROVIDER"
  .venv/bin/python scripts/run_focused_gate.py --label sot2641_neutral_sonnet \
    --dev --workers 6
  code=$?
else
  # Official measurement model (08-10): gemini-3.6-flash on global (no pro on global)
  export VERTEX_LOCATION=global
  export GEN_MODEL=gemini-3.6-flash
  export GEN_MODEL_HARD=gemini-3.6-flash
  export VISION_MODEL=gemini-3.6-flash
  echo "=== SOT-2641 focused gate (flash official) start $(date -u +%FT%TZ) ==="
  echo "GEN_MODEL=$GEN_MODEL VERTEX_LOCATION=$VERTEX_LOCATION RAG_NEUTRAL_PROMPT=$RAG_NEUTRAL_PROMPT"
  .venv/bin/python scripts/run_focused_gate.py --label sot2641_neutral_flash \
    --workers 8
  code=$?
fi
echo "=== SOT-2641 focused gate ($BACKEND) done $(date -u +%FT%TZ) exit=$code ==="
exit $code
