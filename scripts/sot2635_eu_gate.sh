#!/usr/bin/env bash
# SOT-2635 — focused calibration of the commit-time expected-utility gate (RAG_EU_GATE) on the answer
# path (investigator.answer_question), paired with the answer-increasing config. Same champion Wave A base
# and same focused set as the SOT-2634 baseline (so the OFF→ON deltas are apples-to-apples), with the EU
# gate turned ON. The commit threshold τ is the calibration knob:
#
#   bash scripts/sot2635_eu_gate.sh -9    # PROBE: τ=-9 ⇒ commit everything (answers = baseline), record U
#   bash scripts/sot2635_eu_gate.sh 0     # DEFAULT τ=0 ⇒ real gate (倒す U≤0 commits to 棄権)
#   bash scripts/sot2635_eu_gate.sh 0.12  # a calibrated τ chosen offline from the probe's per-idx U
#
# The per-idx eu_gate decision (tier / U / commit / flip / signal bundle) is carried into the focused-gate
# JSON's per-row ``interventions`` (SOT-2629 telemetry), so τ can be picked offline without re-running.
# OFFICIAL flash-3.6 stack (SOT-2625). No serve-path change beyond the RAG_EU_GATE opt-in.
set -euo pipefail
cd /workspaces/signate-messy-drive-rag

TAU="${1:--9}"                       # default: probe threshold (commit everything, record U)
LABEL="${2:-sot2635_eu_tau${TAU}}"
LABEL="${LABEL//./p}"; LABEL="${LABEL//-/m}"   # filesystem-safe label

# Same focused set as SOT-2634 (each cycle2 flag's target idx ∪ cycle2 wrong idx).
TARGET="4,6,8,9,12,24,27,28,32,38,47,50,57,63,67,73,76,83,87,94,98,99"

# Official measurement model (08-10): gemini-3.6-flash on global (no pro on global).
export VERTEX_LOCATION=global
export GEN_MODEL=gemini-3.6-flash
export GEN_MODEL_HARD=gemini-3.6-flash
export VISION_MODEL=gemini-3.6-flash
export PYTHONPATH=/tmp/genai_patch:.

# --- champion answer-increase / error-type flags (r1, net32) ---
export RAG_FIRST_MOVE_ROUTING=1 RAG_SPIN_DETECTION=1 RAG_ADAPTIVE_BUDGET=1 RAG_EVIDENCE_CACHE=1
export RAG_BUDGET_BOUNDARY_RESEARCH=1 RAG_UNANSWERABLE_FALLBACK=1 RAG_PDF_OCR=1 RAG_SHARE_CORPUS_PROFILE=1
export RAG_CANONICAL_MANIFEST=1 RAG_EVIDENCE_INDEX=1 RAG_STRUCTURE_STORE=1
export RAG_GRANULARITY_NORMALIZATION=1 RAG_XLSX_EMBEDDED_IMAGE=1 RAG_CONFLICT_RESOLUTION=1 GATE_EXEC_CORRECT=1
export RAG_NUMERIC_FEATURE_CORR=1 RAG_RELEVANCE_STRICT=1 RAG_HIGHLIGHT_EXTRA=1 RAG_FONT_EMPHASIS=1
export RAG_FILE_GREP_INDEX_CANDIDATES=1
export RAG_FORMAT_EVENTS=1
# --- Wave A champion deterministic router with B1-only (SOT-2618 single-gate config) ---
export RAG_DET_PIPELINE_ROUTER=1
export RAG_DET_PIPELINE_B1=1
export RAG_DET_PIPELINE_B2=0
# --- cycle-2 flags: all OFF (SOT-2634 confirmed all four DROP) ---
export RAG_SPIN_PIVOT=0 RAG_SEARCH_CAP=0 RAG_OPERAND_PREFILL=0 RAG_CONDITION_PREIR=0
# --- THIS issue: the commit-time expected-utility gate ON, at threshold τ ---
export RAG_EU_GATE=1
export RAG_EU_GATE_TAU="$TAU"

echo "=== SOT-2635 EU-gate focused flag=RAG_EU_GATE τ=$TAU label=$LABEL start $(date -u +%FT%TZ) ==="
set +e
.venv/bin/python scripts/run_focused_gate.py \
  --label "$LABEL" --target "$TARGET" --workers 6 --issue SOT-2635
rc=$?
set -e
echo "=== SOT-2635 EU-gate focused τ=$TAU done $(date -u +%FT%TZ) gate_exit=$rc ==="
[ "$rc" = "3" ] && exit 3 || exit 0
