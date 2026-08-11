#!/usr/bin/env bash
# SOT-2634 — single-flag ablation of the four cycle-2 waste-elimination / operand flags.
# For ONE flag name ($1 ∈ spin_pivot|search_cap|operand_prefill|condition_preir) this sets the champion
# Wave A base (net40 = r1 answer-increase + RAG_FORMAT_EVENTS + RAG_DET_PIPELINE_ROUTER with B1-only)
# and turns ON exactly that ONE cycle-2 flag, then runs scripts/run_focused_gate.py over the common
# focused set (all four flags' original targets ∪ cycle2 wrong idx) with the mandatory 10-question
# MATCH sentinel set (SOT-2623) on the OFFICIAL flash-3.6 stack (SOT-2625). No serve-path change.
#
#   .venv/bin/python present; run from repo root:  bash scripts/sot2634_ablation.sh spin_pivot
set -euo pipefail
cd /workspaces/signate-messy-drive-rag

FLAG="${1:?usage: sot2634_ablation.sh <spin_pivot|search_cap|operand_prefill|condition_preir>}"

# Common focused set (issue 実装内容 #1): union of each flag's original target idx + cycle2 wrong idx.
#  spin_pivot/search_cap targets (search-heavy set): 76,50,99,67,38,87,32,63,83,98
#  operand_prefill targets:                          76,47,50,99,8,28,57,63
#  condition_preir targets:                          76,47,57,6,27,94
#  cycle2 wrong idx (issue):                          4,9,12,24,47,73,83,94,99
TARGET="4,6,8,9,12,24,27,28,32,38,47,50,57,63,67,73,76,83,87,94,98,99"

# Official measurement model (08-10): gemini-3.6-flash on global (no pro on global)
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
# --- cycle-2 flags: all OFF, then turn ON exactly the one under test ---
export RAG_SPIN_PIVOT=0 RAG_SEARCH_CAP=0 RAG_OPERAND_PREFILL=0 RAG_CONDITION_PREIR=0
case "$FLAG" in
  spin_pivot)      export RAG_SPIN_PIVOT=1 ;;
  search_cap)      export RAG_SEARCH_CAP=1 ;;
  operand_prefill) export RAG_OPERAND_PREFILL=1 ;;
  condition_preir) export RAG_CONDITION_PREIR=1 ;;
  baseline)        : ;;  # champion Wave A, all four cycle-2 flags OFF (OFF reference for OFF→ON deltas)
  *) echo "[error] unknown flag: $FLAG" >&2; exit 3 ;;
esac

echo "=== SOT-2634 ablation flag=$FLAG start $(date -u +%FT%TZ) ==="
echo "SPIN_PIVOT=$RAG_SPIN_PIVOT SEARCH_CAP=$RAG_SEARCH_CAP OPERAND_PREFILL=$RAG_OPERAND_PREFILL CONDITION_PREIR=$RAG_CONDITION_PREIR"
# gate exit 2 (sentinel regression) is MEANINGFUL ablation data (that flag is a DROP candidate), not a
# script error — capture it instead of aborting. exit 3 = precondition/model-guard error (real abort).
set +e
.venv/bin/python scripts/run_focused_gate.py \
  --label "sot2634_${FLAG}" --target "$TARGET" --workers 6 --issue SOT-2634
rc=$?
set -e
echo "=== SOT-2634 ablation flag=$FLAG done $(date -u +%FT%TZ) gate_exit=$rc ==="
[ "$rc" = "3" ] && exit 3 || exit 0
