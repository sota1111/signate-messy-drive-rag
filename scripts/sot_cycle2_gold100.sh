#!/usr/bin/env bash
# SOT-2622 Cycle-2 consolidated gold100 (waste elimination + operand binding). Champion
# (Wave A net40 = r1 answer-increase + RAG_FORMAT_EVENTS + RAG_DET_PIPELINE_ROUTER with B1-only)
# PLUS the four cycle-2 serve-path flags: RAG_SPIN_PIVOT (SOT-2614) / RAG_SEARCH_CAP (SOT-2620) /
# RAG_OPERAND_PREFILL (SOT-2616) / RAG_CONDITION_PREIR (SOT-2621). Wave B1 stays ON, B2 stays OFF
# (SOT-2618 single-gate config). SOT-2617/2619 formatting contracts are template-based ⇒ always
# active, no flag. Official model = gemini-3.6-flash @ global (judge=codex). Run ONCE — this is the
# single cycle-2 gold100 gate (net>40 promotion). Follows scripts/sot2613_gold100.sh.
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
# --- Wave A champion deterministic router with B1-only (SOT-2618 single-gate config) ---
export RAG_DET_PIPELINE_ROUTER=1
export RAG_DET_PIPELINE_B1=1   # Wave B1 document_extract (default ON)
export RAG_DET_PIPELINE_B2=0   # Wave B2 fact_lookup stays OFF
# --- cycle-2 waste-elimination + operand-binding flags (this issue) ---
export RAG_SPIN_PIVOT=1        # SOT-2614 forced pivot guard
export RAG_SEARCH_CAP=1        # SOT-2620 per-route search cap → registry fallback
export RAG_OPERAND_PREFILL=1   # SOT-2616 numeric operand structure-store prefill
export RAG_CONDITION_PREIR=1   # SOT-2621 what-if BranchCondition IR pre-loop injection

echo "=== SOT-2622 cycle2 gold100 start $(date -u +%FT%TZ) ==="
echo "GEN_MODEL=$GEN_MODEL VERTEX_LOCATION=$VERTEX_LOCATION ROUTER=$RAG_DET_PIPELINE_ROUTER B1=$RAG_DET_PIPELINE_B1 B2=$RAG_DET_PIPELINE_B2"
echo "SPIN_PIVOT=$RAG_SPIN_PIVOT SEARCH_CAP=$RAG_SEARCH_CAP OPERAND_PREFILL=$RAG_OPERAND_PREFILL CONDITION_PREIR=$RAG_CONDITION_PREIR"
.venv/bin/python -m scoring.gold_offline --run --workers 8 \
  --out artifacts/gold100_cycle2.json
echo "=== SOT-2622 cycle2 gold100 done $(date -u +%FT%TZ) exit=$? ==="
