#!/usr/bin/env bash
# SOT-2636 Cycle-3 consolidated gold100 (Sonnet-trace ports + ablation winners + EU gate + format
# contracts). This is the SINGLE full-량 gold100 gate for cycle3 — each child ran focused only.
#
# Config = champion (Wave A + B1, SOT-2610 net40, with the SOT-2619 naturalization fix already merged
# ⇒ effective ~net42 per SOT-2633 idx74 free recovery) PLUS only the child gate winners:
#   * RAG_G2_LOOKUP_PORT=1        — SOT-2632 G2 lookup/derived port. focused GATE PASS, sentinels 10/10,
#                                   idx96 ABSTAIN→Perfect, no new wrong. (also enables the compute
#                                   encrypted-xlsx decrypt tool-gap fix, low-risk decode-only.)
#   * RAG_DERIVED_FORMAT_CONTRACTS=1 — SOT-2617 derived 書式契約 (unit/rounding/verbosity). "effective in
#                                   the answer-increasing config";未export事故で cycle2 は未測定のまま。
#
# DELIBERATELY EXCLUDED (evidence-backed, see docs/ai/gold100_cycle3.md / ledger):
#   * RAG_G1_HIGHLIGHT_PORT   — SOT-2631 focused PASS but target improvement 非実証 (非決定性) = inconclusive.
#   * RAG_EU_GATE             — SOT-2635 single-pass U non-discriminating (utility_discriminates=False).
#   * RAG_SPIN_PIVOT / RAG_SEARCH_CAP / RAG_OPERAND_PREFILL / RAG_CONDITION_PREIR — SOT-2634 all DROP
#                               (spin/cap fire→precision loss; prefill/preir inert=fired 0).
#   * RAG_G3_CHART_PORT       — SOT-2633 idx56 non-feasible (vision-only, SOT-2507 forbidden); not introduced.
#
# Official model = gemini-3.6-flash @ global (judge=codex). Run ONCE — promotion gate: net>40 AND
# match 非劣化 AND wrong ≤ 7. Follows scripts/sot_cycle2_gold100.sh.
set -euo pipefail
cd /workspaces/signate-messy-drive-rag

# --- flag-manifest preflight (SOT-2624): abort if this script exports a RAG_* no source knows ---
# Catches the cycle-2 H3 accident class (typo / retired-flag export / RAG_DERIVED_FORMAT_CONTRACTS
# silently-OFF). Warns (not fails) on impl flags left on default; add --strict to fail on those too.
.venv/bin/python scripts/check_flag_manifest.py "$0" || exit 1

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
# --- cycle-3 child gate winners (this issue) ---
export RAG_G2_LOOKUP_PORT=1            # SOT-2632 G2 lookup/derived port (+encrypted-xlsx decrypt fix)
export RAG_DERIVED_FORMAT_CONTRACTS=1  # SOT-2617 derived unit/rounding/verbosity format contracts

echo "=== SOT-2636 cycle3 gold100 start $(date -u +%FT%TZ) ==="
echo "GEN_MODEL=$GEN_MODEL VERTEX_LOCATION=$VERTEX_LOCATION ROUTER=$RAG_DET_PIPELINE_ROUTER B1=$RAG_DET_PIPELINE_B1 B2=$RAG_DET_PIPELINE_B2"
echo "G2_LOOKUP_PORT=$RAG_G2_LOOKUP_PORT DERIVED_FORMAT_CONTRACTS=$RAG_DERIVED_FORMAT_CONTRACTS"
.venv/bin/python -m scoring.gold_offline --run --workers 8 \
  --out artifacts/gold100_cycle3.json
echo "=== SOT-2636 cycle3 gold100 done $(date -u +%FT%TZ) exit=$? ==="
