#!/usr/bin/env bash
# TEMPLATE for a consolidated gold100 integration-measurement script (SOT-2624).
#
# Copy this to scripts/sot_<issue>_gold100.sh and fill in the flags for the configuration you are
# measuring. The point of the header is the flag-manifest preflight: it reconciles the RAG_* flags
# this script exports against the flags the source actually reads, BEFORE the (often one-shot,
# expensive) measurement — so a typo / export-omission / retired flag can never silently corrupt the
# run the way RAG_DERIVED_FORMAT_CONTRACTS did in cycle 2 (adversarial-review hole H3).
set -euo pipefail
cd /workspaces/signate-messy-drive-rag

# --- flag-manifest preflight (SOT-2624) — keep this at the very top, before any measurement ---
#   default : fails only on (b) exported-but-unknown-to-source RAG_* (typo / retired flag).
#   --strict: ALSO fails on (a) a code-read flag left on its default (not exported) — use when the
#             measurement's correctness depends on every flag being set explicitly.
.venv/bin/python scripts/check_flag_manifest.py "$0" || exit 1
# .venv/bin/python scripts/check_flag_manifest.py "$0" --strict || exit 1   # stricter variant

# Official measurement model (08-10): gemini-3.6-flash on global (no pro on global).
export VERTEX_LOCATION=global
export GEN_MODEL=gemini-3.6-flash
export GEN_MODEL_HARD=gemini-3.6-flash
export VISION_MODEL=gemini-3.6-flash
export PYTHONPATH=/tmp/genai_patch:.

# --- configuration under test: set every RAG_* flag this run depends on (do NOT rely on defaults) ---
# export RAG_DET_PIPELINE_ROUTER=1
# export RAG_FORMAT_EVENTS=1
# ... add the flags for the configuration you are measuring ...

echo "=== gold100 start $(date -u +%FT%TZ) ==="
.venv/bin/python -m scoring.gold_offline --run --workers 8 \
  --out artifacts/gold100_<issue>.json
echo "=== gold100 done $(date -u +%FT%TZ) exit=$? ==="
