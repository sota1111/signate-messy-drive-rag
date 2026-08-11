#!/usr/bin/env bash
# SOT-2642 Cycle-4 convergence — SONNET dev gold100 (model-invariance verification, half 2 of 2).
#
# Identical HEAD / identical champion pipeline (Wave A + B1) / identical cycle-4 flags as
# scripts/sot_cycle4_gold100.sh, differing ONLY in:
#   * RAG_INVESTIGATOR_BACKEND=claude-mcp — drive the investigator tool loop on flat-rate Sonnet
#     (SOT-2627/2628) instead of gemini-3.6-flash.
#   * RAG_COMMIT_GATE_ENFORCE=1 — ENFORCE the gate. The claude-mcp backend has NO inline precision
#     guards (exec_verifier / enum universe / formatting live inside the Gemini loop), so its
#     submit_answer would commit raw values unchecked. Enforcement makes REJECT feed an in-band retry
#     and, after RAG_COMMIT_GATE_ABSTAIN_AFTER consecutive rejects, degrade to ABSTAIN. This is the
#     cycle-4 thesis: the deterministic gate SUBSTITUTES for the missing inline guards, so commit
#     precision stops depending on the model. (flash keeps enforce OFF — its inline guards are already
#     authoritative and enforcing there over-rejects derived values; SOT-2639.)
#
# ***OFFICIAL:FALSE*** — dev measurement only (--no-official ⇒ history official:false). The official
# net / non-regression judgment stays pinned to gemini-3.6-flash. Baseline to beat (SOT-2628, same base
# WITHOUT commit_gate/neutral prompt): net18, match46/abstain26/wrong28. cycle-4 target: wrong ≤ 10,
# net ≥ 35, wrong-side divergence ≈ 0.
#
# COST: the Sonnet tool loop is $0 (flat-rate). USAGE-LIMIT: shared account-wide — run at parallelism 1
# when workers idle. On a limit the backend abstains that question and persists ANSWERED questions to the
# resume sidecar, so re-running continues where it stopped (may span sessions). LB 提出はしない。
set -euo pipefail
cd /workspaces/signate-messy-drive-rag

# --- flag-manifest preflight (SOT-2624): abort if this script exports a RAG_* no source reads ---
.venv/bin/python scripts/check_flag_manifest.py "$0" || exit 1

# Residual-Gemini model for anything OUTSIDE the investigator loop (vision fallback / naturalization).
export VERTEX_LOCATION=global
export GEN_MODEL=gemini-3.6-flash
export GEN_MODEL_HARD=gemini-3.6-flash
export VISION_MODEL=gemini-3.6-flash

# genai per-request timeout injection (SOT-2568 method)
export PYTHONPATH=/tmp/genai_patch:.

# --- champion answer-increase / error-type flags (r1) — IDENTICAL to flash cycle4 ---
export RAG_FIRST_MOVE_ROUTING=1 RAG_SPIN_DETECTION=1 RAG_ADAPTIVE_BUDGET=1 RAG_EVIDENCE_CACHE=1
export RAG_BUDGET_BOUNDARY_RESEARCH=1 RAG_UNANSWERABLE_FALLBACK=1 RAG_PDF_OCR=1 RAG_SHARE_CORPUS_PROFILE=1
export RAG_CANONICAL_MANIFEST=1 RAG_EVIDENCE_INDEX=1 RAG_STRUCTURE_STORE=1
export RAG_GRANULARITY_NORMALIZATION=1 RAG_XLSX_EMBEDDED_IMAGE=1 RAG_CONFLICT_RESOLUTION=1 GATE_EXEC_CORRECT=1
export RAG_NUMERIC_FEATURE_CORR=1 RAG_RELEVANCE_STRICT=1 RAG_HIGHLIGHT_EXTRA=1 RAG_FONT_EMPHASIS=1
export RAG_FILE_GREP_INDEX_CANDIDATES=1
export RAG_FORMAT_EVENTS=1
# --- Wave A champion deterministic router with B1-only (SOT-2618 single-gate config = net40) ---
export RAG_DET_PIPELINE_ROUTER=1
export RAG_DET_PIPELINE_B1=1
export RAG_DET_PIPELINE_B2=0
# --- cycle-4 model-invariance parts (IDENTICAL flags; enforce differs per header) ---
export RAG_COMMIT_GATE=1
export RAG_COMMIT_GATE_ENFORCE=1    # SONNET = ENFORCE (guard-less backend needs the gate to守る)
export RAG_NEUTRAL_PROMPT=1
export RAG_MCP_COMMIT_GATE_LOG=artifacts/gold100_cycle4_sonnet_commit_gate.jsonl

# --- dev-only: drive the investigator tool loop on flat-rate Sonnet via claude CLI + MCP (SOT-2627) ---
export RAG_INVESTIGATOR_BACKEND=claude-mcp
export RAG_CLAUDE_MCP_MODEL=sonnet
export RAG_CLAUDE_MCP_RESUME=artifacts/gold100_cycle4_sonnet_resume.jsonl   # resume sidecar (interrupt/resume)
export RAG_MCP_TOOL_LOG=artifacts/gold100_cycle4_sonnet_tool_calls.jsonl    # per tools/call trace

echo "=== SOT-2642 cycle4 SONNET dev gold100 (official:false) start $(date -u +%FT%TZ) ==="
echo "BACKEND=$RAG_INVESTIGATOR_BACKEND MODEL=$RAG_CLAUDE_MCP_MODEL ROUTER=$RAG_DET_PIPELINE_ROUTER B1=$RAG_DET_PIPELINE_B1 B2=$RAG_DET_PIPELINE_B2"
echo "COMMIT_GATE=$RAG_COMMIT_GATE ENFORCE=$RAG_COMMIT_GATE_ENFORCE NEUTRAL_PROMPT=$RAG_NEUTRAL_PROMPT RESUME=$RAG_CLAUDE_MCP_RESUME"
.venv/bin/python -m scoring.gold_offline --run --workers 1 --no-official \
  --out artifacts/gold100_cycle4_sonnet.json
echo "=== SOT-2642 cycle4 SONNET dev gold100 done $(date -u +%FT%TZ) exit=$? ==="
