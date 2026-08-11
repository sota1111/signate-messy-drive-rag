#!/usr/bin/env bash
# SOT-2642 Cycle-4 convergence — FLASH 公式 gold100 (model-invariance verification, half 1 of 2).
#
# cycle4 の主張は「commit 精度はモデル非依存にできる」。その最終実測。champion（Wave A + B1, SOT-2610
# net40）の deterministic-first パイプラインはそのままに、モデル不変アーキの 2 部品を足す:
#   * RAG_COMMIT_GATE=1     — SOT-2637 の backend 非依存 commit gate を配線（SOT-2639 Gemini 終端 /
#                            SOT-2640 MCP submit_answer）。
#   * RAG_NEUTRAL_PROMPT=1  — SOT-2641 のモデル中立 SYSTEM_PROMPT（探索は積極・commit はゲート任せ）。
#
# ***FLASH 側は enforce OFF（観測モード）***: SOT-2639 の Gemini 配線は等価保存で、RAG_COMMIT_GATE=1 だけ
# ならゲートの判定＋テレメトリは記録するが、ループ自身の inline ガードが権威のまま — commit 値は VERBATIM。
# flash に enforce を掛けると compute-record grounding が導出値（idx30 1.18%）を過剰却下し等価性が壊れる
# （SOT-2639 で実証）。よって flash は観測のみ ⇒ 原則 net 非劣化。劣化したら「配線バグ」として扱う。
# Sonnet 側（guard-less claude-mcp）だけが enforce ON でゲートに commit を守らせる（scripts/
# sot_cycle4_sonnet_gold100.sh）。「同一 HEAD・同一パイプライン、backend と enforce だけが差」= 収束測定。
#
# Official model = gemini-3.6-flash @ global (judge=codex). 本 issue で公式 gold100 は 1 回のみ。
# 昇格ゲート: net > 40 AND wrong ≤ 7（commit_gate 等価配線なので原則非劣化）。LB 提出はしない。
set -euo pipefail
cd /workspaces/signate-messy-drive-rag

# --- flag-manifest preflight (SOT-2624): abort if this script exports a RAG_* no source reads ---
.venv/bin/python scripts/check_flag_manifest.py "$0" || exit 1

# Official measurement model (08-10): gemini-3.6-flash on global (no pro on global)
export VERTEX_LOCATION=global
export GEN_MODEL=gemini-3.6-flash
export GEN_MODEL_HARD=gemini-3.6-flash
export VISION_MODEL=gemini-3.6-flash

# genai per-request timeout injection (SOT-2568 method)
export PYTHONPATH=/tmp/genai_patch:.

# --- champion answer-increase / error-type flags (r1) ---
export RAG_FIRST_MOVE_ROUTING=1 RAG_SPIN_DETECTION=1 RAG_ADAPTIVE_BUDGET=1 RAG_EVIDENCE_CACHE=1
export RAG_BUDGET_BOUNDARY_RESEARCH=1 RAG_UNANSWERABLE_FALLBACK=1 RAG_PDF_OCR=1 RAG_SHARE_CORPUS_PROFILE=1
export RAG_CANONICAL_MANIFEST=1 RAG_EVIDENCE_INDEX=1 RAG_STRUCTURE_STORE=1
export RAG_GRANULARITY_NORMALIZATION=1 RAG_XLSX_EMBEDDED_IMAGE=1 RAG_CONFLICT_RESOLUTION=1 GATE_EXEC_CORRECT=1
export RAG_NUMERIC_FEATURE_CORR=1 RAG_RELEVANCE_STRICT=1 RAG_HIGHLIGHT_EXTRA=1 RAG_FONT_EMPHASIS=1
export RAG_FILE_GREP_INDEX_CANDIDATES=1
export RAG_FORMAT_EVENTS=1
# --- Wave A champion deterministic router with B1-only (SOT-2618 single-gate config = net40) ---
export RAG_DET_PIPELINE_ROUTER=1
export RAG_DET_PIPELINE_B1=1   # Wave B1 document_extract (default ON)
export RAG_DET_PIPELINE_B2=0   # Wave B2 fact_lookup stays OFF
# --- cycle-4 model-invariance parts (this issue) ---
export RAG_COMMIT_GATE=1            # SOT-2637..2640 backend-agnostic commit gate (wired)
export RAG_COMMIT_GATE_ENFORCE=0    # FLASH = observational only (equivalence-preserving; see header)
export RAG_NEUTRAL_PROMPT=1         # SOT-2641 model-neutral SYSTEM_PROMPT
export RAG_MCP_COMMIT_GATE_LOG=artifacts/gold100_cycle4_flash_commit_gate.jsonl

echo "=== SOT-2642 cycle4 FLASH gold100 (official:true) start $(date -u +%FT%TZ) ==="
echo "GEN_MODEL=$GEN_MODEL VERTEX_LOCATION=$VERTEX_LOCATION ROUTER=$RAG_DET_PIPELINE_ROUTER B1=$RAG_DET_PIPELINE_B1 B2=$RAG_DET_PIPELINE_B2"
echo "COMMIT_GATE=$RAG_COMMIT_GATE ENFORCE=$RAG_COMMIT_GATE_ENFORCE NEUTRAL_PROMPT=$RAG_NEUTRAL_PROMPT"
.venv/bin/python -m scoring.gold_offline --run --workers 8 \
  --out artifacts/gold100_cycle4_flash.json
echo "=== SOT-2642 cycle4 FLASH gold100 done $(date -u +%FT%TZ) exit=$? ==="
