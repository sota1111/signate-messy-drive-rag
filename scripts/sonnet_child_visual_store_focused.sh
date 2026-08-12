#!/usr/bin/env bash
# SOT-2653 — focused gate for the xlsx visual-facts store (official:false / Sonnet lane).
#
# Same env as scripts/sonnet_gold_cycle3_focused.sh plus the new lever:
#   RAG_VISUAL_STORE=1  — build-time xlsx 可視事実ストア（chart系列→カラム / ハイライト全数収蔵 /
#                         相関CF / 行・列ハイライト交差）の serve 直答レーン + investigator ツール（既定OFF）
# Targets (cycle4 クラスタB):
#   idx39 (chart_read — 青潮 train.xlsx グラフ1 が可視化するカラム = hum)
#   idx65 (numeric/highlight — 白峰 相関シートの黄CF条件 = 相関係数 < -0.99)
#   idx97 (numeric/highlight — 青葉バイオ 黄ハイライト交差2セルの値差 = 272)
#   idx82 (二次 — 蒼泉会 スケジュール WBS の淡橙ハイライト行の タスクID 列挙)
# Gate FAILS on any Sonnet sentinel regression. --dev ⇒ official:false (non-flash stack).
set -euo pipefail
cd /workspaces/signate-messy-drive-rag

.venv/bin/python scripts/check_flag_manifest.py "$0" || exit 1

# Build the visual store (LLM-free, question-independent) so serve reads the baked artifact.
.venv/bin/python -m src.rag.index.visual_store

export VERTEX_LOCATION=global
export GEN_MODEL=gemini-3.6-flash
export GEN_MODEL_HARD=gemini-3.6-flash
export VISION_MODEL=gemini-3.6-flash
export PYTHONPATH=/tmp/genai_patch:.

# --- champion Wave A + B1 flags, verbatim from SOT-2650/sonnet_gold_cycle3_focused.sh ---
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

# --- cycle-2/3 levers (kept) ---
export RAG_FACT_LAYER=1
export RAG_FORBID_GEMINI=1
export LLM_PROVIDER=claude-cli
export CLAUDE_CLI_MODEL=sonnet
export RAG_OCR_STORE=1
export RAG_FORMAT_STRIP_PAREN=1

# --- SOT-2653 lever: xlsx visual-facts store direct-answer lane + tool ---
export RAG_VISUAL_STORE=1

# --- investigator on flat-rate Sonnet (SOT-2627); parallelism 1 = shared-limit protection ---
export RAG_INVESTIGATOR_BACKEND=claude-mcp
export RAG_CLAUDE_MCP_MODEL=sonnet
export RAG_CLAUDE_MCP_RESUME=artifacts/sonnet_child_visual_store_resume.jsonl
export RAG_MCP_TOOL_LOG=artifacts/sonnet_child_visual_store_tool_calls.jsonl

TARGET="39,65,97,82"

echo "=== SOT-2653 visual-store focused gate start $(date -u +%FT%TZ) ==="
echo "BACKEND=$RAG_INVESTIGATOR_BACKEND FACT_LAYER=$RAG_FACT_LAYER VISUAL_STORE=$RAG_VISUAL_STORE TARGET=$TARGET"
.venv/bin/python scripts/run_focused_gate.py \
  --label sot2653_visual_store \
  --target "$TARGET" \
  --sentinels scripts/sonnet_sentinels.json \
  --issue SOT-2653 \
  --dev --no-smoke \
  --workers 1
echo "=== SOT-2653 visual-store focused gate done $(date -u +%FT%TZ) exit=$? ==="
