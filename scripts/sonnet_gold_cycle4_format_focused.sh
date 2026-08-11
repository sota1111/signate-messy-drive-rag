#!/usr/bin/env bash
# SOT-2656 — focused gate for Sonnet gold100 cycle 4, クラスタE 値保存書式契約 (official:false lane).
#
# Same production env as scripts/sonnet_gold_cycle3_focused.sh plus the cycle-4 value-norm lever:
#   RAG_FORMAT_STRIP_PAREN=1  — SOT-2650 trailing 括弧内付加情報 strip (recovers idx4/59/88 …)
#   RAG_FORMAT_VALUE_NORM=1   — SOT-2656 値保存回答正規化: approximation prefix (idx8/36), counter
#                               suffix for a bare-count ask (idx41/92), sentence frame (idx6).
# Targets: the クラスタE wrong class idx4/6/8/12/41/59/88/92. (idx12「2ページ目」→「2ページ」は gold 両形
# 存在ゆえ変換せず — idx18 回帰防止。番兵 10 問で既存 MATCH の非回帰を検出。)
# Gate FAILS on any Sonnet sentinel regression. --dev ⇒ official:false (non-flash stack).
set -euo pipefail
cd /workspaces/signate-messy-drive-rag

.venv/bin/python scripts/check_flag_manifest.py "$0" || exit 1

export VERTEX_LOCATION=global
export GEN_MODEL=gemini-3.6-flash
export GEN_MODEL_HARD=gemini-3.6-flash
export VISION_MODEL=gemini-3.6-flash
export PYTHONPATH=/tmp/genai_patch:.

# --- champion Wave A + B1 flags, verbatim from sonnet_gold_cycle3_focused.sh ---
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

# --- cycle-4 lever (SOT-2656) ---
export RAG_FORMAT_VALUE_NORM=1

# --- investigator on flat-rate Sonnet; parallelism 1 = shared-limit protection ---
export RAG_INVESTIGATOR_BACKEND=claude-mcp
export RAG_CLAUDE_MCP_MODEL=sonnet
export RAG_CLAUDE_MCP_RESUME=artifacts/sonnet_cycle4_format_focused_resume.jsonl
export RAG_MCP_TOOL_LOG=artifacts/sonnet_cycle4_format_focused_tool_calls.jsonl

TARGET="4,6,8,12,41,59,88,92"

echo "=== SOT-2656 Sonnet cycle4 value-norm focused gate start $(date -u +%FT%TZ) ==="
echo "STRIP_PAREN=$RAG_FORMAT_STRIP_PAREN VALUE_NORM=$RAG_FORMAT_VALUE_NORM FORBID_GEMINI=$RAG_FORBID_GEMINI TARGET=$TARGET"
.venv/bin/python scripts/run_focused_gate.py \
  --label sot2656_sonnet_cycle4_format \
  --target "$TARGET" \
  --sentinels scripts/sonnet_sentinels.json \
  --issue SOT-2656 \
  --dev --no-smoke \
  --workers 1
echo "=== SOT-2656 Sonnet cycle4 value-norm focused gate done $(date -u +%FT%TZ) exit=$? ==="
