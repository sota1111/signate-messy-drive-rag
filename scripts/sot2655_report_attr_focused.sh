#!/usr/bin/env bash
# SOT-2655 (事前計算事実層 追補 — cycle4 クラスタD) — focused gate for the 報告数値属性ストア.
#
# Same production env as scripts/sonnet_gold_cycle4_format_focused.sh (current Sonnet champion) plus the
# one new lever RAG_REPORT_ATTR_STORE=1: build-time 質問非依存 attributes from 最終報告/分析成果物 wired as
# NUMERIC 決定論直答レーン (最良モデル param / フェーズ工数合計 / 高影響特徴量×相関最大).
# Targets = the cluster-D abstains idx5 (max_depth) / idx28 (相関最大特徴量=BMI) / idx64 (フェーズA+B工数).
# These 3 are answered by the deterministic lane BEFORE the LLM loop (SOT-2649: Sonnet はツールを確実に
# 選ばない ⇒ 決定論レーンが回収の主経路). Gate FAILS on any Sonnet sentinel regression. --dev ⇒ official:false.
set -euo pipefail
cd /workspaces/signate-messy-drive-rag

.venv/bin/python scripts/check_flag_manifest.py "$0" || exit 1

# Rebuild the report-attr store on disk (LLM-free: reads persisted OCR + text-layer + analysis JSON +
# train tables — no genai). Cheap & idempotent; guarantees the serve lane reads the current artifact.
.venv/bin/python -c "from src.rag.index import report_attr_store as s; r=s.build()['report']; print('[build] report_attr_store', r)"

export VERTEX_LOCATION=global
export GEN_MODEL=gemini-3.6-flash
export GEN_MODEL_HARD=gemini-3.6-flash
export VISION_MODEL=gemini-3.6-flash
export PYTHONPATH=/tmp/genai_patch:.

# --- champion Wave A + B1 flags, verbatim from sonnet_gold_cycle4_format_focused.sh ---
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

# --- cycle-2/3/4 levers (current champion base) ---
export RAG_FACT_LAYER=1
export RAG_FORBID_GEMINI=1
export LLM_PROVIDER=claude-cli
export CLAUDE_CLI_MODEL=sonnet
export RAG_OCR_STORE=1
export RAG_FORMAT_STRIP_PAREN=1
export RAG_FORMAT_VALUE_NORM=1

# --- the one new lever under test (this issue) ---
export RAG_REPORT_ATTR_STORE=1

# --- investigator on flat-rate Sonnet; parallelism 1 = shared-limit protection ---
export RAG_INVESTIGATOR_BACKEND=claude-mcp
export RAG_CLAUDE_MCP_MODEL=sonnet
export RAG_CLAUDE_MCP_RESUME=artifacts/sot2655_report_attr_focused_resume.jsonl
export RAG_MCP_TOOL_LOG=artifacts/sot2655_report_attr_focused_tool_calls.jsonl

TARGET="5,28,64"

echo "=== SOT-2655 report-attr focused gate start $(date -u +%FT%TZ) ==="
echo "RAG_REPORT_ATTR_STORE=$RAG_REPORT_ATTR_STORE FORBID_GEMINI=$RAG_FORBID_GEMINI TARGET=$TARGET"
.venv/bin/python scripts/run_focused_gate.py \
  --label sot2655_report_attr \
  --target "$TARGET" \
  --sentinels scripts/sonnet_sentinels.json \
  --issue SOT-2655 \
  --dev --no-smoke \
  --workers 1
echo "=== SOT-2655 report-attr focused gate done $(date -u +%FT%TZ) exit=$? ==="
