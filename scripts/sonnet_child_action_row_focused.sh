#!/usr/bin/env bash
# SOT-2652 — focused gate for the scanned-PDF action-row store child (cycle4 クラスタA, official:false).
#
# Same env as scripts/sonnet_gold_cycle3_focused.sh plus the cycle-4 lever:
#   RAG_ACTION_ROW_STORE=1  — question-independent action-row store (会議録アクション表 / 報告資料の
#                             優先タスク・優先フォロー) read at serve time with NO genai call; feeds the
#                             fact-layer deterministic lane (idx20 優先タスク担当集合, idx70 Open優先
#                             フォロー × 会議録完了突合) and the action_row_lookup investigator tool.
# Targets:
#   idx20 (優先タスク: 担当2名条件での行選択 — deterministic priority-task-by-assignee-set lane)
#   idx70 (Open 優先フォロー AI × 会議録完了記録の突合 — deterministic set-difference lane)
#   idx34/93 (証拠はストア/OCRで提供・束縛できなければ棄権のまま — 無理な回答化はしない)
# Gate FAILS on any Sonnet sentinel regression. --dev ⇒ official:false (non-flash stack).
set -euo pipefail
cd /workspaces/signate-messy-drive-rag

.venv/bin/python scripts/check_flag_manifest.py "$0" || exit 1

export VERTEX_LOCATION=global
export GEN_MODEL=gemini-3.6-flash
export GEN_MODEL_HARD=gemini-3.6-flash
export VISION_MODEL=gemini-3.6-flash
export PYTHONPATH=/tmp/genai_patch:.

# --- champion Wave A + B1 flags, verbatim from SOT-2628/sonnet_gold_cycle2.sh ---
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

# --- cycle-4 lever (SOT-2652) ---
export RAG_ACTION_ROW_STORE=1

# --- investigator on flat-rate Sonnet (SOT-2627); parallelism 1 = shared-limit protection ---
export RAG_INVESTIGATOR_BACKEND=claude-mcp
export RAG_CLAUDE_MCP_MODEL=sonnet
# Reuse the reviewed cycle-4 Sonnet cache for unchanged LLM/sentinel lanes; the two
# new target answers are deterministic and bypass this cache.
export RAG_CLAUDE_MCP_RESUME=artifacts/gold100_cycle4_sonnet_resume.jsonl
export RAG_MCP_TOOL_LOG=artifacts/sonnet_child_action_row_tool_calls.jsonl

TARGET="${TARGET:-20,34,70,93}"

echo "=== SOT-2652 Sonnet action-row focused gate start $(date -u +%FT%TZ) ==="
echo "BACKEND=$RAG_INVESTIGATOR_BACKEND FACT_LAYER=$RAG_FACT_LAYER ACTION_ROW_STORE=$RAG_ACTION_ROW_STORE TARGET=$TARGET"
.venv/bin/python scripts/run_focused_gate.py \
  --label sot_child_action_row \
  --target "$TARGET" \
  --sentinels scripts/sonnet_sentinels.json \
  --issue SOT-2652 \
  --dev --no-smoke \
  --workers 1
echo "=== SOT-2652 Sonnet action-row focused gate done $(date -u +%FT%TZ) exit=$? ==="
