#!/usr/bin/env bash
# SOT-2668 — focused gate for the heading→printed-page locator store child (cycle5 クラスタC5, official:false).
#
# Same env as scripts/sonnet_child_action_row_focused.sh plus the cycle-5 lever:
#   RAG_HEADING_PAGE_STORE=1  — question-independent 見出し→印字ページ store (docx=LibreOffice レンダ→フッタ
#                               印字番号 / scan-pdf=永続OCRの[ページN]マーカ) read at serve time with NO genai
#                               call; feeds the fact-layer deterministic lane (idx12 docx見出し→印字ページ,
#                               idx18 会議ID:M04→PDF見出し→印字ページ) and the heading_page_lookup tool.
# Targets:
#   idx12 (docx「WBS観点の進捗状況」の見出し → 印字ページ, gold=2ページ)
#   idx18 (会議ID:M04 会議録の「進捗サマリ」記載ページ, gold=2ページ目)
# Gate FAILS on any Sonnet sentinel regression. --dev ⇒ official:false (non-flash stack).
set -euo pipefail
cd /workspaces/signate-messy-drive-rag

.venv/bin/python scripts/check_flag_manifest.py "$0" || exit 1

# Rebuild the heading→page store (LLM-free, question-independent, idempotent). HOME set so LibreOffice
# has a writable profile dir inside the container.
HOME="${HOME:-/tmp}" .venv/bin/python -m src.rag.index.heading_page_store

export VERTEX_LOCATION=global
export GEN_MODEL=gemini-3.6-flash
export GEN_MODEL_HARD=gemini-3.6-flash
export VISION_MODEL=gemini-3.6-flash
export PYTHONPATH=/tmp/genai_patch:.

# --- champion Wave A + B1 flags, verbatim from sonnet_child_action_row_focused.sh ---
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

# --- cycle-5 lever (SOT-2668) ---
export RAG_HEADING_PAGE_STORE=1

# --- investigator on flat-rate Sonnet; parallelism 1 = shared-limit protection ---
export RAG_INVESTIGATOR_BACKEND=claude-mcp
export RAG_CLAUDE_MCP_MODEL=sonnet
# Reuse the reviewed cycle-4.5 Sonnet cache for unchanged LLM/sentinel lanes; the two
# new target answers are deterministic (heading_page lane) and bypass this cache.
export RAG_CLAUDE_MCP_RESUME=artifacts/gold100_sonnet_manual0812_resume.jsonl
export RAG_MCP_TOOL_LOG=artifacts/sonnet_child_heading_page_tool_calls.jsonl

TARGET="${TARGET:-12,18}"

echo "=== SOT-2668 Sonnet heading→page focused gate start $(date -u +%FT%TZ) ==="
echo "BACKEND=$RAG_INVESTIGATOR_BACKEND FACT_LAYER=$RAG_FACT_LAYER HEADING_PAGE_STORE=$RAG_HEADING_PAGE_STORE TARGET=$TARGET"
.venv/bin/python scripts/run_focused_gate.py \
  --label sot_child_heading_page \
  --target "$TARGET" \
  --sentinels scripts/sonnet_sentinels.json \
  --issue SOT-2668 \
  --dev --no-smoke \
  --workers 1
echo "=== SOT-2668 Sonnet heading→page focused gate done $(date -u +%FT%TZ) exit=$? ==="
