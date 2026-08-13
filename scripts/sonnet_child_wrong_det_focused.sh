#!/usr/bin/env bash
# SOT-2688 — focused gate for cycle7 K5 (wrong 決定論修正 3 件, official:false).
#
# Base = scripts/sonnet_gold_cycle5.sh (net59 実証構成) PLUS the cycle6 promoted stores
# (RAG_DERIVED_COVERAGE / RAG_VDIFF_SUBJECT etc.) PLUS the three K5 levers under test:
#   RAG_CORR_SIGN=1        — 相関レーンの符号対応。負→ r<0 の厳密単独最小、正→ r>0 の厳密単独最大、
#                            修飾なし→従来どおり |r| 最大（idx4 保存）。idx91 の gold=campaign（最も負）。
#   RAG_BIN_RANGE_FORMAT=1 — ビン範囲の区間記法→チルダ書式 naturalization（値保存）。idx29
#                            「(6.088138, 6.288138]」→「6.088138 ~ 6.288138」。
#   RAG_SPECIAL_PROVISION=1 — 「特別規定なし＋一般規定内容」合成回答契約（prompt-only）。idx78
#                            ACTH/200時間超の特別規定不在＋取得済み一般規定 6.1〜6.3 条を1回答に合成。
# 全て serve/prompt gated・既定 OFF ⇒ OFF 時 byte-identical。
# Targets (gold=artifacts/predictions_test_v3_final.csv): 一次 idx91/29/78、レーン回帰番兵 idx4、
#   RAG_NONE_BARE 非衝突番兵 idx0/85（該当なし正の問い）。
# Gate FAILS on any Sonnet sentinel regression. --dev ⇒ official:false (non-flash stack; NOT a promotion
# basis). RAG_CLAUDE_MCP_RESUME=0 必須: serve/prompt change alters answers, so target AND sentinels
# re-derive live (SOT-2664 resume-replay 罠).
set -euo pipefail
cd /workspaces/signate-messy-drive-rag

.venv/bin/python scripts/check_flag_manifest.py "$0" || exit 1

# Rebuild the question-independent stores the base config + this axis rely on (LLM-free, idempotent).
# derived_metrics is rebuilt because the sign-aware correlation lane reads correlations.with_target.
.venv/bin/python -m src.rag.index.derived_metrics
.venv/bin/python -m src.rag.index.visual_store
.venv/bin/python -m src.rag.index.action_row_store
.venv/bin/python -m src.rag.index.case_finance_store
.venv/bin/python -c "from src.rag.index import report_attr_store as s; r=s.build()['report']; print('[build] report_attr_store', r)"
.venv/bin/python -c "from src.rag.index import diff_store as s; r=s.build(); print('[build] diff_store', r.get('pairs') if isinstance(r,dict) else r)"
HOME="${HOME:-/tmp}" .venv/bin/python -m src.rag.index.heading_page_store

export VERTEX_LOCATION=global
export GEN_MODEL=gemini-3.6-flash
export GEN_MODEL_HARD=gemini-3.6-flash
export VISION_MODEL=gemini-3.6-flash
export PYTHONPATH=/tmp/genai_patch:.

# --- champion Wave A + B1 flags (verbatim from scripts/sonnet_gold_cycle5.sh) ---
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

# --- cycle-4 levers (the five children, combined) ---
export RAG_ACTION_ROW_STORE=1
export RAG_VISUAL_STORE=1
export RAG_CASE_FINANCE_STORE=1
export RAG_REPORT_ATTR_STORE=1
export RAG_FORMAT_VALUE_NORM=1

# --- cycle-4.5 manual interim (Cerebras stack + bare-answer) ---
export RAG_TEXT_FTS=1
export RAG_UNIFIED_SEARCH=1
export RAG_PLAN_FANOUT=1
export RAG_BARE_ANSWER=1

# --- cycle-5 promoted children (base net59) ---
export RAG_FANOUT_FINISHER=1
export RAG_FANOUT_FINISHER_MAX=1
export RAG_VDIFF_SUBJECT=1
export RAG_NONE_BARE=1
export RAG_EXACT_LABEL=1
export RAG_HEADING_PAGE_STORE=1

# --- cycle-6 promoted children (idx91 の相関レーンは RAG_DERIVED_COVERAGE に載る) ---
export RAG_DERIVED_COVERAGE=1
export RAG_VDIFF_NORMALIZE=1

# --- axes under test (SOT-2688, cycle7 K5) ---
export RAG_CORR_SIGN=1
export RAG_BIN_RANGE_FORMAT=1
export RAG_SPECIAL_PROVISION=1

# --- investigator on flat-rate Sonnet; parallelism 1 = shared-limit protection ---
export RAG_INVESTIGATOR_BACKEND=claude-mcp
export RAG_CLAUDE_MCP_MODEL=sonnet
# MANDATORY: no resume cache — the serve/prompt change alters answers, so re-derive live (SOT-2664).
export RAG_CLAUDE_MCP_RESUME=0
export RAG_MCP_TOOL_LOG=artifacts/sonnet_child_wrong_det_tool_calls.jsonl

# 一次 idx91/29/78 ＋ レーン回帰番兵 idx4 ＋ NONE_BARE 非衝突番兵 idx0/85。
TARGET="${TARGET:-91,29,78,4,0,85}"

echo "=== SOT-2688 Sonnet K5 wrong-det focused gate start $(date -u +%FT%TZ) ==="
echo "BACKEND=$RAG_INVESTIGATOR_BACKEND CORR_SIGN=$RAG_CORR_SIGN BIN_RANGE=$RAG_BIN_RANGE_FORMAT SPECIAL_PROVISION=$RAG_SPECIAL_PROVISION RESUME=$RAG_CLAUDE_MCP_RESUME TARGET=$TARGET"
.venv/bin/python scripts/run_focused_gate.py \
  --label sot_child_wrong_det \
  --target "$TARGET" \
  --sentinels scripts/sonnet_sentinels.json \
  --issue SOT-2688 \
  --dev --no-smoke \
  --workers 1
echo "=== SOT-2688 Sonnet K5 wrong-det focused gate done $(date -u +%FT%TZ) exit=$? ==="
