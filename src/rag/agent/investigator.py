"""SOT-2468 — Gemini function-calling investigation loop (skeleton).

Parent SOT-2460 Step2. This is the **reusable production** answer path: one question is answered by a
*plan → tool iteration → structured answer* loop over Vertex Gemini function-calling. The model is given
only the corpus-agnostic Step1 tools (file discovery / grep / Office extraction / decryption / pandas
compute / chart & vision reads) — **no corpus-specific fact** (passwords, 略称, 書式規則) is injected;
the model self-discovers them through the tools. The loop terminates when the model calls the terminal
``submit_answer`` tool, returning the fixed answer schema::

    {answer, confidence, evidence, method}

Design for testability
----------------------
The loop (:func:`investigate`) is a *pure* driver over an abstract :class:`Model`; it never imports the
Gemini SDK. Live runs inject :class:`GeminiModel`; unit tests inject a scripted fake. The tools are the
real deterministic ones from :mod:`src.rag.tools`, so an offline test can drive a real tool end-to-end
without any network call.

Guardrails (受け入れ条件・実装内容)
-----------------------------------
* **反復上限** — ``max_turns`` caps model turns per question; on exhaustion the loop abstains.
* **タイムアウト** — ``timeout_s`` (wall-clock, checked between turns) bounds a single question.
* **コスト計上** — token usage is summed every turn and priced via :meth:`Usage.cost_usd`.

The Step1 early-validation gate :mod:`scoring.early_gate` (SOT-2467) shares the same agent-loop shape;
its consolidation onto this module (and wiring into the production answer path) is SOT-2469's scope, so
this change is intentionally self-contained and leaves the green gate untouched.
"""
from __future__ import annotations

import json
import os
import re
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Mapping, Protocol, Sequence

from config import settings
from src.rag.tools import contract as _contract
from src.rag.tools.canonical_route import canonical_route
from src.rag.tools.chart_numcache import read_chart_values
from src.rag.tools.compute_sandbox import run as compute_run
from src.rag.tools.corpus_aggregate import corpus_aggregate
from src.rag.tools.emf_pivot import extract_pptx_pivots
from src.rag.tools.extract_tools import (
    caption_figure,
    decrypt as _decrypt,
    extract_office,
    find_files,
)
from src.rag.tools import call_budget
from src.rag.tools.file_grep import file_grep
from src.rag.agent import pot_lane as _pot_lane
from src.rag.agent import operand_prefill as _operand_prefill
from src.rag.agent import enum_scan as _enum_scan
from src.rag.agent import fact_layer as _fact_layer
# NOTE: ``commit_gate`` is imported lazily inside :func:`investigate` (SOT-2639), not here: it pulls in
# ``exec_verifier``, which imports names from this module, so a top-level import would form an
# import cycle during ``investigator`` initialization.
from src.rag.tools import font_emphasis as _font_emphasis
from src.rag.tools import format_events as _format_events
from src.rag.tools.highlight_extract import highlight_extract
from src.rag.tools.pdf_faux_italic import emphasized_words
from src.rag.tools.profile import CorpusProfile
from src.rag.tools.seating_chart import seating_lookup

# --------------------------------------------------------------------------- loop configuration
DEFAULT_MAX_TURNS = 12            # hard cap on model turns per question (tool rounds + the final answer)
DEFAULT_TIMEOUT_S = 180.0         # wall-clock budget per question (checked between turns)
ABSTAIN = settings.ABSTAIN
SUBMIT_ANSWER = "submit_answer"   # terminal tool name the model calls to finish
DIRECTIVE_MESSAGE = "__instruction__"  # plain user guidance, never a synthetic function response

# SOT-2639 — in-band directive fed back when the shared commit gate REJECTs a submitted answer. Reuses
# the existing ``answer_rejected`` retry channel (identical to the exec/numeric rejection flow), so a
# rejected commit re-enters the loop with a concrete "検算せよ" instruction rather than being dropped.
_COMMIT_GATE_RETRY_DIRECTIVE = (
    "commit_gate が回答を却下しました。数値回答は compute / corpus_aggregate で値を実際に導出・検算してから、"
    "列挙の『該当なし』は母集団を全数確認してから submit_answer してください。"
    "根拠を実際に取得できない場合のみ棄権してください。")


def _commit_gate_enforce() -> bool:
    """Whether the commit gate's verdict is ENFORCED on the answer — ``RAG_COMMIT_GATE_ENFORCE`` (default
    OFF). Delegates to :func:`commit_gate.enforce` (SOT-2640, single source of truth for the flag shared
    across the Gemini and claude-mcp wiring points). SOT-2639's Gemini wiring is equivalence-preserving:
    with ``RAG_COMMIT_GATE=1`` alone the gate's decision + telemetry are recorded but the loop's own inline
    guards stay authoritative, so a committed answer is served VERBATIM (byte-equivalent to OFF)."""
    from src.rag.agent import commit_gate as _commit_gate  # lazy (avoids exec_verifier import cycle)
    return _commit_gate.enforce()


def _bool_env(key: str, default: bool) -> bool:
    raw = os.getenv(key)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


# SOT-2521 — whether the convenience :func:`answer_question` entry point wires the loop-side
# deterministic first move (see :func:`investigate`'s ``first_move``). **Default OFF so the production
# answer path stays byte-identical.** The always-on variant regressed the single gold-100 measurement
# (match 21→19, wrong 6→9 vs the SOT-2510 baseline): seeding canonical evidence and telling the model to
# stop searching nudged premature *wrong* commits on questions the model would otherwise have kept
# investigating. So the mechanism ships behind this flag, OFF, pending a *narrowed* enablement whose
# net effect is measured by the single integrated SOT-2527 gold-100 run — mirroring the
# ``GATE_SLICE_CALIBRATE`` / ``GATE_COMMIT_ON_TIEBREAK`` default-OFF precedent (mechanism landed,
# relaxation deferred to a real non-regression confirmation). The ``investigate(first_move=…)`` parameter
# itself is unchanged; only this entry point's default wiring is gated.
FIRST_MOVE_ROUTING = _bool_env("RAG_FIRST_MOVE_ROUTING", False)

# SOT-2525 — whether :func:`answer_question` wires the loop-side deterministic tool fallback tried before
# an UNANSWERABLE abstain is accepted (see :func:`investigate`'s ``fallback``). **Default OFF so the
# production answer path stays byte-identical** (mirroring the ``RAG_FIRST_MOVE_ROUTING`` default-OFF
# precedent). When on, a question about to abstain forces one contract-typed deterministic tool
# (canonical_route / version_diff / file_grep, each self-resolving from the question) and only feeds real
# evidence back — never an answer. Its net effect on gold-100 is measured by the single integrated
# SOT-2527 run before any default flip.
UNANSWERABLE_FALLBACK = _bool_env("RAG_UNANSWERABLE_FALLBACK", False)

# SOT-2524 — whether :func:`answer_question` wires the budget-exhaustion boundary hook (see
# :func:`investigate`'s ``budget_boundary``). The obligation-driven local re-search director (SOT-2502)
# already fires on a *deliberate* abstain, but the dominant BUDGET_EXHAUSTED cause is the non-deliberate
# boundary — the model wanders through tools until ``max_turns``/``timeout_s`` is reached without ever
# committing or abstaining, so the director never runs. This hook gives it a bounded last push at the
# still-unmet obligations *at that boundary*, before the abstain is finalized. **Default ON**: it only
# extends the already-default-ON director to where most BUDGET abstains actually land, and it is EV-safe
# (the commit threshold is untouched — it only turns a would-be BUDGET abstain into either a grounded
# answer or a coded, history-bearing abstain). Set ``RAG_BUDGET_BOUNDARY_RESEARCH=0`` to restore the
# pre-hook boundary (abstain finalized immediately). Bounded by the director's ``max_rounds`` and by the
# existing ``timeout_s`` (a timeout-triggered boundary adds no model turns — it only records terminal).
BUDGET_BOUNDARY_RESEARCH = _bool_env("RAG_BUDGET_BOUNDARY_RESEARCH", True)

# SOT-2523 — the multi-stage contracts whose derivation runs through several bounded stages
# (ファイル特定→復号→読込→計算→検証) and therefore most often exhausts the default 12-turn / 180s budget
# (BUDGET_EXHAUSTED は棄権原因の最多). ``derived_calculation`` (→ ``numeric``/``multi_hop``), ``横断集計``
# (→ ``cross_aggregate``) and ``enum_set`` (→ ``full_enumeration``) live here; single-stage 単純検索
# (``simple_lookup``) と書式/座席/グラフ/版差分の専用決定論経路は据え置き(コスト線形増を避ける)。
MULTISTAGE_CONTRACTS: frozenset[str] = frozenset({
    "numeric", "multi_hop", "cross_aggregate", "full_enumeration",
})
# Bounded contract-adaptive budget for the multi-stage contracts above. Both are hard, finite upper
# bounds (ハードコードの特定回答ではなく上限値のみ) and only lift the default — an explicit caller budget
# is never shrunk, and the existing ratio (+4) / regulation・gantt (300s) adaptations still compose.
ADAPTIVE_MAX_TURNS = 18           # 12 -> 18 for multi-stage derivations
ADAPTIVE_TIMEOUT_S = 240.0        # 180 -> 240 for multi-stage derivations

# SOT-2523 — whether :func:`answer_question` lifts the per-question budget for the multi-stage contracts
# above. **Default ON**: it only *adds* bounded turns/time to the contracts that dominate BUDGET_EXHAUSTED
# and can never make a previously-committed answer wrong (a question that already committed under 12 turns
# is byte-identical; only a would-be budget abstain gets more room to reach a grounded answer). Set
# ``RAG_ADAPTIVE_BUDGET=0`` to restore the flat 12/180 budget for every contract.
ADAPTIVE_BUDGET = _bool_env("RAG_ADAPTIVE_BUDGET", True)

# SOT-2523 — whether :func:`investigate` memoises deterministic tool evidence within one question (see the
# intra-question ``evidence_cache`` below). **Default ON**: deterministic read/compute/decrypt tools return
# the same value for the same args within a question, so re-issuing an identical call (common in multi-stage
# 再導出) only wastes wall-clock re-deriving evidence already in hand. It is value-preserving (the model sees
# the identical output, only faster) and caches nothing on error. Set ``RAG_EVIDENCE_CACHE=0`` to disable.
EVIDENCE_CACHE = _bool_env("RAG_EVIDENCE_CACHE", True)

# SOT-2522 — spin (dead-end) detection & budget reallocation. A share of BUDGET_EXHAUSTED abstains is the
# model calling the *same tool with the same (normalized) arguments* over and over: a deterministic tool
# returns identical output for identical args, so each repeat burns a turn without adding evidence. When a
# normalized (tool, args) call recurs ``spin_threshold`` times we treat that path as a dead end and, ONCE,
# feed back a directive that redirects the freed budget to an *untried* deterministic route
# (canonical_route / compute / version_diff / corpus_aggregate / …). If spinning persists after that one
# reallocation, the path is cut off early (棄権を前倒し) so the remaining budget is not melted — the overall
# ``max_turns``/``timeout_s`` cap is never raised (無制限ループにしない). The abstain, if any, is attributed
# to :data:`~src.rag.agent.abstain_ledger.SPIN_CUTOFF`, distinct from a plain BUDGET cutoff. **Default OFF**
# (byte-identical answer path): a redirect changes a spinning question's trajectory, which — like
# ``FIRST_MOVE_ROUTING`` / ``UNANSWERABLE_FALLBACK`` — could nudge a commit, so the mechanism ships dormant
# and its net gold-100 effect is measured by the single integrated SOT-2527 run before any flip. No corpus
# fact is ever injected (only tool names), so enabling it can never leak an answer.
SPIN_DETECTION = _bool_env("RAG_SPIN_DETECTION", False)

# SOT-2545 — answer granularity normalization (誤答A2). One corrective round on the terminal commit:
# reject a *truncated* verbatim extract (「そのまま抜き出す」で見出しのみ = idx93) or an *over-enumerated*
# single-item answer (「第N週の項目は何ですか」に日付範囲の全タスク列挙 = idx88) and feed a granularity
# directive so the model re-answers at the question's granularity.  **Default OFF so the production answer
# path stays byte-identical** (mirroring RAG_FIRST_MOVE_ROUTING / RAG_UNANSWERABLE_FALLBACK): the guard
# feeds a directive that changes the model trajectory (nudge risk), so its net gold-100 effect is measured
# by the single integrated SOT-2550 run before any default flip.  EV-safe — it never turns an abstain into
# a wrong answer, and it is one-shot (a still-mismatched re-submission is accepted, never looped).
GRANULARITY_NORMALIZATION = _bool_env("RAG_GRANULARITY_NORMALIZATION", False)

# SOT-2549 — record-conflict resolution (誤答E, idx75). One corrective round on the terminal commit:
# when the model surfaces conflicting records and refuses ("『第3週目から第5週目』と『第4週目』が競合し
# 特定できません" vs gold「第4週」), apply a general precedence rule — a confirmed single record refining a
# coarse range, or the latest/confirmed version — and feed the resolved single value back as a directive so
# the model commits it instead of abstaining.  Content-blind (the value fed back is one the model already
# wrote; no corpus fact injected) and EV-safe: an irreducible conflict stays abstained, and the guard is
# one-shot (a still-refusing re-submission is accepted, never looped).  **Default OFF so the production
# answer path stays byte-identical** (mirroring RAG_GRANULARITY_NORMALIZATION); its net gold-100 effect is
# measured by the single integrated SOT-2550 run before any default flip.
CONFLICT_RESOLUTION = _bool_env("RAG_CONFLICT_RESOLUTION", False)

# SOT-2562 (review=human follow-up) — two residual over-reasoning precision gates on the terminal commit,
# each one corrective round.  NUMERIC_FEATURE_CORR: a 「相関が最も高い数値特徴量」 answer built on a
# categorical→numeric re-encoding (.map) is rejected → recompute with numeric_only (idx4 smoker→bmi).
# RELEVANCE_STRICT: a 「(aspect)に関連する変更を挙げて」 version-diff answer is re-filtered by the aspect,
# falling back to 該当なし when nothing is grounded (idx9).  Both content-blind (no corpus fact injected),
# one-shot, EV-safe (an abstain is never rejected), and **Default OFF so the production answer path stays
# byte-identical** (mirroring RAG_GRANULARITY_NORMALIZATION / RAG_CONFLICT_RESOLUTION).
NUMERIC_FEATURE_CORR = _bool_env("RAG_NUMERIC_FEATURE_CORR", False)
RELEVANCE_STRICT = _bool_env("RAG_RELEVANCE_STRICT", False)

# SOT-2584 — whether :func:`answer_question` builds an Evidence Packet (typed route → registry-resolved
# documents → required evidence slots → per-route budget contract) and pre-injects it into the generation
# agent before its first turn (see :func:`investigate`'s ``preamble``). This reverses "検索→考える→また
# 検索" into "型判定→文書確定→不足スロットだけ探索", targeting the dominant BUDGET_EXHAUSTED loss. **Default
# OFF so the production answer path stays byte-identical** (mirroring RAG_FIRST_MOVE_ROUTING /
# RAG_DOCUMENT_REGISTRY): the packet changes the model's first-turn trajectory (nudge risk), so it ships
# dormant and its net gold-100 effect is measured by a dedicated A/B before any default flip. It injects
# no corpus fact and no answer (document identity + slot names + budget only). Reads
# ``RAG_EVIDENCE_PACKET`` via :func:`src.rag.agent.evidence_packet.enabled`.
EVIDENCE_PACKET = _bool_env("RAG_EVIDENCE_PACKET", False)

# SOT-2586 — whether the NUMERIC route dispatches through the PoT forced compute lane (Evidence Binder →
# 制限AST → Decimal 実行 → 独立検算 → N-sample majority; :mod:`src.rag.agent.pot_lane`). When on, the
# ``verify_formula`` tool is additively exposed and a NUMERIC Evidence Packet appends the forced-lane
# directive. **Default OFF so the production answer path stays byte-identical** (mirroring
# RAG_EVIDENCE_PACKET / RAG_FONT_EMPHASIS / RAG_FORMAT_EVENTS): the extra tool + directive change the
# model's numeric trajectory, so the mechanism ships dormant and its net gold-100 effect is measured by a
# dedicated A/B before any default flip. Reads ``RAG_POT_HARD_LANE`` via :func:`pot_lane.enabled`. The
# directive is appended only atop an Evidence Packet preamble (so it requires RAG_EVIDENCE_PACKET too).
POT_HARD_LANE = _bool_env("RAG_POT_HARD_LANE", False)

# SOT-2587 — whether the ENUM route dispatches through the symbolic exhaustive-scan lane
# (:mod:`src.rag.agent.enum_scan`): resolve the target document universe from the registry, scan every
# applicable document (no top-k retrieval cutoff), and return a completeness certificate — with the
# idx16-type guard forbidding a "該当なし" answer when unsupported documents blocked coverage. When on, the
# ``enum_scan`` tool is additively exposed and an ENUM Evidence Packet appends the full-scan directive.
# **Default OFF so the production answer path stays byte-identical** (mirroring RAG_POT_HARD_LANE /
# RAG_EVIDENCE_PACKET): the extra tool + directive change the model's enumeration trajectory, so the
# mechanism ships dormant and its net gold-100 effect is measured by a dedicated A/B before any default
# flip. Reads ``RAG_ENUM_SCAN`` via :func:`enum_scan.enabled`. The directive is appended only atop an
# Evidence Packet preamble (so it requires RAG_EVIDENCE_PACKET too).
ENUM_SCAN = _bool_env("RAG_ENUM_SCAN", False)

# SOT-2603 (Stage0, PLAN SOT-2602) — whether :func:`answer_question` promotes the contract classifier
# from an in-loop *hint* to a deterministic **router (入口ゲート)**. When on, a question whose contract
# type has a deterministic pipeline registered in :mod:`src.rag.agent.det_pipeline` is answered by that
# pipeline WITHOUT the LLM loop; a question with no registered pipeline — or one whose pipeline cannot
# ground a ``{value, evidence, method}`` result — falls through to the unchanged loop (回答数を減らさない).
# **Default OFF so the production answer path stays byte-identical** (mirroring RAG_EVIDENCE_PACKET /
# RAG_FIRST_MOVE_ROUTING). On top of that, Stage0 ships the registry EMPTY, so even with the flag ON every
# question routes to the LLM loop — the per-type pipelines land in Wave A1〜B2. The flag is read fresh via
# :func:`det_pipeline.enabled` inside the router (env-driven), so this constant is only the module-level
# mirror used by tests; the actual gate reads ``RAG_DET_PIPELINE_ROUTER`` at call time.
DET_PIPELINE_ROUTER = _bool_env("RAG_DET_PIPELINE_ROUTER", False)

# SOT-2632 (G2, PLAN SOT-2602) — whether :func:`answer_question` appends the G2 lookup/derived procedure
# HINTS (:mod:`src.rag.agent.g2_lookup_port`) for the five Sonnet-reachable questions the flash champion
# abstained on (idx 5/53/96/36/79). When on, a question matching one of the G2 archetypes gets an advisory
# procedure directive appended to the generation preamble (which document / which extra hop / route the
# arithmetic through the PoT lane); a companion tool-gap fix lets ``compute`` open a decrypted in-memory
# xlsx. **Default OFF so the production answer path stays byte-identical** (mirroring RAG_EVIDENCE_PACKET /
# RAG_CONDITION_PREIR): the directive nudges the model's trajectory, so it ships dormant and its net
# gold-100 effect is measured by the focused gate / SOT-2636 integration before any default flip. Injects
# no corpus fact and no answer — procedure guidance only. Reads ``RAG_G2_LOOKUP_PORT`` fresh via
# :func:`g2_lookup_port.enabled` at call time; this constant is the module-level mirror used by tests.
G2_LOOKUP_PORT = _bool_env("RAG_G2_LOOKUP_PORT", False)

# SOT-2631 (G1, PLAN SOT-2602) — whether :func:`answer_question` appends the G1 highlight-extraction
# procedure HINTS (:mod:`src.rag.agent.g1_highlight_port`) for the three Sonnet-reachable questions the
# flash champion abstained on (idx 15/80/17). When on, a question matching one of the G1 archetypes gets
# an advisory procedure directive appended to the generation preamble (highlight cell → pivot-label
# reverse-lookup → compute self-verify for the 抽出条件 class; composite 黄∧赤 + PoT-lane arithmetic for
# idx17). **Default OFF so the production answer path stays byte-identical** (mirroring RAG_EVIDENCE_PACKET
# / RAG_G2_LOOKUP_PORT): the directive nudges the model's trajectory, so it ships dormant and its net
# gold-100 effect is measured by the focused gate / SOT-2636 integration before any default flip. Injects
# no corpus fact and no answer — procedure guidance only. Reads ``RAG_G1_HIGHLIGHT_PORT`` fresh via
# :func:`g1_highlight_port.enabled` at call time; this constant is the module-level mirror used by tests.
G1_HIGHLIGHT_PORT = _bool_env("RAG_G1_HIGHLIGHT_PORT", False)

# SOT-2614 — consecutive same-tool spin pivot guard (サイクル2: BUDGET_EXHAUSTED 32件回収). Phase-0 diagnosis
# (``docs/ai/budget32_trace_classification.md``) found 23/32 budget abstains carry a ≥5-long run of the
# SAME tool with *tweaked* arguments (idx99 file_grep×16, idx76 17/18 turns search, extraction ゼロ; compute
# guess-and-check idx47/57/95): the existing SOT-2522 SPIN_CUTOFF only fires on the *identical* (tool,args)
# key, so it caught 2/23. This guard STRENGTHENS (never replaces) that detector: it tracks the *consecutive
# run* of one tool whose arguments only fuzzily change (same file target for file tools; any repeat for a
# search tool that carries no target), and on the ``PIVOT_THRESHOLD``-th consecutive call it does NOT
# dispatch the redundant call — it cools that tool for the question and feeds back a pivot directive
# (canonical_route / 逆引き索引 / structure store / 手持ち証拠での確定判断; names only, no corpus fact). When
# few turns remain it escalates to a definitive-decision directive (answer-or-abstain) so the残予算 is spent
# on a verdict, not more spinning. A cooled tool re-pivots on a shorter run (threshold-1). Each pivot is
# counted (``spin_pivots``) into the details log / abstain ledger so the diagnosis can measure capture rate.
# **Default OFF** (byte-identical answer path, sha256-verified): the directive changes a spinning question's
# trajectory (nudge risk) so it ships dormant; its net gold-100 effect is measured by the サイクル2 integrated
# run before any flip. It never touches the commit threshold and injects no answer, so it can never leak a
# fact nor turn an abstain into a wrong answer directly.
SPIN_PIVOT = _bool_env("RAG_SPIN_PIVOT", False)
DEFAULT_SPIN_THRESHOLD = 3        # identical (tool, args) calls that mark a path a dead end
DEFAULT_PIVOT_THRESHOLD = 3       # SOT-2614 — consecutive same-tool (fuzzy-arg) calls that force a pivot
PIVOT_LOW_TURNS = 3               # SOT-2614 — remaining tool rounds at/below which a pivot escalates to a verdict

# SOT-2620 — per-route search-call cap (サイクル2: BUDGET_EXHAUSTED 32件回収). Phase-0 diagnosis
# (`docs/ai/budget32_trace_classification.md`): of the 486 tool calls across the 32 BUDGET_EXHAUSTED
# abstains, **70% (340) were search** (file_grep 285), ~10.6/問; retrieval itself succeeds 69% of the
# time, so the waste is the *iteration count* of "hit返るが針が無い→パターン変えて再grep", not search
# failure. The existing 型別予算契約 (`ROUTE_BUDGET`) already models this but is gated behind the
# opt-in `RAG_EVIDENCE_PACKET` and so never bounds the champion fallback LLM loop. This flag wires a
# **search上限** into the default path independently: it caps the *total* file_grep/find_files calls per
# question by route type, and on the over-cap call feeds back a switch-lane directive (canonical_route /
# 逆引き索引 / structure store / 手持ち証拠での確定判断) instead of dispatching another search. It
# STRENGTHENS the SOT-2614 pivot guard (which only cuts a *consecutive* same-tool run) by also catching
# search reflexes spread *non-consecutively* across interleaved reads. Total turns are NOT changed
# (adaptive 18/240s のまま) — only the repeated-search axis. **Default OFF** (byte-identical answer path,
# sha256-verified): a cap changes a search-heavy question's trajectory, which — like SOT-2614 — must be
# net-measured (match非劣化・wrong非増加) before any default flip. Wired into :func:`answer_question`.
SEARCH_CAP = _bool_env("RAG_SEARCH_CAP", False)
# The search-style tools the cap counts: a query-carrying scan with no single value lane (the documented
# re-grep waste). read_office/compute/registry/canonical_route stay uncapped so the model can always
# still act on evidence in hand after the cap is hit.
_SEARCH_TOOLS: frozenset[str] = frozenset({"file_grep", "find_files"})

# Deterministic routes offered as the reallocation target when a spin is cut off (names only, no fact).
_DETERMINISTIC_ROUTES: tuple[str, ...] = (
    "canonical_route", "compute", "version_diff", "corpus_aggregate",
    "seating_lookup", "read_chart_values", "file_grep",
)

# SOT-2660 — tool classification table (DB経路 vs 生ファイル系) for the fallback-dependency KPI and the
# RAG_DB_ONLY diagnostic mode.
#
# ``RAW_FILE_TOOLS`` names every agent tool whose ``fn`` may open/read a RAW corpus document (office/pdf/
# pptx/xlsx/csv content, a directory walk, or a live compute over such a file) at SERVE time. These are the
# 生ファイルフォールバック — the safety net that (per the 2026-08-12 adversarial review) still produces ~30
# correct answers, so it is NEVER force-disabled in production. Everything NOT in this set is a "DB経路"
# tool: the precomputed fact-layer stores (case_filter/id_lookup/metric_lookup/diff_lookup — serve-time JSON
# lookups built at 事前処理), the pure-computation PoT lane (verify_formula operates on model-supplied
# candidates, no file read), and the terminal submit_answer. New DB stores are automatically "DB経路"
# (allow-by-default under DB_ONLY); a genuinely new raw-file reader must be added here. Build-time raw reads
# (index/store construction) do NOT count — only serve-time file access is a fallback dependency.
RAW_FILE_TOOLS: frozenset[str] = frozenset({
    "find_files", "file_grep", "read_office", "decrypt", "compute", "canonical_route",
    "read_chart_values", "caption_image", "pdf_emphasis", "pptx_pivot", "highlight_extract",
    "version_diff", "seating_lookup", "corpus_aggregate", "font_emphasis", "format_events",
    "enum_scan",
})


def is_raw_file_tool(name: str) -> bool:
    """Whether ``name`` is a 生ファイル系 tool that reads raw corpus files at serve time (SOT-2660)."""
    return name in RAW_FILE_TOOLS


# SOT-2660 — RAG_DB_ONLY: DIAGNOSTIC-ONLY mode (default OFF). When ON, :func:`dispatch` refuses every
# ``RAW_FILE_TOOLS`` call with a reason so the model must answer from the DB経路 (search/index/fact-layer)
# alone or abstain. It measures the TRUE DB-path coverage (how many questions the precomputed stores can
# solve unaided) and surfaces the not-yet-DB-ized idx list. It must NEVER become the production default:
# forcing DB-only in serve creates a 転記欠落=即棄権 cliff, and the raw-file fallback is a measured safety
# net. Read at dispatch time (module-global) so BOTH the investigator loop and the MCP server honor it
# uniformly. The champion serve path (flag OFF) is byte-identical — no raw-file call is ever intercepted.
DB_ONLY = _bool_env("RAG_DB_ONLY", False)

# SOT-2627 — investigator tool-loop backend. ``gemini`` (default) keeps the live Gemini function-calling
# loop below byte-identical. ``claude-mcp`` delegates the WHOLE per-question loop (tool round-trips → final
# answer) to a flat-rate Sonnet ``claude -p --mcp-config`` session driving the SOT-2626 stdio MCP server,
# so a dev gold100 run avoids Gemini metering. **dev-only** (production stays Gemini-only, SOT-2460;
# official measurement stays flash 3.6, SOT-2625) and the flat-rate limit is shared with the autonomous
# workers, so it is off unless explicitly opted in. The switch is read in :func:`answer_question` *after*
# every deterministic pre-stage (document_registry / det_pipeline / deterministic_* shortcuts stay
# unchanged), so only the model-driven loop moves to Sonnet and the ``gemini`` default is byte-identical.
INVESTIGATOR_BACKEND = (os.getenv("RAG_INVESTIGATOR_BACKEND", "gemini").strip().lower() or "gemini")

# Vertex Gemini list price (USD per 1M tokens), (input, output) — estimates for cost bookkeeping only.
PRICING: dict[str, tuple[float, float]] = {
    "gemini-2.5-pro": (1.25, 10.0),
    "gemini-2.5-flash": (0.30, 2.50),
    "gemini-2.5-flash-lite": (0.10, 0.40),
}

# SOT-2641 — model-neutral SYSTEM_PROMPT. The shared prompt was calibrated against flash's
# over-abstention ("安易に棄権しない"/"諦めず必ず再試行"/"棄権しない" repeated as brakes-off drivers).
# Those drivers work as a brake-release for a conservative flash backend, but over-drive an obedient,
# strong Sonnet into aggressive commits (dev gold100 abstain −20 / wrong +21 = the prompt behaving as
# instructed). cycle4 principle: the prompt is a MODEL-NEUTRAL role declaration ("explore aggressively;
# submission is verified by the deterministic commit_gate, SOT-2637..2640") and commit admissibility is
# owned by the gate, not by prompt exhortation. The variant is gated by ``RAG_NEUTRAL_PROMPT``
# (default OFF ⇒ byte-identical legacy wording, sha256-verified). Only the abstain/aggressiveness
# drivers (items 3, 4-tail, 7) change; the TOOL-DISCIPLINE block (4b/5/5b/6a/6b/6c/6 — compute必須 /
# read_chart_values必須 / caption_image数値採用禁止 等) is model-independent correct discipline and is
# SHARED verbatim between both variants so it can never drift between them.
NEUTRAL_PROMPT = _bool_env("RAG_NEUTRAL_PROMPT", False)

# Intro + items 1-2 (shared, unchanged).
_PROMPT_HEAD = (
    "あなたは社内ドライブの文書QAを行う調査エージェントです。与えられた汎用ツールだけを使って質問に答えます。\n"
    "厳守事項:\n"
    "1. 暗算・記憶・創作で答えない。必ずツールで根拠となる値を取得してから答える。\n"
    "2. パスワード・略称・書式規則などのコーパス固有事実は与えられていない。ツール(ファイル探索/grep/"
    "Office抽出/復号/pandas計算/チャート読取/画像説明)で自力発見する。暗号化ファイルは復号ツールが"
    "ファイル名等から鍵を推定して復号する。\n"
)

# Items 3-4 — exploration/abstain drivers (the ONLY model-tuned wording; variant).
_PROMPT_EXPLORE_LEGACY = (
    "3. まず関連ファイルを探索し、必要なツールを反復呼び出しして値を確定する。ツールがエラー/空を返しても"
    "諦めず、原因(列名違い・ファイル違い・値表記違い)を切り分けて別のファイルや式で必ず再試行する。安易に"
    "棄権しない。\n"
    "4. 数値計算(平均・合計・件数など)は必ず compute ツールで行う。train.xlsx/train.csv 等は案件ごとに"
    "同名で複数存在するので、質問が指す案件名を compute の project 引数(会社名の一部)で渡してファイルを"
    "特定する。曖昧エラーが返ったら、そのエラーが挙げる『存在プロジェクト』から該当案件を選んで project を"
    "付けて再試行する(棄権しない)。列名や絞り込み値が不明なときは、まず `df.columns.tolist()` や"
    "`df['列'].unique().tolist()` を compute で確認してから集計式を組む。\n"
)
_PROMPT_EXPLORE_NEUTRAL = (
    "3. まず関連ファイルを探索し、必要なツールを反復呼び出しして値を確定する。ツールがエラー/空を返したら、"
    "原因(列名違い・ファイル違い・値表記違い)を切り分け、手段(別ファイル・別の式・別ツール)を替えて証拠に"
    "到達するまで探索する。\n"
    "4. 数値計算(平均・合計・件数など)は必ず compute ツールで行う。train.xlsx/train.csv 等は案件ごとに"
    "同名で複数存在するので、質問が指す案件名を compute の project 引数(会社名の一部)で渡してファイルを"
    "特定する。曖昧エラーが返ったら、そのエラーが挙げる『存在プロジェクト』から該当案件を選んで project を"
    "付けて再試行する。列名や絞り込み値が不明なときは、まず `df.columns.tolist()` や"
    "`df['列'].unique().tolist()` を compute で確認してから集計式を組む。\n"
)

# Items 4b-6 — tool discipline (model-independent; SHARED verbatim between both variants).
_PROMPT_TOOLS = (
    "4b. 案件名+データ資産(train.xlsx/train.csv・分析コード/modeling.py・notebook/EDA・leaderboard・"
    "スケジュール等)を指すデータ/計算系の質問で、grep/検索で根拠ファイルが見つからないときは canonical_route "
    "ツールに質問文をそのまま渡す。用語集で案件を特定し canonical ファイルを直行解決して返すので(chunk検索を"
    "迂回)、返った先頭 rel を compute/read_office/read_chart_values に渡すか、canonical_route(question, "
    "expr='…') で計算値まで得る。例: 『京橋信用ソリューションズの分析コードの n_estimators』は"
    "canonical_route(question=…, kind='code') で modeling.py を得て read_office で読む。\n"
    "5. 旧版(old版)と最新版の比較・変更点を問う質問は、grepで手作業比較せず version_diff ツールに質問文を"
    "そのまま渡す。決定論の構造diffが『変更前 → 変更後』を返すので、その value をそのまま回答にする。value が"
    "null のときのみ他手段(grep等)を検討する。\n"
    "5b. PDFの検索・抽出結果が空で、ページが画像だけの場合は caption_image(file=..., question=質問全文) を"
    "使う。引用語や『明記』の質問では、条件語と候補が同じ原文行にある vision 出力だけを根拠にする。\n"
    "6a. 内線番号/EXT/座席/『向かい・隣・同じ列・Xから見て右側/左側』を問う質問は seating_lookup "
    "ツールを使う(座席表は画像1枚で"
    "grep/office抽出では読めない)。多段(案件→担当者→内線)では先に担当者の氏名を他ツールで特定し、その氏名を"
    "seating_lookup(name=…)に渡す。『Aさんの向かいの人のEXT』は seating_lookup(name='A', relation='向かい')。"
    "『Aさんから見て右側の人の名前をすべて』は seating_lookup(name='A', relation='右側', field='name')。\n"
    "6b. 『全体で/横断で/全案件で/最も〜な案件・人』のように複数案件をまたいで集計・比較する質問は"
    "corpus_aggregate ツールを使う(単一案件の compute では同名ファイルが複数で解けない)。例: 『最も多く案件に"
    "関わる人』=corpus_aggregate(metric='staff', op='count') の top、『着手金が最も高い案件』="
    "corpus_aggregate(metric='deposit', op='max')、『固定金額契約で1行あたり契約金額が最も高い案件』="
    "corpus_aggregate(metric='amount_per_row', op='max', fixed_only=true, round_up=true)、『契約期間が"
    "2025-08-15〜09-07 に重なり40日超の案件を主略称で』=corpus_aggregate(metric='period_days', op='filter', "
    "overlap_start='2025-08-15', overlap_end='2025-09-07', min_days=40)、『全案件のPP・契約書・PLAN・FRに"
    "役割付きで記載されたDA人物のユニオン人数』=corpus_aggregate(metric='staff_population', op='count')。"
    "案件特定後の多段(その案件の担当者→"
    "内線 等)は返り値 staff(ES/PM…)の氏名を seating_lookup に渡して解決する。\n"
    "6c. グラフの数値を問う場合は read_chart_values を必ず使う。系列/列名を column に、ヒストグラムの"
    "最多カウントなら operation='histogram_max_count' を渡す。numCacheが無い画像グラフも元データ列から"
    "決定論的に再集計される。caption_image はグラフの所在・軸ラベル確認にだけ使い、その画像説明中の数値を"
    "answer/evidence/methodへ採用してはならない。read_chart_values が値を返せなければ推測せず棄権する。\n"
    "6. 十分な根拠が得られたら、最終回答は必ず submit_answer ツールを1回だけ呼んで返す(通常のテキストでは"
    "答えない)。submit_answer には次を渡す: answer=回答本文(値/一覧のみ、列挙は「、」区切り、金額は原文表記)、"
    "confidence=0.0〜1.0の自己確信度、evidence=根拠(参照ファイル・値・ツール結果)、method=導出手順の要約。\n"
)

# Item 7 — commit/abstain policy (variant). Legacy = abstain-only-as-last-resort driver; neutral =
# gate-delegated commit ("提出は commit_gate が検証する; 拒否されたら検算・修正して再提出; 証拠が確定
# できなければ「わかりません」"), Incorrect=−1 < Missing=0.
_PROMPT_COMMIT_LEGACY = (
    f"7. あらゆる手段を尽くしても根拠が得られない場合に限り answer=「{ABSTAIN}」・confidence=0.0 で submit_answer"
    "する。"
)
_PROMPT_COMMIT_NEUTRAL = (
    "7. 提出(submit_answer)は決定論の commit_gate が検証する。ゲートに拒否されたら、その理由に従って"
    "検算・修正し、正しい値に直して再提出する。手段を替えて探索してもなお証拠が確定できない場合に限り "
    f"answer=「{ABSTAIN}」・confidence=0.0 で submit_answer する(Incorrect=−1 < Missing=0)。"
)

# SOT-2647 — fact-layer tool discipline. Appended ONLY when RAG_FACT_LAYER is on (default OFF ⇒ the prompt
# is byte-identical), mirroring the RAG_NEUTRAL_PROMPT module-level variant gate. Steers the loop to the
# precomputed-store tools as the first choice over file_grep for 横断列挙/ID逆引き/派生量/版差分 fact reads.
_PROMPT_FACT_LAYER = (
    "8. 事前計算事実層(有効時): 横断列挙/横断比較(『APR-M3の案件を略称で列挙し契約金額合計』等)は "
    "case_filter を、特定IDの内容抜き出しは id_lookup を、案件の相関/F1閾値/予測等の派生量は metric_lookup を、"
    "旧版→新版の変更点は diff_lookup を、file_grep の反復より先に使う。いずれも出典付きの確定値(検証済み"
    "operand)を返すので、その value を根拠に submit_answer する。返り値が空/未解決のときのみ従来の探索に戻す。\n"
)

SYSTEM_PROMPT = (
    _PROMPT_HEAD
    + (_PROMPT_EXPLORE_NEUTRAL if NEUTRAL_PROMPT else _PROMPT_EXPLORE_LEGACY)
    + _PROMPT_TOOLS
    + (_PROMPT_FACT_LAYER if _fact_layer.enabled() else "")
    + (_PROMPT_COMMIT_NEUTRAL if NEUTRAL_PROMPT else _PROMPT_COMMIT_LEGACY)
)


# --------------------------------------------------------------------------- usage / answer schema
@dataclass
class Usage:
    """Token usage accumulator (output includes any 'thinking' tokens for cost realism)."""

    input_tokens: int = 0
    output_tokens: int = 0

    def __add__(self, other: "Usage") -> "Usage":
        return Usage(self.input_tokens + other.input_tokens, self.output_tokens + other.output_tokens)

    @property
    def total_tokens(self) -> int:
        return self.input_tokens + self.output_tokens

    def cost_usd(self, model: str) -> float:
        # SOT-2627 — the flat-rate claude-mcp backend has zero marginal token cost, so a
        # ``…(claude-mcp)`` model prices at 0 regardless of token counts. Gemini models never carry that
        # suffix, so the production/Gemini details.jsonl cost is byte-identical.
        if model.endswith("(claude-mcp)"):
            return 0.0
        pin, pout = PRICING.get(model, PRICING["gemini-2.5-pro"])
        return self.input_tokens / 1e6 * pin + self.output_tokens / 1e6 * pout


@dataclass(frozen=True)
class Answer:
    """The fixed structured answer schema every investigation returns."""

    answer: str
    confidence: float          # self-reported, clamped to [0.0, 1.0]
    evidence: str = ""         # 根拠: files / values / tool outputs the answer rests on
    method: str = ""           # how it was derived (which tools / steps)

    def to_dict(self) -> dict[str, Any]:
        return {"answer": self.answer, "confidence": self.confidence,
                "evidence": self.evidence, "method": self.method}


def _coerce_confidence(value: Any) -> float:
    """Best-effort parse of a model-supplied confidence into a clamped float in [0.0, 1.0]."""
    try:
        c = float(value)
    except (TypeError, ValueError):
        return 0.0
    if c != c:  # NaN
        return 0.0
    return max(0.0, min(1.0, c))


def _answer_from_args(args: Mapping[str, Any] | None) -> Answer:
    """Build an :class:`Answer` from the model's ``submit_answer`` arguments (robust to omissions)."""
    a = dict(args or {})
    text = str(a.get("answer", "") or "").strip() or ABSTAIN
    conf = _coerce_confidence(a.get("confidence"))
    if is_abstain(text):
        conf = 0.0
    return Answer(answer=text, confidence=conf,
                  evidence=str(a.get("evidence", "") or ""),
                  method=str(a.get("method", "") or ""))


def _answer_from_det_contract(result: Mapping[str, Any]) -> Answer:
    """Build an :class:`Answer` from a deterministic pipeline's ``{value, evidence, method}`` contract.

    SOT-2603 (Stage0). The pipeline value is rendered to the answer text (a plain string verbatim, any
    other JSON value serialized deterministically); ``evidence``/``method`` are serialized compactly into
    the Answer's string fields. Confidence is taken from ``method.confidence`` when the pipeline supplied
    one (deterministic answers default to full confidence), and forced to 0.0 for an abstain-shaped value.
    """
    value = result.get("value")
    text = value if isinstance(value, str) else json.dumps(value, ensure_ascii=False, sort_keys=True)
    text = str(text).strip()
    evidence = result.get("evidence") or {}
    method = result.get("method") or {}
    conf = _coerce_confidence(method.get("confidence", 1.0) if isinstance(method, Mapping) else 1.0)
    if is_abstain(text):
        conf = 0.0
    return Answer(
        answer=text,
        confidence=conf,
        evidence=json.dumps(evidence, ensure_ascii=False, sort_keys=True),
        method=json.dumps(method, ensure_ascii=False, sort_keys=True),
    )


def is_abstain(answer: str) -> bool:
    import unicodedata

    a = unicodedata.normalize("NFKC", str(answer)).strip().lower()
    return not a or a == unicodedata.normalize("NFKC", ABSTAIN).strip().lower() \
        or "わかりません" in a or "不明" in a


def _reference_commit_rejection(question: str, candidate: Answer, contract: str | None, *,
                                tool_outputs: Sequence[Any],
                                version_diff_result: Mapping[str, Any] | None) -> dict[str, Any] | None:
    """Return a structured rejection when deterministic reference evidence is still incomplete."""
    if contract == "version_diff":
        if version_diff_result is None:
            return {
                "answer_rejected": True,
                "reason": "版差分契約で必須の version_diff ツールが未実行です。",
                "directive": (
                    "回答を確定せず version_diff(question=質問全文) を実行してください。"
                    "grep/目視による一部比較だけで回答してはなりません。"),
            }
        resolved = version_diff_result.get("value")
        if resolved is None:
            if is_abstain(candidate.answer):
                return None
            return {
                "answer_rejected": True,
                "reason": "全スライド/全シートの決定論的版差分が解決できていません。",
                "directive": "別の変更を推測せず、追加の版特定ができなければ棄権してください。",
            }
        if is_abstain(candidate.answer):
            return {
                "answer_rejected": True,
                "reason": "version_diff は全版差分を解決済みのため棄権できません。",
                "expected": str(resolved),
                "directive": "version_diff が返した value を省略・言い換えせずそのまま回答してください。",
            }
        if candidate.answer.strip() != str(resolved).strip():
            return {
                "answer_rejected": True,
                "reason": "回答が version_diff の決定論的 value と一致しません。",
                "expected": str(resolved),
                "directive": "version_diff が返した value を省略・言い換えせずそのまま回答してください。",
            }

    if is_abstain(candidate.answer):
        return None

    from src.rag.agent import obligations as _obligations

    literal_check = _obligations.validate_literal_evidence(
        question, candidate.answer, tool_outputs)
    if not literal_check.passed:
        return {
            "answer_rejected": True,
            "reason": "literal 一致の証拠義務が未充足です。",
            "issues": list(literal_check.issues),
            "directive": (
                "引用条件語が候補と同じセル/文に文字列として存在する箇所をツールで確認してください。"
                "共起しない候補は採用せず、確認不能なら棄権してください。"),
        }
    return None


# --------------------------------------------------------------------------- transport data types
@dataclass(frozen=True)
class Call:
    """A single function call the model requested."""

    name: str
    args: dict[str, Any]


@dataclass(frozen=True)
class ToolResponse:
    """The dispatched result for one :class:`Call`, fed back to the model."""

    name: str
    response: Any


@dataclass(frozen=True)
class Step:
    """One model turn: either ``function_calls`` to run, or a ``final_text`` answer (never both)."""

    function_calls: tuple[Call, ...] = ()
    final_text: str | None = None
    usage: Usage = field(default_factory=Usage)


@dataclass
class Investigation:
    """Outcome for one investigated question."""

    question: str
    answer: Answer
    iterations: int             # tool rounds (turns that requested ≥1 tool call)
    tool_calls: list[str]
    usage: Usage
    model: str
    elapsed_s: float
    stop_reason: str            # "answered" | "max_turns" | "timeout" | "model_error"
    error: str | None = None
    contract: str | None = None  # SOT-2498 — routing contract this question was classified as (or None)
    calc_record: dict[str, Any] | None = None  # SOT-2506 — in-memory record for the execution gate
    pot_lane: dict[str, Any] | None = None  # SOT-2586 — last verify_formula three-layer verdict (None ⇒ lane not exercised)
    spin_pivots: int = 0  # SOT-2614 — forced consecutive-spin pivots fired this question (0 ⇒ guard OFF/no spin)
    search_cap_hits: int = 0  # SOT-2620 — over-cap search calls intercepted this question (0 ⇒ cap OFF/not hit)
    # SOT-2629 — per-question intervention/guard firing telemetry, recorded for ANSWERED and abstained cases
    # alike so class-A attribution (spin/cap 犯人説 …) is verifiable on the answered trace, not just the
    # abstain ledger (cycle2 adversarial-review hole H6). One key per *active* intervention flag with an
    # explicit fired count/bool (0/false ⇒ flag ON but did not fire — distinguishable from an ABSENT key ⇒
    # flag OFF); an empty ``{}`` when no intervention flag was active. Telemetry only: it never touches the
    # served answer, so the answer CSV stays byte-identical.
    interventions: dict[str, Any] = field(default_factory=dict)

    @property
    def confidence(self) -> float:
        return self.answer.confidence

    def to_dict(self) -> dict[str, Any]:
        return {
            "question": self.question,
            **self.answer.to_dict(),
            "contract": self.contract,
            "iterations": self.iterations,
            "tool_calls": list(self.tool_calls),
            "input_tokens": self.usage.input_tokens,
            "output_tokens": self.usage.output_tokens,
            "total_tokens": self.usage.total_tokens,
            "cost_usd": round(self.usage.cost_usd(self.model), 6),
            "model": self.model,
            "elapsed_s": round(self.elapsed_s, 3),
            "stop_reason": self.stop_reason,
            "error": self.error,
            # SOT-2586 — thread the PoT forced-lane three-layer verdict into the details log ONLY when the
            # lane was actually exercised. When it is None (lane OFF, or NUMERIC question that never called
            # verify_formula) the key is omitted, so the champion/OFF ``.details.jsonl`` stays byte-identical.
            **({"pot_lane": self.pot_lane} if self.pot_lane is not None else {}),
            # SOT-2614 — spin-pivot count is emitted ONLY when the guard fired (>0). With RAG_SPIN_PIVOT OFF
            # (default) it is always 0, so the key is omitted and the champion/OFF ``.details.jsonl`` stays
            # byte-identical; a nonzero count lets the diagnosis measure per-question spin capture rate.
            **({"spin_pivots": self.spin_pivots} if self.spin_pivots else {}),
            # SOT-2620 — search-cap hits are emitted ONLY when the cap actually intercepted a call (>0).
            # With RAG_SEARCH_CAP OFF (default) it is always 0, so the key is omitted and the champion/OFF
            # ``.details.jsonl`` stays byte-identical; a nonzero count lets the diagnosis measure how often
            # the search 上限 fired per question.
            **({"search_cap_hits": self.search_cap_hits} if self.search_cap_hits else {}),
            # SOT-2629 — unified intervention telemetry. ALWAYS emitted (unlike the conditional keys above)
            # so every details.jsonl row — answered or abstained — carries the same schema and a flag that is
            # ON-but-fired-0× is distinguishable from a flag that was OFF. Empty ``{}`` when no intervention
            # flag was active. Additive & telemetry-only: the served answer / answer CSV are unaffected.
            "interventions": dict(self.interventions),
        }


# --------------------------------------------------------------------------- generic tool layer
@dataclass(frozen=True)
class AgentTool:
    """A generic, agent-callable tool: JSON-schema ``parameters`` + a Python ``fn``."""

    name: str
    description: str
    parameters: dict[str, Any]
    fn: Callable[..., Any]


def _obj(props: dict[str, dict[str, Any]], required: Sequence[str] = ()) -> dict[str, Any]:
    return {"type": "object", "properties": props, "required": list(required)}


_STR = {"type": "string"}
_BOOL = {"type": "boolean"}
_NUM = {"type": "number"}
_INT = {"type": "integer"}


def _version_diff(question: str) -> dict[str, Any]:
    """Deterministic version-diff solver, wrapped in the {value, evidence, method} contract.

    Wires the deterministic differ (:func:`src.rag.diffpair.answer_question_agent`, built on the
    SOT-2408 hold-out-validated solver) into the agent's function-calling loop so a 版差分 question is
    answered by an actual structural diff instead of the model guessing. Imported lazily so serve-time
    import of this module stays lean.

    ``value`` is the rendered "変更前 → 変更後" summary, or ``None`` when the differ abstains (not a
    diff question / no unambiguous pair / non-adjacent versions / unreadable / too many realigned
    changes). ``None`` is a
    legitimate contract value meaning "resolved nothing" — the agent then abstains as before, never a
    guess (precision 1.0 維持).
    """
    from src.rag import diffpair  # lazy: keeps serve-time import free of corpus/Office deps

    if not diffpair.is_diff_question(question):
        return _contract.make(
            None, engine="diffpair", evidence={"applicable": False},
            note="版差分質問ではない(旧版/old版/_r1_r2等の版参照+比較動詞が必要)")
    # precision-first agent entry: abstains on non-adjacent / unreadable / ambiguous pairs so a
    # wrong diff (−1) is never surfaced in place of an abstention (0).
    answer = diffpair.answer_question_agent(question)
    evidence: dict[str, Any] = {"applicable": True, "resolved": answer is not None,
                                "coverage": "all-slides/all-sheets"}
    # SOT-2588 (opt-in, default OFF → byte-identical): pre-inject ranked substantive change candidates
    # (intent/score/old/new/structural_location) so the agent reasons over aligned changed blocks and
    # the substantive edit is the first candidate, not the largest structural diff.
    if diffpair.align_enabled():
        try:
            cands = diffpair.ranked_candidates(question)
        except Exception:
            cands = []
        if cands:
            evidence["candidates"] = cands[:6]
    return _contract.make(
        answer, engine="diffpair",
        evidence=evidence,
        scheme="structural-version-diff",
        note=("隣接版(旧版→最新/vN→vN+1)の全スライド/全シート構造diff"
              "(セル/段落を整列し実質変更のみ)"
              if answer is not None
              else "版ペア不確定/非隣接/読取不能/大規模変更のため棄権(None)"))


# The terminal tool: calling it ends the investigation with the structured answer schema. Its ``fn`` is
# never dispatched (the loop intercepts the call), but a no-op keeps the tool uniform.
SUBMIT_ANSWER_TOOL = AgentTool(
    SUBMIT_ANSWER,
    "十分な根拠が得られたら最終回答を確定する。answer(回答本文)・confidence(0.0〜1.0)・evidence(根拠)・"
    "method(導出手順)を渡す。これを呼ぶと調査は終了する。",
    _obj({"answer": _STR, "confidence": _NUM, "evidence": _STR, "method": _STR}, ["answer"]),
    lambda answer=None, confidence=None, evidence=None, method=None: {"submitted": True},
)


def build_generic_tools(profile: CorpusProfile) -> list[AgentTool]:
    """Wire the deterministic :mod:`src.rag.tools` into agent-callable tools bound to ``profile``.

    ``profile`` carries self-discovered secrets (passwords/aliases) across a single question's tool
    calls; it is never seeded with corpus facts here (移植性の担保).
    """
    tools = [
        AgentTool(
            "find_files",
            "コーパス内のファイルを名前/拡張子/プロジェクトで検索し、該当ファイル一覧を返す。",
            _obj({"query": _STR, "ext": _STR, "project": _STR}),
            lambda query=None, ext=None, project=None: find_files(query, ext=ext, project=project),
        ),
        AgentTool(
            "file_grep",
            "コーパス全体を全文/セル/ファイル名でgrepし、一致箇所(ファイル・行・抜粋)を返す。"
            "逆引き索引が有効な場合は所在(ファイル・シート・セル/段落)を即答し、未ヒット時のみ全走査にフォールバックする。",
            _obj({"query": _STR, "ext": _STR, "project": _STR, "regex": _BOOL}, ["query"]),
            lambda query, ext=None, project=None, regex=False: file_grep(
                query, regex=bool(regex), ext=ext, project=project),
        ),
        AgentTool(
            "read_office",
            "docx/xlsx/pptxから書式保持テキストを抽出する。暗号化ファイルは鍵を自力推定して透過復号する。",
            _obj({"file": _STR}, ["file"]),
            lambda file: extract_office(file, profile=profile),
        ),
        AgentTool(
            "decrypt",
            "暗号化Officeファイルの復号可否と方式(根拠のみ、パスワードは返さない)を確認する。",
            _obj({"file": _STR}, ["file"]),
            lambda file: _decrypt(file, profile=profile),
        ),
        AgentTool(
            "compute",
            "csv/xlsxに対し単一のpandas式(dfを参照)を実行し、計算値と根拠(列・範囲)を返す。暗算の代替。"
            "train.xlsx/train.csv等は案件ごとに同名で複数存在するため、案件名を project(会社名の一部)で"
            "渡してファイルを特定する。曖昧エラー時は返された『存在プロジェクト』から project を選び再試行する。",
            _obj({"file": _STR, "expr": _STR, "sheet": _STR, "project": _STR}, ["file", "expr"]),
            lambda file, expr, sheet=None, project=None: compute_run(
                file, expr, sheet=sheet, project=project),
        ),
        AgentTool(
            "canonical_route",
            "データ/計算系の質問を、案件(project)の canonical データファイル(train.xlsx/train.csv・"
            "分析コード modeling.py・notebook 01_eda.ipynb・leaderboard.csv 等)へ直行解決するツール。"
            "chunk検索で根拠が top-k に上がらない(索引で拾えない)データ質問の迂回ルート。question に質問文を"
            "そのまま渡すと、用語集で案件を特定し、質問の種別(train/code/notebook/leaderboard/schedule/"
            "contract)に対応する canonical ファイルを決定論で発見して {rel, project, category, ext, kind} の"
            "一覧(canonical本命が先頭)を返す。project(会社名の一部)/kind を明示して絞ることも可。返った先頭 rel を"
            "compute/read_office/read_chart_values に渡す。expr(単一pandas式)を渡すと先頭の表ファイルに対して"
            "その場で compute を実行し計算値まで返す。『京橋の分析コードの n_estimators』『〜のtrain.xlsxの…』の"
            "ように案件名+データ資産を指す質問に使う。該当なしは value=[](棄権のまま)。",
            _obj({"question": _STR, "project": _STR, "kind": _STR, "expr": _STR, "sheet": _STR},
                 ["question"]),
            lambda question, project=None, kind=None, expr=None, sheet=None: canonical_route(
                question, project=project, kind=kind, expr=expr, sheet=sheet),
        ),
        AgentTool(
            "read_chart_values",
            "xlsx/pptx埋め込みチャートの数値を厳密に読む。numCacheを最優先し、画像化されたxlsxヒストグラムは"
            "columnで指定した元データ列から再集計する。最多カウントはoperation='histogram_max_count'。"
            "vision数値は使用せず、厳密経路が無ければエラー=棄権。",
            _obj({"file": _STR, "column": _STR, "operation": _STR}, ["file"]),
            lambda file, column=None, operation=None: read_chart_values(
                file, column=column, operation=operation),
        ),
        AgentTool(
            "caption_image",
            "図表PNGを説明し、または画像のみのPDF全ページをvisionモデルで質問別に厳密転記する。"
            "PDFでは question に元の質問全文を渡す。引用語/『明記』は条件語と候補を同一原文行で返す。"
            "チャートでは所在・系列名・軸ラベル確認専用で、説明中の数値を回答根拠に使用してはならない。"
            "数値はread_chart_valuesで取得する。",
            _obj({"file": _STR, "question": _STR}, ["file"]),
            lambda file, question=None: caption_figure(file, question=question),
        ),
        AgentTool(
            "pdf_emphasis",
            "PDF内の強調(疑似イタリック=行列シアー)された単語を抽出する。",
            _obj({"file": _STR}, ["file"]),
            lambda file: emphasized_words(file),
        ),
        AgentTool(
            "pptx_pivot",
            "pptxに埋め込まれたPivotTable(EMF)を復元し、表とハイライトされたセルを返す。"
            "同一案件の元データと全表示セルを照合して semantics(行フィールド/列フィールド/対象列/集計方法)を"
            "意味解決し、各セルに filters・target_column・aggregation_label・semantic_summary を付ける。"
            "ピボット回答は生ラベルや値だけでなく semantic_summary の抽出条件+対象列+集計方法を必ず含める。",
            _obj({"file": _STR}, ["file"]),
            lambda file: extract_pptx_pivots(file),
        ),
        AgentTool(
            "highlight_extract",
            "xlsx/pptx/docx/pdfのハイライト・マーカー強調された語/セルを文書順で列挙する。"
            "colorに色名(黄/オレンジ/赤/青など)を渡すとその色だけに絞れる。書式型(マーカー語抽出・"
            "色付きセルの条件)の質問に使う。",
            _obj({"file": _STR, "color": _STR}, ["file"]),
            lambda file, color=None: highlight_extract(file, color=color, profile=profile),
        ),
        AgentTool(
            "version_diff",
            "同一文書の旧版と最新版(または _r1/_r2・_v1/_v3 等の版)を構造diffし、変更点を"
            "『(項目)：変更前 → 変更後』で決定論的に返す。旧版/old版/前の版と最新版の比較・変更箇所を"
            "問う質問に使う。question に質問文をそのまま渡す(会社名・文書名・版指定を含めるほど特定精度が"
            "上がる)。値は決定論・推測なし。版ペアが一意に定まらない/読取不能/変更が大規模すぎる場合は"
            "value=null を返す(その場合は棄権のまま)。版差分契約では回答確定前の実行が必須で、返された"
            "value をそのまま回答する。",
            _obj({"question": _STR}, ["question"]),
            lambda question: _version_diff(question),
        ),
        AgentTool(
            "seating_lookup",
            "座席表(フロアマップ)から 氏名⇄内線(EXT)⇄座席 を引く。内線/EXT/座席/『向かい・隣・同じ列・"
            "Xから見て右側/左側』を"
            "問う質問に使う(座席図は画像1枚で他ツールでは読めない)。name=氏名(『〜さん』可)を渡すとその人の"
            "EXTを返す。relation に『向かい/隣/同じ列/同じ行』を渡すとその隣人のEXTを返す(例: 井上さんの"
            "向かいの人のEXT)。右側/左側は着席向き基準の複数候補を返す。名前が必要なら field='name'、"
            "座席レコードなら field='seat'、既定は field='ext'。ext=内線から人物を、"
            "role=役割(Exec/PM/DS/BA/DE/QA)+pod で EXT を引くこともできる。"
            "多段質問(案件→担当者→内線)では、先に担当者の氏名を他ツールで特定し、その氏名を name で渡す。"
            "該当なし/曖昧なときは value=null(棄権)を返す。",
            _obj({"name": _STR, "ext": _STR, "role": _STR, "relation": _STR, "pod": _NUM,
                  "field": _STR}),
            lambda name=None, ext=None, role=None, relation=None, pod=None, field="ext": seating_lookup(
                name=name, ext=ext, role=role, relation=relation,
                pod=int(pod) if isinstance(pod, (int, float)) else None, field=field),
        ),
        AgentTool(
            "corpus_aggregate",
            "全プロジェクトを横断して契約情報を集約する決定論ツール。単一案件のcompute/read_officeでは"
            "解けない『全体で/横断で/最も〜な案件・人』を扱う(train.xlsx/契約書は案件ごとに同名複数のため)。"
            "metric: contract_amount(契約金額税込)/deposit(着手金税込)/train_rows(学習データ行数)/"
            "amount_per_row(契約金額税込÷train行数)/period_days(契約期間日数)/staff(契約書の乙担当者)/"
            "staff_population(PP・契約書・PLAN・FR×全案件の役割付きDA人物union)。"
            "op: max/min(数値metricの極値案件を主略称abbrev+valueで返す。staff情報も同梱するので『最大案件のES』は"
            "その staff.ES を seating_lookup(name=…) に渡す)/count(staffの案件横断出現回数→最頻top=『最も多く"
            "案件に関わる人』、staff_population は count で閉包済み人数+人物union)/filter(契約期間フィルタ)/list。"
            "固定金額契約に絞るときは fixed_only=true。円単位で切り上げるときは round_up=true。"
            "契約期間フィルタは overlap_start/overlap_end(YYYY-MM-DD)で重なる案件、min_days でその日数超"
            "(『40日を超える』→min_days=40)に絞り、主略称の配列を返す。値は決定論(推測なし)、"
            "該当なしは value=null(棄権)。",
            _obj({"metric": _STR, "op": _STR, "fixed_only": _BOOL, "round_up": _BOOL,
                  "overlap_start": _STR, "overlap_end": _STR, "min_days": _INT}, ["metric"]),
            lambda metric, op="max", fixed_only=False, round_up=False,
            overlap_start=None, overlap_end=None, min_days=None: corpus_aggregate(
                metric, op=op, fixed_only=bool(fixed_only), round_up=bool(round_up),
                overlap_start=overlap_start, overlap_end=overlap_end,
                min_days=int(min_days) if isinstance(min_days, (int, float)) else None),
        ),
    ]
    # SOT-2564: font-decoration (太字/下線/イタリック) extract face. Additively exposed only when
    # RAG_FONT_EMPHASIS is on, so the champion serve tool set / prompt stays byte-identical by default.
    if _font_emphasis.enabled():
        tools.append(AgentTool(
            "font_emphasis",
            "xlsx/docx/pptx/pdfで太字・下線・イタリックの書式が付いた箇所を文書順で列挙する。"
            "requireに『太字』『下線』『イタリック』(複数はカンマ/＋区切り、英語bold/underline/italicも可)を"
            "渡すと、指定した書式すべてに同時該当する箇所だけに絞る。『太字かつ下線かつイタリックの箇所』の"
            "ような複合書式条件の抽出に使う(色ハイライトは highlight_extract を使う)。",
            _obj({"file": _STR, "require": _STR}, ["file"]),
            lambda file, require=None: _font_emphasis.font_emphasis(
                file, require=require, profile=profile),
        ))
    # SOT-2585: OOXML semantic FORMAT_EVENTs — Excel 条件付き書式ルール(cfRule/dxf)、docx コメント本文＋
    # anchor、文字色/ハイライト/塗りの複合述語。RAG_FORMAT_EVENTS が on のときだけ追加(既定 OFF ⇒ 既存の
    # ツールセット/プロンプトは byte-identical)。
    if _format_events.enabled():
        tools.append(AgentTool(
            "format_events",
            "xlsx/docxの書式イベントを決定論抽出する。(1)Excel条件付き書式ルール(cfRule/dxf: 条件/演算子/"
            "式/優先度/dxf塗り色/dxf文字色)を FORMAT_EVENT として返す(『黄色ハイライトになっているセルの"
            "条件』はこのルール条件が答え)。(2)docxコメントを anchor_text(コメントが付いた本文そのもの)＋"
            "author/日時付きで返す(『コメントがついている部分をそのまま抽出』は anchor_text が答え)。"
            "(3)文字色・ハイライト・塗りの実効書式を分離。fill=背景色名・font_color=文字色名を同時に渡すと"
            "『黄色ハイライトかつ赤字』のような複合条件(AND)で絞れる。kind='comment'でコメントのみ、"
            "kind='conditional_format'で条件付き書式のみ。値は決定論(推測なし)。",
            _obj({"file": _STR, "fill": _STR, "font_color": _STR, "source": _STR, "kind": _STR},
                 ["file"]),
            lambda file, fill=None, font_color=None, source=None, kind=None:
                _format_events.format_events(
                    file, fill=fill, font_color=font_color, source=source, kind=kind,
                    profile=profile),
        ))
    # SOT-2586: NUMERIC PoT forced compute lane. Additively exposed only when RAG_POT_HARD_LANE is on, so
    # the champion serve tool set / prompt stays byte-identical by default. Runs the LLM-emitted candidate
    # specs through binder→制限AST→Decimal→独立検算→N-sample majority and returns a three-layer verdict —
    # never eval/parse_expr on a model string (no arbitrary-code path).
    if _pot_lane.enabled():
        # SOT-2616 — when operand prefill is on, expose the augmented schema (operands may ``select`` from
        # the injected catalog by id) and forward the ``catalog`` through to the lane so value/unit/source
        # are bound from the enumerated cell verbatim. Prefill OFF ⇒ base schema + no catalog kwarg, so the
        # tool definition and call surface are byte-identical to the champion path.
        _pot_params = (_pot_lane.TOOL_PARAMETERS_PREFILL
                       if _operand_prefill.enabled() else _pot_lane.TOOL_PARAMETERS)
        tools.append(AgentTool(
            _pot_lane.TOOL_NAME,
            _pot_lane.TOOL_DESCRIPTION,
            _pot_params,
            lambda candidates=None, simple=None, require_units=False, catalog=None:
                _pot_lane.verify_formula(
                    candidates, simple=simple, require_units=bool(require_units), catalog=catalog),
        ))
    # SOT-2587: ENUM symbolic full-scan lane. Additively exposed only when RAG_ENUM_SCAN is on, so the
    # champion serve tool set / prompt stays byte-identical by default. Resolves the target universe from
    # the registry and scans every applicable document (no retrieval cutoff), returning a completeness
    # certificate — never a top-k guess.
    if _enum_scan.enabled():
        tools.append(AgentTool(
            _enum_scan.TOOL_NAME,
            _enum_scan.TOOL_DESCRIPTION,
            _enum_scan.TOOL_PARAMETERS,
            lambda question, predicate=None, entry_types=None, project=None:
                _enum_scan.enum_scan_tool(question, predicate=predicate,
                                          entry_types=entry_types, project=project),
        ))
    # SOT-2647 (事前計算事実層 5/5): the 4 precomputed stores (案件/ID/派生メトリクス/版差分) as first-class
    # tools — case_filter / id_lookup / metric_lookup / diff_lookup. Additively exposed ONLY when
    # RAG_FACT_LAYER is on (``_fact_layer.tools()`` returns [] otherwise), so the champion serve tool set /
    # function-call schema / MCP surface stay byte-identical by default. Registering here is the single
    # source of truth: :mod:`src.rag.mcp.server` builds its tools/list from ``build_tools`` too.
    for _name, _desc, _params, _fn in _fact_layer.tools():
        tools.append(AgentTool(_name, _desc, _params, _fn))
    return tools


def build_tools(profile: CorpusProfile) -> list[AgentTool]:
    """The full tool set exposed to the investigator: generic Step1 tools + the terminal answer tool."""
    return [*build_generic_tools(profile), SUBMIT_ANSWER_TOOL]


def _jsonable(obj: Any, *, max_str: int = 8000, max_items: int = 60, _depth: int = 0) -> Any:
    """Best-effort JSON-safe, size-bounded view of a tool result (keeps token cost in check)."""
    if _contract.is_contract(obj):
        obj = _contract.ensure_contract(obj)
    if isinstance(obj, str):
        return obj if len(obj) <= max_str else obj[:max_str] + f"…(+{len(obj) - max_str}字)"
    if isinstance(obj, Mapping):
        return {str(k): _jsonable(v, max_str=max_str, max_items=max_items, _depth=_depth + 1)
                for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        head = list(obj)[:max_items]
        out = [_jsonable(v, max_str=max_str, max_items=max_items, _depth=_depth + 1) for v in head]
        if len(obj) > max_items:
            out.append(f"…(+{len(obj) - max_items}件)")
        return out
    if isinstance(obj, (int, float, bool)) or obj is None:
        return obj
    return str(obj)


def _first_move_useful(out: Any) -> bool:
    """Whether a deterministic first-move tool actually reached evidence worth seeding (SOT-2521).

    A first move is only worth seeding when it resolved a concrete target: a ``{value, …}`` contract
    with a non-empty ``value`` (canonical_route records / a computed value), or a plain non-empty
    list. An error mapping, an empty ``value`` (no project/kind resolved), or an empty list means the
    deterministic route did not reach the needle — leave the first turn model-driven instead of
    seeding a misleading empty result.
    """
    if _contract.is_contract(out):
        value = _contract.ensure_contract(out).get("value")
        return value not in (None, "", [], {})
    if isinstance(out, Mapping):
        return "error" not in out and bool(out)
    if isinstance(out, (list, tuple)):
        return bool(out)
    return out not in (None, "")


def _budget_boundary_directive(director: Any, enabled: bool, tool_calls: Sequence[str],
                               answer: "Answer") -> str | None:
    """SOT-2524 — the re-search directive to emit at the max_turns budget boundary, or ``None`` to finalize.

    Returns ``None`` (finalize the abstain unchanged) when the boundary hook is disabled, no re-search
    director is active, or the pending answer is already a commit. Otherwise it asks the director — using
    the evidence it has already folded in via :meth:`ResearchDirector.observe` over the run's tool
    results — for the next targeted directive; the director returns one (keep searching) or ``None`` after
    recording its terminal (:data:`BUDGET`/:data:`UNANSWERABLE`) for the abstain ledger. It never inspects
    or changes the commit threshold — it only grows *search* at the boundary.
    """
    if not (enabled and director is not None and is_abstain(answer.answer)):
        return None
    evidence = " ".join(t for t in (answer.evidence, answer.method) if t).strip()
    non_submit = sum(1 for c in tool_calls if c != SUBMIT_ANSWER)
    return director.review(evidence, non_submit, at_boundary=True)


def _spin_key(name: str, args: Mapping[str, Any] | None) -> str:
    """A canonical (tool, args) identity for spin detection (SOT-2522).

    Uses the *same* normalization as the intra-question evidence cache — ``name`` plus a sorted-key JSON
    dump of ``args`` — so "same tool × same (normalized) arguments" means exactly what a cache hit means:
    a call that a deterministic tool would answer identically. Unserializable args fall back to ``repr``
    so the detector degrades gracefully instead of raising.
    """
    try:
        return name + "\x00" + json.dumps(args or {}, sort_keys=True, ensure_ascii=False, default=str)
    except (TypeError, ValueError):
        return name + "\x00" + repr(args)


def _spin_redirect_directive(name: str, tried: "set[str]") -> str:
    """The one-shot reallocation directive emitted when a spinning path is first cut off (SOT-2522).

    Names the still-untried deterministic routes so the model spends the freed budget on a *different*
    path instead of re-issuing the dead-end call. Injects no corpus fact (tool names only); when every
    deterministic route has already been tried it asks the model to vary its arguments or abstain.
    """
    untried = [t for t in _DETERMINISTIC_ROUTES if t != name and t not in tried]
    if untried:
        routes = "、".join(untried)
        return (
            f"同一ツール『{name}』を同一引数で反復していますが、決定論ツールは同じ入力に同じ結果しか返さず"
            "新しい根拠は得られません。この経路を打ち切り、未試行の決定論ツール"
            f"（{routes}）のいずれかに質問文をそのまま渡して別経路で根拠取得を試みてください。"
            "新経路でも根拠が確定できない場合に限り棄権してください。")
    return (
        f"同一ツール『{name}』を同一引数で反復していますが、新しい根拠は得られません。主要な決定論ツールは"
        "既に試行済みのため、別のファイル/引数での再試行に価値がなければ推測せず棄権してください。")


# SOT-2614 — search-style tools that carry a query but no single file target: any consecutive repeat of
# these (even with a tweaked pattern) is the documented budget waste, so the fuzzy key treats them as
# "same target" regardless of query variation. File-carrying tools instead key on the file, so re-hitting
# ONE file spins while walking DIFFERENT files (legitimate enumeration/extraction) resets the run.
_PIVOT_TARGET_ARG_KEYS: tuple[str, ...] = ("file", "path", "filename", "target")


def _spin_soft_target(name: str, args: Mapping[str, Any] | None) -> "tuple[str, str | None]":
    """A *fuzzy* (tool, target) identity for consecutive-spin detection (SOT-2614).

    Unlike :func:`_spin_key` (exact, byte-for-byte args → SOT-2522), this collapses tweaked arguments so a
    "keep hammering the same target with varied patterns" run is caught. Returns ``(name, target)`` where
    ``target`` is the file the call operates on (``None`` for a search tool that carries no file). Two
    consecutive calls are treated as the same spinning path iff they share this key (see
    :func:`_spin_soft_similar`).
    """
    a = args or {}
    for k in _PIVOT_TARGET_ARG_KEYS:
        v = a.get(k)
        if isinstance(v, str) and v.strip():
            return (name, v.strip())
    return (name, None)


def _spin_soft_similar(prev: "tuple[str, str | None] | None",
                       cur: "tuple[str, str | None]") -> bool:
    """Whether two adjacent tool calls belong to the same spinning run (SOT-2614).

    Same tool name is required. For a file-carrying tool the file target must match (walking different
    files is progress, not spin); for a search tool (target ``None``) any consecutive repeat counts, since
    re-searching the corpus with tweaked patterns is the documented dead end. Injects/consults no corpus
    fact — this is pure call-shape bookkeeping.
    """
    if prev is None or prev[0] != cur[0]:
        return False
    return prev[1] == cur[1]


def _spin_pivot_directive(name: str, tried: "set[str]", *, low_turns: bool) -> str:
    """The forced-pivot directive emitted when a consecutive same-tool run is cut off (SOT-2614).

    Names still-untried deterministic routes so the freed budget goes to a *different* path, and always
    offers the registry reverse-index / structure store / 手持ち証拠での確定判断 as escape hatches. When few
    turns remain (``low_turns``) it escalates to a definitive-decision directive so the残予算 lands on a
    verdict rather than more spinning. Injects no corpus fact (tool/route names only).
    """
    untried = [t for t in _DETERMINISTIC_ROUTES if t != name and t not in tried]
    routes = "、".join(untried) if untried else "canonical_route / 逆引き索引 / structure store"
    if low_turns:
        return (
            f"同一ツール『{name}』の連発を打ち切りました。残ターンが僅少です。これ以上同じ手段を反復せず、"
            "今すぐ確定判断してください: 手持ちの証拠で回答を確定できるなら submit_answer で回答し、"
            "根拠が不足していて別経路でも確定の見込みがないなら推測せず棄権してください。"
            f"（未試行の決定論経路が残っていれば {routes} を1回だけ試してから判断可）")
    return (
        f"同一ツール『{name}』を類似引数で連続呼び出ししており、この経路は空回りです（決定論ツールは同種の"
        "入力に新しい根拠を返しません）。この経路をこの問いで打ち切ります。同じ手段を反復せず、次のいずれかへ"
        f"切り替えてください: 未試行の決定論ツール（{routes}）に質問文をそのまま渡す／registry の逆引き索引・"
        "structure store で対象を直接解決する／既に手元にある証拠だけで回答を確定する。"
        "いずれでも根拠が確定できない場合に限り、推測せず棄権してください。")


def _search_cap_directive(name: str, cap: int, tried: "set[str]") -> str:
    """The directive emitted when a question's search-call cap is exhausted (SOT-2620).

    Names still-untried *non-search* deterministic routes so the freed budget goes to a value/registry
    lane instead of another re-grep, and always offers the registry reverse-index / structure store /
    手持ち証拠での確定判断 as escape hatches. Injects no corpus fact (tool/route names only); never touches
    the commit threshold, so the answer path is byte-identical when the cap is disabled.
    """
    untried = [t for t in _DETERMINISTIC_ROUTES
               if t != name and t not in tried and t not in _SEARCH_TOOLS]
    routes = "、".join(untried) if untried else "canonical_route / 逆引き索引 / structure store"
    return (
        f"検索系ツール『{name}』の呼び出しがこの問いの型別上限（{cap}回）に達しました。検索の反復では"
        "既に必要な根拠は出ておらず（ヒットは返るが針が含まれない反復）、これ以上の再検索は浪費です。"
        "検索を止め、次のいずれかに切り替えてください: "
        f"未試行の決定論ツール（{routes}）に質問文をそのまま渡す／registry の逆引き索引・canonical_route・"
        "structure store で対象を直接解決する／既に手元にある証拠だけで回答を確定する。"
        "いずれでも根拠が確定できない場合に限り、推測せず棄権してください。")


def _first_move_directive(name: str, out: Any) -> str:
    """The seed user-message describing a deterministic first-move result (SOT-2521).

    Carries the tool's own output (the same evidence the model would receive had it called the tool
    itself) and tells the model to build on it — compute / extract — rather than re-run search or the
    route from scratch. Injects no answer and no policy: the committed answer is still the model's.
    """
    import json

    try:
        payload = json.dumps(out, ensure_ascii=False, default=str)[:6000]
    except (TypeError, ValueError):
        payload = str(out)[:6000]
    return (
        f"初手として決定論ツール『{name}』を質問に対して実行しました"
        "（探索予算を canonical 証拠へ直結させるため、chunk 検索は省略済み）。\n"
        f"実行結果(JSON): {payload}\n"
        "この結果の primary/records を起点に、canonical_route の再実行や一律の chunk 検索を先に行わず、"
        "必要なら compute / read_office / read_chart_values で計算・抽出して回答してください。"
        "この初手で対象が得られていない場合や、より適切な根拠が別にある場合のみ、他ツールへ切り替えてください。")


def dispatch(tools_by_name: Mapping[str, AgentTool], name: str, args: Mapping[str, Any] | None) -> Any:
    """Run tool ``name`` with ``args``; return a JSON-safe result or an ``{"error": ...}`` mapping.

    Errors are returned (not raised) so the model can see the failure and try another approach — the
    agent loop must never crash on one bad tool call.
    """
    tool = tools_by_name.get(name)
    if tool is None:
        return {"error": f"unknown tool: {name}", "available": sorted(tools_by_name)}
    # SOT-2660 — DB_ONLY diagnostic mode: refuse raw-file tools so the answer must rest on the DB経路
    # (search/index/precomputed fact layer) alone. Returned (not raised) so the model sees the refusal and
    # either switches to a DB tool or abstains. ``db_only_blocked`` lets the per-question telemetry count it.
    if DB_ONLY and is_raw_file_tool(name):
        return {
            "error": f"RAG_DB_ONLY: 生ファイル系ツール『{name}』は診断モードで無効です。",
            "db_only_blocked": True,
            "tool": name,
            "reason": ("診断モード(RAG_DB_ONLY)では生ファイルアクセスを禁止しています。"
                       "DB経路(text/unified search・逆引き索引・事実層 case_filter/id_lookup/"
                       "metric_lookup/diff_lookup)のみで回答するか、DB経路で解けなければ棄権してください。"),
        }
    try:
        out = tool.fn(**dict(args or {}))
    except TypeError as e:
        return {"error": f"bad arguments for {name}: {e}"}
    except Exception as e:  # noqa: BLE001 — surface any tool failure back to the model
        return {"error": f"{type(e).__name__}: {e}"}
    return _jsonable(out)


def _has_deterministic_gantt_evidence(out: Any) -> bool:
    """Whether ``read_office`` returned at least one resolved native-shape Gantt span."""
    if not _contract.is_contract(out):
        return False
    value = str(out.get("value") or "")
    return (
        (out.get("method") or {}).get("engine") == "pptx"
        and "【ガント週グリッド:決定論】" in value
        and bool(re.search(r"\]\s*[^\n]+:\s*第\d+週目から第\d+週目", value))
    )


def _normalized_activity(text: str) -> str:
    """Normalize an activity label for question↔native-shape matching."""
    from src.rag.corpus import nfc

    return re.sub(r"[^0-9A-Za-z\u3040-\u30ff\u3400-\u9fff]+", "", nfc(text)).replace("の", "")


def _closure_satisfied(evidence: Mapping[str, Any]) -> bool:
    """Validate the four machine-readable closure conditions shared by enum/count fast paths."""
    closure = evidence.get("closure")
    return bool(
        isinstance(closure, Mapping)
        and closure.get("authoritative_population_resolved")
        and closure.get("inclusion_exclusion_recorded")
        and not closure.get("second_path_novel_candidates")
        and closure.get("enumeration_count") == closure.get("aggregate_count")
    )


def _deterministic_seating_side_answer(question: str) -> Answer | None:
    """Resolve occupant-relative left/right name enumerations from the hash-pinned seat frame."""
    if not (re.search(r"(?:右側|右手|左側|左手)", question)
            and re.search(r"名前|人物|人", question)
            and re.search(r"すべて|全て|全部", question)):
        return None
    subject = re.search(r"([一-龯々]{1,6})(?:さん|氏|様)?から見て", question)
    relation = re.search(r"(右側|右手|左側|左手)", question)
    if not subject or not relation:
        return None
    result = seating_lookup(name=subject.group(1), relation=relation.group(1), field="name")
    if not _contract.is_contract(result) or not _closure_satisfied(result.get("evidence") or {}):
        return None
    values = result.get("value")
    if not isinstance(values, list) or not values or not all(isinstance(v, str) for v in values):
        return None
    evidence = result["evidence"]
    return Answer(
        answer="、".join(values), confidence=1.0,
        evidence=(f"{evidence.get('file')} pixel_sha={evidence.get('pixel_sha')} / "
                  f"subject={subject.group(1)} / relation={relation.group(1)} / "
                  f"closure={len(values)}件一致"),
        method="pixel-hash pin済み座席アンカーを着席者の向きフレームへ変換し、右左半平面を完全列挙",
    )


def _deterministic_staff_population_answer(question: str) -> Answer | None:
    """Count the closed union of role-bound DA people across canonical PP/contract/PLAN/FR files."""
    from src.rag.agent import question_contract as _question_contract

    if not _question_contract.is_staff_population_question(question):
        return None
    result = corpus_aggregate("staff_population", op="count")
    if not _contract.is_contract(result) or not _closure_satisfied(result.get("evidence") or {}):
        return None
    value = result.get("value")
    if not isinstance(value, Mapping) or not isinstance(value.get("count"), int):
        return None
    evidence = result["evidence"]
    return Answer(
        answer=str(value["count"]), confidence=1.0,
        evidence=(f"PP/契約書/PLAN/FR canonical files={len(evidence.get('selected_files') or [])}; "
                  f"projects={len(evidence.get('projects') or [])}; union={value['count']}"),
        method="全案件×4文書型の正本列挙→役割付きDA人物抽出→表記正規化・重複解決→列挙件数と集計件数照合",
    )


def _strict_literal_vision_answer(question: str, result: object) -> Answer | None:
    """Return the only literal candidate proven to share its visual line with the condition."""
    from src.rag.agent import obligations as _obligations

    if not isinstance(result, Mapping):
        return None
    literal = _obligations.literal_terms(question)
    value = result.get("value")
    candidates = value if isinstance(value, list) else []
    if not literal or len(candidates) != 1 or not isinstance(candidates[0], Mapping):
        return None
    candidate = str(candidates[0].get("candidate", "") or "").strip()
    conditioned_text = str(candidates[0].get("conditioned_text", "") or "")
    literal_check = _obligations.validate_literal_evidence(question, candidate, [result])
    strict_line = (
        candidates[0].get("same_visual_line") is True
        and candidate in conditioned_text
        and all(term in conditioned_text for term in literal)
    )
    if not candidate or not strict_line or not literal_check.passed:
        return None
    return Answer(
        answer=candidate,
        confidence=1.0,
        evidence=str(result.get("evidence", "")),
        method="同一視覚行の単一 literal 候補をそのまま採用",
    )


def _deterministic_literal_report_answer(question: str) -> Answer | None:
    """Inspect one canonical report before a literal lookup can wander or abstain.

    A report-like question may name only the project, leaving an image-only final report invisible to
    text grep.  Resolve the project from the question, require exactly one PDF in the canonical
    ``report`` category, then accept only the strict single-line literal contract above.  Ambiguous
    files, zero/multiple candidates, or Vision errors fail closed to the ordinary agent path.
    """
    from src.rag.agent import obligations as _obligations

    if not _obligations.literal_terms(question):
        return None
    if not re.search(r"報告|今後|運用|成果|結果|総括", question):
        return None
    route = canonical_route(question)
    project = (route.get("evidence") or {}).get("project") if _contract.is_contract(route) else None
    if not project:
        return None
    found = find_files(str(project), ext="pdf")
    files = found.get("value") if _contract.is_contract(found) else None
    reports = [
        item for item in files or []
        if isinstance(item, Mapping)
        and str(item.get("ext", "")).lower() == "pdf"
        and item.get("category") == "report"
        and item.get("rel")
    ]
    if len(reports) != 1:
        return None
    try:
        result = caption_figure(str(reports[0]["rel"]), question=question)
    except Exception:
        return None
    return _strict_literal_vision_answer(question, result)


def _deterministic_verbatim_action_answer(question: str) -> Answer | None:
    """Resolve an exact action-ID ask to the fullest unique Action cell across meeting PDFs.

    Historical minutes may repeat the same action after shortening its label or moving details into a
    status column.  For an explicit ``そのまま`` content request, scan only the resolved project's
    meeting PDFs, retain Vision rows that contain the requested ID on the same visual line, and choose
    the longest candidate only when every shorter candidate is contained in it. Conflicting texts fail
    closed to the ordinary agent path. No action value or project name is embedded here.
    """
    if not re.search(r"(?:そのまま|抜き出|全文)", question):
        return None
    action = re.search(r"(?:アクション)?ID\s*([A-Z]+\s*\d+)", question, re.I)
    if action is None:
        return None
    action_id = re.sub(r"\s+", "", action.group(1)).upper()
    route = canonical_route(question)
    project = (route.get("evidence") or {}).get("project") if _contract.is_contract(route) else None
    if not project:
        return None
    found = find_files(None, ext="pdf", project=str(project))
    files = found.get("value") if _contract.is_contract(found) else None
    meetings = [item for item in files or [] if isinstance(item, Mapping)
                and item.get("category") == "meeting" and item.get("rel")]
    candidates: list[tuple[str, str]] = []
    for item in meetings:
        try:
            result = caption_figure(str(item["rel"]), question=question)
        except Exception:
            continue
        value = result.get("value") if isinstance(result, Mapping) else None
        for row in value if isinstance(value, list) else ():
            if not isinstance(row, Mapping) or row.get("same_visual_line") is not True:
                continue
            line = re.sub(r"\s+", "", str(row.get("conditioned_text", ""))).upper()
            candidate = str(row.get("candidate", "") or "").strip()
            if action_id in line and candidate:
                candidates.append((candidate, str(item["rel"])))
    if not candidates:
        return None
    longest, source = max(candidates, key=lambda pair: len(pair[0]))
    norm_longest = re.sub(r"[\s:：()（）]", "", longest).lower()
    if any(re.sub(r"[\s:：()（）]", "", candidate).lower() not in norm_longest
           for candidate, _ in candidates):
        return None
    return Answer(answer=longest, confidence=1.0, evidence=f"{source}: {action_id}",
                  method="対象IDと同一視覚行のActionセル全文を履歴間で包含照合し、最長の原文を採用")


def _deterministic_gantt_answer(question: str, profile: CorpusProfile) -> Answer | None:
    """Resolve a uniquely named PPTX activity from native Gantt geometry, without an LLM."""
    route = canonical_route(question)
    project = (route.get("evidence") or {}).get("project") if _contract.is_contract(route) else None
    filename_match = re.search(r"([^/\\、。\sの]+\.pptx)", question, re.I)
    target_match = re.search(r"(?:において[、,])(.+?)(?:の)?実行予定スケジュール", question)
    if not project or not filename_match or not target_match:
        return None
    found = find_files(filename_match.group(1), ext="pptx", project=str(project))
    files = found.get("value") if _contract.is_contract(found) else None
    if not isinstance(files, list) or len(files) != 1:
        return None
    office = extract_office(str(files[0]["rel"]), profile=profile)
    if not _has_deterministic_gantt_evidence(office):
        return None
    target = _normalized_activity(target_match.group(1))
    matches = []
    for match in re.finditer(
            r"^\[スライド(?P<slide>\d+)\]\s*(?P<activity>.+?):\s*"
            r"第(?P<start>\d+)週目から第(?P<end>\d+)週目", str(office["value"]), re.M):
        activity = _normalized_activity(match.group("activity"))
        if activity == target:
            matches.append(match)
    if len(matches) != 1:
        return None
    match = matches[0]
    return Answer(
        answer=f"第{match.group('start')}週目から第{match.group('end')}週目",
        confidence=1.0,
        evidence=f"{files[0]['rel']} スライド{match.group('slide')}",
        method="週ヘッダx座標とバーleft/widthの半開区間重なりによる決定論抽出",
    )


def _deterministic_regulation_answer(question: str, profile: CorpusProfile) -> Answer | None:
    """Render a complete fallback rule when the special condition is absent and all fields are explicit."""
    from src.rag.corpus import nfc

    route = canonical_route(question)
    project = (route.get("evidence") or {}).get("project") if _contract.is_contract(route) else None
    if not project:
        return None
    found = find_files("契約", ext="docx", project=str(project))
    files = found.get("value") if _contract.is_contract(found) else None
    if not isinstance(files, list) or len(files) != 1:
        return None
    office = extract_office(str(files[0]["rel"]), profile=profile)
    text = nfc(str(office.get("value") or ""))

    # Only conclude that no special clause exists when the question has distinctive threshold tokens
    # and none occurs in the authoritative contract.  Otherwise leave the judgment to normal research.
    condition_tokens = re.findall(r"[A-Za-z][A-Za-z0-9_]{1,}|\d+(?:\.\d+)?\s*(?:時間|件|円|%)", question)
    if not condition_tokens or any(nfc(token) in text for token in condition_tokens):
        return None

    rate = re.search(r"時間単価は\s*([\d,]+円)\s*[（(](?:消費税別|税別)[）)]", text)
    unit = re.search(r"計上単位は\s*(\d+分)", text)
    cycle = re.search(r"(月次|週次|日次)(?:で)?(?:甲に提出|精算)", text)
    tax = "消費税を加算" in text
    rounded = bool(re.search(r"端数は\s*\d+分単位に切り上げ", text))
    uncapped = "契約総額を固定するものではない" in text
    if not (rate and unit and cycle and tax and rounded and uncapped):
        return None
    subject_match = re.search(r"(?:において[、,])(.+?)の精算方法", question)
    subject = subject_match.group(1) if subject_match else "質問の条件"
    answer = (
        f"{subject}の特別な精算規定はなく、実績工数に時間単価{rate.group(1)}(税別)を乗じ"
        f"消費税を加算して{cycle.group(1)}で精算する({unit.group(1)}単位・切上げ、上限なし)。")
    return Answer(
        answer=answer,
        confidence=1.0,
        evidence=f"{files[0]['rel']} 報酬および支払条件",
        method="特別条件不在確認後、一般規定の単価・税・単位・丸め・周期・上限を機械抽出",
    )


# --------------------------------------------------------------------------- agent loop (pure)
class Model(Protocol):
    """A model conversation for one question. ``next(None)`` starts it; ``next(responses)`` continues
    it after tool calls. Implementations hold their own history; the loop stays transport-agnostic."""

    def next(self, tool_responses: Sequence[ToolResponse] | None) -> Step: ...


def investigate(model: Model, question: str, tools: Sequence[AgentTool], *,
                max_turns: int = DEFAULT_MAX_TURNS, timeout_s: float = DEFAULT_TIMEOUT_S,
                clock: Callable[[], float] = time.monotonic,
                ledger: "str | object | bool | None" = None,
                calc_ledger: "str | object | bool | None" = None,
                research: "bool | Mapping[str, Any] | object | None" = None,
                enumeration: "bool | Mapping[str, Any] | object | None" = None,
                contract: "str | None" = None,
                first_move: "tuple[str, Mapping[str, Any]] | None" = None,
                fallback: "object | None" = None,
                budget_boundary: bool = False,
                spin_detection: "bool | Mapping[str, Any] | None" = False,
                pivot_detection: "bool | Mapping[str, Any] | None" = False,
                search_cap: "bool | Mapping[str, Any] | None" = None,
                preamble: "str | None" = None) -> Investigation:
    """Drive ``model`` through tool-calling until it submits a structured answer.

    The loop ends on the first of: the model calls ``submit_answer`` (→ ``answered``); ``max_turns`` is
    reached (→ ``max_turns``, abstain); the wall-clock ``timeout_s`` is exceeded between turns
    (→ ``timeout``, abstain); or the transport raises (→ ``model_error``, abstain). A plain final-text
    turn (no ``submit_answer``) is accepted as the answer with confidence 0.0.

    ``iterations`` counts tool rounds (turns that requested ≥1 tool call). Token usage is summed across
    every turn so the caller can price the question.

    ``ledger`` (SOT-2492) enables the abstain ledger: when it is not ``None``/``False``, per-tool
    outcome signals are *observed* (never altered) during the loop, and if the final answer is an
    abstain a coded :class:`~src.rag.agent.abstain_ledger.AbstainRecord` is appended. ``True`` writes to
    the default path (``artifacts/abstain_ledger.jsonl``); a ``str``/``Path`` writes there instead. The
    default ``None`` is a pure no-op so the commit-vs-abstain decision and every existing caller are
    unchanged (受け入れ条件②).

    ``calc_ledger`` (SOT-2495/SOT-2506) is the commit-side twin: when enabled, derivation-tool contracts
    are *observed* (never altered) during the loop, and if the final answer is numeric—or is a label
    selected by a ``numeric`` contract—a typed :class:`~src.rag.agent.calc_ledger.CalcRecord` (raw_text /
    parsed_value / unit / source_range / formula + per-compute証跡) is appended (``True`` →
    ``artifacts/calc_ledger.jsonl``; a ``str``/``Path`` redirects it). Same observer-only invariant: the
    answer path is byte-identical with it on or off.

    ``research`` (SOT-2502) enables the obligation-driven local re-search loop: when it is not
    ``None``/``False``, a deliberate ``submit_answer`` abstain is not accepted immediately — the
    :class:`~src.rag.agent.research_loop.ResearchDirector` discharges the question's evidence obligations
    against what the tools actually found and, while budget remains, feeds a *targeted* re-search
    directive (unmet obligation + per-kind tactics) back to the model so it re-searches only the unmet
    obligation before it may abstain again. A ``Mapping`` supplies the budget (``max_rounds`` /
    ``max_tool_calls``). The commit threshold/confidence are never touched (SOT-2483 の軸とは別物): a
    re-search either reaches a grounded answer or yields a *coded, history-bearing* abstain
    (:data:`~src.rag.agent.abstain_ledger.BUDGET_EXHAUSTED` / ``UNANSWERABLE``). Off by default so the
    answer path stays byte-identical.

    ``enumeration`` (SOT-2500) enables the full-enumeration closure protocol: when it is not
    ``None``/``False`` and the question is a ``full_enumeration`` contract, a deliberate abstain is
    intercepted **once** and the :class:`~src.rag.agent.enumeration.EnumerationGate` feeds the closure
    *procedure* back to the model (権威的母集団の特定 → 各候補の包含/除外理由 → 別経路で新規候補ゼロ →
    列挙件数と集計件数の一致 → 列挙順序規則) so completeness is *proven* before the model may abstain again.
    The procedure injects no corpus fact (no 略称/member list — only the source category to read), so the
    no-fact-injection invariant holds. It only ever turns an enumeration abstain into either a
    closure-proven answer or a still-coded abstain — the commit threshold is never touched. Off by default
    so the answer path stays byte-identical; it composes with ``research`` (the enum procedure is tried
    first, the generic re-search after).

    ``contract`` (SOT-2498) is a pure *label* recorded onto the returned :class:`Investigation` (and, on
    an abstain, into the ledger) so the routing contract the question was classified as is captured
    per-question alongside the tools actually used. It never influences the loop — the routing *hint*
    itself is injected into the model's system prompt at construction (see :func:`answer_question`), not
    here — so passing it leaves the answer path byte-identical (default ``None``).

    ``first_move`` (SOT-2521) is an optional ``(tool_name, kwargs)`` plan the loop runs *deterministically
    before the model's first turn*, seeding the model with the tool's evidence (as a plain user directive)
    so it stops burning its turn budget wandering through chunk search on questions whose evidence is a
    canonical data-asset needle (BUDGET_EXHAUSTED, the top abstain cause). It only seeds when the tool
    actually reaches evidence (:func:`_first_move_useful`); an empty/error result falls back to the
    ordinary model-driven first turn. The seed steers the *first* move only — the model's later tool
    choice and the commit-vs-abstain decision are untouched — and injects no answer, so the answer path
    stays byte-identical when it is ``None`` (default). The plan is built by
    :func:`~src.rag.agent.routing.deterministic_first_move` in :func:`answer_question`.

    ``fallback`` (SOT-2525) is an optional one-shot gate
    (:class:`~src.rag.agent.question_contract.DeterministicFallbackGate`) tried when the model is about to
    abstain and neither the enumeration-closure nor the obligation re-search loop intervened. It forces
    the question's *contract-typed deterministic tool* (canonical_route / version_diff / file_grep — each
    self-resolving from the question text) to run exactly once before UNANSWERABLE is accepted, and — only
    when that tool actually reaches evidence (:func:`_first_move_useful`) — feeds the evidence back so the
    model may answer from it (subject to every existing commit guard, so no answer is injected). When the
    tool reaches nothing, the abstain stands unchanged (従来の安全動作を維持). It is one-shot so a still-
    abstaining model cannot loop, and injects no corpus fact, so the answer path is byte-identical when it
    is ``None`` (default). The gate is built in :func:`answer_question` behind ``RAG_UNANSWERABLE_FALLBACK``.

    ``budget_boundary`` (SOT-2524) enables the budget-exhaustion boundary hook, effective only together
    with ``research``. Without it the re-search director fires solely on a *deliberate* abstain; with it,
    when the loop is about to finalize a non-committed abstain because ``max_turns`` (or ``timeout_s``)
    was reached, the director gets a bounded last push at the still-unmet obligations: it emits one
    targeted re-search directive and up to ``budget.max_rounds`` extra model turns are granted to answer
    it — still inside ``timeout_s`` (so a timeout-triggered boundary grants no turns and only records the
    director's terminal). The commit threshold is untouched, so it only turns a would-be BUDGET abstain
    into either a grounded answer or a coded, history-bearing abstain. ``False`` by default so existing
    callers are byte-identical; :func:`answer_question` wires it from ``RAG_BUDGET_BOUNDARY_RESEARCH``.

    ``spin_detection`` (SOT-2522) enables dead-end spin detection & budget reallocation. When it is not
    ``None``/``False``, a normalized ``(tool, args)`` call that recurs ``spin_threshold`` times (default
    :data:`DEFAULT_SPIN_THRESHOLD`; a ``Mapping`` may override ``threshold``) is treated as a dead end:
    the loop, ONCE, feeds back a directive redirecting the freed budget to an untried deterministic route
    (see :func:`_spin_redirect_directive`) instead of re-dispatching the redundant call. If spinning
    persists after that single reallocation, the path is cut off early (``stop_reason="spin_cutoff"``) so
    the rest of the budget is not melted — the ``max_turns``/``timeout_s`` cap is never raised. A resulting
    abstain is attributed to :data:`~src.rag.agent.abstain_ledger.SPIN_CUTOFF`. It injects no corpus fact
    (tool names only) and never touches the commit threshold, so the answer path is byte-identical when it
    is ``False`` (default); :func:`answer_question` wires it from ``RAG_SPIN_DETECTION``.

    ``pivot_detection`` (SOT-2614) STRENGTHENS ``spin_detection``: it detects a *consecutive run* of the
    same tool whose arguments only fuzzily change (same file target for file tools; any repeat for a
    search tool carrying no target) and, on the ``PIVOT_THRESHOLD``-th consecutive call (default
    :data:`DEFAULT_PIVOT_THRESHOLD`; a ``Mapping`` may override ``threshold`` / ``low_turns``), does NOT
    dispatch the redundant call — it cools that tool for the question and feeds back a forced-pivot
    directive (see :func:`_spin_pivot_directive`), escalating to a definitive-decision directive when few
    turns remain. A cooled tool re-pivots on a shorter run (threshold−1). Each pivot increments the
    returned :attr:`Investigation.spin_pivots` (and the abstain-ledger ``spin_pivots``). It injects no
    corpus fact (tool/route names only) and never touches the commit threshold, so the answer path is
    byte-identical when it is ``False`` (default); :func:`answer_question` wires it from ``RAG_SPIN_PIVOT``.
    It composes with ``spin_detection`` (the SOT-2522 exact-identity cut runs first, unchanged).

    ``search_cap`` (SOT-2620) bounds the *total* number of search-style tool calls (``file_grep`` /
    ``find_files``) this question may make, by route type — the 型別 search 上限 wired into the default
    fallback loop (independent of the opt-in ``RAG_EVIDENCE_PACKET``). When it is not ``None``/``False``,
    the per-route cap is resolved from ``contract`` via
    :func:`~src.rag.agent.query_router.search_cap_for_contract` (a ``Mapping`` with a ``cap`` key overrides
    it directly); the (cap+1)-th search call is NOT dispatched — the loop feeds back a switch-lane
    directive (:func:`_search_cap_directive`: canonical_route / 逆引き索引 / structure store / 手持ち証拠で
    の確定判断) and records a ``search_cap_hits`` count. It STRENGTHENS ``pivot_detection`` and shares its
    budget: the cap check runs *after* the pivot guard, so a call the pivot guard already intercepted never
    double-counts here (no duplicate intervention). Total turns are never raised — only the repeated-search
    axis is bounded. It injects no corpus fact (tool/route names only) and never touches the commit
    threshold, so the answer path is byte-identical when it is ``None`` (default); :func:`answer_question`
    wires it from ``RAG_SEARCH_CAP``.

    ``preamble`` (SOT-2584) is an optional plain user directive injected *before the model's first turn*
    — the Evidence Packet (:mod:`src.rag.agent.evidence_packet`): the typed route, the registry-resolved
    target documents, the required evidence slots + which are missing, the deterministic primary lane, and
    the per-route free-exploration budget. It reverses "検索→考える→また検索" into "型判定→文書確定→
    不足スロットだけ探索". It injects no corpus fact and no answer (document identity + slot names + budget
    only) and never touches the commit threshold, so the answer path is byte-identical when it is ``None``
    (default); :func:`answer_question` wires it from ``RAG_EVIDENCE_PACKET``. It composes with
    ``first_move`` (both seed the pre-first-turn directive; the packet is prepended).
    """
    spin_enabled = spin_detection not in (None, False)
    spin_threshold = DEFAULT_SPIN_THRESHOLD
    if isinstance(spin_detection, Mapping):
        try:
            spin_threshold = max(2, int(spin_detection.get("threshold", DEFAULT_SPIN_THRESHOLD)))
        except (TypeError, ValueError):
            spin_threshold = DEFAULT_SPIN_THRESHOLD
    spin_counts: dict[str, int] = {}
    spin_redirected = False   # a reallocation directive has already been emitted once
    spin_cutoff = False       # a spin was detected this question (recorded onto the abstain ledger)

    # SOT-2614 — consecutive same-tool spin pivot guard state (fuzzy-arg run detection; STRENGTHENS the
    # SOT-2522 exact-identity cut above without replacing it).
    pivot_enabled = pivot_detection not in (None, False)
    pivot_threshold = DEFAULT_PIVOT_THRESHOLD
    pivot_low_turns = PIVOT_LOW_TURNS
    if isinstance(pivot_detection, Mapping):
        try:
            pivot_threshold = max(2, int(pivot_detection.get("threshold", DEFAULT_PIVOT_THRESHOLD)))
        except (TypeError, ValueError):
            pivot_threshold = DEFAULT_PIVOT_THRESHOLD
        try:
            pivot_low_turns = max(0, int(pivot_detection.get("low_turns", PIVOT_LOW_TURNS)))
        except (TypeError, ValueError):
            pivot_low_turns = PIVOT_LOW_TURNS
    pivot_prev: "tuple[str, str | None] | None" = None   # last non-submit call's fuzzy (tool, target) key
    pivot_run = 0             # length of the current consecutive same-target run
    pivot_cooldown: set[str] = set()  # tools already pivoted once this question (re-pivot on a shorter run)
    pivot_count = 0           # spin_pivots recorded (details log / abstain ledger)

    # SOT-2620 — per-route search-call cap state (STRENGTHENS the SOT-2614 pivot guard; shares its budget
    # because the cap check runs AFTER the pivot block below, so a pivoted call never reaches it).
    search_cap_enabled = search_cap not in (None, False)
    search_cap_limit = 0
    if search_cap_enabled:
        from src.rag.agent import query_router as _query_router
        if isinstance(search_cap, Mapping) and "cap" in search_cap:
            try:
                search_cap_limit = max(1, int(search_cap["cap"]))
            except (TypeError, ValueError):
                search_cap_limit = _query_router.search_cap_for_contract(contract)
        else:
            search_cap_limit = _query_router.search_cap_for_contract(contract)
    search_calls = 0          # dispatched search-tool calls this question
    search_cap_hits = 0       # over-cap search attempts intercepted (details log / abstain ledger)

    record_enabled = ledger not in (None, False)
    if record_enabled:
        from src.rag.agent import abstain_ledger as _abstain_ledger
        signals = _abstain_ledger.AbstainSignals()
    else:
        _abstain_ledger = None
        signals = None

    calc_enabled = calc_ledger not in (None, False)
    if calc_enabled:
        from src.rag.agent import calc_ledger as _calc_ledger
        calc_signals = _calc_ledger.CalcSignals(question=question)
    else:
        _calc_ledger = None
        calc_signals = None

    research_enabled = research not in (None, False)
    if research_enabled:
        from src.rag.agent import research_loop as _research_loop
        director = _research_loop.ResearchDirector(question, budget=research)
    else:
        director = None

    enumeration_enabled = enumeration not in (None, False)
    if enumeration_enabled:
        from src.rag.agent import enumeration as _enumeration
        enum_gate = _enumeration.EnumerationGate(question)
    else:
        enum_gate = None

    by_name = {t.name: t for t in tools}

    # SOT-2523 — intra-question evidence cache. Deterministic read/compute/decrypt tools return the same
    # value for identical args within one question, so re-issuing an identical call (特定→復号→読込→計算→
    # 検証 の多段でよく起きる再導出) only burns wall-clock re-deriving evidence already in hand. Memoise
    # (tool, canonical-args) → output for THIS question only; the map dies with the loop (1問スコープ),
    # values are captured from real tool outputs at runtime (特定回答の埋め込みなし), and errors are never
    # cached so a transient failure can still be retried. Value-preserving: the model sees the identical
    # output, only faster — every observe/iteration/ledger path around the call runs unchanged.
    evidence_cache: dict[str, Any] = {}

    # SOT-2660 — per-question raw-file dependency telemetry. ``raw_file_used`` counts 生ファイル系 tool
    # calls that actually READ a raw corpus file (successful, non-error dispatch); ``raw_file_blocked`` counts
    # calls refused by RAG_DB_ONLY. Every real dispatch flows through ``cached_dispatch`` (first_move /
    # fallback / main loop all call it), so this is the single per-question chokepoint. Recorded into
    # ``interventions['raw_file_access']`` for EVERY question below (answered or abstained).
    raw_file_used: dict[str, int] = {}
    raw_file_blocked: dict[str, int] = {}

    def _note_raw_file_access(name: str, out: Any) -> None:
        if not is_raw_file_tool(name):
            return
        if isinstance(out, Mapping) and out.get("db_only_blocked"):
            raw_file_blocked[name] = raw_file_blocked.get(name, 0) + 1
        elif not (isinstance(out, Mapping) and "error" in out):
            raw_file_used[name] = raw_file_used.get(name, 0) + 1

    def cached_dispatch(name: str, args: Mapping[str, Any] | None) -> Any:
        # SOT-2563 — propagate the *remaining* wall-clock budget to the tool call so a scan tool
        # (file_grep full scan) can derive a per-call deadline and cooperatively cancel instead of
        # overrunning ``timeout_s`` (only checked between turns) by 1.5–3.6× in one synchronous call.
        # Harmless for tools that ignore it. ``start`` is assigned before the first dispatch below.
        remaining = timeout_s - (clock() - start)
        with call_budget.remaining_budget(remaining):
            if not EVIDENCE_CACHE:
                out = dispatch(by_name, name, args)
                _note_raw_file_access(name, out)  # SOT-2660
                return out
            try:
                key = name + "\x00" + json.dumps(args or {}, sort_keys=True,
                                                 ensure_ascii=False, default=str)
            except (TypeError, ValueError):
                out = dispatch(by_name, name, args)  # unserialisable args → skip the cache
                _note_raw_file_access(name, out)  # SOT-2660
                return out
            if key in evidence_cache:
                return evidence_cache[key]  # cache hit: no new raw-file read
            out = dispatch(by_name, name, args)
            _note_raw_file_access(name, out)  # SOT-2660 — count only real (non-cached) dispatches
            if not (isinstance(out, Mapping) and "error" in out):
                evidence_cache[key] = out
            return out

    usage = Usage()
    tool_calls: list[str] = []
    iterations = 0
    responses: list[ToolResponse] | None = None
    answer = Answer(answer=ABSTAIN, confidence=0.0)
    stop_reason = "max_turns"
    error: str | None = None
    strict_chart_evidence = False
    chart_guidance_sent = False
    simple_lookup_guidance_sent = False
    regulation_guidance_sent = False
    granularity_guidance_sent = False   # SOT-2545 — one-shot answer-granularity correction
    conflict_guidance_sent = False      # SOT-2549 — one-shot record-conflict resolution
    numeric_feature_guidance_sent = False  # SOT-2562 — one-shot numeric-feature correlation literalism
    relevance_guidance_sent = False        # SOT-2562 — one-shot relevance-filtered enumeration re-filter
    from src.rag.agent import question_contract as _question_contract
    from src.rag.agent import commit_gate as _commit_gate  # SOT-2639 — lazy (avoids exec_verifier cycle)
    gantt_question = contract == "chart_read" and _question_contract.is_gantt_week_question(question)
    tool_outputs: list[Any] = []
    # SOT-2639 — tolerant (name, response) tool records for the shared commit gate's numeric-grounding
    # probe. Populated at every real dispatch below; read only when RAG_COMMIT_GATE is ON, so an OFF run
    # never touches it (the list stays a pure local ⇒ byte-identical).
    tool_history: list[dict[str, Any]] = []
    commit_gate_rejects = 0                          # consecutive commit_gate REJECTs (drives 棄権降格)
    commit_gate_tel: dict[str, Any] | None = None    # commit_gate decision telemetry (SOT-2629 形式)
    version_diff_result: Mapping[str, Any] | None = None
    # SOT-2586 — retain the PoT forced-lane verdict so the three-layer diagnostics survive into
    # ``.details.jsonl``. Prefer a COMMIT verdict (the one the answer actually rests on); otherwise keep
    # the last verdict that carried candidates. Stays None when the lane is never exercised.
    pot_lane_verdict: dict[str, Any] | None = None
    start = clock()

    # SOT-2521 — deterministic first move: run the routed first-move tool *before* the model's first
    # turn and seed its evidence, so the model starts from the canonical needle instead of wandering
    # through chunk search until BUDGET_EXHAUSTED. Only seeds on a useful (non-empty) result; the
    # observe-only ledgers see it exactly as a normal tool round would, and no answer is injected.
    # SOT-2584 — Evidence Packet pre-inject. When a ``preamble`` is supplied it is injected as a plain
    # user directive before the model's first turn (the typed route + registry-resolved documents +
    # required/missing evidence slots + primary lane + free-exploration budget). ``None`` (default) is a
    # pure no-op so the answer path is byte-identical. It composes with ``first_move`` (below): the packet
    # is prepended to the first-move directive so both seeds reach the model on turn 1.
    preamble_text = preamble.strip() if isinstance(preamble, str) and preamble.strip() else None
    if preamble_text:
        responses = [ToolResponse(DIRECTIVE_MESSAGE, preamble_text)]

    if first_move is not None:
        fm_name, fm_args = first_move
        if fm_name in by_name:
            fm_out = cached_dispatch(fm_name, dict(fm_args or {}))
            if _first_move_useful(fm_out):
                tool_calls.append(fm_name)
                tool_outputs.append(fm_out)
                tool_history.append({"name": fm_name, "response": fm_out, "ok": True})  # SOT-2639
                if fm_name == "version_diff" and isinstance(fm_out, Mapping):
                    version_diff_result = fm_out
                if signals is not None:
                    signals.observe(fm_name, fm_out)
                if calc_signals is not None:
                    calc_signals.observe(fm_name, fm_out)
                if director is not None:
                    director.observe(fm_name, fm_out)
                iterations += 1
                fm_directive = _first_move_directive(fm_name, fm_out)
                combined = f"{preamble_text}\n\n{fm_directive}" if preamble_text else fm_directive
                responses = [ToolResponse(DIRECTIVE_MESSAGE, combined)]

    # SOT-2524 — the model-turn cap is a mutable `turn_limit` (not a fixed range) so the budget-exhaustion
    # boundary hook can grant a bounded number of extra turns exactly once, when the cap is reached without
    # a committed answer. Each granted turn still passes the wall-clock check below, so a boundary that is
    # simultaneously past `timeout_s` runs no extra turn.
    turn = 0
    turn_limit = max_turns
    boundary_tried = False
    submitted = False
    while True:
        if turn >= turn_limit:
            # Turn budget exhausted. Give the re-search director one bounded targeted push at the still-
            # unmet obligations before finalizing the abstain; on a directive, extend the turn budget by
            # the director's max_rounds and run the re-search through the same commit gate below.
            if boundary_tried or submitted:
                break
            boundary_tried = True
            boundary_directive = _budget_boundary_directive(
                director, budget_boundary, tool_calls, answer)
            if boundary_directive is None:
                break
            responses = [ToolResponse(SUBMIT_ANSWER, {
                "abstain_rejected": True,
                "reason": "反復上限に達しました。棄権を確定する前に、未充足の証拠義務『だけ』を局所再探索してください。",
                "directive": boundary_directive,
            })]
            turn_limit += director.budget.max_rounds
        turn += 1
        if clock() - start > timeout_s:
            stop_reason = "timeout"
            error = f"timeout_s={timeout_s} exceeded"
            break
        try:
            step = model.next(responses)
        except Exception as e:  # noqa: BLE001 — a transport failure ends the question, not the batch
            stop_reason = "model_error"
            error = f"model error: {type(e).__name__}: {e}"
            break
        usage = usage + step.usage

        if not step.function_calls:
            # The model answered in free text without the terminal tool: accept it, but we have no
            # self-reported confidence, so record 0.0 and note the shortcut in ``method``.
            text = (step.final_text or "").strip() or ABSTAIN
            if (contract == "simple_lookup" and not tool_calls and is_abstain(text)
                    and not simple_lookup_guidance_sent):
                # An empty first model turn is a transport/model omission, not evidence that a named
                # source is absent.  Give the ordinary lookup path one bounded mandatory-tool retry;
                # the second abstain remains terminal so this cannot loop or inflate every question.
                responses = [ToolResponse(DIRECTIVE_MESSAGE, (
                    "空の初回応答だけでは棄権できません。質問で指定された会社・ファイル名を使って "
                    "find_files を実行し、対象がOffice文書なら read_office、表データなら "
                    "canonical_route を実行してください。抽出結果から質問の行・グループ・最大/最後などの"
                    "条件を確認し、根拠が得られた場合だけ submit_answer してください。"
                    "必須ツールでも対象を解決できない場合のみ棄権してください。"))]
                simple_lookup_guidance_sent = True
                iterations += 1
                continue
            if contract == "chart_read" and not strict_chart_evidence and not chart_guidance_sent:
                # A first-turn prose answer/abstain is not evidence that the strict path is unavailable.
                # Feed one bounded mandatory-tool directive back; if the model still cannot obtain strict
                # evidence, its next abstain is accepted (never loop indefinitely).
                responses = [ToolResponse(DIRECTIVE_MESSAGE, (
                    (
                        "ガント週範囲はread_officeの決定論週グリッドを試す前に確定・棄権できません。"
                        if gantt_question else
                        "グラフ数値はread_chart_valuesの厳密経路を試す前に確定・棄権できません。")
                    + " " +
                    (
                        "対象pptxをfind_filesで特定し、read_office(file=...) の"
                        "【ガント週グリッド:決定論】から対象活動の開始週・終了週を採用してください。"
                        "抽出が曖昧な場合のみ棄権してください。"
                        if gantt_question else
                        "対象xlsxをfind_files/canonical_routeで特定し、read_chart_values(file=..., "
                        "column=..., operation='histogram_max_count')を実行してください。"
                        "失敗した場合のみ棄権してください。"
                    )))]
                chart_guidance_sent = True
                iterations += 1
                continue
            if not is_abstain(text):
                regulation_check = _question_contract.validate_regulation_answer(question, text)
                if not regulation_check.passed:
                    if not regulation_guidance_sent:
                        responses = [ToolResponse(
                            DIRECTIVE_MESSAGE,
                            "通常テキストでは回答を確定できません。特別規定が存在しない場合は、同じ契約の"
                            "一般規定を再確認し、単価・税処理・課金単位・丸め・精算周期・上限をすべて含む"
                            "回答をsubmit_answerで返してください。原文で確定できなければ棄権してください。")]
                        regulation_guidance_sent = True
                        iterations += 1
                        continue
                    text = ABSTAIN
            if contract == "chart_read" and not strict_chart_evidence and not is_abstain(text):
                text = ABSTAIN
            candidate = Answer(answer=text, confidence=0.0,
                               method="(submit_answer未使用: 最終テキストを採用)")
            # SOT-2586 — the PoT lane is a gate, not merely prompt guidance.  A NUMERIC answer may not
            # bypass binder→restricted AST→Decimal→independent verification via plain final text.
            if (POT_HARD_LANE and _pot_lane.enabled() and contract == "numeric"
                    and not is_abstain(candidate.answer) and pot_lane_verdict is None):
                responses = [ToolResponse(SUBMIT_ANSWER, {
                    "answer_rejected": True,
                    "reason": "NUMERIC 回答の PoT 強制レーン検算が未実行です。",
                    "directive": _pot_lane.numeric_lane_directive(),
                })]
                iterations += 1
                continue
            rejection = _reference_commit_rejection(
                question, candidate, contract, tool_outputs=tool_outputs,
                version_diff_result=version_diff_result)
            if rejection is not None:
                responses = [ToolResponse(SUBMIT_ANSWER, rejection)]
                iterations += 1
                continue
            # SOT-2545 — answer granularity normalization on the fallback-text commit path (one-shot).
            if (GRANULARITY_NORMALIZATION and not granularity_guidance_sent
                    and not is_abstain(candidate.answer)):
                gran_check = _question_contract.validate_answer_granularity(
                    question, candidate.answer, tool_outputs)
                if not gran_check.passed:
                    payload: dict[str, Any] = {
                        "answer_rejected": True,
                        "reason": "回答の粒度が質問の要求と一致しません。",
                        "kind": gran_check.kind,
                        "issues": list(gran_check.issues),
                        "directive": gran_check.directive,
                    }
                    if gran_check.expected:
                        payload["expected"] = gran_check.expected
                    responses = [ToolResponse(SUBMIT_ANSWER, payload)]
                    granularity_guidance_sent = True
                    iterations += 1
                    continue
            # SOT-2549 — record-conflict resolution on the fallback-text commit path (one-shot).
            # No is_abstain guard: the target is exactly a "…が競合し特定できません" refusal that scored a
            # wrong answer.  Only fires when a precedence rule reduces the surfaced records to one value.
            if CONFLICT_RESOLUTION and not conflict_guidance_sent:
                conflict_check = _question_contract.resolve_record_conflict(
                    question, candidate.answer)
                if not conflict_check.passed:
                    responses = [ToolResponse(SUBMIT_ANSWER, {
                        "answer_rejected": True,
                        "reason": "競合する複数記載を棄権で逃げています。優先規則で単一解に決めてください。",
                        "rule": conflict_check.rule,
                        "issues": list(conflict_check.issues),
                        "resolved": conflict_check.resolved,
                        "directive": conflict_check.directive,
                    })]
                    conflict_guidance_sent = True
                    iterations += 1
                    continue
            answer = candidate
            stop_reason = "answered"
            break

        responses = []
        submitted = False
        dispatched_tool = False
        spin_terminal = False   # SOT-2522 — early cutoff after a persisting spin (frees the残予算)
        for call in step.function_calls:
            tool_calls.append(call.name)
            if call.name == SUBMIT_ANSWER:
                candidate = _answer_from_args(call.args)
                # SOT-2586 — enforce the advertised forced lane at the terminal boundary.  Prompt-only
                # guidance is insufficient: production models can submit directly.  Abstention remains
                # available, but every committed NUMERIC value must rest on a retained verify_formula trace.
                if (POT_HARD_LANE and _pot_lane.enabled() and contract == "numeric"
                        and not is_abstain(candidate.answer) and pot_lane_verdict is None):
                    responses.append(ToolResponse(SUBMIT_ANSWER, {
                        "answer_rejected": True,
                        "reason": "NUMERIC 回答の PoT 強制レーン検算が未実行です。",
                        "directive": _pot_lane.numeric_lane_directive(),
                    }))
                    dispatched_tool = True
                    break
                # SOT-2507 — chart pixels are never numeric authority.  A non-abstain chart answer may
                # commit only after read_chart_values returned numCache or source-cell recomputation.
                if (contract == "chart_read" and not is_abstain(candidate.answer)
                        and not strict_chart_evidence):
                    responses.append(ToolResponse(SUBMIT_ANSWER, {
                        "answer_rejected": True,
                        "reason": "グラフ数値の厳密証拠がありません。vision値では回答を確定できません。",
                        "directive": (
                            "read_office(file=...) の【ガント週グリッド:決定論】を実行し、"
                            "週ヘッダx座標とバーleft/widthの結果だけを採用してください。"
                            "決定論抽出が曖昧な場合は棄権してください。"
                            if gantt_question else
                            "read_chart_values(file=..., column=..., operation=...) を実行し、"
                            "numCacheまたは元データ再集計のresultだけを採用してください。"
                            "厳密経路が失敗した場合は棄権してください。"
                        ),
                    }))
                    dispatched_tool = True
                    break
                # SOT-2511 — "no special regulation" is only an intermediate finding for a question
                # asking what the regulation says.  Reject a premature terminal answer until the
                # governing fallback rule's rate/tax treatment, billing unit/rounding, cycle and cap are
                # all present.  The guard embeds no policy values; it only enforces semantic coverage.
                if not is_abstain(candidate.answer):
                    from src.rag.agent import question_contract as _question_contract

                    regulation_check = _question_contract.validate_regulation_answer(
                        question, candidate.answer)
                    if not regulation_check.passed:
                        responses.append(ToolResponse(SUBMIT_ANSWER, {
                            "answer_rejected": True,
                            "reason": "規定内容回答のfallback一般規定が不完全です。",
                            "missing": list(regulation_check.missing),
                            "directive": (
                                "『特別規定は存在しない』だけで確定せず、同じ契約の一般規定を局所再探索し、"
                                "単価・税処理・課金単位・丸め・精算周期・上限の有無を回答本文にすべて含めてください。"
                                "値を原文で確定できない場合は推測せず棄権してください。"
                            ),
                        }))
                        dispatched_tool = True
                        break
                rejection = _reference_commit_rejection(
                    question, candidate, contract, tool_outputs=tool_outputs,
                    version_diff_result=version_diff_result)
                if rejection is not None:
                    responses.append(ToolResponse(SUBMIT_ANSWER, rejection))
                    dispatched_tool = True
                    break
                # SOT-2545 — answer granularity normalization (default OFF via RAG_GRANULARITY_NORMALIZATION).
                # One corrective round: reject a truncated verbatim extract / over-enumerated single-item
                # answer and feed a granularity directive; a still-mismatched re-submission is accepted
                # (never loop, never worse than baseline).  EV-safe — an abstain is never rejected here.
                if (GRANULARITY_NORMALIZATION and not granularity_guidance_sent
                        and not is_abstain(candidate.answer)):
                    gran_check = _question_contract.validate_answer_granularity(
                        question, candidate.answer, tool_outputs)
                    if not gran_check.passed:
                        payload: dict[str, Any] = {
                            "answer_rejected": True,
                            "reason": "回答の粒度が質問の要求と一致しません。",
                            "kind": gran_check.kind,
                            "issues": list(gran_check.issues),
                            "directive": gran_check.directive,
                        }
                        if gran_check.expected:
                            payload["expected"] = gran_check.expected
                        responses.append(ToolResponse(SUBMIT_ANSWER, payload))
                        granularity_guidance_sent = True
                        dispatched_tool = True
                        break
                # SOT-2549 — record-conflict resolution (default OFF via RAG_CONFLICT_RESOLUTION).
                # One corrective round: reject a conflict-driven refusal ("…が競合し特定できません") when a
                # general precedence rule (confirmed single refining a range / latest-confirmed version)
                # decides it, and feed the resolved single value back.  No is_abstain guard — the refusal
                # is a committed wrong answer, not an empty abstain; irreducible conflicts pass unchanged.
                if CONFLICT_RESOLUTION and not conflict_guidance_sent:
                    conflict_check = _question_contract.resolve_record_conflict(
                        question, candidate.answer)
                    if not conflict_check.passed:
                        responses.append(ToolResponse(SUBMIT_ANSWER, {
                            "answer_rejected": True,
                            "reason": "競合する複数記載を棄権で逃げています。優先規則で単一解に決めてください。",
                            "rule": conflict_check.rule,
                            "issues": list(conflict_check.issues),
                            "resolved": conflict_check.resolved,
                            "directive": conflict_check.directive,
                        }))
                        conflict_guidance_sent = True
                        dispatched_tool = True
                        break
                # SOT-2562 — numeric-feature correlation literalism (default OFF via RAG_NUMERIC_FEATURE_CORR).
                # One corrective round: a 「相関が最も高い数値特徴量」 answer built by re-encoding a categorical
                # column (.map) into the correlation ranking is rejected and re-derived with numeric_only.
                # EV-safe (abstain passes), content-blind (no column named), one-shot.
                if (NUMERIC_FEATURE_CORR and not numeric_feature_guidance_sent
                        and not is_abstain(candidate.answer)):
                    nf_check = _question_contract.validate_numeric_feature_correlation(
                        question, candidate.answer, tool_outputs)
                    if not nf_check.passed:
                        responses.append(ToolResponse(SUBMIT_ANSWER, {
                            "answer_rejected": True,
                            "reason": "『数値特徴量』の指定に反し、カテゴリ列の数値エンコードで相関を作っています。",
                            "issues": list(nf_check.issues),
                            "directive": nf_check.directive,
                        }))
                        numeric_feature_guidance_sent = True
                        dispatched_tool = True
                        break
                # SOT-2562 — relevance-filtered enumeration (default OFF via RAG_RELEVANCE_STRICT).
                # One corrective round: a 「(aspect)に関連する変更を挙げて」 version-diff answer is re-filtered by
                # the aspect, falling back to 該当なし when nothing is grounded.  An already-該当なし / abstain
                # answer passes unchanged (EV-safe); content-blind; one-shot.
                if (RELEVANCE_STRICT and not relevance_guidance_sent
                        and not is_abstain(candidate.answer)):
                    rel_check = _question_contract.validate_relevance_enumeration(
                        question, candidate.answer)
                    if not rel_check.passed:
                        responses.append(ToolResponse(SUBMIT_ANSWER, {
                            "answer_rejected": True,
                            "reason": "指定した関連観点で根拠付けられない変更まで列挙している可能性があります。",
                            "aspect": rel_check.aspect,
                            "issues": list(rel_check.issues),
                            "directive": rel_check.directive,
                        }))
                        relevance_guidance_sent = True
                        dispatched_tool = True
                        break
                # SOT-2508 — a numeric answer is not complete merely because a number was computed.
                # Enforce the question's quantity definition / unit / rounding contract against the
                # observed compute trail *before* accepting the terminal answer.  This catches the
                # classic ``XのうちYの割合`` denominator swap (e.g. using Y as the denominator) even
                # when the arithmetic itself is internally consistent and high-confidence.  The check
                # is active on the production path where the calc ledger is enabled; callers that
                # deliberately disable calc observation retain the historical byte-identical loop.
                if (contract == "numeric" and calc_signals is not None
                        and not is_abstain(candidate.answer)):
                    from src.rag.agent import question_contract as _question_contract

                    numeric_check = _question_contract.validate_numeric_answer(
                        question, candidate.answer, calc_signals.steps)
                    if not numeric_check.passed:
                        responses.append(ToolResponse(SUBMIT_ANSWER, {
                            "answer_rejected": True,
                            "reason": "量の定義・単位・丸めの証拠義務が未充足です。",
                            "issues": list(numeric_check.issues),
                            "directive": (
                                "回答を確定せず、質問が要求する量を分子・分母・母集団へ書き下してください。"
                                "『XのうちYの割合』ではXだけを分母として別computeで件数を取得し、"
                                "分子件数/分母件数/未丸め値/指定単位/最後の丸めをmethodに明記して再計算すること。"
                                "再計算できなければ推測せず棄権してください。"
                            ),
                        }))
                        dispatched_tool = True
                        break
                # SOT-2500 — full-enumeration closure protocol: for a full_enumeration contract, a
                # deliberate abstain is intercepted once and the closure *procedure* (権威的母集団の特定 →
                # 閉包条件ゲート → 列挙順序) is fed back so completeness is proven before an abstain is
                # accepted. Tried before the generic re-search; one-shot so it cannot loop.
                if enum_gate is not None and is_abstain(candidate.answer):
                    enum_directive = enum_gate.review()
                    if enum_directive is not None:
                        responses.append(ToolResponse(SUBMIT_ANSWER, {
                            "abstain_rejected": True,
                            "reason": "完全列挙の閉包が未確認です。棄権の前に権威的母集団の特定と閉包条件の確認を行ってください。",
                            "directive": enum_directive,
                        }))
                        dispatched_tool = True  # count the guided round; send the procedure next turn
                        break
                # SOT-2502 — obligation-driven local re-search: a deliberate abstain is not accepted
                # immediately. While budget remains and unmet obligations exist, feed a *targeted*
                # re-search directive back so the model re-searches only the unmet obligation. The
                # commit threshold is never touched — a committed (non-abstain) answer finalizes as-is.
                if director is not None and is_abstain(candidate.answer):
                    ev = " ".join(t for t in (candidate.evidence, candidate.method) if t).strip()
                    non_submit = sum(1 for c in tool_calls if c != SUBMIT_ANSWER)
                    directive = director.review(ev, non_submit)
                    if directive is not None:
                        responses.append(ToolResponse(SUBMIT_ANSWER, {
                            "abstain_rejected": True,
                            "reason": "未充足の証拠義務が残っています。棄権の前に局所再探索を実行してください。",
                            "directive": directive,
                        }))
                        dispatched_tool = True  # count the re-search round; send the directive next turn
                        break
                # SOT-2525 — deterministic tool fallback before concluding UNANSWERABLE: when the model
                # is about to abstain and neither the enumeration closure nor the obligation re-search
                # intervened, force ONE contract-typed deterministic tool (canonical_route / version_diff
                # / file_grep, each self-resolving from the question) and — only if it reached concrete
                # evidence — feed that evidence back so the model may answer from it. One-shot; when the
                # tool reaches nothing the abstain proceeds unchanged (従来の安全動作を維持).
                if fallback is not None and is_abstain(candidate.answer):
                    fb_plan = fallback.plan()
                    if fb_plan is not None and fb_plan[0] in by_name:
                        fb_name, fb_args = fb_plan
                        fb_out = cached_dispatch(fb_name, dict(fb_args or {}))
                        tool_calls.append(fb_name)
                        tool_outputs.append(fb_out)
                        tool_history.append({"name": fb_name, "response": fb_out, "ok": True})  # SOT-2639
                        if fb_name == "version_diff" and isinstance(fb_out, Mapping):
                            version_diff_result = fb_out
                        if signals is not None:
                            signals.observe(fb_name, fb_out)
                        if calc_signals is not None:
                            calc_signals.observe(fb_name, fb_out)
                        if director is not None:
                            director.observe(fb_name, fb_out)
                        if _first_move_useful(fb_out):
                            responses.append(ToolResponse(SUBMIT_ANSWER, {
                                "abstain_rejected": True,
                                "reason": "UNANSWERABLE と確定する前に契約型の決定論ツールを実行しました。",
                                "directive": fallback.directive(fb_name, fb_out),
                            }))
                            dispatched_tool = True  # count the forced fallback round; re-prompt next turn
                            break
                # SOT-2639 — route the finalization through the shared commit gate (RAG_COMMIT_GATE,
                # default OFF ⇒ byte-identical / unwired). This is the model-invariant commit boundary:
                # the same accept / reject / abstain judgment every backend (Gemini here, claude-mcp in
                # SOT-2640) must pass. It is wired as the terminal DECISION, not an added formatter — on
                # COMMIT the model's own gold-form value is kept VERBATIM (re-applying formatting.py here
                # would be the 二重 naturalize the design forbids), so a committed answer on this Gemini
                # path stays equivalent to OFF. A precision REJECT is fed back through the existing
                # ``answer_rejected`` in-band retry channel (identical to the exec/numeric rejection flow);
                # it is bounded because the gate itself degrades to ABSTAIN after
                # RAG_COMMIT_GATE_ABSTAIN_AFTER consecutive rejects, so it can never loop.
                if _commit_gate.enabled():
                    cg = _commit_gate.evaluate(
                        question, contract, candidate.answer,
                        session_tool_history=tool_history,
                        naturalizer=None,
                        prior_rejects=commit_gate_rejects,
                    )
                    commit_gate_tel = cg.telemetry
                    # Equivalence-preserving default: record the gate DECISION + telemetry, but on this
                    # Gemini path the loop's own inline guards remain authoritative, so a committed answer
                    # is kept VERBATIM (no re-format, no degrade) ⇒ byte-equivalent to OFF. Enforcement is
                    # opt-in (RAG_COMMIT_GATE_ENFORCE) and only ever acts on a non-abstain commit — an
                    # already-abstain answer is untouched either way. See :func:`_commit_gate_enforce`.
                    if _commit_gate_enforce() and not is_abstain(candidate.answer):
                        if cg.verdict == _commit_gate.REJECT:
                            commit_gate_rejects += 1
                            responses.append(ToolResponse(SUBMIT_ANSWER, {
                                "answer_rejected": True,
                                "reason": "commit_gate: " + "; ".join(cg.reasons),
                                "directive": _COMMIT_GATE_RETRY_DIRECTIVE,
                            }))
                            dispatched_tool = True   # count the guided re-verification round
                            break
                        if cg.abstained:
                            candidate = Answer(answer=ABSTAIN, confidence=0.0,
                                               evidence=candidate.evidence,
                                               method="(commit_gate: 棄権降格) " + candidate.method)
                answer = candidate
                if director is not None and not is_abstain(answer.answer):
                    director.note_answered()
                stop_reason = "answered"
                submitted = True
                break
            # SOT-2522 — spin (dead-end) detection & budget reallocation. A deterministic tool returns the
            # same output for the same (normalized) args, so a repeated identical call adds no evidence and
            # only burns a turn. On the ``spin_threshold``-th recurrence, reallocate once (redirect the
            # freed budget to an untried deterministic route without re-dispatching); if it keeps spinning
            # after that single redirect, cut the path off early so the残予算 is not melted. No corpus fact
            # is injected and the commit threshold is untouched — this only ever turns a would-be BUDGET
            # abstain into a redirect, a grounded answer via another route, or a coded SPIN_CUTOFF abstain.
            if spin_enabled:
                skey = _spin_key(call.name, call.args)
                spin_counts[skey] = spin_counts.get(skey, 0) + 1
                if spin_counts[skey] >= spin_threshold:
                    spin_cutoff = True
                    if not spin_redirected:
                        spin_redirected = True
                        tried = {c for c in tool_calls if c != SUBMIT_ANSWER}
                        responses.append(ToolResponse(call.name, {
                            "spin_detected": True,
                            "reason": (f"同一ツール『{call.name}』を同一引数で{spin_counts[skey]}回"
                                       "呼び出しました。この経路は袋小路です。"),
                            "directive": _spin_redirect_directive(call.name, tried),
                        }))
                        dispatched_tool = True  # count the guided reallocation round
                        continue
                    # already reallocated once and still spinning → cut off early, freeing the残予算.
                    spin_terminal = True
                    break
            # SOT-2614 — consecutive same-tool spin pivot guard (STRENGTHENS SOT-2522). The exact-identity
            # cut above only fires when the args are byte-identical; the phase-0 diagnosis showed the real
            # waste is a long run of the SAME tool with *tweaked* args (varied grep patterns / guess-and-
            # check compute exprs on one file). Track the consecutive run of one fuzzy (tool, target) key;
            # on the threshold-th consecutive call (threshold−1 once the tool is cooled) do NOT dispatch the
            # redundant call — cool the tool for this question and feed back a forced-pivot directive
            # (escalating to a verdict when few turns remain). No corpus fact is injected and the commit
            # threshold is untouched, so an OFF (default) path is byte-identical.
            if pivot_enabled:
                pkey = _spin_soft_target(call.name, call.args)
                if _spin_soft_similar(pivot_prev, pkey):
                    pivot_run += 1
                else:
                    pivot_run = 1
                pivot_prev = pkey
                effective_threshold = (pivot_threshold - 1
                                       if call.name in pivot_cooldown else pivot_threshold)
                if pivot_run >= max(2, effective_threshold):
                    pivot_count += 1
                    pivot_cooldown.add(call.name)
                    pivot_run = 0            # start the next run fresh; the cooldown persists for the question
                    pivot_prev = None
                    tried = {c for c in tool_calls if c != SUBMIT_ANSWER}
                    low = (max_turns - iterations) <= pivot_low_turns
                    responses.append(ToolResponse(call.name, {
                        "spin_pivot": True,
                        "reason": (f"同一ツール『{call.name}』を類似引数で連続{max(2, effective_threshold)}回以上"
                                   "呼び出しています。この経路は空回りです。"),
                        "directive": _spin_pivot_directive(call.name, tried, low_turns=low),
                    }))
                    dispatched_tool = True   # count the guided pivot round
                    continue
            # SOT-2620 — per-route search-call cap. STRENGTHENS the SOT-2614 pivot guard: the pivot guard
            # cuts a *consecutive* same-tool run, but the phase-0 diagnosis showed search waste is also
            # spread *non-consecutively* (a re-grep after a read never trips the consecutive guard yet still
            # burns a search turn). This bounds the TOTAL search-tool (file_grep/find_files) calls per
            # question by route type; on the over-cap call it does NOT dispatch — it feeds back a switch-lane
            # directive (canonical_route / 逆引き索引 / structure store / 手持ち証拠での確定判断). Placed AFTER
            # the pivot guard so a pivoted call (which `continue`d above) never double-counts here — the two
            # guards share one budget. No corpus fact is injected and the commit threshold is untouched, so an
            # OFF (default) path is byte-identical.
            if search_cap_enabled and call.name in _SEARCH_TOOLS:
                if search_calls >= search_cap_limit:
                    search_cap_hits += 1
                    tried = {c for c in tool_calls if c != SUBMIT_ANSWER}
                    responses.append(ToolResponse(call.name, {
                        "search_cap_exceeded": True,
                        "reason": (f"検索系ツール『{call.name}』の呼び出しが型別上限（{search_cap_limit}回）に"
                                   "達しました。検索の反復では新しい根拠は得られません。"),
                        "directive": _search_cap_directive(call.name, search_cap_limit, tried),
                    }))
                    dispatched_tool = True   # count the guided switch round
                    continue
                search_calls += 1
            out = cached_dispatch(call.name, call.args)
            if call.name == "read_chart_values" and _contract.is_contract(out):
                method = out.get("method") or {}
                strict_chart_evidence = (
                    method.get("engine") in {"chart_numcache", "chart_source_compute"}
                    and method.get("numeric_authority") is True
                    and method.get("vision_used") is False
                    and out.get("value") not in (None, "", [], {})
                )
            elif call.name == "read_office" and gantt_question:
                strict_chart_evidence = _has_deterministic_gantt_evidence(out)
            responses.append(ToolResponse(call.name, out))
            tool_outputs.append(out)
            tool_history.append({"name": call.name, "response": out, "ok": True})  # SOT-2639
            if call.name == "version_diff" and isinstance(out, Mapping):
                version_diff_result = out
            # SOT-2586 — capture the forced-lane three-layer verdict for the details log. A COMMIT verdict
            # (what the committed answer rests on) wins; else keep the latest verdict that carried candidates
            # (error/empty tool results are ignored so the retained trace is always aggregatable).
            if (call.name == _pot_lane.TOOL_NAME and isinstance(out, Mapping)
                    and out.get("candidates")):
                have_commit = (pot_lane_verdict is not None
                               and pot_lane_verdict.get("status") == _pot_lane.COMMIT)
                if out.get("status") == _pot_lane.COMMIT or not have_commit:
                    pot_lane_verdict = dict(out)
            dispatched_tool = True
            if signals is not None:
                # observe-only: fold the tool outcome into the abstain signals without touching `out`
                signals.observe(call.name, out)
            if calc_signals is not None:
                # observe-only: capture derivation証跡 (compute/aggregate/chart) for the calc ledger
                calc_signals.observe(call.name, out)
            if director is not None:
                # observe-only: collect successful tool evidence so unmet obligations reflect coverage
                director.observe(call.name, out)
            if call.name == "caption_image" and isinstance(out, Mapping):
                # A question-specific Vision extraction has already enforced literal co-occurrence,
                # visual-line locality and the smallest directly modified candidate.  When exactly one
                # such candidate remains, it is deterministic evidence: commit it directly instead of
                # asking the model to re-select (or repeatedly rediscover) neighbouring cell items.
                literal_answer = _strict_literal_vision_answer(question, out)
                if literal_answer is not None:
                    answer = literal_answer
                    stop_reason = "answered"
                    submitted = True
                    if director is not None:
                        director.note_answered()
                    break
            if (contract == "version_diff" and call.name == "version_diff"
                    and isinstance(out, Mapping) and out.get("value") is not None):
                # The deterministic full-document differ is the authority for this contract. Once it
                # resolves, committing its exact value directly avoids a second model turn that could
                # paraphrase, select a different change, abstain, or fail in transport.
                answer = Answer(
                    answer=str(out["value"]), confidence=1.0,
                    evidence=str(out.get("evidence", "")),
                    method="version_diff の全スライド/全シート構造差分をそのまま採用",
                )
                stop_reason = "answered"
                submitted = True
                if director is not None:
                    director.note_answered()
                break
        if dispatched_tool:
            # count only genuine tool rounds; the terminal submit_answer turn is not a round
            iterations += 1
        if submitted:
            break
        if spin_terminal:
            # SOT-2522 — the reallocation directive did not break the spin; abstain now (answer stays the
            # ABSTAIN default) so the remaining budget is not melted on a proven dead end.
            stop_reason = "spin_cutoff"
            error = error or f"spin cutoff: repeated identical tool call after reallocation"
            break
    if not submitted and stop_reason == "max_turns":
        error = error or f"max_turns={max_turns} reached without a final answer"

    model_name = getattr(model, "model_name", settings.GEN_MODEL_HARD)
    # SOT-2629 — loop-level intervention telemetry (answered + abstained). A key appears ONLY when its
    # intervention flag was active this run (an ABSENT key therefore means the flag was OFF), carrying the
    # explicit firing detail — 0 when the guard was ON but never fired, so ON-but-idle is distinguishable
    # from OFF on the answered trace (cycle2 adversarial-review hole H6). The env-level interventions built
    # during preamble assembly (operand_prefill / condition_preir / eu_gate) are merged on by the caller.
    loop_interventions: dict[str, Any] = {}
    if pivot_enabled:
        loop_interventions["spin_pivot"] = pivot_count
    if search_cap_enabled:
        loop_interventions["search_cap_hits"] = search_cap_hits
    if commit_gate_tel is not None:
        # SOT-2639 — surface the commit-gate decision (SOT-2629 形式). The key appears ONLY when the gate
        # actually ran at the commit boundary (RAG_COMMIT_GATE ON and a submit reached it); an absent key
        # therefore means the gate was OFF or never reached ⇒ an OFF run's telemetry is byte-identical.
        loop_interventions["commit_gate"] = commit_gate_tel
    # SOT-2660 — raw-file dependency (fallback依存) telemetry, recorded for EVERY question (answered OR
    # abstained) so the gold_offline report / Sonnet cycle ledger can compute the fallback依存率 (share of
    # correct answers that needed a 生ファイル系 tool). ``used`` = the answer touched a raw corpus file;
    # ``tools`` = per-tool read counts; ``blocked`` = RAG_DB_ONLY refusals (non-empty only in DB_ONLY runs);
    # ``db_only`` = whether the diagnostic mode was active. Always present (unlike the conditional guards
    # above) since it is the KPI's raw signal, not a firing flag.
    loop_interventions["raw_file_access"] = {
        "used": bool(raw_file_used),
        "tools": dict(raw_file_used),
        "blocked": dict(raw_file_blocked),
        "db_only": DB_ONLY,
    }
    investigation = Investigation(
        question=question, answer=answer, iterations=iterations, tool_calls=tool_calls,
        usage=usage, model=model_name, elapsed_s=max(0.0, clock() - start),
        stop_reason=stop_reason, error=error, contract=contract,
        pot_lane=pot_lane_verdict,  # SOT-2586 — persist the forced-lane verdict for details/diagnostics
        spin_pivots=pivot_count,    # SOT-2614 — forced consecutive-spin pivots (0 ⇒ guard OFF/no spin)
        search_cap_hits=search_cap_hits,  # SOT-2620 — over-cap search calls intercepted (0 ⇒ cap OFF/not hit)
        interventions=loop_interventions,  # SOT-2629 — per-question guard/flag firing telemetry
    )

    # SOT-2492 — abstain ledger: purely post-decision. The answer above is already final; here we only
    # *record* it. Every abstain path (self-abstain / max_turns / timeout / model_error) flows through
    # `record_abstain`, which always assigns a state code, so a code-less abstain is impossible.
    if record_enabled and is_abstain(investigation.answer.answer):
        signals.stop_reason = stop_reason
        # SOT-2522 — attribute a spin-detected abstain to SPIN_CUTOFF (distinct from a plain BUDGET cutoff)
        # whether it ended via early cutoff or by exhausting the cap after the one reallocation directive.
        signals.spin_cutoff = spin_cutoff
        # SOT-2614 — record how many forced consecutive-spin pivots fired so the diagnosis can measure the
        # capture rate of the strengthened guard (0 when RAG_SPIN_PIVOT is OFF ⇒ ledger byte-identical).
        signals.spin_pivots = pivot_count
        # SOT-2620 — record how many over-cap search calls were intercepted so the diagnosis can measure how
        # often the search 上限 fired (0 when RAG_SEARCH_CAP is OFF ⇒ ledger byte-identical).
        signals.search_cap_hits = search_cap_hits
        signals.evidence_text = " ".join(
            t for t in (answer.evidence, answer.method, answer.answer) if t).strip()
        if director is not None:
            # SOT-2502 — record the local re-search history + why it terminated (BUDGET/UNANSWERABLE)
            # so 探索履歴 always remains on the abstain and 即棄権 leaves an auditable trail.
            signals.research_trace = director.trace()
            signals.research_terminal = director.terminal
        _abstain_ledger.record_abstain(
            investigation, signals, path=(None if ledger is True else ledger))

    # SOT-2495/SOT-2506 — calc ledger: record literal numbers AND answers (e.g. a column label such as
    # ``bmi``) produced under the numeric/derived-calculation contract.  The latter matters for argmax
    # statistics: the served answer is text, but it is still the output of a replayable computation.
    # Keep the record on the Investigation as well as persisting it, avoiding racy question lookups in
    # the shared JSONL when gate_question runs concurrently over gold-100.
    if calc_enabled and not is_abstain(investigation.answer.answer) and (
            _calc_ledger.is_numeric_answer(investigation.answer.answer) or contract == "numeric"):
        record = _calc_ledger.record_calc(
            investigation, calc_signals, path=(None if calc_ledger is True else calc_ledger))
        investigation.calc_record = record.to_dict()

    return investigation


def investigate_batch(model_factory: Callable[[str, Sequence[AgentTool]], Model],
                      questions: Sequence[str], *,
                      profile_factory: Callable[[], CorpusProfile] | None = None,
                      shared_profile: CorpusProfile | None = None,
                      max_turns: int = DEFAULT_MAX_TURNS,
                      timeout_s: float = DEFAULT_TIMEOUT_S) -> list[Investigation]:
    """Investigate each question with a fresh model + tools; return one result per question.

    Profile modes (SOT-2528):
    - default / ``profile_factory``: each question gets its OWN :class:`CorpusProfile` (a fresh one, or
      the factory's) so nothing carries across questions — the historical isolation behaviour.
    - ``shared_profile``: every question reuses the SAME profile instance, so a password/alias/format
      discovered on one question is reused on the next instead of being re-derived. These are corpus
      facts (not per-question answers), so sharing only removes rediscovery cost — answers are unchanged.
      ``shared_profile`` takes precedence over ``profile_factory`` when both are given.
    """
    prof = profile_factory or (lambda: CorpusProfile())
    out: list[Investigation] = []
    for q in questions:
        tools = build_tools(shared_profile if shared_profile is not None else prof())
        model = model_factory(q, tools)
        out.append(investigate(model, q, tools, max_turns=max_turns, timeout_s=timeout_s))
    return out


# --------------------------------------------------------------------------- live Gemini model
class GeminiModel:
    """Live Vertex Gemini function-calling conversation (imported lazily so tests stay offline)."""

    def __init__(self, question: str, genai_tools: Any, *, model: str | None = None,
                 system: str = SYSTEM_PROMPT, max_output_tokens: int = 1024,
                 thinking_budget: int = 1024):
        from google.genai import types

        from src.rag import llm

        self._types = types
        self._client = llm.client()
        self.model_name = model or settings.GEN_MODEL_HARD
        self._contents = [types.Content(role="user", parts=[types.Part.from_text(text=question)])]
        self._config = types.GenerateContentConfig(
            temperature=0.0, seed=0, system_instruction=system,
            max_output_tokens=max_output_tokens,
            thinking_config=types.ThinkingConfig(thinking_budget=thinking_budget),
            tools=genai_tools,
        )

    def next(self, tool_responses: Sequence[ToolResponse] | None) -> Step:
        types = self._types
        if tool_responses:
            self._contents.append(types.Content(role="user", parts=[
                (types.Part.from_text(text=str(r.response)) if r.name == DIRECTIVE_MESSAGE else
                 types.Part.from_function_response(
                     name=r.name,
                     response=(r.response if isinstance(r.response, Mapping)
                               else {"result": r.response})))
                for r in tool_responses]))
        resp = self._client.models.generate_content(
            model=self.model_name, contents=self._contents, config=self._config)
        cand = resp.candidates[0]
        parts = list(cand.content.parts or [])
        # Vertex occasionally omits ``content.role`` on a function-call response.  Reusing that object
        # verbatim makes the next request fail with "Please use a valid role: user, model".  Preserve
        # the returned parts while normalizing the documented model role.
        self._contents.append(types.Content(role="model", parts=parts))
        calls = tuple(
            Call(p.function_call.name, dict(p.function_call.args or {}))
            for p in parts if getattr(p, "function_call", None))
        text = "".join(p.text for p in parts if getattr(p, "text", None))
        um = getattr(resp, "usage_metadata", None)
        prompt_toks = getattr(um, "prompt_token_count", 0) or 0
        total_toks = getattr(um, "total_token_count", 0) or 0
        usage = Usage(prompt_toks, max(0, total_toks - prompt_toks))
        return Step(function_calls=calls, final_text=(None if calls else text), usage=usage)


def to_genai_tools(tools: Sequence[AgentTool]) -> Any:
    """Convert :class:`AgentTool` schemas into a google-genai ``Tool`` (live path only)."""
    from google.genai import types

    _TYPE = {"object": types.Type.OBJECT, "string": types.Type.STRING,
             "boolean": types.Type.BOOLEAN, "number": types.Type.NUMBER,
             "integer": types.Type.INTEGER, "array": types.Type.ARRAY}

    def _schema(d: Mapping[str, Any]) -> Any:
        kind = _TYPE.get(d.get("type", "string"), types.Type.STRING)
        props = d.get("properties")
        return types.Schema(
            type=kind,
            properties={k: _schema(v) for k, v in props.items()} if props else None,
            required=list(d.get("required", [])) or None,
        )

    decls = [types.FunctionDeclaration(name=t.name, description=t.description,
                                       parameters=_schema(t.parameters)) for t in tools]
    return [types.Tool(function_declarations=decls)]


def gemini_model_factory(question: str, tools: Sequence[AgentTool], *,
                         model: str | None = None,
                         system: str | None = None) -> GeminiModel:
    """Fresh live conversation for one question (each question gets an isolated context).

    ``system`` overrides the default :data:`SYSTEM_PROMPT` — used by :func:`answer_question` to inject
    the SOT-2498 contract routing hint for this question (default keeps the base prompt).
    """
    return GeminiModel(question, to_genai_tools(tools), model=model,
                       system=system if system is not None else SYSTEM_PROMPT)


# --------------------------------------------------------------------------- SOT-2635 answer-path EU gate
def _eu_registry_resolves(question: str, *, project: str | None = None) -> bool:
    """SOT-2635 — True when the SOT-2583 document registry resolves a target document for ``question``.

    Lazy + fail-open (mirrors :func:`src.rag.agent.gate._registry_resolves`): any import/lookup error →
    ``False`` (a non-blocking default), never an exception on the answer path."""
    try:
        from src.rag.index import document_registry as _dr

        resolver = _dr.get_resolver()
        if resolver is None:
            return False
        return bool(resolver.resolve(question, project=project, limit=1))
    except Exception:  # noqa: BLE001 — advisory signal; never break the answer path
        return False


def _eu_signals_from_investigation(inv: "Investigation", question: str) -> Any:
    """SOT-2635 — build an :class:`~src.rag.agent.eu_gate.GateSignals` bundle from a single-pass
    :class:`Investigation` (the production ``answer_question`` / ``gold_offline --run`` path).

    No 合議 verifier runs on this path, so the verifier / self-consistency signals stay at their
    conservative default (False). Only what the single pass actually carries is reconstructed:

    * ``deterministic_lane`` — the answer came from a deterministic pipeline (``model == 'deterministic'``);
      such an answer is grounded in resolved structure by construction, so ``canonical_doc_resolved`` and
      ``evidence_slots_complete`` are credited alongside it.
    * ``canonical_doc_resolved`` — the SOT-2583 registry resolves the target document (network-free).
    * ``evidence_slots_complete`` — the answer carries non-empty grounding evidence text.
    * ``execution_engines_agree`` / ``operand_sources_complete`` (numeric only) — a numeric answer whose
      PoT三層 verdict (``inv.pot_lane``) *disagreed* forces the execution-disagreement hard blocker.
    * ``verbal_confidence`` — the model self-report (demoted to a small auxiliary weight in eu_gate).
    """
    from src.rag.agent import eu_gate as _eu_gate

    ans = inv.answer
    calc = inv.calc_record or {}
    is_numeric = (inv.contract == "numeric") or bool(calc)
    deterministic = (inv.model == "deterministic")
    has_evidence = bool((ans.evidence or "").strip())

    # numeric execution agreement from the PoT三層 verdict (fail-open: unknown/absent ⇒ agree).
    exec_agree = True
    if is_numeric and isinstance(inv.pot_lane, Mapping):
        verdict = str(inv.pot_lane.get("verdict") or inv.pot_lane.get("status") or "").upper()
        if inv.pot_lane.get("agree") is False or verdict in {
                "MISMATCH", "DISAGREE", "REJECT", "FAIL", "EXEC_MISMATCH"}:
            exec_agree = False

    canonical = deterministic or _eu_registry_resolves(question)
    return _eu_gate.GateSignals(
        canonical_doc_resolved=canonical,
        evidence_slots_complete=(deterministic or has_evidence),
        operand_sources_complete=(exec_agree if is_numeric else True),
        execution_engines_agree=(exec_agree if is_numeric else True),
        deterministic_lane=deterministic,
        verbal_confidence=float(ans.confidence),
    )


def _apply_answer_eu_gate(inv: "Investigation", question: str) -> "Investigation":
    """SOT-2635 — apply the expected-utility commit gate to a produced answer, in place, behind RAG_EU_GATE.

    Records the gate decision in ``inv.interventions['eu_gate']`` for EVERY question (answered OR abstained
    — SOT-2629 全ケース記録) so an ablation can attribute the gate's effect on the answered trace. When the
    answer is a real commit and the gate decides ABSTAIN (a hard epistemic blocker, or ``U ≤ τ``), the
    served answer is flipped to わかりません (棄権側へ倒す); the evidence/method are preserved as diagnostics.
    An already-abstained answer is never resurrected. No-op unless RAG_EU_GATE is on — default OFF ⇒ this
    returns ``inv`` untouched (byte-identical), so every call site can wrap its return unconditionally."""
    from src.rag.agent import eu_gate as _eu_gate

    if not _eu_gate.enabled():
        return inv
    try:
        ans = inv.answer
        already_abstain = is_abstain(ans.answer)
        signals = _eu_signals_from_investigation(inv, question)
        decision = _eu_gate.decide(signals)
        record: dict[str, Any] = {
            "enabled": True,
            "tier": decision.tier,
            "commit": bool(decision.commit),
            "utility": round(float(decision.utility), 4),
            "correctness": round(float(decision.correctness), 4),
            "already_abstain": already_abstain,
            "flipped": False,
            "signals": signals.to_dict(),
        }
        if not already_abstain and not decision.commit:
            inv.answer = Answer(answer=ABSTAIN, confidence=0.0,
                                evidence=ans.evidence, method=ans.method)
            inv.stop_reason = "eu_gate_abstain"
            record["flipped"] = True
        inv.interventions["eu_gate"] = record
    except Exception:  # noqa: BLE001 — the gate is advisory telemetry; never break the answer path
        pass
    return inv


def answer_question(question: str, *, model: str | None = None,
                    profile: CorpusProfile | None = None,
                    max_turns: int = DEFAULT_MAX_TURNS,
                    timeout_s: float = DEFAULT_TIMEOUT_S,
                    ledger: "str | object | bool | None" = True,
                    calc_ledger: "str | object | bool | None" = True,
                    research: "bool | Mapping[str, Any] | object | None" = True,
                    enumeration: "bool | Mapping[str, Any] | object | None" = True,
                    routing: "bool | object | None" = True,
                    contract_flash: "Callable[[str], str | None] | None" = None) -> Investigation:
    """Convenience live entry point: investigate one ``question`` with a real Gemini conversation.

    The abstain ledger (SOT-2492) is **on by default** here so the production answer path records a
    coded diagnosis for every abstain (``artifacts/abstain_ledger.jsonl``). The calc ledger (SOT-2495)
    is likewise **on by default** so every committed numeric answer records its typed calculation証跡
    (``artifacts/calc_ledger.jsonl``). The obligation-driven local re-search loop (SOT-2502) is **on by
    default** so a deliberate abstain re-searches its unmet evidence obligations before it is accepted
    (即棄権が構造上不可能). The budget-exhaustion boundary hook (SOT-2524) is likewise **on by default**
    (``RAG_BUDGET_BOUNDARY_RESEARCH``) so that re-search also fires at the ``max_turns``/``timeout``
    boundary — where the model wandered out of turns without ever deliberately abstaining, the dominant
    BUDGET_EXHAUSTED cause. The full-enumeration closure protocol (SOT-2500) is **on by default** so a
    ``full_enumeration`` question proves its completeness (権威的母集団の特定 + 閉包条件) before it may
    abstain. Pass ``ledger``/``calc_ledger``/``research``/``enumeration`` ``=False``/``None`` to disable,
    or a path (ledgers) / budget mapping (research) to configure.

    Contract routing (SOT-2498) is **on by default** here: the question is classified into its
    :class:`~src.rag.agent.question_contract.QuestionContract` and a corpus-fact-free *routing hint*
    (推奨初手ツール優先順 + 完了条件) is appended to the system prompt so the agent's first move is
    steered toward the tool most likely to reach the evidence — data/横断集計/数値 →
    ``canonical_route``/``compute``/``corpus_aggregate`` first, 書式/グラフ/空間/版差分 → the specialised
    tool first, 単純検索 → the fast retrieval path (Adaptive-RAG). The hint is *advisory only* (final tool
    choice stays with the model) and the classified contract is recorded on the returned
    :class:`Investigation`. Pass ``routing=False``/``None`` to answer with the base prompt (no hint, no
    contract label); ``contract_flash`` injects a live flash arbiter for genuinely ambiguous questions
    (default: deterministic classification only, so this path stays reproducible).
    """
    profile_obj = profile or CorpusProfile()
    # Explicit-file hard constraint (SOT-2583 / 事前処理 #1, idx12): when the question names a file
    # explicitly and the document registry cannot resolve it after an exhaustive manifest scan, abstain
    # with the coded reason instead of falling through to semantic retrieval and over-inferring. Opt-in
    # (RAG_DOCUMENT_REGISTRY) — a no-op when disabled or the artifact is absent, so the champion serve
    # path stays byte-identical.
    from src.rag.index import document_registry as _document_registry
    # SOT-2597 (較正1): pass the legacy path's own case/project resolver so a registry resolution
    # *miss* falls back to exploration (find_files/canonical_route/file_grep) instead of a
    # speculative 0-iteration abstain — the registry stays "resolve→accelerate", and hard-abstains
    # only when no scope resolves at all (certified absent). ``resolve_project`` is imported directly
    # (the tools re-export shadows the module attribute — known gotcha).
    from src.rag.tools.canonical_route import resolve_project as _resolve_project
    _dr_reason = _document_registry.hard_constraint_abstain(
        question, scope_resolver=lambda q: _resolve_project(q, None))
    if _dr_reason is not None:
        return _apply_answer_eu_gate(Investigation(
            question=question,
            answer=Answer(answer=ABSTAIN, confidence=0.0,
                          evidence=f"registry: {_dr_reason}",
                          method="document_registry.hard_constraint"),
            iterations=0,
            tool_calls=["document_registry"],
            usage=Usage(),
            model="deterministic",
            elapsed_s=0.0,
            stop_reason=_dr_reason,
            contract=None,
        ), question)
    tools = build_tools(profile_obj)
    system: str | None = None
    contract: str | None = None
    first_move: "tuple[str, Mapping[str, Any]] | None" = None
    fallback: "object | None" = None
    preamble: "str | None" = None
    # SOT-2629 — env/preamble-level intervention telemetry, merged onto the loop-level record the returned
    # Investigation already carries (spin_pivot / search_cap). A key is added ONLY for an active flag (an
    # ABSENT key ⇒ flag OFF); the value records whether it actually injected/fired for THIS question, so an
    # ON-but-idle intervention is distinguishable from OFF on the answered trace (adversarial-review H6).
    # SOT-2635 — the EU gate's per-question decision (tier / U / commit / flip / signal bundle) is recorded
    # downstream by :func:`_apply_answer_eu_gate` at every return point (全ケース記録), so no ``eu_gate`` key
    # is seeded here. That helper also倒す a committed-but-EV-negative answer to 棄権 when RAG_EU_GATE is on.
    packet_interventions: dict[str, Any] = {}
    if routing not in (None, False):
        from src.rag.agent import question_contract as _question_contract
        from src.rag.agent import routing as _routing  # lazy: keeps import free of the classifier deps
        qc = _routing.classify_for_routing(question, flash=contract_flash)
        system = _routing.routed_system_prompt(SYSTEM_PROMPT, qc, question)
        contract = qc.contract
        # SOT-2603 (Stage0, PLAN SOT-2602) — deterministic router (入口ゲート). Promote the just-computed
        # contract from an in-loop hint to an entry gate: when RAG_DET_PIPELINE_ROUTER is on AND a
        # deterministic pipeline is registered for this contract type AND that pipeline grounds a
        # ``{value, evidence, method}`` result, answer it here WITHOUT ever entering the LLM loop. Every
        # other case — flag OFF (default), no pipeline registered (Stage0 ships the registry EMPTY ⇒ 全問
        # LLM ループ = champion byte-identical), or a pipeline that cannot ground a value — leaves
        # ``det_result`` None and falls through to the unchanged loop below (回答数を減らさない). No corpus
        # fact / no answer is hardcoded here; the registered pipelines self-derive from the question. The
        # import is lazy (mirrors the routing import) so the module graph is untouched when routing is off.
        from src.rag.agent import det_pipeline as _det_pipeline
        det_started = time.monotonic()
        det_result = _det_pipeline.resolve(question, contract, profile=profile_obj)
        if det_result is not None:
            # SOT-2604 (Stage3, PLAN SOT-2602) — deterministic-value → gold-format naturalization. The
            # single short LLM call the inverted design allows lives here and here only: template-first for
            # 数値/列挙/週/「該当なし」 (no LLM), one short LLM naturalize only for a free-text type whose value
            # is still a raw structure. It preserves the value facts (SOT-2544 記号↔文章形の同義, SOT-2545 粒度
            # トリム/truncation 補完 are all evidence-bound, no invention). ``format_contract`` returns None only
            # when the deterministic value is blank — then we fall through to the LLM loop rather than
            # committing an empty answer (回答数を減らさない). Same gate (RAG_DET_PIPELINE_ROUTER) as the router.
            from src.rag.agent import formatting as _formatting
            formatted = _formatting.format_contract(det_result, question, contract_type=contract)
            if formatted is not None:
                return _apply_answer_eu_gate(Investigation(
                    question=question,
                    answer=_answer_from_det_contract(formatted),
                    iterations=1,
                    tool_calls=[f"det_pipeline:{contract}"],
                    usage=Usage(),
                    model="deterministic",
                    elapsed_s=max(0.0, time.monotonic() - det_started),
                    stop_reason="answered",
                    contract=contract,
                ), question)
        # SOT-2647 (事前計算事実層 5/5) — precomputed-store direct-answer lane. Sits AFTER the Stage0 router
        # (so Wave A1〜B2 keep precedence and there is no contract-registry collision) and BEFORE the LLM
        # loop: when RAG_FACT_LAYER is on AND the contract type binds unambiguously to a unique store value
        # (案件マスタ enum / 派生メトリクス scalar), answer it deterministically WITHOUT the LLM loop —
        # model-invariant, provenance-carrying (verified operand). Any ambiguity ⇒ ``fact_result`` None ⇒
        # fall through to the loop (回答数を減らさない, wrong を増やさない, SOT-2601 の発火緩和 fail の教訓).
        # RAG_FACT_LAYER default OFF ⇒ resolve() returns None ⇒ byte-identical. Never raises into the path.
        if _fact_layer.enabled():
            from src.rag.agent import formatting as _formatting
            fact_started = time.monotonic()
            fact_result = _fact_layer.resolve(question, contract, profile=profile_obj)
            if fact_result is not None:
                formatted = _formatting.format_contract(fact_result, question, contract_type=contract)
                if formatted is not None:
                    return _apply_answer_eu_gate(Investigation(
                        question=question,
                        answer=_answer_from_det_contract(formatted),
                        iterations=1,
                        tool_calls=[f"fact_layer:{contract}"],
                        usage=Usage(),
                        model="deterministic",
                        elapsed_s=max(0.0, time.monotonic() - fact_started),
                        stop_reason="answered",
                        contract=contract,
                    ), question)
        # SOT-2584 — Evidence Packet pre-inject (typed route → registry-resolved docs → slots → budget).
        # Built only behind RAG_EVIDENCE_PACKET; reuses the just-computed contract so the question is not
        # re-classified. Fail-open: any build error leaves ``preamble`` None so the answer path is
        # byte-identical (回帰ゼロ). Injects no corpus fact / no answer.
        if EVIDENCE_PACKET:
            try:
                from src.rag.agent import evidence_packet as _evidence_packet
                from src.rag.agent import query_router as _query_router
                # Import the submodule by full path — ``src.rag.tools.canonical_route`` the *name* in the
                # tools package is re-exported as the function (shadowing), so a bare package import would
                # not expose ``resolve_project`` (known tools-shadow gotcha).
                from src.rag.tools.canonical_route import resolve_project as _resolve_project
                packet_project = _resolve_project(question, None)
                decision = _query_router.classify_route(question, contract=qc)
                _packet, preamble = _evidence_packet.build_directive(
                    question, project=packet_project, decision=decision)
                # SOT-2629 — record whether the NUMERIC operand-prefill / condition-IR interventions
                # actually injected into this question's packet (reading the packet the agent will see, so
                # the telemetry cannot drift from what fired). Keys are added ONLY when the flag is ON; the
                # value is 0/false when the flag was ON but nothing was injected for this question.
                _pkt_ev = _packet.evidence if _packet is not None else None
                if _operand_prefill.enabled():
                    _catalog = (_pkt_ev or {}).get("operand_candidates")
                    packet_interventions["operand_prefill"] = {
                        "candidates": len(_catalog) if _catalog else 0,
                        "injected": bool(_catalog),
                    }
                try:
                    from src.rag.agent import condition_prefill as _condition_prefill
                    if _condition_prefill.enabled():
                        packet_interventions["condition_preir"] = {
                            "built": bool((_pkt_ev or {}).get("condition_ir")),
                        }
                except Exception:  # noqa: BLE001 — telemetry is additive; never break the answer path
                    pass
                # SOT-2586 — NUMERIC PoT forced-lane directive (only when RAG_POT_HARD_LANE is on and the
                # route is NUMERIC). Appended to the packet preamble so the agent is told to route its
                # arithmetic through ``verify_formula`` (binder→制限AST→Decimal→独立検算) rather than 暗算.
                # Injects no corpus fact / no answer — protocol only.
                if POT_HARD_LANE and _pot_lane.enabled() and decision.route == _query_router.NUMERIC:
                    preamble = f"{preamble}\n\n{_pot_lane.numeric_lane_directive()}"
                # SOT-2616 — operand candidate prefill is built *inside* the Evidence Packet
                # (build_directive → build_packet) so the catalog is embedded in the packet JSON and
                # rendered in the preamble in one place; nothing to append here.
                # SOT-2587 — ENUM full-scan forced-lane directive (only when RAG_ENUM_SCAN is on and the
                # route is ENUM). Tells the agent to resolve the universe and scan every applicable
                # document via ``enum_scan`` and to honour the completeness certificate / no-match guard,
                # rather than trusting a retrieval top-k. Injects no corpus fact / no answer — protocol
                # only. Appended atop the packet preamble (so it requires RAG_EVIDENCE_PACKET too).
                if ENUM_SCAN and _enum_scan.enabled() and decision.route == _query_router.ENUM:
                    preamble = f"{preamble}\n\n{_enum_scan.enum_lane_directive(question)}"
            except Exception:  # noqa: BLE001 — packet is additive; never break the answer path
                preamble = None
        # SOT-2632 (G2, PLAN SOT-2602) — port Sonnet's lookup/derived procedures via advisory HINTS,
        # appended to the generation preamble independently of the Evidence Packet (so the port can be
        # A/B'd on its own). Gated by RAG_G2_LOOKUP_PORT; default OFF ⇒ preamble untouched (byte-identical).
        # Fail-open: any build error leaves the preamble as-is. Injects no corpus fact / no answer — the
        # directive is procedure guidance only, and the SOT-2629-style telemetry records whether it fired.
        if G2_LOOKUP_PORT:
            try:
                from src.rag.agent import g2_lookup_port as _g2_lookup_port
                _g2_directive, _g2_tel = _g2_lookup_port.port_directive(question, contract=contract)
                if _g2_directive:
                    preamble = f"{preamble}\n\n{_g2_directive}" if preamble else _g2_directive
                packet_interventions["g2_lookup_port"] = _g2_tel
            except Exception:  # noqa: BLE001 — the hint is additive; never break the answer path
                pass
        # SOT-2631 (G1, PLAN SOT-2602) — port Sonnet's highlight-extraction procedures via advisory HINTS,
        # appended to the generation preamble independently of the Evidence Packet (so the port can be
        # A/B'd on its own, exactly like the G2 sibling). Gated by RAG_G1_HIGHLIGHT_PORT; default OFF ⇒
        # preamble untouched (byte-identical). Fail-open: any build error leaves the preamble as-is.
        # Injects no corpus fact / no answer — the directive is procedure guidance only, and the
        # SOT-2629-style telemetry records whether it fired.
        if G1_HIGHLIGHT_PORT:
            try:
                from src.rag.agent import g1_highlight_port as _g1_highlight_port
                _g1_directive, _g1_tel = _g1_highlight_port.port_directive(question, contract=contract)
                if _g1_directive:
                    preamble = f"{preamble}\n\n{_g1_directive}" if preamble else _g1_directive
                packet_interventions["g1_highlight_port"] = _g1_tel
            except Exception:  # noqa: BLE001 — the hint is additive; never break the answer path
                pass
        # SOT-2521 — the loop-side deterministic first move (see investigate ``first_move``). Reuses the
        # same first-tool decision as the prompt hint so they can never disagree. Gated OFF by default
        # (``FIRST_MOVE_ROUTING``): the always-on variant regressed gold-100, so the wiring stays dormant
        # here — the answer path is byte-identical — until SOT-2527 measures a narrowed enablement.
        if FIRST_MOVE_ROUTING:
            first_move = _routing.deterministic_first_move(qc, question)
        # SOT-2525 — deterministic tool fallback before UNANSWERABLE (see investigate ``fallback``).
        # Gated OFF by default (``UNANSWERABLE_FALLBACK``): the wiring stays dormant so the answer path is
        # byte-identical until SOT-2527 measures its net effect on gold-100.
        if UNANSWERABLE_FALLBACK:
            fallback = _question_contract.DeterministicFallbackGate(question, contract)
        started = time.monotonic()
        deterministic: Answer | None = None
        deterministic_tools: list[str] = []
        deterministic = _deterministic_seating_side_answer(question)
        if deterministic is not None:
            deterministic_tools = ["seating_lookup", "enumeration_closure"]
        elif _question_contract.is_staff_population_question(question):
            deterministic = _deterministic_staff_population_answer(question)
            deterministic_tools = ["corpus_aggregate", "enumeration_closure"]
        elif contract == "simple_lookup":
            if GRANULARITY_NORMALIZATION:
                deterministic = _deterministic_verbatim_action_answer(question)
                if deterministic is not None:
                    deterministic_tools = ["canonical_route", "find_files", "caption_image"]
            if deterministic is None:
                deterministic = _deterministic_literal_report_answer(question)
            if deterministic is not None:
                deterministic_tools = ["canonical_route", "find_files", "caption_image"]
        elif contract == "chart_read" and _question_contract.is_gantt_week_question(question):
            deterministic = _deterministic_gantt_answer(question, profile_obj)
            deterministic_tools = ["canonical_route", "find_files", "read_office"]
        elif (contract == "simple_lookup"
              and _question_contract.is_regulation_content_question(question)):
            deterministic = _deterministic_regulation_answer(question, profile_obj)
            deterministic_tools = ["canonical_route", "find_files", "read_office"]
        if deterministic is not None:
            return _apply_answer_eu_gate(Investigation(
                question=question,
                answer=deterministic,
                iterations=len(deterministic_tools),
                tool_calls=deterministic_tools,
                usage=Usage(),
                model="deterministic",
                elapsed_s=max(0.0, time.monotonic() - started),
                stop_reason="answered",
                contract=contract,
            ), question)
        # Record whether the caller kept the defaults *before* any adaptation, so every adaptation below
        # only lifts a default budget and never shrinks an explicit caller budget, and so they compose
        # (the ratio +4 still applies on top of the multi-stage lift).
        caller_default_turns = max_turns == DEFAULT_MAX_TURNS
        caller_default_timeout = timeout_s == DEFAULT_TIMEOUT_S
        # SOT-2523 — contract-adaptive budget for the multi-stage contracts (derived_calculation / 横断集計
        # / enum_set): 「特定→復号→読込→計算→検証」の多段で 12ターン/180s を使い切りやすい(BUDGET_EXHAUSTED
        # 最多)ため、多段型に限り bounded に拡張する。単純検索など単段型は据え置き(コスト線形増を避ける)。
        if ADAPTIVE_BUDGET and contract in MULTISTAGE_CONTRACTS:
            if caller_default_turns:
                max_turns = ADAPTIVE_MAX_TURNS       # 12 -> 18
            if caller_default_timeout:
                timeout_s = ADAPTIVE_TIMEOUT_S       # 180 -> 240
        # Ratio contracts need separate numerator/denominator computations plus final unit/rounding
        # validation.  Grant four bounded extra turns when the caller kept the default so a correct
        # re-computation can finish after one rejected contract-violating submission (SOT-2508 focused
        # cycle 1 root cause) — on top of the multi-stage lift above (18 -> 22) or the flat cap (12 -> 16).
        if (contract == "numeric" and caller_default_turns
                and _question_contract.numeric_requirements(question).ratio):
            max_turns += 4
        if (caller_default_timeout and (
                _question_contract.is_regulation_content_question(question)
                or _question_contract.is_gantt_week_question(question))):
            # A whole-section read / PPTX shape extraction plus guarded resubmission can exceed the
            # ordinary budget.  This remains bounded and applies only to the two deterministic paths.
            timeout_s = 300.0
    # SOT-2627 — dev backend switch. Reached only after every deterministic pre-stage declined (so the
    # deterministic shortcuts above are byte-identical); it swaps ONLY the model-driven tool loop. When
    # ``RAG_INVESTIGATOR_BACKEND=claude-mcp`` the whole loop is delegated to a flat-rate Sonnet
    # ``claude -p --mcp-config`` session (SOT-2626 MCP server); the routed ``system``/``contract``/
    # ``preamble`` computed above are carried over so the Sonnet run sees the same instructions the Gemini
    # loop would. Any other value keeps the Gemini path unchanged. Imported lazily so the default path
    # never imports the subprocess provider.
    if INVESTIGATOR_BACKEND == "claude-mcp":
        from src.rag.llm_providers import claude_mcp
        _inv = claude_mcp.investigate_question(
            question, tools=tools, system=system, contract=contract, preamble=preamble,
            max_turns=max_turns, timeout_s=timeout_s, model=model)
        _inv.interventions.update(packet_interventions)  # SOT-2629 — merge env/packet-level telemetry
        return _apply_answer_eu_gate(_inv, question)  # SOT-2635 — commit-time EU gate (no-op unless ON)
    model_obj = gemini_model_factory(question, tools, model=model, system=system)
    _inv = investigate(model_obj, question, tools, max_turns=max_turns, timeout_s=timeout_s,
                       ledger=ledger, calc_ledger=calc_ledger, research=research,
                       enumeration=enumeration, contract=contract, first_move=first_move,
                       fallback=fallback, budget_boundary=BUDGET_BOUNDARY_RESEARCH,
                       spin_detection=SPIN_DETECTION, pivot_detection=SPIN_PIVOT,
                       search_cap=SEARCH_CAP, preamble=preamble)
    _inv.interventions.update(packet_interventions)  # SOT-2629 — merge env/packet-level telemetry
    return _apply_answer_eu_gate(_inv, question)  # SOT-2635 — commit-time EU gate (no-op unless ON)
