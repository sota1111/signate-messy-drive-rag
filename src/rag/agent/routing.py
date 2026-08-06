"""SOT-2498 — contract-based routing: wire the question-contract classifier into the investigator.

Parent SOT-2460. The classifier (:mod:`src.rag.agent.question_contract`, SOT-2493) labels a question
with one of nine **contracts**; the canonical direct route (:mod:`src.rag.tools.canonical_route`,
SOT-2494) resolves data/計算系 questions straight to their canonical file (bypassing chunk retrieval).
This module joins the two: it turns a classified contract into a **routing hint** that is appended to
the investigator's system prompt so the agent's *first move* is steered toward the tool most likely to
reach the evidence — data/横断集計/数値 → ``canonical_route`` / ``compute`` / ``corpus_aggregate``
first, 書式/グラフ/空間/版差分 → the specialised tool first, 単純検索 → the fast retrieval path
(Adaptive-RAG: no uniform high-cost search on every question).

This is the fix for the retrieval_miss 22 (SOT-2486): the needle is at rank 29–720, so *routing* the
question to the canonical file — not more chunk search — is what reaches it.

Design invariants
-----------------
* **Hint, not mandate.** The injected text is an ordered *priority list*; the final tool choice stays
  with the Gemini agent (過剰な決定論分岐でエージェント性を殺さない). The agent may deviate when a tool
  errors / returns empty, exactly as before.
* **No corpus fact injected.** The hint names only *tool names* and the contract's generic
  completion-condition checklist — never a password / 略称 / member list — so the portability invariant
  (移植性の担保) of the investigator prompt is preserved.
* **Pure / offline.** :func:`classify_for_routing` runs the deterministic classifier by default (a
  ``flash`` arbiter may be injected for the live path); building the hint is a pure string operation
  with no network or filesystem access.
"""
from __future__ import annotations

from typing import Callable

from src.rag.agent import question_contract as _qc
from src.rag.agent.question_contract import QuestionContract
from src.rag.tools.canonical_route import infer_kinds as _infer_kinds

# --------------------------------------------------------------------------- contract → first-move tools
# Recommended first-move tool priority per contract. Only the *ordering* of already-exposed investigator
# tools — no new capability. Data/計算系 contracts lead with the canonical direct route so a needle the
# chunk index cannot surface (retrieval_miss) is reached by routing, not by more search.
CONTRACT_FIRST_TOOLS: dict[str, tuple[str, ...]] = {
    _qc.SIMPLE_LOOKUP: ("file_grep", "find_files", "read_office"),
    _qc.MULTI_HOP: ("file_grep", "find_files", "canonical_route", "seating_lookup"),
    _qc.CROSS_AGGREGATE: ("corpus_aggregate", "canonical_route"),
    _qc.FULL_ENUMERATION: ("file_grep", "find_files", "corpus_aggregate"),
    _qc.FORMAT_CHECK: ("highlight_extract", "pdf_emphasis", "read_office"),
    _qc.CHART_READ: ("read_chart_values", "caption_image"),
    _qc.SPATIAL: ("seating_lookup",),
    _qc.VERSION_DIFF: ("version_diff",),
    _qc.NUMERIC: ("canonical_route", "compute"),
}

# Contracts whose evidence is typically NOT in the chunk index (data assets / cross-file aggregation) —
# for these the hint explicitly tells the agent to prefer the canonical direct route / compute over a
# chunk search that would miss the needle. This is the Adaptive-RAG discriminator.
_CANONICAL_FIRST = frozenset({_qc.NUMERIC, _qc.CROSS_AGGREGATE})

# The retrieval_miss core (SOT-2486): some questions read a single authoritative record that happens to
# live inside a *canonical data asset* (train.xlsx の1セル / modeling.py のハイパーパラメータ /
# leaderboard の1行) whose chunk BM25 rank is 29〜720 — so the contract is ``simple_lookup`` but the
# fast retrieval path misses the needle. When the question names such a data asset (deterministic,
# corpus-fact-free :func:`~src.rag.tools.canonical_route.infer_kinds`) we lead the ``simple_lookup`` hint
# with ``canonical_route`` so the file is reached by *routing*, not by more search. Contracts with their
# own dedicated tool (spatial/chart/format/version_diff) or a cross-file aggregator (cross_aggregate /
# multi_hop) are left as-is: their first move is already correct, and a multi-project "最も…な案件" has no
# single canonical file to route to.
_DATA_ASSET_CANONICAL_CONTRACTS = frozenset({_qc.SIMPLE_LOOKUP})


def classify_for_routing(question: str, *,
                         flash: Callable[[str], str | None] | None = None) -> QuestionContract:
    """Classify ``question`` into its :class:`QuestionContract` for routing (deterministic-first).

    Thin pass-through to :func:`src.rag.agent.question_contract.classify`; ``flash`` is consulted only
    when the deterministic layer is inconclusive (pass :func:`question_contract.flash_classify` on the
    live path, omit for a fully deterministic, network-free classification).
    """
    return _qc.classify(question, flash=flash)


def references_data_asset(question: str) -> bool:
    """True when ``question`` names a canonical data asset (train/code/notebook/leaderboard/…).

    Reuses the exact deterministic, corpus-fact-free matcher the ``canonical_route`` tool itself uses
    (:func:`~src.rag.tools.canonical_route.infer_kinds`), so the routing hint and the tool agree on what
    counts as a data-asset question.
    """
    return bool(_infer_kinds(question or ""))


def first_tools_for(contract: QuestionContract, question: str) -> tuple[str, ...]:
    """The effective ordered first-move tools for ``contract`` on ``question``.

    Starts from the static :data:`CONTRACT_FIRST_TOOLS` list, then applies the data-asset override: a
    ``simple_lookup`` whose evidence lives inside a named canonical data asset leads with
    ``canonical_route`` (the retrieval_miss fix) instead of the fast chunk-search path.
    """
    base = CONTRACT_FIRST_TOOLS.get(contract.contract, ())
    if contract.contract in _DATA_ASSET_CANONICAL_CONTRACTS and references_data_asset(question):
        return ("canonical_route", *(t for t in base if t != "canonical_route"))
    return base


def route_hint(contract: QuestionContract, question: str) -> str:
    """Build the system-prompt fragment that steers the first move for ``contract`` (advisory).

    The fragment states the classified contract, an ordered *推奨初手ツール* list, whether to prefer the
    canonical direct route over chunk search, and the contract's completion-condition checklist (so the
    agent knows what must hold before it may commit). It injects no corpus-specific fact.
    """
    first_tools = first_tools_for(contract, question)
    canonical_first = contract.contract in _CANONICAL_FIRST or (
        first_tools and first_tools[0] == "canonical_route")
    lines = [
        "───── 契約型ルーティング (SOT-2498 / Adaptive-RAG ヒント) ─────",
        f"この質問の契約型: {contract.label}（{contract.contract}）。"
        "以下は推奨する初手であり最終判断ではない。ツール選択は自分の判断で行い、"
        "空振り/エラー時は従来どおり別経路へ切り替える。",
    ]
    if first_tools:
        lines.append("推奨初手ツール(優先順): " + " → ".join(first_tools) + "。")
    if canonical_first:
        lines.append(
            "この質問はデータ資産/横断集計が根拠で chunk 検索では上位に上がりにくい。まず "
            "canonical_route / compute / corpus_aggregate で canonical ファイルへ直行し、"
            "一律の高コスト検索を先に行わない。")
    if contract.completion_conditions:
        checklist = "".join(f"\n  - {c}" for c in contract.completion_conditions)
        lines.append("回答を確定してよい完了条件(満たすまで棄権しない):" + checklist)
    return "\n".join(lines)


def routed_system_prompt(base_system: str, contract: QuestionContract, question: str) -> str:
    """Return ``base_system`` with the contract routing hint appended (advisory, corpus-fact-free)."""
    return f"{base_system}\n{route_hint(contract, question)}"
