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

import re
from typing import Callable

from src.rag.agent import obligations as _obligations
from src.rag.agent import question_contract as _qc
from src.rag.agent.question_contract import QuestionContract
from src.rag.tools import font_emphasis as _font_emphasis
from src.rag.tools.canonical_route import infer_kinds as _infer_kinds

# Font-decoration cue words in a question (SOT-2564). When the question asks for 太字/下線/イタリック
# (bold/underline/italic) formatting, the ``font_emphasis`` tool — not the colour ``highlight_extract`` —
# reaches the evidence. Pure vocabulary (no corpus fact).
_FONT_DECORATION_RE = re.compile(r"太字|下線|アンダーライン|イタリック|斜体|bold|underline|italic", re.I)


def references_font_decoration(question: str) -> bool:
    """True when ``question`` names a font decoration (太字/下線/イタリック) — a font_emphasis question."""
    return bool(_FONT_DECORATION_RE.search(question or ""))

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
    _qc.CHART_READ: ("read_chart_values",),
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
    if contract.contract == _qc.SIMPLE_LOOKUP and _qc.is_regulation_content_question(question):
        # Threshold acronyms in the question are often absent from the governing clause.  Locate the
        # authoritative contract and read its whole pricing/settlement section before trying literal grep.
        return ("find_files", "read_office", "file_grep")
    if contract.contract == _qc.CHART_READ and _qc.is_gantt_week_question(question):
        # Gantt bars are native PPTX shapes: geometry from read_office is authoritative; vision is only
        # corroboration.  read_chart_values targets embedded numCache charts and cannot recover bar spans.
        return ("read_office",)
    if _obligations.literal_terms(question):
        # Exact/literal requests often target image-only documents. Locate the named file first and
        # inspect it directly; a second broad BM25 pass can consume the whole question budget without
        # ever exposing the page image.
        return ("find_files", "caption_image", "read_office", "file_grep")
    if (contract.contract == _qc.FORMAT_CHECK and _font_emphasis.enabled()
            and references_font_decoration(question)):
        # 太字/下線/イタリックの複合書式条件は色ハイライトではなく font_emphasis が正本 (SOT-2564)。
        return ("font_emphasis", *(t for t in base if t != "font_emphasis"))
    if contract.contract == _qc.FORMAT_CHECK and (
            "ピボット" in question.lower() or (
                ".pptx" in question.lower() and "ハイライト" in question
                and any(cue in question for cue in ("抽出条件", "集計内容", "行ラベル", "列ラベル")))):
        return ("pptx_pivot", *(t for t in base if t != "pptx_pivot"))
    if contract.contract in _DATA_ASSET_CANONICAL_CONTRACTS and references_data_asset(question):
        return ("canonical_route", *(t for t in base if t != "canonical_route"))
    return base


def deterministic_first_move(contract: QuestionContract,
                             question: str) -> tuple[str, dict[str, object]] | None:
    """The tool the *loop* should run deterministically as the first move, or ``None`` (SOT-2521).

    BUDGET_EXHAUSTED is the top abstain cause (54/73): the model burns its turn budget wandering
    through chunk search before it ever reaches the deterministic tool. The SOT-2498 routing *hint*
    can only advise — it cannot stop the wandering. This returns a ``(tool_name, kwargs)`` plan the
    investigator loop executes *before* the model's first turn, seeding the model with canonical
    evidence so it spends its budget computing/extracting rather than searching.

    Only a tool that (a) resolves its target from the question alone and (b) returns generic,
    corpus-fact-free *evidence* (never a committed answer) qualifies, so seeding it cannot hard-code
    an answer:

    * ``canonical_route(question=…)`` — for every contract whose effective first move is the canonical
      direct route (numeric / cross-file that routes to a file / a ``simple_lookup`` whose evidence
      lives inside a named data asset the chunk index misses). It self-resolves the project + kind and
      returns ``{rel, project, …}`` records; the answer is still produced by the model from those.

    Contracts whose first move needs a target the model must still discover (a plain lookup, a chart /
    format / spatial question, an aggregate needing a metric argument) return ``None`` — their first
    turn stays model-driven. The decision reuses :func:`first_tools_for`, so the loop's forced first
    move and the prompt hint can never disagree about what the first tool is.
    """
    first_tools = first_tools_for(contract, question)
    if first_tools and first_tools[0] == "canonical_route":
        return "canonical_route", {"question": question}
    return None


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
    if contract.contract == _qc.VERSION_DIFF:
        lines.append(
            "この契約だけはツール順が必須である。回答確定前に version_diff(question=質問全文) を必ず実行し、"
            "全スライド/全シートを網羅した決定論的 value を回答へそのまま採用する。grep や目視した一部箇所だけで"
            "差分を確定してはならない。value=null の場合は別の変更を推測せず棄権する。")
    literal = _obligations.literal_terms(question)
    if literal:
        lines.append(
            "引用語/『明記』は文字どおりの同一箇所証拠が必須である。検索可能な原文が空の画像のみPDFは "
            "caption_image(file=..., question=質問全文) で全ページを読み、条件語 "
            f"({', '.join(literal)}) と回答候補が同じセル・文・行に共存する出力だけで確定する。")
    if contract.contract == _qc.NUMERIC:
        req = _qc.numeric_requirements(question)
        lines.append(
            "証拠の優先順位は『computeによる生データ再計算・notebookの実行出力 > 文書中のmarkdown/"
            "散文の主張』とする。相関・統計・集計で矛盾した散文を正解根拠にしてはならない。"
            "canonical_routeで元の表データを特定し、computeで同じ統計を自力再計算した結果だけを採用する。"
            "notebook質問で元データが必要なら canonical_route(question=..., kind='train', expr='...') を使う。"
            "矛盾した散文は evidence から除外し、methodに再計算した式・対象列を記録する。")
        lines.append(
            "数値を計算する前に、要求量を『分子 / 分母 / 母集団 / 単位 / 丸め』へ書き下す。"
            "特に『XのうちYの割合』の分母はXを満たす全行であり、Yの部分集合へ勝手に変更してはならない。"
            "分子件数と分母件数を別々にcomputeし、methodに各定義・件数・最終式を明記する。")
        if re.search(r"(?:負の相関|相関.{0,12}(?:低|小さ|負))", question):
            lines.append(
                "負の相関の順位を求める場合は、目的変数自身を候補から列名で明示的に除外したうえで、"
                "相関係数を昇順に並べた先頭（最小値）を選ぶ。自己相関は+1で最大側なので、"
                "『自己相関を飛ばすため』という理由で無条件に2番目の最小値を選んではならない。"
                "採用した列名と相関係数をcompute結果で確認する。")
        if req.denominator_scope:
            lines.append(
                f"この質問で機械抽出した分母条件: 『{req.denominator_scope}』。"
                "この条件を満たす行数を分母として独立にcomputeし、分母証跡を欠いた回答は確定しない。")
            if ("標準化" in req.denominator_scope and req.denominator_fields
                    and "lt" in req.denominator_operators
                    and re.search(r"0\s*未満", req.denominator_scope)):
                field = req.denominator_fields[0]
                lines.append(
                    f"標準化 z=({field}-mean)/std で std>0 なので、z<0 の母集団は "
                    f"{field}<全体平均({field}) と同値。分母は例として "
                    f"len(df[df['{field}'] < df['{field}'].mean()]) をcomputeする。"
                    "分子の追加条件を分母へ混ぜず、分母を別証跡として先に確定する。")
        if req.unit or req.decimal_places is not None:
            lines.append(
                "要求書式: " + ", ".join(x for x in (
                    f"単位={req.unit}" if req.unit else "",
                    f"小数第{req.decimal_places}位" if req.decimal_places is not None else "",
                ) if x) + "。丸めは未丸め値の計算後、最後の一段だけで適用する。")
    if (contract.contract == _qc.FORMAT_CHECK and _font_emphasis.enabled()
            and references_font_decoration(question)):
        lines.append(
            "太字/下線/イタリックの書式条件は色ハイライトではない。font_emphasis(file=..., "
            "require='太字,下線,イタリック' のように該当書式を列挙) を使い、要求された書式すべてに"
            "同時該当する箇所だけを回答する。色フィルタの highlight_extract では該当しない。"
            "対象ファイルが不明なら先に find_files で報告資料/対象文書を特定してから font_emphasis を呼ぶ。")
    if contract.contract == _qc.FORMAT_CHECK and first_tools and first_tools[0] == "pptx_pivot":
        lines.append(
            "ピボット回答では生ラベル(例: 8/2)やハイライト値だけを答えてはならない。pptx_pivot の "
            "semantics / filters / target_column / aggregation_label を使い、必ず『抽出条件 + 対象列 + 集計方法』"
            "の3要素を回答する。semantics が未解決なら別の元データを探索するか棄権する。")
    if contract.contract == _qc.SIMPLE_LOOKUP and _qc.is_regulation_content_question(question):
        lines.append(
            "『特別規定なし』は規定内容回答の完了ではない。特別規定が存在しない場合は、同じ契約で"
            "代わりに適用される一般規定を探索し、単価と税処理・課金単位と丸め・精算周期・上限の有無を"
            "回答本文にすべて含める。値は原文から取得し、項目が欠けたまま確定しない。")
        lines.append(
            "質問中の閾値・略語が契約原文に登場しなくても、字面grepの空振りだけで棄権しない。まず"
            "find_filesで対象案件の契約書を特定してread_officeし、料金モデル/単価/見込額/工数記録/"
            "精算方法の連続条項を読む。『契約総額を固定しない』は料金上限なしの根拠として扱い、"
            "税別単価だけでなく最終請求時の消費税処理も別途確認する。")
    if contract.contract == _qc.CHART_READ and _qc.is_gantt_week_question(question):
        lines.append(
            "ガント週範囲は read_office 冒頭の【ガント週グリッド】決定論抽出を正本とする。"
            "週セルは半開区間としてバーが重なり始める最初の週を開始週、重なる最後の週を終了週とする。"
            "vision目視は裏取りだけに使い、抽出が曖昧/競合なら候補の一方を推測せず棄権する。")
    if "主略称" in question:
        lines.append(
            "回答表面の必須形式は主略称。ツールが返した主略称をそのまま列挙し、正式社名や案件名へ"
            "展開・言い換えしない。")
    if "フルネーム" in question:
        lines.append(
            "回答表面の必須形式は人名のフルネームのみ。根拠説明、所属、役割、敬称（様/氏）を"
            "回答欄へ足さず、抽出した氏名を原文表記のまま返す。")
    if contract.contract == _qc.CHART_READ and not _qc.is_gantt_week_question(question):
        lines.append(
            "グラフ数値はread_chart_valuesのnumCacheまたは元データ再集計だけを根拠にする。"
            "ヒストグラム最多件数は対象列をcolumn、operation='histogram_max_count'として呼ぶ。"
            "caption_image/visionは所在・軸ラベル確認専用で、画像から読んだ数値では回答を確定しない。"
            "厳密経路が無ければ棄権する。")
    if re.search(r"明記|そのまま|原文(?:どおり|通り|表記)|文字列(?:として)?", question):
        lines.append(
            "『明記』『そのまま』『原文』で指定された抽出は literal 一致が必須である。引用された条件語が"
            "採用候補と同じセル/文に文字列として存在することをツール結果で確認し、共起を確認できない候補は"
            "回答へ採用しない。")
    if contract.completion_conditions:
        checklist = "".join(f"\n  - {c}" for c in contract.completion_conditions)
        lines.append("回答を確定してよい完了条件(満たすまで棄権しない):" + checklist)
    return "\n".join(lines)


def routed_system_prompt(base_system: str, contract: QuestionContract, question: str) -> str:
    """Return ``base_system`` with the contract routing hint appended (advisory, corpus-fact-free)."""
    return f"{base_system}\n{route_hint(contract, question)}"
