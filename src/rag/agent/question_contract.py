"""SOT-2493 — question-contract classifier (Adaptive-RAG routing の前段).

Parent SOT-2460. Before a question is turned into a retrieval query, this module first classifies it
into a **question contract** — the *kind of promise* the answer must keep — and hands back the
contract's **completion-condition template** and a suggested processing route. Downstream Issues wire
the contract into the investigator loop (SOT-2498) and use the completion conditions as evidence
obligations (SOT-2499) / enumeration closure gates (SOT-2500); **this Issue only adds the classifier —
it does not touch the production answer path.**

The nine contracts
------------------
``simple_lookup`` (単純検索) · ``multi_hop`` (多段関係) · ``cross_aggregate`` (横断集計) ·
``full_enumeration`` (完全列挙) · ``format_check`` (書式判定) · ``chart_read`` (グラフ読取) ·
``spatial`` (空間推論) · ``version_diff`` (版差分) · ``numeric`` (数値推論).

Hybrid, deterministic-first
---------------------------
:func:`classify` runs a **deterministic** layer first (keyword / regex rules + glossary-abbrev aware
:func:`src.rag.archetype.classify` as the coarse backbone) and only falls back to ``gemini-2.5-flash``
when the question is genuinely ambiguous *and* a ``flash`` callable is supplied. The flash call is
injected (never imported at module load, never invoked on a confident deterministic hit), so the whole
classifier — including the accuracy measurement — runs network-free and reproducibly.

Relation to the archetype taxonomy
----------------------------------
``src.rag.archetype.classify`` already labels each question with a fine-grained *archetype*
(``fact_lookup`` / ``document_extract`` / ``derived_calculation`` / ``enum_set`` / ``version_diff`` …),
and the gold-100 ``archetype`` column (``artifacts/gold_100_review.csv``) is produced by exactly that
function (``scoring/gold_offline.py``). The contract taxonomy is a **routing refinement** on top of it:
several contracts (``spatial`` / ``chart_read`` / ``format_check``) are strictly more specific than the
archetype they refine (a seating question is a specialised ``fact_lookup``; a chart-value question is a
``fact_lookup`` / ``derived_calculation``; a bold/underline extraction is a ``document_extract``). So
each contract declares the set of archetypes it is *consistent* with (:data:`CONTRACT_ARCHETYPES`), and
:func:`agreement_rate` measures how often the predicted contract is consistent with the gold archetype
— the acceptance metric (≥90%), with the specificity gains recorded as principled refinements rather
than counted as errors.
"""
from __future__ import annotations

import ast
import re
from dataclasses import dataclass, field
from typing import Any, Callable, Mapping, Sequence

from config import settings
from src.rag import archetype as _archetype
from src.rag.corpus import nfc

# --------------------------------------------------------------------------- the nine contracts
SIMPLE_LOOKUP = "simple_lookup"        # 単純検索: 単一の権威的記載を引く
MULTI_HOP = "multi_hop"                # 多段関係: 中間エンティティを段階的に辿る
CROSS_AGGREGATE = "cross_aggregate"    # 横断集計: 複数文書/行を横断して集計する
FULL_ENUMERATION = "full_enumeration"  # 完全列挙: 母集団を確定し漏れなく挙げる
FORMAT_CHECK = "format_check"          # 書式判定: 太字/下線/色などの書式属性で抽出する
CHART_READ = "chart_read"              # グラフ読取: 図/グラフから数値を読み取る
SPATIAL = "spatial"                    # 空間推論: 座席表など空間関係を解く
VERSION_DIFF = "version_diff"          # 版差分: 旧版と新版の実質的変更を出す
NUMERIC = "numeric"                    # 数値推論: 型付き入力から式で導出する

CONTRACTS: tuple[str, ...] = (
    SIMPLE_LOOKUP, MULTI_HOP, CROSS_AGGREGATE, FULL_ENUMERATION, FORMAT_CHECK,
    CHART_READ, SPATIAL, VERSION_DIFF, NUMERIC,
)

# Human-readable JP label per contract (for reports / prompts).
CONTRACT_LABELS: dict[str, str] = {
    SIMPLE_LOOKUP: "単純検索", MULTI_HOP: "多段関係", CROSS_AGGREGATE: "横断集計",
    FULL_ENUMERATION: "完全列挙", FORMAT_CHECK: "書式判定", CHART_READ: "グラフ読取",
    SPATIAL: "空間推論", VERSION_DIFF: "版差分", NUMERIC: "数値推論",
}

# --------------------------------------------------------------------------- completion-condition templates
# Per-contract "what must be true before an answer may be committed" — the routing payload downstream
# Issues (SOT-2498/2499/2500) consume. Kept as ordered tuples so they read as a checklist.
CONTRACT_COMPLETION: dict[str, tuple[str, ...]] = {
    SIMPLE_LOOKUP: (
        "回答値が単一の権威的文書に明示されている",
        "値の出典(ファイル/スライド/セル)を一意に特定できる",
    ),
    MULTI_HOP: (
        "中間エンティティ(担当者/案件/期間など)を一段ずつ根拠付きで確定する",
        "各ホップの根拠を保持し、最終値までの連鎖が途切れない",
    ),
    CROSS_AGGREGATE: (
        "集計対象の母集団(全プロジェクト/全ファイル)を権威的に確定する",
        "集計関数(合計/平均/最大/件数)と対象列が型整合している",
        "決定論的に再計算できる(corpus_aggregate 等で再現可能)",
    ),
    FULL_ENUMERATION: (
        "列挙対象の母集団(全体集合)を権威的に確定する",
        "閉包条件を満たす(見落とし 0・重複 0 を保証できる)",
    ),
    FORMAT_CHECK: (
        "判定対象の書式属性(太字/下線/色/フォント)を原本から機械抽出する",
        "該当箇所を網羅し、非該当(例:日付)を除外できる",
        "表/ピボットの場合は生の行・列ラベルを元データのフィールド名へ意味解決し、抽出条件・対象列・集計方法を確定する",
    ),
    CHART_READ: (
        "対象グラフ(図番号/シート/系列)を一意に特定する",
        "numCacheを使い、無い場合は対応する元データ列から決定論的に再集計する",
        "visionは所在・軸ラベル確認に限定し、画像から読んだ数値を根拠にしない",
        "読取値の桁数/単位が質問の要求と一致する",
    ),
    SPATIAL: (
        "座席表/レイアウトの空間関係(隣/向かい/座席番号)を確定する",
        "人物⇔座席⇔内線などの対応を一意に解決できる",
    ),
    VERSION_DIFF: (
        "旧版と新版の対応ペアを一意に確定する",
        "diffpair(version_diffツール)で全スライド/全シートを比較し、隣接版間の実質的な変更のみを抽出する",
        "version_diffツールの決定論結果を得るまで回答を確定せず、解決不能なら推測せず棄権する",
    ),
    NUMERIC: (
        "型付き入力(数値/単位/対象列)を根拠から確定する",
        "質問が要求する量を分子・分母・母集団まで定義し、『〜のうち』の直前にある条件を分母から落とさない",
        "再実行可能な式で決定論的に導出できる",
        "計算結果の単位と丸め(小数第N位など)が質問の要求と一致する",
    ),
}

# Suggested processing route (tool / path hint) per contract — advisory only until SOT-2498 wires it.
CONTRACT_ROUTE: dict[str, str] = {
    SIMPLE_LOOKUP: "retrieve→extract",
    MULTI_HOP: "obligation-driven iterative retrieve",
    CROSS_AGGREGATE: "corpus_aggregate",
    FULL_ENUMERATION: "enumeration-closure",
    FORMAT_CHECK: "pdf_emphasis/highlight_extract",
    CHART_READ: "read_chart_values",
    SPATIAL: "seating_lookup",
    VERSION_DIFF: "version_diff",
    NUMERIC: "compute",
}

# Archetypes each contract is *consistent* with (a contract may refine, i.e. be strictly more specific
# than, the coarse archetype). Used by :func:`agreement_rate` to score the classifier against the gold
# archetype column without penalising a legitimate specificity gain. ``unknown`` is always compatible.
CONTRACT_ARCHETYPES: dict[str, frozenset[str]] = {
    SIMPLE_LOOKUP: frozenset({
        "fact_lookup", "document_extract", "glossary_formal", "glossary_abbrev",
        "config_model_type", "config_hyperparam", "metric_score",
    }),
    MULTI_HOP: frozenset({"fact_lookup", "derived_calculation"}),
    CROSS_AGGREGATE: frozenset({"cross_aggregate", "pivot_condition", "derived_calculation"}),
    FULL_ENUMERATION: frozenset({"enum_set", "highlight_set"}),
    FORMAT_CHECK: frozenset({"document_extract", "highlight_set", "fact_lookup"}),
    CHART_READ: frozenset({"fact_lookup", "derived_calculation", "data_shape"}),
    SPATIAL: frozenset({"fact_lookup"}),
    VERSION_DIFF: frozenset({"version_diff"}),
    NUMERIC: frozenset({
        "derived_calculation", "data_shape", "csv_column_mean", "csv_column_max",
        "contract_amount", "cross_aggregate",
    }),
}

# --------------------------------------------------------------------------- deterministic rules
# New-dimension detectors — the contracts that are strictly more specific than any archetype the coarse
# backbone can express. Kept tight so they fire ONLY on genuine chart/spatial/format questions.
# Geometric seating cues (隣/向かい/座席): unambiguously spatial, checked before multi-hop.
_SPATIAL_GEOM_RE = re.compile(
    r"座席表|座席|着席|席次|座って|隣に座|の隣の|左隣|右隣|向かい(に座|の席|の方|の人)|"
    r"レイアウト図|配置図")
# A bare extension/EXT directory lookup (氏名⇔EXT) also routes through the seating directory, but only
# when the question is NOT a multi-hop ("最も…な人の内線") — those are resolved as multi_hop first.
_SPATIAL_DIR_RE = re.compile(r"内線番号を教え|内線を教え|ext を教え|ext を答え", re.I)
_CHART_RE = re.compile(
    r"グラフ\s*\d|棒グラフ|円グラフ|折れ線|折線|散布図|ヒストグラム|チャート|プロット|"
    r"のグラフ(で|の|に)|グラフ(で|の|に).{0,20}(値|カウント|ビン|系列|軸)|"
    r"可視化したもの|ビンの範囲")
_GANTT_RE = re.compile(
    r"ガント|Gantt|(?:実行|実施|作業|予定).{0,16}(?:スケジュール|第\s*\d+\s*週|何週)|"
    r"(?:スケジュール|工程表).{0,24}(?:第何週|何週目|実行予定|実施予定)",
    re.I)
_FORMAT_RE = re.compile(
    r"太字|ボールド|下線|アンダーライン|イタリック|斜体|取り消し線|"
    r"フォント|文字色|背景色|セルの色|塗りつぶし|罫線|ハイライト(?:色|され|した|して)|"
    r"マーカー(の色|で)|ピボット|pivot",
    re.I)
# A multi-hop question threads one entity into the lookup of another ("最も…な人/案件 の …").
_MULTIHOP_RE = re.compile(
    r"(もっとも|最も|一番|最大|最高|最多).{0,24}(人|担当者|案件|案件名|プロジェクト|会社|社|ファイル)"
    r".{0,20}(の|が担当).{0,16}(内線|ext|担当|案件|金額|着手金|報酬|期間|日付|番号|値)")
_STAFF_POPULATION_RE = re.compile(
    r"(?:各案件|全案件|全プロジェクト).{0,40}"
    r"(?:PP|提案書).{0,16}(?:契約書|契約).{0,16}(?:PLAN|計画|スケジュール).{0,16}"
    r"(?:FR|最終報告).{0,48}(?:役割付き|実施体制|担当体制).{0,32}(?:何人|人数|全部で)",
    re.I,
)

# Archetypes the backbone maps to the NUMERIC contract (derivation / computed values).
_NUMERIC_ARCH = frozenset({
    "derived_calculation", "data_shape", "csv_column_mean", "csv_column_max", "contract_amount",
})


def _base_contract(arch: str) -> str:
    """Map the coarse archetype to its base contract (before new-dimension refinement)."""
    if arch == "version_diff":
        return VERSION_DIFF
    if arch in ("enum_set", "highlight_set"):
        return FULL_ENUMERATION
    if arch in ("cross_aggregate", "pivot_condition"):
        return CROSS_AGGREGATE
    if arch in _NUMERIC_ARCH:
        return NUMERIC
    return SIMPLE_LOOKUP  # fact_lookup / document_extract / glossary_* / config_* / metric_score / unknown


# Deterministic confidence: specific structural/new-dimension hits are trusted; a plain simple_lookup
# fallback from an ``unknown`` archetype is the least certain (candidate for flash arbitration).
_CONF_SPECIFIC = 0.9
_CONF_BASE = 0.75
_CONF_WEAK = 0.5


@dataclass(frozen=True)
class QuestionContract:
    """The classification result: the routing contract plus its completion-condition payload."""

    contract: str
    label: str
    archetype: str                       # coarse backbone archetype (src.rag.archetype.classify)
    completion_conditions: tuple[str, ...]
    route: str
    method: str                          # "deterministic" | "flash"
    confidence: float
    evidence: str = ""                   # short why-note
    consistent_archetypes: frozenset[str] = field(default_factory=frozenset)

    def is_consistent_with(self, gold_archetype: str) -> bool:
        """True when ``gold_archetype`` is one this contract may legitimately refine (or unknown)."""
        return gold_archetype == "unknown" or gold_archetype in self.consistent_archetypes

    def to_dict(self) -> dict[str, object]:
        return {
            "contract": self.contract,
            "label": self.label,
            "archetype": self.archetype,
            "completion_conditions": list(self.completion_conditions),
            "route": self.route,
            "method": self.method,
            "confidence": self.confidence,
            "evidence": self.evidence,
        }


# --------------------------------------------------------------------------- regulation-content completeness
# A negative answer to a "what does the regulation say?" question is not complete when a fallback
# general rule governs the case.  These helpers define the *shape* of the required answer without
# embedding any corpus-specific amount or policy value.
_REGULATION_CONTENT_RE = re.compile(
    r"(?:規定|規約|契約条件).{0,24}(?:内容|詳細|精算方法|取扱い)|"
    r"(?:精算方法|取扱い).{0,24}(?:規定|規約|契約条件)")
_NO_SPECIAL_REGULATION_RE = re.compile(
    r"(?:特別(?:な)?(?:精算)?(?:規定|方法|取扱い)|該当(?:する)?規定|そのような規定|規定).{0,18}"
    r"(?:存在し(?:ない|ません)|あり(?:ません|得ない)|設け(?:ていない|られていない)|"
    r"規定されていません|定め(?:がない|られていない|られていません))|"
    r"(?:存在し(?:ない|ません)|あり(?:ません|得ない)).{0,18}(?:特別(?:な)?(?:精算)?規定|該当(?:する)?規定)|"
    # SOT-2548: a "not found"/"no record" phrasing of the same negative finding — e.g. idx78's
    # 「ACTHに関する記述が見つかりませんでした」/「直接の記述はありません」 — must trip the completeness
    # guard just like 「存在しない」. Covers the 記載/記述/条項 subjects the original list omitted, across
    # both the 見つからない and あり/存在しない verb families.
    r"(?:規定|規約|契約条件|条項|記載|記述|定め|該当).{0,20}"
    r"(?:見つか(?:らない|りません|らず|らなかった)|見当たら(?:ない|ず|ません|なかった)|"
    r"確認でき(?:ない|ず|ません|なかった)|あり(?:ません|得ない)|"
    r"存在し(?:ない|ません)|記載(?:がない|されていない|されていません))")
_REGULATION_FALLBACK_COMPLETION = (
    "特別規定が存在しない場合も、代わりに適用される一般規定の要点(単価と税処理・課金単位と丸め・精算周期・上限)をすべて確定する",
)
_GANTT_COMPLETION = (
    "週ヘッダ列のx座標でグリッドを較正し、バーleft/widthとの半開区間の重なりから開始週・終了週を決定論的に確定する",
    "週境界の候補が競合する場合は目視で推測せず、決定論的根拠が得られるまで回答を確定しない",
)


def is_regulation_content_question(question: str) -> bool:
    """Whether the question asks for the contents/handling of a rule, not merely its existence."""
    return bool(_REGULATION_CONTENT_RE.search(nfc(question or "")))


def is_gantt_week_question(question: str) -> bool:
    """Whether a question asks for an activity's week range on a schedule/Gantt visual."""
    return bool(_GANTT_RE.search(nfc(question or "")))


def is_staff_population_question(question: str) -> bool:
    """Whether a question asks for the unique DA roster across PP/contract/PLAN/FR for all projects."""
    return bool(_STAFF_POPULATION_RE.search(nfc(question or "")))


# --------------------------------------------------------------------------- deterministic fallback (SOT-2525)
# Contracts whose evidence is a canonical data asset the chunk index misses (retrieval_miss): the
# canonical direct route self-resolves the project + kind from the question alone, so it is a safe,
# corpus-fact-free tool to force once before concluding UNANSWERABLE.
_CANONICAL_FALLBACK_CONTRACTS = frozenset({NUMERIC, CROSS_AGGREGATE, MULTI_HOP})


def deterministic_fallback_plan(
        question: str, contract: "str | None") -> "tuple[str, dict[str, object]] | None":
    """The one deterministic, self-resolving tool to try before UNANSWERABLE, or ``None`` (SOT-2525).

    The UNANSWERABLE bucket (12/73 abstains) is dominated by questions whose gold answer *does* exist in
    a canonical data asset / version pair — the system gave up before ever routing to the deterministic
    tool. This returns a ``(tool_name, kwargs)`` plan for exactly one such tool, chosen from the contract
    so it (a) resolves its target from the question text alone and (b) returns corpus-fact-free
    *evidence*, never a committed answer:

    * ``version_diff(question=…)`` for a ``version_diff`` contract — the deterministic full-document
      differ self-resolves the version pair.
    * ``canonical_route(question=…)`` for numeric / cross-file / multi-hop contracts, or any question
      that names a canonical data asset (:func:`~src.rag.agent.routing.references_data_asset`) — it
      self-resolves the project + kind and returns ``{rel, project, …}`` records.
    * ``file_grep(query=<literal term>)`` as a last resort when the question carries an exact quoted /
      named term — a deterministic full-corpus grep for that literal.

    Returns ``None`` when no deterministic route self-resolves from the question (the abstain then stands
    unchanged). The chosen tool only ever *seeds evidence*; the answer is still produced by the model
    subject to every existing commit guard, so a wrong answer is never injected (EV-safe).
    """
    from src.rag.agent import obligations as _obligations  # lazy: avoid import cycle
    from src.rag.agent import routing as _routing          # lazy: avoid import cycle

    if contract == VERSION_DIFF:
        return "version_diff", {"question": question}
    if contract in _CANONICAL_FALLBACK_CONTRACTS or _routing.references_data_asset(question):
        return "canonical_route", {"question": question}
    literal = _obligations.literal_terms(question)
    if literal:
        return "file_grep", {"query": literal[0]}
    return None


class DeterministicFallbackGate:
    """Force one contract-typed deterministic tool before an UNANSWERABLE abstain is accepted (SOT-2525).

    The investigator constructs the gate per question (only when :data:`RAG_UNANSWERABLE_FALLBACK` is on)
    and, when the model is about to abstain and neither the enumeration-closure nor the obligation
    re-search loop intervened, calls :meth:`plan` exactly once to get the deterministic tool to run. The
    gate is one-shot: after :meth:`plan` has returned (whether or not the tool reached evidence) it
    returns ``None`` forever, so a still-abstaining model cannot loop on it. It injects no corpus fact and
    no answer — :meth:`directive` only carries the tool's own output back to the model.
    """

    def __init__(self, question: str, contract: "str | None") -> None:
        self.question = question
        self.contract = contract
        self._tried = False

    def plan(self) -> "tuple[str, dict[str, object]] | None":
        """The one deterministic ``(tool_name, kwargs)`` to try, or ``None`` (one-shot)."""
        if self._tried:
            return None
        self._tried = True
        return deterministic_fallback_plan(self.question, self.contract)

    def directive(self, tool_name: str, out: object) -> str:
        """The user-message feeding the deterministic tool's evidence back before an abstain."""
        import json

        try:
            payload = json.dumps(out, ensure_ascii=False, default=str)[:6000]
        except (TypeError, ValueError):
            payload = str(out)[:6000]
        return (
            f"UNANSWERABLE と確定する前に、契約型の決定論ツール『{tool_name}』を質問に対して実行しました"
            "（棄権前の必須フォールバック）。\n"
            f"実行結果(JSON): {payload}\n"
            "この結果に質問の根拠が含まれるなら、compute / read_office / read_chart_values 等で確定し "
            "submit_answer で回答してください。根拠が含まれない・対象が解決できない場合のみ、従来どおり"
            "棄権してください（推測して回答してはならない）。")


def completion_conditions(question: str, contract: str) -> tuple[str, ...]:
    """Return the static contract checklist plus question-specific completion promises."""
    base = CONTRACT_COMPLETION[contract]
    if contract == SIMPLE_LOOKUP and _VERBATIM_EXTRACT_RE.search(nfc(question or "")):
        return (*base,
                "『そのまま/抜き出す』質問では、対象IDや見出しだけでなく該当セル・行・項目の本文全体を省略せず返す",
                "探索手順・謝罪・推測を回答にせず、原文を確定できない場合だけ棄権する")
    if contract == SIMPLE_LOOKUP and is_regulation_content_question(question):
        return (*base, *_REGULATION_FALLBACK_COMPLETION)
    if contract == CHART_READ and is_gantt_week_question(question):
        # Native-shape Gantt bars have no numCache/source column.  Keep target identification, then
        # apply the dedicated week-grid geometry promises instead of contradictory numeric-chart ones.
        return (base[0], *_GANTT_COMPLETION)
    return base


@dataclass(frozen=True)
class RegulationValidation:
    """Result of checking a negative special-rule answer for its fallback-rule details."""

    passed: bool
    applicable: bool = False
    missing: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, object]:
        return {"passed": self.passed, "applicable": self.applicable, "missing": list(self.missing)}


def validate_regulation_answer(question: str, answer: str) -> RegulationValidation:
    """Require rate/unit/rounding/cycle/cap when a content answer says no special rule exists.

    Values are intentionally not prescribed: the validator only checks that the answer carries each
    semantic field.  The actual values must still come from retrieved source evidence.
    """
    if not is_regulation_content_question(question):
        return RegulationValidation(True)
    text = nfc(answer or "")
    if not _NO_SPECIAL_REGULATION_RE.search(text):
        return RegulationValidation(True)

    checks = (
        ("単価", re.compile(r"(?:時間|工数)?単価|(?:円|ドル)\s*(?:[/／]\s*)?(?:時間|時|h\b)", re.I)),
        ("税処理", re.compile(r"(?:消費税|税).{0,10}(?:加算|乗じ|含め|込み)|税込")),
        ("課金単位", re.compile(r"\d+\s*(?:分|時間)\s*単位|(?:分|時間)ごと")),
        ("丸め", re.compile(r"切り?上げ|切上げ|切り?捨て|切捨て|四捨五入|丸め")),
        ("精算周期", re.compile(r"月次|毎月|月ごと|月単位|週次|毎週|日次|一括精算")),
        ("上限", re.compile(r"上限|限度|キャップ", re.I)),
    )
    missing = tuple(label for label, pattern in checks if not pattern.search(text))
    return RegulationValidation(not missing, True, missing)


# --------------------------------------------------------------------------- answer granularity contract
# SOT-2545 — two document_extract failure shapes (誤答A2) where the evidence was reached but the committed
# answer's *granularity* diverged from the question's promise:
#   (1) truncation of a verbatim ("そのまま抜き出す") extract — only the head/title is returned while the
#       source item carries a fuller body (idx93: 「前処理パイプライン」 vs the full action content);
#   (2) over-enumeration of a single-item ask — a 「第N週の項目は何ですか」 singular question is answered
#       with every task in a computed date range instead of the single designated item (idx88).
# The validator is deterministic and content-blind: it embeds no answer values, only detecting the two
# shapes from the question wording + the committed answer (+ tool evidence for the truncation target).
_VERBATIM_EXTRACT_RE = re.compile(r"そのまま|抜き出|原文|全文|一字一句|逐語")
# A verbatim extract that asks for the *content* (内容/本文/詳細/説明) of an item promises the whole body,
# so a short title-only answer is a truncation even when no fuller fragment has been retrieved yet.
_CONTENT_EXTRACT_RE = re.compile(r"内容|本文|詳細|説明|記載事項|全文")
# Enumeration cues make a question genuinely multi-item; they suppress the over-enumeration guard.
_ENUMERATION_CUE_RE = re.compile(r"すべて|全て|全部|列挙|挙げて|一覧|それぞれ|各(?:項目|案件|人|々)|漏れなく|網羅|複数")
# A single-item selector pins the question to one unit (第N週/N番目/…): a list answer then over-enumerates.
_SINGLE_SELECTOR_RE = re.compile(
    r"第\s*\d+\s*(?:週|回|章|項|条|フェーズ|ステップ|段階|日)目?|\d+\s*(?:番目|つ目|個目|件目)")
# List separators between parallel items — NOT 「・」/「／」, which are usually intra-item connectors
# (e.g. a compound phase name joined by 「・」 is one item and must not be split).
_ITEM_SEP_RE = re.compile(r"[、,，;；\n]+")
_META_RESPONSE_RE = re.compile(
    r"(?:i(?:'m| am) sorry|mistake|previous step|reasoning|should be looking|tool (?:output|call)|"
    r"申し訳|誤り|前のステップ|推論|ツール(?:出力|呼び出し))",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class GranularityValidation:
    """Result of the answer-granularity check (SOT-2545)."""

    passed: bool
    kind: str = ""                    # "" | "truncation" | "over_enumeration"
    issues: tuple[str, ...] = ()
    expected: str = ""                # deterministic full-text target when a fuller fragment is in evidence
    directive: str = ""

    def to_dict(self) -> dict[str, object]:
        return {"passed": self.passed, "kind": self.kind, "issues": list(self.issues),
                "expected": self.expected, "directive": self.directive}


def _granularity_fragments(tool_outputs: Sequence[Any]) -> list[str]:
    """Flatten dispatched tool outputs to string fragments (self-contained; no obligations import)."""
    frags: list[str] = []

    def walk(v: Any) -> None:
        if v is None or isinstance(v, bool):
            return
        if isinstance(v, Mapping):
            for nested in v.values():
                walk(nested)
        elif isinstance(v, (list, tuple, set)):
            for item in v:
                walk(item)
        else:
            s = str(v).strip()
            if s:
                frags.append(s)

    for out in tool_outputs or ():
        walk(out)
    return frags


def validate_answer_granularity(question: str, answer: str,
                                tool_outputs: Sequence[Any] = ()) -> GranularityValidation:
    """Detect a truncated verbatim extract or an over-enumerated single-item answer (SOT-2545).

    Content-blind: no answer values are embedded.  Truncation is proven either against a fuller
    prefix-matching fragment already in the tool evidence (a deterministic target) or, for a
    「内容をそのまま抜き出す」 ask, against a title-only body; over-enumeration fires only when a single-item
    selector is present and no enumeration cue widens the ask.  A genuine enumeration question
    (「すべて挙げて」) and an already-full or already-single answer all pass unchanged.
    """
    q = nfc(question or "")
    a = nfc(str(answer or "")).strip()
    if not a:
        return GranularityValidation(True)
    # Abstain is never a granularity error (EV-safe; the production caller also guards on this).
    a_low0 = a.lower()
    if any(tok in a_low0 for tok in ("わかりません", "分かりません", "不明")):
        return GranularityValidation(True)

    # (1) verbatim-extract truncation ------------------------------------------------------------
    if _VERBATIM_EXTRACT_RE.search(q):
        if _META_RESPONSE_RE.search(a):
            return GranularityValidation(
                False, "truncation",
                ("回答が原文抽出ではなく、探索手順・謝罪などのメタ応答になっています",),
                directive=(
                    "探索手順や謝罪を回答にしないでください。対象IDを含む該当セル/行/項目を再確認し、"
                    "その本文全体だけを省略せず submit_answer で返してください。"
                    "原文を確定できない場合のみ従来どおり棄権してください。"),
            )
        a_low = a.lower()
        for frag in _granularity_fragments(tool_outputs):
            f = nfc(frag).strip()
            fl = f.lower()
            # ``a`` is the head of a materially longer contiguous fragment ⇒ the answer was truncated.
            if len(f) >= len(a) + 8 and f != a and (fl.startswith(a_low) or a_low in fl):
                return GranularityValidation(
                    False, "truncation",
                    (f"回答『{a}』は証拠中のより長い記載『{f[:80]}…』の一部に過ぎず全文になっていません",),
                    expected=f,
                    directive=(
                        "『そのまま抜き出す』要求です。該当項目の全文を省略・要約せずそのまま返してください。"
                        "見出しだけでなく本文/説明/続きを含め、証拠に存在する記載の全体を回答にしてください。"),
                )
        # No fuller fragment retrieved yet, but a 「内容をそのまま抜き出す」 ask answered with a short,
        # title-like string (no body punctuation) is almost certainly a head-only truncation.
        if (_CONTENT_EXTRACT_RE.search(q) and len(a) <= 25
                and not re.search(r"[：:。（(、，,]", a)):
            return GranularityValidation(
                False, "truncation",
                (f"回答『{a}』は見出しのみで、内容の本文が欠落しています",),
                directive=(
                    "『内容をそのまま抜き出す』要求です。見出し/IDだけで確定せず、該当項目のセル/行/段落の"
                    "本文を省略せず全文抽出し、説明・続きを含めてそのまま回答してください。"
                    "全文を確認できない場合のみ従来どおり棄権してください。"),
            )

    # (2) single-item over-enumeration -----------------------------------------------------------
    selector = _SINGLE_SELECTOR_RE.search(q)
    if selector is not None and not _ENUMERATION_CUE_RE.search(q):
        # Count only "wordy" items — a comma inside a numeric interval (e.g. "(6.10, 6.30]") is not an
        # enumeration, so items that are purely digits/punctuation are ignored to avoid false positives.
        items = [p.strip() for p in _ITEM_SEP_RE.split(a)
                 if p.strip() and re.search(r"[^\d.,，、()（）\[\]{}~〜\-–—:：/／%\s]", p)]
        if len(items) >= 2:
            sel = selector.group(0)
            return GranularityValidation(
                False, "over_enumeration",
                (f"質問は単一の項目({sel})を尋ねていますが、回答が{len(items)}件を列挙しています",),
                directive=(
                    f"質問は{sel}に対応する単一の項目を尋ねています。日付範囲などから複数タスクを列挙せず、"
                    "資料が当該単位に割り当てている単一の項目/フェーズ名だけを返してください。"
                    "どの項目か一意に確定できない場合のみ従来どおり棄権してください。"),
            )

    return GranularityValidation(True)


# --------------------------------------------------------------------------- numeric quantity contract
# Numeric questions frequently fail even after a correct file lookup because the model silently changes
# *what quantity is being measured*.  In particular, Japanese ``X のうち Y の割合`` fixes X as the
# denominator population.  The helpers below turn that wording, the requested unit, and the rounding
# instruction into a small machine-checkable contract shared by the investigator and exec verifier.
_ASCII_IDENTIFIER_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")
_DECIMAL_PLACES_RE = re.compile(r"小数第\s*(\d+)\s*位")
_NUMERIC_LITERAL_RE = re.compile(r"[+-]?\d+(?:,\d{3})*(?:\.(\d+))?")
_IGNORED_SCOPE_IDENTIFIERS = frozenset({"csv", "tsv", "xlsx", "xlsm", "df", "data"})


@dataclass(frozen=True)
class NumericRequirements:
    """The quantity/unit/rounding promises explicitly requested by one numeric question."""

    ratio: bool = False
    denominator_scope: str = ""
    denominator_fields: tuple[str, ...] = ()
    denominator_operators: tuple[str, ...] = ()  # lt/lte/gt/gte/eq
    unit: str | None = None
    decimal_places: int | None = None


@dataclass(frozen=True)
class NumericValidation:
    """Result of checking a proposed numeric answer and its executed formulas against the question."""

    passed: bool
    issues: tuple[str, ...] = ()
    denominator_formula: str = ""

    def to_dict(self) -> dict[str, object]:
        return {
            "passed": self.passed,
            "issues": list(self.issues),
            "denominator_formula": self.denominator_formula,
        }


def _denominator_scope(question: str) -> str:
    """Return the condition immediately governing ``のうち`` (the denominator population)."""
    if "のうち" not in question:
        return ""
    prefix = question.split("のうち", 1)[0]
    # Drop the document/project preamble while preserving the complete local condition.
    for sep in ("。", "？", "?", "！", "!", "、", ","):
        if sep in prefix:
            prefix = prefix.rsplit(sep, 1)[-1]
    return prefix.strip()


def numeric_requirements(question: str) -> NumericRequirements:
    """Infer the explicit numeric answer contract without consulting a model or corpus facts."""
    q = nfc(question or "")
    ratio = bool(re.search(r"割合|比率|何\s*[%％]|パーセント", q, re.I))
    scope = _denominator_scope(q) if ratio else ""
    fields = tuple(dict.fromkeys(
        token for token in _ASCII_IDENTIFIER_RE.findall(scope)
        if token.lower() not in _IGNORED_SCOPE_IDENTIFIERS
    ))
    ops: list[str] = []
    for pattern, code in (
        (r"未満", "lt"), (r"以下", "lte"), (r"(?:より大き|超え|超の)", "gt"),
        (r"以上", "gte"), (r"(?:=|＝|等しい)", "eq"),
    ):
        if re.search(pattern, scope):
            ops.append(code)

    unit: str | None = None
    for pattern, canonical in (
        (r"[%％]|パーセント", "%"), (r"億円|万円|千円|円", "円"),
        (r"ドル", "ドル"), (r"件", "件"), (r"人", "人"), (r"時間", "時間"),
        (r"日(?:間|数)?", "日"), (r"倍", "倍"),
    ):
        if re.search(pattern, q):
            unit = canonical
            break
    rounding = _DECIMAL_PLACES_RE.search(q)
    return NumericRequirements(
        ratio=ratio,
        denominator_scope=scope,
        denominator_fields=fields,
        denominator_operators=tuple(ops),
        unit=unit,
        decimal_places=(int(rounding.group(1)) if rounding else None),
    )


def _step_parts(step: Any) -> tuple[str, Any]:
    if isinstance(step, Mapping):
        return str(step.get("code", "") or ""), step.get("output")
    return str(getattr(step, "code", "") or ""), getattr(step, "output", None)


def _division_denominator(code: str) -> ast.AST | None:
    try:
        tree = ast.parse(code, mode="eval")
    except (SyntaxError, ValueError):
        return None
    return next((n.right for n in ast.walk(tree) if isinstance(n, ast.BinOp)
                 and isinstance(n.op, ast.Div)), None)


def _same_number(left: Any, right: Any) -> bool:
    try:
        return abs(float(left) - float(right)) <= max(1e-12, abs(float(right)) * 1e-12)
    except (TypeError, ValueError):
        return False


def _resolved_denominator_formula(formulas: Sequence[Any]) -> str:
    """Find the final ratio denominator and follow a numeric literal back to its compute provenance."""
    parts = [_step_parts(s) for s in formulas]
    for index in range(len(parts) - 1, -1, -1):
        code, _output = parts[index]
        denominator = _division_denominator(code)
        if denominator is None:
            continue
        if isinstance(denominator, ast.Constant) and isinstance(denominator.value, (int, float)):
            for prior_code, prior_output in reversed(parts[:index]):
                if prior_code and _same_number(prior_output, denominator.value):
                    return prior_code
        try:
            return ast.unparse(denominator)
        except Exception:  # pragma: no cover - ast.unparse is available on supported Python
            return code
    return ""


def _formula_operators(code: str) -> frozenset[str]:
    try:
        tree = ast.parse(code, mode="eval")
    except (SyntaxError, ValueError):
        return frozenset()
    mapping = {ast.Lt: "lt", ast.LtE: "lte", ast.Gt: "gt", ast.GtE: "gte", ast.Eq: "eq"}
    return frozenset(mapping[type(op)] for n in ast.walk(tree) if isinstance(n, ast.Compare)
                     for op in n.ops if type(op) in mapping)


def validate_numeric_answer(question: str, answer: str,
                            formulas: Sequence[Any] = ()) -> NumericValidation:
    """Check quantity definition, unit and rounding before a numeric answer may be committed.

    ``formulas`` accepts calc-ledger compute-step dicts or ``ComputeStep``-like objects.  For a literal
    denominator (``129 / 10938``), its producing step is followed by output value so the population
    filter remains auditable instead of disappearing behind the count.
    """
    req = numeric_requirements(question)
    issues: list[str] = []
    if not formulas:
        issues.append("決定論的計算証跡がない: compute等で再実行可能な値を取得していない")
    denominator = _resolved_denominator_formula(formulas) if req.ratio else ""
    if req.ratio and req.denominator_scope:
        if not denominator:
            issues.append(
                f"量の定義が未確認: 『{req.denominator_scope}のうち』の分母を計算証跡から特定できない")
        else:
            missing_fields = [f for f in req.denominator_fields
                              if not re.search(rf"\b{re.escape(f)}\b", denominator)]
            if missing_fields:
                issues.append(
                    "量の定義が不一致: 分母に『〜のうち』条件の対象列がない(" +
                    ", ".join(missing_fields) + f"; 分母={denominator})")
            present_ops = _formula_operators(denominator)
            missing_ops = [op for op in req.denominator_operators if op not in present_ops]
            if missing_ops:
                issues.append(
                    "量の定義が不一致: 分母に『〜のうち』条件の比較がない(" +
                    ", ".join(missing_ops) + f"; 分母={denominator})")

    normalized_answer = nfc(answer or "").replace("％", "%")
    if req.unit == "%" and "%" not in normalized_answer:
        issues.append("単位が不一致: 質問は%を要求している")
    elif req.unit and req.unit not in ("%", "円") and req.unit not in normalized_answer:
        issues.append(f"単位が不一致: 質問は{req.unit}を要求している")
    elif req.unit == "円" and "円" not in normalized_answer:
        issues.append("単位が不一致: 質問は円単位を要求している")

    if req.decimal_places is not None:
        match = _NUMERIC_LITERAL_RE.search(normalized_answer)
        actual = len(match.group(1) or "") if match else None
        if actual != req.decimal_places:
            issues.append(
                f"丸めが不一致: 小数第{req.decimal_places}位を要求、回答の小数桁={actual}")
    return NumericValidation(not issues, tuple(issues), denominator)


def _build(contract: str, arch: str, method: str, confidence: float, evidence: str,
           question: str = "") -> QuestionContract:
    return QuestionContract(
        contract=contract,
        label=CONTRACT_LABELS[contract],
        archetype=arch,
        completion_conditions=completion_conditions(question, contract),
        route=CONTRACT_ROUTE[contract],
        method=method,
        confidence=confidence,
        evidence=evidence,
        consistent_archetypes=CONTRACT_ARCHETYPES[contract],
    )


def _deterministic(q: str, arch: str) -> tuple[str, float, str] | None:
    """Return ``(contract, confidence, evidence)`` from deterministic rules, or ``None`` if ambiguous.

    ``None`` means "no confident deterministic decision" — the caller may then consult ``flash``.
    """
    base = _base_contract(arch)

    # Authoritative structural contracts (version diff / enumeration / cross-aggregate) win outright.
    if base in (VERSION_DIFF, FULL_ENUMERATION, CROSS_AGGREGATE):
        return base, _CONF_SPECIFIC, f"archetype={arch}"

    if is_staff_population_question(q):
        return CROSS_AGGREGATE, _CONF_SPECIFIC, "all-project PP/contract/PLAN/FR staff population"

    # New-dimension refinements apply only to lookup/derivation bases.
    if _SPATIAL_GEOM_RE.search(q):
        return SPATIAL, _CONF_SPECIFIC, "spatial cue (座席/隣/向かい)"
    if _GANTT_RE.search(q):
        return CHART_READ, _CONF_SPECIFIC, "Gantt cue (schedule/week range)"
    if _CHART_RE.search(q):
        return CHART_READ, _CONF_SPECIFIC, "chart cue (グラフ/ヒストグラム/系列)"
    if base == SIMPLE_LOOKUP and _FORMAT_RE.search(q):
        return FORMAT_CHECK, _CONF_SPECIFIC, "format cue (太字/下線/色/フォント)"
    if base == SIMPLE_LOOKUP and _MULTIHOP_RE.search(q):
        return MULTI_HOP, _CONF_BASE, "multi-hop cue (最も…な人/案件 の …)"
    if _SPATIAL_DIR_RE.search(q):
        return SPATIAL, _CONF_BASE, "directory cue (内線/EXT lookup)"

    if base == NUMERIC:
        return NUMERIC, _CONF_BASE, f"archetype={arch}"
    # Archetypes with a concrete structural signal (extraction / glossary / config / metric) commit to
    # simple_lookup deterministically.
    if arch in ("document_extract", "glossary_formal", "glossary_abbrev",
                "config_model_type", "config_hyperparam", "metric_score"):
        return SIMPLE_LOOKUP, _CONF_BASE, f"archetype={arch}"

    # A bare ``fact_lookup`` (the backbone's no-signal default) or an ``unknown`` archetype with no
    # structural cue is genuinely ambiguous → defer to flash if one is supplied.
    return None


# Prompt for the flash arbiter. It only decides among the nine contracts for a genuinely ambiguous
# question; the deterministic layer has already handled every confident case.
_FLASH_SYSTEM = (
    "あなたは社内ドライブ文書QAの質問を、回答に必要な『契約型』へ分類する分類器です。"
    "次の9種のいずれか1つをコード(英字)で返してください:\n"
    "simple_lookup=単一記載を引く単純検索 / multi_hop=中間エンティティを辿る多段関係 / "
    "cross_aggregate=複数文書を横断する集計 / full_enumeration=母集団を漏れなく挙げる完全列挙 / "
    "format_check=太字/下線/色などの書式で抽出 / chart_read=図/グラフから数値を読取 / "
    "spatial=座席表など空間関係の推論 / version_diff=旧版と新版の差分 / numeric=式で数値を導出。\n"
    "推測で本文を作らず、質問が要求する処理の種類だけで1つ選び、コードのみ返すこと。"
)

_FLASH_SCHEMA = {
    "type": "object",
    "properties": {"contract": {"type": "string", "enum": list(CONTRACTS)}},
    "required": ["contract"],
}


def flash_classify(question: str, *, model: str | None = None,
                   generate: Callable[..., str] | None = None) -> str | None:
    """Production flash arbiter: ask ``gemini-2.5-flash`` for one contract code; ``None`` on failure.

    Thin wrapper over :func:`src.rag.llm.generate` (imported lazily so importing this module never
    reaches the network or requires the Gemini SDK). Returns a validated contract code, or ``None`` when
    the model is unreachable or returns something outside :data:`CONTRACTS` (the caller then keeps its
    deterministic default). ``generate`` may be injected in tests to avoid the live client.
    """
    import json

    if generate is None:
        from src.rag import llm  # lazy: the google-genai client is only needed in production
        generate = llm.generate

    try:
        raw = generate(
            question,
            system=_FLASH_SYSTEM,
            model=model or settings.GEN_MODEL,
            temperature=0.0,
            max_output_tokens=64,
            response_schema=_FLASH_SCHEMA,
        )
    except Exception:  # noqa: BLE001 — arbitration is best-effort; fall back to deterministic default
        return None
    code = None
    try:
        code = json.loads(raw).get("contract")
    except (ValueError, AttributeError):
        code = raw.strip()
    return code if code in CONTRACTS else None


def classify(question: str, *, flash: Callable[[str], str | None] | None = None) -> QuestionContract:
    """Classify ``question`` into its :class:`QuestionContract` (deterministic first, flash on ambiguity).

    ``flash`` is an injected arbiter ``question -> contract_code | None`` consulted **only** when the
    deterministic layer is inconclusive (an ``unknown`` archetype with no specific cue). Pass
    :func:`flash_classify` in production; pass ``None`` (or omit) for a fully deterministic, network-free
    classification — which is what the gold-100 accuracy measurement uses.
    """
    q = nfc(question)
    arch = _archetype.classify(q)

    decided = _deterministic(q, arch)
    if decided is not None:
        contract, conf, why = decided
        return _build(contract, arch, "deterministic", conf, why, q)

    # Ambiguous: try the flash arbiter, else default to the weakest simple_lookup.
    if flash is not None:
        code = flash(q)
        if code in CONTRACTS:
            return _build(code, arch, "flash", _CONF_WEAK, "flash arbitration", q)
    return _build(SIMPLE_LOOKUP, arch, "deterministic", _CONF_WEAK,
                  f"fallback (archetype={arch})", q)


# --------------------------------------------------------------------------- accuracy measurement
@dataclass(frozen=True)
class AgreementReport:
    """Result of scoring the classifier against a gold archetype column."""

    total: int
    agree: int
    rate: float
    refinements: tuple[dict[str, str], ...]   # contract strictly more specific than the gold archetype
    mismatches: tuple[dict[str, str], ...]     # contract NOT consistent with the gold archetype

    def to_dict(self) -> dict[str, object]:
        return {
            "total": self.total, "agree": self.agree, "rate": self.rate,
            "refinements": list(self.refinements), "mismatches": list(self.mismatches),
        }


# A contract is a "refinement" (not a trivial 1:1 of the archetype's base contract) when it is one of
# the strictly-more-specific new dimensions.
_REFINEMENT_CONTRACTS = frozenset({SPATIAL, CHART_READ, FORMAT_CHECK, MULTI_HOP})


def agreement_rate(rows: list[dict[str, str]], *,
                   flash: Callable[[str], str | None] | None = None) -> AgreementReport:
    """Score classifier vs. the gold ``archetype`` column (``≥0.90`` is the SOT-2493 acceptance target).

    Each row must carry ``question`` and ``archetype``. A row *agrees* when the predicted contract is
    consistent with the gold archetype (:meth:`QuestionContract.is_consistent_with`) — i.e. the contract
    either matches the archetype's base contract or legitimately *refines* it. Refinements and genuine
    mismatches are both recorded so their validity can be reviewed (受け入れ条件①: 不一致は妥当性を記録).
    """
    agree = 0
    refinements: list[dict[str, str]] = []
    mismatches: list[dict[str, str]] = []
    for row in rows:
        q = row.get("question", "")
        gold = (row.get("archetype", "") or "unknown").strip() or "unknown"
        res = classify(q, flash=flash)
        if res.is_consistent_with(gold):
            agree += 1
            if res.contract in _REFINEMENT_CONTRACTS:
                refinements.append({
                    "question": q, "gold_archetype": gold, "contract": res.contract,
                    "note": res.evidence,
                })
        else:
            mismatches.append({
                "question": q, "gold_archetype": gold, "contract": res.contract,
                "note": res.evidence,
            })
    total = len(rows)
    rate = agree / total if total else 1.0
    return AgreementReport(total=total, agree=agree, rate=rate,
                           refinements=tuple(refinements), mismatches=tuple(mismatches))
