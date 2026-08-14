"""SOT-2604 (Stage3) — deterministic-value → gold-format naturalization layer.

Parent PLAN SOT-2602 (決定論先行パイプラインへの反転). This is the **exit** of the inverted
architecture. The deterministic pipelines (Stage0 router :mod:`src.rag.agent.det_pipeline`, Wave A1〜B2)
ground a ``{value, evidence, method}`` contract; this layer *naturalizes* that value into gold's answer
書式 **without altering the facts** — the single, short LLM call the design allows lives here and here
only, and is skipped entirely whenever a deterministic template already produces gold format.

Design invariants
-----------------
* **Template-first, LLM-last.** Types whose 書式 is deterministic (数値・ID列挙・週範囲・「該当なし」…) are
  formatted by a pure template — **no LLM**. Only a free-text-requiring type whose deterministic value is
  still a *raw structure* (or that explicitly asks for it via ``method['naturalize']``) spends **one short
  LLM call** to phrase it, and that call is instructed to preserve the value verbatim and add no new fact.
* **Value facts are never invented.** Every transformation here is *format-level*: canonicalizing a
  none-form to 「該当なし」 (SOT-2544 記号↔文章形の同義), completing a truncated verbatim extract from a
  *fuller fragment already in evidence* (SOT-2545), or trimming an over-enumerated list down to the
  *evidence-designated* single item (SOT-2545). None of these read a corpus fact from outside the supplied
  contract, and none fabricate a value. No answer / no corpus fact is hardcoded in this module.
* **Additive, never subtractive.** A valid non-blank contract in ⇒ a valid contract out (回答数を減らさない).
  The only ``None`` return is the defensive blank-value guard (「決定論値が空なら整形せず上位へ返す」): the
  caller then falls back to abstain / the LLM loop rather than committing an empty answer. A failing /
  unavailable LLM naturalizer degrades to the deterministic template text — it never drops the answer.
* **Gated OFF with the router.** Shares ``RAG_DET_PIPELINE_ROUTER`` (default OFF) with the Stage0 router:
  the layer is only ever reached from the router's det-path, so with the flag off (or the Stage0 registry
  empty) it is never invoked and the champion serve path is byte-identical.
"""
from __future__ import annotations

import os
import re
from decimal import ROUND_HALF_UP, Decimal, InvalidOperation, localcontext
from typing import Any, Callable, Mapping, Sequence

from src.rag.tools import contract as _contract

# A naturalizer turns a deterministic value string + the question into gold-shaped prose, preserving the
# value and adding no fact. ``None`` (the default) means "build the lazy Gemini one-shot on demand"; tests
# inject a stub so the layer stays network-free.
Naturalizer = Callable[[str, str], "str | None"]

# Contract types (question_contract.CONTRACTS) whose 書式 a pure template always produces — these NEVER
# spend an LLM call even if a stray ``method['naturalize']`` flag is set. Numeric values, ID/enumeration
# lists, format/chart/spatial reads are all rendered by the deterministic template below. Every other type
# (simple_lookup / multi_hop / cross_aggregate / version_diff …) may opt into the one short naturalize call.
_TEMPLATE_CONTRACTS: "frozenset[str]" = frozenset(
    {"numeric", "full_enumeration", "format_check", "chart_read", "spatial"}
)

# ------------------------------------------------------------------- SOT-2544 記号↔文章形の同義 (none-form)
# The canonical gold spelling for a 「該当なし」 conclusion, and the none-forms that normalize to it. Matched
# on the *whole* stripped value (fullmatch) so a legitimate answer that merely contains 「なし」 as a
# substring (e.g. 「課題なし体制」) is never collapsed. A 該当なし conclusion is a REAL answer under the
# rubric — this canonicalization does not touch the abstain path.
_NONE_CANONICAL = "該当なし"
_NONE_RE = re.compile(
    r"^(?:"
    r"該当(?:する)?(?:項目|もの|データ|記載事項|記載|情報|値|レコード)?(?:は)?"
    r"(?:なし|無し|ありません|存在しません|存在しない|見つかりません|見当たりません)"
    r"|なし|無し|ありません|存在しません|存在しない|見つかりません|見当たりません"
    r"|N/?A"
    r")[。.]?$",
    re.IGNORECASE,
)

# ------------------------------------------------------------------- SOT-2545 粒度 (granularity) cues
# A single-item selector pins the ask to one unit (第N週/N番目/…); mirrors question_contract._SINGLE_SELECTOR_RE.
_SINGLE_SELECTOR_RE = re.compile(
    r"第\s*\d+\s*(?:週|回|章|項|条|フェーズ|ステップ|段階|日)目?|\d+\s*(?:番目|つ目|個目|件目)")
# Enumeration cues make a question genuinely multi-item and suppress the over-enumeration trim.
_ENUMERATION_CUE_RE = re.compile(
    r"すべて|全て|全部|列挙|挙げて|一覧|それぞれ|各(?:項目|案件|人|々)|漏れなく|網羅|複数")
# A verbatim/full-content extract ask promises the whole body; a shorter prefix is a truncation.
_VERBATIM_EXTRACT_RE = re.compile(r"そのまま|抜き出|原文|全文|一字一句|逐語|内容|本文|詳細")
# Evidence keys a pipeline may set to designate the trimmed / completed target (evidence-driven, no guess).
_SELECTED_KEYS = ("selected", "designated", "chosen", "target")
_FULLTEXT_KEYS = ("full_text", "fulltext", "body", "full", "content", "verbatim")


def _env_flag(key: str, default: bool) -> bool:
    raw = os.getenv(key)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def enabled() -> bool:
    """Whether the formatting layer runs — shares ``RAG_DET_PIPELINE_ROUTER`` (default OFF).

    The layer is only reached from the Stage0 router's det-path, so this flag gates the whole inverted
    exit together with the router: off ⇒ never invoked ⇒ champion byte-identical.
    """
    return _env_flag("RAG_DET_PIPELINE_ROUTER", False)


def derived_contracts_enabled() -> bool:
    """Whether the SOT-2617 derived 書式契約 (unit/rounding/verbosity) fire — ``RAG_DERIVED_FORMAT_CONTRACTS`` (default OFF).

    A *second*, independent gate on top of :func:`enabled`: even when the inverted exit runs, these
    format-class contracts are opt-in so they never regress the SOT-2604 baseline. Off ⇒ the layer's
    output is byte-identical to before this issue.
    """
    return _env_flag("RAG_DERIVED_FORMAT_CONTRACTS", False)


# ------------------------------------------------------------------- SOT-2617 derived 書式契約 (unit/rounding/verbosity)
# All three contract classes are keyed off *question* cues (never off a gold value), operate on the
# rendered deterministic text, and preserve the numeric derivation — they only correct 書式 (単位表記・
# 丸め桁・冗長表現). No answer / corpus fact is hardcoded.

# A generic *counter* suffix a currency/measure question can override without touching the number. These
# are dimensionless tallies (件数/個数…), NOT domain units — a real unit (円/時間/%/人…) is never rewritten.
_GENERIC_COUNTERS = ("件", "個", "つ", "箇所", "点", "コ")
# A leading signed number (comma-grouped, optional decimals) optionally followed by a unit token.
_NUMBER_UNIT_RE = re.compile(r"^\s*(?P<num>-?\d[\d,]*(?:\.\d+)?)\s*(?P<unit>\D*?)\s*$")
# Currency ask — the answer's unit should be 円. Gated away from an explicit count ask (何件/いくつ…).
_CURRENCY_Q_RE = re.compile(r"金額|差額|費用|価格|料金|コスト|請求|予算|単価|総額|売上|利益|収益|価額|いくら")
_COUNT_Q_RE = re.compile(r"何\s*(?:件|個|人|社|名|箇所|つ|回)|いくつ|幾つ")

# Rounding ask — honor an explicit precision directive stated in the question (四捨五入).
_ROUND_DECIMAL_RE = re.compile(r"小数(?:点以下)?第\s*(?P<n>[0-9０-９一二三四五六七八九十]+)\s*位(?:まで)?")
_ROUND_INT_RE = re.compile(r"整数(?:で|に|化)|小数(?:点以下)?を?四捨五入|四捨五入して整数")

# Verbosity — a scalar-quantity ask (何<unit>/いくつ/合計で…) whose deterministic value came back as a
# whole verbose sentence. The asked unit is captured so the summary quantity span can be pulled out.
_QUANTITY_ASK_RE = re.compile(
    r"何\s*(?P<unit>時間|日間|日|件|個|人|回|分|秒|年|ヶ月|か月|月|週間|週|割|名|社|ページ|枚|%|％)")
_SUMMARY_MARKER_RE = re.compile(r"合計|総計|総合|全体|トータル|あわせ|併せ|合わせ|計で|の計")
# Trailing redundant count annotation on a non-count answer: 「…（該当件数: 14件）」「…(該当14件)」.
_TRAILING_COUNT_PAREN_RE = re.compile(
    r"\s*[（(]\s*(?:該当[^）)]*?\d+\s*(?:件|個|セル|箇所|レコード|行)?|\d+\s*(?:件|個|セル|箇所)\s*(?:該当|一致)?)\s*[)）]\s*$")
# A question that asks for a *condition/description*, not a tally — so a trailing count note is 冗長.
_CONDITION_Q_RE = re.compile(r"条件|どのよう|どんな|どういう|説明|理由|内容|定義|方法|状態")

_FULLWIDTH_DIGITS = str.maketrans("０１２３４５６７８９", "0123456789")
_KANJI_SMALL_NUM = {"一": 1, "二": 2, "三": 3, "四": 4, "五": 5, "六": 6, "七": 7, "八": 8, "九": 9, "十": 10}


# ------------------------------------------------------------------- SOT-2650 括弧内付加情報の書式契約
# The "parenthetical addition" wrong class (idx52/84/87/88 …): the committed value is correct but the
# model appends a supplementary annotation — 「解釈・業務示唆整理（担当：松本・鈴木）」「5（スライド6）」
# 「AYM(青葉与信マネジメント株式会社)」 — and the judge scores the extra content Incorrect. Dropping a
# *trailing* balanced （…）/(…) group never changes the answer value (the value is the body before it),
# so this is a value-preserving contract in the same family as the SOT-2617 verbosity trim.
#
# Fail-closed boundaries (some gold answers legitimately END with a parenthetical — e.g. a task name
# 「…確定（タスク割振・ガント更新）」 or a value detail 「n_estimators（1位=500、2位=300）」 — so a blanket
# trailing strip is NOT value-preserving; only *recognized annotation shapes* are dropped):
# * only a TRAILING group is dropped — a mid-string parenthetical (idx93 gold 「疑似欠損（NA）扱い…」)
#   is part of the extracted value and is never touched;
# * the group's CONTENT must look like an annotation: a locator/attribution/name-expansion/metric-detail
#   keyword (:data:`_ANNOTATION_PAREN_RE`), or a qualifier echoed verbatim from the question
#   (idx52 「（別契約）」 ← 「別契約」と明記されているもの). Anything else is treated as value and kept;
# * a verbatim-extraction ask (「そのまま」「抜き出し」) narrows further to meta-commentary shapes only
#   (:data:`_META_ANNOTATION_RE` — 「…の部分」「（スライド6）」), since there the source text, parens
#   included, IS the requested value;
# * the remaining body must be non-empty — a fully parenthesized answer is left alone;
# * multi-line answers are prose, not a value+annotation shape — skipped;
# * gated behind ``RAG_FORMAT_STRIP_PAREN`` (default OFF ⇒ byte-identical serve path).
_PAREN_OPEN = {"(", "（"}
_PAREN_CLOSE = {")", "）"}
_VERBATIM_Q_RE = re.compile(r"そのまま|抜き出し|抜き出す|抜粋")
_QUOTE_WRAP_RE = re.compile(r"^「([^「」]+)」$")
# Supplementary-annotation cues inside a trailing paren: provenance locators (スライド/ページ/セル…),
# attributions (担当), meta commentary (部分/該当/見出し…), corporate-name expansions (株式会社…),
# metric detail (相関係数/F1/約<digit>), and enumeration listings (タスクID: …).
_ANNOTATION_PAREN_RE = re.compile(
    r"担当|スライド|ページ|シート|セル|行目|段落|参照|参考|補足|注記|出典|部分|該当|見出し"
    r"|タスクID|アクションID|株式会社|有限会社|医療法人|相関係数|F1|約\s*[0-9０-９]"
    r"|W[0-9０-９]+\s*[〜～\-]\s*W[0-9０-９]+")
# The narrower meta-commentary subset that is safe even on a verbatim-extraction ask.
_META_ANNOTATION_RE = re.compile(r"部分|記載|該当|出典|スライド|ページ|段落|見出し")
# SOT-2717 (idx8) — a trailing paren whose whole content is a bare unit/counter word (optionally with
# digits): 「17,744ドル（人）」→ gold 「17,744ドル」. This is redundant unit decoration the Gemini answer
# loop appends to a numeric value; the currency/count unit already lives in the value body. Fires ONLY
# when the body is value-shaped (carries a digit) so a proper-noun parenthetical (e.g. 田中（人事部）) —
# whose content 「人事部」 is NOT a bare unit — is never touched. Verified against gold v4: NO gold answer
# ends in such a bare-unit paren, so this only ever drops decoration a real answer never carries.
_UNIT_PAREN_CONTENT_RE = re.compile(r"^[0-9０-９,，、.\s]*(?:人|円|件|名|個|回|箇所|点|ドル|%|％)$")
_HAS_DIGIT_RE = re.compile(r"[0-9０-９]")


def strip_paren_enabled() -> bool:
    """Whether the trailing-parenthetical strip fires — ``RAG_FORMAT_STRIP_PAREN`` (default OFF)."""
    return _env_flag("RAG_FORMAT_STRIP_PAREN", False)


def _trailing_paren_span(text: str) -> "tuple[int, int] | None":
    """(start, end) of a balanced trailing （…）/(…) group in ``text`` (end == len), or ``None``."""
    if not text or text[-1] not in _PAREN_CLOSE:
        return None
    depth = 0
    for i in range(len(text) - 1, -1, -1):
        ch = text[i]
        if ch in _PAREN_CLOSE:
            depth += 1
        elif ch in _PAREN_OPEN:
            depth -= 1
            if depth == 0:
                return i, len(text)
    return None  # unbalanced — leave alone


def strip_trailing_parenthetical(question: str, value: str) -> "tuple[str, list[str]]":
    """Drop trailing supplementary parenthetical group(s) from an answer, value-preserving.

    Returns ``(new_value, fired_rules)`` — ``fired_rules`` is empty when nothing changed. Also unwraps
    a whole-answer 「…」 quotation (the same annotation-verbosity class: 「0.589」 vs gold 0.589).
    """
    rules: list[str] = []
    if not value or "\n" in value.strip():
        return value, rules
    text = value.strip()
    question = question or ""
    verbatim = bool(_VERBATIM_Q_RE.search(question))
    # trailing sentence punctuation after the group (「…（同額）。」) is preserved across the strip
    tail = ""
    while text and text[-1] in "。．.":
        tail = text[-1] + tail
        text = text[:-1].rstrip()
    while True:
        span = _trailing_paren_span(text)
        if span is None:
            break
        body = text[:span[0]].rstrip()
        content = text[span[0] + 1:len(text) - 1].strip()
        if not body or not content:
            break  # fully parenthesized (or empty group) — the group IS the value
        # SOT-2717 — a bare unit/counter parenthetical on a digit-bearing value is redundant decoration
        # (「17,744ドル（人）」→「17,744ドル」), safe to drop on either ask shape.
        unit_annotation = bool(_UNIT_PAREN_CONTENT_RE.match(content)) and bool(_HAS_DIGIT_RE.search(body))
        if verbatim:
            annotation = bool(_META_ANNOTATION_RE.search(content)) or unit_annotation
        else:
            annotation = bool(_ANNOTATION_PAREN_RE.search(content)) or unit_annotation or (
                len(content) >= 2 and content in question)
        if not annotation:
            break  # unrecognized shape ⇒ value-bearing, keep (fail-closed)
        text = body
        if "strip_trailing_paren" not in rules:
            rules.append("strip_trailing_paren")
    quote = _QUOTE_WRAP_RE.match(text)
    if quote and quote.group(1).strip():
        text = quote.group(1).strip()
        rules.append("unwrap_quotes")
    if not rules:
        return value, rules
    return text + tail, rules


# ------------------------------------------------------------------- SOT-2656 値保存回答正規化 (説明文・接頭辞・単位ゆれ)
# cycle4 クラスタE (docs/ai/sonnet_cycle_analysis/cycle4.md): the committed value is CORRECT but wrapped in
# non-value decoration the judge scores Incorrect —
#   * a full sentence frame 「差額は0円です（…）」          → gold 「0円」         (idx6)
#   * an approximation prefix 「約14,744ドル」               → gold 「14,744ドル」  (idx8/36)
#   * a redundant counter 「11件」/「49件」 for a bare-count → gold 「11」/「49」    (idx41/92)
# Each is value-PRESERVING: the number / proper-noun tokens are untouched; only non-value framing is
# dropped. This composes AFTER the SOT-2650 trailing-paren strip (paren strip → value-norm), and is gated
# behind a SEPARATE new flag ``RAG_FORMAT_VALUE_NORM`` (default OFF ⇒ byte-identical serve path).
#
# Fail-closed boundaries — verified against the whole gold100 (artifacts/predictions_test_v3_final.csv):
#   * NO gold answer ends in a bare 「N件」, ends in 「です/ます」, or begins with an approximation 「約」 —
#     so each rule below only ever strips decoration a real gold answer never carries;
#   * approx-prefix drops 約/およそ… ONLY when the very next char is a digit (a value like 「約款」 is safe);
#   * counter strip fires ONLY for a bare-count ask AND when the WHOLE value is 「<number><counter>」
#     (optionally + one trailing paren) — never a counter embedded in a longer phrase;
#   * sentence-frame collapse fires ONLY for a scalar-value ask AND when the extracted core is value-shaped
#     (contains a digit) — a prose answer 「担当は田中です」 has no digit ⇒ left alone;
#   * 「Nページ目」→「Nページ」 unit normalization is deliberately NOT done: gold carries BOTH forms
#     (idx12=2ページ / idx18=2ページ目), so a blanket conversion would regress idx18 (双方向変換禁止 —
#     証拠原文の表記を優先);
#   * a final value-preservation guard refuses any transform whose output numeric tokens are not a subset
#     of the input's (a transform can only ever DROP decoration, never invent/alter a number).
_APPROX_PREFIX_RE = re.compile(r"^(?:約|およそ|おおよそ|ほぼ|概ね|おおむね)\s*(?=[0-9０-９])")
_BARE_COUNT_Q_RE = re.compile(r"いくつ|幾つ|何\s*(?:件|個|名|箇所|つ|回)|件数")
_COUNT_VALUE_RE = re.compile(
    r"^\s*(?P<num>-?\d[\d,]*)\s*(?:件|個|名|箇所|点|つ)\s*(?:[（(][^（）()]*[)）])?\s*$")
_SCALAR_VALUE_Q_RE = re.compile(
    r"金額|差額|費用|価格|料金|コスト|請求|予算|単価|総額|売上|利益|収益|価額|いくら"
    r"|何\s*(?:円|ドル|ページ|人|件|個|回|時間|日)")
_SENTENCE_FRAME_RE = re.compile(
    r"^(?P<lead>[^、。]{0,16}?)は\s*(?P<core>.+?)\s*"
    r"(?:です|でした|だ|となります|となる|になります|になる)。?\s*"
    r"(?:[（(][^（）()]*[)）])?\s*$")
_NUM_TOKEN_RE = re.compile(r"-?\d[\d,]*(?:\.\d+)?")
_HAS_DIGIT_RE = re.compile(r"[0-9０-９]")


def value_norm_enabled() -> bool:
    """Whether the SOT-2656 value-preserving normalization fires — ``RAG_FORMAT_VALUE_NORM`` (default OFF)."""
    return _env_flag("RAG_FORMAT_VALUE_NORM", False)


def _num_tokens(text: str) -> "list[str]":
    return _NUM_TOKEN_RE.findall(text)


def _numeric_tokens_preserved(out: str, src: str) -> bool:
    """Every numeric token in ``out`` also occurs (with multiplicity) in ``src`` — no number invented/changed."""
    from collections import Counter

    co, ci = Counter(_num_tokens(out)), Counter(_num_tokens(src))
    return all(co[tok] <= ci[tok] for tok in co)


def normalize_value_answer(question: str, value: str) -> "tuple[str, list[str]]":
    """Drop non-value framing (sentence/approx/counter) from an answer, value-preserving (SOT-2656).

    Returns ``(new_value, fired_rules)`` — ``fired_rules`` empty when nothing changed. Each rule is
    question-cue-gated and fail-closed; a final numeric-token guard refuses any transform that would
    alter a number. Complements :func:`strip_trailing_parenthetical` (run it first).
    """
    rules: list[str] = []
    if not value or "\n" in value.strip():
        return value, rules
    original = value
    text = value.strip()
    q = question or ""

    # 1) sentence frame 「…は<核>です（…）」 → 核 — scalar-value ask + value-shaped (digit-bearing) core.
    if _SCALAR_VALUE_Q_RE.search(q):
        m = _SENTENCE_FRAME_RE.match(text)
        if m:
            core = m.group("core").strip()
            if core and core != text and _HAS_DIGIT_RE.search(core):
                text = core
                rules.append("sentence_frame")

    # 2) approximation prefix 約/およそ… immediately before a digit → drop the qualifier.
    stripped = _APPROX_PREFIX_RE.sub("", text)
    if stripped != text:
        text = stripped.strip()
        rules.append("approx_prefix")

    # 3) counter suffix 「N件」 → 「N」 — only a bare-count ask whose whole value is <number><counter>(+paren).
    if _BARE_COUNT_Q_RE.search(q):
        m = _COUNT_VALUE_RE.match(text)
        if m:
            text = m.group("num")
            rules.append("count_suffix")

    if not rules:
        return original, []
    if not text.strip() or not _numeric_tokens_preserved(text, original):
        return original, []  # fail-closed: a transform that emptied the value or changed a number is refused
    return text, rules


# ------------------------------------------------------------------- SOT-2682 小数指定問の単位strip書式契約
# idx79: 「…小数第2位で答えてください」への回答が「池田 直哉、7.00時間/タスク」— gold「池田 直哉、7.00」。
# 数値(7.00)は完全一致で、末尾に付いた単位/サフィックス(時間/タスク・%・倍…)のみで Incorrect。回答が明示的に
# 「小数第N位」で数値を求めた問いのとき、末尾の (小数)数値 の直後に来る単位表現だけを落とす — 値保存(数値
# トークン不変)・書式のみ。
#
# 全 gold100 較正 (artifacts/predictions_test_v3_final.csv): 「小数(点以下)第N位」指定問の gold は例外なく
# 裸数値 (idx17=2.21 / 29=6.088138~6.288138 / 30=1.18 / 33/54/57/63/79/83/99 いずれも単位なし)。ゆえに
# 小数指定問で末尾単位を落とす変換は fix(idx79) か no-op にしかならず、単位込み gold を壊さない
# (通貨「いくらですか」等は小数第N位の書式指定を持たないので非該当)。
# Fail-closed:
#   * 発火は質問が「小数第N位」書式指定を持つ場合のみ(単位そのものを問う問いは非該当);
#   * 落とすのは「末尾の(小数)数値 + 単位サフィックス」形のみ — 数値は小数点必須(整数回答 idx27「5」は非対象)、
#     単位は句読点/括弧/空白/数字を含まない短い(≤12字)トークンに限る(説明文・括弧注記は別契約が担当);
#   * 数値トークン保存ガード(:func:`_numeric_tokens_preserved`)で数値の改変を拒否;
#   * 複数行回答は散文とみなしスキップ;
#   * ``RAG_DECIMAL_UNIT_STRIP`` (default OFF ⇒ serve byte-identical)。
_DECIMAL_SPEC_Q_RE = re.compile(r"小数(?:点以下)?第\s*[0-9０-９一二三四五六七八九十]+\s*位")
# 末尾の「(小数)数値 + 単位サフィックス」。単位は句読点・括弧・空白・数字を含まない 1〜12 字トークン。
_TRAILING_DECIMAL_UNIT_RE = re.compile(
    r"(?P<num>-?\d[\d,]*\.\d+)\s*(?P<unit>[^\d\s、。．，,・()（）\[\]「」『』]{1,12})\s*$")


def decimal_unit_strip_enabled() -> bool:
    """SOT-2682 — 小数指定問の単位 strip が発火するか (``RAG_DECIMAL_UNIT_STRIP``, default OFF)。"""
    return _env_flag("RAG_DECIMAL_UNIT_STRIP", False)


def strip_decimal_spec_unit(question: str, value: str) -> "tuple[str, list[str]]":
    """小数第N位指定問の回答末尾に付いた単位サフィックスを落とす、値保存の書式契約 (SOT-2682)。

    Returns ``(new_value, fired_rules)`` — ``fired_rules`` は変化なしなら空。質問が「小数第N位」書式
    指定を持つときだけ発火し、末尾の (小数)数値+単位 の単位のみを除去する(数値トークンは保存)。整数のみの
    回答・単位のない回答・複数行回答・書式指定なしの問いはいずれも no-op。
    """
    rules: list[str] = []
    if not value or "\n" in value.strip():
        return value, rules
    if not _DECIMAL_SPEC_Q_RE.search(question or ""):
        return value, rules
    text = value.strip()
    m = _TRAILING_DECIMAL_UNIT_RE.search(text)
    if not m:
        return value, rules
    new = (text[: m.start()] + m.group("num")).rstrip()
    if not new or not _numeric_tokens_preserved(new, text):
        return value, rules
    rules.append("decimal_spec_unit_strip")
    return new, rules


# --- SOT-2688 (cycle7 K5, idx29) — ビン範囲の区間記法 → チルダ形式 naturalization ----------------
# cycle6 の chart_read wrong idx29 は値・ビン位置とも正しい ((6.088138, 6.288138]) のに、区間記法のまま
# 回答して gold の「6.088138 ~ 6.288138」(チルダ形式) と judge 不一致になった。値保存(両端の数値トークンを
# そのまま温存)で区間記法をチルダ形式へ書式変換するだけの層。
#   * 質問がビン/範囲を問うている時だけ発火(範囲を要求しない問いの座標等は触らない);
#   * 回答が丸ごと 1 個の区間式 (a, b] / [a, b) / (a,b) / [a,b] の時だけ変換(余分な散文があれば no-op);
#   * ``RAG_BIN_RANGE_FORMAT`` (default OFF ⇒ serve byte-identical)。
_BIN_RANGE_Q_RE = re.compile(r"範囲|ビン|レンジ|区間|ヒストグラム|bin|range")
_BIN_RANGE_FULL_RE = re.compile(
    r"^\s*[\(\[]\s*(?P<lo>-?\d[\d,]*(?:\.\d+)?)\s*[,，、]\s*(?P<hi>-?\d[\d,]*(?:\.\d+)?)\s*[\)\]]\s*$")


def bin_range_format_enabled() -> bool:
    """SOT-2688 — ビン範囲の区間記法→チルダ書式 naturalization が発火するか (``RAG_BIN_RANGE_FORMAT``, default OFF)。"""
    return _env_flag("RAG_BIN_RANGE_FORMAT", False)


def naturalize_bin_range(question: str, value: str) -> "tuple[str, list[str]]":
    """区間記法のビン範囲回答を gold のチルダ形式『A ~ B』へ書式変換する、値保存の naturalization (SOT-2688)。

    Returns ``(new_value, fired_rules)`` — 変化なしなら ``fired_rules`` は空。質問がビン/範囲を問い、かつ
    回答が丸ごと単一の区間式のときだけ発火して両端 (lo, hi) をそのまま『lo ~ hi』へ整形する(数値トークンは
    改変しない)。区間式でない回答・範囲を問わない問い・複数値混在はいずれも no-op。
    """
    rules: list[str] = []
    if not value:
        return value, rules
    if not _BIN_RANGE_Q_RE.search(question or ""):
        return value, rules
    m = _BIN_RANGE_FULL_RE.match(str(value))
    if not m:
        return value, rules
    new = f"{m.group('lo')} ~ {m.group('hi')}"
    if new == str(value):
        return value, rules
    rules.append("bin_range_tilde")
    return new, rules


# --- SOT-2718 — 通貨差額型の単位決定論固定 (idx8: 「17744人」→「17,744ドル」) ----------------------
# 症状: 通貨の差額を問う設問（給与差 等）で LLM が値は正しく到達する（idx8=17744）のに、単位を人数系
# （人/名/…）で framing したり裸数値のまま返して gold「17,744ドル」と judge 不一致になる (unit churn)。
# 修正(質問・idx 非依存): 設問が「通貨差額型」（差/違い + 通貨・金額文脈 + いくら/何ドル）で、回答が
#   * 裸数値、または
#   * 明らかに誤った無次元カウンタ単位（人/名/社/件/個/人分/名分）付き、または
#   * 対象通貨だが桁区切り欠落／末尾括弧注記付き
# のとき、値は一切変えずに **設問文脈の通貨**（米国/ドル/USD/$ → ドル、円/日本円 → 円）へ単位を固定し、
# 整数部を3桁カンマ区切りへ整形する。通貨が文脈から一意に定まらない・別の実単位が付く回答は no-op（推測しない）。
_CURRENCY_DIFF_Q_RE = re.compile(
    r"(?=.*(?:差|違い|差額))(?=.*(?:給与|給料|年収|報酬|金額|価格|費用|コスト|単価|総額|売上|利益|額|ドル|円|USD))"
    r".*(?:いくら|何\s*(?:ドル|円)|差額)")
# 対象通貨をカウンタとして誤付与しがちな無次元単位（これらは通貨差額回答では常に誤り ⇒ 上書き対象）。
_WRONG_CURRENCY_UNITS = frozenset({"", "人", "名", "社", "件", "個", "人分", "名分", "者"})
# 桁区切り／末尾括弧を正すため、既に対象通貨で終わる回答も再整形対象に含める。
_CURRENCY_UNIT_TOKENS = frozenset({"ドル", "円", "米ドル", "USドル", "US$", "$", "＄"})
# 末尾の（人）/(人) 等の括弧注記（数値+単位 or 単位のみ）。値本体の後ろに付く冗長注記のみを対象にする。
_TRAILING_UNIT_PAREN_RE = re.compile(r"\s*[（(][0-9０-９,，.\s]*(?:人|名|社|件|個|ドル|円)?[)）]\s*$")


def currency_diff_unit_enabled() -> bool:
    """SOT-2718 — 通貨差額型の単位決定論固定が発火するか (``RAG_CURRENCY_DIFF_UNIT``, default OFF ⇒ byte-identical)。"""
    return _env_flag("RAG_CURRENCY_DIFF_UNIT", False)


def _context_currency(question: str) -> "str | None":
    """設問文脈から対象通貨を一意決定する（米国/ドル/USD/$ → 「ドル」、円/日本円 → 「円」、不明 → None）。"""
    q = question or ""
    if re.search(r"米国|ドル|USD|米ドル|US\$|[\$＄]", q):
        return "ドル"
    if re.search(r"円|日本円|邦貨", q):
        return "円"
    return None


def apply_currency_diff_unit(question: str, value: str) -> "tuple[str, list[str]]":
    """通貨差額型の設問回答の単位を文脈通貨へ決定論固定し、整数部をカンマ整形する値保存の書式契約 (SOT-2718)。

    Returns ``(new_value, fired_rules)`` — 変化なしなら ``fired_rules`` は空。設問が「通貨差額型」でないもの、
    回答が数値+単位形に一意にパースできないもの、既に正しい書式のもの、別の実単位が付くもの、通貨が文脈から
    定まらないものはいずれも no-op。数値そのもの（有効数字）は決して変更しない。
    """
    rules: list[str] = []
    if not value or "\n" in value.strip():
        return value, rules
    if not _CURRENCY_DIFF_Q_RE.search(question or ""):
        return value, rules
    currency = _context_currency(question)
    if currency is None:
        return value, rules
    text = value.strip()
    # 末尾括弧注記（（人）等）を先に落としてから数値+単位をパース（値本体は括弧の前）。
    core = _TRAILING_UNIT_PAREN_RE.sub("", text).strip() or text
    m = _NUMBER_UNIT_RE.match(core)
    if not m:
        return value, rules
    unit = (m.group("unit") or "").strip()
    # 上書きしてよいのは：誤カウンタ単位・裸数値・対象通貨トークン（桁区切り/括弧の是正）に限る。
    # それ以外の実単位（時間/%/ページ 等）が付く回答は検出外 ⇒ 触らない（推測しない）。
    if unit not in _WRONG_CURRENCY_UNITS and unit not in _CURRENCY_UNIT_TOKENS:
        return value, rules
    num_raw = m.group("num").replace(",", "")
    try:
        if "." in num_raw:
            ip, fp = num_raw.split(".", 1)
            formatted_num = f"{int(ip):,}.{fp}"
        else:
            formatted_num = f"{int(num_raw):,}"
    except ValueError:
        return value, rules
    new = f"{formatted_num}{currency}"
    if new == text:
        return value, rules
    rules.append("currency_diff_unit")
    return new, rules


def _parse_small_int(raw: str) -> "int | None":
    """Parse a small positive integer written in ASCII/fullwidth digits or 一〜十 kanji (precision桁 use)."""
    s = raw.translate(_FULLWIDTH_DIGITS).strip()
    if s.isdigit():
        return int(s)
    if s in _KANJI_SMALL_NUM:
        return _KANJI_SMALL_NUM[s]
    return None


def _quantity_span_re(unit: str) -> "re.Pattern[str]":
    """Match a ``<number>`` or ``<number>〜<number>`` span immediately followed by ``unit`` (a range wins)."""
    num = r"-?\d[\d,]*(?:\.\d+)?"
    return re.compile(rf"{num}(?:\s*[〜～\-~－]\s*{num})?\s*{re.escape(unit)}")


def _apply_unit_contract(text: str, question: str, rules: list[str]) -> str:
    """idx6 class — a currency ask answered with a bare number or a *generic counter* ⇒ append/fix 円.

    Only rewrites a dimensionless counter (件/個/…) or a unit-less number; a real domain unit is left
    intact, and the number itself is never changed. Fires only when the question implies currency and
    does NOT explicitly ask for a count.
    """
    if not (_CURRENCY_Q_RE.search(question) and not _COUNT_Q_RE.search(question)):
        return text
    m = _NUMBER_UNIT_RE.match(text)
    if not m:
        return text
    unit = m.group("unit")
    if unit == "円" or unit.endswith("円"):
        return text  # already correct
    if unit == "" or unit in _GENERIC_COUNTERS:
        rules.append("unit_currency")
        return f"{m.group('num')}円"
    return text  # a specific non-generic unit ⇒ our detection is off, leave it


def _apply_rounding_contract(text: str, question: str, rules: list[str]) -> str:
    """丸め class — honor an explicit precision directive (小数第N位 / 整数で) via 四捨五入 (ROUND_HALF_UP).

    Format-level only: re-renders the SAME number at the requested precision. Runs in a local Decimal
    context so global precision is never polluted. No directive ⇒ no-op.
    """
    m = _NUMBER_UNIT_RE.match(text)
    if not m:
        return text
    dec_m = _ROUND_DECIMAL_RE.search(question)
    places: "int | None" = None
    if dec_m:
        places = _parse_small_int(dec_m.group("n"))
    elif _ROUND_INT_RE.search(question):
        places = 0
    if places is None or places < 0:
        return text
    try:
        with localcontext() as ctx:
            ctx.prec = 50
            num = Decimal(m.group("num").replace(",", ""))
            quant = Decimal(1) if places == 0 else Decimal(1).scaleb(-places)
            rounded = num.quantize(quant, rounding=ROUND_HALF_UP)
    except (InvalidOperation, ValueError):
        return text
    formatted = f"{rounded:.{places}f}" if places else str(int(rounded))
    if formatted == m.group("num"):
        return text
    rules.append("rounding")
    return f"{formatted}{m.group('unit')}"


def _apply_verbosity_trim(text: str, question: str, rules: list[str]) -> str:
    """idx64/idx65 class — strip 冗長表現 down to the asked answer, never touching the retained fact.

    Two sub-rules, both evidence-free but question-cue-gated:
    * **summary-quantity extraction**: a 「(合計で)何<unit>」 ask answered by a whole sentence ⇒ reduce to
      the summary ``<number(range)><unit>`` span (the one after a 合計/あわせ marker; or the sole span
      when unambiguous). Ambiguous multi-span with no marker ⇒ left untouched (no guessing).
    * **trailing count-note removal**: a condition/description ask whose answer trails a redundant
      「（該当件数: N件）」 tally ⇒ drop the parenthetical.
    """
    q = question
    ask = _QUANTITY_ASK_RE.search(q)
    if ask and len(text) > len(ask.group(0)) + 6:
        unit = ask.group("unit")
        span_re = _quantity_span_re(unit)
        marker = _SUMMARY_MARKER_RE.search(text)
        chosen: "str | None" = None
        if marker:
            after = span_re.search(text, marker.end())
            if after:
                chosen = after.group(0)
        if chosen is None:
            # No 合計/あわせ marker: only safe to trim when a single unique span exists (else guessing).
            uniq = list(dict.fromkeys(m.group(0).strip() for m in span_re.finditer(text)))
            if len(uniq) == 1:
                chosen = uniq[0]
        if chosen is not None:
            trimmed = re.sub(r"\s+", "", chosen)
            if trimmed and trimmed != text.strip():
                rules.append("verbosity_summary")
                text = trimmed

    stripped = _TRAILING_COUNT_PAREN_RE.sub("", text)
    if stripped != text and stripped.strip() and not _COUNT_Q_RE.search(q):
        # only for a non-count ask; a condition/description ask is the canonical case.
        if _CONDITION_Q_RE.search(q) or not _QUANTITY_ASK_RE.search(q):
            rules.append("verbosity_count_note")
            text = stripped.strip()
    return text


def _apply_derived_format_contracts(text: str, question: str, rules: list[str]) -> str:
    """SOT-2617 — run the three derived 書式契約 in order (verbosity → unit → rounding). Value-preserving."""
    question = question or ""
    text = _apply_verbosity_trim(text, question, rules)
    text = _apply_unit_contract(text, question, rules)
    text = _apply_rounding_contract(text, question, rules)
    return text


# --------------------------------------------------------------------------- value shaping (deterministic)
def _is_blank(value: Any) -> bool:
    """A value that carries no answer: ``None``, an empty/whitespace string, or an empty collection."""
    if value is None:
        return True
    if isinstance(value, str):
        return not value.strip()
    if isinstance(value, (list, tuple, set, dict)):
        return len(value) == 0
    return False


def _evidence_get(evidence: Mapping[str, Any], keys: Sequence[str]) -> Any:
    for k in keys:
        if k in evidence and not _is_blank(evidence[k]):
            return evidence[k]
    return None


def _complete_truncation(value: Any, question: str, evidence: Mapping[str, Any],
                         rules: list[str]) -> Any:
    """SOT-2545 truncation completion — a verbatim/full-content ask answered with a head-only prefix.

    Deterministic and evidence-bound: replaces ``value`` with a *fuller fragment already in evidence*
    (an explicit ``full_text``-family key, or any evidence string of which ``value`` is a strict prefix /
    substring). Never invents text — if evidence carries nothing longer, the value is left unchanged.
    """
    if not isinstance(value, str) or not _VERBATIM_EXTRACT_RE.search(question or ""):
        return value
    head = value.strip()
    if not head:
        return value
    # 1) an explicitly designated full-text field wins.
    designated = _evidence_get(evidence, _FULLTEXT_KEYS)
    if isinstance(designated, str) and len(designated.strip()) >= len(head) + 8 \
            and (designated.strip().startswith(head) or head in designated):
        rules.append("truncation_completed")
        return designated.strip()
    # 2) otherwise any evidence fragment of which the answer is a strict head/substring.
    for frag in _evidence_fragments(evidence):
        f = frag.strip()
        if len(f) >= len(head) + 8 and f != head and (f.startswith(head) or head in f):
            rules.append("truncation_completed")
            return f
    return value


def _trim_over_enumeration(value: Any, question: str, evidence: Mapping[str, Any],
                           rules: list[str]) -> Any:
    """SOT-2545 over-enumeration trim — a single-item ask answered with a list.

    Fires only when the question has a single-item selector (第N週/N番目/…) AND no enumeration cue widens
    it AND ``value`` is a multi-item list AND evidence names the designated item (``selected``-family).
    Returns that evidence-designated single item; otherwise leaves the list untouched (no guessing which
    item to keep — that would fabricate a selection).
    """
    if not isinstance(value, (list, tuple)) or len(value) <= 1:
        return value
    q = question or ""
    if not _SINGLE_SELECTOR_RE.search(q) or _ENUMERATION_CUE_RE.search(q):
        return value
    selected = _evidence_get(evidence, _SELECTED_KEYS)
    if selected is None:
        return value
    items = [str(v).strip() for v in value if not _is_blank(v)]
    sel = str(selected).strip()
    if sel in items:
        rules.append("over_enumeration_trimmed")
        return sel
    return value


def _evidence_fragments(evidence: Mapping[str, Any]) -> list[str]:
    """Flatten evidence values to string fragments (self-contained; no obligations import)."""
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

    walk(dict(evidence))
    return frags


def _render_template(value: Any) -> "str | None":
    """Render a deterministic value to its gold surface form — pure, no LLM. ``None`` when blank.

    * ``str``  → stripped verbatim (structured notation like ``n_estimators（1位=500、2位=300）`` survives).
    * ``bool`` → 「はい」/「いいえ」 (a yes/no contract's canonical Japanese form).
    * ``int``/``float`` → the number without a spurious trailing ``.0``.
    * ``list``/``tuple`` → non-blank items joined with 「、」 (a single item renders bare); the gold
      enumeration 書式.
    * ``dict`` → ``key=value`` pairs joined with 「、」 (deterministic; order preserved).
    """
    if value is None:
        return None
    if isinstance(value, str):
        s = value.strip()
        return s or None
    if isinstance(value, bool):
        return "はい" if value else "いいえ"
    if isinstance(value, int):
        return str(value)
    if isinstance(value, float):
        return str(int(value)) if value.is_integer() else repr(value)
    if isinstance(value, (list, tuple)):
        items = [t for t in (_render_template(v) for v in value) if t]
        if not items:
            return None
        return items[0] if len(items) == 1 else "、".join(items)
    if isinstance(value, Mapping):
        parts = [f"{k}={_render_template(v)}" for k, v in value.items()
                 if not _is_blank(v)]
        return "、".join(parts) if parts else None
    s = str(value).strip()
    return s or None


def _canonical_none(text: str, rules: list[str]) -> str:
    """SOT-2544 — collapse a whole-string none-form to the canonical 「該当なし」 (real answer, not abstain)."""
    if _NONE_RE.match(text.strip()):
        if text.strip() != _NONE_CANONICAL:
            rules.append("none_canonical")
        return _NONE_CANONICAL
    return text


def _needs_llm(value: Any, contract_type: "str | None", method: Mapping[str, Any]) -> bool:
    """Whether the one short LLM naturalize call is warranted — template-first, so **opt-in only**.

    The LLM fires only when the deterministic pipeline *explicitly* requests it via a truthy
    ``method['naturalize']`` — and never for a template-format type (数値/列挙/…), whose 書式 a pure
    template always produces. This keeps the call minimal and predictable: a pipeline that can already
    hand back a gold-shaped string never triggers a model call; only a free-text type that hands back a
    raw structure it wants phrased sets the flag. (The type is checked so a stray flag on a numeric/enum
    contract cannot force a needless call.)
    """
    if _is_blank(value):
        return False
    if not method.get("naturalize"):
        return False
    return contract_type not in _TEMPLATE_CONTRACTS


_NATURALIZE_SYSTEM = (
    "あなたは、決定論パイプラインが確定した回答値を、質問に対する自然な日本語の回答文へ整えるだけの整形器です。"
    "厳守事項:\n"
    "- 与えられた値の事実(数値・ID・固有名・列挙対象)を一切変更・追加・削除しない。新しい事実を創作しない。\n"
    "- 値に含まれない情報を補わない。値が答えそのものなら、そのまま最小限に整えて返す。\n"
    "- 前置き・言い訳・思考過程・引用符を付けず、回答本文のみを1つ返す。"
)


def _default_naturalizer(value_text: str, question: str) -> "str | None":
    """Lazy Gemini one-shot (production default). Best-effort: any failure returns ``None`` → template text.

    Imported lazily so this module stays dependency-light and offline tests never touch the network.
    """
    try:
        from src.rag import llm as _llm
    except Exception:  # noqa: BLE001 — no client available ⇒ keep template text
        return None
    prompt = (
        f"質問:\n{question}\n\n"
        f"決定論パイプラインが確定した回答値:\n{value_text}\n\n"
        "この値を、質問に自然に答える日本語の回答文へ整形してください。値の事実は改変しないこと。"
    )
    try:
        out = _llm.generate(prompt, system=_NATURALIZE_SYSTEM, temperature=0.0,
                            thinking_budget=0, max_output_tokens=512)
    except Exception:  # noqa: BLE001 — naturalize is additive; degrade to the template text
        return None
    out = (out or "").strip()
    return out or None


def _maybe_naturalize(text: str, question: str, naturalizer: "Naturalizer | None",
                      rules: list[str]) -> str:
    """Run the one short LLM naturalize call; keep the template text on any empty/failed result."""
    fn = naturalizer or _default_naturalizer
    try:
        out = fn(text, question)
    except Exception:  # noqa: BLE001 — never break the answer path on a naturalizer error
        out = None
    out = (out or "").strip() if isinstance(out, str) else None
    if out:
        rules.append("llm_naturalized")
        return out
    return text


# --------------------------------------------------------------------------- public entry
def format_contract(contract: Any, question: str = "", *, contract_type: "str | None" = None,
                    naturalizer: "Naturalizer | None" = None,
                    force: bool = False) -> "dict[str, Any] | None":
    """Naturalize a deterministic ``{value, evidence, method}`` into gold 書式 (Stage3 exit).

    Returns the reformatted contract dict, or ``None`` **only** when the deterministic value is blank
    (「決定論値が空なら整形せず上位へ返す」 — the caller then abstains / falls back to the LLM loop). When the
    layer is disabled (flag off and ``force`` unset) it is an identity no-op (returns the normalized
    contract unchanged), so it can never alter the answer outside the gated router path. The formatting
    provenance (which rules fired, whether the LLM was used) is recorded under ``method['formatting']``;
    ``method['confidence']`` and every other method field are preserved.
    """
    if not _contract.is_contract(contract):
        return None
    c = _contract.ensure_contract(contract)
    if not (force or enabled()):
        return c  # identity no-op when the inverted exit is gated off
    value = c.get("value")
    if _is_blank(value):
        return None  # blank deterministic value ⇒ let the caller abstain / fall back (回答数を減らさない)

    evidence: Mapping[str, Any] = c.get("evidence") or {}
    method = dict(c.get("method") or {})
    rules: list[str] = []

    # 1) evidence-bound granularity repair (SOT-2545): complete a truncated verbatim extract, then trim an
    #    over-enumerated single-item list — both only when evidence licenses it (never a guess).
    value = _complete_truncation(value, question, evidence, rules)
    value = _trim_over_enumeration(value, question, evidence, rules)

    # 2) deterministic template render (template-first — no LLM for numbers/lists/strings/none-forms).
    text = _render_template(value)
    if text is None:
        return None  # shaping emptied the value ⇒ nothing to commit; fall back.
    text = _canonical_none(text, rules)  # SOT-2544 none-form → 「該当なし」

    # 2b) SOT-2617 derived 書式契約 (unit/rounding/verbosity) — opt-in second gate, value-preserving.
    if derived_contracts_enabled():
        text = _apply_derived_format_contracts(text, question, rules)

    # 3) at most ONE short LLM naturalize call, and only for a free-text type whose value is still raw.
    if _needs_llm(value, contract_type, method):
        text = _maybe_naturalize(text, question, naturalizer, rules)

    method["formatting"] = {
        "engine": "formatting",
        "template_only": "llm_naturalized" not in rules,
        "rules": rules,
        "contract_type": contract_type,
    }
    return {"value": text, "evidence": dict(evidence), "method": method}
