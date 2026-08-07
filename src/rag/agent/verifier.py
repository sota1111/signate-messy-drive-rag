"""SOT-2470 — heterogeneous verification agent (independent re-derivation).

Parent SOT-2460. The :mod:`~src.rag.agent.investigator` loop answers a question with the *primary*
path (``gemini-2.5-pro`` + the investigator prompt). Correlated errors — a wrong value both the model
and any homogeneous double-check agree on — slip through a self-consistency check. This module adds a
**heterogeneous** second opinion whose errors are *uncorrelated* with the investigator's:

* **別モデル (model diversity)** — the verifier defaults to :data:`~config.settings.GEN_MODEL`
  (``gemini-2.5-flash``), a different tier than the investigator's
  :data:`~config.settings.GEN_MODEL_HARD` (``gemini-2.5-pro``).
* **別プロンプト (prompt diversity)** — :data:`VERIFIER_SYSTEM_PROMPT` frames the task as an independent
  re-derivation with a skeptical stance (enumeration completeness / rounding / 該当なし discipline),
  not the investigator's phrasing.
* **独立ツール経路 (tool-path independence)** — the verifier runs its *own* investigation with a fresh
  :class:`~src.rag.tools.profile.CorpusProfile`, so no self-discovered secret or cached state leaks from
  the investigator's run into the verifier's.

Crucially the verifier is **blind to the investigator's answer** — it solves the question from scratch,
then :func:`compare_answers` reconciles the two answers *externally*. Feeding the investigator's answer
into the verifier prompt would re-introduce the very correlation this agent exists to break.

The comparison is on four axes drawn from the acceptance criteria:

* **値 (value)** — normalized text equality.
* **列挙完全性 (enumeration completeness)** — 「、」-separated lists compared as sets, reporting the
  investigator's *missing* and *extra* items.
* **数値丸め (numeric rounding)** — single numbers compared with a rounding tolerance (the coarser of the
  two operands' decimal precision), so ``1526`` and ``1526.3`` agree but ``1526`` and ``1530`` do not.
* **該当なし (abstain)** — if exactly one side abstains it is a disagreement; both abstaining agree.

Design mirrors the investigator: the driver is a *pure* function over an abstract
:class:`~src.rag.agent.investigator.Model`, so offline tests script a fake model and never touch Vertex;
:func:`verify_question` injects the live Gemini path.
"""
from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass
from typing import Any, Callable, Sequence

from config import settings
from src.rag.agent.investigator import (
    DEFAULT_MAX_TURNS,
    DEFAULT_TIMEOUT_S,
    AgentTool,
    Investigation,
    Model,
    build_tools,
    investigate,
    is_abstain,
)
from src.rag.tools.profile import CorpusProfile

ABSTAIN = settings.ABSTAIN

# The verifier deliberately runs a DIFFERENT tier than the investigator (pro↔flash divergence): its
# mistakes are uncorrelated with the investigator's, which is the whole point of a heterogeneous check.
DEFAULT_VERIFIER_MODEL = settings.GEN_MODEL  # flash, vs investigator's GEN_MODEL_HARD (pro)

VERIFIER_SYSTEM_PROMPT = (
    "あなたは独立検証エージェントです。別の調査エージェントが既にこの質問に答えていますが、その回答は"
    "一切与えられません。あなたは調査エージェントとは独立に、与えられた汎用ツールだけで自力で答えを"
    "再導出します。目的は、調査エージェントと相関しない観点から誤りを炙り出すことです。\n"
    "厳守事項:\n"
    "1. 暗算・記憶・創作で答えない。必ずツールで根拠となる値を取得してから答える。他者の回答に引きずられ"
    "ない(そもそも与えられない)。\n"
    "2. パスワード・略称・書式規則などのコーパス固有事実は与えられていない。ツール(ファイル探索/grep/"
    "Office抽出/復号/pandas計算/チャート読取/画像説明)で自力発見する。\n"
    "3. 検証では次の3点を特に厳密に確認する: (a) 列挙は漏れ・重複なく完全か(該当を全件挙げたか)、"
    "(b) 数値は正しい丸め・単位・原文表記か、(c) 本当に該当が存在するか(存在しないなら安易に値を作らない)。\n"
    "4. 証拠の優先順位は『computeによる生データ再計算・notebookの実行出力 > 文書中のmarkdown/散文の主張』"
    "とする。相関・統計・集計で矛盾を検出したら散文を根拠から除外し、canonical_routeで元の表データを特定して"
    "computeで独立再計算した結果を採用する。notebook質問では必要に応じて canonical_route(question=..., "
    "kind='train', expr='...') を使う。\n"
    "5. 数値計算(平均・合計・件数・相関など)は必ず compute ツールで行い、列名や絞り込み値が不明なときは"
    "`df.columns.tolist()` や `df['列'].unique().tolist()` で先に確認してから式を組む。\n"
    "6. 十分な根拠が得られたら、最終回答は必ず submit_answer ツールを1回だけ呼んで返す。answer=回答本文"
    "(値/一覧のみ、列挙は「、」区切り、金額は原文表記)、confidence=0.0〜1.0、evidence=根拠、method=導出手順。\n"
    f"7. あらゆる手段を尽くしても根拠が該当しない/得られない場合に限り answer=「{ABSTAIN}」・"
    "confidence=0.0 で submit_answer する(存在しない値を捏造しない)。"
)

# Comparison outcome categories (the axis on which agreement / disagreement was decided).
CATEGORY_ABSTAIN = "abstain"
CATEGORY_ENUMERATION = "enumeration"
CATEGORY_NUMERIC = "numeric"
CATEGORY_VALUE = "value"


# --------------------------------------------------------------------------- text/number normalization
_NUMBER_RE = re.compile(r"[-+]?\d[\d,]*(?:\.\d+)?")
_ENUM_SPLIT_RE = re.compile(r"[、\n;；]+")


def _norm(text: Any) -> str:
    """NFKC-normalize, strip and lowercase for tolerant text comparison."""
    return unicodedata.normalize("NFKC", str(text)).strip().lower()


def parse_enumeration(text: Any) -> list[str]:
    """Split an answer into enumerated items on the enumeration separator 「、」 (＋改行/;).

    Deliberately does NOT split on the ASCII/full-width comma, which is a thousands separator inside
    numbers (``1,526``). The investigator prompt fixes 「、」 as the list separator, so this stays aligned.
    Each item is normalized; empties are dropped. A scalar answer yields a single-element list.
    """
    return [item for item in (_norm(p) for p in _ENUM_SPLIT_RE.split(str(text))) if item]


def _number_tokens(text: Any) -> list[str]:
    """Extract numeric tokens (thousands separators removed), preserving decimal precision as text."""
    s = unicodedata.normalize("NFKC", str(text))
    return [m.group().replace(",", "") for m in _NUMBER_RE.finditer(s)]


def _decimals(token: str) -> int:
    return len(token.split(".", 1)[1]) if "." in token else 0


def _rounding_equal(token_a: str, token_b: str) -> bool:
    """Two numeric tokens agree under rounding: round both to the *coarser* observed precision.

    ``1526.3`` vs ``1526`` → round to 0 dp → 1526 == 1526 (agree); ``1526`` vs ``1530`` → 1526 != 1530.
    ``12.34`` vs ``12.3`` → round to 1 dp → 12.3 == 12.3 (agree).
    """
    try:
        a, b = float(token_a), float(token_b)
    except ValueError:
        return False
    nd = min(_decimals(token_a), _decimals(token_b))
    return round(a, nd) == round(b, nd)


# --------------------------------------------------------------------------- comparison result types
@dataclass(frozen=True)
class Comparison:
    """The reconciliation of two answers on one axis."""

    agree: bool
    category: str  # CATEGORY_* — which axis decided the outcome
    reason: str    # human-readable 根拠 for the verdict

    def to_dict(self) -> dict[str, Any]:
        return {"agree": self.agree, "category": self.category, "reason": self.reason}


def compare_answers(investigator_answer: Any, verifier_answer: Any) -> Comparison:
    """Reconcile the investigator's answer with the verifier's independent answer.

    Axis precedence: 該当なし(abstain) → 列挙完全性(enumeration) → 数値丸め(numeric) → 値(value text).
    """
    ia, va = str(investigator_answer), str(verifier_answer)
    ia_abstain, va_abstain = is_abstain(ia), is_abstain(va)

    # 該当なし: one side claims a value, the other says 該当なし → disagreement.
    if ia_abstain or va_abstain:
        if ia_abstain and va_abstain:
            return Comparison(True, CATEGORY_ABSTAIN, "両者とも該当なし(棄権)で一致")
        who = "調査AGのみ" if ia_abstain else "検証AGのみ"
        return Comparison(False, CATEGORY_ABSTAIN,
                          f"{who}が該当なし(棄権) — 片方が値を主張し他方が棄権")

    # 列挙完全性: compare 「、」-lists as sets; surface the investigator's missing / extra items.
    ia_items, va_items = parse_enumeration(ia), parse_enumeration(va)
    if len(ia_items) > 1 or len(va_items) > 1:
        sa, sb = set(ia_items), set(va_items)
        if sa == sb:
            return Comparison(True, CATEGORY_ENUMERATION, f"列挙{len(sa)}件が集合一致")
        missing = sorted(sb - sa)  # verifier found, investigator omitted
        extra = sorted(sa - sb)    # investigator listed, verifier did not
        detail = []
        if missing:
            detail.append(f"調査AG欠落: {', '.join(missing)}")
        if extra:
            detail.append(f"調査AG余剰: {', '.join(extra)}")
        return Comparison(False, CATEGORY_ENUMERATION, "列挙不一致 — " + " / ".join(detail))

    # 数値丸め: single number on each side → compare with rounding tolerance.
    ia_nums, va_nums = _number_tokens(ia), _number_tokens(va)
    if len(ia_nums) == 1 and len(va_nums) == 1:
        if _rounding_equal(ia_nums[0], va_nums[0]):
            return Comparison(True, CATEGORY_NUMERIC,
                              f"数値一致(丸め許容): {ia_nums[0]} ≈ {va_nums[0]}")
        return Comparison(False, CATEGORY_NUMERIC,
                          f"数値不一致: 調査AG={ia_nums[0]} vs 検証AG={va_nums[0]}")

    # 値: normalized text equality.
    if _norm(ia) == _norm(va):
        return Comparison(True, CATEGORY_VALUE, "正規化テキストが一致")
    return Comparison(False, CATEGORY_VALUE, f"値不一致: 調査AG「{ia}」 vs 検証AG「{va}」")


@dataclass(frozen=True)
class Verdict:
    """A verifier verdict for one question: does the independent path agree with the investigator?"""

    question: str
    agree: bool
    category: str
    reason: str
    investigator_answer: str
    verifier_answer: str
    verifier_confidence: float = 0.0
    verifier_evidence: str = ""
    verifier_method: str = ""
    verifier_stop_reason: str = ""

    @property
    def disagree(self) -> bool:
        return not self.agree

    def to_dict(self) -> dict[str, Any]:
        return {
            "question": self.question,
            "agree": self.agree,
            "category": self.category,
            "reason": self.reason,
            "investigator_answer": self.investigator_answer,
            "verifier_answer": self.verifier_answer,
            "verifier_confidence": self.verifier_confidence,
            "verifier_evidence": self.verifier_evidence,
            "verifier_method": self.verifier_method,
            "verifier_stop_reason": self.verifier_stop_reason,
        }


def build_verdict(question: str, investigator_answer: Any, investigation: Investigation) -> Verdict:
    """Assemble a :class:`Verdict` from the verifier's independent ``investigation`` and the
    investigator's answer, comparing the two."""
    va = investigation.answer
    cmp = compare_answers(investigator_answer, va.answer)
    return Verdict(
        question=question,
        agree=cmp.agree,
        category=cmp.category,
        reason=cmp.reason,
        investigator_answer=str(investigator_answer),
        verifier_answer=va.answer,
        verifier_confidence=va.confidence,
        verifier_evidence=va.evidence,
        verifier_method=va.method,
        verifier_stop_reason=investigation.stop_reason,
    )


# --------------------------------------------------------------------------- verifier driver (pure)
def verify_answer(question: str, investigator_answer: Any, *,
                  model: Model | None = None,
                  model_factory: Callable[[str, Sequence[AgentTool]], Model] | None = None,
                  profile_factory: Callable[[], CorpusProfile] | None = None,
                  max_turns: int = DEFAULT_MAX_TURNS,
                  timeout_s: float = DEFAULT_TIMEOUT_S) -> Verdict:
    """Independently re-derive ``question`` and reconcile the result against ``investigator_answer``.

    The verifier gets its own tools bound to a *fresh* :class:`CorpusProfile` (tool-path independence).
    Supply a scripted ``model`` (offline tests) or a ``model_factory`` (live); by default the live
    heterogeneous Gemini path is used. The verifier never sees ``investigator_answer`` — comparison is
    purely external — so its errors stay uncorrelated with the investigator's.
    """
    prof = (profile_factory or CorpusProfile)()
    tools = build_tools(prof)
    if model is None:
        factory = model_factory or verifier_model_factory
        model = factory(question, tools)
    investigation = investigate(model, question, tools, max_turns=max_turns, timeout_s=timeout_s)
    return build_verdict(question, investigator_answer, investigation)


def verify_investigation(investigation: Investigation, **kwargs: Any) -> Verdict:
    """Verify an investigator :class:`Investigation` directly (uses its question + answer)."""
    return verify_answer(investigation.question, investigation.answer.answer, **kwargs)


# --------------------------------------------------------------------------- live Gemini verifier path
def verifier_model_factory(question: str, tools: Sequence[AgentTool], *,
                           model: str | None = None) -> Model:
    """Fresh live Gemini conversation for the verifier: the *flash* tier + the verifier prompt."""
    from src.rag.agent.investigator import GeminiModel, to_genai_tools

    return GeminiModel(question, to_genai_tools(tools),
                       model=model or DEFAULT_VERIFIER_MODEL, system=VERIFIER_SYSTEM_PROMPT)


def verify_question(question: str, investigator_answer: Any, *, model: str | None = None,
                    profile: CorpusProfile | None = None,
                    max_turns: int = DEFAULT_MAX_TURNS,
                    timeout_s: float = DEFAULT_TIMEOUT_S) -> Verdict:
    """Convenience live entry point: independently verify one ``question`` against the investigator's
    answer with a real (heterogeneous) Gemini conversation."""
    return verify_answer(
        question, investigator_answer,
        model_factory=lambda q, t: verifier_model_factory(q, t, model=model),
        profile_factory=(lambda: profile) if profile is not None else None,
        max_turns=max_turns, timeout_s=timeout_s,
    )
