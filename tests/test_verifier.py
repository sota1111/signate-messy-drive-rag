"""SOT-2470 — offline tests for the heterogeneous verification agent.

Network-free: the verifier's independent re-derivation is driven by a *scripted* fake model (same
shape as the investigator's), so no Vertex call happens. The comparison layer (value / enumeration /
numeric-rounding / abstain) is unit-tested directly, and the end-to-end acceptance test drives a known
wrong-answer sample and asserts the verifier reports 不一致.
"""
from __future__ import annotations

import pytest

from config import settings
from src.rag.agent.investigator import ABSTAIN, SUBMIT_ANSWER, Call, Step, Usage
from src.rag.agent import verifier as vf
from src.rag.agent.verifier import (
    Comparison,
    Verdict,
    compare_answers,
    parse_enumeration,
    verify_answer,
    verify_investigation,
)


# --------------------------------------------------------------------------- scripted fake model
class ScriptedModel:
    """A fake :class:`~src.rag.agent.investigator.Model` that replays a fixed list of ``Step`` s."""

    def __init__(self, steps, *, model_name="fake-verifier"):
        self._steps = list(steps)
        self._i = 0
        self.model_name = model_name

    def next(self, tool_responses):
        if self._i >= len(self._steps):
            return Step(function_calls=(), final_text=ABSTAIN, usage=Usage(1, 1))
        step = self._steps[self._i]
        self._i += 1
        return step


def _submit(answer, *, confidence=0.9, evidence="e", method="m") -> Step:
    return Step(function_calls=(Call(SUBMIT_ANSWER, {
        "answer": answer, "confidence": confidence, "evidence": evidence, "method": method}),),
        usage=Usage(50, 10))


def _model_answering(answer, **kw):
    return ScriptedModel([_submit(answer, **kw)])


# --------------------------------------------------------------------------- heterogeneity invariant
def test_verifier_defaults_to_a_different_model_tier_than_investigator():
    # 異種検証の要: verifier は調査AG(pro=GEN_MODEL_HARD)と別tier(flash=GEN_MODEL)を既定にする。
    assert vf.DEFAULT_VERIFIER_MODEL == settings.GEN_MODEL
    assert vf.DEFAULT_VERIFIER_MODEL != settings.GEN_MODEL_HARD


def test_verifier_prompt_is_distinct_and_does_not_embed_the_investigator_answer():
    # 別プロンプト、かつ調査AGの回答をプロンプトに埋め込まない(独立性の担保)。
    from src.rag.agent.investigator import SYSTEM_PROMPT

    assert vf.VERIFIER_SYSTEM_PROMPT != SYSTEM_PROMPT
    assert "検証" in vf.VERIFIER_SYSTEM_PROMPT


# --------------------------------------------------------------------------- enumeration parsing
def test_parse_enumeration_splits_on_ideographic_comma_only():
    assert parse_enumeration("A社、B社、C社") == ["a社", "b社", "c社"]


def test_parse_enumeration_keeps_thousands_separated_number_as_one_item():
    # ASCII/全角カンマは桁区切り。列挙分割してはならない。
    assert parse_enumeration("1,526") == ["1526"] or parse_enumeration("1,526") == ["1,526".lower()]
    assert len(parse_enumeration("1,526円")) == 1


# --------------------------------------------------------------------------- value axis
def test_compare_value_agree_after_normalization():
    c = compare_answers("Ｔｏｋｙｏ", "tokyo")  # full-width vs ascii → NFKC normalizes
    assert c.agree and c.category == "value"


def test_compare_value_disagree():
    c = compare_answers("東京", "大阪")
    assert not c.agree and c.category == "value"
    assert "東京" in c.reason and "大阪" in c.reason


# --------------------------------------------------------------------------- numeric-rounding axis
@pytest.mark.parametrize("a,b", [("1526", "1526.3"), ("12.34", "12.3"), ("1,526円", "1526")])
def test_compare_numeric_agrees_within_rounding(a, b):
    c = compare_answers(a, b)
    assert c.agree and c.category == "numeric"


@pytest.mark.parametrize("a,b", [("1526", "1530"), ("100", "101"), ("12.5", "12.4")])
def test_compare_numeric_disagrees_beyond_rounding(a, b):
    c = compare_answers(a, b)
    assert not c.agree and c.category == "numeric"


# --------------------------------------------------------------------------- enumeration axis
def test_compare_enumeration_agrees_as_a_set_regardless_of_order():
    c = compare_answers("A、B、C", "C、A、B")
    assert c.agree and c.category == "enumeration"


def test_compare_enumeration_disagrees_and_reports_missing_and_extra():
    # investigator missed C, and claimed a spurious D.
    c = compare_answers("A、B、D", "A、B、C")
    assert not c.agree and c.category == "enumeration"
    assert "欠落" in c.reason and "c" in c.reason      # verifier found C, investigator omitted it
    assert "余剰" in c.reason and "d" in c.reason      # investigator listed D, verifier did not


# --------------------------------------------------------------------------- abstain axis
def test_compare_both_abstain_agree():
    c = compare_answers(ABSTAIN, "わかりません")
    assert c.agree and c.category == "abstain"


def test_compare_one_side_abstains_is_a_disagreement():
    c = compare_answers("42件", ABSTAIN)
    assert not c.agree and c.category == "abstain"
    assert "調査AGのみ" not in c.reason  # investigator gave a value; verifier abstained
    assert "検証AGのみ" in c.reason


def test_compare_investigator_abstains_but_verifier_finds_a_value():
    c = compare_answers(ABSTAIN, "3件")
    assert not c.agree and c.category == "abstain"
    assert "調査AGのみ" in c.reason


# --------------------------------------------------------------------------- end-to-end driver
def test_verify_answer_agrees_when_independent_rederivation_matches():
    verdict = verify_answer("平均は?", "1526", model=_model_answering("1526.0"), max_turns=3)
    assert isinstance(verdict, Verdict)
    assert verdict.agree and verdict.category == "numeric"
    assert verdict.investigator_answer == "1526" and verdict.verifier_answer == "1526.0"
    assert verdict.verifier_stop_reason == "answered"
    # full serializable schema
    assert set(verdict.to_dict()) >= {
        "question", "agree", "category", "reason", "investigator_answer", "verifier_answer",
    }


def test_verify_answer_detects_disagreement_on_a_known_wrong_sample():
    """受け入れ条件: 調査AGと独立経路で判定し、既知の誤答サンプルで不一致を返せる。

    The investigator answered ``1530`` (wrong); the independent verifier re-derives ``1526`` and the
    verdict is 不一致 on the numeric axis — the whole reason the heterogeneous check exists.
    """
    verdict = verify_answer("loan_amntの平均は?", "1530", model=_model_answering("1526"), max_turns=3)
    assert verdict.disagree
    assert verdict.category == "numeric"
    assert "1530" in verdict.reason and "1526" in verdict.reason


def test_verify_answer_detects_enumeration_incompleteness():
    # investigator gave an incomplete list; verifier's complete list exposes the missing member.
    verdict = verify_answer("該当企業は?", "A社、B社", model=_model_answering("A社、B社、C社"), max_turns=3)
    assert verdict.disagree and verdict.category == "enumeration"
    assert "欠落" in verdict.reason


def test_verify_answer_flags_investigator_hallucination_vs_verifier_abstain():
    # investigator invented a value where the verifier (correctly) finds nothing.
    verdict = verify_answer("存在しない列の合計?", "9999", model=_model_answering(ABSTAIN), max_turns=3)
    assert verdict.disagree and verdict.category == "abstain"


def test_verify_answer_uses_a_fresh_isolated_profile_per_call():
    seen = []

    def profile_factory():
        from src.rag.tools.profile import CorpusProfile
        p = CorpusProfile()
        seen.append(p)
        return p

    verify_answer("q", "x", model=_model_answering("x"), profile_factory=profile_factory, max_turns=2)
    assert len(seen) == 1  # the verifier built its own tools from an independent profile


def test_verify_investigation_uses_its_question_and_answer():
    from src.rag.agent.investigator import Answer, Investigation

    investigation = Investigation(
        question="件数は?", answer=Answer(answer="10件", confidence=0.8),
        iterations=1, tool_calls=[], usage=Usage(), model="gemini-2.5-pro",
        elapsed_s=0.1, stop_reason="answered",
    )
    verdict = verify_investigation(investigation, model=_model_answering("11件"), max_turns=3)
    assert verdict.question == "件数は?"
    assert verdict.investigator_answer == "10件"
    assert verdict.disagree and verdict.category == "numeric"
