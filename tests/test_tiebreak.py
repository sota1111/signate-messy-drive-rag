"""SOT-2471 — offline tests for the third-judge tie-break + abstain fallback.

Network-free: the investigator answer is a hand-built :class:`Investigation`, and the verifier's and
judge's independent re-derivations are driven by *scripted* fake models (same shape as the investigator),
so no Vertex call happens. Covers the acceptance criterion — 対立を正しく決着 or 棄権 — across each branch
(一致→採用 / 不一致→第3判定で決着 / 決着不能→棄権), plus the invariant that the judge is only consulted on
disagreement.
"""
from __future__ import annotations

from config import settings
from src.rag.agent.investigator import ABSTAIN, SUBMIT_ANSWER, Answer, Call, Investigation, Step, Usage
from src.rag.agent import tiebreak as tb
from src.rag.agent.tiebreak import (
    DEFAULT_JUDGE_MODEL,
    STATUS_ABSTAINED,
    STATUS_AGREED,
    STATUS_TIEBREAK_INVESTIGATOR,
    STATUS_TIEBREAK_VERIFIER,
    Resolution,
    resolve_answer,
)


# --------------------------------------------------------------------------- scripted fake model
class ScriptedModel:
    """A fake :class:`~src.rag.agent.investigator.Model` that replays one ``submit_answer`` step."""

    def __init__(self, answer, *, confidence=0.9, evidence="e", method="m", model_name="fake"):
        self._answer = answer
        self._confidence = confidence
        self._evidence = evidence
        self._method = method
        self.model_name = model_name
        self.calls = 0

    def next(self, tool_responses):
        self.calls += 1
        return Step(function_calls=(Call(SUBMIT_ANSWER, {
            "answer": self._answer, "confidence": self._confidence,
            "evidence": self._evidence, "method": self._method}),), usage=Usage(50, 10))


def _investigation(answer, *, confidence=0.9, evidence="ie", method="im") -> Investigation:
    """A completed investigator :class:`Investigation` carrying ``answer``."""
    return Investigation(
        question="Q?", answer=Answer(answer=answer, confidence=confidence,
                                     evidence=evidence, method=method),
        iterations=1, tool_calls=[], usage=Usage(100, 20), model="gemini-2.5-pro",
        elapsed_s=0.1, stop_reason="answered",
    )


def _resolve(inv_answer, ver_answer, judge_answer=None, **kw):
    """Drive ``resolve_answer`` with scripted verifier/judge models."""
    return resolve_answer(
        "Q?", _investigation(inv_answer, **kw),
        verifier_model=ScriptedModel(ver_answer),
        judge_model=(ScriptedModel(judge_answer) if judge_answer is not None else None),
        max_turns=3,
    )


# --------------------------------------------------------------------------- 一致 → 採用
def test_agreement_adopts_the_primary_answer_without_a_tie_break():
    r = _resolve("1526", "1526.0")  # numeric-rounding agreement; no judge supplied → must not be needed
    assert isinstance(r, Resolution)
    assert r.status == STATUS_AGREED and r.agree
    assert r.answer == "1526"          # primary(investigator) answer adopted verbatim
    assert r.confidence == 0.9
    assert r.judge_answer is None      # tie-break did NOT run
    assert not r.abstained and not r.tie_broken


def test_agreement_carries_the_investigator_evidence_and_method():
    # Ｔｏｋｙｏ vs tokyo agree on the value axis after NFKC; the adopted answer keeps the primary's 根拠.
    r = _resolve("Ｔｏｋｙｏ", "tokyo", evidence="file.xlsx A1", method="grep→read")
    assert r.status == STATUS_AGREED
    assert r.evidence == "file.xlsx A1" and r.method == "grep→read"


# --------------------------------------------------------------------------- 不一致 → 第3判定で決着
def test_disagreement_judge_backs_verifier_adopts_verifier_answer():
    """受け入れ条件: 既知の対立サンプルで正answerが選ばれる。

    Investigator answered 1530 (wrong); verifier independently got 1526; the third judge also derives
    1526 → the tie breaks toward the verifier and 1526 is adopted.
    """
    r = _resolve("1530", "1526", judge_answer="1526")
    assert r.status == STATUS_TIEBREAK_VERIFIER and r.tie_broken
    assert r.answer == "1526"
    assert not r.agree and not r.abstained
    assert "検証AG" in r.judge_reason
    assert r.judge_answer == "1526"


def test_disagreement_judge_backs_investigator_adopts_investigator_answer():
    r = _resolve("1530", "1526", judge_answer="1530")
    assert r.status == STATUS_TIEBREAK_INVESTIGATOR and r.tie_broken
    assert r.answer == "1530"
    assert "調査AG" in r.judge_reason


def test_disagreement_enumeration_tie_break_picks_the_complete_list():
    # investigator missed C社; verifier + judge both list all three → verifier side wins.
    r = _resolve("A社、B社", "A社、B社、C社", judge_answer="C社、A社、B社")
    assert r.status == STATUS_TIEBREAK_VERIFIER
    assert r.answer == "A社、B社、C社"


# --------------------------------------------------------------------------- 決着不能 → 棄権
def test_judge_matches_neither_candidate_abstains():
    # investigator 1530, verifier 1526, judge lands on an independent third value 1600 → 決着不能.
    r = _resolve("1530", "1526", judge_answer="1600")
    assert r.status == STATUS_ABSTAINED and r.abstained
    assert r.answer == ABSTAIN and r.confidence == 0.0
    assert "決着不能" in r.reason


def test_judge_abstains_falls_back_to_abstain():
    # A judge that itself cannot find an answer matches neither candidate → 棄権.
    r = _resolve("1530", "1526", judge_answer=ABSTAIN)
    assert r.status == STATUS_ABSTAINED
    assert r.answer == ABSTAIN


def test_investigator_hallucination_vs_verifier_abstain_tie_broken_to_abstain():
    # investigator invented 9999; verifier abstained (disagreement on abstain axis); judge also abstains
    # → verifier & judge agree there is no value, so the fabrication is rejected and the final answer is
    # 棄権 (the tie broke toward the abstaining verifier — abstained is answer-based, not status-based).
    r = _resolve("9999", ABSTAIN, judge_answer=ABSTAIN)
    assert r.answer == ABSTAIN and r.abstained
    assert r.status == STATUS_TIEBREAK_VERIFIER  # decision-path label: judge sided with the verifier


def test_investigator_abstain_recovered_when_judge_confirms_verifier_value():
    # investigator abstained, verifier found 3件, judge independently confirms 3件 → recover the value.
    r = _resolve(ABSTAIN, "3件", judge_answer="3件")
    assert r.status == STATUS_TIEBREAK_VERIFIER
    assert r.answer == "3件"


# --------------------------------------------------------------------------- judge invocation invariant
def test_judge_is_not_consulted_when_the_first_two_agree():
    judge = ScriptedModel("SHOULD-NOT-BE-CALLED")
    r = resolve_answer("Q?", _investigation("1526"),
                       verifier_model=ScriptedModel("1526"), judge_model=judge, max_turns=3)
    assert r.status == STATUS_AGREED
    assert judge.calls == 0  # tie-break judge must not run on agreement (no wasted Gemini call)


def test_disagreement_without_a_judge_defensively_abstains():
    # If the two disagree and no judge is available, the safe outcome is 棄権 (never guess).
    r = resolve_answer("Q?", _investigation("1530"),
                       verifier_model=ScriptedModel("1526"), judge_model=None, max_turns=3)
    assert r.status == STATUS_ABSTAINED and r.answer == ABSTAIN


# --------------------------------------------------------------------------- pure tie-break unit
def test_tie_break_matching_both_candidates_is_a_contradiction_abstain():
    # Judge agrees with both candidates while they disagreed with each other → inconsistent → 棄権.
    status, reason = tb._tie_break(Answer("x", 0.9), Answer("x", 0.9), Answer("x", 0.9))
    assert status == STATUS_ABSTAINED and "矛盾" in reason


# --------------------------------------------------------------------------- config / schema
def test_judge_defaults_to_the_dedicated_judge_model():
    assert DEFAULT_JUDGE_MODEL == settings.JUDGE_MODEL


def test_judge_prompt_is_distinct_from_investigator_and_verifier_prompts():
    from src.rag.agent.investigator import SYSTEM_PROMPT
    from src.rag.agent.verifier import VERIFIER_SYSTEM_PROMPT

    assert tb.JUDGE_SYSTEM_PROMPT not in (SYSTEM_PROMPT, VERIFIER_SYSTEM_PROMPT)
    assert "審判" in tb.JUDGE_SYSTEM_PROMPT


def test_resolution_to_dict_and_to_answer_are_complete():
    r = _resolve("1530", "1526", judge_answer="1526")
    d = r.to_dict()
    assert set(d) >= {
        "question", "answer", "confidence", "status", "reason", "agree",
        "investigator_answer", "verifier_answer", "judge_answer", "judge_reason", "verdict",
    }
    assert d["verdict"]["agree"] is False
    ans = r.to_answer()
    assert ans.answer == "1526" and ans.confidence == r.confidence
