"""SOT-2502 — offline tests for the obligation-driven local re-search loop.

All network-free: the investigator loop is driven by a scripted fake model, obligation decomposition is
injected (deterministic), and the tool layer uses trivial fakes so the director's observe/review/trace
behaviour and its wiring into the investigator are exercised end-to-end without Vertex.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.rag.agent import obligations as ob
from src.rag.agent import research_loop as rl
from src.rag.agent import abstain_ledger as al
from src.rag.agent.investigator import (
    ABSTAIN,
    SUBMIT_ANSWER,
    AgentTool,
    Call,
    Step,
    Usage,
    investigate,
    is_abstain,
)
from src.rag.agent.research_loop import (
    BUDGET,
    UNANSWERABLE,
    KINDS,
    ResearchBudget,
    ResearchDirector,
    TACTICS,
    _succeeded,
)


# --------------------------------------------------------------------------- scripted fake model
class ScriptedModel:
    def __init__(self, steps, *, model_name="fake-model"):
        self._steps = list(steps)
        self._i = 0
        self.model_name = model_name
        self.calls_seen = []

    def next(self, tool_responses):
        self.calls_seen.append(tool_responses)
        if self._i >= len(self._steps):
            return Step(function_calls=(), final_text=ABSTAIN, usage=Usage(1, 1))
        step = self._steps[self._i]
        self._i += 1
        return step


def _submit(answer, *, confidence=0.9, evidence="", method="", usage=Usage(5, 5)) -> Step:
    return Step(function_calls=(Call(SUBMIT_ANSWER, {
        "answer": answer, "confidence": confidence, "evidence": evidence, "method": method}),),
        usage=usage)


def _tool(name, result):
    return AgentTool(name, "d", {"type": "object", "properties": {}}, lambda **kw: result)


def _decompose_two(question):
    """A fixed two-obligation set (SOURCE_LOCATION, then COMPUTATION) for deterministic tests."""
    return ob.ObligationSet(
        question=question, contract="numeric",
        obligations=(
            ob.Obligation("回答値の出典を特定する", ob.SOURCE_LOCATION),
            ob.Obligation("決定論的に再計算する", ob.COMPUTATION),
        ))


# --------------------------------------------------------------------------- taxonomy invariants
def test_every_kind_has_tactics():
    for k in KINDS:
        assert TACTICS.get(k), f"missing tactics for {k}"


def test_succeeded_outcomes():
    assert _succeeded({"value": 20, "evidence": {}, "method": {}}) is True
    assert _succeeded({"value": None, "evidence": {}, "method": {}}) is False
    assert _succeeded({"error": "boom"}) is False
    assert _succeeded(["a", "b"]) is True
    assert _succeeded([]) is False
    assert _succeeded(None) is False
    assert _succeeded("x") is True


def test_budget_coerce():
    assert ResearchBudget.coerce(None) == ResearchBudget()
    assert ResearchBudget.coerce(True) == ResearchBudget()
    assert ResearchBudget.coerce({"max_rounds": 5}).max_rounds == 5
    b = ResearchBudget(max_rounds=3, max_tool_calls=1)
    assert ResearchBudget.coerce(b) is b


# --------------------------------------------------------------------------- director unit behaviour
def test_director_emits_targeted_directive_for_first_unmet_kind():
    d = ResearchDirector("q", decompose=_decompose_two)
    directive = d.review(evidence_text="", tool_call_count=0)
    assert directive is not None
    # first unmet obligation is SOURCE_LOCATION → its tactics appear in the directive
    assert any(t in directive for t in TACTICS[ob.SOURCE_LOCATION])
    assert len(d.rounds) == 1
    assert d.rounds[0].kind == ob.SOURCE_LOCATION


def test_director_progresses_through_kinds_then_unanswerable():
    # budget wide enough that tactic-exhaustion (not the round cap) is what stops re-search
    d = ResearchDirector("q", budget=ResearchBudget(max_rounds=5), decompose=_decompose_two)
    d.review()                       # round 1 → SOURCE_LOCATION
    d.review()                       # round 2 → COMPUTATION
    out = d.review()                 # no fresh kinds left (both targeted)
    assert out is None
    assert d.terminal == UNANSWERABLE
    assert [r.kind for r in d.rounds] == [ob.SOURCE_LOCATION, ob.COMPUTATION]


def test_director_stops_on_round_budget():
    d = ResearchDirector("q", budget=ResearchBudget(max_rounds=1), decompose=_decompose_two)
    assert d.review() is not None    # round 1 allowed
    assert d.review() is None        # budget hit
    assert d.terminal == BUDGET


def test_director_stops_on_tool_call_budget():
    d = ResearchDirector("q", budget=ResearchBudget(max_tool_calls=3), decompose=_decompose_two)
    assert d.review(tool_call_count=3) is None
    assert d.terminal == BUDGET


def test_director_observe_discharges_obligation_kinds():
    d = ResearchDirector("q", decompose=_decompose_two)
    # a successful compute covers COMPUTATION; find_files covers SOURCE_LOCATION → all met → unanswerable
    d.observe("find_files", {"value": ["a.xlsx"], "evidence": {}, "method": {}})
    d.observe("compute", {"value": 20, "evidence": {}, "method": {}})
    assert d.review() is None
    assert d.terminal == UNANSWERABLE
    assert d.rounds == []             # nothing was re-searched (all obligations already covered)


def test_director_ignores_failed_tool_results_as_evidence():
    d = ResearchDirector("q", decompose=_decompose_two)
    d.observe("find_files", {"error": "not found"})   # failure → not evidence
    assert d.review() is not None                     # SOURCE_LOCATION still unmet → re-search fires


# --------------------------------------------------------------------------- investigator integration
def test_research_turns_immediate_abstain_into_answer():
    """即棄権が構造上不可能: a first-turn abstain re-searches, then the model commits a grounded answer."""
    model = ScriptedModel([
        _submit(ABSTAIN),                       # would-be immediate abstain
        _submit("契約金額は1,320,000円", confidence=0.8, evidence="train.xlsx compute"),
    ])
    res = investigate(model, "q", [], max_turns=6, research={"max_rounds": 2})
    assert res.stop_reason == "answered"
    assert res.answer.answer == "契約金額は1,320,000円"
    # the model received the re-search directive as a submit_answer function-response before answering
    delivered = [tr for turn in model.calls_seen if turn for tr in turn]
    assert any(tr.name == SUBMIT_ANSWER and isinstance(tr.response, dict)
               and tr.response.get("abstain_rejected") for tr in delivered)


def test_research_disabled_accepts_abstain_immediately():
    """Non-regression: with research off, a first-turn abstain finalizes in exactly one model turn."""
    model = ScriptedModel([_submit(ABSTAIN), _submit("late answer")])
    res = investigate(model, "q", [], max_turns=6)          # research defaults off
    assert res.stop_reason == "answered"
    assert is_abstain(res.answer.answer)
    assert len(model.calls_seen) == 1                        # never asked to re-search


def test_research_does_not_touch_committed_answer():
    """Non-regression: a committed (non-abstain) answer is unchanged whether research is on or off."""
    model_off = ScriptedModel([_submit("42", confidence=0.7)])
    model_on = ScriptedModel([_submit("42", confidence=0.7)])
    off = investigate(model_off, "q", [], max_turns=3)
    on = investigate(model_on, "q", [], max_turns=3, research=True)
    assert off.answer.to_dict() == on.answer.to_dict()
    assert len(model_on.calls_seen) == 1                     # no extra turn for a committed answer


def test_research_records_history_and_budget_code_in_abstain_ledger(tmp_path: Path):
    """棄権台帳に探索履歴が残り、予算枯渇は BUDGET_EXHAUSTED として計上される。"""
    ledger = tmp_path / "abstain.jsonl"
    model = ScriptedModel([_submit(ABSTAIN), _submit(ABSTAIN), _submit(ABSTAIN)])
    res = investigate(model, "q", [], max_turns=8, ledger=str(ledger),
                      research={"max_rounds": 1})
    assert is_abstain(res.answer.answer)
    rec = json.loads(ledger.read_text(encoding="utf-8").strip())
    al.validate(rec)                                         # still a well-formed ledger record
    assert rec["research"], "re-search rounds must be recorded on the abstain"
    assert rec["research"][0]["kind"] in KINDS
    assert rec["research_terminal"] == "budget"
    assert rec["state_code"] == al.BUDGET_EXHAUSTED


def test_research_exhausted_tactics_records_unanswerable(tmp_path: Path):
    ledger = tmp_path / "abstain.jsonl"
    # inject the flat two-obligation set so both kinds get re-searched then run dry → UNANSWERABLE
    import src.rag.agent.research_loop as rlmod
    from unittest import mock

    def fake_director(question, **kw):
        return ResearchDirector(question, budget=kw.get("budget"), decompose=_decompose_two)

    with mock.patch.object(rlmod, "ResearchDirector", fake_director):
        model = ScriptedModel([_submit(ABSTAIN)] * 5)
        res = investigate(model, "q", [], max_turns=12, ledger=str(ledger),
                          research={"max_rounds": 5})
    rec = json.loads(ledger.read_text(encoding="utf-8").strip())
    assert rec["research_terminal"] == "unanswerable"
    assert rec["state_code"] == al.UNANSWERABLE
    assert len(rec["research"]) == 2                          # both obligation kinds re-searched


def test_classify_research_terminal_precedence():
    s = al.AbstainSignals(research_terminal="budget")
    assert al.classify(s) == al.BUDGET_EXHAUSTED
    s = al.AbstainSignals(research_terminal="unanswerable")
    assert al.classify(s) == al.UNANSWERABLE
    # empty terminal → unchanged legacy behaviour (nothing succeeded → NOT_RETRIEVED)
    assert al.classify(al.AbstainSignals()) == al.NOT_RETRIEVED
