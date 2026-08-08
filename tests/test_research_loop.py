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
    Answer,
    AgentTool,
    Call,
    Step,
    Usage,
    _budget_boundary_directive,
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


# --------------------------------------------------------------------------- SOT-2524 budget-boundary hook
def _call(name, **args):
    """A model step that invokes one non-terminal tool (burns a turn without committing/abstaining)."""
    return Step(function_calls=(Call(name, args),), usage=Usage(2, 2))


def _inert_tool(name="chunk_search"):
    """A tool whose name is absent from ``_TOOL_KINDS`` so its result discharges no obligation kind."""
    return _tool(name, {"hits": []})


def test_review_at_boundary_tags_round():
    """SOT-2524: a boundary-triggered review tags its round so the ledger can attribute it."""
    d = ResearchDirector("q", decompose=_decompose_two)
    assert d.review(at_boundary=True) is not None
    assert d.rounds[0].boundary is True
    assert d.rounds[0].to_dict()["boundary"] is True
    # a plain (deliberate-abstain) review stays untagged
    d2 = ResearchDirector("q", decompose=_decompose_two)
    d2.review()
    assert d2.rounds[0].boundary is False


def test_budget_boundary_directive_helper_gating():
    """The helper only asks the director when enabled, active, and the pending answer is an abstain."""
    d = ResearchDirector("q", decompose=_decompose_two)
    abstain = Answer(answer=ABSTAIN, confidence=0.0)
    committed = Answer(answer="42", confidence=0.7)
    assert _budget_boundary_directive(None, True, [], abstain) is None          # no director
    assert _budget_boundary_directive(d, False, [], abstain) is None            # disabled
    assert _budget_boundary_directive(d, True, [], committed) is None           # not an abstain
    assert not d.rounds                                                         # none of the above searched
    directive = _budget_boundary_directive(d, True, ["chunk_search"], abstain)  # enabled + abstain + unmet
    assert directive is not None
    assert d.rounds and d.rounds[0].boundary is True


def test_budget_boundary_hook_researches_at_max_turns_then_answers():
    """A model that wanders out of turns (never a deliberate abstain) still gets one bounded re-search
    push at the max_turns boundary, and a grounded answer found there is committed."""
    import src.rag.agent.research_loop as rlmod
    from unittest import mock

    def fake_director(question, **kw):
        return ResearchDirector(question, budget=kw.get("budget"), decompose=_decompose_two)

    with mock.patch.object(rlmod, "ResearchDirector", fake_director):
        model = ScriptedModel([
            _call("chunk_search"), _call("chunk_search"), _call("chunk_search"),  # burn all 3 turns
            _submit("契約書は総務部フォルダにあります", confidence=0.8, evidence="find_files"),
        ])
        res = investigate(model, "q", [_inert_tool()], max_turns=3,
                          research={"max_rounds": 2}, budget_boundary=True)
    assert res.stop_reason == "answered"
    assert res.answer.answer == "契約書は総務部フォルダにあります"
    delivered = [tr for turn in model.calls_seen if turn for tr in turn]
    assert any(tr.name == SUBMIT_ANSWER and isinstance(tr.response, dict)
               and tr.response.get("abstain_rejected") for tr in delivered)


def test_budget_boundary_disabled_finalizes_abstain_at_max_turns():
    """Non-regression: with the hook off (default), max_turns exhaustion finalizes the abstain in exactly
    max_turns model turns — no boundary directive, byte-identical to the pre-hook loop."""
    model = ScriptedModel([_call("chunk_search")] * 3 + [_submit("late")])
    res = investigate(model, "q", [_inert_tool()], max_turns=3, research=True)  # budget_boundary defaults off
    assert res.stop_reason == "max_turns"
    assert is_abstain(res.answer.answer)
    assert len(model.calls_seen) == 3


def test_budget_boundary_hook_inert_without_research():
    """The hook is a no-op when the re-search director is off: no directive, plain max_turns abstain."""
    model = ScriptedModel([_call("chunk_search")] * 3 + [_submit("late")])
    res = investigate(model, "q", [_inert_tool()], max_turns=3, budget_boundary=True)  # research off
    assert res.stop_reason == "max_turns"
    assert is_abstain(res.answer.answer)
    assert len(model.calls_seen) == 3


def test_budget_boundary_records_boundary_round_in_ledger(tmp_path: Path):
    """A boundary re-search that still fails records a boundary-tagged round and a coded abstain."""
    import src.rag.agent.research_loop as rlmod
    from unittest import mock

    ledger = tmp_path / "abstain.jsonl"

    def fake_director(question, **kw):
        return ResearchDirector(question, budget=kw.get("budget"), decompose=_decompose_two)

    with mock.patch.object(rlmod, "ResearchDirector", fake_director):
        model = ScriptedModel([_call("chunk_search")] * 3 + [_submit(ABSTAIN)] * 4)
        res = investigate(model, "q", [_inert_tool()], max_turns=3, ledger=str(ledger),
                          research={"max_rounds": 2}, budget_boundary=True)
    assert is_abstain(res.answer.answer)
    rec = json.loads(ledger.read_text(encoding="utf-8").strip())
    al.validate(rec)
    assert rec["research"], "the boundary re-search must be recorded on the abstain"
    assert any(r.get("boundary") for r in rec["research"]), "a round must be attributed to the boundary hook"
    assert rec["research_terminal"] in ("budget", "unanswerable")
    assert rec["state_code"] in (al.BUDGET_EXHAUSTED, al.UNANSWERABLE)
