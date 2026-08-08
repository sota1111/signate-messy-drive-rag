"""SOT-2525 — deterministic tool fallback before concluding UNANSWERABLE (棄権 方策D).

Parent SOT-2460 / Step12. Offline, network-free: the investigator loop is driven by a scripted fake
model and fake deterministic tools, so the fallback mechanism is exercised end-to-end without Vertex.

Two layers are covered:
* the pure plan/gate in :mod:`src.rag.agent.question_contract`
  (:func:`deterministic_fallback_plan` / :class:`DeterministicFallbackGate`), and
* the investigator wiring — the forced one deterministic tool + evidence re-prompt before an abstain
  is accepted, its one-shot property, and the byte-identical default (``fallback=None``).
"""
from __future__ import annotations

from src.rag.agent import investigator as inv
from src.rag.agent import question_contract as qc
from src.rag.agent import routing as _routing
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


# --------------------------------------------------------------------------- scripted fake model
class ScriptedModel:
    """Replays a fixed list of ``Step`` s (mirrors tests/test_investigator.py)."""

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


def _submit(answer, *, confidence=0.9, evidence="e", method="m") -> Step:
    return Step(function_calls=(Call(SUBMIT_ANSWER, {
        "answer": answer, "confidence": confidence, "evidence": evidence, "method": method}),),
        usage=Usage(50, 10))


def _route_tool(value):
    return AgentTool(
        "canonical_route", "d", {"type": "object", "properties": {}},
        lambda **kw: {"value": value, "evidence": {}, "method": {"engine": "canonical_route"}},
    )


def _delivered_submits(model):
    return [r.response for turn in model.calls_seen if turn for r in turn
            if r.name == SUBMIT_ANSWER]


# --------------------------------------------------------------------------- pure plan / gate
def test_plan_version_diff_contract_routes_to_version_diff():
    assert qc.deterministic_fallback_plan("旧版と新版の差分は？", "version_diff") == (
        "version_diff", {"question": "旧版と新版の差分は？"})


def test_plan_numeric_contract_routes_to_canonical_route():
    q = "京橋案件の学習データの行数は？"
    assert qc.deterministic_fallback_plan(q, "numeric") == ("canonical_route", {"question": q})


def test_plan_data_asset_lookup_routes_to_canonical_route():
    # A simple_lookup whose evidence lives in a named canonical data asset — the retrieval_miss core.
    q = "京橋案件の train.xlsx に記録された n_estimators は？"
    assert qc.deterministic_fallback_plan(q, "simple_lookup") == (
        "canonical_route", {"question": q})


def test_plan_literal_lookup_routes_to_file_grep():
    q = "「重要」とそのまま記載されている箇所を教えてください。"
    assert _routing.references_data_asset(q) is False  # no canonical asset noun ⇒ grep is the fallback
    plan = qc.deterministic_fallback_plan(q, "simple_lookup")
    assert plan == ("file_grep", {"query": "重要"})


def test_plan_returns_none_when_no_deterministic_route():
    # A spatial question with no data asset and no literal request has no self-resolving tool.
    assert qc.deterministic_fallback_plan("向かいの席の人の内線は？", "spatial") is None


def test_gate_is_one_shot():
    gate = qc.DeterministicFallbackGate("旧版と新版の差分は？", "version_diff")
    assert gate.plan() == ("version_diff", {"question": "旧版と新版の差分は？"})
    assert gate.plan() is None  # already tried → never fires again


def test_gate_directive_carries_tool_output_and_no_answer():
    gate = qc.DeterministicFallbackGate("q", "numeric")
    directive = gate.directive("canonical_route", {"value": [{"rel": "train.xlsx"}]})
    assert "canonical_route" in directive
    assert "train.xlsx" in directive
    assert "棄権" in directive  # tells the model to keep abstaining if evidence is insufficient


# --------------------------------------------------------------------------- investigator wiring
def test_fallback_runs_deterministic_tool_and_reprompts_before_abstain():
    route = _route_tool([{"rel": "train.xlsx", "project": "P"}])
    model = ScriptedModel([
        _submit(ABSTAIN),   # model about to abstain
        _submit("42"),      # after the forced fallback evidence, it answers
    ])
    gate = qc.DeterministicFallbackGate("京橋案件の学習データの行数は？", "numeric")
    result = investigate(
        model, "京橋案件の学習データの行数は？", [route, inv.SUBMIT_ANSWER_TOOL],
        contract="numeric", fallback=gate, max_turns=5)
    assert result.answer.answer == "42"
    assert result.tool_calls.count("canonical_route") == 1
    delivered = _delivered_submits(model)
    assert any(r.get("abstain_rejected") for r in delivered)


def test_fallback_empty_evidence_keeps_abstain_and_runs_tool_once():
    route = _route_tool([])  # empty value ⇒ not useful ⇒ abstain must stand
    model = ScriptedModel([_submit(ABSTAIN)])
    gate = qc.DeterministicFallbackGate("京橋案件の学習データの行数は？", "numeric")
    result = investigate(
        model, "京橋案件の学習データの行数は？", [route, inv.SUBMIT_ANSWER_TOOL],
        contract="numeric", fallback=gate, max_turns=5)
    assert is_abstain(result.answer.answer)
    assert result.tool_calls.count("canonical_route") == 1  # forced exactly once
    # No abstain re-prompt was delivered because the deterministic route reached nothing.
    assert not any(r.get("abstain_rejected") for r in _delivered_submits(model))


def test_fallback_none_is_byte_identical_default():
    route = _route_tool([{"rel": "train.xlsx"}])
    model = ScriptedModel([_submit(ABSTAIN)])
    result = investigate(
        model, "京橋案件の学習データの行数は？", [route, inv.SUBMIT_ANSWER_TOOL],
        contract="numeric", fallback=None, max_turns=5)
    assert is_abstain(result.answer.answer)
    assert "canonical_route" not in result.tool_calls  # no forced tool without the gate


def test_fallback_does_not_fire_on_a_committed_answer():
    route = _route_tool([{"rel": "train.xlsx"}])
    model = ScriptedModel([_submit("1234", evidence="行数", method="compute")])
    gate = qc.DeterministicFallbackGate("京橋案件の学習データの行数は？", "numeric")
    result = investigate(
        model, "京橋案件の学習データの行数は？", [route, inv.SUBMIT_ANSWER_TOOL],
        contract="numeric", fallback=gate, max_turns=5)
    assert result.answer.answer == "1234"
    assert "canonical_route" not in result.tool_calls  # only fires when about to abstain
    assert gate.plan() is not None  # gate never consumed on a committed answer
