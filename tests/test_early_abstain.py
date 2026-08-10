"""SOT-2599 — offline tests for the early-abstain recalibration (RAG_EARLY_ABSTAIN_RECAL).

All network-free: the pure decision/directive helpers are exercised directly, the research director is
driven with an injected (empty) obligation decomposition so the fresh-empty → UNANSWERABLE branch is hit
deterministically, and the investigator loop is driven by a scripted fake model. No Vertex.
"""
from __future__ import annotations

from src.rag.agent import early_abstain as ea
from src.rag.agent import evidence_packet as ep
from src.rag.agent import obligations as ob
from src.rag.agent import research_loop as rl
from src.rag.agent.research_loop import UNANSWERABLE, ResearchDirector
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


def _no_obligations(question):
    """An empty obligation set → ``fresh`` is always empty → the review hits the UNANSWERABLE branch."""
    return ob.ObligationSet(question=question, contract="x", obligations=())


# --------------------------------------------------------------------------- pure helpers
def test_disabled_by_default():
    assert ea.enabled() is False


def test_enabled_reads_env(monkeypatch):
    monkeypatch.setenv("RAG_EARLY_ABSTAIN_RECAL", "1")
    assert ea.enabled() is True
    monkeypatch.setenv("RAG_EARLY_ABSTAIN_RECAL", "off")
    assert ea.enabled() is False


def test_cheap_deterministic_tools():
    for name in ("canonical_route", "file_grep", "find_files"):
        assert ea.is_cheap_deterministic_tool(name) is True
    # free chunk retrieval / value tools are deliberately NOT cheap deterministic probes
    for name in ("compute", "read_office", "retrieve", "read_chart_values"):
        assert ea.is_cheap_deterministic_tool(name) is False


def test_may_finalize_off_is_always_true():
    # recal disabled → finalize UNANSWERABLE unchanged regardless of the other signals
    assert ea.may_finalize_unanswerable(
        tool_call_count=0, deterministic_probe_tried=False, probe_forced=False,
        recal_enabled=False) is True


def test_may_finalize_on_early_no_probe_forbids():
    assert ea.may_finalize_unanswerable(
        tool_call_count=0, deterministic_probe_tried=False, probe_forced=False,
        recal_enabled=True) is False
    assert ea.may_finalize_unanswerable(
        tool_call_count=ea.EARLY_ABSTAIN_MAX_CALLS, deterministic_probe_tried=False,
        probe_forced=False, recal_enabled=True) is False


def test_may_finalize_on_allows_when_probe_tried_or_forced_or_late():
    # a probe was already tried → allow
    assert ea.may_finalize_unanswerable(
        tool_call_count=0, deterministic_probe_tried=True, probe_forced=False,
        recal_enabled=True) is True
    # already forced once (one-shot) → allow (never loop)
    assert ea.may_finalize_unanswerable(
        tool_call_count=0, deterministic_probe_tried=False, probe_forced=True,
        recal_enabled=True) is True
    # past the early window → allow
    assert ea.may_finalize_unanswerable(
        tool_call_count=ea.EARLY_ABSTAIN_MAX_CALLS + 1, deterministic_probe_tried=False,
        probe_forced=False, recal_enabled=True) is True


def test_directives_name_the_cheap_tools():
    for text in (ea.probe_directive(), ea.packet_relaxation_clause()):
        for name in ("canonical_route", "file_grep", "find_files"):
            assert name in text


# --------------------------------------------------------------------------- research director unit
def test_director_off_finalizes_early_unanswerable():
    """Byte-identical OFF: an empty-obligation review finalizes UNANSWERABLE immediately (no probe)."""
    d = ResearchDirector("q", decompose=_no_obligations)
    assert d.review(tool_call_count=0) is None
    assert d.terminal == UNANSWERABLE
    assert d.rounds == []


def test_director_on_forces_one_probe_then_unanswerable(monkeypatch):
    monkeypatch.setenv("RAG_EARLY_ABSTAIN_RECAL", "1")
    d = ResearchDirector("q", decompose=_no_obligations)
    directive = d.review(tool_call_count=0)
    assert directive is not None                       # forced a deterministic probe instead of abstaining
    assert d.terminal == ""                            # not finalized yet
    assert len(d.rounds) == 1 and d.rounds[0].kind == ea.PROBE_KIND
    # one-shot: a second review does NOT force another probe, it finalizes UNANSWERABLE
    assert d.review(tool_call_count=0) is None
    assert d.terminal == UNANSWERABLE
    assert len(d.rounds) == 1                          # no second probe round appended


def test_director_on_skips_probe_when_deterministic_already_tried(monkeypatch):
    monkeypatch.setenv("RAG_EARLY_ABSTAIN_RECAL", "1")
    d = ResearchDirector("q", decompose=_no_obligations)
    # a deterministic probe was already tried (even a failed one counts) → no forced probe
    d.observe("file_grep", {"error": "not found"})
    assert d.review(tool_call_count=1) is None
    assert d.terminal == UNANSWERABLE
    assert d.rounds == []


def test_director_on_skips_probe_when_late(monkeypatch):
    monkeypatch.setenv("RAG_EARLY_ABSTAIN_RECAL", "1")
    d = ResearchDirector("q", decompose=_no_obligations)
    assert d.review(tool_call_count=ea.EARLY_ABSTAIN_MAX_CALLS + 1) is None
    assert d.terminal == UNANSWERABLE
    assert d.rounds == []


# --------------------------------------------------------------------------- investigator integration
def test_investigate_recal_forces_probe_then_answer(monkeypatch):
    """Recal ON: an iters≤2 would-be UNANSWERABLE is turned into a probe round, then a grounded answer."""
    monkeypatch.setenv("RAG_EARLY_ABSTAIN_RECAL", "1")
    monkeypatch.setattr(ob, "decompose", _no_obligations)
    model = ScriptedModel([
        _submit(ABSTAIN),                               # would-be immediate UNANSWERABLE at iters≤2
        _submit("答えは42", confidence=0.8, evidence="file_grep hit"),
    ])
    res = investigate(model, "q", [], max_turns=6, research=True)
    assert res.stop_reason == "answered"
    assert res.answer.answer == "答えは42"
    delivered = [tr for turn in model.calls_seen if turn for tr in turn]
    assert any(tr.name == SUBMIT_ANSWER and isinstance(tr.response, dict)
               and tr.response.get("abstain_rejected") for tr in delivered)


def test_investigate_recal_off_accepts_early_unanswerable(monkeypatch):
    """Byte-identical OFF: with the recal off the same early abstain finalizes in one model turn."""
    monkeypatch.setattr(ob, "decompose", _no_obligations)
    model = ScriptedModel([_submit(ABSTAIN), _submit("答えは42")])
    res = investigate(model, "q", [], max_turns=6, research=True)   # RAG_EARLY_ABSTAIN_RECAL unset
    assert res.stop_reason == "answered"
    assert is_abstain(res.answer.answer)
    assert len(model.calls_seen) == 1                               # never asked to probe


# --------------------------------------------------------------------------- evidence packet directive
def test_packet_directive_byte_identical_when_off():
    packet = ep.build_packet("京橋のtrain.xlsxの平均年齢を教えて")
    assert ea.enabled() is False
    directive = ep.packet_directive(packet)
    assert ea.packet_relaxation_clause() not in directive


def test_packet_directive_appends_relaxation_when_on(monkeypatch):
    monkeypatch.setenv("RAG_EARLY_ABSTAIN_RECAL", "1")
    packet = ep.build_packet("京橋のtrain.xlsxの平均年齢を教えて")
    directive = ep.packet_directive(packet)
    assert ea.packet_relaxation_clause() in directive
