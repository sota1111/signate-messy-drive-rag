"""SOT-2492 — offline tests for the abstain ledger.

Two layers are exercised network-free:

* the pure classifier / schema / writer of :mod:`src.rag.agent.abstain_ledger`, and
* its wiring into the investigator loop (:func:`src.rag.agent.investigator.investigate`) driven by a
  scripted fake model over controlled tools — proving every abstain path writes a coded record while
  the commit-vs-abstain decision stays byte-identical to a run with the ledger off.
"""
from __future__ import annotations

import json

import pytest

from src.rag.agent import abstain_ledger as al
from src.rag.agent.abstain_ledger import (
    BUDGET_EXHAUSTED,
    EVIDENCE_CONFLICT,
    EVIDENCE_INCOMPLETE,
    NOT_RETRIEVED,
    PARSED_AMBIGUOUS,
    RETRIEVED_NOT_PARSED,
    SPIN_CUTOFF,
    STATE_CODES,
    UNANSWERABLE,
    AbstainSignals,
    LedgerError,
    classify,
    record_from_investigation,
    validate,
    write_record,
)
from src.rag.agent.investigator import (
    ABSTAIN,
    SUBMIT_ANSWER,
    AgentTool,
    Call,
    Step,
    Usage,
    investigate,
)


# --------------------------------------------------------------------------- helpers
class ScriptedModel:
    """Replays a fixed list of :class:`Step` s; abstains once exhausted."""

    def __init__(self, steps, *, model_name="fake-model"):
        self._steps = list(steps)
        self._i = 0
        self.model_name = model_name

    def next(self, tool_responses):
        if self._i >= len(self._steps):
            return Step(function_calls=(), final_text=ABSTAIN, usage=Usage(1, 1))
        step = self._steps[self._i]
        self._i += 1
        return step


def _call(name, **args):
    return Step(function_calls=(Call(name, args),), usage=Usage(10, 5))


def _submit(answer, *, confidence=0.9, evidence="e", method="m"):
    return Step(function_calls=(Call(SUBMIT_ANSWER, {
        "answer": answer, "confidence": confidence, "evidence": evidence, "method": method}),),
        usage=Usage(20, 8))


def _tool(name, ret):
    """A controlled AgentTool that ignores its args and returns ``ret``."""
    return AgentTool(name, name, {"type": "object", "properties": {}, "required": []},
                     lambda **_k: ret)


def _sig(**kw):
    s = AbstainSignals()
    for k, v in kw.items():
        setattr(s, k, v)
    return s


# --------------------------------------------------------------------------- state-code vocabulary
def test_distinct_state_codes():
    # SOT-2522 added SPIN_CUTOFF as the eighth distinct code.
    assert len(STATE_CODES) == 8
    assert len(set(STATE_CODES)) == 8
    assert SPIN_CUTOFF in STATE_CODES


def test_classify_spin_cutoff_precedes_budget_and_research_terminal():
    # A fired spin detector attributes the abstain to SPIN_CUTOFF even though the stop_reason / research
    # terminal would otherwise read as a plain BUDGET cutoff (spin is the actionable root cause).
    assert classify(_sig(spin_cutoff=True)) == SPIN_CUTOFF
    assert classify(_sig(spin_cutoff=True, stop_reason="max_turns")) == SPIN_CUTOFF
    assert classify(_sig(spin_cutoff=True, stop_reason="spin_cutoff")) == SPIN_CUTOFF
    assert classify(_sig(spin_cutoff=True, research_terminal="budget")) == SPIN_CUTOFF
    # Without the spin flag, an ordinary budget cutoff still classifies as BUDGET_EXHAUSTED.
    assert classify(_sig(stop_reason="max_turns")) == BUDGET_EXHAUSTED


# --------------------------------------------------------------------------- classifier (one per code)
def test_classify_budget_exhausted_on_max_turns_and_timeout():
    assert classify(_sig(stop_reason="max_turns")) == BUDGET_EXHAUSTED
    assert classify(_sig(stop_reason="timeout")) == BUDGET_EXHAUSTED


def test_classify_model_error_is_unanswerable():
    assert classify(_sig(stop_reason="model_error")) == UNANSWERABLE


def test_classify_parsed_ambiguous_when_a_tool_reported_ambiguity():
    assert classify(_sig(retrieval_attempts=1, retrieval_ok=1, ambiguous=1)) == PARSED_AMBIGUOUS


def test_classify_evidence_conflict_from_keyword():
    assert classify(_sig(retrieval_ok=1, evidence_text="2つの値が矛盾している")) == EVIDENCE_CONFLICT


def test_classify_not_retrieved_when_nothing_succeeded():
    assert classify(_sig(retrieval_attempts=2, retrieval_ok=0)) == NOT_RETRIEVED
    assert classify(AbstainSignals()) == NOT_RETRIEVED  # total function: default → NOT_RETRIEVED


def test_classify_retrieved_not_parsed():
    assert classify(_sig(retrieval_ok=1, extraction_attempts=2, extraction_ok=0)) == RETRIEVED_NOT_PARSED


def test_classify_evidence_incomplete_from_keyword_and_from_failed_derivation():
    assert classify(_sig(retrieval_ok=1, extraction_ok=1,
                         evidence_text="根拠が不足している")) == EVIDENCE_INCOMPLETE
    assert classify(_sig(retrieval_ok=1, derivation_attempts=1,
                         derivation_ok=0)) == EVIDENCE_INCOMPLETE


def test_classify_unanswerable_default_when_something_succeeded_but_no_other_signal():
    assert classify(_sig(retrieval_ok=1, extraction_ok=1, derivation_ok=1)) == UNANSWERABLE


def test_classify_is_total_over_a_grid_of_signals():
    reasons = ["answered", "max_turns", "timeout", "model_error"]
    for r in reasons:
        for ra in (0, 1):
            for ea in (0, 1):
                for da in (0, 1):
                    for amb in (0, 1):
                        code = classify(_sig(stop_reason=r, retrieval_ok=ra, extraction_ok=ea,
                                             derivation_ok=da, ambiguous=amb))
                        assert code in STATE_CODES


# --------------------------------------------------------------------------- signal observation
def test_observe_buckets_tools_and_outcomes():
    s = AbstainSignals()
    s.observe("find_files", [])                        # retrieval, empty
    s.observe("file_grep", [{"file": "a"}])            # retrieval, ok
    s.observe("read_office", {"error": "boom"})        # extraction, hard error
    s.observe("compute", {"error": "曖昧なファイル参照 存在プロジェクト: A, B"})  # derivation, ambiguous
    s.observe("corpus_aggregate", {"value": None, "evidence": {}, "method": {}})   # derivation, empty
    assert (s.retrieval_attempts, s.retrieval_ok) == (2, 1)
    assert (s.extraction_attempts, s.extraction_ok) == (1, 0)
    assert (s.derivation_attempts, s.derivation_ok) == (2, 0)
    assert s.ambiguous == 1
    assert s.errors == 1


def test_observe_contract_value_present_is_ok():
    s = AbstainSignals()
    s.observe("compute", {"value": 42, "evidence": {"file": "t.csv"}, "method": {"engine": "pandas"}})
    assert (s.derivation_attempts, s.derivation_ok) == (1, 1)


# --------------------------------------------------------------------------- record + schema
class _FakeAnswer:
    def __init__(self, answer, confidence=0.0, evidence="", method=""):
        self.answer, self.confidence, self.evidence, self.method = answer, confidence, evidence, method


class _FakeInvestigation:
    def __init__(self, **kw):
        self.question = kw.get("question", "Q?")
        self.answer = kw.get("answer", _FakeAnswer(ABSTAIN))
        self.iterations = kw.get("iterations", 1)
        self.tool_calls = kw.get("tool_calls", ["find_files"])
        self.usage = kw.get("usage", Usage(10, 5))
        self.model = kw.get("model", "fake-model")
        self.elapsed_s = kw.get("elapsed_s", 1.234)
        self.stop_reason = kw.get("stop_reason", "answered")
        self.error = kw.get("error", None)


def test_record_from_investigation_is_valid_and_coded():
    inv = _FakeInvestigation(tool_calls=["find_files", "find_files"], stop_reason="answered")
    sig = _sig(stop_reason="answered", retrieval_attempts=2, retrieval_ok=0)
    rec = record_from_investigation(inv, sig, now=lambda: "2026-08-06T00:00:00+00:00")
    d = rec.to_dict()
    validate(d)                                   # does not raise
    assert d["state_code"] == NOT_RETRIEVED
    assert d["explored_paths"] == ["find_files", "find_files"]
    assert d["missing"] and d["missing"][0]["kind"] == "retrieval"
    assert d["missing"][0]["signals"]["retrieval_attempts"] == 2
    assert d["recorded_at"] == "2026-08-06T00:00:00+00:00"
    assert "retrieval miss" in d["evidence_obligation"]


def test_record_folds_model_note_into_obligation():
    inv = _FakeInvestigation(answer=_FakeAnswer(ABSTAIN, evidence="探索したが不明"))
    sig = _sig(stop_reason="answered", retrieval_ok=1, evidence_text="探索したが不明")
    rec = record_from_investigation(inv, sig)
    assert "調査メモ" in rec.evidence_obligation


def test_validate_rejects_bad_records():
    good = record_from_investigation(_FakeInvestigation(), _sig(retrieval_attempts=1)).to_dict()
    validate(good)
    bad_code = dict(good, state_code="NONSENSE")
    with pytest.raises(LedgerError):
        validate(bad_code)
    with pytest.raises(LedgerError):
        validate({k: v for k, v in good.items() if k != "state_code"})
    with pytest.raises(LedgerError):
        validate(dict(good, missing=[]))


# --------------------------------------------------------------------------- writer
def test_write_record_appends_jsonl(tmp_path):
    p = tmp_path / "abstain_ledger.jsonl"
    r1 = record_from_investigation(_FakeInvestigation(question="Q1"), _sig(retrieval_attempts=1))
    r2 = record_from_investigation(_FakeInvestigation(question="Q2"),
                                   _sig(stop_reason="timeout"))
    assert write_record(r1, p) == p
    write_record(r2, p)
    lines = p.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 2
    rows = [json.loads(x) for x in lines]
    assert rows[0]["question"] == "Q1" and rows[0]["state_code"] == NOT_RETRIEVED
    assert rows[1]["question"] == "Q2" and rows[1]["state_code"] == BUDGET_EXHAUSTED
    for row in rows:
        validate(row)


# --------------------------------------------------------------------------- investigator wiring
def _run_with_tools(steps, tools, ledger):
    model = ScriptedModel(steps)
    return investigate(model, "テスト質問", tools, max_turns=8, ledger=ledger)


def test_investigate_writes_coded_record_on_retrieval_miss(tmp_path):
    p = tmp_path / "ledger.jsonl"
    from src.rag.agent.investigator import SUBMIT_ANSWER_TOOL
    tools = [_tool("find_files", []), SUBMIT_ANSWER_TOOL]
    inv = _run_with_tools([_call("find_files", query="x"), _submit(ABSTAIN, confidence=0.0)], tools, p)
    assert inv.answer.answer == ABSTAIN
    rows = [json.loads(x) for x in p.read_text(encoding="utf-8").strip().splitlines()]
    assert len(rows) == 1
    assert rows[0]["state_code"] == NOT_RETRIEVED
    assert rows[0]["explored_paths"] == ["find_files", SUBMIT_ANSWER]
    validate(rows[0])


def test_investigate_does_not_write_when_answer_is_committed(tmp_path):
    p = tmp_path / "ledger.jsonl"
    from src.rag.agent.investigator import SUBMIT_ANSWER_TOOL
    tools = [_tool("find_files", [{"file": "hit.xlsx"}]), SUBMIT_ANSWER_TOOL]
    inv = _run_with_tools([_call("find_files", query="x"), _submit("42", confidence=0.9)], tools, p)
    assert inv.answer.answer == "42"
    assert not p.exists()


def test_investigate_records_budget_exhausted_on_max_turns(tmp_path):
    p = tmp_path / "ledger.jsonl"
    from src.rag.agent.investigator import SUBMIT_ANSWER_TOOL
    tools = [_tool("find_files", [{"file": "hit"}]), SUBMIT_ANSWER_TOOL]
    # never submits → loop exhausts max_turns and abstains
    model = ScriptedModel([_call("find_files") for _ in range(10)])
    inv = investigate(model, "q", tools, max_turns=3, ledger=p)
    assert inv.stop_reason == "max_turns"
    rows = [json.loads(x) for x in p.read_text(encoding="utf-8").strip().splitlines()]
    assert rows[0]["state_code"] == BUDGET_EXHAUSTED


def test_ledger_off_writes_nothing_and_default_is_off(tmp_path, monkeypatch):
    # default path must not be touched when ledger is disabled
    monkeypatch.setattr(al, "default_path", lambda: tmp_path / "should_not_exist.jsonl")
    from src.rag.agent.investigator import SUBMIT_ANSWER_TOOL
    tools = [_tool("find_files", []), SUBMIT_ANSWER_TOOL]
    inv = _run_with_tools([_call("find_files"), _submit(ABSTAIN, confidence=0.0)], tools, None)
    assert inv.answer.answer == ABSTAIN
    assert not (tmp_path / "should_not_exist.jsonl").exists()


def test_ledger_does_not_change_the_decision(tmp_path):
    """受け入れ条件②: commit/abstain の判断自体は台帳のON/OFFで不変。"""
    from src.rag.agent.investigator import SUBMIT_ANSWER_TOOL
    steps = [_call("find_files", query="x"), _submit(ABSTAIN, confidence=0.0, evidence="矛盾する2値")]

    tools_off = [_tool("find_files", [{"file": "a"}]), SUBMIT_ANSWER_TOOL]
    off = investigate(ScriptedModel(steps), "q", tools_off, max_turns=8, ledger=None)

    tools_on = [_tool("find_files", [{"file": "a"}]), SUBMIT_ANSWER_TOOL]
    on = investigate(ScriptedModel(steps), "q", tools_on, max_turns=8, ledger=tmp_path / "l.jsonl")

    assert off.answer == on.answer
    assert off.stop_reason == on.stop_reason
    assert off.iterations == on.iterations
    assert off.tool_calls == on.tool_calls
    # and the recorded code reflects the model's conflict note
    rows = [json.loads(x) for x in (tmp_path / "l.jsonl").read_text(encoding="utf-8").strip().splitlines()]
    assert rows[0]["state_code"] == EVIDENCE_CONFLICT
