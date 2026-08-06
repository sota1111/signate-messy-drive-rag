"""SOT-2467 — offline tests for the Gemini early validation gate.

Everything here runs network-free: the agent loop is driven by a *scripted* fake model, and the tool
layer uses the real deterministic tools (a real ``find_files`` grep over the corpus, plus a temp-CSV
``compute``) so tool wiring is exercised end-to-end without touching Vertex.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from scoring import early_gate as eg
from scoring.early_gate import (
    GATE_THRESHOLD,
    GOLD_QUESTIONS,
    AgentTool,
    Call,
    GoldQuestion,
    Step,
    Usage,
    answers_match,
    build_tools,
    dispatch,
    is_abstain,
    run_gate,
    run_question,
)
from src.rag.tools.profile import CorpusProfile


# --------------------------------------------------------------------------- scripted fake model
class ScriptedModel:
    """A fake :class:`~scoring.early_gate.Model`: replays a fixed list of ``Step`` s, ignoring input."""

    def __init__(self, steps, *, model_name="fake-model"):
        self._steps = list(steps)
        self._i = 0
        self.model_name = model_name
        self.calls_seen = []

    def next(self, tool_responses):
        self.calls_seen.append(tool_responses)
        if self._i >= len(self._steps):
            return Step(final_text=eg.ABSTAIN, usage=Usage(1, 1))
        step = self._steps[self._i]
        self._i += 1
        return step


# --------------------------------------------------------------------------- gold set integrity
def test_gold_set_has_five_distinct_typed_questions():
    assert len(GOLD_QUESTIONS) == 5
    types = {q.type for q in GOLD_QUESTIONS}
    assert types == {"decrypt", "format", "compute", "enumerate", "chart"}
    for q in GOLD_QUESTIONS:
        assert q.id and q.question.strip() and q.gold.strip()


def test_gold_carries_no_password_or_alias_literal():
    """移植性: the gold set must not embed a raw corpus secret (e.g. the かえで password)."""
    blob = "\n".join(q.question + q.gold for q in GOLD_QUESTIONS)
    assert "pw-kaede" not in blob
    assert "20250902" not in blob


# --------------------------------------------------------------------------- matching
@pytest.mark.parametrize("pred,gold,qtype,expected", [
    ("1526", "1526", "compute", True),
    ("平均は1,526です", "1526", "compute", True),
    ("3850000円", "3,850,000円", "decrypt", True),
    ("20", "20日", "chart", True),
    ("21日", "20日", "chart", False),
    ("T12、T09、T11、T10", "T09、T10、T11、T12", "enumerate", True),
    ("T09、T10、T11", "T09、T10、T11、T12", "enumerate", False),
    ("temp、weekday、hr、weathersit", "hr、weekday、weathersit、temp", "format", True),
    ("わかりません", "1526", "compute", False),
])
def test_answers_match(pred, gold, qtype, expected):
    assert answers_match(pred, gold, qtype) is expected


def test_is_abstain():
    assert is_abstain("わかりません")
    assert is_abstain("")
    assert not is_abstain("1526")


# --------------------------------------------------------------------------- tool layer
def test_build_tools_exposes_expected_generic_tools():
    names = {t.name for t in build_tools(CorpusProfile())}
    assert names == {
        "find_files", "file_grep", "read_office", "decrypt", "compute",
        "read_chart_values", "caption_image", "pdf_emphasis", "pptx_pivot",
        "highlight_extract",
    }


def test_dispatch_unknown_tool_returns_error():
    out = dispatch({}, "nope", {})
    assert "error" in out and "unknown tool" in out["error"]


def test_dispatch_bad_arguments_returns_error():
    tools = {t.name: t for t in build_tools(CorpusProfile())}
    out = dispatch(tools, "compute", {"file": "x"})  # missing required 'expr'
    assert "error" in out


def test_dispatch_runs_real_find_files_offline():
    tools = {t.name: t for t in build_tools(CorpusProfile())}
    out = dispatch(tools, "find_files", {"ext": "csv"})
    assert out["method"]["engine"] == "corpus"
    assert isinstance(out["value"], list) and out["value"]


def test_dispatch_runs_real_compute_on_temp_csv(tmp_path: Path):
    csv = tmp_path / "t.csv"
    csv.write_text("a,b\n1,10\n3,30\n", encoding="utf-8")
    tools = {t.name: t for t in build_tools(CorpusProfile())}
    out = dispatch(tools, "compute", {"file": str(csv), "expr": "df['b'].mean()"})
    assert out["value"] == 20
    assert out["method"]["engine"] == "pandas"


def test_dispatch_truncates_long_strings():
    tool = AgentTool("big", "d", {"type": "object", "properties": {}}, lambda: "x" * 10000)
    out = dispatch({"big": tool}, "big", {})
    assert len(out) < 10000 and out.startswith("x")


# --------------------------------------------------------------------------- agent loop
def _tool_that_records(sink):
    return AgentTool("compute", "d", {"type": "object", "properties": {}},
                     lambda **kw: sink.append(kw) or {"value": 20, "evidence": {}, "method": {}})


def test_run_question_dispatches_then_finalizes():
    sink = []
    tools = [_tool_that_records(sink)]
    model = ScriptedModel([
        Step(function_calls=(Call("compute", {"file": "f", "expr": "df['b'].mean()"}),), usage=Usage(100, 20)),
        Step(final_text="平均は20です", usage=Usage(50, 10)),
    ])
    q = GoldQuestion("t", "compute", "…", "20")
    res = run_question(model, q, tools, max_turns=5)
    assert res.correct is True
    assert res.iterations == 1
    assert res.tool_calls == ["compute"]
    assert res.usage.input_tokens == 150 and res.usage.output_tokens == 30
    assert sink == [{"file": "f", "expr": "df['b'].mean()"}]
    assert res.model == "fake-model"


def test_run_question_abstains_on_max_turns():
    tools = [_tool_that_records([])]
    # a model that only ever calls tools, never finalizes
    steps = [Step(function_calls=(Call("compute", {}),), usage=Usage(1, 1)) for _ in range(10)]
    model = ScriptedModel(steps)
    q = GoldQuestion("t", "compute", "…", "20")
    res = run_question(model, q, tools, max_turns=3)
    assert res.correct is False
    assert res.error and "max_turns" in res.error
    assert res.iterations == 3


def test_run_question_handles_model_error():
    class Boom:
        model_name = "boom"

        def next(self, _):
            raise RuntimeError("vertex down")

    q = GoldQuestion("t", "compute", "…", "20")
    res = run_question(Boom(), q, [], max_turns=3)
    assert res.correct is False and "model error" in res.error


# --------------------------------------------------------------------------- gate driver
def _factory_for(answers: dict[str, str]):
    """Build a model factory that answers each question in one turn with a scripted string."""
    def factory(q: GoldQuestion, tools):
        return ScriptedModel([Step(final_text=answers[q.id], usage=Usage(10, 5))])
    return factory


def test_run_gate_go_when_four_of_five_match():
    answers = {q.id: q.gold for q in GOLD_QUESTIONS}
    answers["chart"] = "99日"  # break exactly one → 4/5
    summary = run_gate(_factory_for(answers), max_turns=3)
    assert summary["n_correct"] == 4
    assert summary["threshold"] == GATE_THRESHOLD
    assert summary["go"] is True
    assert summary["total_input_tokens"] == 50 and summary["total_output_tokens"] == 25


def test_run_gate_no_go_when_three_match():
    answers = {q.id: q.gold for q in GOLD_QUESTIONS}
    answers["chart"] = "99日"
    answers["format"] = "違う"
    summary = run_gate(_factory_for(answers), max_turns=3)
    assert summary["n_correct"] == 3
    assert summary["go"] is False


def test_run_gate_uses_isolated_profile_per_question():
    seen = []

    def profile_factory():
        p = CorpusProfile()
        seen.append(p)
        return p

    run_gate(_factory_for({q.id: q.gold for q in GOLD_QUESTIONS}),
             profile_factory=profile_factory, max_turns=2)
    assert len(seen) == len(GOLD_QUESTIONS)
    assert len({id(p) for p in seen}) == len(GOLD_QUESTIONS)


# --------------------------------------------------------------------------- artifact / ledger
def test_write_artifact_and_append_ledger(tmp_path: Path):
    summary = run_gate(_factory_for({q.id: q.gold for q in GOLD_QUESTIONS}), max_turns=2)
    art = eg.write_artifact(summary, tmp_path / "early_gate.json")
    loaded = json.loads(art.read_text(encoding="utf-8"))
    assert loaded["go"] is True and len(loaded["results"]) == 5

    ledger = tmp_path / "experiment_ledger.jsonl"
    eg.append_ledger(summary, ledger, recorded_at="2026-08-05T00:00:00Z")
    entry = json.loads(ledger.read_text(encoding="utf-8").strip())
    assert entry["result"] == "promoted"
    assert entry["axis"].startswith("gemini-only early validation gate")
    assert "n_correct=5/5" in entry["evidence"]


def test_to_genai_tools_builds_function_declarations():
    tools = build_tools(CorpusProfile())
    genai_tools = eg.to_genai_tools(tools)
    assert len(genai_tools) == 1
    decls = genai_tools[0].function_declarations
    assert {d.name for d in decls} == {t.name for t in tools}
