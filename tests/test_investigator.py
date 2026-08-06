"""SOT-2468 — offline tests for the Gemini function-calling investigation loop.

Everything here runs network-free: the loop is driven by a *scripted* fake model, and the tool layer
uses the real deterministic tools (a real ``find_files`` grep over the corpus, plus a temp-CSV
``compute``) so tool wiring is exercised end-to-end without touching Vertex.
"""
from __future__ import annotations

from pathlib import Path

import pytest

from src.rag.agent import investigator as inv
from src.rag.agent.investigator import (
    ABSTAIN,
    SUBMIT_ANSWER,
    AgentTool,
    Answer,
    Call,
    Investigation,
    Step,
    Usage,
    build_tools,
    dispatch,
    investigate,
    investigate_batch,
    is_abstain,
)
from src.rag.tools.profile import CorpusProfile


# --------------------------------------------------------------------------- scripted fake model
class ScriptedModel:
    """A fake :class:`~src.rag.agent.investigator.Model`: replays a fixed list of ``Step`` s."""

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


def _submit(answer, *, confidence=0.9, evidence="e", method="m", usage=Usage(50, 10)) -> Step:
    return Step(function_calls=(Call(SUBMIT_ANSWER, {
        "answer": answer, "confidence": confidence, "evidence": evidence, "method": method}),),
        usage=usage)


# --------------------------------------------------------------------------- answer schema
def test_answer_schema_shape():
    a = Answer(answer="1526", confidence=0.8, evidence="loan_amnt mean", method="compute")
    assert a.to_dict() == {"answer": "1526", "confidence": 0.8,
                           "evidence": "loan_amnt mean", "method": "compute"}


@pytest.mark.parametrize("raw,expected", [
    (0.5, 0.5), ("0.7", 0.7), (1, 1.0), (5, 1.0), (-2, 0.0),
    (None, 0.0), ("nope", 0.0), (float("nan"), 0.0),
])
def test_confidence_is_coerced_and_clamped(raw, expected):
    assert inv._coerce_confidence(raw) == expected


def test_abstain_answer_forces_zero_confidence():
    a = inv._answer_from_args({"answer": ABSTAIN, "confidence": 0.9})
    assert a.answer == ABSTAIN and a.confidence == 0.0


def test_is_abstain():
    assert is_abstain("わかりません")
    assert is_abstain("")
    assert is_abstain("不明です")
    assert not is_abstain("1526")


# --------------------------------------------------------------------------- tool layer
def test_build_tools_exposes_generic_tools_plus_submit_answer():
    names = {t.name for t in build_tools(CorpusProfile())}
    assert names == {
        "find_files", "file_grep", "read_office", "decrypt", "compute", "canonical_route",
        "read_chart_values", "caption_image", "pdf_emphasis", "pptx_pivot",
        "highlight_extract", "version_diff", "seating_lookup", "corpus_aggregate",
        SUBMIT_ANSWER,
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


def test_version_diff_tool_abstains_on_non_diff_question():
    # a plain retrieval question is NOT a diff question → contract value None, applicable False,
    # and the differ is never invoked (no guess).
    tools = {t.name: t for t in build_tools(CorpusProfile())}
    out = dispatch(tools, "version_diff", {"question": "契約金額はいくらですか。"})
    assert out["value"] is None
    assert out["method"]["engine"] == "diffpair"
    assert out["evidence"]["applicable"] is False


def test_version_diff_tool_wraps_solver_answer_in_contract(monkeypatch):
    # a resolved diff question: the deterministic solver's rendered answer is surfaced verbatim as the
    # contract value (the agent path must reproduce diffpair.answer_question, precision unchanged).
    from src.rag import diffpair

    monkeypatch.setattr(diffpair, "is_diff_question", lambda q: True)
    monkeypatch.setattr(diffpair, "answer_question_agent", lambda q: "QAレビューア：池田 直哉 → 小林 直樹")
    tools = {t.name: t for t in build_tools(CorpusProfile())}
    out = dispatch(tools, "version_diff", {"question": "旧版と最新版を比較して変更点を教えて。"})
    assert out["value"] == "QAレビューア：池田 直哉 → 小林 直樹"
    assert out["evidence"] == {"applicable": True, "resolved": True}
    assert out["method"]["engine"] == "diffpair"


def test_version_diff_tool_returns_none_value_when_solver_abstains(monkeypatch):
    # diff question but the solver can't resolve a unique pair → value None, resolved False (棄権のまま).
    from src.rag import diffpair

    monkeypatch.setattr(diffpair, "is_diff_question", lambda q: True)
    monkeypatch.setattr(diffpair, "answer_question_agent", lambda q: None)
    tools = {t.name: t for t in build_tools(CorpusProfile())}
    out = dispatch(tools, "version_diff", {"question": "幻の資料の旧版と最新版を比較して変更点を。"})
    assert out["value"] is None
    assert out["evidence"] == {"applicable": True, "resolved": False}


def test_version_diff_tool_reproduces_idx9_through_agent_path():
    # end-to-end over the real corpus: the deterministic idx9 answer (青嶺 QAレビューア change) must be
    # reachable through the agent tool-dispatch path, not only via diffpair directly.
    from src.rag import corpus

    if not corpus.walk():
        pytest.skip("corpus not present")
    tools = {t.name: t for t in build_tools(CorpusProfile())}
    q = ("青嶺不動産アセットマネジメントの提案書について、oldフォルダ内の旧版と提案フォルダ直下の"
         "最新版を比較し、変更された箇所を変更前と変更後で答えてください。")
    out = dispatch(tools, "version_diff", {"question": q})
    assert out["value"] and "池田 直哉" in out["value"] and "小林 直樹" in out["value"]
    assert out["method"]["engine"] == "diffpair"


def test_version_diff_tool_answers_adjacent_pair_but_abstains_on_gapped_pair():
    # Precision-first: an adjacent step (提案書 v1→v2, clean personnel change) resolves; a non-adjacent
    # explicit pair (v1→v3, skips v2) abstains rather than surface an unreliable single diff row (−1).
    from src.rag import corpus

    if not corpus.walk():
        pytest.skip("corpus not present")
    tools = {t.name: t for t in build_tools(CorpusProfile())}
    q_v2 = ("青葉与信マネジメントの提案書_v1.pptxから提案書_v2.pptxに修正されたもののうち、"
            "案件遂行に関連する変更を挙げてください。")
    q_v3 = ("青葉与信マネジメントの提案書_v1.pptxから提案書_v3.pptxに修正されたもののうち、"
            "案件遂行に関連する変更を挙げてください。")
    out_v2 = dispatch(tools, "version_diff", {"question": q_v2})
    out_v3 = dispatch(tools, "version_diff", {"question": q_v3})
    assert out_v2["value"] and "藤田 彩" in out_v2["value"] and "井上 里奈" in out_v2["value"]
    assert out_v3["value"] is None  # non-adjacent v1→v3 → abstain (precision 1.0 維持)
    assert out_v3["evidence"] == {"applicable": True, "resolved": False}


def test_dispatch_truncates_long_strings():
    tool = AgentTool("big", "d", {"type": "object", "properties": {}}, lambda: "x" * 10000)
    out = dispatch({"big": tool}, "big", {})
    assert len(out) < 10000 and out.startswith("x")


# --------------------------------------------------------------------------- agent loop
def _tool_that_records(sink):
    return AgentTool("compute", "d", {"type": "object", "properties": {}},
                     lambda **kw: sink.append(kw) or {"value": 20, "evidence": {}, "method": {}})


def test_investigate_dispatches_then_submits_structured_answer():
    sink = []
    tools = [_tool_that_records(sink), inv.SUBMIT_ANSWER_TOOL]
    model = ScriptedModel([
        Step(function_calls=(Call("compute", {"file": "f", "expr": "df['b'].mean()"}),), usage=Usage(100, 20)),
        _submit("平均は20です", confidence=0.9, evidence="df['b'].mean()", method="compute"),
    ])
    res = investigate(model, "…", tools, max_turns=5)
    assert isinstance(res, Investigation)
    assert res.stop_reason == "answered"
    assert res.answer.answer == "平均は20です"
    assert res.answer.confidence == 0.9
    assert res.answer.evidence == "df['b'].mean()" and res.answer.method == "compute"
    assert res.iterations == 1  # only the tool round counts
    assert res.tool_calls == ["compute", SUBMIT_ANSWER]
    assert res.usage.input_tokens == 150 and res.usage.output_tokens == 30
    assert sink == [{"file": "f", "expr": "df['b'].mean()"}]
    assert res.model == "fake-model"
    d = res.to_dict()
    assert set(("answer", "confidence", "evidence", "method")) <= set(d)
    assert d["cost_usd"] >= 0.0


def test_investigate_accepts_plain_final_text_with_zero_confidence():
    model = ScriptedModel([Step(function_calls=(), final_text="20日", usage=Usage(10, 5))])
    res = investigate(model, "…", build_tools(CorpusProfile()), max_turns=3)
    assert res.stop_reason == "answered"
    assert res.answer.answer == "20日"
    assert res.answer.confidence == 0.0
    assert "submit_answer" in res.answer.method


def test_investigate_abstains_on_max_turns():
    tools = [_tool_that_records([])]
    steps = [Step(function_calls=(Call("compute", {}),), usage=Usage(1, 1)) for _ in range(10)]
    model = ScriptedModel(steps)
    res = investigate(model, "…", tools, max_turns=3)
    assert res.stop_reason == "max_turns"
    assert res.answer.answer == ABSTAIN and res.answer.confidence == 0.0
    assert res.error and "max_turns" in res.error
    assert res.iterations == 3


def test_investigate_times_out_between_turns():
    # A fake clock that advances 10s per read; timeout_s=5 trips before the first model turn's work.
    ticks = iter([0.0, 0.0, 100.0, 200.0, 300.0])
    tools = [_tool_that_records([])]
    steps = [Step(function_calls=(Call("compute", {}),), usage=Usage(1, 1)) for _ in range(5)]
    model = ScriptedModel(steps)
    res = investigate(model, "…", tools, max_turns=5, timeout_s=5.0,
                      clock=lambda: next(ticks))
    assert res.stop_reason == "timeout"
    assert res.error and "timeout" in res.error
    assert res.answer.answer == ABSTAIN


def test_investigate_handles_model_error():
    class Boom:
        model_name = "boom"

        def next(self, _):
            raise RuntimeError("vertex down")

    res = investigate(Boom(), "…", [], max_turns=3)
    assert res.stop_reason == "model_error"
    assert res.error and "model error" in res.error
    assert res.answer.answer == ABSTAIN


def test_submit_answer_terminates_before_other_calls_in_same_step():
    sink = []
    tools = [_tool_that_records(sink), inv.SUBMIT_ANSWER_TOOL]
    # submit_answer first in the tuple → loop finalizes and never dispatches the trailing compute
    model = ScriptedModel([Step(function_calls=(
        Call(SUBMIT_ANSWER, {"answer": "42", "confidence": 0.5}),
        Call("compute", {"file": "f", "expr": "df.sum()"}),
    ), usage=Usage(1, 1))])
    res = investigate(model, "…", tools, max_turns=3)
    assert res.answer.answer == "42" and res.answer.confidence == 0.5
    assert res.tool_calls == [SUBMIT_ANSWER]
    assert sink == []  # trailing compute never ran


# --------------------------------------------------------------------------- batch loop (acceptance)
def _factory_for(answers: dict[str, str]):
    """A model factory that answers each question in one turn via submit_answer."""
    def factory(question: str, tools):
        return ScriptedModel([_submit(answers[question], confidence=0.8)])
    return factory


def test_investigate_batch_runs_five_questions_and_returns_schema():
    """受け入れ条件: 5問でループが回り回答スキーマ {answer, confidence, evidence, method} を返す。"""
    questions = [f"質問{i}" for i in range(5)]
    answers = {q: f"回答{i}" for i, q in enumerate(questions)}
    results = investigate_batch(_factory_for(answers), questions, max_turns=3)
    assert len(results) == 5
    for q, res in zip(questions, results):
        assert isinstance(res, Investigation)
        assert res.stop_reason == "answered"
        assert res.answer.answer == answers[q]
        # every result carries the full structured answer schema
        assert set(res.answer.to_dict()) == {"answer", "confidence", "evidence", "method"}
        assert 0.0 <= res.answer.confidence <= 1.0


def test_investigate_batch_uses_isolated_profile_per_question():
    seen = []

    def profile_factory():
        p = CorpusProfile()
        seen.append(p)
        return p

    questions = [f"q{i}" for i in range(5)]
    investigate_batch(_factory_for({q: "x" for q in questions}), questions,
                      profile_factory=profile_factory, max_turns=2)
    assert len(seen) == 5
    assert len({id(p) for p in seen}) == 5


# --------------------------------------------------------------------------- cost / live wiring
def test_usage_cost_accounting():
    u = Usage(1_000_000, 1_000_000)
    # gemini-2.5-pro list price: (1.25 in, 10.0 out) per 1M tokens
    assert round(u.cost_usd("gemini-2.5-pro"), 6) == round(1.25 + 10.0, 6)
    # unknown model falls back to the pro price
    assert u.cost_usd("unknown") == u.cost_usd("gemini-2.5-pro")


def test_to_genai_tools_builds_function_declarations_including_submit_answer():
    tools = build_tools(CorpusProfile())
    genai_tools = inv.to_genai_tools(tools)
    assert len(genai_tools) == 1
    decls = genai_tools[0].function_declarations
    assert {d.name for d in decls} == {t.name for t in tools}
    assert SUBMIT_ANSWER in {d.name for d in decls}
