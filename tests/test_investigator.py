"""SOT-2468 — offline tests for the Gemini function-calling investigation loop.

Everything here runs network-free: the loop is driven by a *scripted* fake model, and the tool layer
uses the real deterministic tools (a real ``find_files`` grep over the corpus, plus a temp-CSV
``compute``) so tool wiring is exercised end-to-end without touching Vertex.
"""
from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

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


def test_chart_contract_rejects_numeric_submit_without_strict_chart_evidence():
    model = ScriptedModel([
        _submit("1473", evidence="caption_image", method="vision"),
        _submit(ABSTAIN, confidence=0.0),
    ])
    res = investigate(
        model, "AG_ratioのヒストグラムで最も多いカウントは。",
        [inv.SUBMIT_ANSWER_TOOL], max_turns=3, contract="chart_read")
    assert res.answer.answer == ABSTAIN
    assert model.calls_seen[1][0].response["answer_rejected"] is True
    assert "厳密証拠" in model.calls_seen[1][0].response["reason"]


def test_chart_contract_accepts_numcache_or_source_compute_evidence():
    strict = AgentTool(
        "read_chart_values", "strict", {"type": "object", "properties": {}},
        lambda **_kw: {
            "value": {"result": 958}, "evidence": {"source_range": "train!K2:K3501"},
            "method": {"engine": "chart_source_compute", "numeric_authority": True,
                       "vision_used": False},
        })
    model = ScriptedModel([
        Step(function_calls=(Call("read_chart_values", {}),)),
        _submit("958", evidence="train!K2:K3501", method="chart_source_compute"),
    ])
    res = investigate(
        model, "AG_ratioのヒストグラムで最も多いカウントは。",
        [strict, inv.SUBMIT_ANSWER_TOOL], max_turns=3, contract="chart_read")
    assert res.answer.answer == "958"


def test_gantt_contract_accepts_native_week_grid_evidence_without_numcache():
    strict = AgentTool(
        "read_office", "gantt", {"type": "object", "properties": {}},
        lambda **_kw: {
            "value": "【ガント週グリッド:決定論】\n[スライド5] モデル改善: 第6週目から第8週目\n【/ガント週グリッド】",
            "evidence": {"file": "proposal.pptx"},
            "method": {"engine": "pptx"},
        })
    model = ScriptedModel([
        Step(function_calls=(Call("read_office", {}),)),
        _submit("第6週目から第8週目", evidence="proposal.pptx", method="native Gantt geometry"),
    ])
    res = investigate(
        model, "提案書のモデル改善の実行予定スケジュールは案件開始から第何週目ですか。",
        [strict, inv.SUBMIT_ANSWER_TOOL], max_turns=3, contract="chart_read")
    assert res.answer.answer == "第6週目から第8週目"


def test_chart_contract_free_text_without_strict_evidence_abstains():
    model = ScriptedModel([Step(function_calls=(), final_text="958")])
    res = investigate(model, "グラフの値は。", [inv.SUBMIT_ANSWER_TOOL],
                      max_turns=1, contract="chart_read")
    assert res.answer.answer == ABSTAIN


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


def test_numeric_contract_rejects_wrong_denominator_before_commit(tmp_path: Path):
    q = ("標準化されたloan_amntが0未満の行のうち、purpose=credit_cardでloan_amntが平均を"
         "上回る行の割合は何%ですか。小数第2位まで答えてください。")
    outputs = {
        "len(df[df['purpose'] == 'credit_card'])": 3053,
        "round(129 / 3053 * 100, 2)": 4.23,
        "len(df[df['loan_amnt'] < 1582.99])": 10938,
        "round(129 / 10938 * 100, 2)": 1.18,
    }

    def compute(expr):
        return {
            "value": outputs[expr],
            "evidence": {"file": "train.csv", "columns_used": ["loan_amnt", "purpose"],
                         "rows": 17500},
            "method": {"code": expr, "trace": {"input_rows": 17500}},
        }

    tool = AgentTool(
        "compute", "d",
        {"type": "object", "properties": {"expr": {"type": "string"}}, "required": ["expr"]},
        compute,
    )
    model = ScriptedModel([
        Step(function_calls=(Call("compute", {"expr": "len(df[df['purpose'] == 'credit_card'])"}),)),
        Step(function_calls=(Call("compute", {"expr": "round(129 / 3053 * 100, 2)"}),)),
        _submit("4.23%"),  # internally consistent arithmetic, wrong denominator population
        Step(function_calls=(Call("compute", {"expr": "len(df[df['loan_amnt'] < 1582.99])"}),)),
        Step(function_calls=(Call("compute", {"expr": "round(129 / 10938 * 100, 2)"}),)),
        _submit("1.18%"),
    ])
    result = investigate(
        model, q, [tool, inv.SUBMIT_ANSWER_TOOL], max_turns=8,
        calc_ledger=tmp_path / "calc.jsonl", contract="numeric")
    assert result.answer.answer == "1.18%"
    delivered = [response for turn in model.calls_seen if turn for response in turn]
    assert any(response.name == SUBMIT_ANSWER
               and response.response.get("answer_rejected") is True for response in delivered)


def test_regulation_content_rejects_no_rule_only_then_accepts_complete_fallback():
    q = "契約条件で稼働上限を超えた場合の精算方法に関する規定内容を答えてください。"
    model = ScriptedModel([
        _submit("超過時の特別な精算規定は存在しません。"),
        _submit(
            "特別な精算規定は存在しません。一般規定により時間単価30,000円に消費税を加算し、15分単位で切上げ、"
            "月次精算し、上限はありません。"
        ),
    ])
    result = investigate(
        model, q, [inv.SUBMIT_ANSWER_TOOL], max_turns=3,
        ledger=False, calc_ledger=False, research=False, enumeration=False,
        contract="simple_lookup",
    )
    assert "15分単位" in result.answer.answer and result.stop_reason == "answered"
    delivered = [response for turn in model.calls_seen if turn for response in turn]
    rejection = next(response.response for response in delivered
                     if response.name == SUBMIT_ANSWER and response.response.get("answer_rejected"))
    assert set(rejection["missing"]) == {"単価", "税処理", "課金単位", "丸め", "精算周期", "上限"}


def test_regulation_content_free_text_cannot_bypass_completion_guard():
    q = "契約条件で稼働上限を超えた場合の精算方法に関する規定内容を答えてください。"
    complete = (
        "特別な精算規定は存在しません。一般規定として時間単価30,000円に消費税を加算し、"
        "15分単位で切上げ、月次精算し、上限はありません。")
    model = ScriptedModel([
        Step(function_calls=(), final_text="特別規定はありません。"),
        _submit(complete),
    ])
    result = investigate(
        model, q, [inv.SUBMIT_ANSWER_TOOL], max_turns=3,
        ledger=False, calc_ledger=False, research=False, enumeration=False,
        contract="simple_lookup",
    )
    assert result.answer.answer == complete
    assert model.calls_seen[1][0].name == inv.DIRECTIVE_MESSAGE


def test_real_focused_questions_resolve_deterministically_without_hardcoded_values():
    from src.rag import corpus

    if not corpus.walk():
        pytest.skip("corpus not present")
    profile = CorpusProfile()
    gantt = inv._deterministic_gantt_answer(
        "白峰信用リスク評価の提案書.pptxにおいて、モデルの高度化（説明性・セグメント分析）の"
        "実行予定スケジュールは案件開始から第何週目に実施予定でしょうか。", profile)
    regulation = inv._deterministic_regulation_answer(
        "ひがし丘の契約条件において、ACTHが200時間を超えた場合の精算方法に関する規定内容を答えてください。",
        profile)
    assert gantt is not None and gantt.answer == "第6週目から第8週目"
    assert regulation is not None
    assert "25,000円" in regulation.answer and "消費税を加算" in regulation.answer
    assert "30分単位" in regulation.answer and "月次" in regulation.answer and "上限なし" in regulation.answer


def test_idx44_and_idx86_use_closed_deterministic_paths_without_gemini(monkeypatch):
    from src.rag import corpus

    if not corpus.walk():
        pytest.skip("corpus not present")
    monkeypatch.setattr(inv, "gemini_model_factory", lambda *_a, **_k: pytest.fail("Gemini not needed"))
    seating = inv.answer_question(
        "IMにあるFMにおいて、佐藤さんから見て右側に座っている人の名前をすべて挙げてください。")
    staff = inv.answer_question(
        "各案件のPP・契約書・PLAN・FRにおいて、DA側の実施体制として役割付きで記載されている人物は全部で何人ですか。")
    assert seating.answer.answer == "鈴木、藤田"
    assert seating.contract == "full_enumeration" and seating.model == "deterministic"
    assert staff.answer.answer == "19"
    assert staff.contract == "cross_aggregate" and staff.model == "deterministic"
    assert seating.usage.total_tokens == staff.usage.total_tokens == 0


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


def test_gemini_model_normalizes_missing_candidate_role(monkeypatch):
    from google.genai import types
    from src.rag import llm

    response = SimpleNamespace(
        candidates=[SimpleNamespace(content=types.Content(
            role=None, parts=[types.Part.from_text(text="done")]))],
        usage_metadata=None,
    )
    fake_client = SimpleNamespace(models=SimpleNamespace(
        generate_content=lambda **_kwargs: response))
    monkeypatch.setattr(llm, "client", lambda: fake_client)

    model = inv.GeminiModel("q", [])
    step = model.next(None)

    assert step.final_text == "done"
    assert model._contents[-1].role == "model"
