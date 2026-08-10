"""SOT-2468 — offline tests for the Gemini function-calling investigation loop.

Everything here runs network-free: the loop is driven by a *scripted* fake model, and the tool layer
uses the real deterministic tools (a real ``find_files`` grep over the corpus, plus a temp-CSV
``compute``) so tool wiring is exercised end-to-end without touching Vertex.
"""
from __future__ import annotations

import json
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
    assert out["evidence"] == {
        "applicable": True, "resolved": True, "coverage": "all-slides/all-sheets"}
    assert out["method"]["engine"] == "diffpair"


def test_version_diff_tool_returns_none_value_when_solver_abstains(monkeypatch):
    # diff question but the solver can't resolve a unique pair → value None, resolved False (棄権のまま).
    from src.rag import diffpair

    monkeypatch.setattr(diffpair, "is_diff_question", lambda q: True)
    monkeypatch.setattr(diffpair, "answer_question_agent", lambda q: None)
    tools = {t.name: t for t in build_tools(CorpusProfile())}
    out = dispatch(tools, "version_diff", {"question": "幻の資料の旧版と最新版を比較して変更点を。"})
    assert out["value"] is None
    assert out["evidence"] == {
        "applicable": True, "resolved": False, "coverage": "all-slides/all-sheets"}


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
    assert out_v3["evidence"] == {
        "applicable": True, "resolved": False, "coverage": "all-slides/all-sheets"}


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


def test_investigate_captures_pot_lane_verdict_into_details():
    """SOT-2586 — a verify_formula (PoT forced-lane) verdict is threaded onto the Investigation and
    surfaces in to_dict()/details.jsonl so measure_pot_lane.py --details can aggregate the three-layer
    accuracies. The retained trace carries the operand/formula/execution verdicts."""
    from src.rag.agent import pot_lane as pl

    verify_tool = AgentTool(
        pl.TOOL_NAME, pl.TOOL_DESCRIPTION, pl.TOOL_PARAMETERS,
        lambda candidates=None, simple=None, require_units=False: pl.verify_formula(
            candidates, simple=simple, require_units=bool(require_units)))
    tools = [verify_tool, inv.SUBMIT_ANSWER_TOOL]
    spec = {"operands": [{"name": "h", "value": 174, "unit": "時間", "source": "c:S1!B2"},
                         {"name": "rate", "value": 1500, "unit": "円", "source": "c:S1!B3"}],
            "formula": {"op": "MUL", "args": [{"ref": "h"}, {"ref": "rate"}]},
            "condition": None, "result_unit": "円"}
    model = ScriptedModel([
        Step(function_calls=(Call(pl.TOOL_NAME, {"candidates": [spec]}),), usage=Usage(50, 10)),
        _submit("261000円", confidence=0.95, evidence="verify_formula", method="pot_lane"),
    ])
    res = investigate(model, "…", tools, max_turns=5)
    assert res.stop_reason == "answered"
    assert res.pot_lane is not None and res.pot_lane.get("candidates")
    d = res.to_dict()
    assert "pot_lane" in d and d["pot_lane"] is res.pot_lane
    chosen = res.pot_lane["candidates"][0]
    assert set(chosen["verdicts"]) == {pl.LAYER_OPERAND, pl.LAYER_FORMULA, pl.LAYER_EXECUTION}
    json.dumps(d, ensure_ascii=False)  # details.jsonl serialization must not raise


def test_investigate_omits_pot_lane_when_lane_not_exercised():
    """SOT-2586 — no verify_formula call ⇒ pot_lane stays None ⇒ key absent from to_dict()/details, so a
    lane-OFF run's details.jsonl is byte-identical to the champion's."""
    tools = [_tool_that_records([]), inv.SUBMIT_ANSWER_TOOL]
    model = ScriptedModel([
        Step(function_calls=(Call("compute", {}),), usage=Usage(10, 5)),
        _submit("20", confidence=0.9),
    ])
    res = investigate(model, "…", tools, max_turns=5)
    assert res.pot_lane is None
    assert "pot_lane" not in res.to_dict()


def test_pot_hard_lane_rejects_numeric_submit_until_verified(monkeypatch):
    """SOT-2586 — ON means a real terminal gate: direct NUMERIC submit cannot bypass verify_formula."""
    from src.rag.agent import pot_lane as pl

    monkeypatch.setattr(inv, "POT_HARD_LANE", True)
    monkeypatch.setattr(pl, "enabled", lambda: True)
    verify_tool = AgentTool(
        pl.TOOL_NAME, pl.TOOL_DESCRIPTION, pl.TOOL_PARAMETERS,
        lambda candidates=None, simple=None, require_units=False: pl.verify_formula(
            candidates, simple=simple, require_units=bool(require_units)))
    spec = {"operands": [{"name": "x", "value": 2, "unit": "円", "source": "c:S!A1"},
                         {"name": "y", "value": 3, "unit": "円", "source": "c:S!A2"}],
            "formula": {"op": "ADD", "args": [{"ref": "x"}, {"ref": "y"}]},
            "condition": None, "result_unit": "円"}
    model = ScriptedModel([
        _submit("5円", confidence=0.9),
        Step(function_calls=(Call(pl.TOOL_NAME, {"candidates": [spec]}),), usage=Usage(20, 10)),
        _submit("5円", confidence=0.9),
    ])
    res = investigate(model, "2円と3円の合計はいくらですか？", [verify_tool, inv.SUBMIT_ANSWER_TOOL],
                      max_turns=5, contract="numeric")
    assert res.answer.answer == "5円"
    assert res.tool_calls.count(inv.SUBMIT_ANSWER) == 2
    assert res.pot_lane is not None


def test_pot_hard_lane_failure_degrades_to_freeform_not_forced_abstain(monkeypatch):
    """SOT-2598 — a *failed* verify_formula must NOT force the numeric answer to abstain. Once the lane
    has been exercised (verdict non-None, even a non-COMMIT one), an evidence-supported free-form value
    commits through the normal gate (原則 wrong→abstain 可 / match→abstain 不可). This pins the code
    invariant behind the directive relaxation: the terminal gate rejects only an *unverified* (verdict is
    None) submit, never a submit that follows a failing lane verdict."""
    from src.rag.agent import pot_lane as pl

    monkeypatch.setattr(inv, "POT_HARD_LANE", True)
    monkeypatch.setattr(pl, "enabled", lambda: True)
    verify_tool = AgentTool(
        pl.TOOL_NAME, pl.TOOL_DESCRIPTION, pl.TOOL_PARAMETERS,
        lambda candidates=None, simple=None, require_units=False: pl.verify_formula(
            candidates, simple=simple, require_units=bool(require_units)))
    # A candidate the lane cannot verify: the formula references an unbound operand ``y`` → operand-layer
    # failure → the whole run abstains (status != COMMIT). The model then falls back to a source-grounded
    # free-form value rather than abstaining.
    fail_spec = {"operands": [{"name": "x", "value": 2, "unit": "円", "source": "c:S!A1"}],
                 "formula": {"op": "ADD", "args": [{"ref": "x"}, {"ref": "y"}]},
                 "condition": None, "result_unit": "円"}
    model = ScriptedModel([
        _submit("73260円", confidence=0.9),                                   # unverified → rejected by gate
        Step(function_calls=(Call(pl.TOOL_NAME, {"candidates": [fail_spec]}),), usage=Usage(20, 10)),
        _submit("73260円", confidence=0.9),                                   # free-form degrade → accepted
    ])
    res = investigate(model, "原文に記載の合計金額はいくらですか？",
                      [verify_tool, inv.SUBMIT_ANSWER_TOOL], max_turns=6, contract="numeric")
    assert res.stop_reason == "answered"
    assert res.answer.answer == "73260円"          # NOT dropped to abstain despite the failed lane
    assert not is_abstain(res.answer.answer)
    assert res.pot_lane is not None and res.pot_lane.get("status") != pl.COMMIT  # a failing verdict was recorded


def test_numeric_lane_directive_degrades_on_failure_not_forced_abstain():
    """SOT-2598 — the injected directive no longer force-abstains on lane failure. It must (a) still bar
    number fabrication, (b) tell the model to fall back to the normal answer path when the source supports
    a value, and (c) not instruct an unconditional abstain on any layer failure."""
    from src.rag.agent import pot_lane as pl

    directive = pl.numeric_lane_directive()
    assert "即棄権にはしないでください" in directive        # failure ⇒ degrade, not forced abstain
    assert "submit_answer" in directive                    # names the normal free-form answer path
    assert "創作" in directive                             # still bars fabricating a number
    # The old unconditional "…場合は数値を創作せず棄権してください" instruction must be gone.
    assert "一致しない場合は数値を創作せず棄権してください" not in directive


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


def test_version_diff_contract_requires_tool_and_exact_deterministic_value():
    expected = "スライド6 追加：4.1 データ理解 / 4.2 前処理"
    diff_tool = AgentTool(
        "version_diff", "d", {"type": "object", "properties": {}},
        lambda **kw: {"value": expected,
                      "evidence": {"resolved": True, "coverage": "all-slides/all-sheets"},
                      "method": {"engine": "diffpair"}},
    )
    model = ScriptedModel([
        _submit("費用が変更された"),                 # mandatory tool not run
        Step(function_calls=(Call("version_diff", {"question": "q"}),)),
        _submit("4.1と4.2が追加された"),             # paraphrase is not the tool value
        _submit(expected),
    ])
    result = investigate(
        model, "旧版から新版への変更を比較してください。",
        [diff_tool, inv.SUBMIT_ANSWER_TOOL], contract="version_diff", max_turns=6)
    assert result.answer.answer == expected
    delivered = [r.response for turn in model.calls_seen if turn for r in turn
                 if r.name == SUBMIT_ANSWER]
    assert sum(bool(r.get("answer_rejected")) for r in delivered) == 1
    assert result.tool_calls.count("version_diff") == 1


def test_resolved_version_diff_rejects_abstain_and_commits_exact_value():
    expected = "スライド6 追加：4.1 データ理解"
    diff_tool = AgentTool(
        "version_diff", "d", {"type": "object", "properties": {}},
        lambda **kw: {"value": expected,
                      "evidence": {"resolved": True, "coverage": "all-slides/all-sheets"},
                      "method": {"engine": "diffpair"}},
    )
    model = ScriptedModel([
        Step(function_calls=(Call("version_diff", {"question": "q"}),)),
        _submit(inv.ABSTAIN),
        _submit(expected),
    ])
    result = investigate(
        model, "旧版から新版への変更を比較してください。",
        [diff_tool, inv.SUBMIT_ANSWER_TOOL], contract="version_diff", max_turns=5)
    assert result.answer.answer == expected
    assert result.tool_calls == ["version_diff"]


def test_literal_contract_rejects_candidate_without_same_fragment_evidence():
    lookup = AgentTool(
        "read_office", "d", {"type": "object", "properties": {}},
        lambda **kw: {"value": [
            {"item": "データ移行支援", "condition": "本契約内"},
            {"item": "監視ダッシュボード構築", "condition": "別契約"},
        ], "evidence": {"file": "proposal.pptx"}, "method": {"engine": "office"}},
    )
    q = "データアステル側の役割として「別契約」と明記されたものを抽出してください。"
    model = ScriptedModel([
        Step(function_calls=(Call("read_office", {"file": "proposal.pptx"}),)),
        _submit("データ移行支援"),
        _submit("監視ダッシュボード構築"),
    ])
    result = investigate(model, q, [lookup, inv.SUBMIT_ANSWER_TOOL],
                         contract="simple_lookup", max_turns=5)
    assert result.answer.answer == "監視ダッシュボード構築"
    delivered = [r.response for turn in model.calls_seen if turn for r in turn
                 if r.name == SUBMIT_ANSWER]
    assert any(r.get("answer_rejected") and "literal" in r.get("reason", "")
               for r in delivered)


def test_single_strict_literal_vision_candidate_commits_without_model_reselection():
    vision = AgentTool(
        "caption_image", "d", {"type": "object", "properties": {}},
        lambda **kw: {
            "value": [{
                "page": 8,
                "scope": "データクラフト",
                "candidate": "監視ダッシュボード構築",
                "condition": "別契約",
                "source": "モデル再現コード提供、バッチ設計\n監視ダッシュボード構築（別契約）",
                "conditioned_text": "監視ダッシュボード構築（別契約）",
                "same_visual_line": True,
            }],
            "evidence": {"file": "report.pdf", "question_specific": True},
            "method": {"engine": "vision"},
        },
    )
    q = "データアステル側の役割として『別契約』と明記されたものを抽出してください。"
    model = ScriptedModel([
        Step(function_calls=(Call("caption_image", {"file": "report.pdf", "question": q}),)),
        _submit("隣接する誤候補"),
    ])
    result = investigate(model, q, [vision, inv.SUBMIT_ANSWER_TOOL],
                         contract="simple_lookup", max_turns=4)
    assert result.answer.answer == "監視ダッシュボード構築"
    assert result.answer.confidence == 1.0
    assert result.tool_calls == ["caption_image"]
    assert len(model.calls_seen) == 1


def test_literal_report_answer_resolves_unique_report_and_commits_strict_candidate(monkeypatch):
    q = "ある病院の今後の運用で『別契約』と明記されたものを抽出してください。"
    monkeypatch.setattr(inv, "canonical_route", lambda question: {
        "value": [], "evidence": {"project": "ある病院"}, "method": {"engine": "route"},
    })
    monkeypatch.setattr(inv, "find_files", lambda query, ext=None: {
        "value": [
            {"rel": "project/meeting.pdf", "ext": "pdf", "category": "meeting"},
            {"rel": "project/final.pdf", "ext": "pdf", "category": "report"},
        ],
        "evidence": {"matched": 2}, "method": {"engine": "corpus"},
    })
    seen = []

    def caption(file, question=None):
        seen.append((file, question))
        return {
            "value": [{
                "candidate": "監視ダッシュボード構築",
                "condition": "別契約",
                "conditioned_text": "監視ダッシュボード構築（別契約）",
                "same_visual_line": True,
            }],
            "evidence": {"file": file},
            "method": {"engine": "vision"},
        }

    monkeypatch.setattr(inv, "caption_figure", caption)
    answer = inv._deterministic_literal_report_answer(q)
    assert answer is not None
    assert answer.answer == "監視ダッシュボード構築"
    assert seen == [("project/final.pdf", q)]


def test_literal_report_answer_fails_closed_when_report_is_ambiguous(monkeypatch):
    q = "ある病院の今後の運用で『別契約』と明記されたものを抽出してください。"
    monkeypatch.setattr(inv, "canonical_route", lambda question: {
        "value": [], "evidence": {"project": "ある病院"}, "method": {"engine": "route"},
    })
    monkeypatch.setattr(inv, "find_files", lambda query, ext=None: {
        "value": [
            {"rel": "project/a.pdf", "ext": "pdf", "category": "report"},
            {"rel": "project/b.pdf", "ext": "pdf", "category": "report"},
        ],
        "evidence": {"matched": 2}, "method": {"engine": "corpus"},
    })
    monkeypatch.setattr(inv, "caption_figure", lambda *args, **kwargs: pytest.fail("must not scan"))
    assert inv._deterministic_literal_report_answer(q) is None


def test_simple_lookup_empty_first_turn_gets_one_bounded_tool_retry():
    lookup = AgentTool(
        "read_office", "d", {"type": "object", "properties": {}},
        lambda **kw: {"value": "フェーズNo6 | T27 | 最終報告", "method": {"engine": "office"}},
    )
    model = ScriptedModel([
        Step(function_calls=(), final_text="", usage=Usage(1, 0)),
        Step(function_calls=(Call("read_office", {"file": "schedule.xlsx"}),)),
        _submit("最終報告"),
    ])
    result = investigate(model, "スケジュール.xlsxの最後のタスクは何ですか。",
                         [lookup, inv.SUBMIT_ANSWER_TOOL],
                         contract="simple_lookup", max_turns=4)
    assert result.answer.answer == "最終報告"
    assert result.tool_calls == ["read_office", SUBMIT_ANSWER]
    assert model.calls_seen[1] and model.calls_seen[1][0].name == inv.DIRECTIVE_MESSAGE


def test_simple_lookup_empty_retry_is_not_repeated():
    model = ScriptedModel([
        Step(function_calls=(), final_text="", usage=Usage(1, 0)),
        Step(function_calls=(), final_text="", usage=Usage(1, 0)),
    ])
    result = investigate(model, "対象ファイルの値は何ですか。", [inv.SUBMIT_ANSWER_TOOL],
                         contract="simple_lookup", max_turns=4)
    assert result.answer.answer == ABSTAIN
    assert len(model.calls_seen) == 2


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


def test_investigate_batch_shares_one_profile_when_requested():
    """SOT-2528: shared_profile threads ONE instance through every question so discoveries carry over."""
    shared = CorpusProfile()
    factory_calls = []

    def profile_factory():  # must NOT be consulted when shared_profile is given
        p = CorpusProfile()
        factory_calls.append(p)
        return p

    seen_profiles = []

    def recording_factory(question: str, tools):
        # the tools are bound to whichever profile investigate_batch chose for this question
        seen_profiles.append(tools)
        return ScriptedModel([_submit("x", confidence=0.8)])

    questions = [f"q{i}" for i in range(4)]
    # seed a discovery before the batch; every question must see it (reuse, not re-derive)
    shared.set_password("かえで/train.xlsx", "pw")
    investigate_batch(recording_factory, questions,
                      profile_factory=profile_factory, shared_profile=shared, max_turns=2)

    assert factory_calls == []  # shared_profile takes precedence over the per-question factory
    assert shared.get_password("かえで/train.xlsx") == "pw"  # instance is the one we seeded


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


# --------------------------------------------------------------------------- SOT-2523 adaptive budget
def _stub_investigate_capturing(monkeypatch, captured):
    """Replace ``inv.investigate`` with a stub that records the budget it was handed and returns an
    abstain Investigation, so ``answer_question``'s budget adaptation can be asserted offline."""
    def fake_investigate(model, question, tools, **kw):
        captured["max_turns"] = kw.get("max_turns")
        captured["timeout_s"] = kw.get("timeout_s")
        return Investigation(
            question=question, answer=Answer(ABSTAIN, 0.0), iterations=0, tool_calls=[],
            usage=Usage(), model="stub", elapsed_s=0.0, stop_reason="answered",
            contract=kw.get("contract"))
    monkeypatch.setattr(inv, "investigate", fake_investigate)
    monkeypatch.setattr(inv, "gemini_model_factory", lambda *a, **k: object())


def _force_contract(monkeypatch, contract):
    from src.rag.agent import routing as _routing
    monkeypatch.setattr(_routing, "classify_for_routing",
                        lambda q, flash=None: SimpleNamespace(contract=contract))
    monkeypatch.setattr(_routing, "routed_system_prompt", lambda base, qc, q: base)


@pytest.mark.parametrize("contract", ["numeric", "multi_hop", "cross_aggregate", "full_enumeration"])
def test_multistage_contract_lifts_budget(monkeypatch, contract):
    captured = {}
    _stub_investigate_capturing(monkeypatch, captured)
    _force_contract(monkeypatch, contract)
    inv.answer_question("この案件の集計に関する多段の質問です")
    assert captured["max_turns"] == inv.ADAPTIVE_MAX_TURNS == 18
    assert captured["timeout_s"] == inv.ADAPTIVE_TIMEOUT_S == 240.0


@pytest.mark.parametrize("contract", ["simple_lookup", "spatial", "format_check", "version_diff"])
def test_single_stage_contract_keeps_flat_budget(monkeypatch, contract):
    captured = {}
    _stub_investigate_capturing(monkeypatch, captured)
    _force_contract(monkeypatch, contract)
    inv.answer_question("ある値を1つ引くだけの単純な質問です")
    assert captured["max_turns"] == inv.DEFAULT_MAX_TURNS == 12
    assert captured["timeout_s"] == inv.DEFAULT_TIMEOUT_S == 180.0


def test_multistage_ratio_composes_with_lift(monkeypatch):
    captured = {}
    _stub_investigate_capturing(monkeypatch, captured)
    _force_contract(monkeypatch, "numeric")
    from src.rag.agent import question_contract as _qc
    monkeypatch.setattr(_qc, "numeric_requirements",
                        lambda q: SimpleNamespace(ratio=True))
    inv.answer_question("AのうちBの割合は何%ですか")
    # 12 -> 18 (multi-stage) -> +4 (ratio needs separate num/denom + validation) = 22, bounded.
    assert captured["max_turns"] == inv.ADAPTIVE_MAX_TURNS + 4 == 22
    assert captured["timeout_s"] == inv.ADAPTIVE_TIMEOUT_S == 240.0


def test_adaptive_budget_env_off_restores_flat_budget(monkeypatch):
    captured = {}
    _stub_investigate_capturing(monkeypatch, captured)
    _force_contract(monkeypatch, "cross_aggregate")
    monkeypatch.setattr(inv, "ADAPTIVE_BUDGET", False)
    inv.answer_question("全案件を横断した集計の質問です")
    assert captured["max_turns"] == 12 and captured["timeout_s"] == 180.0


def test_explicit_caller_budget_is_never_shrunk_or_lifted(monkeypatch):
    captured = {}
    _stub_investigate_capturing(monkeypatch, captured)
    _force_contract(monkeypatch, "cross_aggregate")
    # A caller that pins a non-default budget keeps it verbatim (adaptation only lifts the default).
    inv.answer_question("全案件を横断した集計の質問です", max_turns=30, timeout_s=90.0)
    assert captured["max_turns"] == 30 and captured["timeout_s"] == 90.0


# --------------------------------------------------------------------------- SOT-2523 evidence cache
def _counting_tool(sink, *, name="compute", result=None):
    payload = result if result is not None else {"value": 20, "evidence": {}, "method": {}}

    def fn(**kw):
        sink.append(kw)
        return payload
    return AgentTool(name, "d", {"type": "object", "properties": {}}, fn)


def test_evidence_cache_memoises_identical_tool_calls():
    sink = []
    tools = [_counting_tool(sink), inv.SUBMIT_ANSWER_TOOL]
    args = {"file": "f", "expr": "df['b'].mean()"}
    model = ScriptedModel([
        Step(function_calls=(Call("compute", dict(args)),), usage=Usage(1, 1)),
        Step(function_calls=(Call("compute", dict(args)),), usage=Usage(1, 1)),
        _submit("平均は20です"),
    ])
    res = investigate(model, "…", tools, max_turns=6)
    assert res.stop_reason == "answered"
    # The identical second call is served from the intra-question cache: the tool ran exactly once,
    # yet both tool rounds are still counted and both delivered the value to the model.
    assert len(sink) == 1
    assert res.tool_calls.count("compute") == 2 and res.iterations == 2


def test_evidence_cache_distinguishes_different_args():
    sink = []
    tools = [_counting_tool(sink), inv.SUBMIT_ANSWER_TOOL]
    model = ScriptedModel([
        Step(function_calls=(Call("compute", {"file": "a"}),), usage=Usage(1, 1)),
        Step(function_calls=(Call("compute", {"file": "b"}),), usage=Usage(1, 1)),
        _submit("done"),
    ])
    investigate(model, "…", tools, max_turns=6)
    assert len(sink) == 2  # different args are distinct cache keys → both execute


def test_evidence_cache_never_caches_errors():
    sink = []
    tools = [_counting_tool(sink, result={"error": "boom"}), inv.SUBMIT_ANSWER_TOOL]
    args = {"file": "f"}
    model = ScriptedModel([
        Step(function_calls=(Call("compute", dict(args)),), usage=Usage(1, 1)),
        Step(function_calls=(Call("compute", dict(args)),), usage=Usage(1, 1)),
        _submit("done"),
    ])
    investigate(model, "…", tools, max_turns=6)
    assert len(sink) == 2  # an error result is retried, not cached


def test_evidence_cache_env_off_reexecutes(monkeypatch):
    monkeypatch.setattr(inv, "EVIDENCE_CACHE", False)
    sink = []
    tools = [_counting_tool(sink), inv.SUBMIT_ANSWER_TOOL]
    args = {"file": "f", "expr": "x"}
    model = ScriptedModel([
        Step(function_calls=(Call("compute", dict(args)),), usage=Usage(1, 1)),
        Step(function_calls=(Call("compute", dict(args)),), usage=Usage(1, 1)),
        _submit("done"),
    ])
    investigate(model, "…", tools, max_turns=6)
    assert len(sink) == 2  # cache disabled → identical call executes twice


# --------------------------------------------------------------------------- SOT-2522 spin detection
def _grep_step(query):
    return Step(function_calls=(Call("file_grep", {"query": query}),), usage=Usage(1, 1))


def test_spin_detection_off_by_default_is_byte_identical():
    # With spin detection off (the default), a model that repeats the same call spins to max_turns
    # exactly as before — the tool executes every turn and the abstain is a plain budget cutoff.
    sink = []
    tools = [_counting_tool(sink, name="file_grep"), inv.SUBMIT_ANSWER_TOOL]
    steps = [_grep_step("同じ") for _ in range(10)]
    res = investigate(model := ScriptedModel(steps), "…", tools, max_turns=5)
    assert res.stop_reason == "max_turns"
    assert res.answer.answer == ABSTAIN
    assert res.tool_calls == ["file_grep"] * 5  # no spin directive injected
    assert "spin" not in (res.error or "")


def test_spin_detection_redirects_once_then_cuts_off(tmp_path: Path):
    # A model that keeps calling the same tool with the same args is detected as spinning: on the
    # threshold-th repeat it is redirected (once) toward untried deterministic routes; when it keeps
    # spinning, the path is cut off early and the abstain is recorded as SPIN_CUTOFF (not BUDGET).
    from src.rag.agent import abstain_ledger as al

    sink = []
    tools = [_counting_tool(sink, name="file_grep"), inv.SUBMIT_ANSWER_TOOL]
    steps = [_grep_step("同じ") for _ in range(12)]  # model never varies its call
    model = ScriptedModel(steps)
    ledger_path = tmp_path / "abstain.jsonl"
    res = investigate(model, "対象の値は？", tools, max_turns=12,
                      spin_detection={"threshold": 3}, ledger=str(ledger_path))
    assert res.stop_reason == "spin_cutoff"
    assert res.answer.answer == ABSTAIN
    # cut off well before max_turns=12 → budget was freed (無駄ターン減少)
    assert res.iterations < 12
    # the reallocation directive was fed back exactly once as a spin_detected response
    spin_msgs = [tr for turn in model.calls_seen if turn for tr in turn
                 if isinstance(tr.response, dict) and tr.response.get("spin_detected")]
    assert len(spin_msgs) == 1
    directive = spin_msgs[0].response["directive"]
    assert "canonical_route" in directive  # redirect names untried deterministic routes
    # the abstain is attributed to SPIN_CUTOFF in the ledger, distinct from BUDGET_EXHAUSTED
    rows = [json.loads(x) for x in ledger_path.read_text(encoding="utf-8").splitlines()]
    assert len(rows) == 1 and rows[0]["state_code"] == al.SPIN_CUTOFF


def test_spin_redirect_lets_model_recover_via_another_route():
    # After the reallocation directive, a model that switches to a different route and commits an answer
    # finishes normally — the redirect reallocated the budget instead of dead-ending the question.
    sink = []
    tools = [_counting_tool(sink, name="file_grep"),
             _counting_tool(sink, name="canonical_route"), inv.SUBMIT_ANSWER_TOOL]
    steps = [
        _grep_step("同じ"), _grep_step("同じ"), _grep_step("同じ"),  # spins to threshold=3 → redirect
        Step(function_calls=(Call("canonical_route", {"question": "q"}),), usage=Usage(1, 1)),
        _submit("回答42", confidence=0.9, evidence="canonical_route", method="route"),
    ]
    model = ScriptedModel(steps)
    res = investigate(model, "対象の値は？", tools, max_turns=12, spin_detection=True)
    assert res.stop_reason == "answered"
    assert res.answer.answer == "回答42"
    assert "canonical_route" in res.tool_calls


def test_spin_detection_ignores_varied_arguments():
    # Distinct arguments are not a spin: a model exploring different queries is never cut off.
    sink = []
    tools = [_counting_tool(sink, name="file_grep"), inv.SUBMIT_ANSWER_TOOL]
    steps = [
        _grep_step("a"), _grep_step("b"), _grep_step("c"), _grep_step("d"),
        _submit("見つけた", confidence=0.8, evidence="file_grep", method="grep"),
    ]
    model = ScriptedModel(steps)
    res = investigate(model, "…", tools, max_turns=12, spin_detection={"threshold": 3})
    assert res.stop_reason == "answered"
    assert res.answer.answer == "見つけた"
    spin_msgs = [tr for turn in model.calls_seen if turn for tr in turn
                 if isinstance(tr.response, dict) and tr.response.get("spin_detected")]
    assert not spin_msgs  # varied args → no spin directive


# ----------------------------------------------------------------- SOT-2614 consecutive-spin pivot guard
def _pivot_msgs(model):
    return [tr for turn in model.calls_seen if turn for tr in turn
            if isinstance(tr.response, dict) and tr.response.get("spin_pivot")]


def test_spin_pivot_off_by_default_is_byte_identical():
    # Guard off (default): a model that hammers one tool with *varied* queries spins to max_turns exactly
    # as before — every call executes, no pivot directive, and spin_pivots is absent from the details.
    sink = []
    tools = [_counting_tool(sink, name="file_grep"), inv.SUBMIT_ANSWER_TOOL]
    steps = [_grep_step(q) for q in ("a", "b", "c", "d", "e", "f", "g", "h")]
    res = investigate(model := ScriptedModel(steps), "…", tools, max_turns=5)
    assert res.stop_reason == "max_turns"
    assert res.tool_calls == ["file_grep"] * 5     # every varied grep dispatched, no pivot
    assert not _pivot_msgs(model)
    assert res.spin_pivots == 0
    assert "spin_pivots" not in res.to_dict()      # omitted when the guard never fired → byte-identical


def test_spin_pivot_fires_on_consecutive_varied_search():
    # The strengthened guard catches what SOT-2522 misses: a consecutive run of ONE search tool whose
    # query is tweaked each call. On the 3rd consecutive call the redundant dispatch is withheld and a
    # pivot directive is fed back instead; the pivot is counted into the details log.
    sink = []
    tools = [_counting_tool(sink, name="file_grep"), inv.SUBMIT_ANSWER_TOOL]
    steps = [_grep_step("a"), _grep_step("b"), _grep_step("c")]
    model = ScriptedModel(steps)
    res = investigate(model, "対象の値は？", tools, max_turns=12,
                      pivot_detection={"threshold": 3, "low_turns": 0})
    pivots = _pivot_msgs(model)
    assert len(pivots) == 1
    assert "空回り" in pivots[0].response["reason"]
    assert "canonical_route" in pivots[0].response["directive"]
    assert len(sink) == 2                           # 3rd consecutive call was NOT dispatched (pivoted)
    assert res.spin_pivots == 1
    assert res.to_dict()["spin_pivots"] == res.spin_pivots


def test_spin_pivot_lets_model_recover_via_another_route():
    # After the pivot directive, a model that switches routes and commits finishes normally — the guard
    # reallocated the budget instead of dead-ending the question.
    sink = []
    tools = [_counting_tool(sink, name="file_grep"),
             _counting_tool(sink, name="canonical_route"), inv.SUBMIT_ANSWER_TOOL]
    steps = [
        _grep_step("a"), _grep_step("b"), _grep_step("c"),   # 3 consecutive varied greps → pivot
        Step(function_calls=(Call("canonical_route", {"question": "q"}),), usage=Usage(1, 1)),
        _submit("回答42", confidence=0.9, evidence="canonical_route", method="route"),
    ]
    model = ScriptedModel(steps)
    res = investigate(model, "対象の値は？", tools, max_turns=12, pivot_detection=True)
    assert res.stop_reason == "answered"
    assert res.answer.answer == "回答42"
    assert "canonical_route" in res.tool_calls
    assert res.spin_pivots >= 1


def test_spin_pivot_different_file_targets_not_flagged():
    # Walking DIFFERENT files with a file-carrying tool is legitimate progress, not spin: the run resets
    # on each new target, so no pivot fires.
    sink = []
    tools = [_counting_tool(sink, name="read_office"), inv.SUBMIT_ANSWER_TOOL]
    steps = [
        Step(function_calls=(Call("read_office", {"file": f}),), usage=Usage(1, 1))
        for f in ("a.xlsx", "b.xlsx", "c.xlsx", "d.xlsx")
    ] + [_submit("見つけた", evidence="read_office", method="x")]
    model = ScriptedModel(steps)
    res = investigate(model, "…", tools, max_turns=12, pivot_detection={"threshold": 3})
    assert res.stop_reason == "answered"
    assert not _pivot_msgs(model)
    assert res.spin_pivots == 0


def test_spin_pivot_same_file_guess_and_check_is_flagged():
    # Hammering ONE file with tweaked compute exprs (the idx47/57/95 guess-and-check pattern SOT-2522
    # misses because the args differ) is caught: same file target across the consecutive run → pivot.
    sink = []
    tools = [_counting_tool(sink, name="compute"), inv.SUBMIT_ANSWER_TOOL]
    steps = [
        Step(function_calls=(Call("compute", {"file": "t.xlsx", "expr": e}),), usage=Usage(1, 1))
        for e in ("df.a.mean()", "df.b.sum()", "df.c.max()")
    ]
    model = ScriptedModel(steps)
    res = investigate(model, "…", tools, max_turns=12, pivot_detection={"threshold": 3, "low_turns": 0})
    assert _pivot_msgs(model)
    assert len(sink) == 2                            # 3rd same-file compute withheld
    assert res.spin_pivots == 1


def test_spin_pivot_low_turns_escalates_to_verdict():
    # When few turns remain, the pivot escalates to a definitive-decision directive (answer-or-abstain)
    # so the freed budget lands on a verdict rather than more spinning.
    sink = []
    tools = [_counting_tool(sink, name="file_grep"), inv.SUBMIT_ANSWER_TOOL]
    steps = [_grep_step(q) for q in ("a", "b", "c", "d")]
    model = ScriptedModel(steps)
    res = investigate(model, "…", tools, max_turns=12,
                      pivot_detection={"threshold": 3, "low_turns": 99})
    pivots = _pivot_msgs(model)
    assert pivots
    assert "残ターンが僅少" in pivots[0].response["directive"]
    assert "棄権" in pivots[0].response["directive"]


def test_spin_pivot_records_spin_pivots_to_ledger(tmp_path: Path):
    # A pivot count is recorded onto the abstain ledger only when it fired (>0), so the diagnosis can
    # measure the guard's capture rate; the state code is unchanged when no early cutoff occurred.
    sink = []
    tools = [_counting_tool(sink, name="file_grep"), inv.SUBMIT_ANSWER_TOOL]
    steps = [_grep_step(q) for q in ("a", "b", "c", "d", "e", "f")]
    model = ScriptedModel(steps)
    ledger_path = tmp_path / "abstain.jsonl"
    res = investigate(model, "…", tools, max_turns=6,
                      pivot_detection={"threshold": 3, "low_turns": 0}, ledger=str(ledger_path))
    assert res.answer.answer == ABSTAIN
    assert res.spin_pivots >= 1
    rows = [json.loads(x) for x in ledger_path.read_text(encoding="utf-8").splitlines()]
    assert len(rows) == 1
    assert rows[0]["missing"][0]["signals"]["spin_pivots"] == res.spin_pivots


# ----------------------------------------------------------------- SOT-2620 per-route search-call cap
def _cap_msgs(model):
    return [tr for turn in model.calls_seen if turn for tr in turn
            if isinstance(tr.response, dict) and tr.response.get("search_cap_exceeded")]


def test_search_cap_off_by_default_is_byte_identical():
    # Cap off (default: search_cap=None): a model that issues many varied greps executes every one, no
    # cap directive is injected, and search_cap_hits is absent from the details → byte-identical path.
    sink = []
    tools = [_counting_tool(sink, name="file_grep"), inv.SUBMIT_ANSWER_TOOL]
    steps = [_grep_step(q) for q in ("a", "b", "c", "d", "e", "f", "g", "h")]
    res = investigate(model := ScriptedModel(steps), "…", tools, max_turns=5)
    assert res.stop_reason == "max_turns"
    assert res.tool_calls == ["file_grep"] * 5     # every varied grep dispatched, no cap
    assert not _cap_msgs(model)
    assert res.search_cap_hits == 0
    assert "search_cap_hits" not in res.to_dict()  # omitted when the cap never fired → byte-identical


def test_search_cap_caps_total_search_calls():
    # With an explicit cap, the (cap+1)-th search call is NOT dispatched — a switch-lane directive is fed
    # back instead — even though the greps are all distinct (so SOT-2522/2614 would never cut them).
    sink = []
    tools = [_counting_tool(sink, name="file_grep"), inv.SUBMIT_ANSWER_TOOL]
    steps = [_grep_step(q) for q in ("a", "b", "c", "d", "e", "f")] + [_submit("答")]
    model = ScriptedModel(steps)
    res = investigate(model, "対象の値は？", tools, max_turns=12, search_cap={"cap": 4})
    assert res.stop_reason == "answered"
    assert len(sink) == 4                            # only the first 4 searches actually ran
    caps = _cap_msgs(model)
    assert len(caps) == 2                            # 5th and 6th were intercepted
    assert "上限" in caps[0].response["reason"]
    assert "canonical_route" in caps[0].response["directive"]
    assert res.search_cap_hits == 2
    assert res.to_dict()["search_cap_hits"] == 2


def test_search_cap_lets_model_switch_lane_and_commit():
    # After the cap directive, a model that switches to a value/registry lane and commits finishes
    # normally — the cap redirected the budget instead of dead-ending the question.
    sink = []
    tools = [_counting_tool(sink, name="file_grep"),
             _counting_tool(sink, name="canonical_route"), inv.SUBMIT_ANSWER_TOOL]
    steps = [
        _grep_step("a"), _grep_step("b"),            # cap=2 → next grep is capped
        _grep_step("c"),                             # intercepted (not dispatched)
        Step(function_calls=(Call("canonical_route", {"question": "q"}),), usage=Usage(1, 1)),
        _submit("回答42", confidence=0.9, evidence="canonical_route", method="route"),
    ]
    model = ScriptedModel(steps)
    res = investigate(model, "対象の値は？", tools, max_turns=12, search_cap={"cap": 2})
    assert res.stop_reason == "answered"
    assert res.answer.answer == "回答42"
    assert "canonical_route" in res.tool_calls
    assert res.search_cap_hits == 1


def test_search_cap_resolves_per_route_from_contract():
    # search_cap=True (no explicit cap) resolves the cap from the question's contract via query_router:
    # a version_diff question gets the DIFF cap (3), so the 4th+ grep is capped.
    sink = []
    tools = [_counting_tool(sink, name="file_grep"), inv.SUBMIT_ANSWER_TOOL]
    steps = [_grep_step(q) for q in ("a", "b", "c", "d", "e")] + [_submit("答")]
    model = ScriptedModel(steps)
    res = investigate(model, "…", tools, max_turns=12, search_cap=True, contract="version_diff")
    assert len(sink) == 3                            # DIFF cap = 3
    assert res.search_cap_hits == 2


def test_search_cap_shares_budget_with_pivot_guard():
    # SOT-2614 の pivot ガードと重複介入しない: the cap check runs AFTER the pivot guard, so a call the
    # pivot guard already intercepted never also trips the cap (shared budget, no double directive).
    sink = []
    tools = [_counting_tool(sink, name="file_grep"), inv.SUBMIT_ANSWER_TOOL]
    steps = [_grep_step("a"), _grep_step("b"), _grep_step("c")] + [_submit("答")]
    model = ScriptedModel(steps)
    res = investigate(model, "…", tools, max_turns=12,
                      pivot_detection={"threshold": 3, "low_turns": 0}, search_cap={"cap": 2})
    # The 3rd consecutive grep is intercepted by the pivot guard, not the cap — so no cap directive fires
    # on it and search_cap_hits stays 0 even though 3 greps were issued against a cap of 2.
    assert res.spin_pivots == 1
    assert not _cap_msgs(model)
    assert res.search_cap_hits == 0
    assert len(sink) == 2                            # a,b dispatched; c pivoted (not dispatched)


def test_search_cap_records_hits_to_ledger(tmp_path: Path):
    # The cap-hit count is recorded onto the abstain ledger only when it fired (>0) so the diagnosis can
    # measure how often the search 上限 fired; omitted otherwise → byte-identical.
    sink = []
    tools = [_counting_tool(sink, name="file_grep"), inv.SUBMIT_ANSWER_TOOL]
    steps = [_grep_step(q) for q in ("a", "b", "c", "d", "e", "f")]
    model = ScriptedModel(steps)
    ledger_path = tmp_path / "abstain.jsonl"
    res = investigate(model, "…", tools, max_turns=8,
                      search_cap={"cap": 3}, ledger=str(ledger_path))
    assert res.answer.answer == ABSTAIN
    assert res.search_cap_hits >= 1
    rows = [json.loads(x) for x in ledger_path.read_text(encoding="utf-8").splitlines()]
    assert len(rows) == 1
    assert rows[0]["missing"][0]["signals"]["search_cap_hits"] == res.search_cap_hits
