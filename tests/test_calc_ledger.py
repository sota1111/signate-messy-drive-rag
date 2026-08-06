"""SOT-2495 — offline tests for the typed calculation ledger.

Two layers are exercised network-free, mirroring the abstain-ledger tests:

* the pure typed-value parser / schema / writer of :mod:`src.rag.agent.calc_ledger`, and
* its wiring into the investigator loop (:func:`src.rag.agent.investigator.investigate`) driven by a
  scripted fake model over a real ``compute`` on a temp CSV — proving every committed numeric answer
  writes a typed record (grounded when a compute step backs it, else flagged) while the answer path
  stays byte-identical to a run with the calc ledger off (受け入れ条件②).
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.rag.agent import calc_ledger as cl
from src.rag.agent.calc_ledger import (
    CalcSignals,
    LedgerError,
    is_grounded,
    is_numeric_answer,
    record_from_investigation,
    ungrounded,
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
    build_tools,
    investigate,
)
from src.rag.tools.profile import CorpusProfile


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


class _FakeAnswer:
    def __init__(self, answer, confidence=0.9, evidence="", method=""):
        self.answer, self.confidence, self.evidence, self.method = answer, confidence, evidence, method


class _FakeInvestigation:
    def __init__(self, **kw):
        self.question = kw.get("question", "平均は?")
        self.answer = kw.get("answer", _FakeAnswer("1526"))
        self.iterations = kw.get("iterations", 1)
        self.tool_calls = kw.get("tool_calls", ["compute", SUBMIT_ANSWER])
        self.usage = kw.get("usage", Usage(10, 5))
        self.model = kw.get("model", "fake-model")
        self.elapsed_s = kw.get("elapsed_s", 1.234)


# --------------------------------------------------------------------------- numeric detection
@pytest.mark.parametrize("answer,expected", [
    # real derived_calculation golds that are numeric → require a ledger entry
    ("958", True), ("0円", True), ("14,744ドル", True), ("2.21", True),
    ("-11851246", True), ("22,000円", True), ("8時間", True), ("79,200円の増加", True),
    ("1,234万円", True), ("1899年", True),
    # column-name answers (also derived_calculation golds) carry no quantity → excluded
    ("bmi", False), ("BMI", False), ("campaign", False),
    # abstain / empty never require an entry
    (ABSTAIN, False), ("", False), ("該当なし", False),
])
def test_is_numeric_answer(answer, expected):
    assert is_numeric_answer(answer) is expected


# --------------------------------------------------------------------------- typed-value parsing
@pytest.mark.parametrize("answer,parsed,unit,scale", [
    ("958", 958, None, 1),
    ("0円", 0, "円", 1),
    ("14,744ドル", 14744, "ドル", 1),
    ("2.21", 2.21, None, 1),
    ("-11851246", -11851246, None, 1),
    ("22,000円", 22000, "円", 1),
    ("8時間", 8, "時間", 1),
    ("79,200円の増加", 79200, "円", 1),   # unit adjacent to the number; trailing prose ignored
    ("1,234万円", 1234, "円", 10000),      # a 万円 unit normalizes to (円, scale 10000)
    ("35.2%", 35.2, "%", 1),
])
def test_typed_value_parsing(answer, parsed, unit, scale):
    tv = CalcSignals().typed_value(answer)
    assert tv.raw_text == answer
    assert tv.parsed_value == parsed
    assert tv.unit == unit
    assert tv.scale == scale


def test_typed_value_time_period_mined_from_question_and_answer():
    assert CalcSignals(question="2025年10月の売上は?").typed_value("100円").time_period == "2025年10月"
    assert CalcSignals(question="契約期間は?").typed_value("2025-08-15開始").time_period == "2025-08-15"
    assert CalcSignals(question="平均は?").typed_value("958").time_period is None


def test_typed_value_pulls_source_range_and_formula_from_last_step():
    sig = CalcSignals(question="loan_amntの平均は?")
    sig.observe("compute", {
        "value": 1526.0,
        "evidence": {"file": "train.csv", "sheet": "train", "range": "A1:A45",
                     "columns_used": ["loan_amnt"], "rows": 44, "project": "青葉"},
        "method": {"engine": "pandas", "code": "round(df['loan_amnt'].mean(), 2)",
                   "trace": {"input_rows": 44, "excluded_rows": 1,
                             "normalization": ["numeric_coercion: loan_amnt"]}},
    })
    tv = sig.typed_value("1,526円")
    assert tv.formula == "round(df['loan_amnt'].mean(), 2)"
    assert tv.entity == "青葉"
    sr = tv.source_range
    assert (sr.file, sr.sheet, sr.cell, sr.project) == ("train.csv", "train", "A1:A45", "青葉")


# --------------------------------------------------------------------------- signal observation
def test_observe_only_records_numeric_source_contracts():
    s = CalcSignals()
    s.observe("find_files", [{"file": "a"}])                  # not a derivation tool → ignored
    s.observe("read_office", {"value": "text", "evidence": {}, "method": {}})  # not numeric-source
    s.observe("compute", {"error": "boom"})                   # not a contract → ignored
    s.observe("compute", {"value": None, "evidence": {}, "method": {}})        # empty value → ignored
    s.observe("compute", {"value": 42, "evidence": {"file": "t.csv"},
                          "method": {"code": "df.sum()"}})     # recorded
    s.observe("corpus_aggregate", {"value": "青葉", "evidence": {"project": "青葉"}, "method": {}})
    assert len(s.steps) == 2
    assert s.steps[0].tool == "compute" and s.steps[0].code == "df.sum()"
    assert s.steps[1].tool == "corpus_aggregate"
    assert s.entity == "青葉"


# --------------------------------------------------------------------------- record + schema
def test_record_from_investigation_is_valid_and_typed():
    sig = CalcSignals(question="平均は?")
    sig.observe("compute", {"value": 20, "evidence": {"file": "t.csv", "range": "B1:B3",
                            "columns_used": ["b"]}, "method": {"code": "df['b'].mean()"}})
    rec = record_from_investigation(
        _FakeInvestigation(answer=_FakeAnswer("20件")), sig,
        now=lambda: "2026-08-06T00:00:00+00:00")
    d = rec.to_dict()
    validate(d)                                     # does not raise
    assert d["grounded"] is True
    assert d["answer"] == "20件"
    assert d["typed_value"]["parsed_value"] == 20 and d["typed_value"]["unit"] == "件"
    assert d["typed_value"]["formula"] == "df['b'].mean()"
    assert d["compute_steps"][0]["file"] == "t.csv"
    assert d["recorded_at"] == "2026-08-06T00:00:00+00:00"


def test_record_without_compute_step_is_ungrounded():
    rec = record_from_investigation(_FakeInvestigation(answer=_FakeAnswer("42")), CalcSignals())
    d = rec.to_dict()
    validate(d)
    assert d["grounded"] is False and d["compute_steps"] == []
    assert d["typed_value"]["formula"] is None
    # 欠落検出: an ungrounded numeric record is the machine signature of a 暗算 answer
    assert is_grounded(d) is False
    assert ungrounded([d]) == [d]


def test_validate_rejects_bad_records():
    good = record_from_investigation(_FakeInvestigation(), CalcSignals()).to_dict()
    validate(good)
    with pytest.raises(LedgerError):
        validate({k: v for k, v in good.items() if k != "typed_value"})
    with pytest.raises(LedgerError):
        validate(dict(good, typed_value={"raw_text": "x"}))          # typed_value missing fields
    with pytest.raises(LedgerError):
        validate(dict(good, grounded=True))                          # grounded ≠ (steps non-empty)


# --------------------------------------------------------------------------- writer
def test_write_record_appends_jsonl(tmp_path):
    p = tmp_path / "calc_ledger.jsonl"
    r1 = record_from_investigation(_FakeInvestigation(question="Q1", answer=_FakeAnswer("1")),
                                   CalcSignals())
    r2 = record_from_investigation(_FakeInvestigation(question="Q2", answer=_FakeAnswer("2")),
                                   CalcSignals())
    assert write_record(r1, p) == p
    write_record(r2, p)
    rows = [json.loads(x) for x in p.read_text(encoding="utf-8").strip().splitlines()]
    assert [r["question"] for r in rows] == ["Q1", "Q2"]
    for row in rows:
        validate(row)


# --------------------------------------------------------------------------- investigator wiring
def _grounded_run(tmp_path, csv_expr, answer):
    """Drive one numeric commit that computes over a temp CSV, then reads back the single record."""
    csv = tmp_path / "t.csv"
    csv.write_text("loan_amnt,age\n1000,30\n2000,40\n3000,50\n", encoding="utf-8")
    p = tmp_path / "calc.jsonl"
    tools = build_tools(CorpusProfile())
    steps = [_call("compute", file=str(csv), expr=csv_expr), _submit(answer, confidence=0.95)]
    inv = investigate(ScriptedModel(steps), "loan_amntの平均額は?", tools,
                      max_turns=5, calc_ledger=p)
    rows = [json.loads(x) for x in p.read_text(encoding="utf-8").strip().splitlines()]
    return inv, rows


# The gold-100 baseline has derived_calculation n=32 / MATCH=4. Each MATCH is a committed numeric answer;
# these four representatives (count / ratio / 通貨 / 円) must each generate a COMPLETE grounded record.
@pytest.mark.parametrize("expr,answer,parsed,unit", [
    ("int(df['loan_amnt'].count())", "3", 3, None),
    ("round(df['loan_amnt'].mean() / df['age'].mean(), 2)", "50.0", 50.0, None),
    ("int(df['loan_amnt'].max())", "3,000ドル", 3000, "ドル"),
    ("int(df['loan_amnt'].sum())", "6,000円", 6000, "円"),
])
def test_derived_calculation_match_generates_complete_ledger(tmp_path, expr, answer, parsed, unit):
    inv, rows = _grounded_run(tmp_path, expr, answer)
    assert inv.answer.answer == answer
    assert len(rows) == 1
    r = rows[0]
    validate(r)
    assert r["grounded"] is True
    tv = r["typed_value"]
    assert tv["parsed_value"] == parsed and tv["unit"] == unit
    assert tv["formula"] == expr                       # the executed code is preserved as the formula
    step = r["compute_steps"][0]
    assert step["tool"] == "compute"
    assert step["file"].endswith("t.csv")
    assert step["code"] == expr
    assert step["columns_used"]                        # the touched column(s) are recorded
    assert step["range"]                               # A1 source range
    assert step["input_rows"] == 3                     # rows the expression saw (計算証跡)
    assert step["normalization"] is not None


def test_compute_step_carries_structured_trace(tmp_path):
    # SOT-2495 compute_sandbox change: method.trace exposes input rows / excluded rows / normalization,
    # which the ledger folds into the compute step. A CSV with a text-then-numeric column is coerced.
    csv = tmp_path / "t.csv"
    csv.write_text("amount\n1000\n2000\n", encoding="utf-8")
    p = tmp_path / "calc.jsonl"
    tools = build_tools(CorpusProfile())
    steps = [_call("compute", file=str(csv), expr="int(df['amount'].sum())"),
             _submit("3000円", confidence=0.9)]
    investigate(ScriptedModel(steps), "合計は?", tools, max_turns=5, calc_ledger=p)
    step = json.loads(p.read_text(encoding="utf-8").splitlines()[0])["compute_steps"][0]
    assert step["input_rows"] == 2
    assert step["excluded_rows"] == 0
    assert isinstance(step["normalization"], list)


def test_ungrounded_numeric_answer_is_recorded_and_flagged(tmp_path):
    # 暗算: the model submits a number WITHOUT calling compute → a record still exists (必ず対応) but is
    # flagged grounded=False so 欠落検出 (a numeric answer lacking a calculation証跡) is machine-detectable.
    p = tmp_path / "calc.jsonl"
    tools = [_tool("find_files", [{"file": "hit"}]), *build_tools(CorpusProfile())]
    steps = [_call("find_files", query="x"), _submit("42件", confidence=0.9)]
    inv = investigate(ScriptedModel(steps), "件数は?", tools, max_turns=5, calc_ledger=p)
    assert inv.answer.answer == "42件"
    rows = [json.loads(x) for x in p.read_text(encoding="utf-8").strip().splitlines()]
    assert len(rows) == 1 and rows[0]["grounded"] is False
    assert ungrounded(rows) == rows


def test_no_record_for_abstain_or_non_numeric_commit(tmp_path):
    tools = build_tools(CorpusProfile())
    # abstain → no calc record
    p1 = tmp_path / "a.jsonl"
    investigate(ScriptedModel([_submit(ABSTAIN, confidence=0.0)]), "q", tools,
                max_turns=3, calc_ledger=p1)
    assert not p1.exists()
    # committed but non-numeric (a column-name answer) → no calc record
    p2 = tmp_path / "b.jsonl"
    inv = investigate(ScriptedModel([_submit("bmi", confidence=0.9)]), "どの列?", tools,
                      max_turns=3, calc_ledger=p2)
    assert inv.answer.answer == "bmi"
    assert not p2.exists()


def test_calc_ledger_off_is_a_noop_and_default_is_off(tmp_path, monkeypatch):
    monkeypatch.setattr(cl, "default_path", lambda: tmp_path / "should_not_exist.jsonl")
    tools = build_tools(CorpusProfile())
    inv = investigate(ScriptedModel([_submit("958", confidence=0.9)]), "q", tools,
                      max_turns=3, calc_ledger=None)
    assert inv.answer.answer == "958"
    assert not (tmp_path / "should_not_exist.jsonl").exists()


def test_calc_ledger_does_not_change_the_decision(tmp_path):
    """受け入れ条件②: 回答挙動は台帳のON/OFFで不変（記録のみ）。"""
    csv = tmp_path / "t.csv"
    csv.write_text("b\n10\n30\n", encoding="utf-8")
    steps = lambda: [_call("compute", file=str(csv), expr="df['b'].mean()"),
                     _submit("20件", confidence=0.9)]

    off = investigate(ScriptedModel(steps()), "q", build_tools(CorpusProfile()),
                      max_turns=8, calc_ledger=None)
    on = investigate(ScriptedModel(steps()), "q", build_tools(CorpusProfile()),
                     max_turns=8, calc_ledger=tmp_path / "l.jsonl")

    assert off.answer == on.answer
    assert off.stop_reason == on.stop_reason
    assert off.iterations == on.iterations
    assert off.tool_calls == on.tool_calls
    rows = [json.loads(x) for x in (tmp_path / "l.jsonl").read_text(encoding="utf-8").strip().splitlines()]
    assert rows[0]["grounded"] is True and rows[0]["typed_value"]["parsed_value"] == 20
