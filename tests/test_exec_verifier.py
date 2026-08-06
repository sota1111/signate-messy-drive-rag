"""SOT-2501 — offline tests for the execution verifier (計算台帳の独立再実行照合).

Network-free, mirroring the verifier/tiebreak/gate tests:

* the pure reconciliation core :func:`~src.rag.agent.exec_verifier.compare_execution` over two
  SOT-2495 calc-ledger records — proving the systematic numeric errors this exists to catch
  (相関ペアの対象列取り違え idx4/28 ・ 入力集合の包含差 idx10 ・ 値/単位の相違) surface as
  ``EVIDENCE_CONFLICT`` before commit, while a faithful independent re-execution confirms a correct
  number (既存 MATCH 非劣化);
* the driver :func:`~src.rag.agent.exec_verifier.verify_execution` with an injected re-derivation; and
* the live re-execution glue :func:`~src.rag.agent.exec_verifier.rederive_calc` exercised over a real
  ``compute`` on a temp CSV via a scripted fake model (no Vertex).
"""
from __future__ import annotations

from src.rag.agent.calc_ledger import CalcSignals, record_from_investigation, validate
from src.rag.agent.exec_verifier import (
    EXEC_CONFLICT,
    EXEC_MATCH,
    EXEC_UNDECIDABLE,
    EXEC_UNGROUNDED,
    ExecVerdict,
    compare_execution,
    is_numeric_record,
    rederive_calc,
    verify_execution,
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


# --------------------------------------------------------------------------- fakes / builders
class _FakeAnswer:
    def __init__(self, answer, confidence=0.9):
        self.answer, self.confidence, self.evidence, self.method = answer, confidence, "", ""


class _FakeInvestigation:
    def __init__(self, answer, question="Q?"):
        self.question = question
        self.answer = _FakeAnswer(answer)
        self.iterations = 1
        self.tool_calls = ["compute", SUBMIT_ANSWER]
        self.usage = Usage(10, 5)
        self.model = "fake-model"
        self.elapsed_s = 1.0


def _record(answer, *, columns=(), input_rows=None, code="df['x'].mean()", question="Q?",
            grounded=True):
    """A valid SOT-2495 calc-ledger record dict built through the real ledger machinery."""
    sig = CalcSignals(question=question)
    if grounded:
        sig.observe("compute", {
            "value": 1.0,
            "evidence": {"file": "t.csv", "range": "A1:A3",
                         "columns_used": list(columns), "rows": input_rows},
            "method": {"code": code, "trace": {"input_rows": input_rows}},
        })
    rec = record_from_investigation(_FakeInvestigation(answer, question), sig,
                                    now=lambda: "2026-08-06T00:00:00+00:00").to_dict()
    validate(rec)  # every record we feed the verifier is a well-formed ledger line
    return rec


class ScriptedModel:
    """Replays a fixed list of :class:`Step`s; abstains once exhausted (as in test_calc_ledger)."""

    def __init__(self, steps, *, model_name="fake-model"):
        self._steps, self._i, self.model_name = list(steps), 0, model_name

    def next(self, tool_responses):
        if self._i >= len(self._steps):
            return Step(function_calls=(), final_text=ABSTAIN, usage=Usage(1, 1))
        step = self._steps[self._i]
        self._i += 1
        return step


def _call(name, **args):
    return Step(function_calls=(Call(name, args),), usage=Usage(10, 5))


def _submit(answer, *, confidence=0.9):
    return Step(function_calls=(Call(SUBMIT_ANSWER, {
        "answer": answer, "confidence": confidence, "evidence": "e", "method": "m"}),),
        usage=Usage(20, 8))


# --------------------------------------------------------------------------- MATCH (既存 MATCH 非劣化)
def test_faithful_reexecution_confirms_number():
    committed = _record("0.42", columns=("age", "bmi"), input_rows=44)
    rederived = _record("0.42", columns=("age", "bmi"), input_rows=44)
    cmp = compare_execution(committed, rederived)
    assert cmp.category == EXEC_MATCH and cmp.match and not cmp.should_abstain


def test_rounding_tolerance_still_matches():
    # value axis reuses the verifier's rounding reconciliation: 1526 ≈ 1526.3 (coarser precision).
    committed = _record("1,526円", columns=("loan_amnt",), input_rows=44)
    rederived = _record("1526.3", columns=("loan_amnt",), input_rows=44)
    cmp = compare_execution(committed, rederived)
    assert cmp.category == EXEC_MATCH and cmp.match


# --------------------------------------------------------------------------- idx4/28: 相関ペア取り違え
def test_wrong_correlation_column_pair_is_conflict():
    """gold idx4/28 相当: 相関を age×bmi で問われたのに age×loan_amnt で計算した誤答。

    対象列の相違が直接 EVIDENCE_CONFLICT として検出され、commit 前に棄権へ倒せる。"""
    committed = _record("0.61", columns=("age", "loan_amnt"), input_rows=44)   # wrong pair, confident
    rederived = _record("0.12", columns=("age", "bmi"), input_rows=44)          # correct pair
    cmp = compare_execution(committed, rederived)
    assert cmp.category == EXEC_CONFLICT and not cmp.match and cmp.should_abstain
    assert any("対象列" in c for c in cmp.conflicts)     # the column mix-up is named
    assert any("値の相違" in c for c in cmp.conflicts)   # and the resulting value diverges


# --------------------------------------------------------------------------- idx10: 入力集合の包含差
def test_wrong_input_set_row_count_is_conflict():
    """gold idx10 相当: ヒストグラム/件数を誤った入力集合(行の包含/除外)で数えた誤答。"""
    committed = _record("50件", columns=("bmi",), input_rows=50)   # over-included rows
    rederived = _record("44件", columns=("bmi",), input_rows=44)   # correct input set
    cmp = compare_execution(committed, rederived)
    assert cmp.category == EXEC_CONFLICT and cmp.should_abstain
    assert any("入力集合の差" in c and "行" in c for c in cmp.conflicts)


# --------------------------------------------------------------------------- value / unit axes
def test_value_only_mismatch_is_conflict():
    committed = _record("1000", columns=("x",), input_rows=10)
    rederived = _record("1200", columns=("x",), input_rows=10)
    cmp = compare_execution(committed, rederived)
    assert cmp.category == EXEC_CONFLICT and any("値の相違" in c for c in cmp.conflicts)


def test_same_number_different_unit_is_conflict():
    # numerically equal but 単位 differs (円 vs ドル) → still a mismatch.
    committed = _record("1000円", columns=("x",), input_rows=10)
    rederived = _record("1000ドル", columns=("x",), input_rows=10)
    cmp = compare_execution(committed, rederived)
    assert cmp.category == EXEC_CONFLICT and any("単位の相違" in c for c in cmp.conflicts)


# --------------------------------------------------------------------------- 暗算 / 決着不能
def test_ungrounded_committed_cannot_be_confirmed():
    committed = _record("42", grounded=False)          # 暗算: no compute証跡
    rederived = _record("42", columns=("x",), input_rows=10)
    cmp = compare_execution(committed, rederived)
    assert cmp.category == EXEC_UNGROUNDED and not cmp.match and cmp.should_abstain


def test_reexecution_abstain_is_undecidable():
    committed = _record("42", columns=("x",), input_rows=10)
    cmp = compare_execution(committed, _record(ABSTAIN, grounded=False))
    assert cmp.category == EXEC_UNDECIDABLE and cmp.should_abstain


def test_reexecution_missing_is_undecidable():
    committed = _record("42", columns=("x",), input_rows=10)
    cmp = compare_execution(committed, None)
    assert cmp.category == EXEC_UNDECIDABLE and cmp.should_abstain


# --------------------------------------------------------------------------- driver
def test_verify_execution_injects_rederivation():
    committed = _record("0.61", columns=("age", "loan_amnt"), input_rows=44, question="age と bmi の相関は?")
    correct = _record("0.12", columns=("age", "bmi"), input_rows=44, question="age と bmi の相関は?")

    seen = {}

    def rederive(q):
        seen["q"] = q            # the re-executor is driven by the QUESTION, blind to the committed answer
        return correct

    v = verify_execution(committed, rederive=rederive)
    assert isinstance(v, ExecVerdict)
    assert seen["q"] == "age と bmi の相関は?"
    assert v.category == EXEC_CONFLICT and v.should_abstain
    assert v.committed_answer == "0.61" and v.rederived_answer == "0.12"
    d = v.to_dict()
    assert d["category"] == EXEC_CONFLICT and d["match"] is False and d["conflicts"]


# --------------------------------------------------------------------------- live glue (offline)
def test_rederive_calc_reexecutes_over_real_compute(tmp_path):
    """``rederive_calc`` runs an independent investigation and returns its calc record (no Vertex)."""
    csv = tmp_path / "t.csv"
    csv.write_text("loan_amnt,age\n1000,30\n2000,40\n3000,50\n", encoding="utf-8")
    model = ScriptedModel([_call("compute", file=str(csv), expr="int(df['loan_amnt'].sum())"),
                           _submit("6,000円", confidence=0.95)])
    rec = rederive_calc("loan_amntの合計は?", model=model, max_turns=5)
    validate(rec)
    assert rec["answer"] == "6,000円" and rec["grounded"] is True
    assert rec["compute_steps"][0]["columns_used"]           # 対象列 captured for structural照合
    assert rec["compute_steps"][0]["input_rows"] == 3


def test_rederive_calc_abstain_yields_shell_record():
    rec = rederive_calc("q", model=ScriptedModel([_submit(ABSTAIN, confidence=0.0)]), max_turns=3)
    assert rec["answer"] == ABSTAIN and rec["compute_steps"] == []
    # feeding it back to the verifier makes the outcome 決着不能
    cmp = compare_execution(_record("42", columns=("x",), input_rows=3), rec)
    assert cmp.category == EXEC_UNDECIDABLE


def test_end_to_end_offline_conflict_detected(tmp_path):
    """A committed wrong-column-pair answer vs a real independent re-execution → EVIDENCE_CONFLICT."""
    csv = tmp_path / "t.csv"
    csv.write_text("age,bmi,loan_amnt\n30,22,1000\n40,24,2000\n50,26,3000\n", encoding="utf-8")
    committed = _record("2000", columns=("age", "loan_amnt"), input_rows=3, question="bmi の平均は?")

    def rederive(q):
        model = ScriptedModel([_call("compute", file=str(csv), expr="round(df['bmi'].mean(), 1)"),
                               _submit("24.0", confidence=0.9)])
        return rederive_calc(q, model=model, max_turns=5)

    v = verify_execution(committed, rederive=rederive)
    assert v.category == EXEC_CONFLICT and v.should_abstain


# --------------------------------------------------------------------------- numeric routing
def test_is_numeric_record():
    assert is_numeric_record(_record("958", columns=("x",), input_rows=3)) is True
    assert is_numeric_record(_record("bmi", grounded=False)) is False   # column-name answer
    assert is_numeric_record(None) is False
