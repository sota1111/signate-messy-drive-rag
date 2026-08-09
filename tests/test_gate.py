"""SOT-2473 — offline tests for the precision-first confidence gate on the verifier consensus.

Network-free: :class:`~src.rag.agent.tiebreak.Resolution` objects are produced from *scripted* fake
models (same shape as :mod:`tests.test_tiebreak`, no Vertex) and fed to :func:`src.rag.agent.gate.apply_gate`.
Covers the acceptance criterion — **低確信/不一致は棄権** — and the Incorrect-minimisation property that
motivates the gate (Incorrect=−1 / Missing=0): a low-confidence agreement that would be *wrong* is
converted to an abstention, never committed.
"""
from __future__ import annotations

from config import settings
from src.rag.agent.investigator import (
    ABSTAIN,
    SUBMIT_ANSWER,
    Answer,
    Call,
    Investigation,
    Step,
    Usage,
)
from src.rag.agent import gate
from src.rag.agent.gate import GateDecision, apply_gate
from src.rag.agent.tiebreak import (
    STATUS_AGREED,
    STATUS_ABSTAINED,
    STATUS_TIEBREAK_INVESTIGATOR,
    Resolution,
    resolve_answer,
)


# --------------------------------------------------------------------------- scripted resolution build
class _ScriptedModel:
    """Fake :class:`~src.rag.agent.investigator.Model` replaying one ``submit_answer`` step."""

    def __init__(self, answer, *, confidence=0.9):
        self._answer, self._confidence = answer, confidence
        self.model_name = "fake"
        self.calls = 0

    def next(self, tool_responses):
        self.calls += 1
        return Step(function_calls=(Call(SUBMIT_ANSWER, {
            "answer": self._answer, "confidence": self._confidence,
            "evidence": "e", "method": "m"}),), usage=Usage(50, 10))


def _investigation(answer, *, confidence=0.9) -> Investigation:
    return Investigation(
        question="Q?", answer=Answer(answer=answer, confidence=confidence, evidence="ie", method="im"),
        iterations=1, tool_calls=[], usage=Usage(100, 20), model="gemini-2.5-pro",
        elapsed_s=0.1, stop_reason="answered",
    )


def _resolution(inv_answer, ver_answer, judge_answer=None, *, confidence=0.9) -> Resolution:
    """A realistic :class:`Resolution` from the scripted 合議 (agree / tie-break / abstain per inputs)."""
    return resolve_answer(
        "Q?", _investigation(inv_answer, confidence=confidence),
        verifier_model=_ScriptedModel(ver_answer),
        judge_model=(_ScriptedModel(judge_answer) if judge_answer is not None else None),
        max_turns=3,
    )


# --------------------------------------------------------------------------- 合議一致 → 高確信 commit
def test_agreement_high_confidence_commits():
    r = _resolution("1526", "1526.0", confidence=0.9)
    assert r.status == STATUS_AGREED
    d = apply_gate(r, commit_confidence=0.7)
    assert isinstance(d, GateDecision)
    assert d.commit and d.answer == "1526" and d.confidence == 0.9
    assert not d.abstained
    assert "commit" in d.reason


def test_agreement_confidence_exactly_at_threshold_commits():
    d = apply_gate(_resolution("A", "A", confidence=0.7), commit_confidence=0.7)
    assert d.commit and d.answer == "A"  # boundary is inclusive (≥)


# --------------------------------------------------------------------------- 合議一致 → 低確信 棄権
def test_agreement_low_confidence_abstains():
    r = _resolution("1526", "1526", confidence=0.4)
    d = apply_gate(r, commit_confidence=0.7)
    assert not d.commit and d.answer == settings.ABSTAIN and d.confidence == 0.0
    assert d.abstained
    assert "低確信" in d.reason


# --------------------------------------------------------------------------- 合議棄権 → 棄権を維持
def test_consensus_abstention_stays_abstained():
    # investigator/verifier disagree and the judge lands on a third value → 決着不能 → 棄権.
    r = _resolution("1530", "1526", judge_answer="1600")
    assert r.status == STATUS_ABSTAINED and r.abstained
    d = apply_gate(r)
    assert not d.commit and d.answer == settings.ABSTAIN
    assert "合議が棄権" in d.reason


# --------------------------------------------------------------------------- tie-break → 既定棄権
def test_tiebreak_abstains_by_default_even_when_confident():
    # investigator was wrong-but-confident; judge sided with investigator → tie-broken commit candidate.
    r = _resolution("1530", "1526", judge_answer="1530", confidence=0.99)
    assert r.status == STATUS_TIEBREAK_INVESTIGATOR and not r.abstained
    d = apply_gate(r)  # commit_on_tiebreak defaults OFF (precision-first)
    assert not d.commit and d.answer == settings.ABSTAIN
    assert "不一致" in d.reason and "棄権" in d.reason


def test_tiebreak_opt_in_commits_above_stricter_bar():
    r = _resolution("1530", "1526", judge_answer="1530", confidence=0.9)
    d = apply_gate(r, commit_on_tiebreak=True, tiebreak_confidence=0.85)
    assert d.commit and d.answer == "1530" and d.confidence == 0.9


def test_tiebreak_opt_in_still_abstains_below_stricter_bar():
    r = _resolution("1530", "1526", judge_answer="1530", confidence=0.8)
    d = apply_gate(r, commit_on_tiebreak=True, tiebreak_confidence=0.85)
    assert not d.commit and d.answer == settings.ABSTAIN
    assert "低確信" in d.reason


# --------------------------------------------------------------------------- Incorrect minimisation
def test_gate_minimises_incorrect_by_abstaining_low_confidence_wrong_agreements():
    """受け入れ条件 + 検証内容: Incorrect(-1) を最小化する。

    A slate where both AGs *agree* but on a WRONG value at low confidence (a correlated near-miss) — the
    consensus alone would commit it and score −1. The precision-first gate turns each such low-confidence
    agreement into Missing(0), strictly improving the official score, while leaving high-confidence
    correct agreements committed.
    """
    def score(served: str, truth: str) -> int:
        if served == settings.ABSTAIN:
            return 0                       # Missing
        return 1 if served == truth else -1  # Correct / Incorrect

    slate = [
        # (inv, ver, confidence, ground truth)
        ("0.8999", "0.8999", 0.4, "0.9000"),  # low-conf agreement on a near-miss WRONG value
        ("8", "8", 0.5, "9"),                 # low-conf agreement, wrong count
        ("東京", "東京", 0.95, "東京"),          # high-conf agreement, correct → must stay committed
    ]
    consensus_total = 0
    gated_total = 0
    for inv, ver, conf, truth in slate:
        r = _resolution(inv, ver, confidence=conf)
        assert r.status == STATUS_AGREED
        consensus_total += score(r.answer, truth)             # raw consensus commits everything
        gated_total += score(apply_gate(r, commit_confidence=0.7).answer, truth)

    # Raw consensus: -1 -1 +1 = -1 ; gated: 0 0 +1 = +1. Gate minimises Incorrect and lifts the score.
    assert consensus_total == -1
    assert gated_total == 1
    assert gated_total > consensus_total


# --------------------------------------------------------------------------- schema / helpers
def test_gate_decision_to_dict_and_to_answer_commit():
    d = apply_gate(_resolution("42", "42", confidence=0.9), commit_confidence=0.7)
    dd = d.to_dict()
    assert dd["answer"] == "42" and dd["commit"] is True and dd["gate_status"] == "commit"
    assert dd["confidence"] == 0.9 and dd["status"] == STATUS_AGREED and dd["agree"] is True
    assert dd["resolution"]["answer"] == "42"
    a = d.to_answer()
    assert a.answer == "42" and a.confidence == 0.9 and a.evidence == "ie"


def test_gate_decision_to_dict_and_to_answer_abstain():
    d = apply_gate(_resolution("42", "42", confidence=0.1), commit_confidence=0.7)
    dd = d.to_dict()
    assert dd["answer"] == settings.ABSTAIN and dd["commit"] is False
    assert dd["gate_status"] == "abstain" and dd["confidence"] == 0.0
    assert dd["evidence"] == ""            # abstention carries no evidence/method
    a = d.to_answer()
    assert a.answer == settings.ABSTAIN and a.confidence == 0.0


# --------------------------------------------------------------------------- live wiring / thresholds
def test_gate_question_wires_resolve_then_gates(monkeypatch):
    """``gate_question`` runs the consensus then gates it (network-free via monkeypatched resolve)."""
    r = _resolution("1526", "1526", confidence=0.9)
    monkeypatch.setattr(gate, "resolve_question", lambda q, **kw: r)
    d = gate.gate_question("Q?", commit_confidence=0.7)
    assert d.commit and d.answer == "1526"


def test_gate_question_abstains_low_confidence(monkeypatch):
    r = _resolution("1526", "1526", confidence=0.3)
    monkeypatch.setattr(gate, "resolve_question", lambda q, **kw: r)
    d = gate.gate_question("Q?", commit_confidence=0.7)
    assert not d.commit and d.answer == settings.ABSTAIN


def test_default_thresholds_are_precision_first():
    # Defaults: commit bar >0.5 EV break-even, tie-break commit OFF (disagreement abstains).
    assert gate.GATE_COMMIT_CONFIDENCE > 0.5
    assert gate.GATE_COMMIT_ON_TIEBREAK is False
    assert gate.GATE_TIEBREAK_CONFIDENCE >= gate.GATE_COMMIT_CONFIDENCE


# --------------------------------------------------------------------------- SOT-2501 execution gate
from src.rag.agent.exec_verifier import (  # noqa: E402
    EXEC_CONFLICT,
    EXEC_MATCH,
    ExecVerdict,
)


def _exec_verdict(category, *, should_abstain, answer="0.61"):
    return ExecVerdict(
        question="Q?", category=category, match=(category == EXEC_MATCH),
        should_abstain=should_abstain, reason=f"{category} reason",
        committed_answer=answer, rederived_answer="0.12",
        conflicts=(("対象列の相違",) if should_abstain else ()),
    )


def test_exec_gate_default_scope_leaves_non_calculation_contract_untouched():
    """Default ON is scoped: a non-calculation question ignores an unrelated execution verdict."""
    r = _resolution("0.61", "0.61", confidence=0.9)
    conflict = _exec_verdict(EXEC_CONFLICT, should_abstain=True)
    d = apply_gate(r, commit_confidence=0.7, exec_verdict=conflict)
    assert d.commit and d.answer == "0.61"
    assert gate.GATE_EXEC_VERIFY is True
    assert gate.GATE_EXEC_VERIFY_CONTRACTS == frozenset({"numeric"})


def test_exec_gate_downgrades_conflicting_numeric_commit_when_enabled():
    """idx4/28/10 相当: 独立再実行が不一致(EVIDENCE_CONFLICT)なら数値 commit を棄権へ倒す。"""
    r = _resolution("0.61", "0.61", confidence=0.9)
    conflict = _exec_verdict(EXEC_CONFLICT, should_abstain=True)
    d = apply_gate(r, commit_confidence=0.7, exec_verdict=conflict, exec_verify=True)
    assert not d.commit and d.answer == settings.ABSTAIN
    assert "実行検証" in d.reason and EXEC_CONFLICT in d.reason


def test_exec_gate_keeps_confirmed_numeric_commit():
    r = _resolution("0.42", "0.42", confidence=0.9)
    ok = _exec_verdict(EXEC_MATCH, should_abstain=False, answer="0.42")
    d = apply_gate(r, commit_confidence=0.7, exec_verdict=ok, exec_verify=True)
    assert d.commit and d.answer == "0.42"


def test_exec_gate_leaves_non_numeric_commit_to_the_text_verifier():
    """数値系のみ本検証器が主判定; 非数値(列名等)は heterogeneous verifier のまま = 実行検証は無視。"""
    r = _resolution("bmi", "bmi", confidence=0.9)
    conflict = _exec_verdict(EXEC_CONFLICT, should_abstain=True, answer="bmi")
    d = apply_gate(r, commit_confidence=0.7, exec_verdict=conflict, exec_verify=True)
    assert d.commit and d.answer == "bmi"


def test_exec_gate_applies_to_computed_label_in_numeric_contract():
    r = _resolution_q("目的変数と相関が最も高い特徴量は?", "age", confidence=0.9)
    conflict = _exec_verdict(EXEC_CONFLICT, should_abstain=True, answer="age")
    d = apply_gate(
        r, commit_confidence=0.7, exec_verdict=conflict, contract="numeric")
    assert not d.commit and d.answer == settings.ABSTAIN
    assert "計算回答の実行検証" in d.reason


def test_exec_gate_does_not_resurrect_an_abstention():
    # a low-confidence agreement already abstains; a MATCH verdict must not turn it into a commit.
    r = _resolution("0.42", "0.42", confidence=0.3)
    ok = _exec_verdict(EXEC_MATCH, should_abstain=False, answer="0.42")
    d = apply_gate(r, commit_confidence=0.7, exec_verdict=ok, exec_verify=True)
    assert not d.commit and d.answer == settings.ABSTAIN


# --------------------------------------------------------------------------- SOT-2547: 大外し矯正 gate
from src.rag.agent.exec_verifier import EXEC_CORRECTED  # noqa: E402


def _corrected_verdict(*, committed, corrected):
    return ExecVerdict(
        question="Q?", category=EXEC_CORRECTED, match=True, should_abstain=False,
        reason=f"{EXEC_CORRECTED} 桁 reason", committed_answer=committed,
        rederived_answer=corrected, corrected_answer=corrected, conflicts=("値の相違",))


def test_exec_gate_corrects_gross_miss_when_correction_enabled():
    """idx63/97 相当: EXEC_CORRECTED verdict + 矯正ON → 台帳の大外しを再計算値へ置換(commit維持)。"""
    r = _resolution("-30.78416", "-30.78416", confidence=0.9)
    v = _corrected_verdict(committed="-30.78416", corrected="0.15002")
    d = apply_gate(r, commit_confidence=0.7, exec_verdict=v, exec_verify=True, exec_correct=True)
    assert d.commit and d.answer == "0.15002"
    assert "矯正" in d.reason and EXEC_CORRECTED in d.reason


def test_exec_gate_correction_disabled_abstains_on_gross_miss():
    """矯正OFF(既定)で EXEC_CORRECTED を受け取っても大外しを commit せず安全側で棄権。"""
    r = _resolution("18948", "18948", confidence=0.9)
    v = _corrected_verdict(committed="18948", corrected="272")
    d = apply_gate(r, commit_confidence=0.7, exec_verdict=v, exec_verify=True, exec_correct=False)
    assert not d.commit and d.answer == settings.ABSTAIN
    assert gate.GATE_EXEC_CORRECT is False   # default OFF → byte-identical answer path


def test_exec_gate_correction_scoped_to_numeric_like_answer():
    """非数値回答は矯正対象外(heterogeneous verifier のまま)= EXEC_CORRECTED でも original を維持。"""
    r = _resolution("bmi", "bmi", confidence=0.9)
    v = _corrected_verdict(committed="bmi", corrected="age")
    d = apply_gate(r, commit_confidence=0.7, exec_verdict=v, exec_verify=True, exec_correct=True)
    assert d.commit and d.answer == "bmi"


def test_gate_question_runs_live_exec_verification(monkeypatch):
    """``gate_question`` wires the committed numeric record through the execution verifier when enabled."""
    r = _resolution("0.61", "0.61", confidence=0.9)
    monkeypatch.setattr(gate, "resolve_question", lambda q, **kw: r)
    from src.rag.agent import exec_verifier
    monkeypatch.setattr(exec_verifier, "verify_question",
                        lambda rec, **kw: _exec_verdict(EXEC_CONFLICT, should_abstain=True))
    record = {"question": "Q?", "answer": "0.61", "typed_value": {},
              "compute_steps": [{"columns_used": ["age", "loan_amnt"], "input_rows": 44}]}
    d = gate.gate_question("Q?", commit_confidence=0.7, committed_calc_record=record, exec_verify=True)
    assert not d.commit and d.answer == settings.ABSTAIN


def test_gate_question_uses_in_memory_record_for_derived_label(monkeypatch):
    """No shared-ledger lookup: the investigator's exact record flows with the resolution."""
    from dataclasses import replace
    from src.rag.agent import exec_verifier

    record = {
        "question": "目的変数と相関が最も高い特徴量は?",
        "answer": "age",
        "contract": "numeric",
        "typed_value": {"raw_text": "age", "unit": None},
        "compute_steps": [{"columns_used": ["charges", "age"], "input_rows": 1600}],
    }
    r = replace(
        _resolution_q(record["question"], "age", confidence=0.9), calc_record=record)
    monkeypatch.setattr(gate, "resolve_question", lambda q, **kw: r)
    monkeypatch.setattr(
        exec_verifier, "verify_question",
        lambda rec, **kw: _exec_verdict(EXEC_CONFLICT, should_abstain=True, answer="age"))
    d = gate.gate_question(record["question"], commit_confidence=0.7)
    assert not d.commit and d.answer == settings.ABSTAIN


def test_run_gated_backend_routes_to_the_gate(monkeypatch):
    """``run.make_worker('gated')`` serves the gate decision through the run-log schema."""
    from src.rag import run

    d = apply_gate(_resolution("1526", "1526", confidence=0.9), commit_confidence=0.7)
    monkeypatch.setattr(gate, "gate_question", lambda q: d)
    res = run.make_worker("gated", False)(1, "q")[1]
    assert res["answer"] == "1526" and res["commit"] is True
    assert res["used_images"] == 0  # agent backends get the uniform run-log key
    assert "gated" in run.GEN_CHOICES


# --------------------------------------------------------------------------- SOT-2503 per-slice gate
def _resolution_q(question, answer, *, confidence=0.9) -> Resolution:
    """A 合議一致 :class:`Resolution` on ``answer`` for an explicit ``question`` (for contract routing)."""
    return resolve_answer(
        question, _investigation(answer, confidence=confidence),
        verifier_model=_ScriptedModel(answer), max_turns=3,
    )


def test_slice_calibration_default_off_is_byte_identical():
    """既定OFF: even with a relaxed slice threshold present, a mid-confidence agreement still abstains."""
    r = _resolution("A", "A", confidence=0.6)
    d = apply_gate(r, commit_confidence=0.7,
                   slice_thresholds={"simple_lookup": 0.5}, contract="simple_lookup")
    assert not d.commit and d.answer == settings.ABSTAIN  # slice_calibrate defaults to GATE_SLICE_CALIBRATE (OFF)
    assert gate.GATE_SLICE_CALIBRATE is False


def test_slice_calibration_relaxes_commit_for_adopted_slice():
    """緩和スライスでは lower な閾値で commit される（グローバルでは棄権のはずの中確信一致）。"""
    r = _resolution("A", "A", confidence=0.6)
    d = apply_gate(r, commit_confidence=0.7, slice_calibrate=True,
                   slice_thresholds={"simple_lookup": 0.5}, contract="simple_lookup")
    assert d.commit and d.answer == "A"
    assert "0.50" in d.reason  # committed against the relaxed slice bar


def test_slice_calibration_leaves_unlisted_slice_at_global_bar():
    """未採用スライスはグローバル閾値のまま（棄権）。"""
    r = _resolution("A", "A", confidence=0.6)
    d = apply_gate(r, commit_confidence=0.7, slice_calibrate=True,
                   slice_thresholds={"chart_read": 0.5}, contract="simple_lookup")
    assert not d.commit and d.answer == settings.ABSTAIN


def test_slice_calibration_is_one_directional_never_tightens():
    """スライス閾値がグローバルより高くても引き締めない（min で緩和方向のみ）。"""
    r = _resolution("A", "A", confidence=0.75)
    d = apply_gate(r, commit_confidence=0.7, slice_calibrate=True,
                   slice_thresholds={"simple_lookup": 0.9}, contract="simple_lookup")
    assert d.commit and d.answer == "A"  # 0.75 ≥ min(0.7, 0.9)=0.7 → still commits


def test_slice_calibration_classifies_contract_from_question_when_unset():
    """contract 未指定なら質問文から決定論分類（座席=spatial）してスライス閾値を引く。"""
    r = _resolution_q("山田さんの隣に座っているのは誰ですか", "田中", confidence=0.6)
    d = apply_gate(r, commit_confidence=0.7, slice_calibrate=True,
                   slice_thresholds={"spatial": 0.5})  # contract=None → classified as spatial
    assert d.commit and d.answer == "田中"


def test_load_slice_thresholds_reads_full_model(tmp_path):
    from scoring.slice_calibration import save_model
    model = {"schema_version": 1, "adopted_thresholds": {"simple_lookup": 0.55, "numeric": 0.6}}
    p = tmp_path / "slice.json"
    save_model(model, p)
    assert gate.load_slice_thresholds(p) == {"simple_lookup": 0.55, "numeric": 0.6}


def test_load_slice_thresholds_accepts_bare_map_and_drops_out_of_range(tmp_path):
    import json
    p = tmp_path / "bare.json"
    p.write_text(json.dumps({"spatial": 0.5, "numeric": 1.5, "chart_read": "x"}), encoding="utf-8")
    assert gate.load_slice_thresholds(p) == {"spatial": 0.5}  # 1.5 out of range, "x" unparseable


def test_load_slice_thresholds_missing_or_unset_is_empty(tmp_path):
    assert gate.load_slice_thresholds("") == {}
    assert gate.load_slice_thresholds(tmp_path / "nope.json") == {}


def test_gate_question_loads_slice_file_when_enabled(monkeypatch, tmp_path):
    """``gate_question`` reads GATE_SLICE_CALIBRATION_FILE and relaxes the adopted slice."""
    import json
    from src.rag.agent import question_contract as qc
    question = "山田さんの隣に座っているのは誰ですか"     # deterministically classifies as spatial
    contract = qc.classify(question).contract
    p = tmp_path / "slice.json"
    p.write_text(json.dumps({"adopted_thresholds": {contract: 0.5}}), encoding="utf-8")
    monkeypatch.setattr(gate, "GATE_SLICE_CALIBRATION_FILE", str(p))
    r = _resolution_q(question, "田中", confidence=0.6)
    monkeypatch.setattr(gate, "resolve_question", lambda q, **kw: r)
    d = gate.gate_question(question, commit_confidence=0.7, slice_calibrate=True)
    assert d.commit and d.answer == "田中"
