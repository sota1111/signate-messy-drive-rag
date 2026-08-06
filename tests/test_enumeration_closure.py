"""SOT-2500 — offline tests for the full-enumeration closure protocol.

All network-free: population-kind / ordering / closure classification is deterministic, the authoritative
resolver reads an in-memory :class:`CorpusProfile`, and the investigator wiring is driven by a scripted
fake model (no Vertex). Covers the resolver, the closure gate, the ordering rules, the injected procedure,
and the one-shot :class:`EnumerationGate` interception in :func:`investigate`.
"""
from __future__ import annotations

from src.rag.agent import enumeration as en
from src.rag.agent import question_contract as qc
from src.rag.agent import obligations as ob
from src.rag.agent.abstain_ledger import EVIDENCE_INCOMPLETE
from src.rag.tools.profile import CorpusProfile
from src.rag.agent.investigator import (
    ABSTAIN,
    SUBMIT_ANSWER,
    Call,
    Step,
    Usage,
    investigate,
    is_abstain,
)

Q_ABBREV = "中間報告会または中間レビューが2025年7月1日以前に実施された案件を、主略称ですべて挙げてください。"
Q_TASK = "AYMのPLにおいて、探索的分析・仮説整理フェーズに一致するタスクIDをすべて挙げてください。"
Q_SEAT = "3列目に座っているメンバーの氏名をすべて挙げてください。"
Q_NONENUM = "青潮モビリティサービスの担当者は誰ですか？"


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


def _submit(answer, *, confidence=0.9, evidence="", method="") -> Step:
    return Step(function_calls=(Call(SUBMIT_ANSWER, {
        "answer": answer, "confidence": confidence, "evidence": evidence, "method": method}),),
        usage=Usage(5, 5))


# --------------------------------------------------------------------------- taxonomy invariants
def test_taxonomies_are_well_formed():
    assert set(en.POPULATION_LABELS) == set(en.POPULATION_KINDS)
    assert set(en.ORDERING_LABELS) == set(en.ORDERING_RULES)
    assert set(en.CLOSURE_LABELS) == set(en.CLOSURE_CONDITIONS)
    assert len(en.CLOSURE_CONDITIONS) == 4
    # the module aligns an incomplete enumeration with the abstain ledger's state code
    assert en.EVIDENCE_INCOMPLETE == EVIDENCE_INCOMPLETE


# --------------------------------------------------------------------------- population-kind detection
def test_population_kind_deterministic_cues():
    assert en.detect_population_kind(Q_ABBREV) == en.ABBREV
    assert en.detect_population_kind(Q_TASK) == en.TASK
    assert en.detect_population_kind(Q_SEAT) == en.SEAT
    assert en.detect_population_kind("参加者を全員挙げてください。") == en.PERSON
    assert en.detect_population_kind("対象フォルダのファイルをすべて挙げてください。") == en.DOCUMENT
    assert en.detect_population_kind("取引先の会社をすべて挙げてください。") == en.PROJECT
    assert en.detect_population_kind("該当するものをすべて挙げてください。") == en.GENERIC


def test_population_kind_flash_only_on_ambiguity():
    calls = []

    def flash(q):
        calls.append(q)
        return en.PERSON

    # a confident cue never consults flash
    assert en.detect_population_kind(Q_ABBREV, flash=flash) == en.ABBREV
    assert calls == []
    # an ambiguous question falls back to flash
    assert en.detect_population_kind("該当するものをすべて挙げてください。", flash=flash) == en.PERSON
    assert len(calls) == 1
    # off-vocabulary flash output is ignored → GENERIC
    assert en.detect_population_kind("該当を挙げよ", flash=lambda q: "banana") == en.GENERIC


# --------------------------------------------------------------------------- ordering rules
def test_ordering_rule_selection():
    assert en.detect_ordering_rule(Q_TASK) == en.ORDER_ID_ASC
    assert en.detect_ordering_rule(Q_SEAT) == en.ORDER_SEATING
    assert en.detect_ordering_rule("ID昇順で番号をすべて挙げよ", en.GENERIC) == en.ORDER_ID_ASC
    assert en.detect_ordering_rule("マーカーの単語をすべて挙げよ", en.GENERIC) == en.ORDER_APPEARANCE


def test_apply_ordering_numeric_and_dedup():
    assert en.apply_ordering(["T12", "T09", "T11", "T10", "T09"], en.ORDER_ID_ASC) == \
        ["T09", "T10", "T11", "T12"]
    assert en.apply_ordering(["train_0722", "train_0077", "train_0242"], en.ORDER_ID_ASC) == \
        ["train_0077", "train_0242", "train_0722"]
    # appearance/seating preserve the caller's order (still de-duplicated, blanks dropped)
    assert en.apply_ordering(["b", "a", "b", "", "c"], en.ORDER_APPEARANCE) == ["b", "a", "c"]
    assert en.apply_ordering(["s3", "s1", "s2"], en.ORDER_SEATING) == ["s3", "s1", "s2"]


# --------------------------------------------------------------------------- authoritative population resolver
def test_resolve_population_from_profile_master():
    prof = CorpusProfile(aliases={"青葉バイオメディカル機器": ["青葉BM"], "AYM社": ["AYM"]})
    res = en.resolve_population(Q_ABBREV, prof)
    assert res.resolved is True
    assert res.kind == en.ABBREV
    assert res.source == "corpus_profile.aliases"
    assert res.members == ("AYM社", "青葉バイオメディカル機器")  # sorted canonical master
    assert len(res) == 2


def test_resolve_population_unresolved_without_master():
    # empty profile → unresolved, names the source category to read (never a guessed list)
    res = en.resolve_population(Q_ABBREV, CorpusProfile())
    assert res.resolved is False
    assert res.members == ()
    assert "用語集" in res.source
    # a person population has no persisted master → always unresolved, even with a populated profile
    res_person = en.resolve_population(Q_SEAT, CorpusProfile(aliases={"A社": ["A"]}))
    assert res_person.resolved is False


# --------------------------------------------------------------------------- closure gate
def test_closure_gate_satisfied_when_all_conditions_hold():
    res = en.evaluate_closure(en.ClosureState(True, True, True, True))
    assert res.satisfied is True
    assert res.unmet == ()
    assert res.obligations == ()
    assert res.state_code == ""


def test_closure_gate_emits_incomplete_obligations_for_gaps():
    res = en.evaluate_closure(en.ClosureState(population_defined=True, candidate_reasons=True))
    assert res.satisfied is False
    assert res.unmet == (en.C_ALT_PATH_ZERO, en.C_COUNT_MATCH)
    assert res.state_code == EVIDENCE_INCOMPLETE
    assert len(res.obligations) == 2
    assert all(o.kind == ob.ENUMERATION and o.status == ob.UNMET for o in res.obligations)


def test_closure_gate_accepts_dict_and_empty_state():
    res = en.evaluate_closure({"population_defined": True, "candidate_reasons": True,
                               "alt_path_zero_new": True, "count_match": True})
    assert res.satisfied is True
    empty = en.evaluate_closure(None)
    assert empty.satisfied is False
    assert empty.unmet == en.CLOSURE_CONDITIONS  # nothing proven yet → all four unmet


# --------------------------------------------------------------------------- injected procedure
def test_closure_procedure_is_fact_free_and_complete():
    proc = en.closure_procedure(Q_ABBREV)
    for label in en.CLOSURE_LABELS.values():
        assert label in proc              # all four closure conditions are stated
    assert "EVIDENCE_INCOMPLETE" in proc  # incomplete → coded abstain, not a guess
    assert "用語集" in proc               # names the authoritative source *category* for an abbrev pop
    # fact-free: the procedure never embeds a resolved member (no corpus answer injection)
    prof = CorpusProfile(aliases={"青葉バイオメディカル機器": ["青葉BM"]})
    assert "青葉バイオメディカル機器" not in en.closure_procedure(Q_ABBREV)
    assert en.resolve_population(Q_ABBREV, prof).members  # (the master IS resolvable, just not injected)


def test_procedure_reflects_task_id_ordering():
    proc = en.closure_procedure(Q_TASK)
    assert en.ORDERING_LABELS[en.ORDER_ID_ASC] in proc


# --------------------------------------------------------------------------- EnumerationGate
def test_gate_injects_procedure_once_for_enumeration():
    gate = en.EnumerationGate(Q_ABBREV)
    assert gate.is_enumeration() is True
    first = gate.review()
    assert first is not None and "完全列挙" in first
    assert gate.review() is None           # one-shot: a second abstain is not intercepted again
    assert gate.injected is True
    assert gate.terminal == en.INJECTED


def test_gate_is_inert_for_non_enumeration():
    gate = en.EnumerationGate(Q_NONENUM)
    assert gate.is_enumeration() is False
    assert gate.review() is None
    assert gate.terminal == en.NOT_ENUMERATION


def test_gate_honours_explicit_contract_override():
    gate = en.EnumerationGate("何でもよい", contract=qc.FULL_ENUMERATION)
    assert gate.is_enumeration() is True
    assert gate.review() is not None


# --------------------------------------------------------------------------- investigator wiring
def test_investigate_enumeration_intercepts_abstain_then_answers():
    # the model would abstain first, then (after the closure procedure) commits a complete enumeration
    model = ScriptedModel([_submit(ABSTAIN), _submit("MINAMINO、SHR、AYM")])
    res = investigate(model, Q_ABBREV, [], max_turns=6, enumeration=True)
    assert res.answer.answer == "MINAMINO、SHR、AYM"
    assert not is_abstain(res.answer.answer)
    # the closure procedure was fed back on the intercepted turn
    injected = [tr for turn in model.calls_seen if turn for tr in turn
                if tr.name == SUBMIT_ANSWER and isinstance(tr.response, dict)
                and tr.response.get("abstain_rejected")]
    assert injected and "権威的母集団" in injected[0].response["directive"]


def test_investigate_enumeration_off_is_byte_identical_abstain():
    model = ScriptedModel([_submit(ABSTAIN), _submit("late answer")])
    res = investigate(model, Q_ABBREV, [], max_turns=6)  # enumeration defaults off
    assert is_abstain(res.answer.answer)                 # abstain accepted immediately, no retry
    assert model._i == 1


def test_investigate_enumeration_inert_for_non_enumeration_question():
    model = ScriptedModel([_submit(ABSTAIN), _submit("late answer")])
    res = investigate(model, Q_NONENUM, [], max_turns=6, enumeration=True)
    assert is_abstain(res.answer.answer)  # not an enum contract → gate never fires
    assert model._i == 1


def test_investigate_enumeration_does_not_touch_committed_answer():
    model = ScriptedModel([_submit("MINAMINO、SHR")])
    res = investigate(model, Q_ABBREV, [], max_turns=6, enumeration=True)
    assert res.answer.answer == "MINAMINO、SHR"  # a committed enumeration finalizes as-is
