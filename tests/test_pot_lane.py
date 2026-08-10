"""SOT-2586 — tests for the NUMERIC PoT forced lane (binder→制限AST→Decimal→独立検算→N-sample majority)."""
from __future__ import annotations

from fractions import Fraction

import pytest

from src.rag.agent import failure_taxonomy as tax
from src.rag.agent import pot_lane as p


# --------------------------------------------------------------------------- layer 1: Evidence Binder
def test_bind_operand_parses_yen_and_commas_exactly():
    op = p.bind_operand("gross", "1,100,000円", unit="円", source="契約.docx:S1!B3")
    assert op.value == Fraction(1_100_000)
    assert op.sourced and op.has_unit


def test_bind_operand_rejects_non_numeric_and_bool():
    with pytest.raises(p.FormulaError):
        p.bind_operand("x", "N/A")
    with pytest.raises(p.FormulaError):
        p.bind_operand("x", True)


def test_operand_sources_complete_requires_every_source():
    ok = p.build_binding([{"name": "a", "value": 1, "source": "s"},
                          {"name": "b", "value": 2, "source": "t"}])
    assert ok.operand_sources_complete
    missing = p.build_binding([{"name": "a", "value": 1, "source": "s"},
                               {"name": "b", "value": 2}])
    assert not missing.operand_sources_complete


def test_binding_rejects_duplicate_operand_names():
    with pytest.raises(p.FormulaError):
        p.build_binding([{"name": "a", "value": 1, "source": "s"},
                         {"name": "a", "value": 2, "source": "s"}])


def test_float_operand_parsed_as_human_reads_it():
    # 0.1 must bind to exactly 1/10 (str path), not the binary float expansion.
    op = p.bind_operand("r", 0.1, source="s")
    assert op.value == Fraction(1, 10)


# --------------------------------------------------------------------------- layer 3: restricted AST
def test_disallowed_operation_rejected():
    with pytest.raises(p.FormulaError):
        p.build_formula({"op": "POW", "args": [{"ref": "x"}, {"const": 2}]})


def test_code_string_is_not_a_valid_node():
    # A free-Python string can never be a formula node — no eval path exists.
    with pytest.raises(p.FormulaError):
        p.build_formula({"op": "MUL", "args": ["df['x'].sum()", {"const": 2}]})
    with pytest.raises(p.FormulaError):
        p.build_formula("__import__('os').system('x')")  # type: ignore[arg-type]


def test_arity_enforced():
    with pytest.raises(p.FormulaError):
        p.build_formula({"op": "ADD", "args": [{"ref": "x"}]})  # ADD needs 2


def test_all_allowed_ops_are_arity_mapped():
    for op in p.ALLOWED_OPS:
        assert op in p._ARITY


def test_formula_refs_collects_operand_names():
    node = p.build_formula({"op": "ADD", "args": [{"ref": "a"},
                                                  {"op": "MUL", "args": [{"ref": "b"}, {"const": 2}]}]})
    assert p.formula_refs(node) == {"a", "b"}


# --------------------------------------------------------------------------- layers 4/5: exec + verify
def _spec(formula, operands=None, condition=None, unit="円"):
    return {"operands": operands or [{"name": "x", "value": 100, "unit": "円", "source": "s"}],
            "formula": formula, "condition": condition, "result_unit": unit}


def test_exact_arithmetic_add_mul_round():
    spec = _spec({"op": "ROUND",
                  "args": [{"op": "MUL", "args": [{"ref": "h"}, {"ref": "rate"}]}], "ndigits": 0},
                 operands=[{"name": "h", "value": 174, "unit": "時間", "source": "c:S1!B2"},
                           {"name": "rate", "value": 25000, "unit": "円", "source": "c:S1!B3"}])
    res = p.evaluate_candidate(spec)
    assert res.all_layers_pass
    assert res.value == Fraction(4_350_000)


def test_division_stays_exact_and_verified():
    # 1,100,000 / 11 = 100,000 exactly (10% tax back-out).
    spec = _spec({"op": "DIV", "args": [{"ref": "gross"}, {"const": 11}]},
                 operands=[{"name": "gross", "value": 1_100_000, "unit": "円", "source": "r:S1!B4"}])
    res = p.evaluate_candidate(spec)
    assert res.execution_verdict.ok
    assert res.value == Fraction(100_000)


def test_percent_change_rounds_half_up():
    spec = _spec({"op": "ROUND",
                  "args": [{"op": "PERCENT_CHANGE", "args": [{"ref": "old"}, {"ref": "new"}]}],
                  "ndigits": 1},
                 operands=[{"name": "old", "value": 200, "source": "s"},
                           {"name": "new", "value": 255, "source": "s"}], unit="%")
    res = p.evaluate_candidate(spec)
    assert res.value == Fraction(275, 10)  # (255-200)/200*100 = 27.5


def test_round_half_up_boundary():
    assert p._round_fraction_half_up(Fraction(5, 2), 0) == Fraction(3)   # 2.5 -> 3
    assert p._round_fraction_half_up(Fraction(1, 2), 0) == Fraction(1)   # 0.5 -> 1


def test_round_half_up_matches_decimal_for_negatives():
    # ROUND_HALF_UP ties go away from zero — the Fraction verifier must match the Decimal executor,
    # else a negative .5 result (e.g. a PERCENT_CHANGE decrease) would falsely disagree.
    from decimal import ROUND_HALF_UP, Decimal

    for num, nd in [(Fraction(-5, 2), 0), (Fraction(-1, 2), 0), (Fraction(-275, 100), 1),
                    (Fraction(275, 100), 1)]:
        frac = p._round_fraction_half_up(num, nd)
        dec = Decimal(num.numerator) / Decimal(num.denominator)
        dec = dec.quantize(Decimal(1).scaleb(-nd), rounding=ROUND_HALF_UP)
        assert frac == Fraction(dec), (num, nd, frac, dec)


def test_negative_percent_change_execution_agrees():
    spec = _spec({"op": "ROUND",
                  "args": [{"op": "PERCENT_CHANGE", "args": [{"ref": "old"}, {"ref": "new"}]}],
                  "ndigits": 0},
                 operands=[{"name": "old", "value": 200, "source": "s"},
                           {"name": "new", "value": 195, "source": "s"}], unit="%")
    res = p.evaluate_candidate(spec)
    assert res.execution_verdict.ok  # (195-200)/200*100 = -2.5 -> -3, both engines agree
    assert res.value == Fraction(-3)


def test_weighted_sum_pairs():
    spec = _spec({"op": "WEIGHTED_SUM",
                  "args": [{"ref": "v1"}, {"const": 2}, {"ref": "v2"}, {"const": 3}]},
                 operands=[{"name": "v1", "value": 10, "source": "s"},
                           {"name": "v2", "value": 5, "source": "s"}])
    res = p.evaluate_candidate(spec)
    assert res.value == Fraction(35)  # 10*2 + 5*3


def test_division_by_zero_fails_execution_not_crash():
    spec = _spec({"op": "DIV", "args": [{"ref": "x"}, {"const": 0}]})
    res = p.evaluate_candidate(spec)
    assert not res.execution_verdict.ok
    assert res.value is None


# --------------------------------------------------------------------------- three-layer independence
def test_three_layers_judged_independently():
    # operand fails (no source) but the formula itself is a valid AST → formula layer still passes.
    spec = _spec({"op": "ADD", "args": [{"ref": "x"}, {"const": 1}]},
                 operands=[{"name": "x", "value": 100}])  # no source
    res = p.evaluate_candidate(spec)
    assert res.formula_verdict.ok
    assert not res.operand_verdict.ok
    assert res.operand_verdict.code == tax.EVIDENCE_INCOMPLETE
    assert not res.execution_verdict.ok  # gated: operand-incomplete → not executed


def test_formula_fail_is_execution_disagreement_code():
    spec = _spec({"op": "NOPE", "args": [{"ref": "x"}]})
    res = p.evaluate_candidate(spec)
    assert not res.formula_verdict.ok
    assert res.formula_verdict.code == tax.EXECUTION_DISAGREEMENT


def test_require_units_gates_operand_layer():
    spec = _spec({"ref": "x"}, operands=[{"name": "x", "value": 100, "source": "s"}])  # no unit
    assert p.evaluate_candidate(spec, require_units=False).operand_verdict.ok
    assert not p.evaluate_candidate(spec, require_units=True).operand_verdict.ok


def test_unbound_ref_is_operand_layer_failure():
    spec = _spec({"op": "ADD", "args": [{"ref": "x"}, {"ref": "missing"}]},
                 operands=[{"name": "x", "value": 100, "source": "s"}])
    res = p.evaluate_candidate(spec)
    assert not res.operand_verdict.ok
    assert "missing" in res.operand_verdict.note


# --------------------------------------------------------------------------- layer 2: condition / branch
def test_conditional_branch_must_reference_base_quantity():
    # idx76 型: the 減額 branch names ``gross`` as base but the formula multiplies ``net`` — inconsistent.
    spec = _spec({"op": "MUL", "args": [{"ref": "net"}, {"const": Fraction(2, 3)}]},
                 operands=[{"name": "net", "value": 79_200, "unit": "円", "source": "s"}],
                 condition={"predicate": "3分の2に減額", "predicate_truth": True, "base_quantity": "gross"})
    res = p.evaluate_candidate(spec)
    assert not res.formula_verdict.ok
    assert "base_quantity" in res.formula_verdict.note


def test_conditional_requires_resolved_truth():
    spec = _spec({"ref": "x"}, operands=[{"name": "x", "value": 100, "source": "s"}],
                 condition={"predicate": "税込か税抜か", "predicate_truth": None})
    res = p.evaluate_candidate(spec)
    assert not res.formula_verdict.ok


def test_branch_signature_differs_by_truth():
    a = p.build_condition({"predicate": "税込", "predicate_truth": True, "base_quantity": "g"})
    b = p.build_condition({"predicate": "税込", "predicate_truth": False, "base_quantity": "g"})
    assert a.branch_signature() != b.branch_signature()


# --------------------------------------------------------------------------- layer 6: N-sample majority
def _good(const=Fraction(2, 3), src="doc:S1!B3", base="gross"):
    return _spec({"op": "MUL", "args": [{"ref": "gross"}, {"const": const}]},
                 operands=[{"name": "gross", "value": 79_200, "unit": "円", "source": src}],
                 condition={"predicate": "3分の2", "predicate_truth": True, "base_quantity": base})


def test_simple_single_candidate_commits_on_all_layers_pass():
    r = p.run_forced_lane([_good()])
    assert r.decision.status == p.COMMIT
    assert r.decision.value == Fraction(52_800)
    assert r.decision.agreement == 1


def test_multi_majority_requires_full_signature_agreement():
    r = p.run_forced_lane([_good(), _good(), _good()], simple=False)
    assert r.decision.status == p.COMMIT
    assert r.decision.agreement == 3


def test_same_answer_different_source_does_not_agree():
    # All three reach 52,800 but from different operand sources → NOT an agreement of 3.
    r = p.run_forced_lane([_good(src="a"), _good(src="b"), _good(src="c")], simple=False)
    assert r.decision.status == p.NEED_MORE  # only pluralities of 1


def test_majority_needs_more_then_abstains_when_exhausted():
    specs = [_good(const=Fraction(1, k)) for k in (2, 3, 4)]
    assert p.run_forced_lane(specs, simple=False).decision.status == p.NEED_MORE
    five = [_good(const=Fraction(1, k)) for k in (2, 3, 4, 5, 6)]
    assert p.run_forced_lane(five, simple=False).decision.status == p.ABSTAIN


def test_all_candidates_fail_abstains_with_dominant_code():
    bad = _spec({"ref": "x"}, operands=[{"name": "x", "value": 1}])  # no source
    d = p.run_forced_lane([bad, bad]).decision
    assert d.status == p.ABSTAIN
    assert d.code == tax.EVIDENCE_INCOMPLETE


def test_agreement_signature_includes_all_four_axes():
    r = p.run_forced_lane([_good(), _good()], simple=False)
    sig = r.decision.to_dict()["signature"]
    assert set(sig) == {"answer", "operand_source", "unit", "branch"}


# --------------------------------------------------------------------------- serialization / rendering
def test_result_is_json_serializable():
    import json

    r = p.run_forced_lane([_good()])
    json.dumps(r.to_dict(), ensure_ascii=False)  # must not raise


def test_render_value_formats_integer_with_commas():
    assert p.render_value(Fraction(52_800), "円") == "52,800円"
    assert p.render_value(None) is None


def test_sympy_optional_backend_reports_presence():
    assert isinstance(p.have_sympy(), bool)
