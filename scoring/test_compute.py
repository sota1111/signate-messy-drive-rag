from __future__ import annotations

import pandas as pd

from config import settings
from scoring import deterministic, synth
from src.rag import archetype, compute, generate


def _valid(idx: int) -> tuple[str, str]:
    q = pd.read_csv(settings.QUESTIONS_VALID).set_index("index")
    gt = pd.read_csv(settings.VALID_GROUND_TRUTH, header=None, index_col=0)
    return str(q.at[idx, "question"]), str(gt.at[idx, 1])


def test_valid_cross_aggregates_are_computed_exactly():
    for idx in (3, 8, 13):
        question, truth = _valid(idx)
        assert deterministic.score(compute.answer_question(question), truth, "numeric") == "Perfect"
        assert archetype.classify(question) == "cross_aggregate"


def test_generate_routes_computation_without_llm(monkeypatch):
    # SOT-2424: the compute hard module direct-commits only for a hold-out-validated archetype.
    monkeypatch.setattr(generate, "_load_trust",
                        lambda: {"cross_aggregate": {"holdout_validated": True}})
    question, truth = _valid(13)
    monkeypatch.setattr(generate.llm, "generate", lambda *a, **k: (_ for _ in ()).throw(AssertionError()))
    result = generate.answer_question(question)
    assert result["answer"] == truth
    assert result["verified"] is True


def test_unresolved_compute_question_abstains(monkeypatch):
    monkeypatch.setattr(generate, "_load_trust",
                        lambda: {"cross_aggregate": {"holdout_validated": True}})
    result = generate.answer_question("存在しない案件のfoo列の平均値を算出してください。")
    assert result["answer"] == settings.ABSTAIN
    assert result["confidence"] == "compute-unresolved"


def test_synth_contains_cross_aggregate_items():
    items = []
    synth.gen_cross_aggregate(items)
    assert len(items) == 3
    assert all(compute.answer_question(i.question) == i.truth for i in items)


def test_unconditional_csv_column_stats_are_computed_exactly():
    items = []
    synth.gen_csv(items)
    assert items
    for item in items:
        assert deterministic.score(
            compute.column_stat_answer(item.question), item.truth, "numeric"
        ) == "Perfect"


def test_csv_column_stat_rejects_conditional_questions():
    items = []
    synth.gen_csv(items)
    question = items[0].question.replace("の平均値", "が10以上のときの平均値")
    assert compute.column_stat_answer(question) is None


def test_generate_routes_validated_csv_stat_without_llm(monkeypatch):
    items = []
    synth.gen_csv(items)
    item = next(i for i in items if i.archetype == "csv_column_mean")
    monkeypatch.setattr(
        generate,
        "_load_trust",
        lambda: {"csv_column_mean": {"trust": True, "holdout_validated": True}},
    )
    monkeypatch.setattr(
        generate.llm,
        "generate",
        lambda *a, **k: (_ for _ in ()).throw(AssertionError()),
    )
    result = generate.answer_question(item.question)
    assert deterministic.score(result["answer"], item.truth, "numeric") == "Perfect"
    assert result["verified"] is True
