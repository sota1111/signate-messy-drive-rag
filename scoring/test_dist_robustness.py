from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pytest

from scoring import dist_match, perturb


@dataclass
class Item:
    id: str
    question: str
    archetype: str


def test_official_test_distribution_classifies_all_100_questions():
    questions = dist_match.load_questions(Path("data/questions/questions_test.csv"))
    counts = dist_match.histogram(questions)
    assert len(questions) == 100
    assert sum(counts.values()) == 100
    assert counts  # includes unknown explicitly rather than silently dropping it


def test_importance_weights_reproduce_target_archetype_mix():
    items = [Item("a1", "", "enum_set"), Item("a2", "", "document_extract"),
             Item("b1", "", "derived_calculation")]
    weights = dist_match.item_weights(items, {"extract": 1, "calculate": 3})
    total = sum(weights)
    assert sum(w for it, w in zip(items, weights) if dist_match.family(it.archetype) == "extract") / total == pytest.approx(.25)
    assert sum(w for it, w in zip(items, weights) if dist_match.family(it.archetype) == "calculate") / total == pytest.approx(.75)


def test_perturbations_cover_alias_word_order_and_redundancy():
    q = "社内用語PPの正式名称について、答えてください。"
    changed = perturb.variants(q, limit=3)
    assert len(changed) == 3
    assert any("社内で使う略称" in x for x in changed)
    assert any("対象は" in x for x in changed)
    assert any("確認してください" in x for x in changed)


def test_robustness_flags_hard_module_answer_flip_as_overfit():
    result = perturb.score([
        ("hard-4", "5,775,000円", ["5775000 円", "6,000,000円"]),
        ("stable", "hist_gradient_boosting", ["hist gradient boosting"]),
    ])
    assert result.score == pytest.approx(.5)
    assert result.unstable_ids == ("hard-4",)
