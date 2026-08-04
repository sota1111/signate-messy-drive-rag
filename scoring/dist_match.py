"""Match an evaluation benchmark to the public test-question archetype distribution."""
from __future__ import annotations

import collections
import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence, TypeVar

from src.rag.archetype import classify

T = TypeVar("T")


def load_questions(path: Path) -> list[str]:
    """Load questions from a CSV, accepting the competition's common column names."""
    with path.open(encoding="utf-8-sig", newline="") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        return []
    candidates = ("question", "Question", "query", "text", "問題", "質問")
    column = next((c for c in candidates if c in rows[0]), None)
    if column is None:
        # The official file has an id column followed by the question. Prefer the first non-id field.
        column = next((c for c in rows[0] if c.lower() not in {"id", "index"}), None)
    if column is None:
        raise ValueError(f"no question column in {path}")
    return [str(r.get(column, "")).strip() for r in rows if str(r.get(column, "")).strip()]


def histogram(questions: Iterable[str]) -> dict[str, int]:
    return dict(sorted(collections.Counter(classify(q) for q in questions).items()))


_FAMILIES = {
    "enum_set": "extract", "highlight_set": "extract", "document_extract": "extract",
    "cross_aggregate": "calculate", "contract_amount": "calculate",
    "config_hyperparam": "calculate", "metric_score": "calculate", "data_shape": "calculate",
    "csv_column_mean": "calculate", "csv_column_max": "calculate", "derived_calculation": "calculate",
    "version_diff": "compare",
}


def family(archetype: str) -> str:
    return _FAMILIES.get(archetype, "lookup")


def family_histogram(questions: Iterable[str]) -> dict[str, int]:
    return dict(sorted(collections.Counter(family(classify(q)) for q in questions).items()))


def distribution(counts: dict[str, int]) -> dict[str, float]:
    total = sum(counts.values())
    return {k: v / total for k, v in counts.items()} if total else {}


def item_weights(items: Sequence[T], target_counts: dict[str, int], *, archetype_attr: str = "archetype") -> list[float]:
    """Importance weights whose mean is one; absent test archetypes receive zero weight."""
    eval_counts = collections.Counter(family(str(getattr(x, archetype_attr))) for x in items)
    target = distribution(target_counts)
    raw = [target.get(family(str(getattr(x, archetype_attr))), 0.0) /
           eval_counts[family(str(getattr(x, archetype_attr)))]
           for x in items]
    scale = len(raw) / sum(raw) if raw and sum(raw) else 0.0
    return [w * scale for w in raw]


def weighted_mean(values: Sequence[float], weights: Sequence[float]) -> float:
    if len(values) != len(weights):
        raise ValueError("values and weights must have equal length")
    denom = sum(weights)
    return sum(v * w for v, w in zip(values, weights)) / denom if denom else 0.0


@dataclass(frozen=True)
class DistributionMatch:
    test_counts: dict[str, int]
    test_distribution: dict[str, float]
    eval_counts: dict[str, int]
    missing_archetypes: tuple[str, ...]
    test_archetype_counts: dict[str, int]


def assess(items: Sequence[T], test_questions: Iterable[str], *, archetype_attr: str = "archetype") -> DistributionMatch:
    questions = list(test_questions)
    archetypes = histogram(questions)
    tc = family_histogram(questions)
    ec = dict(sorted(collections.Counter(family(str(getattr(x, archetype_attr))) for x in items).items()))
    missing = tuple(sorted(a for a, n in tc.items() if n and not ec.get(a)))
    return DistributionMatch(tc, distribution(tc), ec, missing, archetypes)
