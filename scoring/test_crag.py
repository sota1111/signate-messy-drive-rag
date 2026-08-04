"""Network-free regressions for the hybrid CRAG judge."""
from scoring import crag
from scoring import deterministic as D


def test_official_rounding_examples():
    assert D.score_numeric("12.5", "12.47") == "Acceptable"
    assert D.score_numeric("12.5", "12.45") == "Acceptable"
    assert D.score_numeric("12.4", "12.47") == "Incorrect"


def test_known_misjudgements_are_deterministic():
    assert crag.deterministic_judge("Recall", "Recall") == "Perfect"
    assert crag.deterministic_judge("100円（税込）", "100円（税抜）") == "Incorrect"
    assert crag.deterministic_judge("20", "20日") == "Perfect"


def test_enumeration_is_order_independent_and_complete():
    assert crag.deterministic_judge("A、B、C", "C、A、B") == "Perfect"
    assert crag.deterministic_judge("A、B", "A、B、C") == "Incorrect"


def test_deterministic_route_never_calls_llm(monkeypatch):
    monkeypatch.setattr(crag, "_judge_gemini", lambda *_: (_ for _ in ()).throw(AssertionError("LLM called")))
    assert crag.judge("Recall", "Recall") == "Perfect"


def test_free_text_falls_back_to_majority_three(monkeypatch):
    calls = iter(["Incorrect", "Perfect", "Perfect"])
    monkeypatch.setattr(crag, "_judge_gemini", lambda *_: next(calls))
    assert crag.judge("主要な内容を自然文で説明しています。", "同じ意味を別の自然文で説明しています。") == "Perfect"
