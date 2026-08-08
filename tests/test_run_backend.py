"""SOT-2469 — offline tests for the production answer-backend wiring in :mod:`src.rag.run`.

The production answer path must be the Gemini tool-driven **investigation agent** (Claude-independent):
``run.run`` defaults to ``gen='investigator'`` and the legacy text backends (``gemini``/``opus``) are
kept for backward-compatible experiments only. These tests drive ``run`` network-free by monkeypatching
the backend entry points, so no Vertex/Claude call is made.
"""
from __future__ import annotations

import csv

import pytest

from config import settings
from src.rag import run


class _FakeInvestigation:
    """Minimal stand-in for :class:`~src.rag.agent.investigator.Investigation` (only ``to_dict``)."""

    def __init__(self, answer: str, confidence: float = 0.9):
        self._answer = answer
        self._confidence = confidence

    def to_dict(self) -> dict:
        return {"question": "q", "answer": self._answer, "confidence": self._confidence,
                "evidence": "e", "method": "m", "stop_reason": "answered"}


def _read_predictions(path) -> list[list[str]]:
    with open(path, encoding="utf-8", newline="") as f:
        return list(csv.reader(f))


def test_production_defaults_are_gemini_only():
    """The default backend is the investigator agent; Claude/opus is not the production default."""
    assert run.DEFAULT_GEN == "investigator"
    assert "opus" in run.EXPERIMENTAL_GEN  # Claude backend is excluded from production
    assert "investigator" not in run.EXPERIMENTAL_GEN


def test_run_defaults_to_investigator_and_never_calls_claude(monkeypatch, tmp_path):
    """A default ``run`` answers every question via the investigator and never touches opus/Claude."""
    monkeypatch.setattr(run, "load_questions", lambda split: [(1, "q1"), (2, "q2")])

    seen: list[str] = []

    def fake_answer_question(q: str, profile=None):
        seen.append(q)
        return _FakeInvestigation(f"ans-{q}")

    from src.rag.agent import investigator
    monkeypatch.setattr(investigator, "answer_question", fake_answer_question)

    # Any accidental use of the Claude backend must fail loudly.
    from src.rag import opus_gen
    monkeypatch.setattr(opus_gen, "answer_question",
                        lambda *a, **k: pytest.fail("opus/Claude backend must not run in production"))

    out = tmp_path / "predictions_test.csv"
    run.run("test", out, limit=None, workers=2, hard=False)  # gen defaults to investigator

    assert sorted(seen) == ["q1", "q2"]
    rows = _read_predictions(out)
    assert rows == [["1", "ans-q1"], ["2", "ans-q2"]]


def test_investigator_abstain_is_written_as_abstain(monkeypatch, tmp_path):
    """An investigator abstention flows through to the sanitised ABSTAIN token."""
    monkeypatch.setattr(run, "load_questions", lambda split: [(7, "hard-q")])
    from src.rag.agent import investigator
    monkeypatch.setattr(investigator, "answer_question",
                        lambda q, profile=None: _FakeInvestigation(settings.ABSTAIN, confidence=0.0))

    out = tmp_path / "predictions.csv"
    run.run("valid", out, limit=None, workers=1, hard=False)
    assert _read_predictions(out) == [["7", settings.ABSTAIN]]


def test_make_worker_selects_backend(monkeypatch):
    """``make_worker`` routes each backend name to its own module; unknown names raise."""
    from src.rag.agent import investigator, tiebreak
    from src.rag import generate, opus_gen

    monkeypatch.setattr(investigator, "answer_question",
                        lambda q, profile=None: _FakeInvestigation("inv"))
    monkeypatch.setattr(tiebreak, "resolve_question", lambda q: _FakeInvestigation("res"))
    monkeypatch.setattr(generate, "answer_question",
                        lambda q, hard=False: {"answer": "gem", "confidence": "high", "used_images": 0})
    monkeypatch.setattr(opus_gen, "answer_question",
                        lambda q: {"answer": "opus", "confidence": "high", "used_images": 0})

    assert run.make_worker("investigator", False)(1, "q")[1]["answer"] == "inv"
    assert run.make_worker("resolve", False)(1, "q")[1]["answer"] == "res"
    assert run.make_worker("gemini", False)(1, "q")[1]["answer"] == "gem"
    assert run.make_worker("opus", False)(1, "q")[1]["answer"] == "opus"
    # agent backends get a uniform used_images key for the run-log schema
    assert run.make_worker("investigator", False)(1, "q")[1]["used_images"] == 0

    with pytest.raises(ValueError):
        run.make_worker("mystery", False)


# --------------------------------------------------------------------------- SOT-2528 run-wide profile
def test_run_shares_one_persisted_profile_across_questions_when_enabled(monkeypatch, tmp_path):
    """RAG_SHARE_CORPUS_PROFILE on ⇒ ONE loaded profile is threaded through every question and saved."""
    from src.rag.tools import profile as _profile

    monkeypatch.setenv("RAG_SHARE_CORPUS_PROFILE", "1")
    profile_path = tmp_path / "corpus_profile.json"
    # seed an on-disk profile so we can prove ``run`` loads it (a prior run's discovery)
    _profile.CorpusProfile(passwords={"seed.docx": "pw0"}).save(profile_path)
    monkeypatch.setattr(_profile, "DEFAULT_PROFILE_PATH", profile_path)

    monkeypatch.setattr(run, "load_questions", lambda split: [(1, "q1"), (2, "q2"), (3, "q3")])
    seen_profiles: list[object] = []

    def fake_answer_question(q: str, profile=None):
        seen_profiles.append(profile)
        # simulate a per-question discovery landing on the shared instance
        profile.set_password(f"{q}.docx", f"pw-{q}")
        return _FakeInvestigation(f"ans-{q}")

    from src.rag.agent import investigator
    monkeypatch.setattr(investigator, "answer_question", fake_answer_question)

    out = tmp_path / "predictions.csv"
    run.run("test", out, limit=None, workers=1, hard=False)

    # every question saw the SAME non-None shared instance, seeded from disk
    assert all(p is not None for p in seen_profiles)
    assert len({id(p) for p in seen_profiles}) == 1
    assert seen_profiles[0].get_password("seed.docx") == "pw0"
    # discoveries were persisted back after the pool joined
    reloaded = _profile.CorpusProfile.load(profile_path)
    assert reloaded.get_password("seed.docx") == "pw0"
    assert {reloaded.get_password(f"q{i}.docx") for i in (1, 2, 3)} == {"pw-q1", "pw-q2", "pw-q3"}


def test_run_uses_fresh_per_question_profile_when_disabled(monkeypatch, tmp_path):
    """Default (flag OFF) ⇒ investigator gets ``profile=None`` (historical per-question fresh profile)."""
    from src.rag.tools import profile as _profile

    monkeypatch.delenv("RAG_SHARE_CORPUS_PROFILE", raising=False)
    # ensure no stray profile is written to the real artifacts dir
    monkeypatch.setattr(_profile, "DEFAULT_PROFILE_PATH", tmp_path / "corpus_profile.json")
    monkeypatch.setattr(run, "load_questions", lambda split: [(1, "q1"), (2, "q2")])

    seen_profiles: list[object] = []

    def fake_answer_question(q: str, profile=None):
        seen_profiles.append(profile)
        return _FakeInvestigation(f"ans-{q}")

    from src.rag.agent import investigator
    monkeypatch.setattr(investigator, "answer_question", fake_answer_question)

    out = tmp_path / "predictions.csv"
    run.run("test", out, limit=None, workers=1, hard=False)

    assert seen_profiles == [None, None]
    assert not (tmp_path / "corpus_profile.json").exists()  # nothing saved when disabled
