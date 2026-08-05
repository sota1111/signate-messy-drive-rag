"""Repo-wide test fixtures."""
import pytest


@pytest.fixture(autouse=True)
def _pin_judge_backend_gemini(monkeypatch):
    """The network-free judge tests were written against the Gemini path (they stub
    crag._judge_gemini). SOT-2457 made codex the default backend whenever the CLI is on
    PATH, which would reroute those tests to a live `codex exec`. Pin the explicit
    backend for tests; codex-specific tests override with their own setenv."""
    monkeypatch.setenv("JUDGE_BACKEND", "gemini")
