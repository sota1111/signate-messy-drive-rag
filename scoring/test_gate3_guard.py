"""関門3 human-permission guard (SOT-2457): submission never runs without explicit approval."""
import subprocess

import pytest

from scoring import gate3


def test_submit_blocked_without_permission_env(monkeypatch, tmp_path):
    monkeypatch.delenv("SIGNATE_SUBMIT_ALLOWED", raising=False)

    def no_signate(*_a, **_k):
        raise AssertionError("signate submit must not run without permission")

    monkeypatch.setattr(subprocess, "run", no_signate)
    with pytest.raises(SystemExit, match="human permission required"):
        gate3.submit(tmp_path / "submission.zip", "memo")


def test_submit_runs_with_permission_env(monkeypatch, tmp_path):
    monkeypatch.setenv("SIGNATE_SUBMIT_ALLOWED", "1")
    calls = []
    monkeypatch.setattr(subprocess, "run",
                        lambda cmd, **kw: calls.append(cmd) or
                        subprocess.CompletedProcess(cmd, 0))
    gate3.submit(tmp_path / "submission.zip", "memo")
    assert calls and calls[0][:2] == ["signate", "submit"]
