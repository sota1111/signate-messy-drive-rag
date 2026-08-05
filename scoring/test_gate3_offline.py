"""関門3 offline ゴールド採点 前段ゲート (SOT-2477).

The real SIGNATE submission is guarded by TWO gates: an offline gold match-rate gate and the
existing ``SIGNATE_SUBMIT_ALLOWED`` human gate. These tests pin the offline gate + the wiring in
``gate3.main`` without any network call (the CRAG judge / runner / signate are all stubbed)."""
import subprocess

import pytest

from scoring import gate3, gold_offline


def _report(match: int, total: int) -> gold_offline.Report:
    """A gold_offline.Report with a controllable match rate (`match`/`total`)."""
    items = [
        gold_offline.Item(index=i, question="q", answer="a", gold="g", archetype="x",
                          verdict="Perfect" if i < match else "Incorrect")
        for i in range(total)
    ]
    return gold_offline.Report(items=items)


# --------------------------------------------------------------------------- offline scoring works
def test_score_offline_reports_match_rate(tmp_path):
    """offline採点が機能: predictions are matched against gold via an injected judge."""
    gold = tmp_path / "gold.csv"
    gold.write_text("0,A\n1,B\n2,C\n", encoding="utf-8")
    preds = tmp_path / "preds.csv"
    preds.write_text("index,answer\n0,A\n1,X\n2,\n", encoding="utf-8")

    # index 2 is an empty (sentinel) answer → pre-bucketed Missing without a judge call; the judge
    # therefore only sees index 0 (correct) and index 1 (wrong), in index order.
    def stub_judge(pairs):
        assert pairs == [("A", "A"), ("X", "B")]
        return ["Perfect", "Incorrect"]

    report = gate3.score_offline(preds, gold, judge=stub_judge)
    assert report.n == 3
    assert report.match_rate == pytest.approx(1 / 3)


def test_offline_gate_boundary_is_inclusive():
    report = _report(5, 10)  # 50%
    assert gate3.offline_gate(report, 0.5) is True   # >= threshold passes
    assert gate3.offline_gate(report, 0.5001) is False


def test_gold_threshold_default_env_and_invalid(monkeypatch):
    monkeypatch.delenv("GATE3_GOLD_THRESHOLD", raising=False)
    assert gate3.gold_threshold() == pytest.approx(gate3.DEFAULT_GOLD_THRESHOLD)
    monkeypatch.setenv("GATE3_GOLD_THRESHOLD", "0.42")
    assert gate3.gold_threshold() == pytest.approx(0.42)
    monkeypatch.setenv("GATE3_GOLD_THRESHOLD", "not-a-number")
    with pytest.raises(SystemExit):
        gate3.gold_threshold()


# --------------------------------------------------------------------------- CLI gating wiring
def _patch_pipeline(monkeypatch, report):
    """Stub the network-bound steps of gate3.main; return the recorded (zip, signate) calls list."""
    calls: list[str] = []
    monkeypatch.setattr(gate3, "ensure_predictions", lambda preds, *a, **k: preds)
    monkeypatch.setattr(gate3, "score_offline", lambda *a, **k: report)
    monkeypatch.setattr(gate3, "build_zip", lambda preds: calls.append("zip") or preds)
    monkeypatch.setattr(subprocess, "run",
                        lambda cmd, **k: calls.append("signate") or
                        subprocess.CompletedProcess(cmd, 0))
    return calls


def test_main_submit_blocked_below_offline_threshold(monkeypatch):
    """Even with the human gate open, a below-threshold offline score must NOT build/submit."""
    calls = _patch_pipeline(monkeypatch, _report(1, 10))  # 10%
    monkeypatch.setenv("SIGNATE_SUBMIT_ALLOWED", "1")
    with pytest.raises(SystemExit, match="offline gold match"):
        gate3.main(["--no-run", "--submit", "--threshold", "0.5"])
    assert calls == []  # neither zip built nor signate invoked


def test_main_submit_proceeds_when_both_gates_pass(monkeypatch):
    calls = _patch_pipeline(monkeypatch, _report(9, 10))  # 90%
    monkeypatch.setenv("SIGNATE_SUBMIT_ALLOWED", "1")
    rc = gate3.main(["--no-run", "--submit", "--threshold", "0.5", "--memo", "m"])
    assert rc == 0
    assert calls == ["zip", "signate"]


def test_main_submit_passes_offline_but_human_gate_blocks(monkeypatch):
    """Offline gate PASS but human gate closed → build zip, then submit() blocks (no signate)."""
    calls = _patch_pipeline(monkeypatch, _report(9, 10))
    monkeypatch.delenv("SIGNATE_SUBMIT_ALLOWED", raising=False)
    with pytest.raises(SystemExit, match="human permission required"):
        gate3.main(["--no-run", "--submit", "--threshold", "0.5"])
    assert calls == ["zip"]  # zip built, but signate never runs


def test_main_dry_run_never_submits(monkeypatch):
    calls = _patch_pipeline(monkeypatch, _report(9, 10))
    rc = gate3.main(["--no-run"])
    assert rc == 0
    assert calls == []
