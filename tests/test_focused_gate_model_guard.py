"""SOT-2625 — model guard for the focused gate runner (scripts/run_focused_gate.py).

Offline unit tests (no Gemini / no codex): they exercise the pure resolution/guard helpers, so an
OFFICIAL non-regression verdict can only come from the flash-3.6 production stack and Sonnet/other
configs are mechanically demoted to official:false.
"""
from __future__ import annotations

import importlib

import pytest

gate = importlib.import_module("scripts.run_focused_gate")


@pytest.fixture()
def clean_env(monkeypatch):
    for k in ("LLM_PROVIDER", "GEN_MODEL", "GEN_MODEL_HARD", "VISION_MODEL",
              "VERTEX_LOCATION", "CLAUDE_CLI_MODEL"):
        monkeypatch.delenv(k, raising=False)
    return monkeypatch


def _official_env(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "gemini")
    monkeypatch.setenv("GEN_MODEL", gate.OFFICIAL_MODEL)
    monkeypatch.setenv("GEN_MODEL_HARD", gate.OFFICIAL_MODEL)
    monkeypatch.setenv("VISION_MODEL", gate.OFFICIAL_MODEL)
    monkeypatch.setenv("VERTEX_LOCATION", gate.OFFICIAL_LOCATION)


def test_official_stack_passes_guard(clean_env):
    _official_env(clean_env)
    info = gate._model_env()
    assert info["model"] == gate.OFFICIAL_MODEL
    assert info["provider"] == "gemini"
    assert gate._official_guard_reasons(info) == []


def test_claude_cli_is_rejected_as_official(clean_env):
    # Sonnet dev path: right on GEN_MODEL/location is irrelevant — provider alone breaks official.
    clean_env.setenv("LLM_PROVIDER", "claude-cli")
    clean_env.setenv("GEN_MODEL", gate.OFFICIAL_MODEL)
    clean_env.setenv("GEN_MODEL_HARD", gate.OFFICIAL_MODEL)
    clean_env.setenv("VISION_MODEL", gate.OFFICIAL_MODEL)
    clean_env.setenv("VERTEX_LOCATION", gate.OFFICIAL_LOCATION)
    info = gate._model_env()
    reasons = gate._official_guard_reasons(info)
    assert reasons, "claude-cli must not qualify as official"
    assert any("LLM_PROVIDER" in r for r in reasons)
    # the honest ledger `model` records the dev text model, not the gemini gen model
    clean_env.setenv("CLAUDE_CLI_MODEL", "sonnet")
    assert gate._model_env()["model"] == "claude-cli:sonnet"


def test_wrong_location_rejected(clean_env):
    _official_env(clean_env)
    clean_env.setenv("VERTEX_LOCATION", "us-central1")  # flash 3.6 404s off global
    reasons = gate._official_guard_reasons(gate._model_env())
    assert any("VERTEX_LOCATION" in r for r in reasons)


def test_wrong_gen_model_rejected(clean_env):
    _official_env(clean_env)
    clean_env.setenv("GEN_MODEL", "gemini-2.5-flash")
    reasons = gate._official_guard_reasons(gate._model_env())
    assert any("GEN_MODEL=" in r for r in reasons)


def test_pro_hard_model_rejected(clean_env):
    # global has no pro tier, so a leftover pro HARD/VISION must break official.
    _official_env(clean_env)
    clean_env.setenv("GEN_MODEL_HARD", "gemini-2.5-pro")
    reasons = gate._official_guard_reasons(gate._model_env())
    assert any("GEN_MODEL_HARD" in r for r in reasons)


def test_ledger_required_fields_enforced():
    ok = {"model": "gemini-3.6-flash", "provider": "gemini", "official": True, "gate": "PASS"}
    gate._require_ledger_fields(ok)  # no raise

    for missing in ("model", "provider", "official"):
        bad = dict(ok)
        bad.pop(missing)
        with pytest.raises(SystemExit):
            gate._require_ledger_fields(bad)

    # empty-string / None also count as missing
    with pytest.raises(SystemExit):
        gate._require_ledger_fields({"model": "", "provider": "gemini", "official": False})
    # official:False is a valid value (must NOT be treated as missing)
    gate._require_ledger_fields({"model": "claude-cli:sonnet", "provider": "claude-cli", "official": False})
