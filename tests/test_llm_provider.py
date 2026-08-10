"""SOT-2606 — offline tests for the pluggable LLM text-role provider abstraction.

Network-free: the Gemini path (:func:`llm._gemini_generate`), the claude-cli subprocess
(``subprocess.run``), and the Vertex client (:func:`llm.client`) are monkeypatched, so the routing
/ guard logic runs (not a stub). Invariants under test:

* default ``LLM_PROVIDER`` == ``gemini`` and text generate() goes to the Gemini path (byte-identical
  routing — same args forwarded);
* ``LLM_PROVIDER=claude-cli`` routes tool-free text generate() to the claude-cli provider;
* GUARD (misroute prevention): with ``LLM_PROVIDER=claude-cli`` an image (vision) prompt is forced
  to Gemini and NEVER reaches claude-cli; the investigator function-calling loop
  (``GeminiModel``) stays pinned to the Vertex client + ``GEN_MODEL_HARD`` regardless of provider;
* ``anthropic`` raises NotImplementedError, an unknown provider raises ValueError;
* the claude-cli provider builds the right ``claude -p`` command, feeds the prompt on stdin, parses
  the ``result`` field, and raises (no silent Gemini fallback) on error/empty/nonzero exit.
"""
from __future__ import annotations

import json
import subprocess

import pytest

from src.rag import llm
from src.rag.llm_providers import claude_cli


@pytest.fixture(autouse=True)
def _clear_provider_env(monkeypatch):
    """Every test starts from an unset provider env so ordering can't leak state."""
    monkeypatch.delenv("LLM_PROVIDER", raising=False)
    monkeypatch.delenv("CLAUDE_CLI_MODEL", raising=False)
    monkeypatch.setattr(claude_cli.time, "sleep", lambda _s: None)  # keep retry-path tests fast


# --------------------------------------------------------------------------- routing / guards
def test_default_provider_is_gemini(monkeypatch):
    assert llm._provider() == "gemini"

    seen = {}

    def _fake_gemini(prompt, **kw):
        seen["prompt"] = prompt
        seen["kw"] = kw
        return "GEMINI"

    monkeypatch.setattr(llm, "_gemini_generate", _fake_gemini)
    monkeypatch.setattr(
        claude_cli, "generate_text",
        lambda *a, **k: pytest.fail("claude-cli must not be called on the default gemini provider"),
    )

    out = llm.generate("hello", system="sys", model="m", temperature=0.0, response_schema={"x": 1})
    assert out == "GEMINI"
    # Byte-identical routing: the same call args are forwarded to the Gemini path unchanged.
    assert seen["prompt"] == "hello"
    assert seen["kw"]["system"] == "sys"
    assert seen["kw"]["model"] == "m"
    assert seen["kw"]["response_schema"] == {"x": 1}
    assert seen["kw"]["images"] is None


def test_claude_cli_text_routes_to_provider(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "claude-cli")
    monkeypatch.setattr(
        llm, "_gemini_generate",
        lambda *a, **k: pytest.fail("gemini must not be called for a claude-cli text prompt"),
    )
    captured = {}

    def _fake_cli(prompt, **kw):
        captured["prompt"] = prompt
        captured["kw"] = kw
        return "CLAUDE"

    monkeypatch.setattr(claude_cli, "generate_text", _fake_cli)

    out = llm.generate("hi there", system="s", model=None, response_schema={"a": 2})
    assert out == "CLAUDE"
    assert captured["prompt"] == "hi there"
    assert captured["kw"]["system"] == "s"
    assert captured["kw"]["response_schema"] == {"a": 2}


def test_claude_cli_with_images_forced_to_gemini(monkeypatch):
    """GUARD: vision prompts are pinned to Gemini even when the provider is claude-cli."""
    monkeypatch.setenv("LLM_PROVIDER", "claude-cli")
    monkeypatch.setattr(
        claude_cli, "generate_text",
        lambda *a, **k: pytest.fail("vision (images) must NEVER route to claude-cli"),
    )
    monkeypatch.setattr(llm, "_gemini_generate", lambda *a, **k: "GEMINI_VISION")

    out = llm.generate("describe", images=[llm.Image(b"\x89PNG", "image/png")])
    assert out == "GEMINI_VISION"


def test_investigator_loop_pinned_to_gemini(monkeypatch):
    """GUARD: the function-calling loop uses the Vertex client + GEN_MODEL_HARD, ignoring LLM_PROVIDER."""
    from config import settings
    from src.rag.agent import investigator

    monkeypatch.setenv("LLM_PROVIDER", "claude-cli")
    sentinel = object()
    monkeypatch.setattr(llm, "client", lambda: sentinel)

    m = investigator.GeminiModel("q?", None)
    assert m._client is sentinel                       # the Vertex client, not any claude-cli path
    assert m.model_name == settings.GEN_MODEL_HARD     # a Gemini model, chosen without consulting provider


def test_anthropic_provider_not_implemented(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "anthropic")
    with pytest.raises(NotImplementedError):
        llm.generate("hi")


def test_unknown_provider_raises(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "totally-bogus")
    with pytest.raises(ValueError):
        llm.generate("hi")


def test_provider_name_is_normalized(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "  Claude-CLI  ")
    assert llm._provider() == "claude-cli"


# --------------------------------------------------------------------------- claude_cli provider unit
def _fake_completed(stdout: str, returncode: int = 0, stderr: str = ""):
    return subprocess.CompletedProcess(args=["claude"], returncode=returncode, stdout=stdout, stderr=stderr)


def test_claude_cli_builds_command_and_parses_result(monkeypatch):
    monkeypatch.setattr(claude_cli.shutil, "which", lambda _bin: "/usr/bin/claude")
    calls = {}

    def _fake_run(cmd, *, input, capture_output, text, timeout):
        calls["cmd"] = cmd
        calls["input"] = input
        return _fake_completed(json.dumps({"is_error": False, "result": "  hello world  "}))

    monkeypatch.setattr(claude_cli.subprocess, "run", _fake_run)

    out = claude_cli.generate_text("PROMPT", system="SYS", model="sonnet")
    assert out == "hello world"                                  # parsed + stripped
    assert calls["input"] == "PROMPT"                            # prompt fed on stdin
    assert calls["cmd"][:2] == ["claude", "-p"]
    assert "--model" in calls["cmd"] and "sonnet" in calls["cmd"]
    assert "--output-format" in calls["cmd"] and "json" in calls["cmd"]
    assert "--append-system-prompt" in calls["cmd"] and "SYS" in calls["cmd"]


def test_claude_cli_schema_hint_appended(monkeypatch):
    monkeypatch.setattr(claude_cli.shutil, "which", lambda _bin: "/usr/bin/claude")
    captured = {}

    def _fake_run(cmd, *, input, capture_output, text, timeout):
        captured["input"] = input
        return _fake_completed(json.dumps({"is_error": False, "result": "{}"}))

    monkeypatch.setattr(claude_cli.subprocess, "run", _fake_run)

    claude_cli.generate_text("Q", response_schema={"type": "object"})
    assert "JSON schema" in captured["input"]
    assert "\"type\": \"object\"" in captured["input"]


def test_claude_cli_missing_binary_raises(monkeypatch):
    monkeypatch.setattr(claude_cli.shutil, "which", lambda _bin: None)
    with pytest.raises(RuntimeError, match="not on PATH"):
        claude_cli.generate_text("hi")


def test_claude_cli_error_result_raises_no_silent_fallback(monkeypatch):
    monkeypatch.setattr(claude_cli.shutil, "which", lambda _bin: "/usr/bin/claude")
    monkeypatch.setattr(
        claude_cli.subprocess, "run",
        lambda *a, **k: _fake_completed(json.dumps({"is_error": True, "result": None})),
    )
    with pytest.raises(RuntimeError):
        claude_cli.generate_text("hi", retries=1)


def test_claude_cli_nonzero_exit_raises(monkeypatch):
    monkeypatch.setattr(claude_cli.shutil, "which", lambda _bin: "/usr/bin/claude")
    monkeypatch.setattr(
        claude_cli.subprocess, "run",
        lambda *a, **k: _fake_completed("", returncode=1, stderr="boom"),
    )
    with pytest.raises(RuntimeError, match="exited 1|failed after"):
        claude_cli.generate_text("hi", retries=1)


def test_claude_cli_empty_result_raises(monkeypatch):
    monkeypatch.setattr(claude_cli.shutil, "which", lambda _bin: "/usr/bin/claude")
    monkeypatch.setattr(
        claude_cli.subprocess, "run",
        lambda *a, **k: _fake_completed(json.dumps({"is_error": False, "result": "   "})),
    )
    with pytest.raises(RuntimeError):
        claude_cli.generate_text("hi", retries=1)
