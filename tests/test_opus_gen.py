"""Network-free regressions for the Claude Opus answer backend (SOT-2457).

No test launches the real `claude` CLI — subprocess / pipeline seams are stubbed.
"""
import json
import subprocess

import pytest

from config import settings
from src.rag import generate, opus_gen


# ---- output parsing ----

def test_parse_clean_json():
    assert opus_gen._parse('{"answer": "20日", "confidence": "high"}') == {
        "answer": "20日", "confidence": "high"}


def test_parse_fenced_json():
    raw = '説明します。\n```json\n{"answer": "5ページ", "confidence": "medium"}\n```'
    assert opus_gen._parse(raw)["answer"] == "5ページ"


def test_parse_json_with_surrounding_prose():
    raw = '回答は以下です。 {"answer": "hr、weekday", "confidence": "high"} 以上。'
    assert opus_gen._parse(raw)["answer"] == "hr、weekday"


def test_parse_garbage_degrades_to_low_confidence():
    obj = opus_gen._parse("すみません、判断できませんでした。")
    assert obj == {"answer": "", "confidence": "low"}


# ---- claude CLI invocation ----

def _stub_claude(monkeypatch, stdout="", returncode=0, calls=None):
    def fake_run(cmd, **kwargs):
        if calls is not None:
            calls.append((cmd, kwargs))
        return subprocess.CompletedProcess(cmd, returncode, stdout=stdout, stderr="")

    monkeypatch.setattr(opus_gen, "available", lambda: True)
    monkeypatch.setattr(subprocess, "run", fake_run)


def test_run_claude_passes_model_and_prompt_on_stdin(monkeypatch):
    calls = []
    _stub_claude(monkeypatch, stdout="ok", calls=calls)
    monkeypatch.setenv("OPUS_GEN_MODEL", "opus")
    assert opus_gen._run_claude("PROMPT") == "ok"
    cmd, kwargs = calls[0]
    assert cmd[:2] == ["claude", "-p"]
    assert cmd[cmd.index("--model") + 1] == "opus"
    assert kwargs["input"] == "PROMPT"


def test_run_claude_nonzero_exit_raises(monkeypatch):
    _stub_claude(monkeypatch, stdout="", returncode=1)
    with pytest.raises(opus_gen.OpusGenError):
        opus_gen._run_claude("PROMPT")


def test_run_claude_missing_cli_raises(monkeypatch):
    monkeypatch.setattr(opus_gen, "available", lambda: False)
    with pytest.raises(opus_gen.OpusGenError, match="not found"):
        opus_gen._run_claude("PROMPT")


# ---- answer_question pipeline ----

class _StubRetriever:
    def retrieve(self, question, k=16):
        return [{"text": "根拠テキスト", "rel": "a/b.txt", "kind": "text"}]


def _stub_pipeline(monkeypatch, front=(None, []), claude_stdout=""):
    monkeypatch.setattr(generate, "deterministic_front",
                        lambda q, **kw: front)
    monkeypatch.setattr(opus_gen.retrieve, "get", lambda: _StubRetriever())
    _stub_claude(monkeypatch, stdout=claude_stdout)


def test_high_confidence_answer_commits(monkeypatch):
    _stub_pipeline(monkeypatch,
                   claude_stdout=json.dumps({"answer": "20日", "confidence": "high"}))
    res = opus_gen.answer_question("TG平均が最も低い日は?")
    assert res["answer"] == "20日"
    assert res["verified"] is True
    assert res["evidence_files"] == ["a/b.txt"]


def test_non_high_confidence_abstains(monkeypatch):
    _stub_pipeline(monkeypatch,
                   claude_stdout=json.dumps({"answer": "20日", "confidence": "medium"}))
    res = opus_gen.answer_question("TG平均が最も低い日は?")
    assert res["answer"] == settings.ABSTAIN


def test_unparseable_output_abstains(monkeypatch):
    _stub_pipeline(monkeypatch, claude_stdout="判断できません")
    res = opus_gen.answer_question("何日?")
    assert res["answer"] == settings.ABSTAIN


def test_deterministic_front_short_circuits_without_claude_call(monkeypatch):
    front_result = {"question": "q", "answer": "42", "confidence": "high",
                    "raw_answer": "42", "verified": True, "evidence_files": [],
                    "used_images": 0}
    monkeypatch.setattr(generate, "deterministic_front", lambda q, **kw: (front_result, []))

    def no_claude(*_a, **_k):
        raise AssertionError("claude must not be called when the front resolves")

    monkeypatch.setattr(opus_gen, "_run_claude", no_claude)
    assert opus_gen.answer_question("q") is front_result


def test_advisory_hints_reach_the_prompt(monkeypatch):
    captured = {}
    monkeypatch.setattr(generate, "deterministic_front",
                        lambda q, **kw: (None, ["候補: 42"]))
    monkeypatch.setattr(opus_gen.retrieve, "get", lambda: _StubRetriever())

    def fake_run_claude(prompt):
        captured["prompt"] = prompt
        return json.dumps({"answer": "42", "confidence": "high"})

    monkeypatch.setattr(opus_gen, "_run_claude", fake_run_claude)
    opus_gen.answer_question("q")
    assert "候補: 42" in captured["prompt"]
    assert "未検証の自動抽出候補" in captured["prompt"]
