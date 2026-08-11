"""SOT-2627 — offline tests for the claude-mcp investigator backend.

Network-free / no subprocess: they exercise the stream-json parser, usage-limit detection, the resume
sidecar round-trip, the process-wide usage-limit short-circuit, the flat-rate cost zeroing, and the
server-side tool-call budget cap directly. No ``claude`` process is spawned.
"""
from __future__ import annotations

import json

import pytest

from src.rag.agent import investigator as inv
from src.rag.agent.investigator import AgentTool, build_tools
from src.rag.llm_providers import claude_mcp
from src.rag.mcp import server as mcp_server
from src.rag.tools.profile import CorpusProfile


@pytest.fixture(autouse=True)
def _clear_latch():
    claude_mcp.reset_usage_limit()
    yield
    claude_mcp.reset_usage_limit()


def _stream(*events: dict) -> str:
    return "\n".join(json.dumps(e, ensure_ascii=False) for e in events)


def test_parse_stream_json_extracts_submit_answer_and_tools():
    stdout = _stream(
        {"type": "system", "subtype": "init"},
        # a claude built-in (schema discovery) must NOT be recorded as an investigator tool round
        {"type": "assistant", "message": {"content": [
            {"type": "tool_use", "name": "ToolSearch", "input": {"query": "compute"}}],
            "usage": {"input_tokens": 3, "output_tokens": 1}}},
        {"type": "assistant", "message": {"content": [
            {"type": "tool_use", "name": "mcp__investigator__file_grep", "input": {"query": "x"}}],
            "usage": {"input_tokens": 10, "output_tokens": 5}}},
        {"type": "assistant", "message": {"content": [
            {"type": "tool_use", "name": "mcp__investigator__submit_answer",
             "input": {"answer": "42件", "confidence": 0.9, "evidence": "train.xlsx", "method": "compute"}}],
            "usage": {"input_tokens": 20, "output_tokens": 8}}},
        {"type": "result", "subtype": "success", "is_error": False, "result": "done",
         "usage": {"input_tokens": 30, "output_tokens": 13}},
    )
    p = claude_mcp._parse_stream_json(stdout)
    assert p["tool_calls"] == ["file_grep", "submit_answer"]
    assert p["submit_args"] == {"answer": "42件", "confidence": 0.9,
                                "evidence": "train.xlsx", "method": "compute"}
    assert p["is_error"] is False
    # result-message usage is cumulative and preferred over per-turn summing.
    assert p["usage"].input_tokens == 30 and p["usage"].output_tokens == 13


def test_parse_stream_json_plain_final_text_no_submit():
    stdout = _stream(
        {"type": "assistant", "message": {"content": [{"type": "text", "text": "以下が回答です"}],
                                          "usage": {"input_tokens": 5, "output_tokens": 3}}},
        {"type": "result", "subtype": "success", "is_error": False, "result": "最終回答テキスト"},
    )
    p = claude_mcp._parse_stream_json(stdout)
    assert p["submit_args"] is None
    assert p["final_text"] == "最終回答テキスト"
    assert p["tool_calls"] == []


@pytest.mark.parametrize("text,expected", [
    ("Claude AI usage limit reached", True),
    ("rate limit exceeded (429)", True),
    ("overloaded_error", True),
    ("something normal happened", False),
    ("", False),
])
def test_is_usage_limit(text, expected):
    assert claude_mcp._is_usage_limit(text) is expected


def test_resume_record_roundtrip():
    orig = inv.Investigation(
        question="Q?", answer=inv.Answer("A", 0.8, "ev", "meth"), iterations=3,
        tool_calls=["file_grep", "compute", "submit_answer"], usage=inv.Usage(100, 50),
        model="sonnet(claude-mcp)", elapsed_s=4.2, stop_reason="answered", error=None,
        contract="numeric")
    rec = claude_mcp._investigation_to_record(orig)
    rebuilt = claude_mcp._record_to_investigation(rec)
    assert rebuilt.to_dict() == orig.to_dict()


def test_usage_limit_short_circuits_without_subprocess(monkeypatch, tmp_path):
    monkeypatch.setenv("RAG_CLAUDE_MCP_RESUME", str(tmp_path / "resume.jsonl"))
    # If _run_claude is called after the latch is set, that's a failure (it must short-circuit first).
    monkeypatch.setattr(claude_mcp, "_run_claude",
                        lambda *a, **k: pytest.fail("must not call claude after usage-limit latch"))
    claude_mcp._USAGE_LIMIT.set()
    tools = build_tools(CorpusProfile())
    out = claude_mcp.investigate_question("何件?", tools=tools, contract="numeric")
    assert out.stop_reason == "usage_limit"
    assert out.answer.answer == inv.ABSTAIN
    assert out.answer.confidence == 0.0


def test_resume_hit_returns_cached_without_subprocess(monkeypatch, tmp_path):
    resume = tmp_path / "resume.jsonl"
    monkeypatch.setenv("RAG_CLAUDE_MCP_RESUME", str(resume))
    q = "京橋案件の担当者は?"
    label = f"{claude_mcp.DEFAULT_MODEL}({claude_mcp._MODEL_TAG})"
    key = claude_mcp._resume_key(q, label)
    cached = inv.Investigation(
        question=q, answer=inv.Answer("田中", 0.7, "seat", "lookup"), iterations=1,
        tool_calls=["seating_lookup", "submit_answer"], usage=inv.Usage(1, 1),
        model=label, elapsed_s=1.0, stop_reason="answered", error=None, contract="simple_lookup")
    claude_mcp._append_resume(resume, key, q, cached)
    monkeypatch.setattr(claude_mcp, "_run_claude",
                        lambda *a, **k: pytest.fail("resume hit must not spawn claude"))
    out = claude_mcp.investigate_question(q, tools=build_tools(CorpusProfile()), contract="simple_lookup")
    assert out.answer.answer == "田中"
    assert out.stop_reason == "answered"


def test_flat_rate_cost_is_zero():
    assert inv.Usage(1_000_000, 1_000_000).cost_usd("sonnet(claude-mcp)") == 0.0
    # a Gemini model is priced exactly as before (byte-identical details cost).
    assert inv.Usage(1_000_000, 0).cost_usd("gemini-2.5-pro") == pytest.approx(1.25)


# -- server-side tool-call budget cap (SOT-2627) ---------------------------------------------------
def _make_capped_server(cap: int) -> mcp_server.InvestigatorMCPServer:
    return mcp_server.build_server(CorpusProfile(), log_path=None, max_tool_calls=cap)


def _call(server, name, args=None):
    return server.handle({"jsonrpc": "2.0", "id": 1, "method": "tools/call",
                          "params": {"name": name, "arguments": args or {}}})


def test_tool_cap_disabled_by_default_is_byte_identical():
    server = mcp_server.build_server(CorpusProfile(), log_path=None)
    assert server.max_tool_calls == 0
    # many non-terminal calls, never a budget message
    for _ in range(5):
        resp = _call(server, "find_files", {"query": "train", "ext": "xlsx"})
        assert "budget_exhausted" not in json.dumps(resp["result"], ensure_ascii=False)


def test_tool_cap_blocks_after_budget_and_allows_submit():
    server = _make_capped_server(2)
    # first two non-terminal calls dispatch normally
    for _ in range(2):
        resp = _call(server, "find_files", {"query": "train", "ext": "xlsx"})
        assert resp["result"]["isError"] is False
        assert "budget_exhausted" not in resp["result"]["content"][0]["text"]
    # third non-terminal call is refused with a finalize directive (not an error)
    blocked = _call(server, "find_files", {"query": "train", "ext": "xlsx"})
    assert blocked["result"]["isError"] is False
    assert "budget_exhausted" in blocked["result"]["content"][0]["text"]
    assert "submit_answer" in blocked["result"]["content"][0]["text"]
    # submit_answer is never capped
    submit = _call(server, "submit_answer", {"answer": "わかりません"})
    assert submit["result"]["isError"] is False
    assert "budget_exhausted" not in submit["result"]["content"][0]["text"]


def test_missing_claude_binary_returns_model_error(monkeypatch, tmp_path):
    monkeypatch.setenv("RAG_CLAUDE_MCP_RESUME", "0")  # disable resume for isolation
    monkeypatch.setattr(claude_mcp.shutil, "which", lambda _b: None)
    out = claude_mcp.investigate_question("Q?", tools=build_tools(CorpusProfile()), contract=None)
    assert out.stop_reason == "model_error"
    assert out.answer.answer == inv.ABSTAIN
    assert "not on PATH" in (out.error or "")
