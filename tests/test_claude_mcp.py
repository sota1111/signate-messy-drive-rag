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


# -- planner -> parallel fan-out -> synthesis flow (SOT-2661) --------------------------------------
def test_plan_fanout_off_preserves_prompt_and_budget(monkeypatch):
    monkeypatch.delenv("RAG_PLAN_FANOUT", raising=False)
    assert inv.plan_fanout_enabled() is False
    assert inv.plan_fanout_budget(18) == 18
    assert inv._PROMPT_PLAN_FANOUT not in inv.SYSTEM_PROMPT


def test_plan_fanout_on_uses_five_nonterminal_calls_for_six_total(monkeypatch):
    monkeypatch.setenv("RAG_PLAN_FANOUT", "1")
    assert inv.plan_fanout_enabled() is True
    assert inv.plan_fanout_budget(18) == 5
    assert inv.plan_fanout_budget(4) == 4  # never loosen a tighter caller cap
    monkeypatch.setenv("RAG_PLAN_FANOUT_MAX_TURNS", "3")
    assert inv.plan_fanout_budget(18) == 3
    assert "分子・分母・最終式を1つの compute 式" in inv._PROMPT_PLAN_FANOUT
    assert "budget_exhausted" in inv._PROMPT_PLAN_FANOUT
    assert "要求外の補足を混ぜない" in inv._PROMPT_PLAN_FANOUT


def test_plan_fanout_telemetry_counts_search_and_supplements(monkeypatch):
    monkeypatch.delenv("RAG_FANOUT_FINISHER", raising=False)
    tel = claude_mcp._plan_fanout_telemetry(
        ["search", "metric_lookup", "file_grep", "submit_answer"], 5)
    assert tel == {
        "enabled": True,
        "budget": 5,
        "first_tool": "search",
        "search_first": True,
        "search_calls": 1,
        "supplement_calls": 2,
        "extra_searches": 0,
        "tool_turns": 3,
        "finisher_enabled": False,
        "finisher_max": 0,
        "finisher_calls": 0,
    }


# -- bounded plan-fanout finisher (SOT-2664) -------------------------------------------------------
def test_fanout_finisher_off_by_default(monkeypatch):
    monkeypatch.delenv("RAG_FANOUT_FINISHER", raising=False)
    assert inv.fanout_finisher_enabled() is False
    assert inv.fanout_finisher_budget() == 0
    # eligibility is a pure predicate — independent of the enabled flag (the server multiplies it in)
    assert inv.fanout_finisher_eligible("read_office", {"file": "a.pptx"}) is True


def test_fanout_finisher_budget_reads_env(monkeypatch):
    monkeypatch.setenv("RAG_FANOUT_FINISHER", "1")
    assert inv.fanout_finisher_budget() == inv.FANOUT_FINISHER_DEFAULT_MAX
    monkeypatch.setenv("RAG_FANOUT_FINISHER_MAX", "5")
    assert inv.fanout_finisher_budget() == 5
    monkeypatch.setenv("RAG_FANOUT_FINISHER_MAX", "0")  # invalid (must be >0) ⇒ default
    assert inv.fanout_finisher_budget() == inv.FANOUT_FINISHER_DEFAULT_MAX


def test_fanout_finisher_eligible_only_targeted_reads():
    # targeted raw-evidence readers with a resolved file target qualify
    assert inv.fanout_finisher_eligible("read_office", {"file": "b.pptx"}) is True
    assert inv.fanout_finisher_eligible("format_events", {"file": "m.xlsx", "fill": "黄"}) is True
    assert inv.fanout_finisher_eligible("compute", {"file": "t.csv", "expr": "df.x.sum()"}) is True
    # exploratory/search/resolve tools never qualify (the abstain→wrong guard)
    assert inv.fanout_finisher_eligible("file_grep", {"file": "x", "query": "q"}) is False
    assert inv.fanout_finisher_eligible("find_files", {"query": "q"}) is False
    assert inv.fanout_finisher_eligible("version_diff", {"question": "q"}) is False
    assert inv.fanout_finisher_eligible("canonical_route", {"question": "q"}) is False
    # a targeted reader still searching for its file (empty/missing target) does not qualify
    assert inv.fanout_finisher_eligible("read_office", {"file": ""}) is False
    assert inv.fanout_finisher_eligible("read_office", {}) is False


def test_plan_fanout_wires_cap_and_records_intervention(monkeypatch):
    monkeypatch.setenv("RAG_CLAUDE_MCP_RESUME", "0")
    monkeypatch.setenv("RAG_PLAN_FANOUT", "1")
    monkeypatch.setattr(claude_mcp.shutil, "which", lambda _b: "/usr/bin/claude")
    stdout = _stream(
        {"type": "assistant", "message": {"content": [{
            "type": "tool_use", "name": "mcp__investigator__search",
            "input": {"query": "Q"}}]}},
        {"type": "assistant", "message": {"content": [{
            "type": "tool_use", "name": "mcp__investigator__submit_answer",
            "input": {"answer": "42件", "confidence": 0.9,
                      "evidence": "doc.xlsx#row=2", "method": "search"}}]}},
        {"type": "result", "subtype": "success", "is_error": False, "result": "done"},
    )

    def fake_run(*args, **kwargs):
        cfg = json.loads(open(kwargs["cfg_path"], encoding="utf-8").read())
        env = cfg["mcpServers"]["investigator"]["env"]
        assert env["RAG_MCP_MAX_TOOL_CALLS"] == "5"
        return stdout, "", 0, False

    monkeypatch.setattr(claude_mcp, "_run_claude", fake_run)
    out = claude_mcp.investigate_question(
        "Q", tools=build_tools(CorpusProfile()), contract="numeric", max_turns=18)
    assert out.answer.answer == "42件"
    assert out.interventions["plan_fanout"]["search_first"] is True
    assert out.interventions["plan_fanout"]["tool_turns"] == 1
    assert out.interventions["plan_fanout"]["supplement_calls"] == 0


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


def test_plain_final_text_commit_gate_rejects_to_abstain(monkeypatch):
    monkeypatch.setenv("RAG_CLAUDE_MCP_RESUME", "0")
    monkeypatch.setenv("RAG_COMMIT_GATE", "1")
    monkeypatch.setenv("RAG_COMMIT_GATE_ENFORCE", "1")
    monkeypatch.setattr(claude_mcp.shutil, "which", lambda _b: "/usr/bin/claude")
    stdout = _stream({"type": "result", "subtype": "success", "is_error": False,
                      "result": "42件"})
    monkeypatch.setattr(claude_mcp, "_run_claude", lambda *a, **k: (stdout, "", 0, False))
    out = claude_mcp.investigate_question(
        "合計は何件ですか?", tools=build_tools(CorpusProfile()), contract="numeric")
    assert out.answer.answer == inv.ABSTAIN
    assert out.interventions["commit_gate"]["verdict"] == "REJECT"


def test_submit_honors_server_commit_gate_decision(monkeypatch, tmp_path):
    monkeypatch.setenv("RAG_CLAUDE_MCP_RESUME", "0")
    monkeypatch.setenv("RAG_COMMIT_GATE", "1")
    monkeypatch.setenv("RAG_COMMIT_GATE_ENFORCE", "1")
    monkeypatch.setattr(claude_mcp.shutil, "which", lambda _b: "/usr/bin/claude")
    stdout = _stream({"type": "assistant", "message": {"content": [{
        "type": "tool_use", "name": "mcp__investigator__submit_answer",
        "input": {"answer": "42件", "confidence": 0.9}}]}},
        {"type": "result", "subtype": "success", "is_error": False, "result": "done"})

    def fake_run(*args, **kwargs):
        cfg = json.loads(open(kwargs["cfg_path"], encoding="utf-8").read())
        path = cfg["mcpServers"]["investigator"]["env"]["RAG_MCP_COMMIT_GATE_LOG"]
        with open(path, "w", encoding="utf-8") as fh:
            fh.write(json.dumps({"commit_gate": {
                "verdict": "ABSTAIN", "final_answer": inv.ABSTAIN,
                "reasons": ["reject_streak_abstain"],
                "telemetry": {"enabled": True, "verdict": "ABSTAIN"}}}) + "\n")
        return stdout, "", 0, False

    monkeypatch.setattr(claude_mcp, "_run_claude", fake_run)
    out = claude_mcp.investigate_question(
        "合計は何件ですか?", tools=build_tools(CorpusProfile()), contract="numeric")
    assert out.answer.answer == inv.ABSTAIN
    assert out.answer.confidence == 0.0
    assert out.interventions["commit_gate"]["verdict"] == "ABSTAIN"


def test_observational_gate_records_telemetry_without_changing_submit(monkeypatch):
    monkeypatch.setenv("RAG_CLAUDE_MCP_RESUME", "0")
    monkeypatch.setenv("RAG_COMMIT_GATE", "1")
    monkeypatch.delenv("RAG_COMMIT_GATE_ENFORCE", raising=False)
    monkeypatch.setattr(claude_mcp.shutil, "which", lambda _b: "/usr/bin/claude")
    stdout = _stream({"type": "assistant", "message": {"content": [{
        "type": "tool_use", "name": "mcp__investigator__submit_answer",
        "input": {"answer": "42件", "confidence": 0.9}}]}},
        {"type": "result", "subtype": "success", "is_error": False, "result": "done"})

    def fake_run(*args, **kwargs):
        cfg = json.loads(open(kwargs["cfg_path"], encoding="utf-8").read())
        path = cfg["mcpServers"]["investigator"]["env"]["RAG_MCP_COMMIT_GATE_LOG"]
        with open(path, "w", encoding="utf-8") as fh:
            fh.write(json.dumps({"commit_gate": {
                "verdict": "REJECT", "final_answer": "42件", "reasons": ["ungrounded"],
                "telemetry": {"enabled": True, "verdict": "REJECT"}}}) + "\n")
        return stdout, "", 0, False

    monkeypatch.setattr(claude_mcp, "_run_claude", fake_run)
    out = claude_mcp.investigate_question(
        "合計は何件ですか?", tools=build_tools(CorpusProfile()), contract="numeric")
    assert out.answer.answer == "42件"
    assert out.interventions["commit_gate"]["verdict"] == "REJECT"


# --------------------------------------------------------------------------- SOT-2665 「該当なし」裸形式契約
def test_none_bare_contract_absent_by_default(monkeypatch):
    # Default OFF ⇒ the system suffix carries no 該当なし contract (serve byte-identical).
    monkeypatch.delenv("RAG_NONE_BARE", raising=False)
    assert "該当なし契約" not in claude_mcp._harness_system_suffix()


def test_none_bare_contract_appended_when_flag_on(monkeypatch):
    # SOT-2665 (idx9/85): flag ON ⇒ the suffix binds a bare「該当なし」none-answer form.
    monkeypatch.setenv("RAG_NONE_BARE", "1")
    suffix = claude_mcp._harness_system_suffix()
    assert "該当なし契約" in suffix
    assert "『該当なし』のみ" in suffix
    # composes with the base bare-answer contract when both are on (base config keeps RAG_BARE_ANSWER=1)
    monkeypatch.setenv("RAG_BARE_ANSWER", "1")
    both = claude_mcp._harness_system_suffix()
    assert "回答書式契約" in both and "該当なし契約" in both
