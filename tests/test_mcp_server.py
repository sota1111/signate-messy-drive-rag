"""SOT-2626 — offline tests for the investigator stdio MCP server.

Network-free: they exercise the JSON-RPC handler and the tool-schema conversion directly, and drive one
real tool call (``find_files``) end-to-end through the MCP ``tools/call`` path to prove parity with the
investigator's own dispatch. No claude/Gemini process is spawned.
"""
from __future__ import annotations

import io
import json

import pytest

from src.rag.agent.investigator import AgentTool, build_tools, dispatch
from src.rag.tools.profile import CorpusProfile
from src.rag.mcp import server as mcp_server


def _srv(logger=None):
    return mcp_server.build_server(CorpusProfile(), log_path=None) if logger is None else logger


def test_tool_specs_single_source_of_truth():
    tools = build_tools(CorpusProfile())
    specs = mcp_server.tool_specs(tools)
    assert [s["name"] for s in specs] == [t.name for t in tools]
    for spec, tool in zip(specs, tools):
        # inputSchema IS the AgentTool.parameters object — no second schema is maintained.
        assert spec["inputSchema"]["type"] == "object"
        assert set(spec["inputSchema"].get("properties", {})) == set(
            tool.parameters.get("properties", {}))
        assert spec["description"] == tool.description


def test_build_server_is_read_only():
    server = mcp_server.build_server(CorpusProfile(), log_path=None)
    names = {t.name for t in server.tools}
    # the canonical read/compute tools are present; no write/shell tool leaks in.
    assert {"find_files", "file_grep", "read_office", "compute", "submit_answer"} <= names
    for name in names:
        assert not any(m in name.lower() for m in mcp_server._WRITE_OR_SHELL_MARKERS)


def test_build_server_rejects_write_tool(monkeypatch):
    """The read-only tripwire must reject a write/shell tool even if one is injected into build_tools."""
    poison = AgentTool("write_file", "danger", {"type": "object", "properties": {}}, lambda: None)
    monkeypatch.setattr(mcp_server, "build_tools", lambda profile: [poison])
    with pytest.raises(RuntimeError):
        mcp_server.build_server(CorpusProfile(), log_path=None)


def test_initialize_handshake_echoes_protocol():
    server = mcp_server.build_server(CorpusProfile(), log_path=None)
    resp = server.handle({"jsonrpc": "2.0", "id": 1, "method": "initialize",
                          "params": {"protocolVersion": "2025-06-18"}})
    assert resp["id"] == 1
    assert resp["result"]["protocolVersion"] == "2025-06-18"
    assert resp["result"]["capabilities"]["tools"] is not None
    assert resp["result"]["serverInfo"]["name"] == mcp_server.SERVER_NAME
    # default when the client omits a version
    resp2 = server.handle({"jsonrpc": "2.0", "id": 2, "method": "initialize", "params": {}})
    assert resp2["result"]["protocolVersion"] == mcp_server.DEFAULT_PROTOCOL_VERSION


def test_notifications_produce_no_response():
    server = mcp_server.build_server(CorpusProfile(), log_path=None)
    assert server.handle({"jsonrpc": "2.0", "method": "notifications/initialized"}) is None


def test_tools_list_matches_build_tools():
    server = mcp_server.build_server(CorpusProfile(), log_path=None)
    resp = server.handle({"jsonrpc": "2.0", "id": 3, "method": "tools/list", "params": {}})
    listed = [t["name"] for t in resp["result"]["tools"]]
    assert listed == [t.name for t in build_tools(CorpusProfile())]


def test_unknown_method_returns_method_not_found():
    server = mcp_server.build_server(CorpusProfile(), log_path=None)
    resp = server.handle({"jsonrpc": "2.0", "id": 9, "method": "does/not/exist", "params": {}})
    assert resp["error"]["code"] == mcp_server._METHOD_NOT_FOUND


def test_tools_call_parity_with_dispatch():
    profile = CorpusProfile()
    server = mcp_server.build_server(profile, log_path=None)
    args = {"query": "train", "ext": "xlsx"}
    resp = server.handle({"jsonrpc": "2.0", "id": 4, "method": "tools/call",
                          "params": {"name": "find_files", "arguments": args}})
    assert resp["result"]["isError"] is False
    text = resp["result"]["content"][0]["text"]
    # parity: the MCP text payload equals the investigator's own dispatch result, JSON-encoded.
    expected = dispatch({t.name: t for t in build_tools(profile)}, "find_files", args)
    parsed = json.loads(text) if not isinstance(expected, str) else text
    assert parsed == expected


def test_tools_call_unknown_tool_is_error():
    server = mcp_server.build_server(CorpusProfile(), log_path=None)
    resp = server.handle({"jsonrpc": "2.0", "id": 5, "method": "tools/call",
                          "params": {"name": "nope", "arguments": {}}})
    assert resp["result"]["isError"] is True
    payload = json.loads(resp["result"]["content"][0]["text"])
    assert "error" in payload


def test_tool_call_logger_writes_jsonl(tmp_path):
    log = tmp_path / "calls.jsonl"
    server = mcp_server.build_server(CorpusProfile(), log_path=str(log))
    server.handle({"jsonrpc": "2.0", "id": 6, "method": "tools/call",
                   "params": {"name": "find_files", "arguments": {"query": "train"}}})
    lines = log.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 1
    rec = json.loads(lines[0])
    assert rec["tool"] == "find_files"
    assert rec["ok"] is True
    assert "elapsed_ms" in rec and "ts" in rec


def test_serve_stdio_roundtrip():
    server = mcp_server.build_server(CorpusProfile(), log_path=None)
    stdin = io.StringIO(
        json.dumps({"jsonrpc": "2.0", "id": 1, "method": "initialize", "params": {}}) + "\n"
        + json.dumps({"jsonrpc": "2.0", "method": "notifications/initialized"}) + "\n"
        + json.dumps({"jsonrpc": "2.0", "id": 2, "method": "tools/list", "params": {}}) + "\n")
    stdout = io.StringIO()
    mcp_server.serve_stdio(server, stdin, stdout)
    out_lines = [json.loads(l) for l in stdout.getvalue().strip().splitlines()]
    # two requests → two responses; the notification produced none.
    assert [r["id"] for r in out_lines] == [1, 2]
    assert out_lines[1]["result"]["tools"]


def test_commit_gate_rejects_then_abstains_in_band(monkeypatch):
    monkeypatch.setenv("RAG_COMMIT_GATE", "1")
    monkeypatch.setenv("RAG_COMMIT_GATE_ENFORCE", "1")
    server = mcp_server.build_server(CorpusProfile(), log_path=None,
                                     question="合計は何件ですか?", contract="numeric")
    first = server.handle({"jsonrpc": "2.0", "id": 1, "method": "tools/call",
                           "params": {"name": "submit_answer", "arguments": {"answer": "42件"}}})
    text = first["result"]["content"][0]["text"]
    assert "commit_gate が回答を却下" in text
    assert server._commit_gate_rejects == 1

    second = server.handle({"jsonrpc": "2.0", "id": 2, "method": "tools/call",
                            "params": {"name": "submit_answer", "arguments": {"answer": "42件"}}})
    payload = json.loads(second["result"]["content"][0]["text"])
    assert payload["submitted"] is True
    assert payload["commit_gate"]["verdict"] == "ABSTAIN"
    assert payload["commit_gate"]["final_answer"] == "わかりません"


def test_commit_gate_off_submit_response_is_unchanged(monkeypatch):
    monkeypatch.delenv("RAG_COMMIT_GATE", raising=False)
    server = mcp_server.build_server(CorpusProfile(), log_path=None,
                                     question="合計は何件ですか?", contract="numeric")
    response = server.handle({"jsonrpc": "2.0", "id": 1, "method": "tools/call",
                              "params": {"name": "submit_answer", "arguments": {"answer": "42件"}}})
    assert json.loads(response["result"]["content"][0]["text"]) == {"submitted": True}
    assert server._commit_gate_rejects == 0


def test_commit_gate_accepts_numeric_value_grounded_by_compute(monkeypatch):
    monkeypatch.setenv("RAG_COMMIT_GATE", "1")
    monkeypatch.setenv("RAG_COMMIT_GATE_ENFORCE", "1")
    tools = [
        AgentTool("compute", "compute", {"type": "object", "properties": {}},
                  lambda **_kwargs: {"value": 42}),
        AgentTool("submit_answer", "submit", {"type": "object", "properties": {}},
                  lambda **_kwargs: {"submitted": True}),
    ]
    server = mcp_server.InvestigatorMCPServer(
        tools, question="合計は何件ですか?", contract="numeric")
    server.handle({"jsonrpc": "2.0", "id": 1, "method": "tools/call",
                   "params": {"name": "compute", "arguments": {}}})
    response = server.handle({"jsonrpc": "2.0", "id": 2, "method": "tools/call",
                              "params": {"name": "submit_answer", "arguments": {"answer": "42件"}}})
    payload = json.loads(response["result"]["content"][0]["text"])
    assert payload["commit_gate"]["verdict"] == "COMMIT"
    assert payload["commit_gate"]["final_answer"] == "42件"


# --- SOT-2664 — bounded plan-fanout finisher --------------------------------------------------------
def _finisher_tools():
    """A targeted reader (``read_office``), an exploratory tool (``file_grep``), and the terminal."""
    return [
        AgentTool("read_office", "read", {"type": "object", "properties": {}},
                  lambda **_kwargs: {"text": "ok"}),
        AgentTool("file_grep", "grep", {"type": "object", "properties": {}},
                  lambda **_kwargs: {"hits": []}),
        AgentTool("submit_answer", "submit", {"type": "object", "properties": {}},
                  lambda **_kwargs: {"submitted": True}),
    ]


def _call(server, name, args, msg_id=1):
    return server.handle({"jsonrpc": "2.0", "id": msg_id, "method": "tools/call",
                          "params": {"name": name, "arguments": args}})


def test_budget_gate_refuses_over_budget_without_finisher():
    """Baseline (finisher OFF): the first over-budget non-terminal call is refused, targeted or not."""
    server = mcp_server.InvestigatorMCPServer(_finisher_tools(), max_tool_calls=1, finisher_max=0)
    ok = _call(server, "read_office", {"file": "a.pptx"})
    assert ok["result"]["isError"] is False and "budget_exhausted" not in ok["result"]["content"][0]["text"]
    over = _call(server, "read_office", {"file": "b.pptx"})
    assert "budget_exhausted" in over["result"]["content"][0]["text"]
    assert server._finisher_used == 0


def test_finisher_allows_bounded_targeted_reads():
    """With the finisher ON, up to ``finisher_max`` extra *targeted* reads past the budget are allowed,
    then the gate refuses again — the extension is bounded, not a budget reset."""
    server = mcp_server.InvestigatorMCPServer(_finisher_tools(), max_tool_calls=1, finisher_max=2)
    assert "budget_exhausted" not in _call(server, "read_office", {"file": "a.pptx"})["result"]["content"][0]["text"]
    # calls #2 and #3 are over budget but targeted ⇒ granted by the finisher
    assert "budget_exhausted" not in _call(server, "read_office", {"file": "b.pptx"})["result"]["content"][0]["text"]
    assert "budget_exhausted" not in _call(server, "read_office", {"file": "c.pptx"})["result"]["content"][0]["text"]
    assert server._finisher_used == 2
    # the finisher budget is spent ⇒ the next over-budget read is refused
    assert "budget_exhausted" in _call(server, "read_office", {"file": "d.pptx"})["result"]["content"][0]["text"]


def test_finisher_refuses_exploratory_over_budget():
    """The wrong→abstain guard: an over-budget *exploratory* tool is refused even with the finisher ON,
    so the finisher can never re-open the wandering search that produces spurious answers."""
    server = mcp_server.InvestigatorMCPServer(_finisher_tools(), max_tool_calls=1, finisher_max=3)
    _call(server, "file_grep", {"query": "x"})  # consumes the budget
    over = _call(server, "file_grep", {"query": "y"})
    assert "budget_exhausted" in over["result"]["content"][0]["text"]
    assert server._finisher_used == 0


def test_finisher_refuses_targeted_read_without_resolved_target():
    """A targeted reader still searching for its file (no non-empty ``file`` arg) does not qualify."""
    server = mcp_server.InvestigatorMCPServer(_finisher_tools(), max_tool_calls=1, finisher_max=3)
    _call(server, "read_office", {"file": "a.pptx"})  # consumes the budget
    over = _call(server, "read_office", {"file": ""})
    assert "budget_exhausted" in over["result"]["content"][0]["text"]
    assert server._finisher_used == 0


def test_finisher_off_is_byte_identical_to_baseline():
    """finisher_max=0 ⇒ the over-budget targeted read is refused exactly as without the flag."""
    off = mcp_server.InvestigatorMCPServer(_finisher_tools(), max_tool_calls=1, finisher_max=0)
    _call(off, "read_office", {"file": "a.pptx"})
    refused = _call(off, "read_office", {"file": "b.pptx"})
    assert "budget_exhausted" in refused["result"]["content"][0]["text"]


def test_build_server_reads_finisher_env(monkeypatch):
    monkeypatch.delenv("RAG_FANOUT_FINISHER", raising=False)
    assert mcp_server.build_server(CorpusProfile(), log_path=None).finisher_max == 0
    monkeypatch.setenv("RAG_FANOUT_FINISHER", "1")
    monkeypatch.setenv("RAG_FANOUT_FINISHER_MAX", "4")
    assert mcp_server.build_server(CorpusProfile(), log_path=None).finisher_max == 4
