"""SOT-2626 — stdio MCP server exposing the investigator's read-only tool set.

Goal
----
A flat-rate ``claude -p --mcp-config <cfg>`` session cannot be handed the investigator's home-grown
tool schemas directly. The Model-Context-Protocol (MCP) is the injection point: ``claude`` speaks MCP
over stdio via ``--mcp-config``. This module turns the *existing* agent tool definitions
(:func:`src.rag.agent.investigator.build_tools`) into an MCP server so the same deterministic corpus
tools the live Gemini loop uses become callable from the flat-rate CLI. This is the dev-only tool
substrate for SOT-2627 (claude-mcp investigator backend); the production Gemini answer path is untouched.

Design invariants
-----------------
* **Single source of truth.** Tool ``name`` / ``description`` / ``inputSchema`` are derived from the very
  same :class:`~src.rag.agent.investigator.AgentTool` list that feeds ``to_genai_tools`` — no second
  schema is maintained here (:func:`tool_specs`).
* **Read-only.** Only ``build_tools`` is exposed; that set is corpus-read + restricted-pandas compute
  only. There is no filesystem-write or shell tool in it, and :func:`build_server` asserts the exposed
  set contains none (:data:`_WRITE_OR_SHELL_MARKERS`), so a future write tool cannot leak in silently.
* **Zero new dependencies.** A minimal JSON-RPC 2.0 stdio transport (newline-delimited messages) is
  implemented here rather than pulling in the ``mcp`` SDK, keeping this dev-only path dependency-free and
  the repo's ``requirements`` untouched.
* **serve path unchanged.** Nothing in the Gemini answer path imports this module.

Transport
---------
MCP stdio framing: each JSON-RPC message is a single UTF-8 line on stdout; all diagnostics go to stderr
so stdout stays pure protocol. Handled methods: ``initialize``, ``notifications/initialized``, ``ping``,
``tools/list``, ``tools/call``. Unknown methods return JSON-RPC error ``-32601``.
"""
from __future__ import annotations

import json
import os
import sys
import time
from typing import Any, Mapping, Sequence

from src.rag.agent.investigator import SUBMIT_ANSWER, AgentTool, build_tools, dispatch
from src.rag.tools.profile import CorpusProfile

SERVER_NAME = "signate-investigator"
SERVER_VERSION = "0.1.0"
# Protocol revision advertised when the client omits one. We echo the client's requested version when
# present (forward/backward compatible), otherwise fall back to this widely-supported revision.
DEFAULT_PROTOCOL_VERSION = "2024-11-05"

# A tool whose name contains any of these is a write/shell capability and must never be exposed over the
# read-only surface. ``build_tools`` contains none today; this is a defensive tripwire for the future.
_WRITE_OR_SHELL_MARKERS = ("write", "delete", "remove", "shell", "exec", "bash", "run_command", "subprocess")

# JSON-RPC error codes
_METHOD_NOT_FOUND = -32601
_INVALID_PARAMS = -32602
_INTERNAL_ERROR = -32603


def tool_specs(tools: Sequence[AgentTool]) -> list[dict[str, Any]]:
    """Convert :class:`AgentTool` schemas into MCP ``tools/list`` entries (single source of truth).

    The ``inputSchema`` is the tool's own JSON-Schema ``parameters`` object verbatim — the same object
    :func:`~src.rag.agent.investigator.to_genai_tools` reads — so the MCP surface and the live Gemini
    surface can never drift.
    """
    specs: list[dict[str, Any]] = []
    for t in tools:
        params = dict(t.parameters or {})
        params.setdefault("type", "object")
        params.setdefault("properties", {})
        specs.append({"name": t.name, "description": t.description, "inputSchema": params})
    return specs


class ToolCallLogger:
    """Append-only jsonl recorder of ``{ts, tool, args, elapsed_ms, ok}`` per ``tools/call``.

    Enables the SOT-2627 details.jsonl-compatible reconstruction of a Sonnet tool loop. Best-effort: a
    logging failure never breaks a tool call (diagnostics only).
    """

    def __init__(self, path: str | None) -> None:
        self.path = path

    def record(self, tool: str, args: Mapping[str, Any] | None, elapsed_ms: float,
               ok: bool, error: str | None = None) -> None:
        if not self.path:
            return
        rec = {
            "ts": time.time(),
            "tool": tool,
            "args": _safe_args(args),
            "elapsed_ms": round(elapsed_ms, 3),
            "ok": ok,
        }
        if error:
            rec["error"] = error
        try:
            os.makedirs(os.path.dirname(self.path) or ".", exist_ok=True)
            with open(self.path, "a", encoding="utf-8") as fh:
                fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
        except Exception as exc:  # noqa: BLE001 — logging must never break the call
            _log(f"tool-call log write failed: {exc}")


def _safe_args(args: Mapping[str, Any] | None) -> Any:
    try:
        json.dumps(args, ensure_ascii=False)
        return args
    except Exception:  # noqa: BLE001
        return {k: str(v) for k, v in dict(args or {}).items()}


class InvestigatorMCPServer:
    """A minimal, transport-agnostic MCP request handler over the investigator tool set.

    ``handle(message)`` maps one decoded JSON-RPC message to its response dict (or ``None`` for a
    notification). Kept transport-free so it is directly unit-testable without spawning a process.
    """

    def __init__(self, tools: Sequence[AgentTool], logger: ToolCallLogger | None = None,
                 max_tool_calls: int = 0) -> None:
        self.tools = list(tools)
        self.by_name = {t.name: t for t in self.tools}
        self.logger = logger or ToolCallLogger(None)
        # SOT-2627 — optional per-session budget: cap the number of *non-terminal* tool calls so the
        # flat-rate claude-mcp loop cannot spin unbounded (``claude`` has no --max-turns). 0 ⇒ disabled
        # (default), so the SOT-2626 MCP behaviour is byte-identical unless a budget is explicitly set.
        self.max_tool_calls = max(0, int(max_tool_calls or 0))
        self._non_terminal_calls = 0

    # -- request handlers ---------------------------------------------------
    def _on_initialize(self, params: Mapping[str, Any]) -> dict[str, Any]:
        client_version = params.get("protocolVersion") if isinstance(params, Mapping) else None
        return {
            "protocolVersion": client_version or DEFAULT_PROTOCOL_VERSION,
            "capabilities": {"tools": {"listChanged": False}},
            "serverInfo": {"name": SERVER_NAME, "version": SERVER_VERSION},
        }

    def _on_tools_list(self, _params: Mapping[str, Any]) -> dict[str, Any]:
        return {"tools": tool_specs(self.tools)}

    def _on_tools_call(self, params: Mapping[str, Any]) -> dict[str, Any]:
        name = params.get("name")
        args = params.get("arguments") or {}
        if not isinstance(name, str):
            raise _RpcError(_INVALID_PARAMS, "tools/call requires a string 'name'")
        if not isinstance(args, Mapping):
            raise _RpcError(_INVALID_PARAMS, "tools/call 'arguments' must be an object")
        # SOT-2627 — budget gate: once the non-terminal tool-call cap is reached, decline further
        # exploratory calls and steer the model to finalize via submit_answer (which is never capped).
        # Returned as a normal (non-error) tool result so the model reads and acts on it.
        if (self.max_tool_calls and name != SUBMIT_ANSWER
                and self._non_terminal_calls >= self.max_tool_calls):
            msg = (f"budget_exhausted: reached the {self.max_tool_calls}-call tool budget for this "
                   f"question. Do not call more exploratory tools — call {SUBMIT_ANSWER} now with your "
                   f"best grounded answer, or answer='わかりません' if no evidence was found.")
            self.logger.record(name, args, 0.0, ok=False, error="budget_exhausted")
            return {"content": [{"type": "text", "text": msg}], "isError": False}
        started = time.perf_counter()
        result = dispatch(self.by_name, name, args)
        if name != SUBMIT_ANSWER:
            self._non_terminal_calls += 1
        elapsed_ms = (time.perf_counter() - started) * 1000.0
        is_error = isinstance(result, Mapping) and "error" in result and name not in self.by_name
        self.logger.record(name, args, elapsed_ms, ok=not is_error,
                           error=(str(result.get("error")) if is_error else None))
        text = result if isinstance(result, str) else json.dumps(result, ensure_ascii=False)
        return {"content": [{"type": "text", "text": text}], "isError": bool(is_error)}

    def handle(self, message: Mapping[str, Any]) -> dict[str, Any] | None:
        """Dispatch one JSON-RPC message; return a response dict, or ``None`` for a notification."""
        method = message.get("method")
        msg_id = message.get("id")
        params = message.get("params") or {}
        is_notification = "id" not in message
        try:
            if method == "initialize":
                result: Any = self._on_initialize(params)
            elif method in ("notifications/initialized", "notifications/cancelled",
                            "initialized", "$/cancelRequest"):
                return None  # fire-and-forget notifications
            elif method == "ping":
                result = {}
            elif method == "tools/list":
                result = self._on_tools_list(params)
            elif method == "tools/call":
                result = self._on_tools_call(params)
            else:
                if is_notification:
                    return None
                return _error(msg_id, _METHOD_NOT_FOUND, f"method not found: {method}")
        except _RpcError as exc:
            if is_notification:
                return None
            return _error(msg_id, exc.code, exc.message)
        except Exception as exc:  # noqa: BLE001 — never crash the loop on a single bad message
            _log(f"handler error on {method}: {type(exc).__name__}: {exc}")
            if is_notification:
                return None
            return _error(msg_id, _INTERNAL_ERROR, f"{type(exc).__name__}: {exc}")
        if is_notification:
            return None
        return {"jsonrpc": "2.0", "id": msg_id, "result": result}


class _RpcError(Exception):
    def __init__(self, code: int, message: str) -> None:
        super().__init__(message)
        self.code = code
        self.message = message


def _error(msg_id: Any, code: int, message: str) -> dict[str, Any]:
    return {"jsonrpc": "2.0", "id": msg_id, "error": {"code": code, "message": message}}


def _log(msg: str) -> None:
    print(f"[mcp-investigator] {msg}", file=sys.stderr, flush=True)


def build_server(profile: CorpusProfile | None = None,
                 log_path: str | None = None,
                 max_tool_calls: int | None = None) -> InvestigatorMCPServer:
    """Build the server over ``build_tools`` and enforce the read-only invariant.

    Raises ``RuntimeError`` if any exposed tool name looks like a write/shell capability, so the
    read-only guarantee cannot regress silently.

    ``max_tool_calls`` (SOT-2627) caps the non-terminal tool calls per session; ``None`` reads the
    ``RAG_MCP_MAX_TOOL_CALLS`` env (0/unset ⇒ disabled = byte-identical SOT-2626 behaviour).
    """
    tools = build_tools(profile or CorpusProfile())
    for t in tools:
        low = t.name.lower()
        if any(marker in low for marker in _WRITE_OR_SHELL_MARKERS):
            raise RuntimeError(
                f"refusing to expose non-read-only tool over MCP: {t.name!r}")
    path = log_path if log_path is not None else os.getenv("RAG_MCP_TOOL_LOG") or None
    if max_tool_calls is None:
        try:
            max_tool_calls = int(os.getenv("RAG_MCP_MAX_TOOL_CALLS", "0") or 0)
        except ValueError:
            max_tool_calls = 0
    return InvestigatorMCPServer(tools, ToolCallLogger(path), max_tool_calls=max_tool_calls)


def serve_stdio(server: InvestigatorMCPServer, stdin: Any, stdout: Any) -> None:
    """Run the newline-delimited JSON-RPC stdio loop until stdin closes."""
    for raw in stdin:
        line = raw.strip()
        if not line:
            continue
        try:
            message = json.loads(line)
        except json.JSONDecodeError as exc:
            _log(f"skipping non-JSON line: {exc}")
            continue
        messages = message if isinstance(message, list) else [message]
        responses = []
        for m in messages:
            if not isinstance(m, Mapping):
                continue
            resp = server.handle(m)
            if resp is not None:
                responses.append(resp)
        for resp in responses:
            stdout.write(json.dumps(resp, ensure_ascii=False) + "\n")
            stdout.flush()


def main(argv: Sequence[str] | None = None) -> int:
    server = build_server()
    _log(f"ready: {len(server.tools)} tools "
         f"({', '.join(t.name for t in server.tools)})")
    serve_stdio(server, sys.stdin, sys.stdout)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
