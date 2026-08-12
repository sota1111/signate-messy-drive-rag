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

from src.rag.agent.investigator import (
    SUBMIT_ANSWER, AgentTool, build_tools, dispatch, is_raw_file_tool, scored_answer_text)
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

# SOT-2640 — in-band feedback returned as the submit_answer tool result when the shared commit gate
# REJECTs a submitted answer. The flat-rate Sonnet loop reads this (a normal, non-error tool result) and
# re-derives + re-submits, exactly as the Gemini loop's own ``answer_rejected`` retry channel does. The
# retry is bounded: the gate itself degrades to ABSTAIN after RAG_COMMIT_GATE_ABSTAIN_AFTER rejects.
_COMMIT_GATE_RETRY_DIRECTIVE = (
    "commit_gate が回答を却下しました。数値回答は compute / corpus_aggregate で値を実際に導出・検算してから、"
    "列挙の『該当なし』は母集団を全数確認してから submit_answer を呼び直してください。"
    "根拠を実際に取得できない場合のみ answer=「わかりません」で submit してください。")


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
            # SOT-2660 — mark 生ファイル系 calls so the details-reconstruction can compute this session's
            # raw_file_access (a raw-file tool with ok=True means the answer touched a raw corpus file; a
            # RAG_DB_ONLY refusal lands here with ok=False).
            "raw_file": is_raw_file_tool(tool),
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
                 max_tool_calls: int = 0, *, question: str = "", contract: str | None = None,
                 commit_gate_log: str | None = None) -> None:
        self.tools = list(tools)
        self.by_name = {t.name: t for t in self.tools}
        self.logger = logger or ToolCallLogger(None)
        # SOT-2627 — optional per-session budget: cap the number of *non-terminal* tool calls so the
        # flat-rate claude-mcp loop cannot spin unbounded (``claude`` has no --max-turns). 0 ⇒ disabled
        # (default), so the SOT-2626 MCP behaviour is byte-identical unless a budget is explicitly set.
        self.max_tool_calls = max(0, int(max_tool_calls or 0))
        self._non_terminal_calls = 0
        # SOT-2640 — commit-gate session state. ``submit_answer`` is the gate's execution point for the
        # guard-less claude-mcp backend: the question/contract identify the commit, ``_tool_history`` is
        # the session's successful tool records the gate grounds numerics against, and ``_commit_gate_rejects``
        # counts consecutive rejects so the in-band retry degrades to ABSTAIN after the threshold. All of
        # this is inert unless RAG_COMMIT_GATE (+ RAG_COMMIT_GATE_ENFORCE) is set ⇒ OFF byte-identical.
        self.question = str(question or "")
        self.contract = contract
        self.commit_gate_log = commit_gate_log
        self._tool_history: list[dict[str, Any]] = []
        self._commit_gate_rejects = 0

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
        # SOT-2640 — commit gate at the submit_answer execution point. Returns a response dict when the gate
        # is active AND governs this submit (REJECT retry feedback, or a COMMIT/ABSTAIN marker for the
        # client to honor); returns None when the gate is OFF/observational so the normal dispatch below runs
        # unchanged (⇒ OFF byte-identical).
        if name == SUBMIT_ANSWER:
            gate_resp = self._commit_gate_submit(args)
            if gate_resp is not None:
                return gate_resp
        started = time.perf_counter()
        result = dispatch(self.by_name, name, args)
        if name != SUBMIT_ANSWER:
            self._non_terminal_calls += 1
            # SOT-2640 — record the successful tool outcome so the gate can ground a later numeric commit
            # against a real compute/corpus_aggregate value (mirrors the Gemini loop's ``tool_history``).
            is_err = isinstance(result, Mapping) and "error" in result and name not in self.by_name
            self._tool_history.append({"name": name, "response": result, "ok": not is_err})
        elapsed_ms = (time.perf_counter() - started) * 1000.0
        is_error = isinstance(result, Mapping) and "error" in result and name not in self.by_name
        self.logger.record(name, args, elapsed_ms, ok=not is_error,
                           error=(str(result.get("error")) if is_error else None))
        text = result if isinstance(result, str) else json.dumps(result, ensure_ascii=False)
        return {"content": [{"type": "text", "text": text}], "isError": bool(is_error)}

    def _commit_gate_submit(self, args: Mapping[str, Any]) -> dict[str, Any] | None:
        """Run the shared commit gate at ``submit_answer`` (SOT-2640). Backend-agnostic commit judgment for
        the guard-less claude-mcp loop, wired identically to the Gemini finalization (SOT-2639).

        Returns ``None`` when the gate is inert for this submit — RAG_COMMIT_GATE OFF (byte-identical) or
        ON-but-observational (record telemetry, let the raw value commit) — so the caller's normal dispatch
        proceeds. Returns a tool-result dict when the gate governs the submit:

        * ``REJECT`` (retry streak below threshold) → non-error feedback text; the submit does NOT finalize
          and the model re-derives + re-submits in-band. The consecutive-reject counter advances.
        * ``COMMIT`` / ``ABSTAIN`` → a terminal marker (``{"submitted": true, "commit_gate": {...}}``) whose
          ``final_answer`` the client honors (formatted value on COMMIT, ABSTAIN on the 棄権 degrade).
        """
        from src.rag.agent import commit_gate as _commit_gate  # lazy — avoids the exec_verifier cycle
        if not _commit_gate.enabled():
            return None  # OFF ⇒ byte-identical
        # SOT-2670 — the gate validates the SCORED answer, which is bare_answer when RAG_TWO_TIER_ANSWER
        # is ON (else the legacy ``answer`` field, verbatim ⇒ byte-identical).
        submitted = scored_answer_text(args)
        decision = _commit_gate.evaluate(
            self.question, self.contract, submitted,
            session_tool_history=self._tool_history,
            prior_rejects=self._commit_gate_rejects,
        )
        self._log_commit_gate(decision)
        if not _commit_gate.enforce():
            return None  # observational: record telemetry, keep the raw commit (equivalence)
        if decision.verdict == _commit_gate.REJECT:
            self._commit_gate_rejects += 1
            reason = "; ".join(decision.reasons)
            msg = _COMMIT_GATE_RETRY_DIRECTIVE + (f" (却下理由: {reason})" if reason else "")
            self.logger.record(SUBMIT_ANSWER, args, 0.0, ok=False, error="commit_gate_reject")
            return {"content": [{"type": "text", "text": msg}], "isError": False}
        # COMMIT or ABSTAIN — terminal. Embed the gate decision so the client serves final_answer.
        payload = {"submitted": True, "commit_gate": decision.to_dict()}
        self.logger.record(SUBMIT_ANSWER, args, 0.0, ok=True, error=None)
        return {"content": [{"type": "text", "text": json.dumps(payload, ensure_ascii=False)}],
                "isError": False}

    def _log_commit_gate(self, decision: Any) -> None:
        """Best-effort append of the commit-gate decision telemetry (SOT-2629 形式) to a jsonl sink; a
        logging failure never affects the submit."""
        if not self.commit_gate_log:
            return
        try:
            rec = {"ts": time.time(), "commit_gate": decision.to_dict()}
            os.makedirs(os.path.dirname(self.commit_gate_log) or ".", exist_ok=True)
            with open(self.commit_gate_log, "a", encoding="utf-8") as fh:
                fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
        except Exception as exc:  # noqa: BLE001 — telemetry must never break the submit
            _log(f"commit-gate log write failed: {exc}")

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
                 max_tool_calls: int | None = None,
                 *, question: str | None = None, contract: str | None = None,
                 commit_gate_log: str | None = None) -> InvestigatorMCPServer:
    """Build the server over ``build_tools`` and enforce the read-only invariant.

    Raises ``RuntimeError`` if any exposed tool name looks like a write/shell capability, so the
    read-only guarantee cannot regress silently.

    ``max_tool_calls`` (SOT-2627) caps the non-terminal tool calls per session; ``None`` reads the
    ``RAG_MCP_MAX_TOOL_CALLS`` env (0/unset ⇒ disabled = byte-identical SOT-2626 behaviour).

    ``question`` / ``contract`` / ``commit_gate_log`` (SOT-2640) supply the commit-gate context; ``None``
    reads ``RAG_MCP_QUESTION`` / ``RAG_MCP_CONTRACT`` / ``RAG_MCP_COMMIT_GATE_LOG`` (unset ⇒ empty, and the
    gate stays inert unless RAG_COMMIT_GATE is also set).
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
    q = question if question is not None else (os.getenv("RAG_MCP_QUESTION") or "")
    c = contract if contract is not None else (os.getenv("RAG_MCP_CONTRACT") or None)
    cg_log = commit_gate_log if commit_gate_log is not None else (os.getenv("RAG_MCP_COMMIT_GATE_LOG") or None)
    return InvestigatorMCPServer(tools, ToolCallLogger(path), max_tool_calls=max_tool_calls,
                                 question=q, contract=c, commit_gate_log=cg_log)


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
