"""claude-mcp investigator backend (SOT-2627) — run one question's whole tool loop on flat-rate Sonnet.

Where :mod:`src.rag.llm_providers.claude_cli` (SOT-2606) only covers *tool-free text* roles, this module
delegates the investigator's **entire function-calling loop** — tool round-trips and the final answer —
to a flat-rate ``claude -p --mcp-config`` session driving the SOT-2626 stdio MCP server
(:mod:`src.rag.mcp.server`). The 2026-08-10 smoke proved ``LLM_PROVIDER=claude-cli`` alone runs the
gold100 loop with **zero** Sonnet calls (the loop is genai-fixed at ``investigator.py:2316``); the only
way to move that loop off Gemini metering is to let ``claude`` itself drive the tools over MCP.

**Dev-only.** Selected by ``RAG_INVESTIGATOR_BACKEND=claude-mcp`` (default ``gemini``). The production
answer path stays Gemini-only (SOT-2460); official measurement stays on flash 3.6 (SOT-2625). The
flat-rate usage limit is **shared account-wide with the autonomous workers**, so this throttles them —
run it only when workers are idle, at parallelism 1 (the investigator run default for this backend).

What it produces
----------------
A :class:`~src.rag.agent.investigator.Investigation` byte-compatible with the details.jsonl schema
(``index`` is added by :mod:`src.rag.run`), so ``scoring.gold_offline --answers/--run`` scores it exactly
like the Gemini backend. ``model`` is reported as ``"sonnet(claude-mcp)"`` and ``cost_usd`` is 0 (flat
rate — see :meth:`Usage.cost_usd`, which zeroes any ``(claude-mcp)`` model).

Budget & interruption
---------------------
``claude`` exposes no ``--max-turns`` (checked, CLI 2.1.x), so the per-question budget is enforced two
ways: a subprocess wall-clock ``timeout`` (= ``timeout_s``) and a server-side tool-call cap
(``RAG_MCP_MAX_TOOL_CALLS`` = ``max_turns``, passed through the generated mcp-config). On a detected
flat-rate **usage-limit** the run trips a process-wide latch so every remaining question short-circuits
to an abstain immediately (no further ``claude`` calls hammer the shared limit); answered questions are
persisted to a resume sidecar so a later re-run continues where it stopped.
"""
from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
import tempfile
import threading
import time
from pathlib import Path
from typing import Any, Mapping, Sequence

from src.rag.agent.investigator import (
    ABSTAIN,
    SUBMIT_ANSWER,
    SYSTEM_PROMPT,
    AgentTool,
    Answer,
    Investigation,
    Usage,
    _answer_from_args,
    fanout_finisher_budget,
    fanout_finisher_enabled,
    is_abstain,
    plan_fanout_budget,
    plan_fanout_enabled,
)

CLAUDE_BIN = "claude"
MCP_SERVER_NAME = "investigator"          # the --mcp-config key ⇒ tools surface as mcp__investigator__*
DEFAULT_MODEL = "sonnet"                   # flat-rate Sonnet; RAG_CLAUDE_MCP_MODEL overrides
_MODEL_TAG = "claude-mcp"                  # model label suffix ⇒ Usage.cost_usd zeroes the flat-rate cost

# Built-in claude tools we never want the investigator loop to use — the loop must go through the MCP
# corpus tools only (parity with the Gemini function-calling surface). Disallowed tools are auto-denied
# in non-interactive ``-p`` mode, so claude routes to the MCP tools instead.
_DISALLOWED_BUILTINS = (
    "Bash", "Read", "Edit", "Write", "NotebookEdit", "WebFetch", "WebSearch",
    "Glob", "Grep", "Task", "TodoWrite",
)

# --- process-wide usage-limit latch (SOT-2627 §4) -------------------------------------------------
# Once a flat-rate usage limit is seen, every remaining question in the batch short-circuits instead of
# issuing more ``claude`` calls that would only burn the shared account limit faster.
_USAGE_LIMIT = threading.Event()
_RESUME_LOCK = threading.Lock()


class UsageLimitError(RuntimeError):
    """Raised internally when the flat-rate Claude usage limit is detected (see :func:`_is_usage_limit`)."""


def usage_limit_tripped() -> bool:
    """Whether a flat-rate usage limit has been detected in this process (test/inspection hook)."""
    return _USAGE_LIMIT.is_set()


def reset_usage_limit() -> None:
    """Clear the process-wide usage-limit latch (used by tests / a deliberate fresh run)."""
    _USAGE_LIMIT.clear()


# --- resume sidecar (SOT-2627 §4) -----------------------------------------------------------------
def _resume_path() -> Path | None:
    """Resolve the resume-sidecar path, or ``None`` when resume is disabled.

    ``RAG_CLAUDE_MCP_RESUME`` unset ⇒ default ``<ARTIFACTS_DIR>/claude_mcp_resume.jsonl``; an explicit
    path redirects it; ``0``/``off``/``false``/``no``/empty disables the sidecar entirely.
    """
    raw = os.getenv("RAG_CLAUDE_MCP_RESUME")
    if raw is not None:
        low = raw.strip().lower()
        if low in {"", "0", "off", "false", "no"}:
            return None
        return Path(raw)
    from config import settings
    return settings.ARTIFACTS_DIR / "claude_mcp_resume.jsonl"


def _resume_key(question: str, model: str) -> str:
    return hashlib.sha256(f"{model}\x00{question}".encode("utf-8")).hexdigest()


def _load_resume(path: Path) -> dict[str, dict]:
    out: dict[str, dict] = {}
    if not path.exists():
        return out
    try:
        for line in path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            key = rec.get("key")
            if key:
                out[key] = rec
    except Exception:  # noqa: BLE001 — a corrupt sidecar must never break a run; just ignore it
        return out
    return out


def _append_resume(path: Path, key: str, question: str, inv: Investigation) -> None:
    rec = {"key": key, "question": question, "record": _investigation_to_record(inv)}
    try:
        with _RESUME_LOCK:
            path.parent.mkdir(parents=True, exist_ok=True)
            with path.open("a", encoding="utf-8") as fh:
                fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
    except Exception:  # noqa: BLE001 — persistence is best-effort; a write failure only costs resumability
        pass


def _investigation_to_record(inv: Investigation) -> dict[str, Any]:
    """Compact, self-contained serialization sufficient to rebuild the :class:`Investigation`."""
    return {
        "question": inv.question,
        "answer": inv.answer.answer,
        "confidence": inv.answer.confidence,
        "evidence": inv.answer.evidence,
        "method": inv.answer.method,
        "iterations": inv.iterations,
        "tool_calls": list(inv.tool_calls),
        "input_tokens": inv.usage.input_tokens,
        "output_tokens": inv.usage.output_tokens,
        "model": inv.model,
        "elapsed_s": inv.elapsed_s,
        "stop_reason": inv.stop_reason,
        "error": inv.error,
        "contract": inv.contract,
    }


def _record_to_investigation(rec: Mapping[str, Any]) -> Investigation:
    return Investigation(
        question=str(rec.get("question", "")),
        answer=Answer(
            answer=str(rec.get("answer", ABSTAIN) or ABSTAIN),
            confidence=float(rec.get("confidence", 0.0) or 0.0),
            evidence=str(rec.get("evidence", "") or ""),
            method=str(rec.get("method", "") or ""),
        ),
        iterations=int(rec.get("iterations", 0) or 0),
        tool_calls=list(rec.get("tool_calls", []) or []),
        usage=Usage(int(rec.get("input_tokens", 0) or 0), int(rec.get("output_tokens", 0) or 0)),
        model=str(rec.get("model", f"{DEFAULT_MODEL}({_MODEL_TAG})")),
        elapsed_s=float(rec.get("elapsed_s", 0.0) or 0.0),
        stop_reason=str(rec.get("stop_reason", "answered")),
        error=rec.get("error"),
        contract=rec.get("contract"),
    )


# --- claude invocation ----------------------------------------------------------------------------
def _repo_root() -> Path:
    # src/rag/llm_providers/claude_mcp.py → repo root is three parents up from the package dir.
    return Path(__file__).resolve().parents[3]


def _mcp_config(tool_log: str, max_tool_calls: int,
                question: str = "", contract: str | None = None,
                commit_gate_log: str | None = None) -> dict[str, Any]:
    """Build a --mcp-config dict launching the SOT-2626 stdio server with per-question budget/log env.

    SOT-2640: the question/contract are forwarded so the server-side commit gate (submit_answer execution
    point) has the commit context. They are inert unless RAG_COMMIT_GATE is set in the ambient env, which
    the launched server process inherits (the CLI merges this ``env`` over the parent environment)."""
    launcher = str(_repo_root() / "scripts" / "mcp_investigator_server.sh")
    env = {"RAG_MCP_TOOL_LOG": tool_log}
    if max_tool_calls and max_tool_calls > 0:
        env["RAG_MCP_MAX_TOOL_CALLS"] = str(int(max_tool_calls))
    # SOT-2664 — carry the bounded finisher config alongside the per-question budget so the launched
    # server enables it deterministically (does not rely on ambient-env inheritance). OFF ⇒ keys omitted
    # ⇒ byte-identical config.
    _fin = fanout_finisher_budget()
    if _fin > 0:
        env["RAG_FANOUT_FINISHER"] = "1"
        env["RAG_FANOUT_FINISHER_MAX"] = str(int(_fin))
    if question:
        env["RAG_MCP_QUESTION"] = question
    if contract:
        env["RAG_MCP_CONTRACT"] = str(contract)
    if commit_gate_log:
        env["RAG_MCP_COMMIT_GATE_LOG"] = commit_gate_log
    return {"mcpServers": {MCP_SERVER_NAME: {"command": "bash", "args": [launcher], "env": env}}}


def _allowed_tools(tools: Sequence[AgentTool]) -> list[str]:
    return [f"mcp__{MCP_SERVER_NAME}__{t.name}" for t in tools]


def _bare_answer_enabled() -> bool:
    """裸回答書式契約 (cycle4 調査: wrong 21 のうち 12 件が「値は正しいが説明文/注記/記法で judge 落ち」)。

    プロンプトのみの契約 — 値の後処理は一切行わない (値保存)。既定 OFF。
    """
    return os.getenv("RAG_BARE_ANSWER", "0").strip().lower() in ("1", "true", "yes", "on")


_BARE_ANSWER_CONTRACT = (
    "\n\n【回答書式契約】answer フィールドには質問が要求する値・語句そのものだけを書く(裸の値):"
    "理由・計算過程・出典・セル位置・補足説明・括弧書きの注記は answer に一切含めず、evidence フィールドへ書く。"
    "例:『差額は0円(…のため)』ではなく『0円』/『13ページ(スライド13)「8. 費用見積」』ではなく『13ページ』。"
    "数値は原文・計算結果の精度をそのまま書き、勝手に丸めたり『約』を付けたりしない。"
    "範囲は『A ~ B』の形式で書く(区間記法 (a, b] 等は使わない)。"
    "一覧を要求されたら項目のみを読点区切りで列挙し、各項目に説明を付けない。"
)


def _none_bare_enabled() -> bool:
    """SOT-2665 (cycle5 C2) — 「該当なし」裸形式契約。

    cycle5 の wrong のうち idx9/85 は gold=「該当なし」なのに、変更点を列挙(idx9)したり
    「なし(全6項目達成)」と達成状況を注記(idx85)して judge 落ちした。存在しないことが確定した回答は
    装飾なしの「該当なし」のみに縛る。プロンプトのみ(値の後処理なし)・既定 OFF。
    """
    return os.getenv("RAG_NONE_BARE", "0").strip().lower() in ("1", "true", "yes", "on")


_NONE_BARE_CONTRACT = (
    "\n\n【該当なし契約】質問が要求する項目・変更・値が確定的に存在しない/見つからない場合、"
    "answer は装飾なしで『該当なし』のみとする(『なし』『特になし』ではなく『該当なし』)。"
    "『なし(全6項目達成)』のような理由・達成状況・件数の注記や、"
    "存在した別項目の列挙・言い換えを answer に一切付けない(補足は evidence へ書く)。"
)


def _two_tier_answer_enabled() -> bool:
    """SOT-2670 — two-tier submit_answer schema (full_answer + bare_answer). Prompt-side mirror of the
    structural flag defined in ``investigator`` so the tool description AND the system prompt both steer
    the model toward bare_answer as the scored value. Default OFF."""
    return os.getenv("RAG_TWO_TIER_ANSWER", "0").strip().lower() in ("1", "true", "yes", "on")


_TWO_TIER_ANSWER_CONTRACT = (
    "\n\n【二段回答契約】submit_answer には full_answer と bare_answer を必ず両方渡す。"
    "full_answer には根拠・限定条件を含む完全な文を書いて『考える場所』とし、bare_answer には"
    "質問が要求する値・語句そのものだけを書く(採点対象は bare_answer)。bare_answer では"
    "ラベル・肩書・単位・記法を原文どおり完全に写し、切り詰め・言い換え・注記の付加をしない。"
    "存在しないことが確定した場合の bare_answer は『該当なし』のみとする。"
)


def _harness_system_suffix() -> str:
    base = (
        "\n\n【実行環境(dev)】ツールは MCP 経由で `mcp__investigator__<ツール名>` として提供される"
        "(例: `mcp__investigator__file_grep`, `mcp__investigator__compute`)。これら以外のツールは使わない。"
        f"最終回答は必ず `mcp__investigator__{SUBMIT_ANSWER}` を1回だけ呼んで確定する"
        "(通常の文章では答えない)。根拠が得られない場合のみ answer=「" + ABSTAIN + "」で submit する。"
    )
    if _bare_answer_enabled():
        base += _BARE_ANSWER_CONTRACT
    if _none_bare_enabled():
        base += _NONE_BARE_CONTRACT
    if _two_tier_answer_enabled():
        base += _TWO_TIER_ANSWER_CONTRACT
    return base


def _is_usage_limit(text: str) -> bool:
    """Heuristic detection of a flat-rate Claude usage/rate limit from CLI output."""
    t = (text or "").lower()
    needles = (
        "usage limit", "rate limit", "rate_limit", "limit reached", "too many requests",
        "429", "resets at", "quota", "overloaded_error", "usage_limit",
    )
    return any(n in t for n in needles)


def _run_claude(prompt: str, *, system: str, model: str, cfg_path: str,
                allowed: Sequence[str], timeout: float) -> tuple[str, str, int, bool]:
    """Run one ``claude -p`` session; return (stdout, stderr, returncode, timed_out)."""
    cmd = [
        CLAUDE_BIN, "-p",
        "--model", model,
        "--mcp-config", cfg_path,
        "--strict-mcp-config",
        "--output-format", "stream-json",
        "--verbose",
        "--append-system-prompt", system,
        "--allowedTools", *allowed,
        "--disallowedTools", *_DISALLOWED_BUILTINS,
    ]
    try:
        proc = subprocess.run(
            cmd, input=prompt, capture_output=True, text=True, timeout=timeout,
        )
        return proc.stdout or "", proc.stderr or "", proc.returncode, False
    except subprocess.TimeoutExpired as exc:
        out = exc.stdout or ""
        err = exc.stderr or ""
        if isinstance(out, bytes):
            out = out.decode("utf-8", "replace")
        if isinstance(err, bytes):
            err = err.decode("utf-8", "replace")
        return out, err, -1, True


def _parse_stream_json(stdout: str) -> dict[str, Any]:
    """Parse claude ``stream-json`` JSONL into the fields we need to rebuild an Investigation.

    Returns ``tool_calls`` (bare tool names, in order), ``submit_args`` (the last submit_answer input, or
    None), ``final_text`` (final assistant text), ``usage`` (input/output tokens), ``is_error``,
    ``subtype``, and ``result_text`` (the terminal result message text, if any).
    """
    tool_calls: list[str] = []
    submit_args: dict[str, Any] | None = None
    final_text = ""
    input_tokens = 0
    output_tokens = 0
    is_error = False
    subtype = ""
    result_text = ""
    prefix = f"mcp__{MCP_SERVER_NAME}__"
    for line in stdout.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            evt = json.loads(line)
        except json.JSONDecodeError:
            continue
        etype = evt.get("type")
        if etype == "assistant":
            msg = evt.get("message") or {}
            for block in msg.get("content") or []:
                if not isinstance(block, Mapping):
                    continue
                if block.get("type") == "tool_use":
                    name = str(block.get("name") or "")
                    # Record ONLY investigator MCP tools (the mcp__investigator__* surface). Claude's own
                    # built-ins (e.g. ToolSearch used for schema discovery) are excluded so tool_calls /
                    # iterations reflect the same investigator-tool loop the Gemini trace does — a fair
                    # reachability comparison, not a claude-harness artifact.
                    if not name.startswith(prefix):
                        continue
                    bare = name[len(prefix):]
                    tool_calls.append(bare)
                    if bare == SUBMIT_ANSWER:
                        inp = block.get("input")
                        submit_args = dict(inp) if isinstance(inp, Mapping) else {}
                elif block.get("type") == "text":
                    final_text = str(block.get("text") or "") or final_text
            usage = msg.get("usage") or {}
            input_tokens += int(usage.get("input_tokens", 0) or 0)
            output_tokens += int(usage.get("output_tokens", 0) or 0)
        elif etype == "result":
            is_error = bool(evt.get("is_error"))
            subtype = str(evt.get("subtype") or "")
            rt = evt.get("result")
            if isinstance(rt, str):
                result_text = rt
            usage = evt.get("usage") or {}
            # The result message carries cumulative usage; prefer it when present (avoids double count).
            if usage:
                input_tokens = int(usage.get("input_tokens", 0) or 0)
                output_tokens = int(usage.get("output_tokens", 0) or 0)
    return {
        "tool_calls": tool_calls,
        "submit_args": submit_args,
        "final_text": (result_text or final_text).strip(),
        "usage": Usage(input_tokens, max(0, output_tokens)),
        "is_error": is_error,
        "subtype": subtype,
    }


def _abstain_investigation(question: str, *, model: str, elapsed_s: float, stop_reason: str,
                           error: str | None, contract: str | None,
                           tool_calls: Sequence[str] = (), iterations: int = 0,
                           usage: Usage | None = None) -> Investigation:
    return Investigation(
        question=question,
        answer=Answer(answer=ABSTAIN, confidence=0.0, evidence="", method=f"claude-mcp: {stop_reason}"),
        iterations=iterations,
        tool_calls=list(tool_calls),
        usage=usage or Usage(),
        model=model,
        elapsed_s=elapsed_s,
        stop_reason=stop_reason,
        error=error,
        contract=contract,
    )


def _load_commit_gate_log(path: str) -> list[dict[str, Any]]:
    """Read the server-side decision stream. A missing/truncated telemetry file is non-fatal."""
    out: list[dict[str, Any]] = []
    try:
        with open(path, encoding="utf-8") as fh:
            for line in fh:
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError:
                    continue
                decision = rec.get("commit_gate") if isinstance(rec, Mapping) else None
                if isinstance(decision, Mapping):
                    out.append(dict(decision))
    except OSError:
        pass
    return out


def _plan_fanout_telemetry(tool_calls: Sequence[str], budget: int) -> dict[str, Any]:
    """SOT-2661 — per-stage telemetry for the planner→fan-out→synthesis flow (SOT-2629 form).

    Derives the stage accounting from the executed tool-call sequence:
    * ①計画/②収集 — ``search_first`` / ``search_calls`` (the unified fan-out call(s));
    * ③合成 — ``supplement_calls`` (the ≤2 additional narrowing probes after the first search).
    ``tool_turns`` is the total investigator-tool turns (submit_answer excluded), i.e. the段数 the
    3-stage flow is meant to hold at ≤ ``budget``.
    """
    from src.rag.tools import unified_search as _unified_search
    search_name = _unified_search.TOOL_NAME
    non_submit = [t for t in tool_calls if t != SUBMIT_ANSWER]
    search_calls = sum(1 for t in non_submit if t == search_name)
    first = non_submit[0] if non_submit else None
    # supplement = narrowing tool calls that are NOT the unified search (stage ③); when search never
    # fired, every non-submit call counts as a supplement probe (the flow degraded to individual tools).
    supplement = (len(non_submit) - search_calls) if search_calls else len(non_submit)
    # SOT-2664 — non-terminal tool turns executed beyond the plan-fanout budget. When the bounded finisher
    # is ON these are the extra targeted raw-evidence reads it granted; when OFF this is always 0 (the
    # server refuses every over-budget call), so the field doubles as a finisher-fired indicator.
    finisher_max = fanout_finisher_budget()
    finisher_calls = max(0, len(non_submit) - int(budget)) if budget and budget > 0 else 0
    return {
        "enabled": True,
        "budget": int(budget),
        "first_tool": first,
        "search_first": bool(first == search_name),
        "search_calls": int(search_calls),
        "supplement_calls": int(max(0, supplement)),
        "extra_searches": int(max(0, search_calls - 1)),
        "tool_turns": int(len(non_submit)),
        "finisher_enabled": bool(finisher_max > 0),
        "finisher_max": int(finisher_max),
        "finisher_calls": int(finisher_calls),
    }


def investigate_question(question: str, *, tools: Sequence[AgentTool],
                         system: str | None = None, contract: str | None = None,
                         preamble: str | None = None,
                         max_turns: int = 12, timeout_s: float = 180.0,
                         model: str | None = None) -> Investigation:
    """Run one question's whole tool loop on flat-rate Sonnet via claude CLI + MCP.

    Mirrors :func:`src.rag.agent.investigator.investigate`'s contract: always returns an
    :class:`Investigation` (never raises), abstaining with a coded ``stop_reason`` on any failure so a
    batch keeps going and the failure is visible in the details log.
    """
    mdl = model or os.getenv("RAG_CLAUDE_MCP_MODEL") or DEFAULT_MODEL
    model_label = f"{mdl}({_MODEL_TAG})"
    resume = _resume_path()

    # Resume/short-circuit gates (checked before any subprocess spend).
    key = _resume_key(question, model_label)
    if resume is not None:
        cached = _load_resume(resume).get(key)
        if cached is not None:
            return _record_to_investigation(cached.get("record") or {})
    if _USAGE_LIMIT.is_set():
        return _abstain_investigation(
            question, model=model_label, elapsed_s=0.0, stop_reason="usage_limit",
            error="flat-rate usage limit tripped earlier this run; skipped", contract=contract)

    if shutil.which(CLAUDE_BIN) is None:
        return _abstain_investigation(
            question, model=model_label, elapsed_s=0.0, stop_reason="model_error",
            error=f"{CLAUDE_BIN!r} not on PATH (claude-mcp backend requires the Claude Code CLI)",
            contract=contract)

    sys_prompt = (system or SYSTEM_PROMPT) + _harness_system_suffix()
    user_prompt = f"{preamble}\n\n---\n\n{question}" if preamble else question
    allowed = _allowed_tools(tools)
    # SOT-2661 — redefine the per-question budget to the 3-stage shape when RAG_PLAN_FANOUT is on
    # (≤5 non-terminal tool calls + the uncapped submit ⇒ ≤6 total). No-op (== max_turns) when the flag is off,
    # so the server-side RAG_MCP_MAX_TOOL_CALLS cap stays byte-identical.
    eff_max_turns = plan_fanout_budget(max_turns)

    started = time.monotonic()
    tmp = tempfile.NamedTemporaryFile("w", suffix=".mcp.json", delete=False, encoding="utf-8")
    log_fd, log_path = tempfile.mkstemp(suffix=".mcp_tool_calls.jsonl")
    os.close(log_fd)
    gate_fd, gate_log_path = tempfile.mkstemp(suffix=".mcp_commit_gate.jsonl")
    os.close(gate_fd)
    try:
        json.dump(_mcp_config(log_path, eff_max_turns, question=question, contract=contract,
                              commit_gate_log=gate_log_path),
                  tmp, ensure_ascii=False)
        tmp.flush()
        tmp.close()
        stdout, stderr, rc, timed_out = _run_claude(
            user_prompt, system=sys_prompt, model=mdl, cfg_path=tmp.name,
            allowed=allowed, timeout=timeout_s)
        gate_decisions = _load_commit_gate_log(gate_log_path)
    finally:
        for p in (tmp.name, log_path, gate_log_path):
            try:
                os.unlink(p)
            except OSError:
                pass
    elapsed = max(0.0, time.monotonic() - started)

    parsed = _parse_stream_json(stdout)
    tool_calls = parsed["tool_calls"]
    iterations = sum(1 for t in tool_calls if t != SUBMIT_ANSWER)
    usage = parsed["usage"]

    # Usage-limit detection: trip the latch so the rest of the batch short-circuits; do NOT persist this
    # abstain to the resume sidecar so a later re-run retries the question.
    combined = "\n".join((stderr, parsed["subtype"], parsed["final_text"]))
    if (timed_out is False and rc != 0 and _is_usage_limit(combined)) or (
            parsed["is_error"] and _is_usage_limit(combined)):
        _USAGE_LIMIT.set()
        return _abstain_investigation(
            question, model=model_label, elapsed_s=elapsed, stop_reason="usage_limit",
            error=f"claude usage limit: {combined.strip()[:400]}", contract=contract,
            tool_calls=tool_calls, iterations=iterations, usage=usage)

    if timed_out:
        inv = _abstain_investigation(
            question, model=model_label, elapsed_s=elapsed, stop_reason="timeout",
            error=f"claude -p exceeded timeout_s={timeout_s}", contract=contract,
            tool_calls=tool_calls, iterations=iterations, usage=usage)
    elif rc != 0 and parsed["submit_args"] is None and not parsed["final_text"]:
        inv = _abstain_investigation(
            question, model=model_label, elapsed_s=elapsed, stop_reason="model_error",
            error=f"claude -p exited {rc}: {(stderr or stdout)[:400]}", contract=contract,
            tool_calls=tool_calls, iterations=iterations, usage=usage)
    else:
        gate_tel: dict[str, Any] | None = None
        if parsed["submit_args"] is not None:
            answer = _answer_from_args(parsed["submit_args"])
            if gate_decisions:
                from src.rag.agent import commit_gate as _commit_gate
                last = gate_decisions[-1]
                gate_tel = dict(last.get("telemetry") or {})
                # The server is the submit execution point. Honor its formatted/abstained terminal value;
                # a trailing REJECT means Claude ended without a successful re-submit and must fail closed.
                if _commit_gate.enforce():
                    verdict = str(last.get("verdict") or "")
                    if verdict in {"COMMIT", "ABSTAIN"}:
                        final = str(last.get("final_answer") or ABSTAIN)
                    else:
                        final = ABSTAIN
                    answer = Answer(answer=final,
                                    confidence=(answer.confidence if verdict == "COMMIT" else 0.0),
                                    evidence=answer.evidence,
                                    method=f"claude-mcp: commit_gate {verdict or 'REJECT'}")
            stop_reason = "answered"
        elif parsed["final_text"]:
            # Plain final-text answer with no submit_answer call — accepted at confidence 0.0, mirroring
            # the Gemini loop's plain-final-text branch.
            answer = Answer(answer=parsed["final_text"], confidence=0.0, evidence="",
                            method="claude-mcp: plain final text (no submit_answer)")
            if is_abstain(answer.answer):
                answer = Answer(answer=ABSTAIN, confidence=0.0, evidence="", method=answer.method)
            # A final text bypasses submit_answer, so no in-band retry is possible. Apply the same gate
            # once client-side and fail closed on REJECT, as required by SOT-2640.
            from src.rag.agent import commit_gate as _commit_gate
            if _commit_gate.enabled():
                decision = _commit_gate.evaluate(question, contract, answer.answer)
                gate_tel = decision.telemetry
                if _commit_gate.enforce():
                    final = (decision.final_answer if decision.verdict == _commit_gate.COMMIT else ABSTAIN)
                    answer = Answer(answer=final,
                                    confidence=(answer.confidence if decision.verdict == _commit_gate.COMMIT else 0.0),
                                    evidence=answer.evidence,
                                    method=f"claude-mcp: plain final commit_gate {decision.verdict}")
            stop_reason = "answered"
        else:
            answer = Answer(answer=ABSTAIN, confidence=0.0, evidence="",
                            method="claude-mcp: no answer emitted")
            stop_reason = "max_turns"
        # SOT-2650 — 値を変えない括弧内付加情報の書式契約 (RAG_FORMAT_STRIP_PAREN, default OFF). Applied at
        # the backend boundary so it covers both the submit_answer and plain-final-text commits even when
        # the commit gate itself is off (the cycle2 dev lever set runs gate-less).
        strip_tel: dict[str, Any] | None = None
        if stop_reason == "answered" and not is_abstain(answer.answer):
            from src.rag.agent import formatting as _fmt
            if _fmt.strip_paren_enabled():
                stripped, fired = _fmt.strip_trailing_parenthetical(question, answer.answer)
                strip_tel = {"applied": bool(fired), "rules": fired}
                if fired:
                    answer = Answer(answer=stripped, confidence=answer.confidence,
                                    evidence=answer.evidence, method=answer.method + " +paren_strip")
            # SOT-2656 — 値保存回答正規化 (説明文/接頭辞/カウンタ) (RAG_FORMAT_VALUE_NORM, default OFF).
            if _fmt.value_norm_enabled():
                normed, nfired = _fmt.normalize_value_answer(question, answer.answer)
                if nfired:
                    strip_tel = {"applied": True,
                                 "rules": list((strip_tel or {}).get("rules", [])) + nfired}
                    answer = Answer(answer=normed, confidence=answer.confidence,
                                    evidence=answer.evidence, method=answer.method + " +value_norm")
        inv = Investigation(
            question=question, answer=answer, iterations=iterations, tool_calls=tool_calls,
            usage=usage, model=model_label, elapsed_s=elapsed, stop_reason=stop_reason,
            error=None, contract=contract)
        if gate_tel is not None:
            inv.interventions["commit_gate"] = gate_tel
        if strip_tel is not None:
            inv.interventions["format_strip_paren"] = strip_tel

    # SOT-2661 — always record the 3-stage flow telemetry when RAG_PLAN_FANOUT is on (answered OR
    # abstained: timeout/model_error paths still carry the tool sequence up to the failure). Off ⇒
    # interventions untouched (byte-identical).
    if plan_fanout_enabled():
        inv.interventions["plan_fanout"] = _plan_fanout_telemetry(tool_calls, eff_max_turns)

    if resume is not None and inv.stop_reason != "usage_limit":
        _append_resume(resume, key, question, inv)
    return inv
