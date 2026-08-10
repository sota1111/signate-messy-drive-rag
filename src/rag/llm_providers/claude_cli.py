"""claude-cli text-role LLM provider (SOT-2606).

Calls the Claude Code fixed-plan Sonnet through the ``claude -p`` subprocess so that DEV/diagnostic
loops (gold100 local answer-reachability checks, judge, rerank, Stage3 formatting) avoid Gemini API
billing. **Text generate only** — no tools, no images.

Auth reuses the existing Claude Code login (no API key needed). The prompt is fed on **stdin** (not
argv) so large text-role prompts never hit ARG_MAX; the system instruction, when given, is passed
via ``--append-system-prompt``.

WARNING: the fixed-plan usage limit is shared account-wide with the autonomous workers, so heavy
loops through this provider throttle them. Dev-only, limited use. Production stays on Gemini
(``LLM_PROVIDER`` default ``gemini``).
"""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import time

# Guard against argv/stdin binaries other than the Claude Code CLI shadowing the name.
CLAUDE_BIN = "claude"
_DEFAULT_TIMEOUT = 300  # seconds per attempt — text roles are short; a hang should not wedge a run.


def _schema_hint(response_schema: dict | None) -> str:
    """Turn a Gemini-style response_schema into a plain-text JSON instruction for claude -p.

    claude -p has no native structured-output mode, so we ask for JSON matching the schema and let
    the caller parse it (the same callers — e.g. rerank — already degrade gracefully on parse error).
    """
    if not response_schema:
        return ""
    return (
        "\n\nRespond with ONLY a single JSON value (no prose, no code fences) that conforms to this "
        "JSON schema:\n" + json.dumps(response_schema, ensure_ascii=False)
    )


def generate_text(
    prompt: str,
    *,
    system: str | None = None,
    model: str | None = None,
    temperature: float = 0.0,  # accepted for signature parity; claude -p has no temperature flag
    max_output_tokens: int = 2048,  # accepted for parity; not enforced by the CLI
    response_schema: dict | None = None,
    retries: int = 4,
    timeout: int = _DEFAULT_TIMEOUT,
) -> str:
    """Return the text produced by ``claude -p --model <model>`` for a single text prompt.

    Raises ``RuntimeError`` after ``retries`` failed attempts (non-zero exit, timeout, ``is_error``
    result, or unparseable output) so the caller sees an explicit failure — there is no silent
    fallback to Gemini (that would defeat the cost-avoidance intent and hide misconfiguration).
    """
    if shutil.which(CLAUDE_BIN) is None:
        raise RuntimeError(
            f"claude-cli provider selected but {CLAUDE_BIN!r} is not on PATH "
            "(Claude Code CLI required for LLM_PROVIDER=claude-cli)"
        )

    mdl = model or os.getenv("CLAUDE_CLI_MODEL") or "sonnet"
    full_prompt = prompt + _schema_hint(response_schema)
    cmd = [CLAUDE_BIN, "-p", "--model", mdl, "--output-format", "json"]
    if system:
        cmd += ["--append-system-prompt", system]

    last_err: Exception | None = None
    for attempt in range(retries):
        try:
            proc = subprocess.run(
                cmd, input=full_prompt, capture_output=True, text=True, timeout=timeout,
            )
            if proc.returncode != 0:
                raise RuntimeError(
                    f"claude -p exited {proc.returncode}: {(proc.stderr or proc.stdout)[:500]}"
                )
            data = json.loads(proc.stdout)
            if data.get("is_error"):
                raise RuntimeError(f"claude -p returned error result: {str(data)[:500]}")
            text = data.get("result")
            if not isinstance(text, str) or not text.strip():
                raise RuntimeError(f"claude -p returned empty result: {str(data)[:300]}")
            return text.strip()
        except Exception as e:  # noqa: BLE001 — back off and retry, mirroring the Gemini path
            last_err = e
            time.sleep(min(2 ** attempt * 2, 30))
    raise RuntimeError(f"claude-cli generate failed after {retries} attempts: {last_err}")
