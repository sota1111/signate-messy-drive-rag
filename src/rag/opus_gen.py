"""Claude Opus answer backend (SOT-2457) — 正答作成を Opus が担当する.

The Vertex Gemini generation path plateaued (real-LB commit accuracy collapses on unseen
questions), so per the SOT-2457 directive the answer set is produced by **Claude Opus** via
the Claude CLI (`claude -p --model opus`) instead of Gemini, and scored locally by the
Codex batch judge. This module mirrors ``generate.answer_question``'s contract:

- the shared deterministic front (trust gate + hard modules) runs first, unchanged;
- evidence comes from the existing hybrid retriever (the index — including the vision
  captions baked into it — is reused as-is; no new Gemini calls are made);
- ONE Opus call per question drafts + self-verifies and returns
  ``{"answer", "confidence"}``; anything but confidence="high" abstains
  (Incorrect = −1 < Missing = 0, so the precision-first gate stays).

Raw PNGs are NOT re-attached (the CLI call is text-only); figure questions see the
indexed vision captions instead.

Env knobs (read directly from the environment — config/ is intentionally untouched):
- OPUS_GEN_MODEL    model passed to `claude --model` (default "opus")
- OPUS_GEN_TIMEOUT  seconds per claude call (default 600)
"""
from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import tempfile

from config import settings
from src.rag import generate, retrieve


class OpusGenError(RuntimeError):
    """claude CLI failed / timed out / returned unusable output."""


def available() -> bool:
    return shutil.which("claude") is not None


def model() -> str:
    return os.getenv("OPUS_GEN_MODEL", "").strip() or "opus"


def timeout_s() -> int:
    try:
        return max(1, int(os.getenv("OPUS_GEN_TIMEOUT", "") or 600))
    except ValueError:
        return 600


_OUTPUT_CONTRACT = """
上記の根拠のみに基づき、質問へ回答してください。ツールやファイルアクセスは使わず、
与えられたテキストのみで判断してください。

回答前に自己検証すること（誤答は0点より悪い-1点、棄権は0点）:
- 答えの値・要素が根拠に明示的に現れ、そのまま読み取れる場合のみ confidence="high"。
- 複数資料をまたぐ計算・集計・差分でその数値が根拠に直接書かれていない場合、
  列挙の網羅を根拠から確認しきれない場合、根拠が断片的で推測を含む場合は confidence="low"。

出力は次のJSONオブジェクト1つのみ（前置き・説明・コードフェンス不要）:
{"answer": "<最終回答（問われた値・要素そのものだけ）>", "confidence": "high|medium|low"}
""".strip()


def _build_prompt(question: str, evidence: list[dict], advisory: list[str]) -> str:
    return (
        f"{generate.SYSTEM}\n\n"
        f"質問:\n{question}\n\n"
        f"根拠資料:\n{generate._format_evidence(evidence)}"
        f"{generate.advisory_block(advisory)}\n\n"
        f"{_OUTPUT_CONTRACT}"
    )


def _run_claude(prompt: str) -> str:
    """One `claude -p` call → raw text output. Runs in an empty temp cwd so no project
    context (CLAUDE.md) leaks into the prompt and tool use has nothing to read."""
    if not available():
        raise OpusGenError("claude CLI not found on PATH")
    with tempfile.TemporaryDirectory(prefix="opus-gen-") as tmp:
        cmd = ["claude", "-p", "--model", model()]
        try:
            proc = subprocess.run(cmd, input=prompt, capture_output=True, text=True,
                                  timeout=timeout_s(), cwd=tmp)
        except subprocess.TimeoutExpired as e:
            raise OpusGenError(f"claude -p timed out after {timeout_s()}s") from e
    if proc.returncode != 0:
        tail = (proc.stderr or proc.stdout or "").strip()[-400:]
        raise OpusGenError(f"claude -p failed (exit {proc.returncode}): {tail!r}")
    return (proc.stdout or "").strip()


def _parse(raw: str) -> dict:
    """Extract the {"answer", "confidence"} object; unparseable output → low confidence."""
    text = raw.strip()
    fenced = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    if fenced:
        text = fenced.group(1)
    start = text.find("{")
    if start >= 0:
        try:
            obj = json.loads(text[start:text.rfind("}") + 1])
            if isinstance(obj, dict) and obj.get("answer") is not None:
                return obj
        except Exception:  # noqa: BLE001 — fall through to the abstain default
            pass
    return {"answer": "", "confidence": "low"}


def answer_question(question: str, *, k: int = 16) -> dict:
    resolved, advisory = generate.deterministic_front(question)
    if resolved is not None:
        return resolved

    evidence = retrieve.get().retrieve(question, k=k)
    obj = _parse(_run_claude(_build_prompt(question, evidence, advisory)))
    ans = str(obj.get("answer") or "").strip()
    conf = str(obj.get("confidence") or "low").strip().lower()
    if not ans or conf != "high":
        return generate._result(question, settings.ABSTAIN, conf or "low", ans,
                                evidence, [], verified=False)
    return generate._result(question, generate._clip_tokens(ans), conf, ans,
                            evidence, [], verified=True)
