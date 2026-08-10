"""Thin, retrying wrapper around Vertex AI (Gemini generation + text embeddings).

Uses the google-genai SDK in Vertex mode, authenticated via Application Default
Credentials (ADC). All generation is deterministic (temperature=0, fixed seed) so runs
are reproducible, per the competition's reproducibility requirement.
"""
from __future__ import annotations

import os
import time
from dataclasses import dataclass, field
from typing import Any, Sequence

from google import genai
from google.genai import types

from config import settings

_client: genai.Client | None = None


# --------------------------------------------------- LLM provider selection (SOT-2606) -------------
# ``LLM_PROVIDER`` picks the backend for tool-free *text* generate() calls (Stage3 formatting / judge
# / rerank).  Default ``gemini`` == current behaviour, byte-identical.  ``claude-cli`` routes text
# generate() through the Claude Code fixed-plan Sonnet (`claude -p`) for DEV cost avoidance.
#
# GUARD (misroute prevention): this ONLY affects tool-free text generate().  The investigator
# function-calling loop (``src/rag/agent/investigator.py`` — native Vertex ``tools``) never calls
# generate(); it uses :func:`client` directly, so it is always Gemini.  Any generate() call carrying
# ``images`` (vision) is ALSO forced to Gemini regardless of ``LLM_PROVIDER`` — claude-cli is
# text-only (no tools, no images).
#
# WARNING: the claude-cli fixed-plan usage limit is shared account-wide with the autonomous workers,
# so running heavy loops (e.g. gold100) through it will throttle them.  Dev-only, limited use.
# Config lives here (not config/settings.py, a hook-protected path) and is read at call time so
# tests / dev sessions can flip it via the environment.
_TEXT_PROVIDERS = ("gemini", "claude-cli", "anthropic")


def _provider() -> str:
    """The configured text-role LLM provider (env ``LLM_PROVIDER``; default ``gemini``)."""
    return (os.getenv("LLM_PROVIDER") or getattr(settings, "LLM_PROVIDER", "gemini") or "gemini").strip().lower()


def _claude_cli_model() -> str:
    """Model passed to ``claude -p --model`` (env ``CLAUDE_CLI_MODEL``; default ``sonnet``)."""
    return (os.getenv("CLAUDE_CLI_MODEL") or getattr(settings, "CLAUDE_CLI_MODEL", "sonnet") or "sonnet").strip()


def client() -> genai.Client:
    global _client
    if _client is None:
        _client = genai.Client(
            vertexai=True,
            project=settings.GCP_PROJECT_ID,
            location=settings.VERTEX_LOCATION,
        )
    return _client


@dataclass
class Image:
    """An inline image part for multimodal prompts."""

    data: bytes
    mime_type: str = "image/png"


def _parts(prompt: str, images: Sequence[Image] | None) -> list[types.Part]:
    parts: list[types.Part] = [types.Part.from_text(text=prompt)]
    for img in images or []:
        parts.append(types.Part.from_bytes(data=img.data, mime_type=img.mime_type))
    return parts


def generate(
    prompt: str,
    *,
    system: str | None = None,
    model: str | None = None,
    images: Sequence[Image] | None = None,
    temperature: float = 0.0,
    max_output_tokens: int = 2048,
    thinking_budget: int = 0,
    response_schema: dict | None = None,
    retries: int = 4,
) -> str:
    """Return the text (or JSON string) produced by Gemini for a single prompt.

    thinking_budget: Gemini 2.5 reasoning tokens. 0 disables thinking (fast, and avoids
    thinking consuming the whole max_output_tokens budget); raise it for hard reasoning.

    Provider routing (SOT-2606): tool-free *text* prompts honour ``LLM_PROVIDER`` (default
    ``gemini``). Any call carrying ``images`` (vision) stays on Gemini regardless — claude-cli is
    text-only. See the module docstring for the guard rationale.
    """
    provider = _provider()
    # Vision guard: image prompts are ALWAYS Gemini (claude-cli/anthropic text providers can't see).
    if provider != "gemini" and not images:
        if provider == "claude-cli":
            from src.rag.llm_providers import claude_cli

            return claude_cli.generate_text(
                prompt, system=system, model=model, temperature=temperature,
                max_output_tokens=max_output_tokens, response_schema=response_schema,
                retries=retries,
            )
        if provider == "anthropic":
            raise NotImplementedError(
                "LLM_PROVIDER=anthropic (raw Anthropic API) is not yet implemented; "
                "use 'gemini' (default) or 'claude-cli'."
            )
        raise ValueError(
            f"unknown LLM_PROVIDER={provider!r}; expected one of {_TEXT_PROVIDERS}"
        )
    return _gemini_generate(
        prompt, system=system, model=model, images=images, temperature=temperature,
        max_output_tokens=max_output_tokens, thinking_budget=thinking_budget,
        response_schema=response_schema, retries=retries,
    )


def _gemini_generate(
    prompt: str,
    *,
    system: str | None = None,
    model: str | None = None,
    images: Sequence[Image] | None = None,
    temperature: float = 0.0,
    max_output_tokens: int = 2048,
    thinking_budget: int = 0,
    response_schema: dict | None = None,
    retries: int = 4,
) -> str:
    """The Vertex AI Gemini text/vision generate path (also the forced path for image prompts)."""
    mdl = model or settings.GEN_MODEL
    cfg = types.GenerateContentConfig(
        temperature=temperature,
        seed=0,
        max_output_tokens=max_output_tokens,
        system_instruction=system,
        thinking_config=types.ThinkingConfig(thinking_budget=thinking_budget),
    )
    if response_schema is not None:
        cfg.response_mime_type = "application/json"
        cfg.response_schema = response_schema

    contents = [types.Content(role="user", parts=_parts(prompt, images))]
    last_err: Exception | None = None
    for attempt in range(retries):
        try:
            resp = client().models.generate_content(model=mdl, contents=contents, config=cfg)
            text = resp.text
            if text is None:
                # Blocked or empty — treat as recoverable, back off once.
                raise RuntimeError(f"empty response (finish={getattr(resp, 'candidates', None)})")
            return text.strip()
        except Exception as e:  # noqa: BLE001 — broad on purpose, we back off and retry
            last_err = e
            time.sleep(min(2 ** attempt * 2, 30))
    raise RuntimeError(f"Gemini generate failed after {retries} attempts: {last_err}")


# ------------------------------------- LLM reranker (R4 / SOT-2450) --------------------------
_RERANK_SYSTEM = (
    "あなたは検索結果の関連度を採点するリランカです。質問に答えるための根拠として"
    "各資料がどれだけ直接的に役立つかを 0〜10 の整数で採点してください。\n"
    "- 質問が求める具体的な値・条件・識別子・列挙対象がその資料に明示されていれば高得点(8〜10)。\n"
    "- 話題は近いが答えそのものは含まない資料は中程度(3〜6)。\n"
    "- 無関係・別案件・別ファイルの資料は低得点(0〜2)。\n"
    "推測や外部知識で補わず、資料の記載内容のみで判断してください。"
)

_RERANK_SCHEMA = {
    "type": "object",
    "properties": {
        "scores": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "index": {"type": "integer"},
                    "score": {"type": "number"},
                },
                "required": ["index", "score"],
            },
        }
    },
    "required": ["scores"],
}


def rerank(question: str, candidates: Sequence[str], *, model: str | None = None,
           max_chars: int = 600, retries: int = 3) -> list[float]:
    """Score each candidate snippet 0–10 for relevance to ``question`` (same order as input).

    Deterministic (temperature 0, thinking disabled, fixed seed) so a rerank run is reproducible.
    Any failure — parse error, missing index, empty candidates — degrades to all-zero scores, which
    leaves the caller's stable sort on the original fusion order untouched (a safe no-op fallback).
    """
    if not candidates:
        return []
    listing = "\n".join(
        f"[{i}] {c[:max_chars]}" for i, c in enumerate(candidates)
    )
    prompt = (
        f"質問:\n{question}\n\n"
        f"候補資料(各行 [番号] 本文):\n{listing}\n\n"
        "各候補の関連度を 0〜10 で採点し、JSONで {\"scores\":[{\"index\":番号,\"score\":点数}, ...]} "
        "を全候補分返してください。"
    )
    scores = [0.0] * len(candidates)
    try:
        raw = generate(
            prompt, system=_RERANK_SYSTEM, model=model or settings.GEN_MODEL,
            temperature=0.0, thinking_budget=0, max_output_tokens=1024,
            response_schema=_RERANK_SCHEMA, retries=retries,
        )
        import json
        for item in (json.loads(raw).get("scores") or []):
            idx = item.get("index")
            if isinstance(idx, int) and 0 <= idx < len(scores):
                try:
                    scores[idx] = float(item.get("score", 0.0))
                except (TypeError, ValueError):
                    scores[idx] = 0.0
    except Exception:  # noqa: BLE001 — rerank is best-effort; fall back to fusion order
        return [0.0] * len(candidates)
    return scores


def _embed_batch(mdl: str, chunk: list[str], task_type: str, retries: int) -> list[list[float]]:
    """Embed one batch; on a token/size 400 error, split and recurse (Vertex caps ~20k tok/req)."""
    for attempt in range(retries):
        try:
            resp = client().models.embed_content(
                model=mdl, contents=chunk,
                config=types.EmbedContentConfig(task_type=task_type),
            )
            return [list(e.values) for e in resp.embeddings]
        except Exception as e:  # noqa: BLE001
            msg = str(e)
            if ("token count" in msg or "INVALID_ARGUMENT" in msg or "400" in msg) and len(chunk) > 1:
                mid = len(chunk) // 2
                return (_embed_batch(mdl, chunk[:mid], task_type, retries)
                        + _embed_batch(mdl, chunk[mid:], task_type, retries))
            if attempt == retries - 1:
                raise RuntimeError(f"embed failed: {e}") from e
            time.sleep(min(2 ** attempt * 2, 30))
    return []


def embed(texts: Sequence[str], *, model: str | None = None, batch: int = 16,
          task_type: str = "RETRIEVAL_DOCUMENT", retries: int = 4) -> list[list[float]]:
    """Embed a list of texts, batched (adaptive split on size errors)."""
    mdl = model or settings.EMBED_MODEL
    out: list[list[float]] = []
    for i in range(0, len(texts), batch):
        chunk = [t if t.strip() else " " for t in texts[i : i + batch]]
        out.extend(_embed_batch(mdl, chunk, task_type, retries))
    return out
