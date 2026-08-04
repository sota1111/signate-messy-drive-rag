"""Thin, retrying wrapper around Vertex AI (Gemini generation + text embeddings).

Uses the google-genai SDK in Vertex mode, authenticated via Application Default
Credentials (ADC). All generation is deterministic (temperature=0, fixed seed) so runs
are reproducible, per the competition's reproducibility requirement.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Sequence

from google import genai
from google.genai import types

from config import settings

_client: genai.Client | None = None


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
    """
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


def embed(texts: Sequence[str], *, model: str | None = None, batch: int = 64,
          task_type: str = "RETRIEVAL_DOCUMENT", retries: int = 4) -> list[list[float]]:
    """Embed a list of texts, batched. task_type is RETRIEVAL_DOCUMENT or RETRIEVAL_QUERY."""
    mdl = model or settings.EMBED_MODEL
    out: list[list[float]] = []
    for i in range(0, len(texts), batch):
        chunk = [t if t.strip() else " " for t in texts[i : i + batch]]
        for attempt in range(retries):
            try:
                resp = client().models.embed_content(
                    model=mdl,
                    contents=chunk,
                    config=types.EmbedContentConfig(task_type=task_type),
                )
                out.extend([list(e.values) for e in resp.embeddings])
                break
            except Exception as e:  # noqa: BLE001
                if attempt == retries - 1:
                    raise RuntimeError(f"embed failed: {e}") from e
                time.sleep(min(2 ** attempt * 2, 30))
    return out
