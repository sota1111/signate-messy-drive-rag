"""Pluggable LLM text-role providers (SOT-2606).

Each provider exposes ``generate_text(prompt, *, system, model, temperature, max_output_tokens,
response_schema, retries) -> str`` and serves only tool-free *text* generate() calls (Stage3
formatting / judge / rerank). Vision and the investigator function-calling loop always stay on
Gemini — see :mod:`src.rag.llm` for the guard.
"""
