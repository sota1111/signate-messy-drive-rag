"""SOT-2626 — stdio MCP surface for the investigator's read-only tool set.

Exposes :func:`src.rag.agent.investigator.build_tools` (the *single* source of truth for the agent's
tool schemas) as a Model-Context-Protocol server so a flat-rate ``claude -p --mcp-config`` session can
drive the same deterministic corpus tools the live Gemini function-calling loop uses. Dev-only: the
production answer path (Gemini) is untouched. See :mod:`src.rag.mcp.server`.
"""
