"""Agent layer — Gemini function-calling investigation loop (SOT-2460 Step2).

The production answer path: hand ``gemini-2.5-pro`` the corpus-agnostic Step1 tools and let it
plan → call tools → return a structured ``{answer, confidence, evidence, method}`` result.
"""
