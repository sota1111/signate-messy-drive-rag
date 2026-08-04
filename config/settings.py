"""Central configuration, loaded from environment (.env) with sensible defaults.

Everything the pipeline needs — GCP project, Vertex models, paths — resolves here so
no project-specific value is hard-coded across the codebase.
"""
from __future__ import annotations

import os
from pathlib import Path

try:
    from dotenv import load_dotenv

    load_dotenv(Path(__file__).resolve().parent.parent / ".env")
except Exception:  # dotenv optional
    pass

REPO_ROOT = Path(__file__).resolve().parent.parent


def _path(env_key: str, default: str) -> Path:
    return (REPO_ROOT / os.getenv(env_key, default)).resolve()


# ---- GCP / Vertex ----
GCP_PROJECT_ID = os.getenv("GCP_PROJECT_ID", "signate-messy-drive-rag")
VERTEX_LOCATION = os.getenv("VERTEX_LOCATION", "us-central1")
GEN_MODEL = os.getenv("GEN_MODEL", "gemini-2.5-flash")
GEN_MODEL_HARD = os.getenv("GEN_MODEL_HARD", "gemini-2.5-pro")
VISION_MODEL = os.getenv("VISION_MODEL", "gemini-2.5-flash")
EMBED_MODEL = os.getenv("EMBED_MODEL", "text-embedding-005")

# ---- Local self-scoring ----
JUDGE_BACKEND = os.getenv("JUDGE_BACKEND", "gemini")  # "gemini" | "openai"
JUDGE_MODEL = os.getenv("JUDGE_MODEL", "gemini-2.5-pro")

# ---- SIGNATE ----
SIGNATE_COMPETITION_KEY = os.getenv("SIGNATE_COMPETITION_KEY", "098a4365f4514c2fa197d4e16548b3bb")
SIGNATE_TASK_KEY = os.getenv("SIGNATE_TASK_KEY", "90c19eddfdc746be88534b7e45fe1dd0")

# ---- Paths ----
DATA_DIR = _path("DATA_DIR", "data")
CORPUS_DIR = _path("CORPUS_DIR", "data/share_drive")
QUESTIONS_DIR = _path("QUESTIONS_DIR", "data/questions")
INDEX_DIR = _path("INDEX_DIR", "index_store")
ARTIFACTS_DIR = _path("ARTIFACTS_DIR", "artifacts")

QUESTIONS_VALID = QUESTIONS_DIR / "questions_valid.csv"
QUESTIONS_TEST = QUESTIONS_DIR / "questions_test.csv"
VALID_GROUND_TRUTH = QUESTIONS_DIR / "valid_txt.csv"

# Answer hard cap enforced by the official validator (tiktoken tokens).
MAX_ANSWER_TOKENS = 1000
# Standard abstention string (scores 0 = Missing, strictly better than -1 Incorrect).
ABSTAIN = "わかりません"

for _d in (INDEX_DIR, ARTIFACTS_DIR):
    _d.mkdir(parents=True, exist_ok=True)
