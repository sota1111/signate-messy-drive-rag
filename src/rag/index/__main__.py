"""``python -m src.rag.index`` — build the hybrid retrieval index (chunks + embeddings).

Kept as a package entry point so the historical invocation ``python -m src.rag.index`` (used by
``scripts/deploy.sh`` and the ops runbooks) still builds the index after ``index.py`` became the
``index/`` package (SOT-2531). The typed evidence store (SOT-2531 / #4a) is written as a side
effect of :func:`build`; to (re)build only that artifact without embedding, run
``python -m src.rag.index.evidence_index`` instead.
"""
from __future__ import annotations

import argparse

from src.rag.index import build

ap = argparse.ArgumentParser(prog="python -m src.rag.index")
ap.add_argument("--no-images", action="store_true", help="skip Gemini PNG captioning")
args = ap.parse_args()
build(caption_images=not args.no_images)
