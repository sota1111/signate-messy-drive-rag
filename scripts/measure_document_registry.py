#!/usr/bin/env python3
"""SOT-2583 — measure the document registry's canonical_doc_recall@1/@3 + explicit-file resolution.

Builds a *deterministic* labelled set from the validation questions that name a file explicitly:
for each such question the target document is the corpus file whose folded filename equals the named
token, scoped to the question's glossary-resolved project (kept only when a single corpus file
matches, so the label is unambiguous). It then runs :class:`Resolver.resolve` over the persisted
registry and records ``canonical_doc_recall@1/@3`` and ``explicit_filename_resolution_rate``.

    .venv/bin/python scripts/measure_document_registry.py

Writes ``artifacts/document_registry_recall.json`` (machine) and prints a summary. Deterministic,
offline (no Gemini/LLM) — the resolver's chain is structural.
"""
from __future__ import annotations

import json
import importlib
import sys
from pathlib import Path

import pandas as pd

# Allow the documented direct invocation from the repository root without requiring PYTHONPATH.
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from config import settings
from src.rag.corpus import nfc, walk
from src.rag.index import document_registry as dr

# ``src.rag.tools.__init__`` re-exports the ``canonical_route`` *function*, shadowing the submodule;
# import the module object explicitly for ``resolve_project`` (same idiom as canonical_manifest).
canonical_route = importlib.import_module("src.rag.tools.canonical_route")


def _target_for(named: str, project: str | None) -> str | None:
    """The canonical corpus doc_id whose filename matches ``named`` within ``project``.

    The named token is peeled to its real filename keys (company/possessive prefix stripped). When a
    single filename matches, that is the target; when several copies of the same filename exist (e.g.
    ``train.csv`` under both ``03.データ`` and ``04.分析/.../data``) the *shallowest* rel is taken as the
    canonical copy — a corpus-layout convention, independent of the resolver's own ranking, so the
    label is not circular.
    """
    keys = set(dr.filename_match_keys(named))
    hits = []
    for ref in walk():
        if project and nfc(ref.project) != nfc(project):
            continue
        if dr.fold(ref.name) in keys:
            hits.append(nfc(ref.rel))
    if not hits:
        return None
    hits.sort(key=lambda r: (r.count("/"), r))
    return hits[0]


def build_cases() -> list[tuple[str, str | None, str]]:
    cases: list[tuple[str, str | None, str]] = []
    seen: set[str] = set()
    for path in (settings.QUESTIONS_VALID, settings.QUESTIONS_TEST):
        try:
            df = pd.read_csv(path)
        except Exception:
            continue
        for _, row in df.iterrows():
            question = str(row["question"])
            if question in seen:
                continue
            tokens = dr.explicit_file_tokens(question)
            if not tokens:
                continue
            project = canonical_route.resolve_project(question, None)
            for tok in tokens:
                target = _target_for(tok, project)
                if target:
                    cases.append((question, project, target))
                    seen.add(question)
                    break
    return cases


def main() -> None:
    dr.reset_cache()
    if not dr.load():
        dr.build(walk())
    cases = build_cases()
    result = dr.measure_recall(cases)
    out = settings.ARTIFACTS_DIR / "document_registry_recall.json"
    out.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"labelled explicit-file cases: {result['n_cases']}")
    print(f"canonical_doc_recall: {result['canonical_doc_recall']}")
    print(f"explicit_filename_cases: {result['explicit_filename_cases']}")
    print(f"explicit_filename_resolution_rate: {result['explicit_filename_resolution_rate']}")
    print(f"-> {out}")


if __name__ == "__main__":
    main()
