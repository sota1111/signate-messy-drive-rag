# SOT-2583 Document Registry verification

Recorded: 2026-08-09 UTC

## Corpus build

- Command: `.venv/bin/python -m src.rag.index`
- Result: PASS
- Files visible in the mounted corpus and registered: **403 / 403**
- Versioned documents: 10
- Documents with predecessor links: 5
- Documents with sheet metadata: 19
- Extraction warnings: 1 (the encrypted/binary file remains represented in the registry)

The Issue description says 418 files, but the corpus mounted for this run contains 403 files according
to the repository's canonical `src.rag.corpus.walk()` traversal. The registry covers every currently
available file; it does not invent records for the 15 files absent from the mounted corpus.

## Locator-question measurement

- Command: `.venv/bin/python scripts/measure_document_registry.py`
- Labelled explicit-file questions: 44
- `canonical_doc_recall@1`: **0.9773** (43/44)
- `canonical_doc_recall@3`: **1.0000** (44/44)
- `explicit_filename_resolution_rate`: **1.0000** (44/44)

The measurement is deterministic and offline. Labels are derived from exact filename matches scoped by
the existing project glossary; ambiguous duplicate filenames use the shallowest corpus path as the
canonical copy.

## Regression verification

- `.venv/bin/python -m pytest -q`: **983 passed**, 8 existing openpyxl WMF warnings
- `.venv/bin/python -m compileall -q src scripts tests`: PASS
- Runtime registry consultation is opt-in through `RAG_DOCUMENT_REGISTRY`; when unset/OFF, the hard
  constraint returns without changing the existing answer path.
