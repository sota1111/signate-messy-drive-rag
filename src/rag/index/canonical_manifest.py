"""Canonical-route file manifest (SOT-2530 / 事前処理 #3).

Index-time precomputation of the ``project → {kind: [rel_paths]}`` map that
:func:`src.rag.tools.canonical_route.discover` needs at run time. Today ``discover`` re-walks the
corpus and runs ``_matches_kind`` on *every* file on *every* call — find_files/canonical_route fire
~1,119× across the BUDGET questions, so that per-call scan is pure repeated work over a static corpus.
This module runs the discovery **once** at build time (every project × every :data:`KINDS` kind) and
persists the ranked result, so the runtime route becomes an O(1) dict lookup.

Design invariants (mirroring the sibling pre-stores — evidence_index #4a, structure_store #5)
---------------------------------------------------------------------------------------------
* **Answers never change (回答不変).**  A stored rel list is exactly ``_discover_live(project, kind)``
  with ``ext_hint=None`` — the same match+rank the tool computes live. The runtime ``ext_hint`` filter
  is applied on the looked-up list identically to :func:`_matches_kind`, so the resolved files (and
  their order) are byte-identical to the live scan.
* **Zero-regression fallback (回帰ゼロ).**  A disabled flag, an absent/schema-mismatched artifact, a
  corpus-signature drift (a file added/removed/renamed), a KINDS-signature drift (the matcher/ranking
  vocabulary changed in code without a rebuild), or a project the manifest never saw all fall through
  to the live walk in :func:`discover`.
* **Opt-in at serve time.**  :func:`enabled` (``RAG_CANONICAL_MANIFEST``) gates *runtime consultation*
  only; the artifact is always (re)built at index time as an additive, guarded step. Default OFF ⇒ the
  champion serve path is byte-identical — matching the ``RAG_FACT_INDEX`` / ``RAG_EVIDENCE_INDEX`` /
  ``RAG_STRUCTURE_STORE`` conventions.
* **No hard-coded answers, Gemini-only, redistribution-safe.**  The manifest holds only structural
  file paths discovered by the corpus-agnostic matcher; no corpus-specific answer/secret is embedded.
  The 同名 train.xlsx×9 disambiguation stays project-scoped exactly as before (per-project entries).
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any

import importlib

from config import settings
from src.rag.corpus import FileRef, nfc, walk

# Resolve the *module* via importlib: ``src.rag.tools.__init__`` re-exports the ``canonical_route``
# *function*, shadowing the submodule attribute, so both ``from src.rag.tools import canonical_route``
# and ``import src.rag.tools.canonical_route as canonical_route`` would bind the function (the latter's
# ``as`` target is resolved by getattr, which returns the shadowing function). import_module returns
# the sys.modules entry — the actual module object exposing ``KINDS`` / ``_matches_kind`` / ``_rank_key``.
canonical_route = importlib.import_module("src.rag.tools.canonical_route")

SCHEMA = "canonical-manifest"
SCHEMA_VERSION = 1

_ON = {"1", "true", "yes", "on"}


def enabled() -> bool:
    """True when :func:`discover` should consult the persisted manifest (default OFF — opt-in)."""
    return os.getenv("RAG_CANONICAL_MANIFEST", "0").strip().lower() in _ON


def default_out_path() -> Path:
    return settings.ARTIFACTS_DIR / "canonical_manifest.json"


# --------------------------------------------------------------------------- signatures (staleness)
def corpus_signature(refs: list[FileRef]) -> str:
    """SHA-256 over the sorted corpus rel set — invalidates on any file add/remove/rename.

    The manifest maps which files exist per (project, kind); content edits are irrelevant (the rel set
    is unchanged), but a structural change to the file listing must invalidate it. Hashing the sorted
    NFC rels captures exactly that.
    """
    h = hashlib.sha256()
    for rel in sorted(nfc(r.rel) for r in refs):
        h.update(rel.encode("utf-8"))
        h.update(b"\n")
    return h.hexdigest()


def kinds_signature() -> str:
    """SHA-256 over the match/rank-relevant fields of :data:`KINDS`.

    Only the fields that determine the stored rels (``exts``/``stem_tokens``/``categories`` for
    matching, ``prefer`` for ranking, plus ``name`` for keying) are hashed — a code edit to any of
    them invalidates a manifest that was not rebuilt, forcing the live fallback.
    """
    payload = [[k.name, list(k.exts), list(k.stem_tokens), list(k.categories), list(k.prefer)]
               for k in canonical_route.KINDS]
    blob = json.dumps(payload, ensure_ascii=False, sort_keys=True)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


# --------------------------------------------------------------------------- build / persist
def build(refs: list[FileRef] | None = None, *, out: Path | None = None) -> dict[str, int]:
    """Precompute and persist ``project → {kind: [rel_paths]}`` for the whole corpus.

    Computes directly over ``refs`` (defaulting to :func:`corpus.walk`) using the same match+rank that
    :func:`_discover_live` uses, and stores the corpus/KINDS signatures of *those* refs so a runtime
    consult can detect drift. Every walked project gets an entry (possibly with no kinds) so a
    validated-but-empty project resolves to ``[]`` at run time instead of a spurious live fallback.
    """
    refs = refs if refs is not None else walk()
    projects = sorted({r.project for r in refs if r.project})
    manifest: dict[str, dict[str, list[str]]] = {}
    for proj in projects:
        entry: dict[str, list[str]] = {}
        for kind in canonical_route.KINDS:
            hits = [r for r in refs
                    if r.project == proj and canonical_route._matches_kind(r, kind, None)]
            hits.sort(key=lambda r: canonical_route._rank_key(r, kind))
            rels = [r.rel for r in hits]
            if rels:  # omit empty kinds to keep the artifact small (absent kind == [] at lookup)
                entry[kind.name] = rels
        manifest[proj] = entry
    return write_manifest(manifest, corpus_signature(refs), kinds_signature(), out)


def write_manifest(manifest: dict[str, dict[str, list[str]]], corpus_sig: str, kinds_sig: str,
                   path: Path | None = None) -> dict[str, int]:
    """Atomically write the reproducible manifest JSON (schema header + signatures + sorted content)."""
    out = path or default_out_path()
    out.parent.mkdir(parents=True, exist_ok=True)
    doc = {
        "schema": SCHEMA,
        "version": SCHEMA_VERSION,
        "corpus_signature": corpus_sig,
        "kinds_signature": kinds_sig,
        "projects": manifest,
    }
    tmp = out.with_suffix(out.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(doc, handle, ensure_ascii=False, sort_keys=True, indent=2)
        handle.write("\n")
    tmp.replace(out)
    reset_cache()
    return {
        "projects": len(manifest),
        "kind_entries": sum(len(kinds) for kinds in manifest.values()),
        "files": sum(len(rels) for kinds in manifest.values() for rels in kinds.values()),
    }


# --------------------------------------------------------------------------- load / lookup
_LOAD_CACHE: dict[str, dict[str, Any]] = {}
_VALIDATED: dict[tuple[str, str], dict[str, Any] | None] = {}
_REL_INDEX: dict[str, dict[str, FileRef]] = {}


def reset_cache() -> None:
    """Drop every in-process cache (used by the build path and tests)."""
    _LOAD_CACHE.clear()
    _VALIDATED.clear()
    _REL_INDEX.clear()


def load(path: Path | None = None) -> dict[str, Any]:
    """Load the manifest JSON (memoized). Returns ``{}`` when absent/unreadable/schema-mismatch."""
    out = path or default_out_path()
    key = str(out)
    if key in _LOAD_CACHE:
        return _LOAD_CACHE[key]
    doc: dict[str, Any] = {}
    try:
        with open(out, encoding="utf-8") as handle:
            loaded = json.load(handle)
        if isinstance(loaded, dict) and loaded.get("schema") == SCHEMA \
                and loaded.get("version") == SCHEMA_VERSION:
            doc = loaded
    except Exception:
        doc = {}
    _LOAD_CACHE[key] = doc
    return doc


def _validated_projects(corpus_dir: Path | None = None,
                        path: Path | None = None) -> dict[str, Any] | None:
    """The manifest ``projects`` map iff enabled AND both signatures still match, else None.

    The signature comparison is done once per (artifact, corpus) and memoized, so the recurring
    :func:`lookup_rels` stays a pure dict lookup rather than re-hashing the corpus on every call.
    """
    if not enabled():
        return None
    doc = load(path)
    if not doc:
        return None
    key = (str(path or default_out_path()), str(corpus_dir) if corpus_dir is not None else "")
    if key in _VALIDATED:
        return _VALIDATED[key]
    refs = walk(corpus_dir)
    ok = (doc.get("corpus_signature") == corpus_signature(refs)
          and doc.get("kinds_signature") == kinds_signature())
    result = doc.get("projects") if ok else None
    _VALIDATED[key] = result
    return result


def lookup_rels(project: str | None, kind_name: str, *, corpus_dir: Path | None = None,
                path: Path | None = None) -> list[str] | None:
    """Precomputed ranked rels for ``(project, kind)``, or None to signal a live fallback.

    Returns ``[]`` (authoritative empty) when the project *is* in a validated manifest but has no file
    of this kind — that is a real "no match", not a miss, so the caller must not re-scan. Returns None
    (fallback) when disabled / manifest absent / signature drift / the project was never indexed.
    """
    if project is None:
        return None
    projects = _validated_projects(corpus_dir, path)
    if not projects:
        return None
    entry = projects.get(nfc(project))
    if entry is None:
        return None
    return entry.get(kind_name, [])


def rel_index(corpus_dir: Path | None = None) -> dict[str, FileRef]:
    """``nfc(rel) → FileRef`` for the corpus (memoized), to rehydrate manifest rels into FileRefs."""
    key = str(corpus_dir) if corpus_dir is not None else ""
    idx = _REL_INDEX.get(key)
    if idx is None:
        idx = {nfc(r.rel): r for r in walk(corpus_dir)}
        _REL_INDEX[key] = idx
    return idx


if __name__ == "__main__":
    print(build())
