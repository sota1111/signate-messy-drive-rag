"""Deterministic structure pre-store (SOT-2533 / 事前処理 #5).

Index-time precomputation and persistence of the *deterministic-but-runtime-expensive* structure
extractions — highlighted spans/cells, embedded-chart exact numbers (``numCache``), PivotTable
condition cells, the seating directory, and version-diff pairs — so the runtime tools **read a
persisted answer** instead of re-opening and re-parsing the source file on every question. This is
the index-time complement to the runtime relaxations of Step 12: the discovery/extraction cost is
paid once at build time rather than repeatedly inside the per-question turn budget.

Design invariants
-----------------
* **Answers never change (回答不変).**  Every stored payload is the *serialized output of the same
  existing extractor* the tool would call live, so a tool consulting the store returns exactly what
  it would have computed — the committed answers stay byte-identical.
* **Zero-regression fallback (回帰ゼロ).**  A store miss, a stale entry (the source file changed →
  content-sha mismatch), or the disabled flag all fall through to the original live extraction.
* **Opt-in at serve time.**  :func:`enabled` (``RAG_STRUCTURE_STORE``) gates *runtime consultation*
  only; the artifact is always (re)built at index time as an additive, guarded step.  With the flag
  unset the champion serve path is unchanged — mirroring the fact-index (``RAG_FACT_INDEX``) and
  typed evidence-index conventions.
* **Deterministic & LLM-free.**  Only deterministic extractors feed the store: Office highlights
  (never the PDF vision fallback), XML chart ``numCache``, PivotTable/AutoFilter conditions, the
  *hash-pinned reviewed* seating directory (never generic vision), and structural version diffs.
* **No hard-coded answers, Gemini-only, redistribution-safe.**  The store holds only reusable
  structural facts with provenance; no corpus-specific answer is embedded.
"""
from __future__ import annotations

import copy
import hashlib
import json
import os
from pathlib import Path
from typing import Any

from config import settings
from src.rag.corpus import FileRef, nfc

SCHEMA = "structure-store"
SCHEMA_VERSION = 1

# Office formats whose formatting layer (highlights) is read deterministically. PDF is excluded on
# purpose: its highlight path can fall back to Gemini vision, which is neither deterministic nor
# reproducible, so it must not be baked into a persisted store.
_HIGHLIGHT_EXTS = {"xlsx", "xlsm", "pptx", "docx"}
# Formats that may carry an embedded OOXML chart with a numeric cache.
_CHART_EXTS = {"xlsx", "xlsm", "pptx"}

_ON = {"1", "true", "yes", "on"}


def enabled() -> bool:
    """True when runtime tools should consult the persisted store (default OFF — opt-in)."""
    return os.getenv("RAG_STRUCTURE_STORE", "0").strip().lower() in _ON


def content_sha(path: Path | str) -> str:
    """SHA-256 of a file's bytes, or ``""`` when it cannot be read.

    Used as the staleness key: a stored payload is trusted only while the source file hashes to the
    same value, so editing/replacing a document transparently invalidates its cached structures.
    """
    try:
        with open(path, "rb") as handle:
            return hashlib.sha256(handle.read()).hexdigest()
    except Exception:
        return ""


def default_out_path() -> Path:
    return settings.ARTIFACTS_DIR / "structure_store.json"


# --------------------------------------------------------------------------- per-file extractors
def _highlight_items(ref: FileRef) -> list[dict[str, Any]] | None:
    """Unfiltered highlighted items for an Office file (deterministic; no PDF/vision path)."""
    if ref.ext not in _HIGHLIGHT_EXTS or ref.name.startswith("~$"):
        return None
    from src.rag.tools.highlight_extract import _office_items

    items = _office_items(ref, None)
    return items or None


def _chart_contract(ref: FileRef) -> dict[str, Any] | None:
    """Full ``extract_chart_numcache`` contract for a file that carries a chart numCache, else None."""
    if ref.ext not in _CHART_EXTS or ref.name.startswith("~$"):
        return None
    from src.rag.tools.chart_numcache import ContractError, _extract_chart_numcache_live

    try:
        return _extract_chart_numcache_live(ref)
    except ContractError:
        return None


def _pivot_facts(ref: FileRef) -> list[str] | None:
    """Deterministic PivotTable/AutoFilter condition fact rows for an xlsx, else None."""
    if ref.ext != "xlsx" or ref.name.startswith("~$"):
        return None
    from src.rag import facts

    rows = facts._pivot_facts(ref)
    return rows or None


def _file_payload(ref: FileRef) -> dict[str, Any]:
    """Every deterministic structure payload for one file, keyed for staleness by ``content_sha``.

    Each extractor is guarded independently so a single malformed document degrades to fewer stored
    structures, never an exception that aborts the whole build.
    """
    payload: dict[str, Any] = {"content_sha": content_sha(ref.path)}
    for key, fn in (("highlights", _highlight_items), ("charts", _chart_contract),
                    ("pivots", _pivot_facts)):
        try:
            value = fn(ref)
        except Exception:
            value = None
        if value:
            payload[key] = value
    return payload


# --------------------------------------------------------------------------- corpus-wide extractors
def _seating_payload() -> dict[str, Any] | None:
    """The hash-pinned reviewed seating directory, or None when it is not deterministically available.

    Only a ``reviewed`` (pixel-hash pinned) directory is stored — the generic Gemini-vision path is
    non-deterministic and must never be persisted as an authoritative structure.
    """
    from src.rag.tools import seating_chart
    from src.rag.tools.extract_tools import resolve_ref

    try:
        ref = resolve_ref("座席表.pptx")
    except Exception:
        return None
    directory = seating_chart._build_directory_live(ref, use_cache=False)
    if directory.origin != "reviewed" or not directory.seats:
        return None
    return {
        "file": directory.source,
        "content_sha": content_sha(ref.path),
        "pixel_sha": directory.pixel_sha,
        "origin": directory.origin,
        "seats": [seat.to_dict() for seat in directory.seats],
    }


def _version_diff_payload() -> list[dict[str, Any]]:
    """Every resolvable version pair's structural diff, keyed for staleness by both files' sha."""
    from src.rag import diffpair

    out: list[dict[str, Any]] = []
    try:
        pairs = diffpair.find_pairs()
    except Exception:
        return out
    for pair in pairs:
        try:
            changes = diffpair._structural_diff_live(pair)
        except Exception:
            changes = None
        if not changes:
            continue
        out.append({
            "old": pair.old.rel,
            "new": pair.new.rel,
            "base": pair.base,
            "basis": pair.basis,
            "old_sha": content_sha(pair.old.path),
            "new_sha": content_sha(pair.new.path),
            "changes": [{"label": c.label, "before": c.before, "after": c.after, "kind": c.kind}
                        for c in changes],
        })
    return sorted(out, key=lambda rec: (rec["old"], rec["new"]))


# --------------------------------------------------------------------------- build / persist
def build(refs: list[FileRef] | None = None, *, out: Path | None = None) -> dict[str, int]:
    """Precompute and persist every deterministic structure for the corpus.

    Iterates the corpus once, storing only files that yield at least one structure, plus the
    corpus-wide seating directory and version-diff pairs. Returns a small count summary.
    """
    from src.rag import corpus

    refs = refs if refs is not None else corpus.walk()
    files: dict[str, dict[str, Any]] = {}
    for ref in refs:
        try:
            payload = _file_payload(ref)
        except Exception:
            continue
        if len(payload) > 1:  # more than just content_sha
            files[nfc(ref.rel)] = payload
    corpus_payload: dict[str, Any] = {}
    seating = _seating_payload()
    if seating:
        corpus_payload["seating"] = seating
    diffs = _version_diff_payload()
    if diffs:
        corpus_payload["version_diffs"] = diffs
    return write_store(files, corpus_payload, out)


def write_store(files: dict[str, dict[str, Any]], corpus_payload: dict[str, Any],
                path: Path | None = None) -> dict[str, int]:
    """Atomically write the reproducible store JSON (schema header + sorted content)."""
    out = path or default_out_path()
    out.parent.mkdir(parents=True, exist_ok=True)
    doc = {"schema": SCHEMA, "version": SCHEMA_VERSION, "files": files, "corpus": corpus_payload}
    tmp = out.with_suffix(out.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(doc, handle, ensure_ascii=False, sort_keys=True, indent=2)
        handle.write("\n")
    tmp.replace(out)
    reset_cache()
    return {
        "files": len(files),
        "highlights": sum(1 for f in files.values() if f.get("highlights")),
        "charts": sum(1 for f in files.values() if f.get("charts")),
        "pivots": sum(1 for f in files.values() if f.get("pivots")),
        "seating": 1 if corpus_payload.get("seating") else 0,
        "version_pairs": len(corpus_payload.get("version_diffs") or []),
    }


# --------------------------------------------------------------------------- load / lookup
_STORE_CACHE: dict[str, dict[str, Any]] = {}


def reset_cache() -> None:
    """Drop the in-process load cache (used by the build path and tests)."""
    _STORE_CACHE.clear()


def load(path: Path | None = None) -> dict[str, Any]:
    """Load the store JSON (memoized per path). Returns ``{}`` when absent/unreadable/schema-mismatch."""
    out = path or default_out_path()
    key = str(out)
    if key in _STORE_CACHE:
        return _STORE_CACHE[key]
    doc: dict[str, Any] = {}
    try:
        with open(out, encoding="utf-8") as handle:
            loaded = json.load(handle)
        if isinstance(loaded, dict) and loaded.get("schema") == SCHEMA \
                and loaded.get("version") == SCHEMA_VERSION:
            doc = loaded
    except Exception:
        doc = {}
    _STORE_CACHE[key] = doc
    return doc


def _fresh_file_entry(ref: FileRef, path: Path | None = None) -> dict[str, Any] | None:
    """The stored payload for ``ref`` iff present AND the source file still hashes the same."""
    entry = load(path).get("files", {}).get(nfc(ref.rel))
    if not entry:
        return None
    if entry.get("content_sha") != content_sha(ref.path):
        return None  # source changed → stale → caller falls back to live extraction
    return entry


def lookup_highlight_items(ref: FileRef, path: Path | None = None) -> list[dict[str, Any]] | None:
    """Cached unfiltered highlighted items for ``ref``, or None (disabled / miss / stale)."""
    if not enabled():
        return None
    entry = _fresh_file_entry(ref, path)
    items = entry.get("highlights") if entry else None
    return copy.deepcopy(items) if items else None


def lookup_chart_contract(ref: FileRef, path: Path | None = None) -> dict[str, Any] | None:
    """Cached ``extract_chart_numcache`` contract for ``ref``, or None (disabled / miss / stale)."""
    if not enabled():
        return None
    entry = _fresh_file_entry(ref, path)
    contract = entry.get("charts") if entry else None
    return copy.deepcopy(contract) if contract else None


def lookup_pivot_facts(ref: FileRef, path: Path | None = None) -> list[str] | None:
    """Cached deterministic pivot condition fact rows for ``ref``, or None."""
    if not enabled():
        return None
    entry = _fresh_file_entry(ref, path)
    rows = entry.get("pivots") if entry else None
    return list(rows) if rows else None


def lookup_seating(ref: FileRef, path: Path | None = None) -> dict[str, Any] | None:
    """Cached reviewed seating directory for ``ref``, or None (disabled / miss / stale)."""
    if not enabled():
        return None
    seating = load(path).get("corpus", {}).get("seating")
    if not seating or nfc(seating.get("file", "")) != nfc(ref.rel):
        return None
    if seating.get("content_sha") != content_sha(ref.path):
        return None
    return copy.deepcopy(seating)


def lookup_version_changes(old: FileRef, new: FileRef,
                           path: Path | None = None) -> list[dict[str, Any]] | None:
    """Cached structural changes for the ``old→new`` pair, or None (disabled / miss / stale)."""
    if not enabled():
        return None
    for rec in load(path).get("corpus", {}).get("version_diffs") or []:
        if nfc(rec.get("old", "")) == nfc(old.rel) and nfc(rec.get("new", "")) == nfc(new.rel):
            if rec.get("old_sha") == content_sha(old.path) \
                    and rec.get("new_sha") == content_sha(new.path):
                return copy.deepcopy(rec.get("changes") or [])
            return None
    return None


if __name__ == "__main__":
    print(build())
