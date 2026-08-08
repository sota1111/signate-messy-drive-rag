"""Deterministic typed evidence-location index built from extracted document text.

This is the build half of SOT-2531.  It deliberately contains no retrieval policy and no
question-specific answers: it records reusable tokens and formatting facts with provenance.
"""
from __future__ import annotations

import json
import os
import re
import unicodedata
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable

from config import settings
from src.rag.corpus import FileRef, nfc

SCHEMA = "typed-evidence-index"
SCHEMA_VERSION = 1

_ON = {"1", "true", "yes", "on"}
_SNIPPET_MAX = 300

_SHEET = re.compile(r"^\[シート:\s*(?P<name>[^]]+)\]")
_SLIDE = re.compile(r"^\[スライド(?P<num>\d+)\]")
_HL_CELL = re.compile(r"^\s*(?P<cell>[A-Z]+\d+)\((?P<color>[^)]+)\):\s*(?P<value>.+)$")
_MARK = re.compile(r"^【(?P<kind>ハイライト|ハイライト:[^】]+|図形塗り:[^】]+)】(?P<value>.+)$")
_EXT = re.compile(r"(?i)\b(?:EXT[.\-:\s]*|内線[：:\s]*)(?P<value>\d{2,8})\b")
_NUMBER = re.compile(
    r"(?<![\w])(?:\d{4}[-/]\d{1,2}[-/]\d{1,2}|(?:¥|￥|\$)?\d[\d,]*(?:\.\d+)?(?:円|万円|億円|%|％)?)(?![\w])")
_PERSON = re.compile(r"(?<![一-龯々])(?P<value>[一-龯々]{1,8}(?:さん|様|氏|部長|課長|社長))(?![一-龯々])")
_ALIAS = re.compile(r"(?P<left>[\w一-龯ぁ-んァ-ヶー]{2,40})\s*(?:（|\()(?P<right>[^()（）\n]{2,40})(?:）|\))")
_PARAM = re.compile(r"^\s*(?P<name>[A-Za-z_][A-Za-z0-9_.-]*)\s*=\s*(?P<value>[^#\n]+?)\s*(?:#.*)?$")
_CONDITIONS = re.compile(r"別契約|明記|必須|任意|対象外|除外|条件|ただし|以上|以下|未満")


def normalize(value: object) -> str:
    """NFC-normalize and collapse whitespace for stable keys and provenance."""
    return " ".join(nfc(str(value)).split())


@dataclass(frozen=True)
class EvidenceEntry:
    type: str
    key: str
    value: str
    project: str
    file: str
    rel: str
    sheet: str = ""
    cell: str = ""
    paragraph: str = ""
    marker: str = ""


def _entry(ref: FileRef, kind: str, value: str, *, sheet: str = "", cell: str = "",
           paragraph: str = "", marker: str = "", key: str | None = None) -> EvidenceEntry:
    value = normalize(value)
    return EvidenceEntry(kind, normalize(key if key is not None else value).casefold(), value,
                         normalize(ref.project), normalize(ref.name), normalize(ref.rel),
                         normalize(sheet), normalize(cell), normalize(paragraph), normalize(marker))


def scan_doc(ref: FileRef, text: str) -> list[EvidenceEntry]:
    """Return typed evidence entries from one already-extracted document.

    ``paragraph`` is a stable extracted-line locator for formats without native cells.  XLSX
    highlighted-cell annotations retain their native sheet and A1 coordinate.
    """
    entries: list[EvidenceEntry] = []
    sheet = ""
    paragraph = ""
    in_highlights = False
    lines = unicodedata.normalize("NFC", text).splitlines()
    for line_no, raw in enumerate(lines, 1):
        line = raw.rstrip()
        if match := _SHEET.match(line):
            sheet, paragraph, in_highlights = normalize(match.group("name")), "", False
        elif match := _SLIDE.match(line):
            paragraph, in_highlights = f"slide:{match.group('num')}", False
        elif line.strip() == "【ハイライトされたセル】":
            in_highlights = True
            continue
        location = paragraph or f"line:{line_no}"
        if in_highlights and (match := _HL_CELL.match(line)):
            value = match.group("value")
            entries.append(_entry(ref, "highlight", value, sheet=sheet, cell=match.group("cell"),
                                  marker=match.group("color")))
        elif in_highlights and line.strip():
            in_highlights = False
        if match := _MARK.match(line.strip()):
            entries.append(_entry(ref, "highlight", match.group("value"), sheet=sheet,
                                  paragraph=location, marker=match.group("kind")))
        if line.startswith("【太字箇所】"):
            for value in line.removeprefix("【太字箇所】").split(" / "):
                if value.strip():
                    entries.append(_entry(ref, "bold", value, sheet=sheet, paragraph=location))
        for match in _EXT.finditer(line):
            entries.append(_entry(ref, "ext", match.group("value"), sheet=sheet, paragraph=location,
                                  key=match.group("value")))
        for match in _PERSON.finditer(line):
            value = match.group("value")
            key = re.sub(r"(?:さん|様|氏|部長|課長|社長)$", "", value)
            entries.append(_entry(ref, "person", value, sheet=sheet, paragraph=location, key=key))
        for match in _CONDITIONS.finditer(line):
            entries.append(_entry(ref, "condition", match.group(), sheet=sheet, paragraph=location))
        for match in _ALIAS.finditer(line):
            left, right = match.group("left"), match.group("right")
            entries.append(_entry(ref, "alias", right, sheet=sheet, paragraph=location, key=left))
            entries.append(_entry(ref, "alias", left, sheet=sheet, paragraph=location, key=right))
        if match := _PARAM.match(line):
            entries.append(_entry(ref, "param", match.group("value"), sheet=sheet,
                                  paragraph=location, key=match.group("name")))
        for match in _NUMBER.finditer(line):
            entries.append(_entry(ref, "number", match.group(), sheet=sheet, paragraph=location))

    # Header/column names are the first non-marker pipe-delimited row after each sheet marker.
    for index, line in enumerate(lines):
        match = _SHEET.match(line)
        if not match:
            continue
        current_sheet = normalize(match.group("name"))
        for candidate in lines[index + 1:]:
            if _SHEET.match(candidate):
                break
            if " | " in candidate and not candidate.startswith("【"):
                for col_no, value in enumerate(candidate.split(" | "), 1):
                    if value.strip():
                        entries.append(_entry(ref, "column", value, sheet=current_sheet,
                                              cell=f"column:{col_no}"))
                break
    unique = {tuple(asdict(item).values()): item for item in entries}
    return sorted(unique.values(), key=lambda item: (item.project, item.rel, item.type, item.key,
                                                       item.sheet, item.cell, item.paragraph))


def default_out_path() -> Path:
    return settings.ARTIFACTS_DIR / "evidence_index.jsonl"


def write_index(entries: list[EvidenceEntry], path: Path | None = None) -> dict[str, int]:
    """Atomically replace a reproducible JSONL index (schema header + sorted entries)."""
    out = path or default_out_path()
    out.parent.mkdir(parents=True, exist_ok=True)
    ordered = sorted(entries, key=lambda item: (item.project, item.rel, item.type, item.key,
                                                 item.sheet, item.cell, item.paragraph, item.value))
    tmp = out.with_suffix(out.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8", newline="\n") as handle:
        handle.write(json.dumps({"schema": SCHEMA, "version": SCHEMA_VERSION},
                                ensure_ascii=False, sort_keys=True) + "\n")
        for entry in ordered:
            handle.write(json.dumps(asdict(entry), ensure_ascii=False, sort_keys=True) + "\n")
    tmp.replace(out)
    return {"entries": len(ordered), "files": len({(x.project, x.rel) for x in ordered})}


def build_only(*, out: Path | None = None, caption_images: bool = False) -> dict[str, int]:
    from src.rag import corpus
    from src.rag.extract import extract

    entries: list[EvidenceEntry] = []
    for ref in corpus.walk():
        try:
            doc = extract(ref, caption_images=caption_images)
            entries.extend(scan_doc(ref, doc.text))
        except Exception:
            continue
    return write_index(entries, out)


# --------------------------------------------------------------------------- runtime read / lookup
#
# The build half above is unconditional (the artifact is always regenerated at index time). This
# read half is the SOT-2532 / 事前処理 #4b addition: it lets ``file_grep`` consult the typed index
# at question time to answer discovery queries by lookup instead of extracting every corpus file.
#
# Design invariants (mirroring the structure-store, SOT-2533):
#   * Opt-in at serve time — :func:`enabled` (``RAG_EVIDENCE_INDEX``) gates *runtime consultation*
#     only; with the flag unset the champion serve path is byte-identical (file_grep full scan).
#   * Zero-regression fallback — a missing / unreadable / schema-mismatched index yields ``[]`` so
#     the caller falls through to the live scan; a query the typed index does not cover is a miss,
#     which also falls back to the full scan (index is a *fast path*, never a truncation of truth).
#   * Deterministic & answer-preserving — a hit returns the entry's own recorded provenance
#     (file / sheet / cell / paragraph); no question-specific answer or hard-coded location.

_INDEX_CACHE: dict[str, list[dict[str, str]]] = {}


def enabled() -> bool:
    """True when runtime tools should consult the persisted evidence index (default OFF — opt-in).

    Mirrors the ``RAG_STRUCTURE_STORE`` / ``RAG_FACT_INDEX`` conventions: the artifact is always
    (re)built at index time, but *runtime consultation* is gated so the champion serve path stays
    byte-identical unless the flag is set.
    """
    return os.getenv("RAG_EVIDENCE_INDEX", "0").strip().lower() in _ON


def reset_cache() -> None:
    """Drop the in-process load cache (used by the build path and tests)."""
    _INDEX_CACHE.clear()


def load(path: Path | None = None) -> list[dict[str, str]]:
    """Load the index entry rows (memoized per path).

    Returns ``[]`` when the index is absent, unreadable, or its schema/version header does not match
    — the caller then falls back to the live scan (回帰ゼロ). The schema header line is validated and
    dropped; only entry rows are returned.
    """
    out = path or default_out_path()
    key = str(out)
    if key in _INDEX_CACHE:
        return _INDEX_CACHE[key]
    entries: list[dict[str, str]] = []
    try:
        with open(out, encoding="utf-8") as handle:
            header = handle.readline()
            meta = json.loads(header) if header.strip() else {}
            if isinstance(meta, dict) and meta.get("schema") == SCHEMA \
                    and meta.get("version") == SCHEMA_VERSION:
                for line in handle:
                    line = line.strip()
                    if line:
                        entries.append(json.loads(line))
    except Exception:
        entries = []
    _INDEX_CACHE[key] = entries
    return entries


def _rel_ext(rel: str, file: str) -> str:
    name = rel or file
    return name.rsplit(".", 1)[-1].lower() if "." in name else ""


def _lookup_where(cell: str, sheet: str, paragraph: str) -> str:
    """Most-specific locator for an index hit, mirroring ``file_grep._where`` shapes."""
    if cell:
        return f"cell:{cell}" + (f"@sheet:{sheet}" if sheet else "")
    if sheet:
        return f"sheet:{sheet}"
    if paragraph:
        return paragraph  # already a "line:N" / "slide:N" locator
    return "index"


def lookup(pattern: "re.Pattern[str]", *, ext: str | Iterable[str] | None = None,
           project: str | None = None, max_hits_per_file: int = 50, limit: int = 500,
           path: Path | None = None) -> list[dict[str, Any]]:
    """Return ``file_grep``-shaped hit records for entries whose value/key matches ``pattern``.

    ``pattern`` is a pre-compiled :class:`re.Pattern` (the caller applies the same literal/regex,
    ignore-case and NFC handling used for the live scan, so matching semantics stay identical). A
    hit is index-derived: ``line`` is ``None`` and ``where`` is the stored cell/sheet/paragraph
    locator. Returns ``[]`` when disabled or on any miss so the caller falls back to the live scan.
    """
    if not enabled():
        return []
    entries = load(path)
    if not entries:
        return []
    items = [ext] if isinstance(ext, str) else list(ext or [])
    exts = {e.lower().lstrip(".") for e in items if e} or None
    proj = nfc(project) if project else None

    hits: list[dict[str, Any]] = []
    per_file: dict[str, int] = {}
    seen: set[tuple[str, str, str]] = set()
    for entry in entries:
        rel = entry.get("rel") or entry.get("file") or ""
        if proj is not None and nfc(entry.get("project", "")) != proj:
            continue
        entry_ext = _rel_ext(rel, entry.get("file", ""))
        if exts is not None and entry_ext not in exts:
            continue
        value = entry.get("value", "")
        key = entry.get("key", "")
        if not (pattern.search(nfc(value)) or (key and pattern.search(key))):
            continue
        where = _lookup_where(entry.get("cell", ""), entry.get("sheet", ""),
                              entry.get("paragraph", ""))
        snippet = value if len(value) <= _SNIPPET_MAX else value[:_SNIPPET_MAX] + "…"
        dedup = (rel, where, snippet)
        if dedup in seen or per_file.get(rel, 0) >= max_hits_per_file:
            continue
        seen.add(dedup)
        per_file[rel] = per_file.get(rel, 0) + 1
        hits.append({
            "file": rel,
            "project": entry.get("project", ""),
            "ext": entry_ext,
            "line": None,
            "where": where,
            "snippet": snippet,
        })
        if len(hits) >= limit:
            break
    return hits


if __name__ == "__main__":
    print(build_only())
