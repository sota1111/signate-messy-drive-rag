"""Deterministic typed evidence-location index built from extracted document text.

This is the build half of SOT-2531.  It deliberately contains no retrieval policy and no
question-specific answers: it records reusable tokens and formatting facts with provenance.
"""
from __future__ import annotations

import json
import re
import unicodedata
from dataclasses import asdict, dataclass
from pathlib import Path

from config import settings
from src.rag.corpus import FileRef, nfc

SCHEMA = "typed-evidence-index"
SCHEMA_VERSION = 1

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


if __name__ == "__main__":
    print(build_only())
