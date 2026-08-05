"""file_grep — NFC-aware cross-corpus full-text / cell grep tool.

The starting point of any 案件・区分をまたぐ探索 ("where is X mentioned across the whole drive?"):
a single tool that scans **every** content file — docx / xlsx / pptx / pdf / csv / md / txt / … —
for a query (literal 語 or 正規表現) and returns each hit as ``file + 位置`` (sheet / page / slide /
cell / line). It returns the unified :mod:`src.rag.tools.contract` shape ``{value, evidence, method}``
so the generation layer can call it exactly like the other tools.

NFC everywhere (受け入れ条件)
---------------------------
The share drive is NFD-normalized (macOS origin), so both **filenames** and **text** are matched
after ``unicodedata.normalize("NFC", …)``. File discovery goes through :func:`src.rag.corpus.walk`
(already NFC-aware) and every candidate line is NFC-normalized before matching, so a query typed in
NFC finds content and filenames that live on disk in NFD.

Location (``file + cell/page/line``)
------------------------------------
Content is read via the existing :func:`src.rag.extract.extract` dispatch, whose text embeds
structural markers (``[ページN]`` for PDFs, ``[シート: …]`` for xlsx, ``[スライドN]`` for pptx,
``A1(色): 値`` for highlighted cells). :func:`file_grep` tracks the nearest such marker as it scans,
so each hit carries a best-effort ``where`` locator (``page:2`` / ``sheet:train`` / ``slide:3`` /
``cell:A1`` / ``line:12``) alongside the 1-based ``line`` index within the extracted text.
"""
from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Iterable

from src.rag.corpus import FileRef, nfc, walk
from src.rag.extract import extract as _extract
from src.rag.tools import contract


class FileGrepError(ValueError):
    """Raised for an invalid query (e.g. a malformed regular expression)."""


# Extensions whose *content* is searched. Anything else is matched by filename only. This is the
# union of what :func:`src.rag.extract.extract` can turn into text (Office + pdf + tabular + text).
SEARCHABLE_EXT = {
    "docx", "xlsx", "xlsm", "pptx", "pdf", "csv", "tsv",
    "md", "txt", "json", "ipynb", "py", "toml", "lock",
}

# Structural markers emitted by the extract layer → the current location context while scanning.
_PAGE_RE = re.compile(r"^\[ページ\s*(?P<n>\d+)\]")            # extract.plain.extract_pdf
_SHEET_RE = re.compile(r"^\[シート:\s*(?P<name>.*?)\]")        # extract.office.extract_xlsx
_SLIDE_RE = re.compile(r"^\[スライド\s*(?P<n>\d+)\]")          # extract.office.extract_pptx
# A highlighted-cell line ("  A1(オレンジ): 値") or a plain "A1: …" carries an A1 coordinate.
_CELL_RE = re.compile(r"^\s*(?P<cell>[A-Z]{1,3}\d+)\s*[(:]")

_SNIPPET_MAX = 300


def _norm_exts(ext: str | Iterable[str] | None) -> set[str] | None:
    if ext is None:
        return None
    items = [ext] if isinstance(ext, str) else list(ext)
    return {e.lower().lstrip(".") for e in items if e}


def _compile(query: str, *, regex: bool, ignore_case: bool) -> re.Pattern[str]:
    flags = re.IGNORECASE if ignore_case else 0
    q = nfc(query)
    if not q:
        raise FileGrepError("empty query")
    if regex:
        try:
            return re.compile(q, flags)
        except re.error as e:
            raise FileGrepError(f"invalid regex {query!r}: {e}") from e
    return re.compile(re.escape(q), flags)


def _where(cell: str | None, page: int | None, sheet: str | None,
           slide: int | None, lineno: int) -> str:
    """Most-specific available locator for a hit (cell > page/sheet/slide > line)."""
    if cell:
        return f"cell:{cell}" + (f"@sheet:{sheet}" if sheet else "")
    if page is not None:
        return f"page:{page}"
    if sheet is not None:
        return f"sheet:{sheet}"
    if slide is not None:
        return f"slide:{slide}"
    return f"line:{lineno}"


def _scan_text(text: str, pattern: re.Pattern[str], ref: FileRef,
               *, max_hits: int) -> list[dict[str, Any]]:
    """Yield hit records for every matching line, tracking the current location context."""
    hits: list[dict[str, Any]] = []
    page: int | None = None
    sheet: str | None = None
    slide: int | None = None
    for lineno, raw in enumerate(text.split("\n"), 1):
        line = nfc(raw)
        # update location context from any structural marker on this line
        m = _PAGE_RE.match(line)
        if m:
            page, sheet, slide = int(m["n"]), None, None
        elif (m := _SHEET_RE.match(line)):
            sheet, page, slide = m["name"].strip(), None, None
        elif (m := _SLIDE_RE.match(line)):
            slide, page, sheet = int(m["n"]), None, None
        if not pattern.search(line):
            continue
        cm = _CELL_RE.match(line)
        cell = cm["cell"] if cm else None
        snippet = line.strip()
        if len(snippet) > _SNIPPET_MAX:
            snippet = snippet[:_SNIPPET_MAX] + "…"
        hits.append({
            "file": ref.rel or ref.name,
            "project": ref.project,
            "ext": ref.ext,
            "line": lineno,
            "where": _where(cell, page, sheet, slide, lineno),
            "snippet": snippet,
        })
        if len(hits) >= max_hits:
            break
    return hits


def file_grep(query: str, *, regex: bool = False, ext: str | Iterable[str] | None = None,
              project: str | None = None, ignore_case: bool = True,
              match_filename: bool = True, max_hits_per_file: int = 50,
              limit: int = 500, corpus_dir: str | Path | None = None) -> dict[str, Any]:
    """Grep the corpus for ``query`` and return contract ``{value, evidence, method}``.

    Parameters
    ----------
    query
        A literal substring (default) or, when ``regex=True``, a Python regular expression.
        NFC-normalized before matching.
    regex
        Treat ``query`` as a regular expression rather than a literal.
    ext, project
        Restrict the scan to files of an extension (str or iterable) and/or a project/company
        (NFC-matched against ``FileRef.project``).
    ignore_case
        Case-insensitive matching (default True; harmless for Japanese, friendly for ASCII tokens).
    match_filename
        Also match the query against each file's NFC path/name (a filename-only hit has
        ``line=None`` and ``where="filename"``).
    max_hits_per_file, limit
        Per-file and total caps on returned hits (``evidence.truncated`` flags an overflow).
    corpus_dir
        Override the corpus root (defaults to ``settings.CORPUS_DIR``); mainly for tests.

    ``value`` is a list of hit records ``{file, project, ext, line, where, snippet}``.
    """
    pattern = _compile(query, regex=regex, ignore_case=ignore_case)
    exts = _norm_exts(ext)
    proj = nfc(project) if project else None

    refs = walk(Path(corpus_dir)) if corpus_dir is not None else walk()
    all_hits: list[dict[str, Any]] = []
    files_scanned = 0
    matched_files: set[str] = set()

    for ref in refs:
        if exts is not None and ref.ext not in exts:
            continue
        if proj is not None and nfc(ref.project) != proj:
            continue
        files_scanned += 1
        file_hits: list[dict[str, Any]] = []

        if match_filename and pattern.search(nfc(ref.rel)):
            file_hits.append({
                "file": ref.rel or ref.name,
                "project": ref.project,
                "ext": ref.ext,
                "line": None,
                "where": "filename",
                "snippet": ref.rel or ref.name,
            })

        if ref.ext in SEARCHABLE_EXT and len(file_hits) < max_hits_per_file:
            doc = _extract(ref)
            if doc.text:
                file_hits.extend(
                    _scan_text(doc.text, pattern, ref,
                               max_hits=max_hits_per_file - len(file_hits)))

        if file_hits:
            matched_files.add(ref.rel or ref.name)
            all_hits.extend(file_hits)

    total_hits = len(all_hits)
    returned = all_hits[:limit]
    return contract.make(
        returned,
        engine="file_grep",
        evidence={
            "files_scanned": files_scanned,
            "files_matched": len(matched_files),
            "total_hits": total_hits,
            "returned": len(returned),
            "truncated": total_hits > len(returned),
        },
        query=query,
        regex=regex,
        ignore_case=ignore_case,
        filters={"ext": sorted(exts) if exts else None, "project": proj},
    )
