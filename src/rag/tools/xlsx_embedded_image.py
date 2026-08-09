"""xlsx_embedded_image — read highlighted cells from a screenshot *embedded as a picture* in a sheet.

Some corpus workbooks store a table not as cells but as a **picture pasted onto a sheet** (e.g. a
PivotTable screenshot exported as an EMF and anchored on ``Sheet2``).  openpyxl then reports the sheet
as empty (no ``sheetData``), so ``read_office`` / ``highlight_extract`` conclude "the sheet is empty
with no highlighted cells" — a false negative even though the picture clearly contains a yellow-marked
cell and its surrounding table (SOT-2548 idx80: *Sheet2 誤空判定*).

This module surfaces those embedded pictures.  It walks the ``.xlsx`` OPC relationships
(``workbook → sheet → drawing → media``) to recover each sheet's picture bytes, renders a **vector**
picture (EMF/WMF) to SVG with the system ``emf2svg-conv`` tool, and reconstructs the highlighted cells
**deterministically** from the SVG geometry — no vision model, no corpus-specific colour or value is
hard-coded:

* a *highlighted cell* is a solid-filled rectangle whose colour classifies via the same
  :func:`src.rag.extract.office._color_name` map the cell extractor uses, sized like one table cell
  (so the full-width header band, the pivot expand/collapse glyphs and the 1-px grid lines are all
  excluded by geometry, not by a colour allow-list);
* its **extraction condition** is the pivot *grouping path* — for every column to its left, the nearest
  preceding non-empty text in that column band (a fill-forward of merged/grouped group headers);
* its **aggregation content** is the column header the highlighted cell sits under, plus the cell's own
  value.

The result reuses the :mod:`src.rag.tools.contract` per-item shape so it can be merged straight into
:func:`src.rag.tools.highlight_extract.highlight_extract`'s ``value`` list.

Everything here is gated behind ``RAG_XLSX_EMBEDDED_IMAGE`` (default OFF), mirroring the
``RAG_PDF_OCR`` / ``RAG_STRUCTURE_STORE`` convention: the champion serve path stays byte-identical unless
the flag is set, and the end-to-end effect is measured with the flag ON in the consolidated gold run.
"""
from __future__ import annotations

import os
import re
import shutil
import subprocess
import tempfile
import zipfile
from pathlib import Path
from typing import Any
from xml.etree import ElementTree as ET

from src.rag.corpus import FileRef, nfc
from src.rag.extract import office as _office

_ON = {"1", "true", "yes", "on"}

# OPC relationship namespaces used to walk workbook → sheet → drawing → media.
_NS_PKG_REL = "http://schemas.openxmlformats.org/package/2006/relationships"
_NS_SPREADSHEET = "http://schemas.openxmlformats.org/spreadsheetml/2006/main"
_NS_DOC_REL = "http://schemas.openxmlformats.org/officeDocument/2006/relationships"

# Vector picture formats we can rasterise to SVG deterministically (emf2svg-conv handles EMF).
_VECTOR_EXTS = {"emf"}


def enabled() -> bool:
    """True when the embedded-picture highlight reader is active (default OFF, like RAG_PDF_OCR)."""
    return os.getenv("RAG_XLSX_EMBEDDED_IMAGE", "0").strip().lower() in _ON


# --------------------------------------------------------------------------- OPC relationship walk
def _rel_targets(zf: zipfile.ZipFile, rels_path: str, base_dir: str) -> dict[str, str]:
    """Map ``Id -> resolved archive path`` for one ``*.rels`` part (``base_dir``-relative targets)."""
    try:
        root = ET.fromstring(zf.read(rels_path))
    except (KeyError, ET.ParseError):
        return {}
    out: dict[str, str] = {}
    for rel in root.findall(f"{{{_NS_PKG_REL}}}Relationship"):
        rid = rel.get("Id")
        target = rel.get("Target")
        if not rid or not target:
            continue
        out[rid] = os.path.normpath(os.path.join(base_dir, target)).replace(os.sep, "/")
    return out


def _sheet_parts(zf: zipfile.ZipFile) -> list[tuple[str, str]]:
    """Return ``(sheet_title, worksheet_archive_path)`` for every worksheet, in workbook order."""
    try:
        wb = ET.fromstring(zf.read("xl/workbook.xml"))
    except (KeyError, ET.ParseError):
        return []
    rels = _rel_targets(zf, "xl/_rels/workbook.xml.rels", "xl")
    out: list[tuple[str, str]] = []
    for sheet in wb.findall(f"{{{_NS_SPREADSHEET}}}sheets/{{{_NS_SPREADSHEET}}}sheet"):
        title = sheet.get("name") or ""
        rid = sheet.get(f"{{{_NS_DOC_REL}}}id")
        target = rels.get(rid or "")
        if target:
            out.append((title, target))
    return out


def _drawing_images(zf: zipfile.ZipFile, sheet_path: str) -> list[str]:
    """Archive paths of every image referenced by the sheet's drawing part(s)."""
    sheet_dir = os.path.dirname(sheet_path)
    sheet_rels = _rel_targets(
        zf, f"{sheet_dir}/_rels/{os.path.basename(sheet_path)}.rels", sheet_dir)
    images: list[str] = []
    for target in sheet_rels.values():
        if "/drawings/" not in target or not target.endswith(".xml"):
            continue
        draw_dir = os.path.dirname(target)
        draw_rels = _rel_targets(
            zf, f"{draw_dir}/_rels/{os.path.basename(target)}.rels", draw_dir)
        for media in draw_rels.values():
            if "/media/" in media:
                images.append(media)
    return images


def embedded_sheet_images(ref: FileRef, data: bytes | None) -> list[tuple[str, str, bytes]]:
    """Every picture embedded on a worksheet as ``(sheet_title, image_name, image_bytes)``.

    ``data`` carries decrypted bytes for an encrypted workbook (else the path is opened directly).
    Returns ``[]`` for a non-xlsx/xlsm file or a workbook with no embedded pictures.
    """
    if ref.ext not in ("xlsx", "xlsm"):
        return []
    src: Any = ref.path
    if data is not None:
        import io
        src = io.BytesIO(data)
    out: list[tuple[str, str, bytes]] = []
    try:
        with zipfile.ZipFile(src) as zf:
            names = set(zf.namelist())
            for title, sheet_path in _sheet_parts(zf):
                for media in _drawing_images(zf, sheet_path):
                    if media in names:
                        out.append((title, os.path.basename(media), zf.read(media)))
    except (zipfile.BadZipFile, OSError):
        return []
    return out


# --------------------------------------------------------------------------- vector rendering
def _emf_to_svg(image_bytes: bytes) -> str | None:
    """Render EMF picture bytes to an SVG string via ``emf2svg-conv``; ``None`` when unavailable."""
    tool = shutil.which("emf2svg-conv")
    if tool is None:
        return None
    with tempfile.TemporaryDirectory() as tmp:
        emf = Path(tmp) / "in.emf"
        svg = Path(tmp) / "out.svg"
        emf.write_bytes(image_bytes)
        try:
            subprocess.run([tool, "--input", str(emf), "--output", str(svg)],
                           check=True, capture_output=True, timeout=120)
        except (subprocess.SubprocessError, OSError):
            return None
        try:
            return svg.read_text(encoding="utf-8", errors="replace")
        except OSError:
            return None


# --------------------------------------------------------------------------- SVG table reconstruction
_FILL_RE = re.compile(r'fill:#([0-9a-fA-F]{6})"\s+d="([^"]+)"')
_TEXT_RE = re.compile(
    r'<text[^>]*?\bx="([\d.]+)"[^>]*?\by="([\d.]+)"[^>]*?>'
    r'<!\[CDATA\[(.*?)\]\]></text>', re.S)
_NUM_RE = re.compile(r"-?\d+(?:\.\d+)?")


def _rects(svg: str) -> list[tuple[str, float, float, float, float]]:
    """Every filled path as ``(hexcolor, x0, y0, x1, y1)`` (its axis-aligned bounding box)."""
    out: list[tuple[str, float, float, float, float]] = []
    for m in _FILL_RE.finditer(svg):
        nums = [float(n) for n in _NUM_RE.findall(m.group(2))]
        xs, ys = nums[0::2], nums[1::2]
        if xs and ys:
            out.append((m.group(1).lower(), min(xs), min(ys), max(xs), max(ys)))
    return out


def _texts(svg: str) -> list[tuple[float, float, str]]:
    """Every text run as ``(x, y, text)`` with whitespace-normalised, NFC text (drops empties)."""
    out: list[tuple[float, float, str]] = []
    for m in _TEXT_RE.finditer(svg):
        text = nfc(" ".join(m.group(3).split()))
        if text:
            out.append((float(m.group(1)), float(m.group(2)), text))
    return out


def _highlight_items_from_svg(svg: str, *, file: str, sheet: str) -> list[dict[str, Any]]:
    """Reconstruct highlighted table cells (with pivot grouping context) from a rendered SVG."""
    texts = _texts(svg)
    if not texts:
        return []
    texts.sort(key=lambda t: (t[1], t[0]))
    header_y = texts[0][1]
    # Header row = every text sharing the topmost baseline; its x positions are the column left edges.
    header = sorted((x, t) for x, y, t in texts if abs(y - header_y) < 8.0)
    if len(header) < 2:
        return []
    col_x = [x for x, _t in header]
    col_name = [t for _x, t in header]
    page_w = max(x1 for _c, _x0, _y0, x1, _y1 in _rects(svg)) if _rects(svg) else col_x[-1]
    row_h = header_y * 1.6 if header_y else 24.0  # header baseline ≈ one row height from the top edge

    def band(x: float) -> int:
        idx = 0
        for i, cx in enumerate(col_x):
            if x >= cx - 5.0:
                idx = i
        return idx

    def context(ybase: float) -> dict[str, str]:
        """Fill-forward: nearest preceding non-empty text in each column band at/above ``ybase``."""
        ctx: dict[str, str] = {}
        for i, cx in enumerate(col_x):
            cx_max = col_x[i + 1] - 5.0 if i + 1 < len(col_x) else float("inf")
            best: tuple[float, str] | None = None
            for x, y, t in texts:
                if cx - 5.0 <= x < cx_max and header_y + 8.0 < y <= ybase + row_h * 0.5:
                    if best is None or y > best[0]:
                        best = (y, t)
            if best is not None:
                ctx[col_name[i]] = best[1]
        return ctx

    items: list[dict[str, Any]] = []
    for color, x0, y0, x1, y1 in _rects(svg):
        name = _office._color_name(color)
        if name is None:
            continue
        w, h = x1 - x0, y1 - y0
        # A highlighted cell is one table cell: exclude the full-width header band, the tiny pivot
        # expand/collapse glyphs and the 1-px grid lines purely by geometry (portable, no allow-list).
        if not (0.5 * row_h <= h <= 2.0 * row_h and 40.0 <= w <= 0.9 * page_w):
            continue
        ybase = (y0 + y1) / 2.0
        ctx = context(ybase)
        col = band((x0 + x1) / 2.0)
        agg_col = col_name[col]
        value = ctx.get(agg_col, "")
        condition = {k: v for k, v in ctx.items() if k != agg_col}
        items.append({
            "value": value,
            "evidence": {
                "file": file, "sheet": sheet, "source": "embedded_image",
                "aggregation_column": agg_col,
                "extraction_condition": condition,
                "cell_context": ctx,
            },
            "method": {"engine": "xlsx_embedded_image", "color": name, "kind": "embedded_cell"},
        })
    return items


# --------------------------------------------------------------------------- public API
def embedded_highlight_items(ref: FileRef, data: bytes | None) -> list[dict[str, Any]]:
    """Highlighted cells reconstructed from pictures embedded on the workbook's sheets.

    Returns per-item contracts (``{value, evidence, method}``) in the same shape as
    :func:`src.rag.tools.highlight_extract`'s cell items, so the caller can merge them directly.
    ``[]`` when the flag is off, the file has no vector embedded picture, or ``emf2svg-conv`` is
    unavailable — a pure additive path that never breaks the existing cell-grid extraction.
    """
    if not enabled():
        return []
    items: list[dict[str, Any]] = []
    for sheet, name, image_bytes in embedded_sheet_images(ref, data):
        ext = name.rsplit(".", 1)[-1].lower()
        if ext not in _VECTOR_EXTS:
            continue
        svg = _emf_to_svg(image_bytes)
        if not svg:
            continue
        items.extend(_highlight_items_from_svg(
            svg, file=ref.rel or str(ref.path), sheet=sheet))
    return items
