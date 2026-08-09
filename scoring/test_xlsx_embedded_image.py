"""SOT-2548: highlighted-cell recovery from a picture *embedded on a worksheet* (``xlsx_embedded_image``).

A sheet can hold a table as a pasted picture (an embedded EMF screenshot) instead of real cells, so
openpyxl reports it empty and the grid highlight scan finds nothing (idx80 *Sheet2 誤空判定*). This pins:
  ① the OPC walk recovers the picture bytes anchored on a sheet (workbook → sheet → drawing → media);
  ② the deterministic SVG reconstruction isolates one highlighted *cell* by geometry (the full-width
     header band, the tiny pivot glyphs and the grid lines are excluded) and resolves its pivot grouping
     path (fill-forward of the sparse leading columns) + its aggregation column — no hard-coded value;
  ③ the whole path is gated OFF by default (champion serve stays byte-identical), plus a guarded smoke
     test over the real share drive where 東都人材プラットフォーム/train.xlsx Sheet2 carries the pivot.
"""
from __future__ import annotations

import os
import shutil
import zipfile
from pathlib import Path

import pytest

from src.rag.corpus import walk
from src.rag.tools import xlsx_embedded_image as xei
from src.rag.tools.extract_tools import resolve_ref


# --------------------------------------------------------------------------- synthetic SVG (emf2svg shape)
def _fill(hexc: str, x0: float, y0: float, x1: float, y1: float) -> str:
    return (f'<path style="fill:#{hexc}" d="M {x0},{y0} L {x1},{y0} '
            f'L {x1},{y1} L {x0},{y1} Z" />')


def _text(x: float, y: float, s: str) -> str:
    return (f'<text clip-path="url(#c)" font-family="X" fill="#000000" style ="white-space:pre;" '
            f'font-weight="400" text-anchor="start" x="{x}" y="{y}" font-size="22.0" >'
            f'<![CDATA[{s}]]></text>')


def _pivot_svg() -> str:
    cols = [(5.0, "Gender"), (268.0, "target"), (531.0, "Age"),
            (812.0, "Profession"), (1095.0, "個数 / id")]
    parts = ['<svg xmlns="http://www.w3.org/2000/svg">']
    parts.append(_fill("c0e6f5", 0, 0, 1194, 37))       # full-width header band -> excluded (too wide)
    parts.append(_fill("aab5c9", 250, 50, 262, 62))     # tiny pivot expand glyph -> excluded (too small)
    parts.append(_fill("e0e0e0", 0, 90, 1194, 91))      # grid line, grey -> _color_name None
    parts.append(_fill("ffff00", 1090, 445, 1194, 482))  # the yellow highlighted 個数 cell
    for x, name in cols:                                  # header row (topmost baseline)
        parts.append(_text(x + 2, 24.0, name))
    # Sparse grouping cells (as a PivotTable renders merged group headers only once):
    parts.append(_text(15.0, 100.0, "Male"))
    parts.append(_text(278.0, 100.0, "3"))
    parts.append(_text(541.0, 200.0, "30-34"))
    parts.append(_text(822.0, 460.0, "Software Engineer"))
    parts.append(_text(1100.0, 460.0, "61"))             # the value inside the yellow cell's row/col
    parts.append("</svg>")
    return "".join(parts)


def test_svg_reconstruction_isolates_cell_and_resolves_grouping():
    items = xei._highlight_items_from_svg(_pivot_svg(), file="train.xlsx", sheet="Sheet2")
    assert len(items) == 1                                # header band / glyph / grid line all excluded
    item = items[0]
    assert item["method"] == {"engine": "xlsx_embedded_image", "color": "黄",
                              "kind": "embedded_cell"}
    assert item["value"] == "61"
    ev = item["evidence"]
    assert ev["sheet"] == "Sheet2" and ev["source"] == "embedded_image"
    assert "個数" in ev["aggregation_column"]
    assert ev["extraction_condition"] == {
        "Gender": "Male", "target": "3", "Age": "30-34", "Profession": "Software Engineer"}


# --------------------------------------------------------------------------- OPC relationship walk
def _xlsx_with_embedded_emf(path: Path, sheet_title: str, emf_bytes: bytes) -> Path:
    """Hand-build a minimal 1-sheet .xlsx whose sheet anchors a drawing referencing an embedded EMF."""
    R = "http://schemas.openxmlformats.org/package/2006/relationships"
    RD = "http://schemas.openxmlformats.org/officeDocument/2006/relationships"
    SS = "http://schemas.openxmlformats.org/spreadsheetml/2006/main"
    with zipfile.ZipFile(path, "w") as z:
        z.writestr("[Content_Types].xml",
                   '<?xml version="1.0"?><Types xmlns="http://schemas.openxmlformats.org/package/'
                   '2006/content-types"><Default Extension="emf" ContentType="image/x-emf"/>'
                   '<Default Extension="rels" ContentType="application/vnd.openxmlformats-package.'
                   'relationships+xml"/></Types>')
        z.writestr("_rels/.rels",
                   f'<?xml version="1.0"?><Relationships xmlns="{R}"><Relationship Id="rId1" '
                   f'Type="{RD}/officeDocument" Target="xl/workbook.xml"/></Relationships>')
        z.writestr("xl/workbook.xml",
                   f'<?xml version="1.0"?><workbook xmlns="{SS}" xmlns:r="{RD}"><sheets>'
                   f'<sheet name="{sheet_title}" sheetId="1" r:id="rId1"/></sheets></workbook>')
        z.writestr("xl/_rels/workbook.xml.rels",
                   f'<?xml version="1.0"?><Relationships xmlns="{R}"><Relationship Id="rId1" '
                   f'Type="{RD}/worksheet" Target="worksheets/sheet1.xml"/></Relationships>')
        z.writestr("xl/worksheets/sheet1.xml",
                   f'<?xml version="1.0"?><worksheet xmlns="{SS}" xmlns:r="{RD}"><sheetData/>'
                   f'<drawing r:id="rId1"/></worksheet>')
        z.writestr("xl/worksheets/_rels/sheet1.xml.rels",
                   f'<?xml version="1.0"?><Relationships xmlns="{R}"><Relationship Id="rId1" '
                   f'Type="{RD}/drawing" Target="../drawings/drawing1.xml"/></Relationships>')
        z.writestr("xl/drawings/drawing1.xml",
                   '<?xml version="1.0"?><xdr:wsDr xmlns:xdr="http://schemas.openxmlformats.org/'
                   'drawingml/2006/spreadsheetDrawing"/>')
        z.writestr("xl/drawings/_rels/drawing1.xml.rels",
                   f'<?xml version="1.0"?><Relationships xmlns="{R}"><Relationship Id="rId1" '
                   f'Type="{RD}/image" Target="../media/image1.emf"/></Relationships>')
        z.writestr("xl/media/image1.emf", emf_bytes)
    return path


def test_embedded_sheet_images_walks_opc_relationships(tmp_path):
    p = _xlsx_with_embedded_emf(tmp_path / "s.xlsx", "Sheet2", b"\x01\x00\x00\x00EMFDATA")
    imgs = xei.embedded_sheet_images(resolve_ref(p), None)
    assert len(imgs) == 1
    sheet, name, data = imgs[0]
    assert sheet == "Sheet2" and name == "image1.emf" and data == b"\x01\x00\x00\x00EMFDATA"


def test_embedded_highlight_items_gated_off_by_default(tmp_path, monkeypatch):
    monkeypatch.delenv("RAG_XLSX_EMBEDDED_IMAGE", raising=False)
    p = _xlsx_with_embedded_emf(tmp_path / "s.xlsx", "Sheet2", b"emf")
    assert xei.embedded_highlight_items(resolve_ref(p), None) == []  # flag off → no-op


def test_embedded_highlight_items_degrades_without_renderer(tmp_path, monkeypatch):
    monkeypatch.setenv("RAG_XLSX_EMBEDDED_IMAGE", "1")
    monkeypatch.setattr(xei.shutil, "which", lambda _tool: None)  # emf2svg-conv unavailable
    p = _xlsx_with_embedded_emf(tmp_path / "s.xlsx", "Sheet2", b"emf")
    assert xei.embedded_highlight_items(resolve_ref(p), None) == []  # graceful, never raises


# --------------------------------------------------------------------------- real-corpus smoke (idx80)
def _corpus_ref(*substrs: str, ext: str):
    for r in walk():
        if r.ext == ext and all(s in r.rel for s in substrs):
            return r
    return None


@pytest.mark.skipif(
    _corpus_ref("東都人材", "train", ext="xlsx") is None or shutil.which("emf2svg-conv") is None,
    reason="share drive or emf2svg-conv not present")
def test_real_corpus_sheet2_pivot_highlight_smoke(monkeypatch):
    monkeypatch.setenv("RAG_XLSX_EMBEDDED_IMAGE", "1")
    ref = _corpus_ref("東都人材", "train", ext="xlsx")
    items = xei.embedded_highlight_items(ref, None)
    yellow = [it for it in items if it["method"]["color"] == "黄"]
    assert yellow, "Sheet2's embedded pivot must yield the yellow-highlighted cell"
    cond = yellow[0]["evidence"]["extraction_condition"]
    assert cond["Gender"] == "Male" and cond["target"] == "3"
    assert cond["Age"] == "30-34" and cond["Profession"] == "Software Engineer"
    assert "個数" in yellow[0]["evidence"]["aggregation_column"]
