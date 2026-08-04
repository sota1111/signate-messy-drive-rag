"""Office extraction (docx / xlsx / pptx) that PRESERVES formatting signals.

Many questions depend on visual formatting — bold runs in contracts, orange/yellow
highlighted cells/rows, PivotTable conditions. We surface these as explicit, searchable
annotations (e.g. "【太字】…", "【ハイライト:オレンジ】…") rather than flattening to plain text.
Encrypted files are transparently decrypted via extract.passwords.
"""
from __future__ import annotations

import io

import docx
import openpyxl
from openpyxl.styles.colors import COLOR_INDEX
from openpyxl.utils import get_column_letter
from pptx import Presentation

from src.rag.corpus import FileRef, nfc

# ---- ARGB → coarse colour name (highlight questions say オレンジ/黄/…) ----
# Exact matches for common theme / conditional-format fills (override the HSV heuristic).
_COLOR_NAMES = {
    "FFFF00": "黄",
    "FFA500": "オレンジ", "ED7D31": "オレンジ", "FFC000": "オレンジ", "F4B183": "オレンジ",
    "FF0000": "赤", "00B050": "緑", "92D050": "緑", "00FF00": "緑",
    "0070C0": "青", "00B0F0": "水色",
    "7030A0": "紫", "B4A7D6": "紫",
    "FFC7CE": "赤", "C6EFCE": "緑", "FFEB9C": "黄",  # Excel conditional-format palette
}


def _hue_name(h: float) -> str:
    """Map hue degrees (0-360) to a coarse Japanese colour name."""
    if h < 18 or h >= 345:
        return "赤"
    if h < 45:
        return "オレンジ"
    if h < 70:
        return "黄"
    if h < 90:
        return "黄緑"
    if h < 160:
        return "緑"
    if h < 200:
        return "水色"
    if h < 255:
        return "青"
    if h < 290:
        return "紫"
    return "ピンク"


def _color_name(argb: str | None) -> str | None:
    """Coarse highlight colour from an ARGB/RGB hex, or None for no meaningful highlight.

    Uses HSV so pale Excel tints (e.g. F2E0D0 light peach → オレンジ) classify correctly,
    while low-saturation grays (header/table styling) and near-white/black are dropped.
    """
    import colorsys

    if not argb:
        return None
    hex6 = str(argb).upper()[-6:]
    if hex6 in _COLOR_NAMES:
        return _COLOR_NAMES[hex6]
    try:
        r, g, b = (int(hex6[i:i + 2], 16) / 255 for i in (0, 2, 4))
    except ValueError:
        return None
    h, s, v = colorsys.rgb_to_hsv(r, g, b)
    # drop grayscale styling and near-white/near-black backgrounds
    if s < 0.12 or v < 0.20 or (v > 0.96 and s < 0.06):
        return None
    return _hue_name(h * 360)


def _excel_color_name(color) -> str | None:
    """Resolve direct and indexed Excel fills through the same deterministic HSV classifier."""
    if color is None:
        return None
    kind = getattr(color, "type", None)
    if kind == "rgb":
        return _color_name(getattr(color, "rgb", None))
    if kind == "indexed":
        try:
            index = int(color.indexed)
            return _color_name(COLOR_INDEX[index]) if 0 <= index < len(COLOR_INDEX) else None
        except (TypeError, ValueError):
            return None
    # Theme colours depend on the workbook theme. Treating their integer index as RGB caused
    # unstable false highlights, so unresolved theme/auto colours deliberately fail closed.
    return None


# ---------------- DOCX ----------------
def raw_docx_text(path) -> str:
    """Plain text of a (non-encrypted) docx; returns '' if unreadable. No decryption."""
    try:
        d = docx.Document(str(path))
    except Exception:
        return ""
    out = [p.text for p in d.paragraphs if p.text.strip()]
    for t in d.tables:
        for r in t.rows:
            out.append(" | ".join(c.text for c in r.cells))
    return "\n".join(out)


def _docx_from_bytes(b: bytes) -> "docx.Document":
    return docx.Document(io.BytesIO(b))


def extract_docx(ref: FileRef, data: bytes | None) -> str:
    d = _docx_from_bytes(data) if data else docx.Document(str(ref.path))
    lines: list[str] = []
    bold_terms: list[str] = []
    for p in d.paragraphs:
        if not p.text.strip():
            continue
        lines.append(p.text)
        for run in p.runs:
            txt = run.text.strip()
            if txt and (run.bold or (run.font and run.font.bold)):
                bold_terms.append(txt)
            hl = getattr(run.font, "highlight_color", None)
            if txt and hl is not None:
                lines.append(f"【ハイライト】{txt}")
    for ti, t in enumerate(d.tables):
        lines.append(f"[表{ti + 1}]")
        for r in t.rows:
            lines.append(" | ".join(c.text for c in r.cells))
    if bold_terms:
        # dedupe preserving order
        seen, terms = set(), []
        for b in bold_terms:
            if b not in seen:
                seen.add(b)
                terms.append(b)
        lines.insert(0, "【太字箇所】" + " / ".join(terms))
    return "\n".join(lines)


# ---------------- XLSX ----------------
def _xlsx_from(ref: FileRef, data: bytes | None):
    src = io.BytesIO(data) if data else str(ref.path)
    return openpyxl.load_workbook(src, data_only=True)


def extract_xlsx(ref: FileRef, data: bytes | None) -> str:
    wb = _xlsx_from(ref, data)
    out: list[str] = []
    for ws in wb.worksheets:
        out.append(f"[シート: {ws.title}]  範囲 {ws.dimensions}")
        highlights: list[str] = []
        rows_repr: list[str] = []
        maxr = min(ws.max_row or 0, 400)
        for row in ws.iter_rows(max_row=maxr):
            cells = []
            for c in row:
                v = c.value
                if v is None:
                    continue
                cells.append(str(v))
                # highlight / fill
                fill = c.fill
                if fill and fill.patternType:
                    color = None
                    fg = getattr(fill, "fgColor", None)
                    if fg is not None:
                        color = _excel_color_name(fg)
                    if color:
                        highlights.append(f"{c.coordinate}({color}): {v}")
            if cells:
                rows_repr.append(" | ".join(cells))
        out.extend(rows_repr[:400])
        if highlights:
            out.append("【ハイライトされたセル】")
            out.extend(f"  {h}" for h in highlights[:200])
    return "\n".join(out)


# ---------------- PPTX ----------------
def _shape_fill_color(shape) -> str | None:
    try:
        fill = shape.fill
        if fill.type is not None and fill.fore_color and fill.fore_color.rgb is not None:
            return _color_name(str(fill.fore_color.rgb))
    except Exception:
        pass
    return None


def _run_highlight(run) -> str | None:
    # python-pptx has no direct highlight API; inspect the run XML for a:highlight
    try:
        rpr = run._r.find("{http://schemas.openxmlformats.org/drawingml/2006/main}rPr")
        if rpr is not None:
            hl = rpr.find("{http://schemas.openxmlformats.org/drawingml/2006/main}highlight")
            if hl is not None:
                srgb = hl.find("{http://schemas.openxmlformats.org/drawingml/2006/main}srgbClr")
                if srgb is not None:
                    return _color_name(srgb.get("val"))
                return "ハイライト"
    except Exception:
        pass
    return None


def extract_pptx(ref: FileRef, data: bytes | None) -> str:
    prs = Presentation(io.BytesIO(data) if data else str(ref.path))
    out: list[str] = []
    for si, slide in enumerate(prs.slides, 1):
        out.append(f"[スライド{si}]")
        # top-to-bottom, left-to-right ordering (per enumeration rules in the task)
        shapes = sorted(slide.shapes, key=lambda s: (int(getattr(s, "top", 0) or 0),
                                                     int(getattr(s, "left", 0) or 0)))
        for shape in shapes:
            fillc = _shape_fill_color(shape)
            if shape.has_text_frame:
                for para in shape.text_frame.paragraphs:
                    line = "".join(r.text for r in para.runs) or para.text
                    marks = []
                    for r in para.runs:
                        hl = _run_highlight(r)
                        if hl and r.text.strip():
                            marks.append(f"【ハイライト:{hl}】{r.text.strip()}")
                    if line.strip():
                        out.append(line)
                    out.extend(marks)
                if fillc and shape.text_frame.text.strip():
                    out.append(f"【図形塗り:{fillc}】{shape.text_frame.text.strip()[:60]}")
            if shape.has_table:
                out.append("[表]")
                for r in shape.table.rows:
                    out.append(" | ".join(c.text for c in r.cells))
    return "\n".join(out)
