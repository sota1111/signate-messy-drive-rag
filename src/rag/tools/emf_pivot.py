"""emf_pivot — recover a PivotTable's cell text/values (and its highlighted cell) from an embedded EMF.

Some corpus documents paste an Excel **PivotTable as a picture**: the表 lives inside an embedded
Enhanced Metafile (``ppt/media/imageN.emf`` in a .pptx) rather than as real cells. A plain text-layer
or vision read throws the table geometry away, so "黄色ハイライトされている数値に対応する抽出条件"
questions become unanswerable from the extracted string alone. EMF is a *vector* format, though: the
glyphs are still there as ``EMR_EXTTEXTOUTW`` records carrying the exact text and its reference point,
and cell shading is a ``PATCOPY`` ``EMR_BITBLT`` painted with a selected solid brush. This tool walks
those records to (1) reconstruct the table grid from the text positions and (2) recover each shaded
(highlighted) region and map it back to the covered cell(s) — value, row label and column header.

It returns the unified :mod:`src.rag.tools.contract` shape ``{value, evidence, method}`` so the
generation layer can call it exactly like the other tools:

* ``value``    — ``{table, n_rows, n_cols, captions, highlights}``. ``table`` is the reconstructed grid
  (row 0 = the header row, column 0 = the row labels) as a list of rows of strings; ``highlights`` is a
  list of shaded regions ``{color, rgb, bbox, cells:[{row,col,row_label,col_header,value}]}``.
* ``evidence`` — ``file`` + record/geometry provenance (n_text_records, n_rows, n_cols, bounds, n_fills).
* ``method``   — ``engine="emf_pivot"``, the clustering gaps used and the distinct highlight colors seen
  (never a secret — pure geometry/color read from the metafile).

Fallback
--------
An EMF with no text records (a *raster*-only picture) cannot be table-reconstructed. Per the design
(失敗時は vision へフォールバック) :func:`extract_emf_pivot` then routes to a vision caption: an injected
``vision_fallback(bytes)->str`` if given, else :func:`src.rag.tools.extract_tools.caption_figure` on a
sibling ``.png`` if one exists next to the EMF; if neither is available it raises :class:`ContractError`
rather than emit a fabricated table.
"""
from __future__ import annotations

import math
import struct
import zipfile
from pathlib import Path
from typing import Any, Callable, Iterable

import pandas as pd

from src.rag.corpus import FileRef, nfc
from src.rag.tools import contract
from src.rag.tools.contract import ContractError

# --- EMF record types we care about (MS-EMF §2.1.1) ---------------------------------------------
_EMR_HEADER = 1
_EMR_EOF = 14
_EMR_SELECTOBJECT = 37
_EMR_CREATEBRUSHINDIRECT = 39
_EMR_BITBLT = 76
_EMR_EXTTEXTOUTA = 83
_EMR_EXTTEXTOUTW = 84

# Raster ops whose *source* is the currently-selected brush pattern → a solid cell fill (MS-WMF §2.1.1.3).
_BRUSH_FILL_ROPS = frozenset({0x00F00021, 0x005A0049})  # PATCOPY, PATINVERT

# Stock brush handles (selected with the high bit set) we can attribute a color to (MS-WMF §2.1.1.7).
_STOCK_BRUSH_RGB = {
    0x80000000: (255, 255, 255),  # WHITE_BRUSH
    0x80000004: (0, 0, 0),        # BLACK_BRUSH
    0x80000001: (192, 192, 192),  # LTGRAY_BRUSH
    0x80000002: (128, 128, 128),  # GRAY_BRUSH
    0x80000003: (64, 64, 64),     # DKGRAY_BRUSH
}

# Fills we never treat as a highlight: the page background white, and (default) thin ≤ this-px lines —
# EMF often paints 1px grid/border strokes as tiny PATCOPY blits which are structure, not shading.
_DEFAULT_IGNORE_RGB = frozenset({(255, 255, 255)})
_DEFAULT_MIN_FILL_PX = 2

# Column/row clustering gaps (device units). Pivot cells drawn by Office sit on a ~36px row pitch with
# ~100px column pitch; numbers are right-aligned so a column's left edges spread ~30px. These defaults
# split real corpus tables cleanly and are exposed as parameters for atypical exports.
_DEFAULT_Y_GAP = 18.0
_DEFAULT_X_GAP = 50.0


class _Span:
    """One drawn text run: its NFC text and integer reference point ``(x, y)`` in device units."""

    __slots__ = ("text", "x", "y", "row", "col")

    def __init__(self, text: str, x: int, y: int) -> None:
        self.text = text
        self.x = x
        self.y = y
        self.row = -1
        self.col = -1


class _Fill:
    """One solid-brush fill region: its ``rgb`` and device-unit ``bbox`` = ``(x0, y0, x1, y1)``."""

    __slots__ = ("rgb", "bbox")

    def __init__(self, rgb: tuple[int, int, int], bbox: tuple[int, int, int, int]) -> None:
        self.rgb = rgb
        self.bbox = bbox


# --------------------------------------------------------------------------- EMF binary parse
def _decode_w(raw: bytes, n_chars: int) -> str:
    return raw[: n_chars * 2].decode("utf-16-le", "replace")


def _decode_a(raw: bytes, n_chars: int) -> str:
    b = raw[:n_chars]
    for enc in ("utf-8", "cp932", "cp1252"):
        try:
            return b.decode(enc)
        except UnicodeDecodeError:
            continue
    return b.decode("latin-1", "replace")


def _clean(text: str) -> str:
    """NFC-normalize and drop NULs / surrounding whitespace from a raw metafile string."""
    return nfc(text.replace("\x00", "").strip())


def _parse_emf(data: bytes) -> tuple[list[_Span], list[_Fill], tuple[int, int, int, int]]:
    """Walk every EMF record, returning the text spans, solid-brush fills, and the header bounds.

    Raises :class:`ContractError` if ``data`` is not a well-formed EMF (bad header / truncated record).
    """
    if len(data) < 88 or struct.unpack_from("<I", data, 0)[0] != _EMR_HEADER:
        raise ContractError("not an EMF: missing EMR_HEADER")
    signature = struct.unpack_from("<I", data, 40)[0]
    if signature != 0x464D4520:  # ' EMF'
        raise ContractError(f"not an EMF: bad signature {signature:#010x}")
    bounds = struct.unpack_from("<4i", data, 8)

    spans: list[_Span] = []
    fills: list[_Fill] = []
    brushes: dict[int, tuple[int, int, int]] = {}
    cur_rgb: tuple[int, int, int] | None = None

    off = 0
    while off + 8 <= len(data):
        itype, size = struct.unpack_from("<II", data, off)
        if size < 8 or off + size > len(data):
            raise ContractError(f"corrupt EMF record at offset {off} (type={itype}, size={size})")

        if itype in (_EMR_EXTTEXTOUTW, _EMR_EXTTEXTOUTA):
            ref_x, ref_y = struct.unpack_from("<2i", data, off + 36)
            n_chars, off_string = struct.unpack_from("<2I", data, off + 44)
            if off_string and off + off_string <= len(data):
                raw = data[off + off_string : off + size]
                text = (_decode_w if itype == _EMR_EXTTEXTOUTW else _decode_a)(raw, n_chars)
                text = _clean(text)
                if text:
                    spans.append(_Span(text, ref_x, ref_y))
        elif itype == _EMR_CREATEBRUSHINDIRECT:
            ih, style, color = struct.unpack_from("<3I", data, off + 8)
            if style == 0:  # BS_SOLID — color is 0x00BBGGRR
                brushes[ih] = (color & 0xFF, (color >> 8) & 0xFF, (color >> 16) & 0xFF)
        elif itype == _EMR_SELECTOBJECT:
            ih = struct.unpack_from("<I", data, off + 8)[0]
            if ih in brushes:
                cur_rgb = brushes[ih]
            elif ih in _STOCK_BRUSH_RGB:
                cur_rgb = _STOCK_BRUSH_RGB[ih]
        elif itype == _EMR_BITBLT:
            xd, yd, cxd, cyd = struct.unpack_from("<4i", data, off + 24)
            rop = struct.unpack_from("<I", data, off + 40)[0]
            if rop in _BRUSH_FILL_ROPS and cur_rgb is not None:
                x0, x1 = sorted((xd, xd + cxd))
                y0, y1 = sorted((yd, yd + cyd))
                fills.append(_Fill(cur_rgb, (x0, y0, x1, y1)))
        elif itype == _EMR_EOF:
            break

        off += size

    return spans, fills, bounds


# --------------------------------------------------------------------------- geometry → grid
def _cluster_1d(values: Iterable[float], gap: float) -> list[float]:
    """Sorted cluster *centers* of ``values``: a new cluster starts wherever the sorted gap > ``gap``."""
    vals = sorted(values)
    if not vals:
        return []
    centers: list[float] = []
    group = [vals[0]]
    for v in vals[1:]:
        if v - group[-1] > gap:
            centers.append(sum(group) / len(group))
            group = [v]
        else:
            group.append(v)
    centers.append(sum(group) / len(group))
    return centers


def _nearest(centers: list[float], v: float) -> int:
    return min(range(len(centers)), key=lambda i: abs(centers[i] - v))


def _build_grid(spans: list[_Span], y_gap: float, x_gap: float
                ) -> tuple[list[list[str]], list[str], list[float], list[float]]:
    """Assign each span a ``(row, col)`` and return ``(table, captions, row_centers, col_centers)``.

    Rows are the y-clusters; the *grid* is built from rows holding ≥ 2 cells (single-cell rows are
    returned separately as ``captions`` — e.g. a pivot title above the表). Columns are the x-clusters of
    those grid rows, so a stray caption never invents a column. Missing cells are filled with ``""``.
    Each span's ``row``/``col`` are set in place so highlight mapping can reuse them.
    """
    row_centers = _cluster_1d((s.y for s in spans), y_gap)
    rows: list[list[_Span]] = [[] for _ in row_centers]
    for s in spans:
        rows[_nearest(row_centers, s.y)].append(s)

    grid_row_idx = [i for i, r in enumerate(rows) if len(r) >= 2]
    captions = [_clean(rows[i][0].text) for i, r in enumerate(rows) if len(r) == 1]

    col_centers = _cluster_1d(
        (s.x for i in grid_row_idx for s in rows[i]), x_gap)
    n_cols = len(col_centers)

    table: list[list[str]] = []
    for out_row, i in enumerate(grid_row_idx):
        cells = [""] * n_cols
        for s in sorted(rows[i], key=lambda sp: sp.x):
            c = _nearest(col_centers, s.x) if n_cols else -1
            s.row, s.col = out_row, c
            if 0 <= c < n_cols:
                cells[c] = (cells[c] + " " + s.text).strip() if cells[c] else s.text
        table.append(cells)

    return table, captions, row_centers, col_centers


def _map_highlights(fills: list[_Fill], spans: list[_Span], table: list[list[str]],
                    ignore_rgb: frozenset[tuple[int, int, int]], min_fill_px: int
                    ) -> list[dict[str, Any]]:
    """Turn each meaningful solid fill into ``{color, rgb, bbox, cells:[…]}`` over the covered cells.

    Fills that are ignored-color (page white) or thinner than ``min_fill_px`` in either axis (grid/border
    strokes, not shading) are dropped. A cell is *covered* when its span's reference point lies inside the
    fill rectangle; each covered cell reports its value plus its row label (col 0) and column header (row 0).
    """
    n_rows, n_cols = len(table), (len(table[0]) if table else 0)
    out: list[dict[str, Any]] = []
    for f in fills:
        x0, y0, x1, y1 = f.bbox
        if f.rgb in ignore_rgb or (x1 - x0) < min_fill_px or (y1 - y0) < min_fill_px:
            continue
        cells: list[dict[str, Any]] = []
        for s in spans:
            if s.row < 0 or s.col < 0:
                continue
            if x0 <= s.x <= x1 and y0 <= s.y <= y1:
                cells.append({
                    "row": s.row,
                    "col": s.col,
                    "value": s.text,
                    "row_label": table[s.row][0] if 0 <= s.row < n_rows and n_cols else "",
                    "col_header": table[0][s.col] if n_rows and 0 <= s.col < n_cols else "",
                })
        cells.sort(key=lambda c: (c["row"], c["col"]))
        out.append({
            "color": "#{:02X}{:02X}{:02X}".format(*f.rgb),
            "rgb": list(f.rgb),
            "bbox": [x0, y0, x1, y1],
            "cells": cells,
        })
    return out


# --------------------------------------------------------------------------- reconstructed grid → source-data semantics
# A pasted PivotTable contains display labels and values but often omits the Excel field buttons.  Raw
# ``row_label=8 / col_header=2`` is not a meaningful extraction condition until those labels are mapped
# back to source columns.  We resolve that mapping deterministically: candidate source columns must cover
# the visible row/column label domains, then each supported aggregation is replayed and compared against
# every visible pivot cell.  Only a unique, near-exact semantic signature is accepted (otherwise fail
# closed with ``None``); no project-specific field name or answer is embedded here.
_AGGREGATIONS: tuple[tuple[str, str], ...] = (
    ("max", "最大"), ("mean", "平均"), ("min", "最小"), ("sum", "合計"),
    ("count", "個数"), ("median", "中央値"),
)


def _cell_scalar(text: Any) -> Any:
    s = nfc(str(text or "")).strip()
    if not s:
        return None
    try:
        value = float(s.replace(",", ""))
        return int(value) if value.is_integer() else value
    except ValueError:
        return s


def _scalar_key(value: Any) -> tuple[str, Any]:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return ("null", None)
    if isinstance(value, bool):
        return ("text", str(value).lower())
    try:
        number = float(value)
        if math.isfinite(number):
            return ("number", round(number, 12))
    except (TypeError, ValueError):
        pass
    return ("text", nfc(str(value)).strip())


def _json_scalar(value: Any) -> Any:
    if hasattr(value, "item") and not isinstance(value, (str, bytes)):
        try:
            value = value.item()
        except (ValueError, AttributeError):
            pass
    if isinstance(value, float) and value.is_integer():
        return int(value)
    return value


def _candidate_fields(df: pd.DataFrame, labels: list[Any]) -> list[str]:
    wanted = {_scalar_key(v) for v in labels if v is not None}
    out: list[str] = []
    for col in df.columns:
        values = {_scalar_key(v) for v in df[col].dropna().unique()}
        # Limit accidental high-cardinality candidates while allowing a partially visible pivot (e.g.
        # visible hours 0..15 from a source domain 0..23).
        if wanted and wanted <= values and len(values) <= max(64, len(wanted) * 4):
            out.append(str(col))
    return out


def _display_tolerance(text: str) -> float:
    clean = str(text).strip().replace(",", "")
    decimals = len(clean.rsplit(".", 1)[1]) if "." in clean else 0
    return max(1e-9, 0.5 * (10 ** -decimals))


def resolve_pivot_semantics(
    table: list[list[str]], data_sources: Iterable[tuple[str, pd.DataFrame]],
) -> dict[str, Any] | None:
    """Resolve ``table`` to ``row field × column field → aggregation(value field)``.

    ``data_sources`` is an iterable of ``(source label, DataFrame)``.  A result is returned only when
    one semantic signature explains at least four cells and ≥98% of all numeric visible cells, and no
    different signature ties it.  Duplicate copies of the same source table collapse to one signature.
    """
    if len(table) < 2 or not table[0] or len(table[0]) < 2:
        return None
    column_labels = [_cell_scalar(x) for x in table[0][1:]]
    row_labels = [_cell_scalar(row[0]) for row in table[1:] if row]
    if not row_labels or not column_labels:
        return None

    candidates: list[dict[str, Any]] = []
    for source, frame in data_sources:
        if frame.empty:
            continue
        df = frame.copy()
        df.columns = [nfc(str(c)) for c in df.columns]
        row_fields = _candidate_fields(df, row_labels)
        column_fields = _candidate_fields(df, column_labels)
        numeric_fields = [str(c) for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
        for row_field in row_fields:
            for column_field in column_fields:
                if row_field == column_field:
                    continue
                row_lookup = {_scalar_key(v): v for v in df[row_field].dropna().unique()}
                col_lookup = {_scalar_key(v): v for v in df[column_field].dropna().unique()}
                for target in numeric_fields:
                    if target in (row_field, column_field):
                        continue
                    for aggregation, aggregation_label in _AGGREGATIONS:
                        try:
                            pivot = pd.pivot_table(
                                df, index=row_field, columns=column_field, values=target,
                                aggfunc=aggregation, observed=False)
                        except (KeyError, TypeError, ValueError):
                            continue
                        compared = matched = 0
                        for rpos, row in enumerate(table[1:]):
                            if not row or rpos >= len(row_labels):
                                continue
                            rvalue = row_lookup.get(_scalar_key(row_labels[rpos]))
                            if rvalue is None or rvalue not in pivot.index:
                                continue
                            for cpos, raw in enumerate(row[1:]):
                                expected = _cell_scalar(raw)
                                if not isinstance(expected, (int, float)) or cpos >= len(column_labels):
                                    continue
                                cvalue = col_lookup.get(_scalar_key(column_labels[cpos]))
                                if cvalue is None or cvalue not in pivot.columns:
                                    continue
                                actual = pivot.loc[rvalue, cvalue]
                                if pd.isna(actual):
                                    continue
                                compared += 1
                                if abs(float(actual) - float(expected)) <= _display_tolerance(raw):
                                    matched += 1
                        if compared >= 4:
                            candidates.append({
                                "source_file": source,
                                "row_field": row_field,
                                "column_field": column_field,
                                "target_column": target,
                                "aggregation": aggregation,
                                "aggregation_label": aggregation_label,
                                "matched_cells": matched,
                                "compared_cells": compared,
                                "match_rate": matched / compared,
                            })

    viable = [c for c in candidates if c["match_rate"] >= 0.98]
    if not viable:
        return None
    viable.sort(key=lambda c: (c["match_rate"], c["matched_cells"], -len(c["source_file"])),
                reverse=True)
    best = viable[0]
    signature = lambda c: (c["row_field"], c["column_field"], c["target_column"], c["aggregation"])
    tied_signatures = {
        signature(c) for c in viable
        if c["match_rate"] == best["match_rate"] and c["matched_cells"] == best["matched_cells"]
    }
    if len(tied_signatures) != 1:
        return None
    best = dict(best)
    best["match_rate"] = round(float(best["match_rate"]), 6)
    return best


def _pptx_data_sources(pptx: str | Path | FileRef) -> list[tuple[str, pd.DataFrame]]:
    """Load likely tabular sources from the PPTX's project, preferring canonical ``03.データ`` files."""
    from src.rag.corpus import walk
    from src.rag.tools.compute_sandbox import _read_frame
    from src.rag.tools.extract_tools import resolve_ref

    if isinstance(pptx, FileRef):
        pptx_ref = pptx
    else:
        path = Path(pptx)
        if path.exists():
            normalized = nfc(str(path.resolve()))
            pptx_ref = next(
                (r for r in walk() if nfc(str(r.path.resolve())) == normalized), None)
            if pptx_ref is None:
                return []
        else:
            pptx_ref = resolve_ref(pptx)
    if not pptx_ref.project:
        return []
    refs = [r for r in walk() if r.project == pptx_ref.project
            and r.ext in {"csv", "tsv", "xlsx", "xlsm"}]
    refs.sort(key=lambda r: (
        r.category != "data", not r.stem.lower().startswith("train"), len(r.rel), r.rel))
    out: list[tuple[str, pd.DataFrame]] = []
    for ref in refs:
        try:
            if ref.ext == "tsv":
                df = pd.read_csv(ref.path, sep="\t")
            else:
                df, _sheet, _trace = _read_frame(ref, None)
        except Exception:  # noqa: BLE001 — one malformed candidate must not hide other source tables
            continue
        out.append((ref.rel, df))
    return out


# --------------------------------------------------------------------------- source loading
def _read_source(file: str | Path | FileRef | bytes | bytearray) -> tuple[bytes, str, Path | None]:
    """Load EMF bytes from raw bytes / a FileRef / an ``.emf`` path, returning ``(data, label, path)``."""
    if isinstance(file, (bytes, bytearray)):
        return bytes(file), "<bytes>", None
    if isinstance(file, FileRef):
        return file.path.read_bytes(), file.rel or str(file.path), file.path
    p = Path(file)
    if p.exists():
        return p.read_bytes(), str(p), p
    # corpus-relative reference → resolve NFC-aware
    from src.rag.tools.extract_tools import resolve_ref
    ref = resolve_ref(file)
    return ref.path.read_bytes(), ref.rel or str(ref.path), ref.path


def _vision_fallback(data: bytes, path: Path | None, label: str,
                     vision_fallback: Callable[[bytes], str] | None) -> dict[str, Any]:
    """Route a text-less (raster) EMF to a vision caption, or raise if no fallback is available."""
    if vision_fallback is not None:
        caption = vision_fallback(data)
        return contract.make(caption, engine="vision", evidence={"file": label},
                             fallback=True, reason="emf_no_text_records")
    if path is not None:
        sibling = path.with_suffix(".png")
        if sibling.exists():
            from src.rag.tools.extract_tools import caption_figure
            result = caption_figure(sibling)
            result["method"]["fallback"] = True
            result["method"]["reason"] = "emf_no_text_records"
            result["evidence"]["emf_file"] = label
            return result
    raise ContractError(
        f"EMF {label} has no text records and no vision fallback (pass vision_fallback= or "
        "provide a sibling .png)")


# --------------------------------------------------------------------------- public API
def extract_emf_pivot(file: str | Path | FileRef | bytes | bytearray, *,
                      y_gap: float = _DEFAULT_Y_GAP, x_gap: float = _DEFAULT_X_GAP,
                      min_fill_px: int = _DEFAULT_MIN_FILL_PX,
                      ignore_rgb: frozenset[tuple[int, int, int]] = _DEFAULT_IGNORE_RGB,
                      vision_fallback: Callable[[bytes], str] | None = None) -> dict[str, Any]:
    """Reconstruct an embedded EMF PivotTable image into a ``{value, evidence, method}`` contract.

    Parameters
    ----------
    file
        Raw EMF ``bytes``, a :class:`FileRef`, an ``.emf`` path, or an NFC corpus-relative reference.
    y_gap, x_gap
        Row / column clustering gaps in device units (defaults tuned for Office pivot exports).
    min_fill_px
        Minimum width AND height (px) for a solid fill to count as a highlight; thinner blits are
        treated as grid/border strokes and ignored.
    ignore_rgb
        Fill colors never treated as a highlight (default: page-background white).
    vision_fallback
        Optional ``bytes -> caption`` callable used when the EMF holds no text (raster picture).

    ``value`` is ``{table, n_rows, n_cols, captions, highlights}``. Raises :class:`ContractError` on a
    malformed EMF, or on a text-less EMF with no vision fallback available.
    """
    data, label, path = _read_source(file)
    spans, fills, bounds = _parse_emf(data)

    if not spans:
        return _vision_fallback(data, path, label, vision_fallback)

    table, captions, _row_c, col_c = _build_grid(spans, y_gap, x_gap)
    highlights = _map_highlights(fills, spans, table, ignore_rgb, min_fill_px)
    colors = sorted({h["color"] for h in highlights})

    value = {
        "table": table,
        "n_rows": len(table),
        "n_cols": len(col_c),
        "captions": captions,
        "highlights": highlights,
    }
    return contract.make(
        value,
        engine="emf_pivot",
        evidence={
            "file": label,
            "n_text_records": len(spans),
            "n_rows": len(table),
            "n_cols": len(col_c),
            "n_fills": len(fills),
            "bounds": list(bounds),
        },
        y_gap=y_gap,
        x_gap=x_gap,
        min_fill_px=min_fill_px,
        highlight_colors=colors,
        fallback=False,
    )


def highlighted_cells(file: str | Path | FileRef | bytes | bytearray, *,
                      color: str | None = None, **kwargs: Any) -> list[dict[str, Any]]:
    """Convenience: the flat list of highlighted cells (``{value,row_label,col_header,...}``).

    Optionally filter by hex ``color`` (e.g. ``"#FFFF00"`` for the yellow-highlight questions). Thin
    wrapper over :func:`extract_emf_pivot`; ``kwargs`` are forwarded. Returns ``[]`` when the EMF fell
    back to vision (no reconstructed table).
    """
    result = extract_emf_pivot(file, **kwargs)
    if result["method"].get("fallback"):
        return []
    want = color.upper() if color else None
    out: list[dict[str, Any]] = []
    for h in result["value"]["highlights"]:
        if want is None or h["color"].upper() == want:
            out.extend(h["cells"])
    return out


def emf_blobs_from_pptx(pptx: str | Path | FileRef) -> list[tuple[str, bytes]]:
    """Return ``(member_name, bytes)`` for every embedded ``ppt/media/*.emf`` in a .pptx, in order.

    A PivotTable-as-picture lives here; feed each blob to :func:`extract_emf_pivot`. Resolves the pptx
    NFC-aware via the corpus so an NFD-on-disk filename matches an NFC-typed reference.
    """
    if isinstance(pptx, FileRef):
        path = pptx.path
    else:
        p = Path(pptx)
        if p.exists():
            path = p
        else:
            from src.rag.tools.extract_tools import resolve_ref
            path = resolve_ref(pptx).path
    out: list[tuple[str, bytes]] = []
    with zipfile.ZipFile(path) as zf:
        for name in sorted(zf.namelist()):
            low = name.lower()
            if low.startswith("ppt/media/") and low.endswith(".emf"):
                out.append((name, zf.read(name)))
    return out


def extract_pptx_pivots(pptx: str | Path | FileRef, **kwargs: Any) -> list[dict[str, Any]]:
    """Run :func:`extract_emf_pivot` over every embedded EMF in a .pptx.

    Returns one contract per embedded EMF, each with an added ``evidence.pptx`` and ``evidence.member``
    so the caller can tell which picture a table/highlight came from. ``kwargs`` are forwarded.
    """
    label = pptx.rel if isinstance(pptx, FileRef) else str(pptx)
    data_sources: list[tuple[str, pd.DataFrame]] | None = None
    results: list[dict[str, Any]] = []
    for name, blob in emf_blobs_from_pptx(pptx):
        try:
            result = extract_emf_pivot(blob, **kwargs)
        except ContractError:
            continue
        result["evidence"]["pptx"] = label
        result["evidence"]["member"] = name
        # Semantic replay is intentionally limited to highlighted tables.  The investigator exposes
        # this tool for highlighted-pivot questions; running a project-wide DataFrame search for every
        # decorative/non-pivot EMF in an Office deck adds substantial CPU/memory without producing a
        # cell the caller can answer from.
        if not result["method"].get("fallback") and result["value"]["highlights"]:
            if data_sources is None:
                data_sources = _pptx_data_sources(pptx)
            semantics = resolve_pivot_semantics(result["value"]["table"], data_sources)
            result["value"]["semantics"] = semantics
            result["method"]["semantic_resolution"] = bool(semantics)
            if semantics:
                result["evidence"]["semantic_source"] = semantics["source_file"]
                for highlight in result["value"]["highlights"]:
                    for cell in highlight["cells"]:
                        row_value = _json_scalar(_cell_scalar(cell.get("row_label")))
                        column_value = _json_scalar(_cell_scalar(cell.get("col_header")))
                        cell["filters"] = {
                            semantics["row_field"]: row_value,
                            semantics["column_field"]: column_value,
                        }
                        cell["target_column"] = semantics["target_column"]
                        cell["aggregation"] = semantics["aggregation"]
                        cell["aggregation_label"] = semantics["aggregation_label"]
                        cell["semantic_summary"] = (
                            f"{semantics['row_field']}={row_value}、"
                            f"{semantics['column_field']}={column_value}で抽出されたデータに対する"
                            f"{semantics['aggregation_label']} / {semantics['target_column']}")
        results.append(result)
    return results
