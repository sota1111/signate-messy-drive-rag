"""SOT-2463: embedded-EMF PivotTable reconstruction + highlight recovery (``emf_pivot``).

Covers the two acceptance criteria:
  ① an EMF ピボット問 returns the correct value — the表 grid is reconstructed from the
     ``EMR_EXTTEXTOUTW`` text positions and the 黄色ハイライト cell maps back to the right value /
     row label / column header (thin border strokes and white background are NOT mistaken for shading);
  ② every result is the common contract ``{value, evidence, method}``.

Behaviour is pinned against a tiny, hand-built EMF whose records we control exactly (fast, no heavy
fixtures and no committing non-redistributable SIGNATE data), plus a guarded smoke test over the real
share drive, where 青潮モビリティサービスの基礎分析.pptx embeds the pivot picture whose yellow cell is
hr=8 / weekday=2 → 0.78.
"""
from __future__ import annotations

import struct

import pytest

from src.rag.tools import (
    ContractError,
    extract_emf_pivot,
    extract_pptx_pivots,
    highlighted_cells,
    is_contract,
)


# --------------------------------------------------------------------------- minimal EMF builder
def _rec_exttextoutw(text: str, x: int, y: int) -> bytes:
    """One EMR_EXTTEXTOUTW placing ``text`` (UTF-16LE) at reference point ``(x, y)`` (no Dx array)."""
    raw = text.encode("utf-16-le")
    pad = (-len(raw)) % 4
    body = raw + b"\x00" * pad
    off_string = 76
    n_size = off_string + len(body)
    rec = struct.pack("<II", 84, n_size)              # iType, nSize
    rec += struct.pack("<4i", x, y, x + 50, y + 20)   # rclBounds
    rec += struct.pack("<I", 1)                        # iGraphicsMode
    rec += struct.pack("<ff", 1.0, 1.0)               # ex/eyScale
    rec += struct.pack("<2i", x, y)                    # ptlReference
    rec += struct.pack("<I", len(text))               # nChars
    rec += struct.pack("<I", off_string)              # offString
    rec += struct.pack("<I", 0)                        # fOptions
    rec += struct.pack("<4i", 0, 0, 0, 0)             # rcl (clip)
    rec += struct.pack("<I", 0)                        # offDx
    rec += body
    return rec


def _rec_createbrush(ih: int, rgb: tuple[int, int, int]) -> bytes:
    color = rgb[0] | (rgb[1] << 8) | (rgb[2] << 16)   # 0x00BBGGRR
    return struct.pack("<IIIIII", 39, 24, ih, 0, color, 0)  # BS_SOLID, hatch=0


def _rec_selectobject(ih: int) -> bytes:
    return struct.pack("<III", 37, 12, ih)


def _rec_bitblt(x: int, y: int, cx: int, cy: int, rop: int = 0x00F00021) -> bytes:
    rec = struct.pack("<II", 76, 100)                 # iType, nSize
    rec += struct.pack("<4i", x, y, x + cx, y + cy)   # rclBounds
    rec += struct.pack("<4i", x, y, cx, cy)           # xDest,yDest,cxDest,cyDest
    rec += struct.pack("<I", rop)                     # dwRop (PATCOPY)
    rec += struct.pack("<2i", 0, 0)                   # xSrc,ySrc
    rec += struct.pack("<6f", 1, 0, 0, 1, 0, 0)      # xformSrc
    rec += struct.pack("<I", 0)                       # crBkColorSrc
    rec += struct.pack("<I", 0)                       # iUsageSrc
    rec += struct.pack("<4I", 0, 0, 0, 0)            # off/cb Bmi/Bits (no source bitmap)
    return rec


def _rec_eof() -> bytes:
    return struct.pack("<IIIII", 14, 20, 0, 16, 20)


def _build_emf(records: list[bytes], bounds=(0, 0, 300, 200), *, n_handles: int = 8) -> bytes:
    """Wrap ``records`` in a minimal valid ENHMETAHEADER + EOF, patching sizes/counts."""
    records = list(records) + [_rec_eof()]
    body = b"".join(records)
    header = struct.pack("<II", 1, 88)                # iType, nSize (88-byte header)
    header += struct.pack("<4i", *bounds)             # rclBounds
    header += struct.pack("<4i", 0, 0, bounds[2] * 15, bounds[3] * 15)  # rclFrame (.01mm)
    header += struct.pack("<I", 0x464D4520)          # dSignature ' EMF'
    header += struct.pack("<I", 0x00010000)          # nVersion
    total = 88 + len(body)
    header += struct.pack("<I", total)               # nBytes
    header += struct.pack("<I", len(records) + 1)    # nRecords (+ header)
    header += struct.pack("<H", n_handles)           # nHandles (WORD)
    header += struct.pack("<H", 0)                    # sReserved
    header += struct.pack("<I", 0)                    # nDescription
    header += struct.pack("<I", 0)                    # offDescription
    header += struct.pack("<I", 0)                    # nPalEntries
    header += struct.pack("<2i", 300, 200)           # szlDevice
    header += struct.pack("<2i", 100, 67)            # szlMillimeters
    assert len(header) == 88
    return header + body


# A 3×3 pivot: header (行ラベル / A / B), two data rows keyed 0 and 1, plus a single-cell title above.
_TABLE_RECORDS = [
    _rec_exttextoutw("タイトル", 100, 4),     # caption (single-cell row)
    _rec_exttextoutw("行ラベル", 10, 40), _rec_exttextoutw("A", 110, 40), _rec_exttextoutw("B", 210, 40),
    _rec_exttextoutw("0", 10, 80), _rec_exttextoutw("1.1", 110, 80), _rec_exttextoutw("2.2", 210, 80),
    _rec_exttextoutw("1", 10, 120), _rec_exttextoutw("3.3", 110, 120), _rec_exttextoutw("9.9", 210, 120),
]


@pytest.fixture()
def pivot_emf() -> bytes:
    """A pivot EMF with a light-blue header band and a yellow highlight over the '9.9' (row 1 / col B).

    Also paints a 1px-tall gray line and a full white background — both must be ignored as non-shading.
    """
    fills = [
        _rec_createbrush(1, (255, 255, 255)), _rec_selectobject(1), _rec_bitblt(0, 0, 300, 200),  # bg white
        _rec_createbrush(2, (192, 230, 245)), _rec_selectobject(2), _rec_bitblt(0, 25, 300, 30),  # header band
        _rec_createbrush(3, (128, 128, 128)), _rec_selectobject(3), _rec_bitblt(0, 100, 300, 1),  # 1px line
        _rec_createbrush(4, (255, 255, 0)), _rec_selectobject(4), _rec_bitblt(195, 105, 60, 30),  # yellow cell
    ]
    return _build_emf(fills + _TABLE_RECORDS)


# --------------------------------------------------------------------------- contract (②)
def test_returns_contract(pivot_emf: bytes):
    out = extract_emf_pivot(pivot_emf)
    assert is_contract(out)
    assert out["method"]["engine"] == "emf_pivot"
    assert out["method"]["fallback"] is False
    assert out["evidence"]["n_text_records"] == 10
    v = out["value"]
    assert v["n_rows"] == 3 and v["n_cols"] == 3
    assert set(v) == {"table", "n_rows", "n_cols", "captions", "highlights"}


# --------------------------------------------------------------------------- grid reconstruction (①)
def test_table_reconstructed(pivot_emf: bytes):
    v = extract_emf_pivot(pivot_emf)["value"]
    assert v["table"] == [
        ["行ラベル", "A", "B"],
        ["0", "1.1", "2.2"],
        ["1", "3.3", "9.9"],
    ]
    assert v["captions"] == ["タイトル"]  # single-cell title is not folded into the grid


# --------------------------------------------------------------------------- highlight → cell (①)
def test_yellow_highlight_maps_to_cell(pivot_emf: bytes):
    cells = highlighted_cells(pivot_emf, color="#FFFF00")
    assert cells == [{"row": 2, "col": 2, "value": "9.9", "row_label": "1", "col_header": "B"}]


def test_thin_and_white_fills_are_not_highlights(pivot_emf: bytes):
    colors = extract_emf_pivot(pivot_emf)["method"]["highlight_colors"]
    # white background + 1px gray line dropped; only the header band + yellow cell remain.
    assert colors == ["#C0E6F5", "#FFFF00"]


def test_header_band_covers_whole_header_row(pivot_emf: bytes):
    band = highlighted_cells(pivot_emf, color="#C0E6F5")
    assert [c["value"] for c in band] == ["行ラベル", "A", "B"]


# --------------------------------------------------------------------------- vision fallback
def test_raster_emf_uses_vision_fallback():
    raster = _build_emf([])  # header + EOF only → no text records
    out = extract_emf_pivot(raster, vision_fallback=lambda data: "a pivot chart")
    assert is_contract(out)
    assert out["value"] == "a pivot chart"
    assert out["method"]["engine"] == "vision"
    assert out["method"]["fallback"] is True
    assert highlighted_cells(raster, vision_fallback=lambda data: "x") == []


def test_raster_emf_without_fallback_raises():
    raster = _build_emf([])
    with pytest.raises(ContractError):
        extract_emf_pivot(raster)


# --------------------------------------------------------------------------- input guards
def test_non_emf_raises():
    with pytest.raises(ContractError):
        extract_emf_pivot(b"not an emf at all, definitely not a metafile header padding........")


def test_corrupt_record_raises():
    good = _build_emf(_TABLE_RECORDS)
    # blow up the size field of the first record after the header (offset 88 → nSize at 92)
    bad = bytearray(good)
    struct.pack_into("<I", bad, 92, 0xFFFFFF)
    with pytest.raises(ContractError):
        extract_emf_pivot(bytes(bad))


# --------------------------------------------------------------------------- real corpus smoke
def test_real_corpus_emf_pivot_smoke():
    from src.rag.corpus import walk
    ref = next((r for r in walk()
                if r.ext == "pptx" and "基礎分析" in r.name
                and "青潮モビリティ" in (r.project or r.rel)), None)
    if ref is None:
        pytest.skip("no real corpus present")
    results = extract_pptx_pivots(ref)
    assert results, "expected at least one embedded EMF pivot"
    yellow = [c for res in results for h in res["value"]["highlights"]
              if h["color"] == "#FFFF00" for c in h["cells"]]
    # 基礎分析.pptx page: 黄色ハイライト = hr=8 (row label) × weekday=2 (col header) → 0.78.
    assert any(c["value"] == "0.78" and c["row_label"] == "8" and c["col_header"] == "2"
               for c in yellow)
    semantics = results[0]["value"]["semantics"]
    assert semantics is not None
    assert semantics["row_field"] == "hr" and semantics["column_field"] == "weekday"
    assert semantics["target_column"] == "temp"
    assert semantics["aggregation"] == "max" and semantics["match_rate"] == 1.0
    assert any(c["filters"] == {"hr": 8, "weekday": 2}
               and c["semantic_summary"] == "hr=8、weekday=2で抽出されたデータに対する最大 / temp"
               for c in yellow)
