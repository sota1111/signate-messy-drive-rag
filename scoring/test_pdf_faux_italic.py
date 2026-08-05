"""SOT-2464: faux-italic (text-matrix shear) emphasis detector (``pdf_faux_italic``).

Covers the two acceptance criteria:
  ① a 強調依存問 returns the correct word set — the emphasised (sheared) words are extracted and
     upright / genuine-italic / rotated glyphs are NOT mistaken for emphasis;
  ② every result is the common contract ``{value, evidence, method}``.

The behaviour is pinned against a tiny, hand-built PDF whose text matrices we control exactly (fast,
no heavy fixtures), plus a guarded smoke test over the real share drive, where 青嶺不動産アセット
マネジメントの報告資料 carries genuine faux italic (shear c/d = 0.3333).
"""
from __future__ import annotations

import io
from pathlib import Path

import pytest

from src.rag.corpus import FileRef, walk
from src.rag.tools import ContractError, detect_faux_italic, emphasized_words, is_contract


# --------------------------------------------------------------------------- minimal PDF builder
def _build_pdf(spans: list[tuple[str, tuple[float, float, float, float], float, float]]) -> bytes:
    """Assemble a 1-page PDF placing each ``(text, (a,b,c,d), x, y)`` span at its own text matrix.

    ``(a,b,c,d)`` are the Tm shear/rotation components (``c`` is the horizontal shear that makes
    faux italic). A single standard Helvetica font backs every span so pdfminer has glyph widths.
    """
    lines: list[str] = []
    for text, (a, b, c, d), x, y in spans:
        esc = text.replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)")
        lines.append(f"BT /F1 24 Tf {a} {b} {c} {d} {x} {y} Tm ({esc}) Tj ET")
    content = ("\n".join(lines) + "\n").encode("latin-1")
    objs = [
        b"<< /Type /Catalog /Pages 2 0 R >>",
        b"<< /Type /Pages /Kids [4 0 R] /Count 1 >>",
        b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>",
        b"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] "
        b"/Resources << /Font << /F1 3 0 R >> >> /Contents 5 0 R >>",
        b"<< /Length %d >>\nstream\n" % len(content) + content + b"endstream",
    ]
    out = io.BytesIO()
    out.write(b"%PDF-1.4\n")
    offsets: list[int] = []
    for i, body in enumerate(objs, 1):
        offsets.append(out.tell())
        out.write(b"%d 0 obj\n" % i + body + b"\nendobj\n")
    xref = out.tell()
    out.write(b"xref\n0 %d\n" % (len(objs) + 1) + b"0000000000 65535 f \n")
    for off in offsets:
        out.write(("%010d 00000 n \n" % off).encode())
    out.write(b"trailer\n<< /Size %d /Root 1 0 R >>\nstartxref\n%d\n%%%%EOF\n"
              % (len(objs) + 1, xref))
    return out.getvalue()


_UPRIGHT = (1, 0, 0, 1)
_SHEAR = (1, 0, 0.3333, 1)          # faux italic — horizontal shear, ~18°
_ROT45 = (0.70710678, 0.70710678, -0.70710678, 0.70710678)  # true rotation (b != 0)


@pytest.fixture()
def faux_pdf(tmp_path: Path) -> FileRef:
    """A PDF with upright, faux-italic (sheared) and rotated spans; only the sheared are emphasis."""
    pdf = _build_pdf([
        ("Plain heading", _UPRIGHT, 72, 720),
        ("Emphasis", _SHEAR, 72, 680),
        ("normal body", _UPRIGHT, 72, 640),
        ("Slanted words", _SHEAR, 72, 600),
        ("Rotated", _ROT45, 72, 480),
    ])
    p = tmp_path / "sample.pdf"
    p.write_bytes(pdf)
    return FileRef(path=p, project="", category="", rel="sample.pdf", name="sample.pdf", ext="pdf")


# --------------------------------------------------------------------------- contract (②)
def test_returns_contract(faux_pdf: FileRef):
    out = detect_faux_italic(faux_pdf)
    assert is_contract(out)
    assert out["method"]["engine"] == "pdf_faux_italic"
    assert out["method"]["shear_threshold"] == pytest.approx(0.1)
    assert out["method"]["shear_values"] == [0.3333]
    assert isinstance(out["value"], list)
    assert out["evidence"]["file"].endswith("sample.pdf")
    assert out["evidence"]["pages"] == 1
    assert out["evidence"]["n_spans"] == len(out["value"])


# --------------------------------------------------------------------------- emphasised words (①)
def test_only_sheared_words_are_emphasis(faux_pdf: FileRef):
    words = emphasized_words(faux_pdf)
    # faux-italic spans (grouped into words) — and nothing upright or rotated.
    assert words == ["Emphasis", "Slanted", "words"]
    assert "Plain" not in words and "normal" not in words  # upright excluded
    assert "Rotated" not in words                          # true rotation is not shear


def test_spans_carry_page_and_bbox(faux_pdf: FileRef):
    spans = detect_faux_italic(faux_pdf)["value"]
    assert {s["text"] for s in spans} == {"Emphasis", "Slanted", "words"}
    for s in spans:
        assert s["page"] == 1
        assert len(s["bbox"]) == 4 and s["bbox"][2] > s["bbox"][0]  # x1 > x0


# --------------------------------------------------------------------------- threshold behaviour
def test_high_threshold_suppresses_all(faux_pdf: FileRef):
    # 0.3333 shear < 0.5 threshold → nothing counts as emphasis, but still a valid empty contract.
    out = detect_faux_italic(faux_pdf, shear_threshold=0.5)
    assert is_contract(out)
    assert out["value"] == []
    assert out["method"]["shear_threshold"] == 0.5


def test_page_filter(faux_pdf: FileRef):
    out = detect_faux_italic(faux_pdf, pages=[2])  # this PDF has only page 1
    assert out["value"] == []
    assert out["evidence"]["pages_scanned"] == 0


# --------------------------------------------------------------------------- input guards
def test_non_pdf_raises(tmp_path: Path):
    p = tmp_path / "note.txt"
    p.write_text("hi", encoding="utf-8")
    ref = FileRef(path=p, project="", category="", rel="note.txt", name="note.txt", ext="txt")
    with pytest.raises(ContractError):
        detect_faux_italic(ref)


def test_negative_threshold_raises(faux_pdf: FileRef):
    with pytest.raises(ContractError):
        detect_faux_italic(faux_pdf, shear_threshold=-0.1)


# --------------------------------------------------------------------------- real corpus smoke
def test_real_corpus_faux_italic_smoke():
    ref = next((r for r in walk()
                if r.ext == "pdf" and r.name == "報告資料_2025-08-06.pdf"), None)
    if ref is None:
        pytest.skip("no real corpus present")
    words = emphasized_words(ref)
    # 青嶺不動産アセットマネジメントの報告資料 page 1 emphasises the (faux-italic) 金額.
    assert "4,675,000円" in words
    out = detect_faux_italic(ref)
    assert is_contract(out)
    assert out["method"]["shear_values"] == [0.3333]
