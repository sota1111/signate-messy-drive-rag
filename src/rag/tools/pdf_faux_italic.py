"""pdf_faux_italic — detect *synthetic* (faux) italic emphasis in a PDF via the text-matrix shear.

Some report authors emphasise a word not with a real italic font but by **shearing** upright glyphs
— the PDF content stream sets a text/render matrix ``[a b c d e f]`` with a non-zero horizontal shear
component ``c`` (``c/d ≈ 0.3333`` in this corpus, i.e. an ~18° slant). A plain text-layer extraction
throws that geometry away, so "which words are emphasised?" questions become unanswerable from the
extracted string alone. This tool walks every glyph's matrix and returns the spans whose shear exceeds
a threshold as the emphasised (回答対象) word set.

It returns the unified :mod:`src.rag.tools.contract` shape ``{value, evidence, method}`` so the
generation layer can call it exactly like the other tools:

* ``value``    — a list of emphasised span records ``{text, page, bbox}`` in reading order.
* ``evidence`` — ``file`` + page/count provenance (which file, how many pages, how many spans).
* ``method``   — ``engine="pdf_faux_italic"``, the ``shear_threshold`` used, and the distinct shear
  ratios actually observed (never any secret — this is pure geometry).

Detection
---------
For each glyph pdfplumber exposes ``matrix = (a, b, c, d, e, f)``. A *faux italic* glyph is an
**upright** glyph (no rotation, ``b ≈ 0``) with a horizontal shear ``|c/d| ≥ shear_threshold``.
Genuine italic *fonts* leave ``c = 0`` (the slant lives in the glyph outlines), so they are correctly
ignored — this tool detects the matrix-shear technique specifically. Rotated text (``b ≠ 0``, e.g.
vertical axis labels) is excluded so a rotation is never mistaken for emphasis.

NFC / NFD
---------
File resolution goes through :func:`src.rag.tools.extract_tools.resolve_ref` (NFC-aware) and opens the
byte-exact :attr:`FileRef.path` from the walked corpus, so an NFD-on-disk filename is matched from an
NFC-typed reference without a "file not found" from re-encoding.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable

from src.rag.corpus import FileRef, nfc
from src.rag.tools import contract
from src.rag.tools.contract import ContractError
from src.rag.tools.extract_tools import resolve_ref

# Faux italic in this corpus shears by c/d ≈ 0.3333 (~18°). 0.1 sits well below that while staying
# clear of the 0.0 of upright/genuine-italic glyphs, so it catches the synthetic slant without noise.
DEFAULT_SHEAR_THRESHOLD = 0.1

# A glyph is treated as unrotated when its off-diagonal ``b`` is negligible relative to the diagonal.
# Faux italic is a pure horizontal shear (b == 0, c != 0); rotated text has b != 0 and is excluded.
_ROTATION_EPS = 1e-3


def _shear_ratio(matrix: Any) -> float | None:
    """Horizontal shear ``c/d`` of a glyph matrix, or ``None`` when it isn't a usable upright shear.

    Returns ``None`` for a missing/short matrix, a zero vertical scale (``d == 0``), or a rotated
    glyph (``|b|`` not negligible vs. the diagonal) — those are never faux-italic emphasis.
    """
    if not matrix or len(matrix) < 4:
        return None
    a, b, c, d = matrix[0], matrix[1], matrix[2], matrix[3]
    if not d:
        return None
    scale = max(abs(a), abs(d), 1.0)
    if abs(b) > _ROTATION_EPS * scale:  # rotated, not a pure shear
        return None
    return c / d


def _sheared_chars(chars: Iterable[dict[str, Any]], threshold: float) -> list[dict[str, Any]]:
    """Chars whose glyph matrix is an upright shear at/above ``threshold`` (each tagged ``_shear``)."""
    out: list[dict[str, Any]] = []
    for ch in chars:
        ratio = _shear_ratio(ch.get("matrix"))
        if ratio is not None and abs(ratio) >= threshold:
            tagged = dict(ch)
            tagged["_shear"] = ratio
            out.append(tagged)
    return out


def detect_faux_italic(file: str | Path | FileRef, *,
                       shear_threshold: float = DEFAULT_SHEAR_THRESHOLD,
                       pages: Iterable[int] | None = None,
                       x_tolerance: float = 1.5,
                       y_tolerance: float = 3.0) -> dict[str, Any]:
    """Detect faux-italic (matrix-shear) emphasised spans in a PDF as a contract result.

    Parameters
    ----------
    file
        A :class:`FileRef`, absolute path, or NFC corpus-relative name/rel (resolved NFC-aware).
    shear_threshold
        Minimum ``|c/d|`` for a glyph to count as sheared (default ``0.1``; corpus faux italic is
        ``0.3333``). Genuine italic fonts and upright text have shear ``0`` and are ignored.
    pages
        Optional 1-based page numbers to restrict the scan; ``None`` scans every page.
    x_tolerance, y_tolerance
        Word-grouping tolerances passed to pdfplumber's word extractor.

    ``value`` is a list of ``{text, page, bbox}`` span records (``bbox = [x0, top, x1, bottom]``) in
    reading order. Raises :class:`ContractError` if ``file`` is not a ``.pdf`` or cannot be opened.
    """
    ref = resolve_ref(file)
    if ref.ext != "pdf":
        raise ContractError(f"not a PDF: .{ref.ext} (expected pdf)")
    if shear_threshold < 0:
        raise ContractError(f"shear_threshold must be non-negative, got {shear_threshold}")

    try:
        import pdfplumber
        from pdfplumber.utils import extract_words
    except Exception as e:  # pragma: no cover - pdfplumber is a hard dependency
        raise ContractError(f"pdfplumber unavailable: {e}") from e

    want_pages = {int(p) for p in pages} if pages is not None else None
    spans: list[dict[str, Any]] = []
    observed_shears: set[float] = set()
    n_pages = 0

    try:
        with pdfplumber.open(str(ref.path)) as pdf:
            n_pages = len(pdf.pages)
            for pi, page in enumerate(pdf.pages, 1):
                if want_pages is not None and pi not in want_pages:
                    continue
                sheared = _sheared_chars(page.chars, shear_threshold)
                if not sheared:
                    continue
                observed_shears.update(round(ch["_shear"], 4) for ch in sheared)
                for w in extract_words(sheared, x_tolerance=x_tolerance,
                                       y_tolerance=y_tolerance):
                    text = nfc(w.get("text", ""))
                    if not text:
                        continue
                    spans.append({
                        "text": text,
                        "page": pi,
                        "bbox": [round(w["x0"], 2), round(w["top"], 2),
                                 round(w["x1"], 2), round(w["bottom"], 2)],
                    })
    except ContractError:
        raise
    except Exception as e:
        raise ContractError(f"could not read PDF {ref.rel or ref.path}: {e}") from e

    scanned_pages = (len(want_pages & set(range(1, n_pages + 1)))
                     if want_pages is not None else n_pages)
    return contract.make(
        spans,
        engine="pdf_faux_italic",
        evidence={
            "file": ref.rel or str(ref.path),
            "pages": n_pages,
            "pages_scanned": scanned_pages,
            "n_spans": len(spans),
        },
        shear_threshold=shear_threshold,
        shear_values=sorted(observed_shears),
    )


def emphasized_words(file: str | Path | FileRef, **kwargs: Any) -> list[str]:
    """Convenience: the ordered, de-duplicated faux-italic span texts (bare list, not a contract).

    Thin wrapper over :func:`detect_faux_italic` for the common "give me the emphasised 語 set"
    call; ``kwargs`` are forwarded (``shear_threshold`` / ``pages`` / tolerances).
    """
    result = detect_faux_italic(file, **kwargs)
    seen: set[str] = set()
    words: list[str] = []
    for span in result["value"]:
        text = span["text"]
        if text not in seen:
            seen.add(text)
            words.append(text)
    return words
