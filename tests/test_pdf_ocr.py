"""Image-only (scanned) PDF → Gemini-vision OCR fallback (SOT-2526, 棄権E / RETRIEVED_NOT_PARSED).

A scanned 会議録 / 報告書 PDF has no searchable text layer, so pdfplumber/pypdf return only the page
markers and the needle (アクションアイテム表 A08…/担当者/マイルストーン M01…/ステータス) is unreadable.
These tests pin the OCR fallback: default OFF ⇒ byte-identical marker-only text; ON ⇒ the page rasters
are transcribed. All hermetic — the vision model and the page-raster reader are stubbed, no network.
"""
from __future__ import annotations

import sys
import types

import pytest

from src.rag.corpus import FileRef
from src.rag.extract import plain, vision


def _ref(tmp_path, name="会議録_2025-04-24.pdf"):
    p = tmp_path / name
    p.write_bytes(b"%PDF-1.4 stub")  # real file so stat()/cache signature works; content unused (stubbed)
    return FileRef(path=p, project="案件A", category="meeting", rel=f"05.会議/{name}", name=name, ext="pdf")


class _FakePage:
    def __init__(self, text="", tables=None):
        self._text = text
        self._tables = tables or []

    def extract_text(self):
        return self._text

    def extract_tables(self):
        return self._tables


class _FakePdf:
    def __init__(self, pages):
        self.pages = pages

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


def _install_fake_pdfplumber(monkeypatch, pages):
    mod = types.ModuleType("pdfplumber")
    mod.open = lambda _path: _FakePdf(pages)
    monkeypatch.setitem(sys.modules, "pdfplumber", mod)


@pytest.fixture(autouse=True)
def _reset(monkeypatch):
    vision._ocr_pdf_cached.cache_clear()
    monkeypatch.delenv("RAG_PDF_OCR", raising=False)
    yield
    vision._ocr_pdf_cached.cache_clear()


# --------------------------------------------------------------------------- flag + helper
@pytest.mark.parametrize("val,expected", [("1", True), ("true", True), ("on", True),
                                          ("0", False), ("", False), ("no", False)])
def test_pdf_ocr_flag(monkeypatch, val, expected):
    monkeypatch.setenv("RAG_PDF_OCR", val)
    assert vision.pdf_ocr_enabled() is expected


def test_pdf_ocr_flag_default_off():
    assert vision.pdf_ocr_enabled() is False


def test_text_layer_body_ignores_markers():
    assert plain._text_layer_body("[ページ1]\n[ページ2]\n[表]") == ""
    assert "A08" in plain._text_layer_body("[ページ1]\nA08 伊藤 完了")


# --------------------------------------------------------------------------- ocr_image_pdf
def test_ocr_image_pdf_joins_pages(monkeypatch, tmp_path):
    monkeypatch.setattr(vision, "pdf_page_images",
                        lambda _p: [(b"img1", "image/jpeg"), (b"img2", "image/jpeg")])
    calls = []
    monkeypatch.setattr(vision.llm, "generate",
                        lambda *a, **k: (calls.append(1), f"転記{len(calls)} A08 伊藤")[1])
    ref = _ref(tmp_path)
    out = vision.ocr_image_pdf(ref.path)
    assert out is not None
    assert "[ページ1]" in out and "[ページ2]" in out
    assert "A08" in out and out.count("転記") == 2


def test_ocr_image_pdf_none_when_not_image_only(monkeypatch, tmp_path):
    monkeypatch.setattr(vision, "pdf_page_images", lambda _p: [])
    monkeypatch.setattr(vision.llm, "generate", lambda *a, **k: "should not be called")
    assert vision.ocr_image_pdf(_ref(tmp_path).path) is None


def test_ocr_image_pdf_none_when_pages_blank(monkeypatch, tmp_path):
    monkeypatch.setattr(vision, "pdf_page_images", lambda _p: [(b"img", "image/jpeg")])
    monkeypatch.setattr(vision.llm, "generate", lambda *a, **k: "   ")  # nothing readable
    assert vision.ocr_image_pdf(_ref(tmp_path).path) is None


def test_ocr_image_pdf_is_cached(monkeypatch, tmp_path):
    calls = []
    monkeypatch.setattr(vision, "pdf_page_images",
                        lambda _p: (calls.append(1), [(b"i", "image/jpeg")])[1])
    monkeypatch.setattr(vision.llm, "generate", lambda *a, **k: "A08")
    ref = _ref(tmp_path)
    assert vision.ocr_image_pdf(ref.path) == vision.ocr_image_pdf(ref.path)
    assert len(calls) == 1  # second read served from the (path,size,mtime) cache


# --------------------------------------------------------------------------- extract_pdf gating
def test_extract_pdf_off_is_marker_only_no_ocr(monkeypatch, tmp_path):
    _install_fake_pdfplumber(monkeypatch, [_FakePage(""), _FakePage("")])

    def _boom(_p):
        raise AssertionError("OCR must not run when RAG_PDF_OCR is off")

    monkeypatch.setattr(vision, "ocr_image_pdf", _boom)
    assert plain.extract_pdf(_ref(tmp_path)) == "[ページ1]\n[ページ2]"


def test_extract_pdf_on_empty_layer_uses_ocr(monkeypatch, tmp_path):
    _install_fake_pdfplumber(monkeypatch, [_FakePage(""), _FakePage("")])
    monkeypatch.setenv("RAG_PDF_OCR", "1")
    monkeypatch.setattr(vision, "ocr_image_pdf", lambda _p: "[ページ1]\nA08 A09 伊藤 M01 M02 完了")
    out = plain.extract_pdf(_ref(tmp_path))
    assert "A08" in out and "伊藤" in out and "M02" in out


def test_extract_pdf_on_textlayer_skips_ocr(monkeypatch, tmp_path):
    # A PDF whose text layer is non-empty must NOT be OCR'd even with the flag on.
    _install_fake_pdfplumber(monkeypatch, [_FakePage("本文テキストあり 契約金額 500万円")])
    monkeypatch.setenv("RAG_PDF_OCR", "1")

    def _boom(_p):
        raise AssertionError("OCR must not run when the text layer is non-empty")

    monkeypatch.setattr(vision, "ocr_image_pdf", _boom)
    out = plain.extract_pdf(_ref(tmp_path))
    assert "契約金額" in out


def test_extract_pdf_on_empty_ocr_returns_falls_back_to_text(monkeypatch, tmp_path):
    # OCR enabled but the PDF is not actually image-only (ocr returns None) → keep the marker text.
    _install_fake_pdfplumber(monkeypatch, [_FakePage("")])
    monkeypatch.setenv("RAG_PDF_OCR", "1")
    monkeypatch.setattr(vision, "ocr_image_pdf", lambda _p: None)
    assert plain.extract_pdf(_ref(tmp_path)) == "[ページ1]"
