"""OCR persistence store (SOT-2650): build-time transcription served without Gemini.

Pins the store contract: default OFF ⇒ byte-identical serve path; ON ⇒ an effectively-image-only
PDF resolves to the persisted build-time transcription with no genai call; size drift / unknown
rel / near-empty threshold all fail open to the live behaviour. All hermetic — vision + pdfplumber
are stubbed, no network.
"""
from __future__ import annotations

import json
import sys
import types

import pytest

from src.rag.corpus import FileRef, nfc
from src.rag.extract import plain
from src.rag.index import ocr_store


def _ref(tmp_path, name="最終報告.pdf", content=b"%PDF-1.4 stub"):
    p = tmp_path / name
    p.write_bytes(content)
    return FileRef(path=p, project="案件A", category="report", rel=f"06.報告書/{name}", name=name, ext="pdf")


def _write_store(tmp_path, ref, text="[ページ1]\n転記 A08 伊藤", size=None):
    out = tmp_path / "ocr_store.jsonl"
    size = size if size is not None else ref.path.stat().st_size
    rows = [{"schema": ocr_store.SCHEMA, "version": ocr_store.SCHEMA_VERSION},
            {"rel": nfc(ref.rel), "size": size, "mtime": 0, "pages": 1,
             "model": "stub-vision", "text": text}]
    out.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in rows), encoding="utf-8")
    return out


class _FakePage:
    def __init__(self, text=""):
        self._text = text

    def extract_text(self):
        return self._text

    def extract_tables(self):
        return []


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
    ocr_store.load.cache_clear()
    monkeypatch.delenv("RAG_OCR_STORE", raising=False)
    monkeypatch.delenv("RAG_OCR_STORE_BUILD", raising=False)
    monkeypatch.delenv("RAG_PDF_OCR", raising=False)
    yield
    ocr_store.load.cache_clear()


# --------------------------------------------------------------------------- flags + eligibility
@pytest.mark.parametrize("val,expected", [("1", True), ("true", True), ("on", True),
                                          ("0", False), ("", False), ("no", False)])
def test_serve_flag(monkeypatch, val, expected):
    monkeypatch.setenv("RAG_OCR_STORE", val)
    assert ocr_store.enabled() is expected


def test_flags_default_off():
    assert ocr_store.enabled() is False
    assert ocr_store.build_enabled() is False


def test_eligibility_threshold():
    assert ocr_store.is_effectively_image_only("")
    assert ocr_store.is_effectively_image_only("[ページ1]\n[ページ2]")
    assert ocr_store.is_effectively_image_only("[ページ1]\n株式会社 データアステル")  # 近空 (最終報告 letterhead)
    assert not ocr_store.is_effectively_image_only("[ページ1]\n" + "本文" * ocr_store.BODY_MAX_CHARS)


# --------------------------------------------------------------------------- lookup fail-open
def test_lookup_hit(tmp_path):
    ref = _ref(tmp_path)
    out = _write_store(tmp_path, ref)
    assert "転記 A08 伊藤" in ocr_store.lookup(ref, out)


def test_lookup_unknown_rel_none(tmp_path):
    ref = _ref(tmp_path)
    out = _write_store(tmp_path, ref)
    other = _ref(tmp_path, name="別文書.pdf")
    assert ocr_store.lookup(other, out) is None


def test_lookup_size_drift_fails_open(tmp_path):
    ref = _ref(tmp_path)
    out = _write_store(tmp_path, ref, size=ref.path.stat().st_size + 1)
    assert ocr_store.lookup(ref, out) is None


def test_lookup_schema_version_mismatch_fails_open(tmp_path):
    ref = _ref(tmp_path)
    out = tmp_path / "ocr_store.jsonl"
    rows = [{"schema": ocr_store.SCHEMA, "version": ocr_store.SCHEMA_VERSION + 1},
            {"rel": nfc(ref.rel), "size": ref.path.stat().st_size, "text": "x"}]
    out.write_text("\n".join(json.dumps(r) for r in rows), encoding="utf-8")
    assert ocr_store.lookup(ref, out) is None


def test_lookup_missing_store_fails_open(tmp_path):
    assert ocr_store.lookup(_ref(tmp_path), tmp_path / "absent.jsonl") is None


# --------------------------------------------------------------------------- extract_pdf wiring
def test_extract_pdf_off_ignores_store(monkeypatch, tmp_path):
    ref = _ref(tmp_path)
    out = _write_store(tmp_path, ref)
    monkeypatch.setattr(ocr_store, "default_out_path", lambda: out)
    _install_fake_pdfplumber(monkeypatch, [_FakePage("")])
    assert plain.extract_pdf(ref) == "[ページ1]"  # byte-identical marker-only text


def test_extract_pdf_on_serves_store_without_genai(monkeypatch, tmp_path):
    ref = _ref(tmp_path)
    out = _write_store(tmp_path, ref)
    monkeypatch.setattr(ocr_store, "default_out_path", lambda: out)
    monkeypatch.setenv("RAG_OCR_STORE", "1")
    _install_fake_pdfplumber(monkeypatch, [_FakePage("")])
    from src.rag.extract import vision

    monkeypatch.setattr(vision, "ocr_image_pdf",
                        lambda _p: (_ for _ in ()).throw(AssertionError("no live OCR with a store hit")))
    assert "転記 A08 伊藤" in plain.extract_pdf(ref)


def test_extract_pdf_on_near_empty_layer_serves_store(monkeypatch, tmp_path):
    ref = _ref(tmp_path)
    out = _write_store(tmp_path, ref)
    monkeypatch.setattr(ocr_store, "default_out_path", lambda: out)
    monkeypatch.setenv("RAG_OCR_STORE", "1")
    # a stray letterhead line: strictly non-empty, so SOT-2526 live OCR never fires — the store must
    _install_fake_pdfplumber(monkeypatch, [_FakePage("株式会社 データアステル")])
    assert "転記 A08 伊藤" in plain.extract_pdf(ref)


def test_extract_pdf_on_text_pdf_keeps_text_layer(monkeypatch, tmp_path):
    ref = _ref(tmp_path)
    out = _write_store(tmp_path, ref)
    monkeypatch.setattr(ocr_store, "default_out_path", lambda: out)
    monkeypatch.setenv("RAG_OCR_STORE", "1")
    body = "本文" * ocr_store.BODY_MAX_CHARS
    _install_fake_pdfplumber(monkeypatch, [_FakePage(body)])
    assert body in plain.extract_pdf(ref)  # real text layer wins; store never overrides it


def test_extract_pdf_store_miss_falls_back_to_live_ocr(monkeypatch, tmp_path):
    ref = _ref(tmp_path, name="未収載.pdf")
    out = _write_store(tmp_path, _ref(tmp_path))
    monkeypatch.setattr(ocr_store, "default_out_path", lambda: out)
    monkeypatch.setenv("RAG_OCR_STORE", "1")
    monkeypatch.setenv("RAG_PDF_OCR", "1")
    _install_fake_pdfplumber(monkeypatch, [_FakePage("")])
    from src.rag.extract import vision

    monkeypatch.setattr(vision, "ocr_image_pdf", lambda _p: "生転記 M01")
    assert plain.extract_pdf(ref) == "生転記 M01"


# --------------------------------------------------------------------------- build
def test_build_transcribes_eligible_and_reuses_prior(monkeypatch, tmp_path):
    ref = _ref(tmp_path)
    out = tmp_path / "ocr_store.jsonl"
    monkeypatch.setattr(ocr_store, "default_report_path", lambda: tmp_path / "report.json")
    monkeypatch.setattr(ocr_store, "_text_layer", lambda _r: "[ページ1]")  # image-only
    calls = []
    monkeypatch.setattr(ocr_store, "_ocr_pages",
                        lambda _p: (calls.append(1), ("[ページ1]\n転記済", 1))[1])
    r1 = ocr_store.build([ref], out)
    assert r1["records"] == 1 and r1["report"]["transcribed"] == 1 and len(calls) == 1
    r2 = ocr_store.build([ref], out)  # unchanged file ⇒ reuse, no second vision spend
    assert r2["report"]["reused"] == 1 and r2["report"]["transcribed"] == 0 and len(calls) == 1
    assert "転記済" in ocr_store.lookup(ref, out)


def test_build_skips_text_pdfs_and_failures(monkeypatch, tmp_path):
    text_ref = _ref(tmp_path, name="text.pdf")
    bad_ref = _ref(tmp_path, name="bad.pdf")
    out = tmp_path / "ocr_store.jsonl"
    monkeypatch.setattr(ocr_store, "default_report_path", lambda: tmp_path / "report.json")
    monkeypatch.setattr(ocr_store, "_text_layer",
                        lambda r: "本文" * 200 if r.name == "text.pdf" else "")
    monkeypatch.setattr(ocr_store, "_ocr_pages", lambda _p: (None, 0))
    r = ocr_store.build([text_ref, bad_ref], out)
    assert r["records"] == 0
    assert [s["rel"] for s in r["report"]["skipped"]] == [nfc(bad_ref.rel)]
