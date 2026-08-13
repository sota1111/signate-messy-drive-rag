"""SOT-2684 — 画像OCR事実ストア + extract 合流の offline テスト（ネット/LLM 不要）。

固定する不変量:
* EMF メタファイルのテキストを **vision を使わず決定論** で復元し、font/positioning ノイズを除去する。
* build の metafile-text 経路は vision を消費しない（vision_model=None）。record_key で increに再利用する。
* ON（``RAG_IMAGE_OCR_STORE``）⇒ image-OCR 由来テキストが :func:`extract` 出力へ locator 付きで合流し、
  FTS/vector/file_grep/read_office が到達できる。OFF（既定）⇒ appendix は "" ⇒ extract 出力 byte-identical。
* load() のスキーマ不一致 / 欠損 ⇒ []（回帰ゼロ）。ツール集合は不変（新規ツールを足さない）。
"""
from __future__ import annotations

import json

import pytest

from src.rag.index import image_ocr_store as S


# --------------------------------------------------------------------------- real corpus EMF (deterministic)
def _emf_docx_ref():
    from src.rag.corpus import walk, nfc
    for r in walk():
        if r.ext == "docx" and "データサイエンティスト調査" in nfc(r.name):
            return r
    return None


def test_emf_text_recovers_table_values_without_vision():
    ref = _emf_docx_ref()
    if ref is None:
        pytest.skip("corpus EMF docx not present")
    import zipfile
    blob = zipfile.ZipFile(str(ref.path)).read("word/media/image1.emf")
    text = S.emf_text(blob)
    # The ML/DE salary table values idx8 needs (140,000 − 125,256) are present…
    assert "140,000" in text and "125,256" in text
    assert "職務タイトル" in text and "データエンジニア" in text
    # …and the metafile font/positioning noise is stripped.
    assert "Arial" not in text and "耀" not in text
    assert not any(0x3400 <= ord(ch) <= 0x4DBF or 0xE000 <= ord(ch) <= 0xF8FF for ch in text)


def test_emf_text_empty_on_garbage():
    assert S.emf_text(b"") == ""
    assert S.emf_text(b"\x00\x01\x02\x03") == ""


# --------------------------------------------------------------------------- build (metafile path, no vision)
def test_build_metafile_path_is_deterministic_and_visionless(tmp_path):
    ref = _emf_docx_ref()
    if ref is None:
        pytest.skip("corpus EMF docx not present")
    out = tmp_path / "image_ocr_store.jsonl"
    res = S.build(refs=[ref], out=out, write_report=False)
    recs = S.load(out)
    assert res["records"] == len(recs) >= 1
    meta = [r for r in recs if r.get("source") == "metafile-text"]
    assert meta, "EMF docx must yield a metafile-text record"
    rec = meta[0]
    assert rec["vision_model"] is None            # deterministic — no Gemini spent
    assert rec["provenance"] == "image-ocr"
    assert "140,000" in rec["full_text"] and "125,256" in rec["full_text"]
    assert rec["record_key"].endswith(rec["content_sha256"])


def test_build_incremental_reuses_prior(tmp_path):
    ref = _emf_docx_ref()
    if ref is None:
        pytest.skip("corpus EMF docx not present")
    out = tmp_path / "s.jsonl"
    first = S.load(S.build(refs=[ref], out=out, write_report=False) and out)
    second = S.load(S.build(refs=[ref], out=out, write_report=False) and out)
    assert first == second and first  # record_key match ⇒ verbatim reuse, stable output


# --------------------------------------------------------------------------- load robustness
def test_load_schema_mismatch_returns_empty(tmp_path):
    p = tmp_path / "bad.jsonl"
    p.write_text(json.dumps({"schema": "image-ocr-store", "version": 999}) + "\n"
                 + json.dumps({"rel": "x", "full_text": "y"}) + "\n", encoding="utf-8")
    assert S.load(p) == []


def test_load_missing_returns_empty(tmp_path):
    assert S.load(tmp_path / "nope.jsonl") == []


# --------------------------------------------------------------------------- extract appendix
class _Ref:
    def __init__(self, rel, ext="pdf"):
        self.rel, self.ext, self.name = rel, ext, rel.split("/")[-1]


@pytest.fixture()
def synth(tmp_path, monkeypatch):
    store = [
        {"rel": "プロジェクト/東都/未来予測.pdf", "locus": "ページ5", "source": "pdf-page-ocr",
         "full_text": "投資実装係数=(生産性向上率+コスト削減率)×ROI倍率"},
        {"rel": "プロジェクト/東都/調査.docx", "locus": "image1.emf", "source": "metafile-text",
         "full_text": "機械学習(ML)エンジニア 中央値 140,000 データエンジニア 平均 125,256"},
    ]
    monkeypatch.setattr(S, "load", lambda path=None: store)
    S._BYREL_CACHE.clear()
    return store


def test_appendix_off_is_empty(synth, monkeypatch):
    monkeypatch.delenv("RAG_IMAGE_OCR_STORE", raising=False)
    assert S.appendix_for(_Ref("プロジェクト/東都/未来予測.pdf")) == ""


def test_appendix_on_pdf_uses_page_marker(synth, monkeypatch):
    monkeypatch.setenv("RAG_IMAGE_OCR_STORE", "1")
    ap = S.appendix_for(_Ref("プロジェクト/東都/未来予測.pdf"))
    assert "[ページ5]" in ap and "投資実装係数" in ap        # file_grep page attribution works


def test_prepend_leads_the_text(synth, monkeypatch):
    monkeypatch.setenv("RAG_IMAGE_OCR_STORE", "1")
    out = S.prepend(_Ref("プロジェクト/東都/未来予測.pdf"), "元の本文")
    assert out.startswith("[ページ5]") and out.endswith("元の本文")  # evidence leads ⇒ survives 8000 trunc


def test_prepend_off_returns_text_unchanged(synth, monkeypatch):
    monkeypatch.delenv("RAG_IMAGE_OCR_STORE", raising=False)
    assert S.prepend(_Ref("プロジェクト/東都/未来予測.pdf"), "元の本文") == "元の本文"


def test_appendix_docx_metafile_served(synth, monkeypatch):
    """docx EMF (metafile-text) IS merged — the exact ML平均 143,000/125,256 idx8 needs is only in the EMF."""
    monkeypatch.setenv("RAG_IMAGE_OCR_STORE", "1")
    ap = S.appendix_for(_Ref("プロジェクト/東都/調査.docx", ext="docx"))
    assert "[画像OCR: image1.emf]" in ap and "125,256" in ap


def test_appendix_unknown_rel_empty(synth, monkeypatch):
    monkeypatch.setenv("RAG_IMAGE_OCR_STORE", "1")
    assert S.appendix_for(_Ref("プロジェクト/東都/無関係.pdf")) == ""


# --------------------------------------------------------------------------- extract() integration + OFF identity
def test_extract_docx_off_identical_on_prepends_emf(monkeypatch):
    """docx extract is byte-identical when OFF; when ON the EMF salary table LEADS the text (idx8 evidence,
    within the read_office 8000-char window)."""
    from src.rag.extract import extract
    ref = _emf_docx_ref()
    if ref is None:
        pytest.skip("corpus EMF docx not present")
    monkeypatch.delenv("RAG_IMAGE_OCR_STORE", raising=False)
    S.reset_cache()
    off = extract(ref).text
    assert "143,000" not in off                               # EMF not in the plain body extract
    monkeypatch.setenv("RAG_IMAGE_OCR_STORE", "1")
    S.reset_cache()
    on = extract(ref).text
    assert on.startswith("[画像OCR: image1.emf]") and on.endswith(off)  # prepended, additive
    assert on.index("143,000") < 8000                         # within read_office truncation window


def test_extract_pdf_prepends_page_ocr(monkeypatch):
    """A scanned/image PDF page's OCR is prepended to the PDF extract when the store is ON."""
    from src.rag.corpus import walk, nfc
    from src.rag.extract import extract
    pdf = next((r for r in walk() if r.ext == "pdf" and "未来予測" in nfc(r.name)), None)
    if pdf is None:
        pytest.skip("corpus PDF not present")
    rec = {"rel": nfc(pdf.rel), "locus": "ページ5", "source": "pdf-page-ocr",
           "full_text": "投資実装係数=(生産性向上率+コスト削減率)×ROI倍率"}
    monkeypatch.setattr(S, "load", lambda path=None: [rec])
    S._BYREL_CACHE.clear()
    monkeypatch.setenv("RAG_IMAGE_OCR_STORE", "1")
    on = extract(pdf).text
    assert on.startswith("[ページ5]") and "投資実装係数" in on
    assert on.index("投資実装係数") < 8000


# --------------------------------------------------------------------------- surface unchanged (no new tools)
def test_no_new_serve_tools(monkeypatch):
    """SOT-2684 integrates via the extractor, not a new tool ⇒ the fact_layer tool set is unchanged."""
    from src.rag.agent import fact_layer
    monkeypatch.setenv("RAG_FACT_LAYER", "1")
    monkeypatch.setenv("RAG_IMAGE_OCR_STORE", "1")
    monkeypatch.delenv("RAG_DOC_REACH_STORE", raising=False)
    names = {t[0] for t in fact_layer.tools()}
    # image OCR adds NO tool of its own (doc_reach tools stay gated on their own flag)
    assert "image_ocr_lookup" not in names
    assert "doc_table_lookup" not in names and "doc_fulltext_search" not in names
