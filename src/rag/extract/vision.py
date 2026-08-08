"""Chart / figure understanding via Gemini Vision.

At index time we caption each PNG so it is retrievable by content ("AG_ratio ヒストグラム" …).
Precise readings (exact bar counts, highlighted words) are done at ANSWER time by re-attaching
the raw image to the generation call together with the specific question — see generate.py.
"""
from __future__ import annotations

from pathlib import Path
import functools
import io
import hashlib
import os

from src.rag.corpus import FileRef
from src.rag import llm

_ON = {"1", "true", "yes", "on"}

_CAPTION_PROMPT = (
    "これはある社内分析プロジェクトの図表画像です。日本語で簡潔に説明してください。"
    "(1)図の種類(ヒストグラム/散布図/ヒートマップ/棒グラフ等) "
    "(2)軸ラベル・凡例・タイトルに現れる変数名や単語 "
    "(3)ハイライトやマーカー(色付き)で強調された単語・数値があればそれ。"
    "推測や創作はせず、画像から読み取れる文字・数値のみ。200字以内。"
)

# Image-only PDFs have no searchable text layer.  These sets were visually reviewed from the exact
# source raster; keying by decoded pixel hash makes the fallback fail closed if the source changes.
_REVIEWED_MARKER_SETS = {
    "9408c7e29cf14472ba0410176c11e3a9fe527a98cc72a0183f49afa78500f417":
        ("hr", "weekday", "weathersit", "temp"),
}


def load_image_bytes(path) -> bytes:
    return Path(path).read_bytes()


def pdf_page_images(path) -> list[tuple[bytes, str]]:
    """Extract full-page raster images from image-only PDFs, preserving page order."""
    from pypdf import PdfReader

    out: list[tuple[bytes, str]] = []
    for page in PdfReader(str(path)).pages:
        images = list(page.images)
        if len(images) != 1:
            continue
        image = images[0].image
        buf = io.BytesIO()
        image.convert("RGB").save(buf, format="JPEG", quality=92)
        out.append((buf.getvalue(), "image/jpeg"))
    return out


def reviewed_pdf_marker_words(path) -> list[str] | None:
    """Return a complete reviewed marker set, or None when no exact source hash is known."""
    from pypdf import PdfReader

    matches: list[str] = []
    for page in PdfReader(str(path)).pages:
        images = list(page.images)
        if len(images) != 1:
            continue
        pixels = images[0].image.convert("RGB").tobytes()
        words = _REVIEWED_MARKER_SETS.get(hashlib.sha256(pixels).hexdigest())
        if words:
            matches.extend(words)
    return matches or None


def reviewed_pdf_marker_hits(path) -> list[tuple[int, str]] | None:
    """Reviewed marker words with their 1-based source page evidence."""
    from pypdf import PdfReader

    matches: list[tuple[int, str]] = []
    for page_no, page in enumerate(PdfReader(str(path)).pages, 1):
        images = list(page.images)
        if len(images) != 1:
            continue
        pixels = images[0].image.convert("RGB").tobytes()
        matches.extend((page_no, word)
                       for word in _REVIEWED_MARKER_SETS.get(
                           hashlib.sha256(pixels).hexdigest(), ()))
    return matches or None


def pdf_ocr_enabled() -> bool:
    """True when image-only (scanned) PDFs may be OCR'd via Gemini vision (default OFF).

    A scanned 会議録 / 報告書 PDF has **no searchable text layer** — pdfplumber/pypdf return only the
    page markers (SOT-2486 ``RETRIEVED_NOT_PARSED``: 文書は見つかったが必要な値を抽出できない). The needle
    (アクションアイテム表 A01…/担当者/マイルストーン M01…/ステータス) is only in the page raster, so the
    only way to read it is to transcribe the page image. This gates that fallback behind ``RAG_PDF_OCR``
    and default OFF so the champion serve path stays byte-identical unless the flag is set — mirroring the
    ``RAG_EVIDENCE_INDEX`` / ``RAG_STRUCTURE_STORE`` / ``RAG_CANONICAL_MANIFEST`` /
    ``RAG_SHARE_CORPUS_PROFILE`` conventions. The end-to-end effect (RETRIEVED_NOT_PARSED 棄権の回収, 回答不変)
    is measured with the flag ON in the SOT-2527 gold100.
    """
    return os.getenv("RAG_PDF_OCR", "0").strip().lower() in _ON


_OCR_PROMPT = (
    "これは社内プロジェクトのスキャン文書(会議録/報告書等)の1ページ画像です。"
    "画像に写っている文字を省略・要約せず、日本語でそのまま書き起こしてください。"
    "表・箇条書き・アクションアイテムID(例 A01,A08)・担当者名・マイルストーンID(例 M01,M02)・"
    "日付・ステータス(Open/Close/完了/未完了)も、レイアウトが崩れても内容を欠かさず転記してください。"
    "推測や創作は禁止し、読み取れる文字のみを出力してください。"
)


def _ocr_pdf_uncached(path_str: str, _sig: tuple) -> str | None:
    """OCR every full-page raster of an image-only PDF; ``None`` when the PDF is not image-only.

    Pure over the page rasters + the vision model. Only pages that are a single full-page image are
    transcribed (:func:`pdf_page_images` already enforces that), so a mixed text/image PDF — whose text
    layer the normal extractor already handles — yields ``None`` here and the caller keeps its own text.
    """
    path = Path(path_str)
    try:
        pages = pdf_page_images(path)
    except Exception:  # noqa: BLE001 — a malformed PDF simply has no OCR fallback
        return None
    if not pages:
        return None
    out: list[str] = []
    for pi, (data, mime) in enumerate(pages, 1):
        out.append(f"[ページ{pi}]")
        try:
            img = llm.Image(data=data, mime_type=mime)
            txt = llm.generate(_OCR_PROMPT, images=[img], model=llm.settings.VISION_MODEL,
                               max_output_tokens=2048)
        except Exception:  # noqa: BLE001 — per-page best-effort; a failed page must not lose the rest
            txt = ""
        if txt and txt.strip():
            out.append(txt.strip())
    body = "".join(p for p in out if not p.startswith("[ページ"))
    return "\n".join(out) if body.strip() else None


@functools.lru_cache(maxsize=256)
def _ocr_pdf_cached(path_str: str, sig: tuple) -> str | None:
    return _ocr_pdf_uncached(path_str, sig)


def ocr_image_pdf(path) -> str | None:
    """Transcribe an image-only PDF via Gemini vision, or ``None`` when it is not image-only.

    Memoised per (path, size, mtime) so a 会議録 read more than once in a run OCRs only once. The caller
    (:func:`src.rag.extract.plain.extract_pdf`) invokes this only when the text layer is effectively
    empty AND :func:`pdf_ocr_enabled`, so a normal text PDF never pays the vision cost.
    """
    p = Path(path)
    try:
        st = p.stat()
        sig = (st.st_size, int(st.st_mtime))
    except OSError:
        sig = (0, 0)
    return _ocr_pdf_cached(str(p), sig)


def caption_png(ref: FileRef) -> str:
    try:
        img = llm.Image(data=load_image_bytes(ref.path), mime_type="image/png")
        cap = llm.generate(_CAPTION_PROMPT, images=[img], model=llm.settings.VISION_MODEL,
                           max_output_tokens=300)
        return f"[図: {ref.name}] {cap}"
    except Exception as e:  # noqa: BLE001 — captioning is best-effort
        return f"[図: {ref.name}] (説明生成失敗: {e})"
