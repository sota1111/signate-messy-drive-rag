"""Chart / figure understanding via Gemini Vision.

At index time we caption each PNG so it is retrievable by content ("AG_ratio ヒストグラム" …).
Precise readings (exact bar counts, highlighted words) are done at ANSWER time by re-attaching
the raw image to the generation call together with the specific question — see generate.py.
"""
from __future__ import annotations

from pathlib import Path
import io
import hashlib

from src.rag.corpus import FileRef
from src.rag import llm

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


def caption_png(ref: FileRef) -> str:
    try:
        img = llm.Image(data=load_image_bytes(ref.path), mime_type="image/png")
        cap = llm.generate(_CAPTION_PROMPT, images=[img], model=llm.settings.VISION_MODEL,
                           max_output_tokens=300)
        return f"[図: {ref.name}] {cap}"
    except Exception as e:  # noqa: BLE001 — captioning is best-effort
        return f"[図: {ref.name}] (説明生成失敗: {e})"
