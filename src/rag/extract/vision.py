"""Chart / figure understanding via Gemini Vision.

At index time we caption each PNG so it is retrievable by content ("AG_ratio ヒストグラム" …).
Precise readings (exact bar counts, highlighted words) are done at ANSWER time by re-attaching
the raw image to the generation call together with the specific question — see generate.py.
"""
from __future__ import annotations

from pathlib import Path

from src.rag.corpus import FileRef
from src.rag import llm

_CAPTION_PROMPT = (
    "これはある社内分析プロジェクトの図表画像です。日本語で簡潔に説明してください。"
    "(1)図の種類(ヒストグラム/散布図/ヒートマップ/棒グラフ等) "
    "(2)軸ラベル・凡例・タイトルに現れる変数名や単語 "
    "(3)ハイライトやマーカー(色付き)で強調された単語・数値があればそれ。"
    "推測や創作はせず、画像から読み取れる文字・数値のみ。200字以内。"
)


def load_image_bytes(path) -> bytes:
    return Path(path).read_bytes()


def caption_png(ref: FileRef) -> str:
    try:
        img = llm.Image(data=load_image_bytes(ref.path), mime_type="image/png")
        cap = llm.generate(_CAPTION_PROMPT, images=[img], model=llm.settings.VISION_MODEL,
                           max_output_tokens=300)
        return f"[図: {ref.name}] {cap}"
    except Exception as e:  # noqa: BLE001 — captioning is best-effort
        return f"[図: {ref.name}] (説明生成失敗: {e})"
