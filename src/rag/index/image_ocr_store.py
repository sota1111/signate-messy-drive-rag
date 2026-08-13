"""Build-time image/OCR evidence store (SOT-2684 / cycle7 K1 — 画像ロック証拠の焼き込み).

Sonnet gold100 の cycle6 abstain のうち idx8/50/52/68 は、答えに必要な数値/語が **テキスト抽出で
到達できない画像に閉じている** ため BUDGET_EXHAUSTED 棄権になっていた（``docs/ai/sonnet_cycle_analysis/
cycle7.md`` §1 K1）:

* **idx8**（東都 データサイエンティスト調査.docx）— 職種別 米国平均給与の比較表が本文テキストに無く、
  埋め込み ``word/media/image1.emf`` の中にある（ML エンジニア中央値 140,000 − データエンジニア 125,256）。
  EMF はベクタメタファイルで、文字は EXTTEXTOUTW レコードの UTF-16 として **決定論的に取り出せる**
  （vision 不要）。
* **idx52**（みなみ野 最終報告.pdf）— 「別契約」の記載が画像ページ（テキストレイヤ空）にある。
* **idx68**（東都 未来予測.pdf）— 「投資実装係数」の計算式ページが画像（テキストレイヤ空）。

本モジュールは build 時に一度だけ、**質問非依存**に「テキスト抽出でカバーされない画像/画像ページを網羅」
して OCR/抽出し、``artifacts/image_ocr_store.jsonl`` へ焼き込む。serve 側（:mod:`src.rag.agent.doc_lookup_lane`）
は既存の ``doc_fulltext_search`` / ``doc_table_lookup`` の検索対象へこの画像由来チャンクを合流させるだけで、
**genai 呼び出しゼロ**（``RAG_FORBID_GEMINI=1`` でも serve は動く）。

質問非依存の universe（gold 参照なし・ハードコードなし）:
* **docx/pptx 埋め込みメタファイル**（EMF/WMF）: 文字レコードを決定論抽出（vision を使わない）。
* **docx/pptx 埋め込みラスタ画像**（png/jpeg/…, 一定サイズ以上）: build 限定 vision で OCR。
* **PDF の画像/テキスト希薄ページ**（ページ本文テキスト < :data:`PAGE_BODY_MAX_CHARS`）: ページを
  ラスタライズして build 限定 vision で OCR。

Design invariants（sibling の ocr_store / nb_chart_store と同一）:
* **Opt-in at serve time.** :func:`enabled` (``RAG_IMAGE_OCR_STORE``) が runtime 参照のみ gate。default OFF ⇒
  champion serve path は byte-identical。
* **Opt-in at build time too.** vision を spend しうるので build も ``RAG_IMAGE_OCR_STORE_BUILD`` (default
  OFF) で gate し、prior 記録(content sha)を再利用して再課金しない。
* **Question-independent / No hardcoding.** universe は文書構造から導く。gold 値・idx は一切持たない。
* **Fail-open.** artifact 欠落・解析不能・vision 失敗はすべて空/スキップへフォールバック（回帰ゼロ）。
"""
from __future__ import annotations

import hashlib
import io
import json
import os
import re
import unicodedata
import zipfile
from pathlib import Path
from typing import Any, Sequence

from config import settings
from src.rag.corpus import FileRef, nfc, walk

SCHEMA = "image-ocr-store"
SCHEMA_VERSION = 1

_ON = {"1", "true", "yes", "on"}

# A PDF page is "image / text-poor" when its own extractable text body is below this many chars — the
# same corpus-measured gap ocr_store uses at document scope (scans yield 0-43, real text pages ≥ hundreds).
# 200 keeps a mixed PDF's genuine text pages out while capturing its scanned/图 pages.
PAGE_BODY_MAX_CHARS = 200
# Raster media below this pixel area are logos/icons/decorations, not evidence — skip (vision spend guard).
MIN_RASTER_AREA = 300 * 200
# Bound the total vision reads a single build spends (fail-safe; ~136 in the current corpus).
DEFAULT_MAX_VISION = 400
MAX_FULLTEXT_CHARS = 200_000
_RASTER_EXTS = ("png", "jpg", "jpeg", "gif", "bmp", "tif", "tiff")
_META_EXTS = ("emf", "wmf")


def enabled() -> bool:
    """True when the serve path may consult the image-OCR chunks (default OFF — opt-in)."""
    return os.getenv("RAG_IMAGE_OCR_STORE", "0").strip().lower() in _ON


def build_enabled() -> bool:
    """True when the index build may spend Gemini vision calls to (re)build the store (default OFF)."""
    return os.getenv("RAG_IMAGE_OCR_STORE_BUILD", "0").strip().lower() in _ON


def _max_vision() -> int:
    try:
        return max(0, int(os.getenv("IMAGE_OCR_MAX_VISION", str(DEFAULT_MAX_VISION))))
    except ValueError:
        return DEFAULT_MAX_VISION


def default_out_path() -> Path:
    return settings.ARTIFACTS_DIR / "image_ocr_store.jsonl"


def default_report_path() -> Path:
    return settings.ARTIFACTS_DIR / "image_ocr_store_build_report.json"


# --------------------------------------------------------------------------- EMF/WMF text (deterministic)
# Font-table / MS style tokens that appear as noise between the real strings in a metafile's text records.
_META_NOISE = {"arial", "unicode", "ms", "ゴシック", "icode", "mincho", "明朝", "calibri", "meiryo",
               "メイリオ", "times", "newroman", "symbol", "wingdings", "segoe"}
# A run of readable characters (CJK / alnum / common punctuation) inside the UTF-16 decoded blob.
_META_RUN = re.compile(r"[0-9A-Za-z぀-ヿ㐀-鿿＀-￯,.\-%()／・：:円ドル万千]{2,}")
_NUM_HEAD = re.compile(r"^[\d][\d,\.]*")
# Metafile positioning/record bytes decode to CJK-ext-A (U+3400–U+4DBF) and PUA (U+E000–U+F8FF) glyphs that
# are pure noise between the real strings; strip them so a residual "䀀㘀Arial" collapses to font noise.
_ARTIFACT_CHARS = re.compile("[\u3400-\u4dbf\ue000-\uf8ff]")
# EMF style-run marker glyph (U+8000) + a trailing style code (\u8000% / \u8000K / \u8000F) \u2014 pure noise.
_STYLE_MARK = re.compile("^\u8000.?$")


def _clean_meta_token(tok: str) -> str:
    """Normalise (NFKC), drop font/style noise, and strip metafile-artifact glyphs from a text run."""
    tok = unicodedata.normalize("NFKC", tok)          # ＭＳ→MS, full-width digits→ascii
    tok = _ARTIFACT_CHARS.sub("", tok).strip()        # remove ext-A / PUA positioning glyphs
    if len(tok) < 2:
        return ""
    if _STYLE_MARK.match(tok):  # "耀%" / "耀K" style-run marker glyphs
        return ""
    low = tok.lower().strip("・.")
    if not low or low in _META_NOISE:
        return ""
    if any(low == n or (low.startswith(n) and len(low) - len(n) <= 1) for n in _META_NOISE):
        return ""
    m = _NUM_HEAD.match(tok)
    if m and len(m.group(0)) >= len(tok) - 1:  # "94,000獬" → "94,000" (strip 1-char tail artifact)
        return m.group(0)
    return tok


def emf_text(blob: bytes) -> str:
    """Deterministically recover the readable text of an EMF/WMF metafile (UTF-16 text records).

    Metafiles store their glyph strings as UTF-16LE inside EMR_EXTTEXTOUTW records. Decoding the whole
    blob as UTF-16LE and keeping the readable runs (minus font-table noise) recovers the table/label text
    without any raster/vision step — reproducible and $0. Returns "" when nothing legible is found.
    """
    if not blob:
        return ""
    try:
        u16 = blob.decode("utf-16le", "ignore")
    except Exception:  # noqa: BLE001
        return ""
    runs = [_clean_meta_token(t) for t in _META_RUN.findall(u16)]
    runs = [t for t in runs if t]
    # Collapse consecutive duplicate font-name residue and join.
    out: list[str] = []
    for t in runs:
        if out and out[-1] == t:
            continue
        out.append(t)
    return " ".join(out).strip()


# --------------------------------------------------------------------------- vision OCR (build-only)
_PAGE_OCR_PROMPT = (
    "これは社内プロジェクト文書（提案書/報告書/会議録等）の1ページ画像です。"
    "画像に写る文字・表・数値・箇条書きを省略や要約をせず、日本語でそのまま書き起こしてください。"
    "数式・係数・契約区分（別契約等）・ID・担当者・日付・ステータスも欠かさず転記。"
    "レイアウトが崩れても内容を欠かさず、推測や創作はせず、読み取れる文字のみを出力してください。"
)


def _vision_ocr(png: bytes) -> str:
    """Transcribe one page/image raster via Gemini vision → verbatim text ("" on failure)."""
    from src.rag import llm

    try:
        txt = llm.generate(_PAGE_OCR_PROMPT, images=[llm.Image(data=png, mime_type="image/png")],
                           model=llm.settings.VISION_MODEL, max_output_tokens=2048, temperature=0.0)
    except Exception:  # noqa: BLE001 — one failed vision call must not sink the record
        return ""
    return (txt or "").strip()


def _render_pdf_page_png(page: Any, dpi: int = 170) -> bytes | None:
    try:
        pix = page.get_pixmap(dpi=dpi)
        return pix.tobytes("png")
    except Exception:  # noqa: BLE001
        return None


def _raster_area(data: bytes) -> int:
    try:
        from PIL import Image
        with Image.open(io.BytesIO(data)) as im:
            return int(im.size[0]) * int(im.size[1])
    except Exception:  # noqa: BLE001
        return 0


def _raster_to_png(data: bytes) -> bytes | None:
    """Normalise an arbitrary raster to PNG bytes for vision (already-PNG passes through)."""
    try:
        from PIL import Image
        with Image.open(io.BytesIO(data)) as im:
            buf = io.BytesIO()
            im.convert("RGB").save(buf, format="PNG")
            return buf.getvalue()
    except Exception:  # noqa: BLE001
        return None


# --------------------------------------------------------------------------- per-document extraction
def _record(ref: FileRef, *, source: str, locus: str, sha: str, full_text: str,
            vision_model: str | None) -> dict[str, Any]:
    ft = full_text[:MAX_FULLTEXT_CHARS]
    return {
        "project": nfc(ref.project), "rel": nfc(ref.rel), "name": nfc(ref.name), "ext": ref.ext,
        "provenance": "image-ocr", "source": source, "locus": locus,
        "content_sha256": sha, "vision_model": vision_model,
        "n_chars": len(ft), "n_tables": 0, "tables": [], "full_text": ft,
        "record_key": f"{nfc(ref.rel)}#{locus}#{sha}",
    }


def _office_media(ref: FileRef) -> list[tuple[str, bytes]]:
    """``(zip-entry-name, bytes)`` for every embedded media file of a docx/pptx (encrypted ⇒ [])."""
    try:
        with zipfile.ZipFile(str(ref.path)) as z:
            return [(n, z.read(n)) for n in z.namelist() if "/media/" in n.lower()]
    except Exception:  # noqa: BLE001 — encrypted/corrupt office file → no media (fail-open)
        return []


def compute_office(ref: FileRef, *, prior: dict[str, dict[str, Any]],
                   vision_budget: list[int]) -> list[dict[str, Any]]:
    """One docx/pptx's image records: metafile text (deterministic) + large raster OCR (vision)."""
    out: list[dict[str, Any]] = []
    for entry, data in _office_media(ref):
        ext = entry.rsplit(".", 1)[-1].lower()
        sha = hashlib.sha256(data).hexdigest()
        locus = nfc(entry.split("/media/", 1)[-1]) if "/media/" in entry.lower() else nfc(entry)
        rec_key = f"{nfc(ref.rel)}#{locus}#{sha}"
        old = prior.get(rec_key)
        if old is not None:
            out.append(old)
            continue
        if ext in _META_EXTS:
            text = emf_text(data)
            if len(text) >= 8:
                out.append(_record(ref, source="metafile-text", locus=locus, sha=sha,
                                   full_text=text, vision_model=None))
            continue
        if ext in _RASTER_EXTS and _raster_area(data) >= MIN_RASTER_AREA and vision_budget[0] > 0:
            png = _raster_to_png(data)
            if png is None:
                continue
            text = _vision_ocr(png)
            vision_budget[0] -= 1
            if len(text) >= 8:
                from src.rag import llm
                out.append(_record(ref, source="media-ocr", locus=locus, sha=sha,
                                   full_text=text, vision_model=llm.settings.VISION_MODEL))
    return out


def compute_pdf(ref: FileRef, *, prior: dict[str, dict[str, Any]],
                vision_budget: list[int]) -> list[dict[str, Any]]:
    """One PDF's image/text-poor pages OCR'd via vision (one record per such page)."""
    try:
        import fitz  # PyMuPDF
    except Exception:  # noqa: BLE001
        return []
    out: list[dict[str, Any]] = []
    try:
        doc = fitz.open(str(ref.path))
    except Exception:  # noqa: BLE001 — unreadable/encrypted PDF → skip
        return []
    try:
        for pi, page in enumerate(doc, 1):
            try:
                body = (page.get_text() or "").strip()
            except Exception:  # noqa: BLE001
                body = ""
            if len(body) >= PAGE_BODY_MAX_CHARS:
                continue
            png = _render_pdf_page_png(page)
            if png is None:
                continue
            sha = hashlib.sha256(png).hexdigest()
            locus = f"ページ{pi}"
            rec_key = f"{nfc(ref.rel)}#{locus}#{sha}"
            old = prior.get(rec_key)
            if old is not None:
                out.append(old)
                continue
            if vision_budget[0] <= 0:
                continue
            text = _vision_ocr(png)
            vision_budget[0] -= 1
            if len(text) >= 8:
                from src.rag import llm
                out.append(_record(ref, source="pdf-page-ocr", locus=locus, sha=sha,
                                   full_text=text, vision_model=llm.settings.VISION_MODEL))
    finally:
        doc.close()
    return out


# --------------------------------------------------------------------------- build / io
def _universe(refs: Sequence[FileRef]) -> list[FileRef]:
    out = [r for r in refs if r.ext in ("docx", "pptx", "pdf") and not nfc(r.name).startswith("~$")]
    out.sort(key=lambda r: r.rel)
    return out


def build(refs: Sequence[FileRef] | None = None, *, out: Path | None = None,
          write_report: bool = True) -> dict[str, Any]:
    """Extract every text-hidden image's content once and bake the store (question-independent).

    Incremental: an existing record whose ``record_key`` (rel#locus#content-sha) still matches is reused
    verbatim — a rebuild never re-spends vision on an unchanged image/page.
    """
    refs = list(refs) if refs is not None else list(walk())
    prior = _load_prior(out)
    vision_budget = [_max_vision()]

    records: list[dict[str, Any]] = []
    for ref in _universe(refs):
        try:
            if ref.ext in ("docx", "pptx"):
                records.extend(compute_office(ref, prior=prior, vision_budget=vision_budget))
            elif ref.ext == "pdf":
                records.extend(compute_pdf(ref, prior=prior, vision_budget=vision_budget))
        except Exception:  # noqa: BLE001 — one bad document must not sink the build
            continue

    stats = write_store(records, out)
    report = {
        "schema": SCHEMA, "version": SCHEMA_VERSION,
        "records": len(records),
        "metafile_text": sum(1 for r in records if r.get("source") == "metafile-text"),
        "media_ocr": sum(1 for r in records if r.get("source") == "media-ocr"),
        "pdf_page_ocr": sum(1 for r in records if r.get("source") == "pdf-page-ocr"),
        "vision_remaining": vision_budget[0],
        "vision_capped": vision_budget[0] <= 0,
        "docs": len({r.get("rel") for r in records}),
    }
    if write_report:
        rp = default_report_path()
        rp.parent.mkdir(parents=True, exist_ok=True)
        rp.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return {"records": stats["records"], "report": report}


def write_store(records: Sequence[dict[str, Any]], path: Path | None = None) -> dict[str, int]:
    """Atomically write a reproducible JSONL store (schema header + (rel, locus) sorted rows)."""
    out = Path(path) if path is not None else default_out_path()
    out.parent.mkdir(parents=True, exist_ok=True)
    ordered = sorted(records, key=lambda r: (r.get("rel", ""), str(r.get("locus", ""))))
    tmp = out.with_suffix(out.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8", newline="\n") as handle:
        handle.write(json.dumps({"schema": SCHEMA, "version": SCHEMA_VERSION},
                                ensure_ascii=False, sort_keys=True) + "\n")
        for rec in ordered:
            handle.write(json.dumps(rec, ensure_ascii=False, sort_keys=True) + "\n")
    tmp.replace(out)
    reset_cache()
    return {"records": len(ordered)}


def _load_prior(out: Path | None) -> dict[str, dict[str, Any]]:
    prior: dict[str, dict[str, Any]] = {}
    for rec in load(out):
        key = rec.get("record_key")
        if key:
            prior[key] = rec
    return prior


# --------------------------------------------------------------------------- load / read
_LOAD_CACHE: dict[str, list[dict[str, Any]]] = {}


def reset_cache() -> None:
    _LOAD_CACHE.clear()
    _BYREL_CACHE.clear()


def load(path: Path | None = None) -> list[dict[str, Any]]:
    """Load the image-OCR records (memoized). ``[]`` when absent/unreadable/schema-mismatch (回帰ゼロ)."""
    out = Path(path) if path is not None else default_out_path()
    key = str(out)
    if key in _LOAD_CACHE:
        return _LOAD_CACHE[key]
    rows: list[dict[str, Any]] = []
    try:
        with open(out, encoding="utf-8") as handle:
            header = handle.readline()
            meta = json.loads(header) if header.strip() else {}
            if isinstance(meta, dict) and meta.get("schema") == SCHEMA \
                    and meta.get("version") == SCHEMA_VERSION:
                for line in handle:
                    line = line.strip()
                    if line:
                        rows.append(json.loads(line))
    except Exception:  # noqa: BLE001
        rows = []
    _LOAD_CACHE[key] = rows
    return rows


_BYREL_CACHE: dict[str, dict[str, list[dict[str, Any]]]] = {}


def _by_rel(path: Path | None = None) -> dict[str, list[dict[str, Any]]]:
    """``nfc(rel) → [record, …]`` (memoized) for the extract-layer appendix."""
    out = Path(path) if path is not None else default_out_path()
    key = str(out)
    cached = _BYREL_CACHE.get(key)
    if cached is not None:
        return cached
    index: dict[str, list[dict[str, Any]]] = {}
    for rec in load(out):
        index.setdefault(nfc(rec.get("rel") or ""), []).append(rec)
    _BYREL_CACHE[key] = index
    return index


# All three image sources fill a genuine text-layer gap and are merged into the extract text:
#  * ``pdf-page-ocr``  — scanned/image PDF pages whose native text layer is empty (idx52「別契約」/
#    idx68「投資実装係数」の計算式ページ).
#  * ``metafile-text`` — docx/pptx embedded EMF/WMF tables (idx8: 職種別 米国平均給与表; the EXACT
#    ML エンジニア平均 143,000 / データエンジニア 125,256 that idx8 (gold v4 = 17,744) needs is ONLY in the
#    EMF — the docx body prose carries only an approximate「約140,000」, so the table image is load-bearing).
#  * ``media-ocr``     — docx/pptx embedded raster images (charts/tables) above the logo-size guard.
_SERVE_SOURCES = frozenset({"pdf-page-ocr", "metafile-text", "media-ocr"})


def appendix_for(ref: FileRef, path: Path | None = None) -> str:
    """The image-OCR text for ``ref`` (locator-marked blocks), or "" when OFF/absent (回帰ゼロ).

    SOT-2684: the store is joined into the corpus **inside the format extractors** (:func:`office.extract_docx`
    / ``extract_pptx`` / :func:`plain.extract_pdf`) and **prepended** to the native text, so it flows into
    EVERY consumer — the ``read_office`` / ``read_pdf`` tools (which call the extractors directly and
    truncate at 8000 chars, so the evidence must lead), ``file_grep``, and the FTS / evidence indexes —
    not just an optional lookup tool the LLM may never call. Records in :data:`_SERVE_SOURCES` are joined,
    each prefixed with a locator marker (``[ページN]`` for PDF pages so ``file_grep`` page attribution
    works, ``[画像OCR: <locus>]`` for docx/pptx embedded media). ``RAG_IMAGE_OCR_STORE`` OFF ⇒ "" ⇒ the
    extract output (and therefore the whole serve path) is byte-identical.
    """
    if not enabled():
        return ""
    recs = _by_rel(path).get(nfc(ref.rel))
    if not recs:
        return ""
    blocks: list[str] = []
    for rec in sorted(recs, key=lambda r: str(r.get("locus") or "")):
        if rec.get("source") not in _SERVE_SOURCES:
            continue
        text = (rec.get("full_text") or "").strip()
        if not text:
            continue
        locus = str(rec.get("locus") or "")
        marker = f"[{locus}]" if locus.startswith("ページ") else f"[画像OCR: {locus}]"
        blocks.append(f"{marker}\n{text}")
    return "\n".join(blocks)


def prepend(ref: FileRef, text: str, path: Path | None = None) -> str:
    """Prepend :func:`appendix_for` (leading, so the read tools' 8000-char window keeps it) to ``text``.

    OFF/absent ⇒ returns ``text`` unchanged (byte-identical). Used by the three format extractors so the
    image-OCR evidence reaches ``read_office`` / ``read_pdf`` / ``file_grep`` / FTS uniformly."""
    ap = appendix_for(ref, path)
    return f"{ap}\n\n{text}" if ap else text


if __name__ == "__main__":
    summary = build()
    print(json.dumps(summary["report"], ensure_ascii=False, indent=2))
