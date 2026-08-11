"""OCR persistence store (SOT-2650 / 事前計算事実層 追補 — 画像PDFの build 時転記永続化).

SOT-2526 added a *live* Gemini-vision OCR fallback for image-only PDFs (``RAG_PDF_OCR``), with two
known limits it named as follow-ups:

* the transcription happens **at serve time** (473s/question latency, and it silently degrades to
  nothing under ``RAG_FORBID_GEMINI=1`` — the Sonnet dev run therefore cannot read any scanned PDF);
* the strict "text layer is empty" trigger misses **near-empty** scans (最終報告 PDFs whose only text
  layer is a stray letterhead line), so those never OCR at all.

This module is the "index 時に一度だけ OCR して永続" correction: at build time — where Gemini use is
allowed (前処理限定) — transcribe every *effectively image-only* PDF once and bake the text into
``artifacts/ocr_store.jsonl``. The serve path then reads the persisted text with **no genai call**.

Design invariants (mirroring the sibling pre-stores — case_master / id_master / diff_store):
* **Opt-in at serve time.** :func:`enabled` (``RAG_OCR_STORE``) gates runtime consultation only;
  default OFF ⇒ the champion serve path is byte-identical.
* **Opt-in at build time too.** Unlike the LLM-free siblings this build step *spends Gemini vision
  calls*, so it only runs under ``RAG_OCR_STORE_BUILD`` (default OFF) and reuses prior records
  (matched by rel+size) instead of re-transcribing on every rebuild.
* **Fail-open.** A missing/mismatched artifact, a size drift, or a non-eligible PDF all fall through
  to the existing text-layer / live-OCR behaviour in :func:`src.rag.extract.plain.extract_pdf`.
* **Question-independent.** The universe is structural: every corpus PDF whose extracted text-layer
  body is under :data:`BODY_MAX_CHARS` (measured corpus gap: scans ≤43 chars vs text PDFs ≥3375).
  No question is consulted; the whole page raster is transcribed verbatim.
"""
from __future__ import annotations

import functools
import json
import os
import unicodedata
from pathlib import Path
from typing import Any, Sequence

from config import settings
from src.rag.corpus import FileRef, nfc, walk

SCHEMA = "ocr-store"
SCHEMA_VERSION = 1

_ON = {"1", "true", "yes", "on"}

# An "effectively image-only" PDF: text-layer body (page markers stripped) below this many chars.
# Corpus-measured gap: scanned PDFs yield 0-43 chars (stray letterhead lines), the smallest real
# text-layer PDF yields 3375 — 200 sits safely inside that gap on the fail-closed side.
BODY_MAX_CHARS = 200


def enabled() -> bool:
    """True when the serve path may consult the persisted OCR text (default OFF — opt-in)."""
    return os.getenv("RAG_OCR_STORE", "0").strip().lower() in _ON


def build_enabled() -> bool:
    """True when the index build may spend Gemini vision calls to (re)build the store (default OFF)."""
    return os.getenv("RAG_OCR_STORE_BUILD", "0").strip().lower() in _ON


def default_out_path() -> Path:
    return settings.ARTIFACTS_DIR / "ocr_store.jsonl"


def default_report_path() -> Path:
    return settings.ARTIFACTS_DIR / "ocr_store_build_report.json"


def is_effectively_image_only(text_layer: str) -> bool:
    """Structural eligibility: the pdfplumber/pypdf text layer carries (almost) no body text."""
    from src.rag.extract.plain import _text_layer_body

    return len(_text_layer_body(text_layer or "").strip()) < BODY_MAX_CHARS


def _sig(path: Path) -> tuple[int, int]:
    try:
        st = Path(path).stat()
        return int(st.st_size), int(st.st_mtime)
    except OSError:
        return (0, 0)


def _text_layer(ref: FileRef) -> str:
    """The raw pdf text layer WITHOUT any OCR/store fallback (build-side eligibility probe)."""
    out: list[str] = []
    try:
        import pdfplumber

        with pdfplumber.open(str(ref.path)) as pdf:
            for pi, page in enumerate(pdf.pages, 1):
                out.append(f"[ページ{pi}]")
                txt = page.extract_text() or ""
                if txt.strip():
                    out.append(txt)
    except Exception:  # noqa: BLE001 — an unreadable PDF is simply not eligible here
        return ""
    return "\n".join(out)


def _ocr_pages(path: Path) -> tuple[str | None, int]:
    """Transcribe every full-page raster via Gemini vision → (text, n_pages); (None, 0) when not rasterised."""
    from src.rag import llm
    from src.rag.extract.vision import _OCR_PROMPT, pdf_page_images

    try:
        pages = pdf_page_images(path)
    except Exception:  # noqa: BLE001
        return None, 0
    if not pages:
        return None, 0
    out: list[str] = []
    for pi, (data, mime) in enumerate(pages, 1):
        out.append(f"[ページ{pi}]")
        txt = llm.generate(_OCR_PROMPT, images=[llm.Image(data=data, mime_type=mime)],
                           model=llm.settings.VISION_MODEL, max_output_tokens=2048)
        if txt and txt.strip():
            out.append(txt.strip())
    body = "".join(p for p in out if not p.startswith("[ページ"))
    return ("\n".join(out) if body.strip() else None), len(pages)


def build(refs: Sequence[FileRef] | None = None, out_path: Path | None = None) -> dict[str, Any]:
    """OCR every effectively-image-only corpus PDF once and persist the transcription.

    Incremental: an existing record whose (rel, size) still matches is reused verbatim — a rebuild
    never re-spends vision calls on unchanged files. Per-file failures are recorded, not raised.
    """
    refs = list(refs) if refs is not None else list(walk())
    out = Path(out_path) if out_path is not None else default_out_path()

    prior: dict[str, dict[str, Any]] = {}
    if out.exists():
        try:
            for line in out.read_text(encoding="utf-8").splitlines():
                rec = json.loads(line)
                if rec.get("rel"):
                    prior[rec["rel"]] = rec
        except Exception:  # noqa: BLE001 — a corrupt prior store is rebuilt from scratch
            prior = {}

    records: list[dict[str, Any]] = []
    skipped: list[dict[str, str]] = []
    reused = transcribed = 0
    for ref in refs:
        if ref.ext != "pdf":
            continue
        layer = _text_layer(ref)
        if not is_effectively_image_only(layer):
            continue
        rel = nfc(ref.rel)
        size, mtime = _sig(ref.path)
        old = prior.get(rel)
        if old and old.get("size") == size and old.get("text"):
            records.append(old)
            reused += 1
            continue
        try:
            text, n_pages = _ocr_pages(ref.path)
        except Exception as e:  # noqa: BLE001 — one bad file must not sink the store
            skipped.append({"rel": rel, "reason": f"{type(e).__name__}: {e}"})
            continue
        if not text:
            skipped.append({"rel": rel, "reason": "no full-page rasters / empty transcription"})
            continue
        from src.rag import llm

        records.append({"rel": rel, "size": size, "mtime": mtime, "pages": n_pages,
                        "model": llm.settings.VISION_MODEL, "text": text})
        transcribed += 1

    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        f.write(json.dumps({"schema": SCHEMA, "version": SCHEMA_VERSION}, ensure_ascii=False) + "\n")
        for rec in sorted(records, key=lambda r: r["rel"]):
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    report = {"files": len(records), "transcribed": transcribed, "reused": reused,
              "pages": sum(r.get("pages", 0) for r in records),
              "chars": sum(len(r.get("text", "")) for r in records),
              "skipped": skipped}
    default_report_path().write_text(json.dumps(report, ensure_ascii=False, indent=1), encoding="utf-8")
    load.cache_clear()
    return {"records": len(records), "report": report, "out": str(out)}


@functools.lru_cache(maxsize=4)
def _load_cached(path_str: str, sig: tuple) -> dict[str, dict[str, Any]]:
    store: dict[str, dict[str, Any]] = {}
    try:
        for line in Path(path_str).read_text(encoding="utf-8").splitlines():
            rec = json.loads(line)
            if rec.get("schema") == SCHEMA:
                if rec.get("version") != SCHEMA_VERSION:
                    return {}
                continue
            if rec.get("rel") and rec.get("text"):
                store[rec["rel"]] = rec
    except Exception:  # noqa: BLE001 — fail-open: unreadable store ⇒ empty ⇒ live behaviour
        return {}
    return store


def load(path: Path | None = None) -> dict[str, dict[str, Any]]:
    p = Path(path) if path is not None else default_out_path()
    return _load_cached(str(p), _sig(p))


# lru_cache is on the inner helper; expose cache_clear for build()/tests.
load.cache_clear = _load_cached.cache_clear  # type: ignore[attr-defined]


def lookup(ref: FileRef, path: Path | None = None) -> str | None:
    """Persisted OCR text for ``ref``, or ``None`` (unknown rel / size drift ⇒ fail-open)."""
    rec = load(path).get(nfc(ref.rel))
    if not rec:
        return None
    size, _ = _sig(ref.path)
    if rec.get("size") != size:
        return None
    return rec.get("text") or None
