"""Extraction for non-Office files: pdf, csv/tsv, md/txt/toml/lock, py, ipynb, json."""
from __future__ import annotations

import json
import re
from pathlib import Path

from src.rag.corpus import FileRef


def read_text_any(path: Path) -> str:
    try:
        return Path(path).read_text(encoding="utf-8", errors="ignore")
    except Exception:
        return ""


# A page-marker line ("[ページ12]") carries no content; strip these to judge whether the text layer
# actually yielded anything.  An image-only (scanned) PDF returns only markers → an empty body.
_PAGE_MARKER_RE = re.compile(r"^\[ページ\d+\]$")


def _text_layer_body(pdf_text: str) -> str:
    """The extracted PDF text minus its structural page markers (for the image-only-PDF check)."""
    return "".join(ln for ln in pdf_text.splitlines()
                   if ln.strip() and not _PAGE_MARKER_RE.match(ln.strip()) and ln.strip() != "[表]")


def extract_pdf(ref: FileRef) -> str:
    """Text layer via pdfplumber (keeps tables as pipe rows); image-only PDFs fall back to vision OCR.

    A scanned 会議録 / 報告書 has no searchable text layer, so pdfplumber/pypdf return only the page
    markers.  When the text layer is effectively image-only and the persisted OCR store is enabled
    (:func:`src.rag.index.ocr_store.enabled`, ``RAG_OCR_STORE`` default OFF), return the build-time
    transcription — no genai call at serve time.  When the text layer is strictly empty AND the live
    OCR fallback is enabled (:func:`src.rag.extract.vision.pdf_ocr_enabled`, ``RAG_PDF_OCR`` default
    OFF), transcribe the page rasters via Gemini vision.  Both default OFF ⇒ byte-identical to the
    previous text-only behaviour.
    """
    out: list[str] = []
    try:
        import pdfplumber

        with pdfplumber.open(str(ref.path)) as pdf:
            for pi, page in enumerate(pdf.pages, 1):
                out.append(f"[ページ{pi}]")
                txt = page.extract_text() or ""
                if txt.strip():
                    out.append(txt)
                for tbl in page.extract_tables() or []:
                    out.append("[表]")
                    for row in tbl:
                        out.append(" | ".join("" if c is None else str(c) for c in row))
    except Exception:
        try:
            import pypdf

            for pi, page in enumerate(pypdf.PdfReader(str(ref.path)).pages, 1):
                out.append(f"[ページ{pi}]\n{page.extract_text() or ''}")
        except Exception:
            out = []
    text = "\n".join(out)
    from src.rag.index import ocr_store

    result: str | None = None
    if ocr_store.enabled() and ocr_store.is_effectively_image_only(text):
        stored = ocr_store.lookup(ref)
        if stored and stored.strip():
            result = stored
    if result is None and not _text_layer_body(text).strip():
        from src.rag.extract import vision

        if vision.pdf_ocr_enabled():
            ocr = vision.ocr_image_pdf(ref.path)
            if ocr and ocr.strip():
                result = ocr
    if result is None:
        result = text
    # SOT-2684: prepend the build-time image-OCR store (per-page OCR of text-poor pages) so read_pdf /
    # file_grep / FTS reach it (OFF ⇒ unchanged ⇒ byte-identical). Leads the text ⇒ survives the 8000-char
    # read_pdf truncation.
    from src.rag.index import image_ocr_store
    return image_ocr_store.prepend(ref, result)


def extract_csv(ref: FileRef) -> str:
    """CSV/TSV → header + row sample + light schema; big tables are summarised, not dumped."""
    import pandas as pd

    sep = "\t" if ref.ext == "tsv" else ","
    try:
        df = pd.read_csv(ref.path, sep=sep, encoding="utf-8", on_bad_lines="skip")
    except Exception:
        return read_text_any(ref.path)[:20000]
    lines = [f"列: {list(df.columns)}", f"行数: {len(df)}"]
    # numeric summary helps aggregation questions
    try:
        desc = df.describe(include="all").transpose()
        lines.append("[統計サマリ]")
        lines.append(desc.to_string())
    except Exception:
        pass
    lines.append("[先頭20行]")
    lines.append(df.head(20).to_string())
    return "\n".join(lines)


def extract_ipynb(ref: FileRef) -> str:
    try:
        nb = json.loads(read_text_any(ref.path))
    except Exception:
        return read_text_any(ref.path)[:20000]
    out: list[str] = []
    for cell in nb.get("cells", []):
        src = "".join(cell.get("source", []))
        if not src.strip():
            continue
        out.append(f"[{cell.get('cell_type', 'code')}セル]\n{src}")
        for o in cell.get("outputs", []):
            txt = o.get("text") or (o.get("data", {}) or {}).get("text/plain")
            if txt:
                out.append("[出力] " + ("".join(txt) if isinstance(txt, list) else str(txt))[:1500])
    return "\n".join(out)


def extract_code(ref: FileRef) -> str:
    return f"[コード {ref.name}]\n" + read_text_any(ref.path)


def extract_json(ref: FileRef) -> str:
    txt = read_text_any(ref.path)
    try:
        return f"[JSON {ref.name}]\n" + json.dumps(json.loads(txt), ensure_ascii=False, indent=1)
    except Exception:
        return f"[JSON {ref.name}]\n" + txt


def extract_generic_text(ref: FileRef) -> str:
    return read_text_any(ref.path)
