"""Unified extraction dispatch: FileRef -> ExtractedDoc(text + metadata).

Dispatches by extension, transparently decrypting password-protected Office files.
Vision captioning of PNGs is optional (costs Gemini calls) and off by default here;
build_index enables it.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from src.rag.corpus import FileRef

# office/plain/passwords pull heavy, build-only deps (openpyxl, python-pptx, msoffcrypto,
# pdfplumber). They are imported lazily inside extract() so that merely importing a light
# submodule of this package (e.g. `extract.glossary` at serve time) does NOT require them.


@dataclass
class ExtractedDoc:
    ref: FileRef
    text: str
    kind: str                       # docx/xlsx/pptx/pdf/csv/code/ipynb/json/text/image
    encrypted: bool = False
    error: str = ""
    meta: dict = field(default_factory=dict)


_OFFICE = {"docx", "xlsx", "pptx"}
_CODE = {"py", "toml", "lock"}
_TEXT = {"md", "txt"}


def extract(ref: FileRef, *, caption_images: bool = False) -> ExtractedDoc:
    from src.rag.extract import office, plain, passwords  # heavy, build-only deps

    ext = ref.ext
    try:
        if ext in _OFFICE:
            data = None
            encrypted = passwords.is_encrypted(ref.path)
            if encrypted:
                data = passwords.resolve(ref)
                if data is None:
                    return ExtractedDoc(ref, "", ext, encrypted=True,
                                        error="decryption failed")
            if ext == "docx":
                text = office.extract_docx(ref, data)
            elif ext == "xlsx":
                text = office.extract_xlsx(ref, data)
            else:
                text = office.extract_pptx(ref, data)
            return ExtractedDoc(ref, text, ext, encrypted=encrypted)

        if ext == "pdf":
            return ExtractedDoc(ref, plain.extract_pdf(ref), "pdf")
        if ext in ("csv", "tsv"):
            return ExtractedDoc(ref, plain.extract_csv(ref), "csv")
        if ext == "ipynb":
            return ExtractedDoc(ref, plain.extract_ipynb(ref), "ipynb")
        if ext == "json":
            return ExtractedDoc(ref, plain.extract_json(ref), "json")
        if ext in _CODE:
            return ExtractedDoc(ref, plain.extract_code(ref), "code")
        if ext in _TEXT:
            return ExtractedDoc(ref, plain.extract_generic_text(ref), "text")
        if ext == "png":
            if caption_images:
                from src.rag.extract import vision
                return ExtractedDoc(ref, vision.caption_png(ref), "image",
                                    meta={"image": True})
            return ExtractedDoc(ref, f"[図: {ref.name}]", "image", meta={"image": True})
    except Exception as e:  # noqa: BLE001
        return ExtractedDoc(ref, "", ext, error=f"{type(e).__name__}: {e}")
    return ExtractedDoc(ref, "", ext, error="unsupported")
