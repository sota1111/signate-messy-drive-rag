"""Resolve passwords for encrypted Office files in the corpus.

Two schemes are used in practice (see docs/corpus-notes.md):
  A) filename `..._pw-<token>.<ext>` → the password is literally `<token>`.
  B) rule `DA-<案件略号>-<契約開始日YYYYMMDD>-<拡張子>` where 案件略号 is the glossary 主略称
     for the file's project and the date is the contract start (mined from readable siblings).
A bounded brute force over {code, aliases} × {candidate dates} × {formats} covers variants.
"""
from __future__ import annotations

import io
import re
import functools

import msoffcrypto

from src.rag.corpus import FileRef, nfc
from src.rag.extract import glossary

_PW_HINT = re.compile(r"pw-([A-Za-z0-9]+)")
_DATE8 = re.compile(r"(20\d{2})[-/年\.]?\s?(\d{1,2})[-/月\.]?\s?(\d{1,2})")


def is_encrypted(path) -> bool:
    try:
        with open(path, "rb") as fh:
            return msoffcrypto.OfficeFile(fh).is_encrypted()
    except Exception:
        return False


def _try(path, pw: str) -> bytes | None:
    try:
        with open(path, "rb") as fh:
            off = msoffcrypto.OfficeFile(fh)
            off.load_key(password=pw)
            buf = io.BytesIO()
            off.decrypt(buf)
        return buf.getvalue()
    except Exception:
        return None


def _candidate_dates(ref: FileRef) -> list[str]:
    """8-digit YYYYMMDD candidates: filename digits + dates in readable project siblings."""
    dates: list[str] = []

    def add(y, m, d):
        try:
            dates.append(f"{int(y):04d}{int(m):02d}{int(d):02d}")
        except Exception:
            pass

    for m in _DATE8.finditer(ref.name):
        add(*m.groups())
    # scan sibling readable files (contract start usually stated in proposal/minutes/report)
    proj_dir = ref.path.parents[2] if len(ref.path.parents) >= 3 else ref.path.parent
    for sib in proj_dir.rglob("*"):
        if not sib.is_file() or sib.suffix.lower() not in (".docx", ".pdf", ".md", ".txt"):
            continue
        try:
            from src.rag.extract import office, plain
            text = (office.raw_docx_text(sib) if sib.suffix.lower() == ".docx"
                    else plain.read_text_any(sib))
        except Exception:
            continue
        for kw in ("契約開始", "開始日", "契約期間", "着手", "キックオフ", "契約日"):
            for km in re.finditer(kw, text):
                seg = text[km.start(): km.start() + 30]
                for dm in _DATE8.finditer(seg):
                    add(*dm.groups())
    # de-dup, keep order
    return list(dict.fromkeys(dates))


def resolve(ref: FileRef) -> bytes | None:
    """Return decrypted bytes for an encrypted Office file, or None if unresolved."""
    # Scheme A: filename hint
    hint = _PW_HINT.search(ref.name)
    if hint:
        for pw in (hint.group(1), hint.group(1).lower(), hint.group(1).upper()):
            b = _try(ref.path, pw)
            if b:
                return b

    g = glossary.load()
    # 案件略号 candidates from the file's project
    codes: list[str] = []
    if ref.project:
        code = g.company_to_code.get(ref.project)
        if code:
            codes.append(code)
        codes += g.company_aliases.get(ref.project, [])
    codes = list(dict.fromkeys(codes)) or ["INTERNAL"]

    ext = ref.ext
    dates = _candidate_dates(ref) or []
    seps = ["-", "_", ""]
    prefixes = ["DA", ""]
    ext_codes = [ext, ext.upper(), ""]

    for code in codes:
        for date in dates:
            # Scheme A-like: bare lowercase code+date (as the contract docx uses)
            for pw in (f"{code}{date}", f"{code.lower()}{date}"):
                b = _try(ref.path, pw)
                if b:
                    return b
            # Scheme B: rule DA-code-date-ext and variants
            for pre in prefixes:
                for sep in seps:
                    for ec in ext_codes:
                        parts = [p for p in [pre, code, date, ec] if p]
                        b = _try(ref.path, sep.join(parts))
                        if b:
                            return b
    return None


@functools.lru_cache(maxsize=64)
def resolve_cached(path_str: str, project: str, name: str, ext: str) -> bytes | None:
    ref = FileRef(path=__import__("pathlib").Path(path_str), project=project,
                  category="", rel=name, name=name, ext=ext)
    return resolve(ref)
