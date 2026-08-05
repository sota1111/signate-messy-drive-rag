"""Thin contract wrappers turning the existing extract modules into uniform, agent-callable tools.

Each function here wraps one existing capability (``corpus`` walking, ``passwords`` decryption,
``glossary`` term/company lookup, ``office`` Office extraction, ``vision`` chart captioning) and
returns the unified :mod:`src.rag.tools.contract` shape ``{value, evidence, method}`` — so the
generation layer can call any of them the same way. The underlying extract modules are unchanged;
these are deliberately *thin* adapters plus optional read/write against the adaptation layer
(:mod:`src.rag.tools.profile`), which caches self-discovered facts (passwords / aliases / formats).

No raw secret is ever placed in a returned ``method``/``evidence`` (受け入れ条件②): the password
*scheme name* and *whether* decryption succeeded are reported, never the password itself. Discovered
secrets go only into the runtime, gitignored ``corpus_profile.json`` via ``profile``.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

from src.rag.corpus import FileRef, nfc, walk
from src.rag.extract import glossary as _glossary
from src.rag.extract import office as _office
from src.rag.extract import passwords as _passwords
from src.rag.extract import vision as _vision
from src.rag.tools import contract
from src.rag.tools.contract import ContractError
from src.rag.tools.profile import CorpusProfile

_OFFICE_ENGINE = {"docx": "docx", "xlsx": "xlsx", "xlsm": "xlsx", "pptx": "pptx"}


# --------------------------------------------------------------------------- ref resolution
def _ref_from_path(path: Path) -> FileRef:
    p = Path(path)
    name = nfc(p.name)
    return FileRef(path=p, project="", category="", rel=name, name=name,
                   ext=p.suffix.lower().lstrip("."))


def resolve_ref(file: str | Path | FileRef) -> FileRef:
    """Resolve a FileRef / path / corpus-relative string (NFC-aware) to a :class:`FileRef`.

    Matches the walked corpus by exact rel, rel-suffix, then filename; falls back to a bare
    filesystem path. Raises :class:`ContractError` when nothing (or something ambiguous) matches.
    """
    if isinstance(file, FileRef):
        return file
    if isinstance(file, Path) or (isinstance(file, str) and Path(file).is_absolute()):
        p = Path(file)
        want = nfc(str(p))
        for r in walk():
            if str(r.path) == str(p) or nfc(str(r.path)) == want:
                return r
        if p.exists():
            return _ref_from_path(p)
        raise ContractError(f"file not found: {file}")

    key = nfc(str(file))
    refs = walk()
    for pool in ([r for r in refs if nfc(r.rel) == key],
                 [r for r in refs if nfc(r.rel).endswith(key)],
                 [r for r in refs if nfc(r.name) == key]):
        if len(pool) == 1:
            return pool[0]
        if len(pool) > 1:
            raise ContractError(
                f"ambiguous file reference {file!r} → {len(pool)} matches: "
                + ", ".join(sorted(r.rel for r in pool)[:5]))
    p = Path(file)
    if p.exists():
        return _ref_from_path(p)
    raise ContractError(f"no corpus file matches {file!r}")


# --------------------------------------------------------------------------- corpus tool
def find_files(query: str | None = None, *, ext: str | None = None,
               project: str | None = None, limit: int = 50) -> dict[str, Any]:
    """List corpus files (optionally filtered by name/path ``query``, ``ext``, ``project``).

    ``value`` is a list of ``{rel, project, category, ext}`` records; ``evidence`` counts the
    matches out of the full corpus. Pure classification over :func:`src.rag.corpus.walk`.
    """
    q = nfc(query) if query else None
    ext_l = ext.lower().lstrip(".") if ext else None
    proj = nfc(project) if project else None
    hits: list[FileRef] = []
    for r in walk():
        if ext_l and r.ext != ext_l:
            continue
        if proj and nfc(r.project) != proj:
            continue
        if q and q not in nfc(r.rel):
            continue
        hits.append(r)
    total = len(hits)
    records = [{"rel": r.rel, "project": r.project, "category": r.category, "ext": r.ext}
               for r in hits[:limit]]
    return contract.make(
        records,
        engine="corpus",
        evidence={"corpus_dir": "data/share_drive", "matched": total, "returned": len(records)},
        query=query, filters={"ext": ext_l, "project": proj}, truncated=total > len(records),
    )


# --------------------------------------------------------------------------- glossary tool
def expand_terms(text: str) -> dict[str, Any]:
    """Expand 社内用語/略称 found in ``text`` to their正式名称/company (glossary lookup)."""
    g = _glossary.load()
    extra = g.expand_terms(text)
    return contract.make(
        extra, engine="glossary",
        evidence={"source": "社内用語集", "input_len": len(text), "n_expansions": len(extra)},
        op="expand_terms")


def company_of(text: str, *, profile: CorpusProfile | None = None) -> dict[str, Any]:
    """Best-effort project/company a question refers to (by code/alias/name substring).

    When it resolves and a ``profile`` is given, the matched alias is cached under the company so
    later calls (and the password derivation) can reuse the self-discovered alias.
    """
    g = _glossary.load()
    company = g.company_of(text)
    matched_alias = None
    if company:
        t = nfc(text)
        for a in sorted(g.company_aliases.get(company, []), key=len, reverse=True):
            if a and a in t:
                matched_alias = a
                break
        if profile is not None and matched_alias:
            profile.add_alias(company, matched_alias)
    return contract.make(
        company, engine="glossary",
        evidence={"source": "社内用語集", "matched_alias": matched_alias},
        op="company_of")


# --------------------------------------------------------------------------- passwords tool
def decrypt(file: str | Path | FileRef, *, profile: CorpusProfile | None = None) -> dict[str, Any]:
    """Decrypt an encrypted Office file, returning *provenance only* — never the password.

    ``value`` is ``{"decrypted": bool, "bytes_len": int|None}``. If a ``profile`` is supplied, a
    previously discovered password is tried first; a freshly brute-forced one is cached back into
    the profile (in memory — the caller persists via ``profile.save()``). The returned ``method``
    reports the *scheme* (``cache`` / ``derivation`` / ``none``) but no secret material.
    """
    ref = resolve_ref(file)
    encrypted = _passwords.is_encrypted(ref.path)
    if not encrypted:
        return contract.make(
            {"decrypted": False, "bytes_len": None}, engine="msoffcrypto",
            evidence={"file": ref.rel or str(ref.path), "encrypted": False}, scheme="none")

    scheme = "none"
    data: bytes | None = None
    cached = profile.get_password(ref.rel) if profile is not None else None
    if cached is not None:
        data = _passwords._try(ref.path, cached)
        if data is not None:
            scheme = "cache"
    if data is None:
        pw = _passwords.resolve_password(ref)
        if pw is not None:
            data = _passwords._try(ref.path, pw)
            if data is not None:
                scheme = "derivation"
                if profile is not None:
                    profile.set_password(ref.rel, pw)

    return contract.make(
        {"decrypted": data is not None, "bytes_len": len(data) if data is not None else None},
        engine="msoffcrypto",
        evidence={"file": ref.rel or str(ref.path), "encrypted": True},
        scheme=scheme)


# --------------------------------------------------------------------------- office tool
def extract_office(file: str | Path | FileRef, *,
                   profile: CorpusProfile | None = None) -> dict[str, Any]:
    """Extract formatting-preserving text from a docx/xlsx/pptx (transparently decrypting first).

    ``value`` is the extracted text; ``evidence`` records the file, ext and whether it was
    encrypted; ``method.engine`` names the Office reader used. Decryption reuses the passwords
    tool (and the profile cache) so encrypted files are handled without embedding any secret here.
    """
    ref = resolve_ref(file)
    engine = _OFFICE_ENGINE.get(ref.ext)
    if engine is None:
        raise ContractError(f"not an Office file: .{ref.ext} (expected docx/xlsx/xlsm/pptx)")

    encrypted = _passwords.is_encrypted(ref.path)
    data: bytes | None = None
    if encrypted:
        cached = profile.get_password(ref.rel) if profile is not None else None
        if cached is not None:
            data = _passwords._try(ref.path, cached)
        if data is None:
            pw = _passwords.resolve_password(ref)
            if pw is not None:
                data = _passwords._try(ref.path, pw)
                if data is not None and profile is not None:
                    profile.set_password(ref.rel, pw)
        if data is None:
            raise ContractError(f"could not decrypt {ref.rel or ref.path}")

    if engine == "docx":
        text = _office.extract_docx(ref, data)
    elif engine == "xlsx":
        text = _office.extract_xlsx(ref, data)
    else:
        text = _office.extract_pptx(ref, data)

    return contract.make(
        text, engine=engine,
        evidence={"file": ref.rel or str(ref.path), "ext": ref.ext,
                  "encrypted": encrypted, "chars": len(text)},
        decrypted=bool(data) if encrypted else None)


# --------------------------------------------------------------------------- vision tool
def caption_figure(file: str | Path | FileRef) -> dict[str, Any]:
    """Caption a chart/figure PNG via the vision model, as a contract result.

    ``value`` is the caption string; ``method`` records the vision model. (Network call — callers
    that must stay offline should stub :func:`src.rag.extract.vision.caption_png`.)
    """
    ref = resolve_ref(file)
    caption = _vision.caption_png(ref)
    from src.rag import llm
    return contract.make(
        caption, engine="vision",
        evidence={"file": ref.rel or str(ref.path)},
        model=getattr(llm.settings, "VISION_MODEL", None))
