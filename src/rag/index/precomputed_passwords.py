"""Pre-resolve encrypted-Office passwords at index-build time (事前処理 #2 / SOT-2529).

``passwords.resolve()`` Scheme B brute-forces over sibling documents × mined contract dates ×
format variants (``src/rag/extract/passwords.py``). Paid *per question* at run time this is a
dominant 反復上限 / timeout cost — 探索の 73% は発見系で、同じ暗号化ファイルを毎問やり直している。

This module walks the corpus **once at build time**, resolves the password for every encrypted
Office file, and folds the results into the persisted ``corpus_profile.json`` — the same run-wide
discovery cache SOT-2528 (事前処理 #1) shares across a run. At run time with
``RAG_SHARE_CORPUS_PROFILE=1`` the profile is loaded once and the very first ``decrypt`` /
``extract_office`` finds the password already in it (``scheme="cache"``), skipping the brute force
entirely; un-precomputed files still fall back to the live derivation path unchanged.

Security invariant (受け入れ条件③ / SOT-2528 contract)
------------------------------------------------------
A resolved password is a raw secret, so it is written **only** to the gitignored runtime
``corpus_profile.json`` (``artifacts/`` — ``.gitignore`` ignores ``corpus_profile.json`` and
``**/corpus_profile.json``); never to source, never committed. Merge is *additive, existing keys
win*, so a live discovery from a prior run is never clobbered.
"""
from __future__ import annotations

from pathlib import Path

from src.rag import corpus
from src.rag.extract import passwords as _passwords
from src.rag.tools.profile import CorpusProfile

# Office containers msoffcrypto can hold an encrypted payload for.
_OFFICE_EXTS = ("docx", "xlsx", "xlsm", "pptx")


def build(refs=None, *, profile_path: str | Path | None = None) -> dict:
    """Pre-resolve every encrypted Office file's password into the persisted profile.

    Loads the on-disk ``corpus_profile.json`` (empty when absent — fail-open), resolves the
    password for each encrypted Office ref that is not already cached, writes the new entries back,
    and atomically saves. Additive: an existing profile password is kept, not re-derived. Per-file
    failures are swallowed so one unreadable file can never abort the precompute.

    Returns counts ``{"encrypted", "resolved", "unresolved", "cached", "path"}`` where ``resolved``
    counts passwords newly derived this build, ``cached`` counts files already present in the
    profile, and ``unresolved`` counts encrypted files no candidate password decrypted.
    """
    if refs is None:
        refs = corpus.walk()

    prof = CorpusProfile.load(profile_path)
    encrypted = resolved = unresolved = cached = 0

    for ref in refs:
        if ref.ext not in _OFFICE_EXTS:
            continue
        try:
            if not _passwords.is_encrypted(ref.path):
                continue
            encrypted += 1
            # Additive: keep a password already discovered by a prior run/build (existing wins).
            if prof.get_password(ref.rel) is not None:
                cached += 1
                continue
            pw = _passwords.resolve_password(ref)
            if pw is not None:
                prof.set_password(ref.rel, pw)
                resolved += 1
            else:
                unresolved += 1
        except Exception:
            # Never let a single unreadable/odd file abort the whole precompute pass.
            continue

    p = prof.save(profile_path)
    return {
        "encrypted": encrypted,
        "resolved": resolved,
        "unresolved": unresolved,
        "cached": cached,
        "path": str(p),
    }
