"""SOT-2529 — pre-index decrypt cache: resolve encrypted-Office passwords at build time.

The precompute walks the corpus once and folds each resolved password into the persisted
``corpus_profile.json`` (SOT-2528 cache), so a run-time ``decrypt``/``extract_office`` finds the
password in the profile and skips the brute force. These tests are hermetic: a real encrypted xlsx
is built with ``msoffcrypto``/``openpyxl`` in tmp, and the profile is written to a tmp path — no
corpus walk, no LLM. The security invariant (secrets only in the gitignored runtime profile) is
exercised via the additive/existing-wins merge and the fail-open unresolved path.
"""
from __future__ import annotations

import io
import json
from pathlib import Path

import msoffcrypto
import openpyxl
import pytest

from src.rag.corpus import FileRef
from src.rag.extract import passwords as _passwords
from src.rag.index import precomputed_passwords
from src.rag.tools.profile import CorpusProfile


def _make_encrypted_xlsx(path: Path, password: str) -> None:
    """Write a real password-protected xlsx at ``path`` (encrypt-in-place via msoffcrypto)."""
    wb = openpyxl.Workbook()
    ws = wb.active
    ws.append(["月", "売上"])
    ws.append(["1月", 100])
    plain = io.BytesIO()
    wb.save(plain)
    plain.seek(0)
    enc = io.BytesIO()
    off = msoffcrypto.OfficeFile(plain)
    off.load_key(password=password)
    off.encrypt(password, enc)
    path.write_bytes(enc.getvalue())


def _make_plain_xlsx(path: Path) -> None:
    wb = openpyxl.Workbook()
    wb.active.append(["月", "売上"])
    wb.save(path)


def _ref(path: Path, project: str = "かえで") -> FileRef:
    # filename carries a pw-<token> hint so Scheme A resolves without a glossary/corpus.
    return FileRef(path=path, project=project, category="data", rel=path.name,
                   name=path.name, ext="xlsx")


# --------------------------------------------------------------------------- build / persist
def test_build_resolves_and_persists_password(tmp_path: Path) -> None:
    p = tmp_path / "train_pw-secret123.xlsx"
    _make_encrypted_xlsx(p, "secret123")
    prof_path = tmp_path / "corpus_profile.json"

    counts = precomputed_passwords.build([_ref(p)], profile_path=prof_path)

    assert counts["encrypted"] == 1 and counts["resolved"] == 1 and counts["unresolved"] == 0
    doc = json.loads(prof_path.read_text(encoding="utf-8"))
    assert doc["passwords"][p.name] == "secret123"
    # And the persisted password actually decrypts the file at run time (scheme=cache would hit).
    assert CorpusProfile.load(prof_path).get_password(p.name) == "secret123"


def test_build_skips_plain_and_non_office_files(tmp_path: Path) -> None:
    plain = tmp_path / "open.xlsx"
    _make_plain_xlsx(plain)
    txt = tmp_path / "notes.txt"
    txt.write_text("hello", encoding="utf-8")
    prof_path = tmp_path / "corpus_profile.json"

    counts = precomputed_passwords.build(
        [_ref(plain), FileRef(txt, "p", "", txt.name, txt.name, "txt")],
        profile_path=prof_path)

    assert counts["encrypted"] == 0 and counts["resolved"] == 0
    assert json.loads(prof_path.read_text(encoding="utf-8"))["passwords"] == {}


def test_build_is_additive_existing_password_wins(tmp_path: Path) -> None:
    """A password already in the profile is kept and not re-derived (existing wins / secret-safe)."""
    p = tmp_path / "train_pw-secret123.xlsx"
    _make_encrypted_xlsx(p, "secret123")
    prof_path = tmp_path / "corpus_profile.json"
    CorpusProfile(path=prof_path).save()  # seed empty file
    seeded = CorpusProfile.load(prof_path)
    seeded.set_password(p.name, "USER-SET")
    seeded.save()

    # resolve_password must NOT be called for an already-cached file → poison it to prove skip.
    def _boom(_ref):  # pragma: no cover - only fires on regression
        raise AssertionError("resolve_password should be skipped for a cached file")

    import unittest.mock as mock
    with mock.patch.object(_passwords, "resolve_password", _boom):
        counts = precomputed_passwords.build([_ref(p)], profile_path=prof_path)

    assert counts["encrypted"] == 1 and counts["cached"] == 1 and counts["resolved"] == 0
    assert CorpusProfile.load(prof_path).get_password(p.name) == "USER-SET"


def test_build_counts_unresolved_when_no_candidate_matches(tmp_path: Path) -> None:
    """An encrypted file whose password can't be derived is counted, never guessed/hard-coded."""
    p = tmp_path / "mystery.xlsx"  # no pw- hint, no project/glossary/corpus → Scheme A/B both miss
    _make_encrypted_xlsx(p, "totally-unguessable-xyz")
    prof_path = tmp_path / "corpus_profile.json"

    counts = precomputed_passwords.build(
        [FileRef(p, "", "", p.name, p.name, "xlsx")], profile_path=prof_path)

    assert counts["encrypted"] == 1 and counts["resolved"] == 0 and counts["unresolved"] == 1
    assert CorpusProfile.load(prof_path).get_password(p.name) is None


def test_build_is_guarded_against_per_file_errors(tmp_path: Path, monkeypatch) -> None:
    """A single unreadable file raises inside the loop but never aborts the whole precompute."""
    good = tmp_path / "train_pw-secret123.xlsx"
    _make_encrypted_xlsx(good, "secret123")
    bad = tmp_path / "broken_pw-x.xlsx"
    bad.write_bytes(b"not a real office file")
    prof_path = tmp_path / "corpus_profile.json"

    real_is_encrypted = _passwords.is_encrypted

    def flaky(path):
        if Path(path).name == bad.name:
            raise RuntimeError("boom")
        return real_is_encrypted(path)

    monkeypatch.setattr(_passwords, "is_encrypted", flaky)
    counts = precomputed_passwords.build([_ref(bad), _ref(good)], profile_path=prof_path)

    # The bad file was swallowed; the good file still resolved and persisted.
    assert counts["resolved"] == 1
    assert CorpusProfile.load(prof_path).get_password(good.name) == "secret123"


def test_runtime_decrypt_hits_cache_and_skips_bruteforce(tmp_path: Path) -> None:
    """受け入れ条件②: after build(), the run-time decrypt tool resolves from the profile cache and
    never brute-forces (resolve_password poisoned would raise if the fallback path were taken)."""
    import unittest.mock as mock

    from src.rag.tools import extract_tools

    p = tmp_path / "train_pw-secret123.xlsx"
    _make_encrypted_xlsx(p, "secret123")
    prof_path = tmp_path / "corpus_profile.json"

    precomputed_passwords.build([_ref(p)], profile_path=prof_path)
    prof = CorpusProfile.load(prof_path)  # what a shared run (RAG_SHARE_CORPUS_PROFILE=1) loads

    def _boom(_ref):  # pragma: no cover - only fires on regression
        raise AssertionError("brute force must be skipped when the profile is pre-warmed")

    with mock.patch.object(extract_tools._passwords, "resolve_password", _boom):
        res = extract_tools.decrypt(_ref(p), profile=prof)

    assert res["value"]["decrypted"] is True
    assert res["method"]["scheme"] == "cache"
