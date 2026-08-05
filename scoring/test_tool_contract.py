"""SOT-2466: unified tool contract {value, evidence, method} + corpus_profile adaptation layer.

Covers the two acceptance criteria:
  ① every tool returns the common contract (contract-compliance across all wrappers);
  ② the generic layer bundles NO raw secret (a discovered password never appears in a returned
     contract nor in the tools' source), while the *runtime* profile may cache it.
Plus the profile read-write roundtrip.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.rag.corpus import walk
from src.rag.extract import passwords as _passwords
from src.rag.extract import vision as _vision
from src.rag.tools import (
    ContractError,
    CorpusProfile,
    ToolResult,
    caption_figure,
    company_of,
    compute_run,
    decrypt,
    ensure_contract,
    expand_terms,
    extract_office,
    find_files,
    is_contract,
    make,
)
from src.rag.tools import contract as contract_mod
from src.rag.tools import profile as profile_mod


# --------------------------------------------------------------------------- contract module
def test_is_contract_accepts_valid_and_rejects_malformed():
    good = {"value": 1, "evidence": {}, "method": {"engine": "x"}}
    assert is_contract(good)
    assert is_contract(ToolResult(value=None))          # dataclass form, empty maps
    assert not is_contract({"value": 1, "evidence": {}})               # missing method
    assert not is_contract({"value": 1, "evidence": {}, "method": {}, "x": 1})  # extra key
    assert not is_contract({"value": 1, "evidence": [], "method": {}})  # evidence not a map
    assert not is_contract("nope")


def test_toolresult_roundtrip_and_make():
    tr = ToolResult(value=[1, 2], evidence={"file": "a"}, method={"engine": "pandas"})
    d = tr.to_dict()
    assert is_contract(d)
    assert ToolResult.from_dict(d) == tr
    built = make(42, engine="glossary", evidence={"source": "x"}, op="lookup")
    assert is_contract(built) and built["method"] == {"engine": "glossary", "op": "lookup"}


def test_ensure_contract_raises_on_bad():
    with pytest.raises(ContractError):
        ensure_contract({"value": 1})
    assert ensure_contract(ToolResult(value=1))["value"] == 1


# --------------------------------------------------------------------------- adaptation layer
def test_profile_read_write_roundtrip(tmp_path: Path):
    p = tmp_path / "corpus_profile.json"
    prof = CorpusProfile(path=p)
    prof.set_password("proj/契約書.docx", "SECRET-123")
    prof.add_alias("かえで総合病院", "kaede")
    prof.add_alias("かえで総合病院", "kaede")   # dedup
    prof.set_format("schedulexlsx/sheet", "train")
    written = prof.save()
    assert written == p and p.exists()

    back = CorpusProfile.load(p)
    assert back.get_password("proj/契約書.docx") == "SECRET-123"
    assert back.get_aliases("かえで総合病院") == ["kaede"]
    assert back.get_format("schedulexlsx/sheet") == "train"
    assert back.to_dict() == prof.to_dict()
    assert back.version == profile_mod.SCHEMA_VERSION


def test_profile_load_missing_is_fail_open(tmp_path: Path):
    empty = CorpusProfile.load(tmp_path / "nope.json")
    assert empty.to_dict() == {"version": profile_mod.SCHEMA_VERSION,
                               "passwords": {}, "aliases": {}, "formats": {}}


def test_default_profile_path_is_gitignored_artifacts():
    # criterion ②: the runtime secret cache lives under the gitignored artifacts dir.
    assert "artifacts" in profile_mod.DEFAULT_PROFILE_PATH.parts


# --------------------------------------------------------------------------- corpus / glossary tools
def test_find_files_returns_contract():
    out = find_files(ext="csv", limit=5)
    assert is_contract(out)
    assert out["method"]["engine"] == "corpus"
    assert isinstance(out["value"], list)
    assert out["evidence"]["matched"] >= len(out["value"])


def test_glossary_tools_return_contract():
    for out in (expand_terms("かえでの契約について"), company_of("かえで総合病院の契約")):
        assert is_contract(out)
        assert out["method"]["engine"] == "glossary"


# --------------------------------------------------------------------------- office / compute tools
def _first_encrypted():
    for r in walk():
        if r.ext in ("docx", "xlsx", "xlsm", "pptx") and _passwords.is_encrypted(r.path):
            return r
    return None


def _first_docx():
    for r in walk():
        if r.ext == "docx" and not _passwords.is_encrypted(r.path):
            return r
    return None


def test_extract_office_returns_contract():
    ref = _first_docx() or _first_encrypted()
    if ref is None:
        pytest.skip("no Office file in corpus")
    out = extract_office(ref)
    assert is_contract(out)
    assert out["method"]["engine"] in ("docx", "xlsx", "pptx")
    assert isinstance(out["value"], str)


def test_extract_office_rejects_non_office():
    csv = next((r for r in walk() if r.ext == "csv"), None)
    if csv is None:
        pytest.skip("no csv in corpus")
    with pytest.raises(ContractError):
        extract_office(csv)


def test_compute_run_returns_contract():
    csv = next((r for r in walk() if r.ext == "csv" and r.name == "train.csv"), None)
    if csv is None:
        pytest.skip("no train.csv in corpus")
    out = compute_run(csv.rel, "len(df)")
    assert is_contract(out) and out["method"]["engine"] == "pandas"


def test_caption_figure_returns_contract(monkeypatch):
    # keep offline: stub the vision model call
    monkeypatch.setattr(_vision, "caption_png", lambda ref: "[図] スタブ説明")
    png = next((r for r in walk() if r.ext == "png"), None)
    ref = png or next((r for r in walk() if r.ext == "csv"), None)
    if ref is None:
        pytest.skip("no corpus file")
    out = caption_figure(ref)
    assert is_contract(out)
    assert out["method"]["engine"] == "vision"
    assert out["value"] == "[図] スタブ説明"


# --------------------------------------------------------------------------- passwords tool + no-secret invariant
def test_decrypt_returns_contract_without_leaking_password():
    ref = _first_encrypted()
    if ref is None:
        pytest.skip("no encrypted Office file in corpus")
    pw = _passwords.resolve_password(ref)
    assert pw, "expected the encrypted file to be resolvable"

    out = decrypt(ref)
    assert is_contract(out)
    assert out["value"]["decrypted"] is True
    assert out["value"]["bytes_len"] > 0
    # ② the tool reports provenance/how, never the answer secret: the password must not appear
    # in value or method (evidence.file legitimately carries the source path — even when, as here,
    # the filename itself embeds a pw-token — that is provenance, not a bundled secret).
    assert pw not in json.dumps(out["value"], ensure_ascii=False)
    assert pw not in json.dumps(out["method"], ensure_ascii=False)
    assert "password" not in out["value"]


def test_decrypt_caches_password_into_profile(tmp_path: Path):
    ref = _first_encrypted()
    if ref is None:
        pytest.skip("no encrypted Office file in corpus")
    prof = CorpusProfile(path=tmp_path / "corpus_profile.json")
    first = decrypt(ref, profile=prof)
    assert first["method"]["scheme"] == "derivation"
    assert prof.get_password(ref.rel)   # discovered secret cached in the runtime profile
    # a second call reuses the cached password instead of re-deriving
    second = decrypt(ref, profile=prof)
    assert second["method"]["scheme"] == "cache"
    assert second["value"]["decrypted"] is True


def test_extract_office_decrypts_encrypted_via_profile(tmp_path: Path):
    ref = _first_encrypted()
    if ref is None or ref.ext not in ("docx", "xlsx", "xlsm", "pptx"):
        pytest.skip("no encrypted Office file in corpus")
    prof = CorpusProfile(path=tmp_path / "corpus_profile.json")
    out = extract_office(ref, profile=prof)
    assert is_contract(out)
    assert out["evidence"]["encrypted"] is True
    assert prof.get_password(ref.rel)   # decryption populated the profile cache


def test_no_raw_password_hardcoded_in_tools_source():
    """② the generic layer must not bundle the raw secret — assert a discovered password string
    does not appear literally in any tools/*.py source file."""
    ref = _first_encrypted()
    if ref is None:
        pytest.skip("no encrypted Office file in corpus")
    pw = _passwords.resolve_password(ref)
    assert pw
    tools_dir = Path(contract_mod.__file__).parent
    for src in tools_dir.glob("*.py"):
        assert pw not in src.read_text(encoding="utf-8"), f"raw password leaked into {src.name}"
