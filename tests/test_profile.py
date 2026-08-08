"""SOT-2528 — unit tests for the run-wide sharing / persistence of :class:`CorpusProfile`.

These exercise the discovery-cache semantics only (passwords/aliases/formats are corpus facts, not
per-question answers), the opt-in ``RAG_SHARE_CORPUS_PROFILE`` flag, the additive ``merge`` helper, and
that concurrent writers never corrupt a shared instance. All offline; no Vertex/LLM call.
"""
from __future__ import annotations

import threading

import pytest

from src.rag.tools import profile as _profile
from src.rag.tools.profile import CorpusProfile


# --------------------------------------------------------------------------- opt-in flag
@pytest.mark.parametrize("val,expected", [
    ("1", True), ("true", True), ("YES", True), ("On", True),
    ("0", False), ("false", False), ("", False), ("nope", False),
])
def test_share_enabled_reads_env(monkeypatch, val, expected):
    monkeypatch.setenv("RAG_SHARE_CORPUS_PROFILE", val)
    assert _profile.share_enabled() is expected


def test_share_enabled_defaults_off(monkeypatch):
    monkeypatch.delenv("RAG_SHARE_CORPUS_PROFILE", raising=False)
    assert _profile.share_enabled() is False


# --------------------------------------------------------------------------- persistence roundtrip
def test_persist_reuses_discovered_password_across_load(tmp_path):
    """A password discovered in one run is reused (not re-derived) after load in the next."""
    p = tmp_path / "corpus_profile.json"
    prof = CorpusProfile(path=p)
    assert prof.get_password("かえで/train.xlsx") is None
    prof.set_password("かえで/train.xlsx", "secret123")
    prof.save()

    reloaded = CorpusProfile.load(p)
    assert reloaded.get_password("かえで/train.xlsx") == "secret123"


def test_load_missing_file_is_empty_fail_open(tmp_path):
    prof = CorpusProfile.load(tmp_path / "does_not_exist.json")
    assert prof.passwords == {} and prof.aliases == {} and prof.formats == {}


# --------------------------------------------------------------------------- merge (additive)
def test_merge_is_additive_and_keeps_existing_values():
    a = CorpusProfile(passwords={"f": "keep"}, aliases={"社": ["A"]}, formats={"k": 1})
    b = CorpusProfile(passwords={"f": "override-ignored", "g": "new"},
                      aliases={"社": ["A", "B"], "他社": ["C"]}, formats={"k": 99, "k2": 2})
    a.merge(b)
    assert a.get_password("f") == "keep"          # existing key wins
    assert a.get_password("g") == "new"           # new key folded in
    assert a.get_aliases("社") == ["A", "B"]       # new alias appended, no dup
    assert a.get_aliases("他社") == ["C"]
    assert a.get_format("k") == 1 and a.get_format("k2") == 2


# --------------------------------------------------------------------------- thread safety
def test_concurrent_writers_do_not_corrupt_shared_instance():
    """A single shared profile written by many threads keeps every discovery (no lost/corrupt dict)."""
    prof = CorpusProfile()
    n = 200

    def worker(i: int):
        prof.set_password(f"file{i}.docx", f"pw{i}")
        prof.add_alias("社", f"alias{i}")
        prof.set_format(f"key{i}", i)

    threads = [threading.Thread(target=worker, args=(i,)) for i in range(n)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert len(prof.passwords) == n
    assert all(prof.get_password(f"file{i}.docx") == f"pw{i}" for i in range(n))
    assert len(prof.formats) == n
    assert sorted(prof.get_aliases("社")) == sorted(f"alias{i}" for i in range(n))


def test_lock_excluded_from_serialization():
    """The internal lock never leaks into the on-disk schema."""
    assert set(CorpusProfile().to_dict()) == {"version", "passwords", "aliases", "formats"}
