"""SOT-2530 — canonical-route manifest: build/persist + zero-regression O(1) runtime lookup.

The manifest is a cache of the *same* ``discover`` match+rank, so a consult must return a
byte-identical result to a live walk; a miss / signature drift / disabled flag must fall back to the
live scan. All tests are hermetic (a tmp mini-drive + a tmp manifest path), no real corpus / no LLM.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.rag import corpus
from src.rag.corpus import walk
from src.rag.index import canonical_manifest
from src.rag.tools import canonical_route
from src.rag.tools.canonical_route import KINDS, _discover_live, discover


def _kind(name: str) -> canonical_route.Kind:
    return next(k for k in KINDS if k.name == name)


@pytest.fixture()
def mini_drive(tmp_path: Path) -> Path:
    """A synthetic corpus with two projects, exercising several canonical kinds."""
    root = tmp_path / "drive"
    files = [
        "プロジェクト/AlphaCorp/03.データ/train.xlsx",
        "プロジェクト/AlphaCorp/03.データ/train.csv",
        "プロジェクト/AlphaCorp/04.分析/modeling.py",
        "プロジェクト/AlphaCorp/04.分析/01_eda.ipynb",
        "プロジェクト/BetaCorp/03.データ/train.xlsx",
    ]
    for rel in files:
        p = root / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text("x", encoding="utf-8")
    walk.cache_clear()  # the corpus lru-cache is keyed by dir; ensure a fresh scan of this mini-drive
    return root


@pytest.fixture()
def manifest_path(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    out = tmp_path / "canonical_manifest.json"
    monkeypatch.setattr(canonical_manifest, "default_out_path", lambda: out)
    canonical_manifest.reset_cache()
    return out


def _enable(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("RAG_CANONICAL_MANIFEST", "1")
    canonical_manifest.reset_cache()


def _disable(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("RAG_CANONICAL_MANIFEST", "0")
    canonical_manifest.reset_cache()


# --------------------------------------------------------------------------- build / persist
def test_build_persists_schema_signatures_and_ranked_rels(mini_drive: Path, manifest_path: Path) -> None:
    refs = walk(mini_drive)
    counts = canonical_manifest.build(refs)
    doc = json.loads(manifest_path.read_text(encoding="utf-8"))

    assert doc["schema"] == canonical_manifest.SCHEMA
    assert doc["version"] == canonical_manifest.SCHEMA_VERSION
    assert doc["corpus_signature"] == canonical_manifest.corpus_signature(refs)
    assert doc["kinds_signature"] == canonical_manifest.kinds_signature()

    # Every walked project has an entry; ranked rels match the live discover for that (project, kind).
    assert set(doc["projects"]) == {"AlphaCorp", "BetaCorp"}
    alpha_train = doc["projects"]["AlphaCorp"]["train"]
    assert alpha_train == [r.rel for r in _discover_live("AlphaCorp", _kind("train"), corpus_dir=mini_drive)]
    assert counts["projects"] == 2 and counts["files"] >= 4


def test_write_manifest_is_reproducible(mini_drive: Path, manifest_path: Path) -> None:
    refs = walk(mini_drive)
    canonical_manifest.build(refs)
    first = manifest_path.read_bytes()
    canonical_manifest.build(refs)
    assert manifest_path.read_bytes() == first  # deterministic / reproducible


# --------------------------------------------------------------------------- runtime consult
def test_discover_lookup_is_byte_identical_to_live(mini_drive: Path, manifest_path: Path,
                                                   monkeypatch: pytest.MonkeyPatch) -> None:
    canonical_manifest.build(walk(mini_drive))
    _enable(monkeypatch)
    for pname in ("AlphaCorp", "BetaCorp"):
        for kname in ("train", "code", "notebook", "contract"):
            k = _kind(kname)
            got = discover(pname, k, corpus_dir=mini_drive)
            live = _discover_live(pname, k, corpus_dir=mini_drive)
            assert got == live, f"{pname}/{kname}: manifest lookup must equal live discover"


def test_ext_hint_narrows_lookup_exactly_like_live(mini_drive: Path, manifest_path: Path,
                                                   monkeypatch: pytest.MonkeyPatch) -> None:
    canonical_manifest.build(walk(mini_drive))
    _enable(monkeypatch)
    k = _kind("train")  # AlphaCorp has both train.xlsx and train.csv
    for hint in ("xlsx", "csv", "docx", None):  # in-exts, in-exts, not-in-exts, none
        got = discover("AlphaCorp", k, ext_hint=hint, corpus_dir=mini_drive)
        live = _discover_live("AlphaCorp", k, ext_hint=hint, corpus_dir=mini_drive)
        assert got == live, f"ext_hint={hint}: manifest+filter must equal live"
    # sanity: the xlsx hint really does drop the csv
    xlsx_only = discover("AlphaCorp", k, ext_hint="xlsx", corpus_dir=mini_drive)
    assert xlsx_only and all(r.ext == "xlsx" for r in xlsx_only)


def test_validated_empty_kind_returns_empty_not_fallback(mini_drive: Path, manifest_path: Path,
                                                         monkeypatch: pytest.MonkeyPatch) -> None:
    canonical_manifest.build(walk(mini_drive))
    _enable(monkeypatch)
    # BetaCorp has only a train.xlsx → no contract file. The project IS validated, so the lookup is an
    # authoritative empty ([]) — not a None that would trigger a live re-scan.
    assert canonical_manifest.lookup_rels("BetaCorp", "contract", corpus_dir=mini_drive) == []
    assert discover("BetaCorp", _kind("contract"), corpus_dir=mini_drive) == []


def test_disabled_flag_never_consults_manifest(mini_drive: Path, manifest_path: Path,
                                               monkeypatch: pytest.MonkeyPatch) -> None:
    # Poison the manifest, then prove the disabled path ignores it entirely (byte-identical champion).
    canonical_manifest.write_manifest(
        {"AlphaCorp": {"train": ["プロジェクト/AlphaCorp/03.データ/POISON.xlsx"]}},
        canonical_manifest.corpus_signature(walk(mini_drive)),
        canonical_manifest.kinds_signature())
    _disable(monkeypatch)
    assert canonical_manifest.lookup_rels("AlphaCorp", "train", corpus_dir=mini_drive) is None
    got = discover("AlphaCorp", _kind("train"), corpus_dir=mini_drive)
    assert got == _discover_live("AlphaCorp", _kind("train"), corpus_dir=mini_drive)
    assert all("POISON" not in r.rel for r in got)


def test_corpus_signature_drift_falls_back_to_live(mini_drive: Path, manifest_path: Path,
                                                   monkeypatch: pytest.MonkeyPatch) -> None:
    canonical_manifest.build(walk(mini_drive))
    # Tamper the stored corpus signature → the manifest is stale relative to the live corpus.
    doc = json.loads(manifest_path.read_text(encoding="utf-8"))
    doc["corpus_signature"] = "deadbeef"
    manifest_path.write_text(json.dumps(doc, ensure_ascii=False), encoding="utf-8")
    _enable(monkeypatch)
    assert canonical_manifest.lookup_rels("AlphaCorp", "train", corpus_dir=mini_drive) is None
    assert discover("AlphaCorp", _kind("train"), corpus_dir=mini_drive) == \
        _discover_live("AlphaCorp", _kind("train"), corpus_dir=mini_drive)


def test_kinds_signature_drift_falls_back_to_live(mini_drive: Path, manifest_path: Path,
                                                  monkeypatch: pytest.MonkeyPatch) -> None:
    canonical_manifest.build(walk(mini_drive))
    doc = json.loads(manifest_path.read_text(encoding="utf-8"))
    doc["kinds_signature"] = "deadbeef"  # simulate a KINDS code edit without a rebuild
    manifest_path.write_text(json.dumps(doc, ensure_ascii=False), encoding="utf-8")
    _enable(monkeypatch)
    assert canonical_manifest.lookup_rels("AlphaCorp", "train", corpus_dir=mini_drive) is None


def test_schema_mismatch_load_returns_empty(mini_drive: Path, manifest_path: Path,
                                            monkeypatch: pytest.MonkeyPatch) -> None:
    manifest_path.write_text(json.dumps({"schema": "other", "version": 99}), encoding="utf-8")
    canonical_manifest.reset_cache()
    assert canonical_manifest.load() == {}
    _enable(monkeypatch)
    assert canonical_manifest.lookup_rels("AlphaCorp", "train", corpus_dir=mini_drive) is None


def test_unknown_project_falls_back(mini_drive: Path, manifest_path: Path,
                                    monkeypatch: pytest.MonkeyPatch) -> None:
    canonical_manifest.build(walk(mini_drive))
    _enable(monkeypatch)
    # A project the manifest never saw → None (fallback), never a spurious empty.
    assert canonical_manifest.lookup_rels("GammaCorp", "train", corpus_dir=mini_drive) is None
