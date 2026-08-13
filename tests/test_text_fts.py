"""Offline tests for the full-corpus lexical index (SOT-2657 / text_fts + text_search).

No corpus / no network: a tiny synthetic set of text/markdown/csv files is built into a temp FTS5
index, then searched. Asserts (1) literal/字句 needles hit at rank 1, (2) rare tokens outrank generic
substrings via IDF, (3) build is deterministic (same fingerprint over two builds), (4) the tool is a
no-op with an empty value when the flag is OFF (byte-identical serve invariant), and (5) the tool
returns the contract shape and falls back (empty value) on a miss.
"""
from __future__ import annotations

import os
from pathlib import Path

import pytest

from src.rag.corpus import FileRef
from src.rag.index import text_fts
from src.rag.tools import text_search


def _write(dir_: Path, rel: str, text: str) -> FileRef:
    p = dir_ / rel
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(text, encoding="utf-8")
    ext = rel.rsplit(".", 1)[-1].lower()
    return FileRef(path=p, project="ACME", category="", rel=rel, name=Path(rel).name, ext=ext)


@pytest.fixture()
def corpus(tmp_path: Path) -> list[FileRef]:
    root = tmp_path / "share"
    return [
        _write(root, "契約書_ACME.txt",
               "本契約の内線番号は EXT1234 です。\n担当は山田さん。\n条件: 別契約が必須。"),
        _write(root, "設定_flags.md",
               "# 設定\nRAG_TEXT_FTS フラグを有効化する。\nflag_name = true\n共通の説明文がここに入る。"),
        _write(root, "notes_general.txt",
               "共通の説明文がここに入る。\n内線 5678 の担当は別部署。\n共通の説明文がここに入る。"),
    ]


@pytest.fixture()
def built(tmp_path: Path, corpus: list[FileRef], monkeypatch):
    db = tmp_path / "text_fts.db"
    # report writer targets settings.ARTIFACTS_DIR; redirect it into tmp so tests never touch artifacts.
    monkeypatch.setattr(text_fts, "default_report_path", lambda: tmp_path / "report.json")
    stats = text_fts.build(corpus, out_path=db)
    text_fts.reset_cache()
    return db, stats


def test_tokenize_literals_and_cjk():
    toks = text_fts.tokenize("内線番号は EXT1234 / flag_name")
    assert "ext1234" in toks          # ascii id kept intact (digits included)
    assert "flag_name" in toks        # underscore kept (tokenchars)
    assert "内線番号は" in toks         # full contiguous CJK/kana run (particle attaches)
    assert "内線" in toks and "番号" in toks  # 2-char shingles for substring recall
    # a query without the trailing particle still matches via the shared shingles
    assert {"内線", "線番", "番号"} <= set(text_fts.tokenize("内線番号"))


def test_build_records_and_report(built):
    _db, stats = built
    assert stats["records"] > 0
    assert stats["docs"] == 3
    assert stats["report"]["tokens_distinct"] > 0
    assert stats["report"]["fingerprint"]


def _search(db: Path, monkeypatch, query: str, **kw):
    monkeypatch.setenv("RAG_TEXT_FTS", "1")
    text_fts.reset_cache()
    return text_fts.search(query, path=db, **kw)


def test_ascii_literal_needle_hits_rank1(built, monkeypatch):
    db, _ = built
    hits = _search(db, monkeypatch, "EXT1234")
    assert hits, "literal needle must hit"
    assert hits[0]["doc_id"] == "契約書_ACME.txt"
    assert "EXT1234" in hits[0]["text"]


def test_flag_name_needle(built, monkeypatch):
    db, _ = built
    hits = _search(db, monkeypatch, "RAG_TEXT_FTS")
    assert hits and hits[0]["doc_id"] == "設定_flags.md"


def test_rare_token_outranks_generic_substring(built, monkeypatch):
    db, _ = built
    # "内線番号" appears once (rare) in 契約書; the generic "共通" substring is everywhere. The specific
    # full-run match must dominate — IDF prioritises the rare needle.
    hits = _search(db, monkeypatch, "内線番号")
    assert hits[0]["doc_id"] == "契約書_ACME.txt"
    assert hits[0]["score"] >= hits[-1]["score"]


def test_locator_present(built, monkeypatch):
    db, _ = built
    hits = _search(db, monkeypatch, "EXT1234")
    assert hits[0]["locator"].startswith("line:")


def test_project_and_ext_filters(built, monkeypatch):
    db, _ = built
    assert _search(db, monkeypatch, "共通", project="ACME")
    assert _search(db, monkeypatch, "共通", project="NOPE") == []
    assert _search(db, monkeypatch, "共通", ext="md")
    assert _search(db, monkeypatch, "EXT1234", ext="md") == []  # literal only lives in the .txt


# --------------------------------------------------------------------------- SOT-2702: permissive project filter

def _write_proj(dir_: Path, rel: str, project: str, text: str) -> FileRef:
    p = dir_ / rel
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(text, encoding="utf-8")
    ext = rel.rsplit(".", 1)[-1].lower()
    return FileRef(path=p, project=project, category="", rel=rel, name=Path(rel).name, ext=ext)


_P1 = "医療法人社団 蒼樹会 みなみ野女性医療センター"
_P2 = "医療法人社団 蒼泉会 ひがし丘総合病院"


@pytest.fixture()
def proj_built(tmp_path: Path, monkeypatch):
    """Two projects sharing the substring '蒼' / '医療法人社団'; only _P1 carries 'みなみ野'."""
    root = tmp_path / "share"
    refs = [
        _write_proj(root, "minamino_report.txt", _P1, "監視ダッシュボード構築(別契約)の記載。"),
        _write_proj(root, "higashioka_report.txt", _P2, "別契約に関する記載がここにある。"),
    ]
    db = tmp_path / "text_fts.db"
    monkeypatch.setattr(text_fts, "default_report_path", lambda: tmp_path / "report.json")
    text_fts.build(refs, out_path=db)
    text_fts.reset_cache()
    return db


def _psearch(db: Path, monkeypatch, alias: bool, **kw):
    monkeypatch.setenv("RAG_TEXT_FTS", "1")
    if alias:
        monkeypatch.setenv("RAG_TEXT_FTS_PROJECT_ALIAS", "1")
    else:
        monkeypatch.delenv("RAG_TEXT_FTS_PROJECT_ALIAS", raising=False)
    text_fts.reset_cache()
    return text_fts.search("別契約", path=db, limit=20, **kw)


def test_project_matches_pure():
    aliases = {"mn": _P1, "minamino": _P1}
    assert text_fts._project_matches("みなみ野", _P1, {})          # short name is a substring
    assert text_fts._project_matches(_P1, _P1, {})                # exact
    assert text_fts._project_matches("MN".casefold(), _P1, aliases)  # alias resolves to formal
    assert not text_fts._project_matches("かえで", _P1, {})        # unrelated short name
    assert text_fts._project_matches("", _P1, {})                 # empty filter accepts all


def test_project_alias_off_is_exact(proj_built, monkeypatch):
    # OFF (historical): a short project name matches nothing → the bug the issue documents (idx52).
    assert _psearch(proj_built, monkeypatch, alias=False, project="みなみ野") == []
    # OFF still honours an exact project name.
    assert _psearch(proj_built, monkeypatch, alias=False, project=_P1)


def test_project_alias_short_name_narrows(proj_built, monkeypatch):
    hits = _psearch(proj_built, monkeypatch, alias=True, project="みなみ野")
    assert hits, "short name must resolve to the unique matching project (idx52 recovery)"
    assert {h["project"] for h in hits} == {_P1}
    assert any("監視ダッシュボード" in h["text"] for h in hits)


def test_project_alias_formal_name_still_works(proj_built, monkeypatch):
    hits = _psearch(proj_built, monkeypatch, alias=True, project=_P1)
    assert hits and {h["project"] for h in hits} == {_P1}


def test_project_alias_glossary_code(proj_built, monkeypatch):
    # A glossary code (not a substring of the project) resolves via the alias table.
    monkeypatch.setattr(text_fts, "_project_alias_map", lambda: {"minamino": _P1})
    hits = _psearch(proj_built, monkeypatch, alias=True, project="MINAMINO")
    assert hits and {h["project"] for h in hits} == {_P1}


def test_project_alias_ambiguous_fails_open(proj_built, monkeypatch):
    # '蒼' (or '医療法人社団') substrings BOTH projects → ambiguous → no narrowing (fail-open),
    # so rows from both projects are returned rather than silently mis-dropped.
    hits = _psearch(proj_built, monkeypatch, alias=True, project="医療法人社団")
    assert {h["project"] for h in hits} == {_P1, _P2}


def test_deterministic_build(tmp_path: Path, corpus: list[FileRef], monkeypatch):
    monkeypatch.setattr(text_fts, "default_report_path", lambda: tmp_path / "r.json")
    a = text_fts.build(corpus, out_path=tmp_path / "a.db")
    b = text_fts.build(corpus, out_path=tmp_path / "b.db")
    assert a["report"]["fingerprint"] == b["report"]["fingerprint"]
    assert a["records"] == b["records"]


def test_disabled_returns_empty(built, monkeypatch):
    db, _ = built
    monkeypatch.delenv("RAG_TEXT_FTS", raising=False)
    text_fts.reset_cache()
    assert text_fts.search("EXT1234", path=db) == []


def test_tool_contract_shape(built, monkeypatch):
    db, _ = built
    monkeypatch.setenv("RAG_TEXT_FTS", "1")
    text_fts.reset_cache()
    monkeypatch.setattr(text_fts, "default_out_path", lambda: db)
    res = text_search.text_search("EXT1234")
    assert set(res.keys()) == {"value", "evidence", "method"}
    assert res["method"]["engine"] == "text_search"
    assert res["value"] and res["value"][0]["doc_id"] == "契約書_ACME.txt"


def test_tool_miss_falls_back_empty(built, monkeypatch):
    db, _ = built
    monkeypatch.setenv("RAG_TEXT_FTS", "1")
    text_fts.reset_cache()
    monkeypatch.setattr(text_fts, "default_out_path", lambda: db)
    # a specific absent literal (the file_grep-spin needle case) → no shingle overlap → empty.
    res = text_search.text_search("ZZQXNOTFOUND")
    assert res["value"] == []
    assert res["evidence"]["reason"] == "no_match"


def test_tool_disabled_is_noop(built, monkeypatch):
    monkeypatch.delenv("RAG_TEXT_FTS", raising=False)
    text_fts.reset_cache()
    res = text_search.text_search("EXT1234")
    assert res["value"] == []
    assert res["evidence"]["reason"] == "disabled"
