"""SOT-2532 / #4b — file_grep routed through the typed evidence index.

Covers the read half added to :mod:`src.rag.index.evidence_index` (``enabled``/``load``/``lookup``)
and the file_grep fast path: index hit ⇒ index-answered, index miss / disabled ⇒ full-scan fallback.
"""
from __future__ import annotations

import unicodedata
from pathlib import Path

from src.rag.corpus import FileRef
from src.rag.index import evidence_index
from src.rag.tools.file_grep import _compile, file_grep


def _ref(tmp_path: Path) -> FileRef:
    nfd_name = unicodedata.normalize("NFD", "顧客一覧.xlsx")
    return FileRef(tmp_path / nfd_name, unicodedata.normalize("NFD", "青葉"), "data",
                   f"プロジェクト/青葉/{nfd_name}", nfd_name, "xlsx")


def _write_index(tmp_path: Path) -> Path:
    text = """[シート: 連絡先]
氏名 | EXT | 備考
佐藤さん | EXT: 1234 | 別契約と明記
【ハイライトされたセル】
  C3(オレンジ): 要確認
"""
    entries = evidence_index.scan_doc(_ref(tmp_path), text)
    out = tmp_path / "evidence_index.jsonl"
    evidence_index.write_index(entries, out)
    evidence_index.reset_cache()
    return out


def test_load_returns_entries_and_ignores_schema_mismatch(tmp_path: Path) -> None:
    out = _write_index(tmp_path)
    assert evidence_index.load(out)  # header validated, entry rows returned

    bad = tmp_path / "bad.jsonl"
    bad.write_text('{"schema": "other", "version": 99}\n{"type": "x"}\n', encoding="utf-8")
    evidence_index.reset_cache()
    assert evidence_index.load(bad) == []

    evidence_index.reset_cache()
    assert evidence_index.load(tmp_path / "missing.jsonl") == []


def test_lookup_returns_hits_when_enabled_and_empty_when_disabled(
        tmp_path: Path, monkeypatch) -> None:
    out = _write_index(tmp_path)
    pattern = _compile("佐藤", regex=False, ignore_case=True)

    monkeypatch.delenv("RAG_EVIDENCE_INDEX", raising=False)
    assert evidence_index.lookup(pattern, path=out) == []  # default OFF

    monkeypatch.setenv("RAG_EVIDENCE_INDEX", "1")
    hits = evidence_index.lookup(pattern, path=out)
    assert hits, "person entry '佐藤さん' should match the query '佐藤'"
    hit = hits[0]
    assert hit["line"] is None
    assert hit["ext"] == "xlsx"
    assert hit["where"].startswith("sheet:連絡先") or "@sheet:連絡先" in hit["where"]
    assert set(hit) == {"file", "project", "ext", "line", "where", "snippet"}


def test_lookup_respects_project_and_ext_filters(tmp_path: Path, monkeypatch) -> None:
    out = _write_index(tmp_path)
    monkeypatch.setenv("RAG_EVIDENCE_INDEX", "1")
    pattern = _compile("佐藤", regex=False, ignore_case=True)

    assert evidence_index.lookup(pattern, path=out, project="青葉")
    assert evidence_index.lookup(pattern, path=out, project="存在しない会社") == []
    assert evidence_index.lookup(pattern, path=out, ext="xlsx")
    assert evidence_index.lookup(pattern, path=out, ext="pdf") == []


def test_file_grep_uses_index_hit_without_scanning(tmp_path: Path, monkeypatch) -> None:
    out = _write_index(tmp_path)
    monkeypatch.setattr(evidence_index, "default_out_path", lambda: out)
    monkeypatch.setenv("RAG_EVIDENCE_INDEX", "1")
    evidence_index.reset_cache()

    empty_corpus = tmp_path / "empty"
    empty_corpus.mkdir()
    res = file_grep("佐藤", corpus_dir=empty_corpus)

    assert res["method"].get("source") == "evidence_index"
    assert res["evidence"]["files_scanned"] == 0  # no full extract happened
    assert res["value"], "index hit should be returned even though the corpus dir is empty"


def test_file_grep_falls_back_to_full_scan_on_index_miss(tmp_path: Path, monkeypatch) -> None:
    out = _write_index(tmp_path)
    monkeypatch.setattr(evidence_index, "default_out_path", lambda: out)
    monkeypatch.setenv("RAG_EVIDENCE_INDEX", "1")
    evidence_index.reset_cache()

    corpus = tmp_path / "corpus"
    corpus.mkdir()
    (corpus / "note.txt").write_text("固有語ZZZが本文に登場する", encoding="utf-8")

    res = file_grep("固有語ZZZ", corpus_dir=corpus)  # not a typed-index entry ⇒ miss
    assert res["method"].get("source") != "evidence_index"
    assert res["evidence"]["files_scanned"] >= 1  # full scan ran
    assert res["value"], "the term lives in the corpus, so the full-scan fallback finds it"


def test_file_grep_candidate_fallback_bounds_the_scan(tmp_path: Path, monkeypatch) -> None:
    """SOT-2562: on an index miss with RAG_FILE_GREP_INDEX_CANDIDATES=1, only candidate files are
    extracted — never the whole corpus — so a single call stays inside the time budget."""
    out = _write_index(tmp_path)
    monkeypatch.setattr(evidence_index, "default_out_path", lambda: out)
    monkeypatch.setenv("RAG_EVIDENCE_INDEX", "1")
    monkeypatch.setenv("RAG_FILE_GREP_INDEX_CANDIDATES", "1")
    monkeypatch.setenv("RAG_FILE_GREP_MAX_CANDIDATES", "2")
    evidence_index.reset_cache()

    corpus = tmp_path / "corpus"
    (corpus / "プロジェクト/青葉").mkdir(parents=True)
    # 顧客一覧.xlsx is the file the typed index knows about (via _write_index); the term "連絡先"
    # is that file's indexed sheet-name/paragraph context but not a direct value hit for "連絡先メモ".
    target = corpus / "プロジェクト/青葉/顧客一覧の連絡先メモ.txt"
    target.write_text("固有語ZZZ はこの連絡先メモにだけ載る", encoding="utf-8")
    for i in range(6):  # decoys that must NOT be extracted once the candidate cap is hit
        (corpus / f"decoy_{i}.txt").write_text("無関係な本文", encoding="utf-8")

    res = file_grep("連絡先メモ", corpus_dir=corpus)
    assert res["method"].get("source") == "evidence_index_candidates"
    assert res["evidence"]["files_scanned"] <= 2  # capped, not a full 7-file scan
    assert res["evidence"]["candidates_considered"] >= 1


def test_file_grep_candidate_fallback_off_still_full_scans(tmp_path: Path, monkeypatch) -> None:
    """With the candidate flag unset, an index miss keeps the original full-scan fallback."""
    out = _write_index(tmp_path)
    monkeypatch.setattr(evidence_index, "default_out_path", lambda: out)
    monkeypatch.setenv("RAG_EVIDENCE_INDEX", "1")
    monkeypatch.delenv("RAG_FILE_GREP_INDEX_CANDIDATES", raising=False)
    evidence_index.reset_cache()

    corpus = tmp_path / "corpus"
    corpus.mkdir()
    (corpus / "note.txt").write_text("固有語ZZZが本文に登場する", encoding="utf-8")
    res = file_grep("固有語ZZZ", corpus_dir=corpus)
    assert res["method"].get("source") not in ("evidence_index", "evidence_index_candidates")
    assert res["evidence"]["files_scanned"] >= 1  # full-scan fallback unchanged


def test_file_grep_disabled_never_consults_index(tmp_path: Path, monkeypatch) -> None:
    out = _write_index(tmp_path)
    monkeypatch.setattr(evidence_index, "default_out_path", lambda: out)
    monkeypatch.delenv("RAG_EVIDENCE_INDEX", raising=False)  # default OFF
    evidence_index.reset_cache()

    corpus = tmp_path / "corpus"
    corpus.mkdir()
    (corpus / "renraku.txt").write_text("佐藤さん EXT 1234", encoding="utf-8")

    res = file_grep("佐藤", corpus_dir=corpus)
    assert res["method"].get("source") != "evidence_index"  # champion path unchanged
    assert res["evidence"]["files_scanned"] >= 1
