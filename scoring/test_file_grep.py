"""SOT-2465: NFC-aware cross-corpus full-text / cell grep tool (``file_grep``).

Covers the two acceptance criteria:
  ① a known word's 所在 is correctly enumerated (right file + a usable location);
  ② every result is the common contract ``{value, evidence, method}``.

Content-search behaviour is pinned against a tiny deterministic temp corpus (fast, no heavy
Office/PDF deps); a guarded smoke test exercises the real share drive when present.
"""
from __future__ import annotations

import unicodedata
from pathlib import Path

import pytest

from src.rag.corpus import walk
from src.rag.tools import FileGrepError, file_grep, is_contract


# --------------------------------------------------------------------------- temp corpus fixture
def _write(p: Path, text: str) -> None:
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(text, encoding="utf-8")


@pytest.fixture()
def mini_corpus(tmp_path: Path) -> Path:
    """A tiny share-drive-shaped corpus with an NFD-normalized Japanese folder/file name."""
    root = tmp_path / "share_drive"
    # NFD-normalize the whole tree (mimics the macOS-origin real drive).
    nfd = lambda s: unicodedata.normalize("NFD", s)  # noqa: E731
    _write(root / nfd("プロジェクト/かえで総合病院/01.契約") / "契約メモ.md",
           "契約条件の概要\n特記事項: 更新条項は自動更新とする\nおわり\n")
    _write(root / nfd("プロジェクト/かえで総合病院/03.データ") / "train.csv",
           "id,name,score\n1,あおば,42\n2,かえで,777\n")
    _write(root / "社内管理" / "メモ.txt",
           "社内用語: RAG は検索拡張生成\n担当: 山田\n")
    return root


# --------------------------------------------------------------------------- contract (②)
def test_returns_contract(mini_corpus: Path):
    out = file_grep("契約", corpus_dir=mini_corpus)
    assert is_contract(out)
    assert out["method"]["engine"] == "file_grep"
    assert isinstance(out["value"], list)
    assert out["evidence"]["files_scanned"] >= 3


# --------------------------------------------------------------------------- known word 所在 (①)
def test_known_word_located_with_file_and_line(mini_corpus: Path):
    out = file_grep("自動更新", corpus_dir=mini_corpus)
    hits = out["value"]
    assert hits, "known word must be enumerated"
    hit = next(h for h in hits if h["file"].endswith("契約メモ.md"))
    assert hit["project"] == "かえで総合病院"
    assert hit["line"] == 2                      # 1-based line within the extracted text
    assert hit["where"] == "line:2"
    assert "自動更新" in hit["snippet"]
    assert out["evidence"]["files_matched"] == 1


def test_word_in_csv_cell_located(mini_corpus: Path):
    # a value living in a spreadsheet cell is found via the tabular extractor
    out = file_grep("777", corpus_dir=mini_corpus)
    files = {h["file"] for h in out["value"]}
    assert any(f.endswith("train.csv") for f in files)


# --------------------------------------------------------------------------- NFC filename match
def test_filename_matched_via_nfc(mini_corpus: Path):
    # the file is stored NFD on disk; querying the NFC form must still find it by name
    out = file_grep("かえで総合病院", corpus_dir=mini_corpus, ext="md")
    names = [h for h in out["value"] if h["where"] == "filename"]
    assert names, "NFC query should match the NFD-on-disk filename"
    assert all(unicodedata.is_normalized("NFC", h["file"]) for h in out["value"])


# --------------------------------------------------------------------------- filters + regex
def test_ext_and_project_filters(mini_corpus: Path):
    out = file_grep("あ", corpus_dir=mini_corpus, ext="csv", project="かえで総合病院",
                    match_filename=False)
    assert out["value"]
    assert all(h["ext"] == "csv" and h["project"] == "かえで総合病院" for h in out["value"])


def test_regex_mode(mini_corpus: Path):
    out = file_grep(r"score|担当", regex=True, corpus_dir=mini_corpus)
    assert out["method"]["regex"] is True
    files = {h["file"] for h in out["value"]}
    assert any(f.endswith("train.csv") for f in files)
    assert any(f.endswith("メモ.txt") for f in files)


def test_no_match_is_empty_contract(mini_corpus: Path):
    out = file_grep("存在しない語句XYZ", corpus_dir=mini_corpus)
    assert is_contract(out)
    assert out["value"] == []
    assert out["evidence"]["files_matched"] == 0


def test_bad_regex_raises(mini_corpus: Path):
    with pytest.raises(FileGrepError):
        file_grep(r"(unclosed", regex=True, corpus_dir=mini_corpus)


def test_limit_truncates(mini_corpus: Path):
    out = file_grep(r".", regex=True, corpus_dir=mini_corpus, limit=1)  # matches every non-empty line
    assert out["evidence"]["total_hits"] > 1
    assert out["evidence"]["returned"] == 1
    assert out["evidence"]["truncated"] is True


# --------------------------------------------------------------------------- real corpus smoke
def test_real_corpus_nfc_filename_smoke():
    if not walk():
        pytest.skip("no real corpus present")
    out = file_grep("青葉", ext=("pdf", "docx", "pptx", "xlsx", "csv", "md"))
    assert is_contract(out)
    # 青葉 appears in several project folder names → at least one filename hit, all NFC.
    assert out["value"], "expected 青葉 to occur somewhere in the real drive"
    assert all(unicodedata.is_normalized("NFC", h["file"]) for h in out["value"])
