from __future__ import annotations

import json
import unicodedata
from pathlib import Path

from src.rag.corpus import FileRef
from src.rag.index import evidence_index


def _ref(tmp_path: Path) -> FileRef:
    nfd_name = unicodedata.normalize("NFD", "顧客一覧.xlsx")
    return FileRef(tmp_path / nfd_name, unicodedata.normalize("NFD", "青葉"), "data",
                   f"プロジェクト/青葉/{nfd_name}", nfd_name, "xlsx")


def test_scan_doc_builds_typed_locations_and_normalizes_nfc(tmp_path: Path) -> None:
    text = """[シート: 連絡先]  範囲 A1:C3
氏名 | EXT | 備考
佐藤さん | EXT: 1234 | 別契約と明記
【ハイライトされたセル】
  C3(オレンジ): 要確認
"""
    entries = evidence_index.scan_doc(_ref(tmp_path), text)

    assert any(e.type == "person" and e.key == "佐藤" and e.sheet == "連絡先" for e in entries)
    assert any(e.type == "ext" and e.key == "1234" and e.sheet == "連絡先" for e in entries)
    assert any(e.type == "condition" and e.value == "別契約" for e in entries)
    assert any(e.type == "condition" and e.value == "明記" for e in entries)
    assert any(e.type == "highlight" and (e.sheet, e.cell, e.value) ==
               ("連絡先", "C3", "要確認") for e in entries)
    assert any(e.type == "column" and e.value == "氏名" for e in entries)
    assert all(unicodedata.is_normalized("NFC", e.project + e.file + e.rel) for e in entries)


def test_scan_doc_indexes_sheet_names_and_headings(tmp_path: Path) -> None:
    """SOT-2562 coverage expansion: sheet names + slide/markdown headings become searchable types."""
    text = """[シート: 売上サマリ]
項目 | 金額
[スライド1]
新税率の提案について
本文行
# 導入セクション見出し
"""
    entries = evidence_index.scan_doc(_ref(tmp_path), text)
    assert any(e.type == "sheet" and e.value == "売上サマリ" and e.sheet == "売上サマリ"
               for e in entries), "sheet name should be indexed as a `sheet` entry"
    assert any(e.type == "heading" and e.value == "新税率の提案について" and e.paragraph == "slide:1"
               for e in entries), "the first line after a slide marker is its title heading"
    assert any(e.type == "heading" and e.value == "導入セクション見出し" for e in entries), \
        "a markdown '# …' line is indexed as a heading"
    # a slide title must not be mistaken for the marker line itself
    assert not any(e.type == "heading" and e.value.startswith("[スライド") for e in entries)


def test_candidate_files_ranks_filename_and_index_tokens(tmp_path: Path, monkeypatch) -> None:
    """A miss query narrows to a small candidate set (no extraction) via filename + index tokens."""
    from src.rag import corpus

    target = FileRef(tmp_path / "ニューヨーク不動産市場の最新動向調査.pdf", "青嶺不動産", "proposal",
                     "プロジェクト/青嶺不動産/ニューヨーク不動産市場の最新動向調査.pdf",
                     "ニューヨーク不動産市場の最新動向調査.pdf", "pdf")
    other = FileRef(tmp_path / "train.xlsx", "別会社", "data",
                    "プロジェクト/別会社/train.xlsx", "train.xlsx", "xlsx")
    monkeypatch.setattr(corpus, "walk", lambda *a, **k: [target, other])

    out = tmp_path / "evidence_index.jsonl"
    evidence_index.write_index(evidence_index.scan_doc(other, "[シート: 連絡先]\n田中様 EXT 1234\n"), out)
    evidence_index.reset_cache()

    # filename token drives the ranking: the distinctively-named PDF is the top candidate.
    cands = evidence_index.candidate_files("ニューヨーク不動産市場の最新動向調査", path=out, limit=5)
    assert cands and cands[0] == target.rel
    # an empty/garbage query yields no candidates (caller then decides its own fallback).
    assert evidence_index.candidate_files("", path=out) == []


def test_write_index_has_schema_is_reproducible_and_scoped_by_project(tmp_path: Path) -> None:
    first = _ref(tmp_path)
    second = FileRef(tmp_path / "same.txt", "赤坂", "data", "赤坂/same.txt", "same.txt", "txt")
    entries = (evidence_index.scan_doc(first, "田中様 EXT 9876") +
               evidence_index.scan_doc(second, "田中様 EXT 9876"))
    out = tmp_path / "evidence_index.jsonl"
    counts = evidence_index.write_index(entries, out)
    original = out.read_bytes()
    evidence_index.write_index(list(reversed(entries)), out)

    rows = [json.loads(line) for line in out.read_text(encoding="utf-8").splitlines()]
    assert rows[0] == {"schema": evidence_index.SCHEMA, "version": evidence_index.SCHEMA_VERSION}
    assert counts == {"entries": len(entries), "files": 2}
    assert {row["project"] for row in rows[1:]} == {"青葉", "赤坂"}
    assert out.read_bytes() == original
