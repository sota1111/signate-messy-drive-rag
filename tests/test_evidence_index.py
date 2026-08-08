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
