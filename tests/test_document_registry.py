"""SOT-2583 document registry and explicit-file hard-constraint tests."""
from __future__ import annotations

import json
import unicodedata
from pathlib import Path

import pytest

from src.rag.corpus import walk
from src.rag.index import document_registry as dr


@pytest.fixture()
def registry(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    root = tmp_path / "drive"
    names = [
        "プロジェクト/Alpha/02.計画/提案書old.pptx",
        "プロジェクト/Alpha/02.計画/提案書new.pptx",
        "プロジェクト/Alpha/03.データ/train.csv",
        "プロジェクト/Beta/03.データ/train.csv",
    ]
    # Exercise an NFD path on disk while FileRef/registry output remains NFC.
    for rel in names:
        path = root / unicodedata.normalize("NFD", rel)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"fixture")
    out = tmp_path / "document_registry.jsonl"
    monkeypatch.setattr(dr, "default_out_path", lambda: out)
    walk.cache_clear()
    dr.reset_cache()
    counts = dr.build(walk(root), probe=False)
    assert counts["files"] == 4
    return out


def test_build_has_full_schema_and_is_reproducible(registry: Path) -> None:
    first = registry.read_bytes()
    rows = dr.load(registry)
    required = {
        "doc_id", "case_id", "full_path", "normalized_filename", "filename_aliases",
        "extension", "title", "title_aliases", "named_entities", "detected_dates",
        "version_tokens", "version_family_id", "predecessor_doc_id", "sheet_names",
        "table_names", "slide_titles", "pivot_names", "comment_count",
        "conditional_format_count", "formatting_features", "parser_capabilities",
        "extraction_warnings", "synopsis", "content_fingerprint",
    }
    assert len(rows) == 4
    assert required <= rows[0].keys()
    assert all(unicodedata.is_normalized("NFC", row["doc_id"]) for row in rows)
    dr.write_registry([dr.DocRecord(**row) for row in rows], registry)
    assert registry.read_bytes() == first


def test_resolution_chain_exact_alias_version_and_project_scope(registry: Path) -> None:
    resolver = dr.Resolver(dr.load(registry))
    exact = resolver.resolve("Alpha の train.csv を確認", project="Alpha")
    assert exact[0].doc_id.endswith("Alpha/03.データ/train.csv")
    assert exact[0].tier == "exact_path_filename"
    alias = resolver.resolve("提案書.pptx を確認", project="Alpha")
    assert alias and alias[0].tier in {"alias", "version_family"}
    assert all(hit.case_id == "Alpha" for hit in alias)


def test_hard_constraint_is_opt_in_and_requires_every_named_file(
        registry: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    question = "train.csv と missing.xlsx を比較して"
    monkeypatch.delenv("RAG_DOCUMENT_REGISTRY", raising=False)
    assert dr.hard_constraint_abstain(question, project="Alpha", path=registry) is None
    monkeypatch.setenv("RAG_DOCUMENT_REGISTRY", "1")
    dr.reset_cache()
    assert dr.hard_constraint_abstain(question, project="Alpha", path=registry) == dr.DOC_NOT_FOUND_REASON
    assert dr.hard_constraint_abstain("train.csv を確認", project="Alpha", path=registry) is None


def test_measurement_records_recall_at_1_and_3(registry: Path) -> None:
    target = "プロジェクト/Alpha/03.データ/train.csv"
    result = dr.measure_recall([("train.csv を確認", "Alpha", target)], path=registry)
    assert result["canonical_doc_recall"] == {"@1": 1.0, "@3": 1.0}
    assert result["explicit_filename_resolution_rate"] == 1.0
