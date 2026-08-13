"""SOT-2693 — 書式付き数値系列ストア（cycle8 C4）の決定論ヘルパの offline テスト（LLM/corpus 不要）。"""
from __future__ import annotations

from types import SimpleNamespace

from src.rag.index import format_series_store as S


def _ref(rel, name):
    return SimpleNamespace(rel=rel, name=name)


def test_series_type_from_path():
    assert S._series_type("プロジェクト/X/05.会議/会議録/会議録_2025-04-09.pdf") == "会議録"
    assert S._series_type("プロジェクト/X/05.会議/報告資料/報告資料_2025-04-29.docx") == "報告資料"
    assert S._series_type("プロジェクト/X/00.提案/提案書.pptx") is None


def test_doc_date_parsing():
    assert S._doc_date(_ref("a/会議録_2025-04-09.pdf", "会議録_2025-04-09.pdf")) == "2025-04-09"
    assert S._doc_date(_ref("a/報告_20250527.docx", "報告_20250527.docx")) == "2025-05-27"
    assert S._doc_date(_ref("a/no_date.pdf", "no_date.pdf")) is None
    assert S._doc_date(_ref("a/2025-13-40.pdf", "2025-13-40.pdf")) is None  # invalid m/d


def test_numbers_and_float():
    assert S._numbers("F1(macro): 0.589 / 4,620,000") == ["0.589", "4,620,000"]
    assert S._to_float("4,620,000") == 4620000.0
    assert S._to_float("x") is None


def test_color_predicates():
    assert S._is_red((1.0, 0.0, 0.0)) and not S._is_red((0.0, 0.0, 1.0))
    assert S._is_red((0.1, 1.0, 1.0, 0.0))  # CMYK red
    assert S._is_yellow((1.0, 1.0, 0.0)) and not S._is_yellow((1.0, 1.0, 1.0))
    assert not S._is_red(None) and not S._is_yellow(0.5)
