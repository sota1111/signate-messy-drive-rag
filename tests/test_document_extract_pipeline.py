"""SOT-2611 (Wave B1) — offline tests for the deterministic ``document_extract`` (format_check) pipeline.

Network-free. The unit tests drive the pipeline's own logic — colour/sheet parsing, the aggregation-column
detector (incl. the ``Country`` ⊅ ``count`` word-boundary guard), the outline forward-fill with digit-prefix
migration, and the 2D cross-tab renderer with a stubbed ``resolve_pivot_semantics`` — with canned grids, so
the recognizer runs (not a stub). Invariants under test: the pipeline self-registers ``format_check``; the
outline recognizer forward-fills grouping columns to the highlighted row and renders ``列=値、…で抽出された
データに対する<集計>`` (idx15/80 型); the 2D recognizer renders ``行f=行ラベル、列f=列見出し…<集計> / <値列>``
(idx7 型); ambiguity / missing structure / unknown colour ⇒ ``None`` (LLM fallback); OFF ⇒ byte-identical.
A corpus-gated integration test proves idx7/15/80 match against the real corpus when present.
"""
from __future__ import annotations

import os

import pytest

from src.rag.agent import det_pipeline as dp
from src.rag.agent import formatting
from src.rag.agent.pipelines import document_extract as de


# --------------------------------------------------------------------------- registry cleanup fixture
@pytest.fixture(autouse=True)
def _restore_registry():
    saved = dict(dp._REGISTRY)
    try:
        yield
    finally:
        dp._REGISTRY.clear()
        dp._REGISTRY.update(saved)


# --------------------------------------------------------------------------- registration
def test_pipeline_registers_for_format_check():
    de.register(replace=True)
    assert dp._REGISTRY.get("format_check") is de.pipeline


def test_router_off_is_byte_identical(monkeypatch):
    monkeypatch.delenv("RAG_DET_PIPELINE_ROUTER", raising=False)
    assert dp.enabled() is False
    # Flag OFF ⇒ resolve short-circuits to None even for a format_check question (LLM loop unchanged).
    assert dp.resolve("黄色にハイライトされたセルの抽出条件と集計", "format_check") is None


# --------------------------------------------------------------------------- colour / sheet / cue parsing
@pytest.mark.parametrize("q,expected", [
    ("黄色にハイライトされたセル", "黄"),
    ("黄にマーカーされた", "黄"),
    ("オレンジにハイライトされている行", "オレンジ"),
    ("水色で塗りつぶされたセル", "水色"),
    ("特に色の指定なし", None),
])
def test_resolve_color(q, expected):
    assert de._resolve_color(q) == expected


@pytest.mark.parametrize("q,expected", [
    ("Sheet1の黄色", "Sheet1"),
    ("Sheet2 の黄色", "Sheet2"),
    ("sheet 3 のセル", "sheet3"),
    ("シート指定なし", None),
])
def test_target_sheet(q, expected):
    assert de._target_sheet(q) == expected


# --------------------------------------------------------------------------- aggregation-column detection
def test_agg_column_word_boundary_guard():
    # ``Country`` must NOT be read as the ascii aggregation token ``count`` — the value header is ``個数``.
    header = ["Gender", "target", "Age", "Country", "個数"]
    assert de._agg_column(header) == (4, "個数")


def test_agg_column_strips_value_suffix():
    # ``個数 / id`` renders as the bare aggregation ``個数`` (the ``/ <value column>`` suffix is dropped).
    assert de._agg_column(["Gender", "Age", "個数 / id", ""]) == (2, "個数")


def test_agg_column_none_when_no_aggregation_header():
    # A 2D cross-tab header (行ラベル + numeric column labels) carries no aggregation word.
    assert de._agg_column(["行ラベル", "0", "1", "2"]) is None


# --------------------------------------------------------------------------- outline forward-fill
def test_outline_condition_forward_fill_real_cells():
    # A flattened outline pivot in genuine cells (idx15 型): grouping columns repeat as blanks below.
    table = [
        ["Gender", "target", "Age", "Country", "個数"],
        ["Male", "2", "40-44", "France", "5"],
        ["", "", "", "Spain", "12"],          # ← highlighted row: Gender/target/Age inherited from above
    ]
    got = de._outline_condition(table, hl_row=2)
    assert got is not None
    assert got["answer"] == "Gender=Male、target=2、Age=40-44、Country=Spainで抽出されたデータに対する個数"


def test_outline_condition_digit_prefix_migration():
    # The EMF reader clusters the numeric ``target`` group value as a ``"<digit> <rest>"`` prefix into the
    # next (Age) column; the recognizer migrates the digit back to its own column (idx80 型).
    table = [
        ["Gender", "target", "Age", "Profession", "個数 / id", ""],
        ["Male", "", "3 18-21", "Analyst", "", "4"],
        ["", "", "30-34", "Software Engineer", "", "61"],   # ← highlighted row
    ]
    got = de._outline_condition(table, hl_row=2)
    assert got is not None
    assert got["answer"] == (
        "Gender=Male、target=3、Age=30-34、Profession=Software Engineerで抽出されたデータに対する個数")


def test_outline_condition_abstains_on_unfilled_group():
    # No ancestor ever sets Gender ⇒ the condition cannot be fully stated ⇒ fallback (None).
    table = [
        ["Gender", "Age", "個数"],
        ["", "40-44", "12"],
    ]
    assert de._outline_condition(table, hl_row=1) is None


# --------------------------------------------------------------------------- 2D cross-tab
def test_twod_condition_renders_row_and_column(monkeypatch):
    table = [["行ラベル", "0", "1", "2"], ["8", "", "", "0.78"]]
    hl_cell = {"row": 1, "row_label": "8", "col_header": "2"}
    monkeypatch.setattr(de, "_resolve_file_ref", lambda q: None)  # not used directly here
    monkeypatch.setattr(
        "src.rag.tools.emf_pivot._pptx_data_sources", lambda ref: [])
    monkeypatch.setattr(
        "src.rag.tools.emf_pivot.resolve_pivot_semantics",
        lambda table, sources: {
            "row_field": "hr", "column_field": "weekday",
            "target_column": "temp", "aggregation_label": "最大"})
    got = de._twod_condition(table, hl_cell, ref=object())
    assert got is not None
    assert got["answer"] == "hr=8、weekday=2で抽出されたデータに対する最大 / temp"


def test_twod_condition_abstains_without_labels():
    # A highlighted cell that maps to no row label / column header cannot state a condition.
    assert de._twod_condition([["行ラベル", "0"], ["8", "0.78"]],
                              {"row": 1, "row_label": "", "col_header": ""}, ref=object()) is None


# --------------------------------------------------------------------------- recognizer gating
def test_recognizer_requires_highlight_and_condition_cues():
    # Missing the ハイライト cue ⇒ the recognizer never even resolves a file (None).
    assert de._highlight_condition("train.xlsx のセルの値を教えて") is None


def test_pipeline_never_raises_on_garbage():
    assert de.pipeline("") is None
    assert de.pipeline("!!!") is None


# --------------------------------------------------------------------------- corpus-gated integration (idx7/15/80)
_CORPUS_PRESENT = (
    __import__("config").settings.CORPUS_DIR.exists()
    if os.getenv("RAG_SKIP_CORPUS_TESTS") not in {"1", "true", "yes", "on"} else False
)


@pytest.mark.skipif(not _CORPUS_PRESENT, reason="corpus (data/share_drive) not present")
@pytest.mark.parametrize("question,gold", [
    ("青潮モビリティサービスの基礎分析.pptxにおいて、黄色ハイライトされている数値に対応するデータの"
     "抽出条件と集計内容を答えてください。",
     "hr=8、weekday=2で抽出されたデータに対する最大 / temp"),
    ("東都人材プラットフォームのtrain.xlsxにおいて、Sheet1の黄色にハイライトされたセルの抽出条件と"
     "集計内容を答えてください。",
     "Gender=Male、target=2、Age=40-44、Country=Spainで抽出されたデータに対する個数"),
    ("東都人材プラットフォームのtrain.xlsxにおいて、Sheet2の黄色にハイライトされたセルの抽出条件と"
     "集計内容を答えてください。",
     "Gender=Male、target=3、Age=30-34、Profession=Software Engineerで抽出されたデータに対する個数"),
])
def test_real_corpus_idx7_idx15_idx80_match(question, gold):
    contract = dp.resolve(question, "format_check", force=True)
    assert contract is not None, "recognizer did not ground the extraction condition on the real corpus"
    formatted = formatting.format_contract(contract, question, contract_type="format_check", force=True)
    assert formatted["value"] == gold
