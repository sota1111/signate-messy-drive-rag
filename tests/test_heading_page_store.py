"""SOT-2668 — 見出し→印字ページ locator ストア + serve-path 決定論レーンの offline テスト.

ネットワーク/LLM/実コーパス/soffice 非依存。印字ページ検出・番号付き見出し抽出・自称ID抽出・docx 見出しの
ページ割当という決定論ヘルパを合成入力で検証し、合成ストア（``heading_page_store.load`` を monkeypatch）で
2 型の決定論レーン（idx12 docx見出し→印字ページ / idx18 会議ID:M04→PDF見出し→印字ページ）と精度優先の
deferral、RAG_HEADING_PAGE_STORE 既定 OFF の byte-identical 挙動、ツール contract を検証する。
"""
from __future__ import annotations

import pytest

from src.rag.agent import fact_layer as fl
from src.rag.agent import heading_page_lane as hpl
from src.rag.index import heading_page_store as hps
from src.rag.tools import contract as _contract


# =========================================================================== printed-page detection
def test_printed_page_slash_footer():
    # "N / TOTAL" フッタ（最後の出現）を印字番号として採る。
    assert hps._printed_page("本文...\n5 / 15", physical=6) == 5
    assert hps._printed_page("表\n1 / 7\n本文\n3 / 7", physical=9) == 3


def test_printed_page_trailing_number():
    # 末尾行が数字のみ = フッタのページ番号。物理と一致する採番も拾う。
    assert hps._printed_page("見出し\n本文です。\n2", physical=2) == 2
    # 本文中の数字は末尾行でなければ拾わない → 物理へ退避。
    assert hps._printed_page("当日実施対象タスク数\n2\nT01、T02\n本文", physical=4) == 4


def test_printed_page_fallback_physical():
    assert hps._printed_page("番号なしの本文だけ", physical=3) == 3


# =========================================================================== numbered heading extraction
def test_numbered_headings_from_pages():
    pages = [
        (1, "分析進捗報告書\n2. 進捗状況\n本文本文\n1"),
        (2, "2.3 WBS観点の進捗状況\n現時点で参照可能な...\n4. 進捗サマリ\n2"),
    ]
    heads = hps._numbered_headings(pages)
    got = {(h["text"], h["page"]) for h in heads}
    assert ("進捗状況", 1) in got
    assert ("WBS観点の進捗状況", 2) in got
    assert ("進捗サマリ", 2) in got


def test_numbered_headings_skip_table_rows():
    # パイプ表の行は見出しにしない。
    heads = hps._numbered_headings([(1, "1. 見出し\nAI-05 | 内容 | 担当 | 状態\n2 | x | y")])
    assert [h["text"] for h in heads] == ["見出し"]


# =========================================================================== self-declared doc IDs (page-1 only)
def test_doc_ids_from_page1_header_only():
    # 会議ID はページ1ヘッダの自称のみ採る（本文の次回会議 M04 参照は拾わない）。
    pages = [
        (1, "会議録\n会議ID：M03 (中間レビュー)\n出席者: ..."),
        (4, "次回会議予定\n会議ID:M04 (モデル比較レビュー)"),
    ]
    assert hps._doc_ids(pages) == ["M03"]


def test_strip_heading_number():
    assert hps.strip_heading_number("2.3 WBS観点の進捗状況") == "WBS観点の進捗状況"
    assert hps.strip_heading_number("4. 進捗サマリ") == "進捗サマリ"
    assert hps.strip_heading_number("進捗サマリ") == "進捗サマリ"


# =========================================================================== docx heading → rendered page
def test_assign_docx_heading_pages_monotonic():
    pages = [
        (1, "分析進捗報告書 2. 進捗状況 2.1 チェックポイント時点の全体進捗 本文 1"),
        (2, "2.3 WBS観点の進捗状況 現時点で参照可能な 2"),
        (3, "3. 主要な分析結果 3"),
    ]
    headings = ["分析進捗報告書", "2. 進捗状況", "2.3 WBS観点の進捗状況", "3. 主要な分析結果"]
    out = hps._assign_docx_heading_pages(headings, pages)
    by = {h["text"]: h["page"] for h in out}
    assert by["2.3 WBS観点の進捗状況"] == 2
    assert by["3. 主要な分析結果"] == 3


# =========================================================================== synthetic store + lanes
_DOCX = "プロジェクト/医療法人社団 蒼泉会 ひがし丘総合病院/05.会議/報告資料/報告資料_2025-07-08.docx"
_PDF = "プロジェクト/白峰信用リスク評価株式会社/05.会議/会議録/会議録_2025-07-15.pdf"
# 撹乱: 同案件の別会議録（M03）— 会議ID で正しく弁別できることを確かめる。
_PDF_M03 = "プロジェクト/白峰信用リスク評価株式会社/05.会議/会議録/会議録_2025-06-17.pdf"


def _rows():
    return [
        {"doc_id": _DOCX, "project": "医療法人社団 蒼泉会 ひがし丘総合病院", "category": "meeting",
         "ext": "docx", "date": "2025-07-08", "doc_name": "報告資料_2025-07-08.docx",
         "doc_ids": [], "page_count": 8,
         "headings": [
             {"text": "分析進捗報告書", "page": 1},
             {"text": "2. 進捗状況", "page": 1},
             {"text": "2.1 チェックポイント時点の全体進捗", "page": 1},
             {"text": "2.3 WBS観点の進捗状況", "page": 2},
             {"text": "3. 主要な分析結果", "page": 3}]},
        {"doc_id": _PDF, "project": "白峰信用リスク評価株式会社", "category": "meeting",
         "ext": "pdf", "date": "2025-07-15", "doc_name": "会議録_2025-07-15.pdf",
         "doc_ids": ["M04"], "page_count": 4,
         "headings": [
             {"text": "会議の目的", "page": 1},
             {"text": "進捗サマリ", "page": 2},
             {"text": "アクション一覧", "page": 3}]},
        {"doc_id": _PDF_M03, "project": "白峰信用リスク評価株式会社", "category": "meeting",
         "ext": "pdf", "date": "2025-06-17", "doc_name": "会議録_2025-06-17.pdf",
         "doc_ids": ["M03"], "page_count": 4,
         "headings": [
             {"text": "会議の目的", "page": 1},
             {"text": "進捗サマリ", "page": 3}]},
    ]


@pytest.fixture
def store(monkeypatch):
    monkeypatch.setattr(hps, "load", lambda path=None: _rows())
    monkeypatch.setenv("RAG_HEADING_PAGE_STORE", "1")
    monkeypatch.setenv("RAG_FACT_LAYER", "1")
    return None


_Q12 = ("蒼泉会 ひがし丘総合病院の報告資料_2025-07-08.docxにおいて、"
        "WBS観点の進捗状況の見出しがあるのは何ページですか。")
_Q18 = ("白峰信用リスク評価の会議ID：M04の会議録にて、"
        "進捗サマリが記載されているページ番号を答えてください。")


def test_idx12_docx_heading_to_printed_page(store):
    res = hpl.resolve(_Q12)
    assert res is not None and res["value"] == "2ページ"
    assert res["evidence"]["doc_name"] == "報告資料_2025-07-08.docx"
    assert res["evidence"]["heading"] == "2.3 WBS観点の進捗状況"


def test_idx18_meeting_id_resolves_correct_pdf(store):
    # 会議ID:M04 → 07-15 の会議録（M03 の 06-17 ではない）で「進捗サマリ」= p2。
    res = hpl.resolve(_Q18)
    assert res is not None and res["value"] == "2ページ"
    assert res["evidence"]["doc_name"] == "会議録_2025-07-15.pdf"


def test_longest_heading_wins_over_substring(store):
    # 「進捗状況」も部分一致するが、より特定的な「WBS観点の進捗状況」(p2) を採る。
    res = hpl.resolve(_Q12)
    assert res["evidence"]["printed_page"] == 2


def test_defer_when_document_ambiguous(store):
    # 案件も自称IDも名指さない → 文書を一意に束縛できず defer。
    assert hpl.resolve("進捗サマリの見出しは何ページですか。") is None


def test_defer_when_no_page_cue(store):
    # ページを問うていない（見出しの内容質問）→ このレーンは発火しない。
    assert hpl.resolve("会議ID：M04の会議録の進捗サマリの内容を教えてください。") is None


def test_defer_when_heading_absent(store):
    # 文書は特定できるが、その文書に存在しない見出し → defer（偽ページを作らない）。
    assert hpl.resolve("会議ID：M04の会議録にて、リスク管理の見出しは何ページですか。") is None


# --------------------------------------------------------------------------- OFF byte-identical
def test_off_is_inert(monkeypatch):
    monkeypatch.setattr(hps, "load", lambda path=None: _rows())
    monkeypatch.delenv("RAG_HEADING_PAGE_STORE", raising=False)
    assert hpl.enabled() is False
    assert hpl.resolve(_Q12) is None
    assert hpl.tool() is None


def test_off_lane_not_in_fact_layer(monkeypatch):
    monkeypatch.setattr(hps, "load", lambda path=None: _rows())
    monkeypatch.setenv("RAG_FACT_LAYER", "1")
    monkeypatch.delenv("RAG_HEADING_PAGE_STORE", raising=False)
    assert fl.resolve(_Q12, "fact_lookup") is None
    assert hpl.HEADING_PAGE_LOOKUP not in {t[0] for t in fl.tools()}


# --------------------------------------------------------------------------- fact_layer wiring (ON)
def test_fact_layer_routes_and_publishes_tool(store):
    res = fl.resolve(_Q12, "fact_lookup")
    assert res is not None and _contract.is_contract(res) and res["value"] == "2ページ"
    assert hpl.HEADING_PAGE_LOOKUP in {t[0] for t in fl.tools()}


# --------------------------------------------------------------------------- investigator tool contract
def test_tool_lookup_by_meeting_id(store):
    name, desc, schema, handler = hpl.tool()
    assert name == hpl.HEADING_PAGE_LOOKUP
    out = handler("会議ID:M04", heading="進捗サマリ")
    assert _contract.is_contract(out)
    docs = out["value"]
    assert len(docs) == 1 and docs[0]["doc_name"] == "会議録_2025-07-15.pdf"
    assert any(h["page"] == 2 for h in docs[0]["headings"])
