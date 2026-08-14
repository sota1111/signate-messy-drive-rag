"""SOT-2714 — pptx ノート「スコープ対象外」✖項目カウント事実ストア + serve-path 決定論レーンの offline テスト.

2 層で検証する:
* **合成ストア**(``pptx_note_scope_store.load`` を monkeypatch)+ **偽 Glossary**(``_glossary`` を monkeypatch)で、
  ネットワーク/LLM/実コーパス非依存に、決定論レーンの束縛(案件 + doc-kind『提案書』)・count>0 の一意判定・
  精度優先の deferral・RAG_PPTX_NOTE_SCOPE 既定 OFF の byte-identical・ツール contract を検証。
* **段落カウント helper の単体**(見出し検出 / ✖項目 / 直下ブロック打ち切り)。
* **実 pptx レコード断言**(恒一会 かえで総合病院 提案書.pptx): notesSlide の ✖項目 = 7。
  zipfile + lxml のみ(soffice/genai 非依存)。コーパス未配置環境では skip。
"""
from __future__ import annotations

import pytest

from src.rag.agent import fact_layer as fl
from src.rag.agent import pptx_note_scope_lane as pl
from src.rag.index import pptx_note_scope_store as ps
from src.rag.tools import contract as _contract


# =========================================================================== store-level paragraph helpers
def test_is_heading_exact_and_decorated():
    assert ps._is_heading("スコープ対象外")
    assert ps._is_heading("■ スコープ対象外")
    assert ps._is_heading("【スコープ対象外】")
    # 見出し語を内包するだけの文は見出しでない(素のテキストが見出し語そのものでない)。
    assert not ps._is_heading("スコープ対象外について補足する")
    assert not ps._is_heading("当初合意スコープ外")


def test_is_xmark_item():
    assert ps._is_xmark_item("✖  実運用システムへの組込み")
    assert ps._is_xmark_item("  ✖ 先頭空白あり")
    assert not ps._is_xmark_item("・箇条書き")
    assert not ps._is_xmark_item("6")


def test_count_scope_items_stops_at_first_non_mark():
    paras = ["【備考】", "スコープ対象外",
             "✖  A", "✖  B、C", "✖  D", "6", "✖  E（別ブロック）"]
    items = ps._count_scope_items(paras)
    # 見出し直下の連続 ✖ 3本のみ(『6』で打ち切り、その後の ✖ は別ブロックで取り込まない)。
    assert items == ["✖  A", "✖  B、C", "✖  D"]


def test_count_scope_items_ignores_blank_paragraphs():
    paras = ["スコープ対象外", "", "✖  A", "", "✖  B", "本文へ", "✖  C"]
    assert ps._count_scope_items(paras) == ["✖  A", "✖  B"]


def test_count_scope_items_no_heading():
    assert ps._count_scope_items(["当初合意スコープ外", "✖  例1", "✖  例2"]) == []


# =========================================================================== synthetic store + fake glossary
class _FakeGlossary:
    formal_to_abbrev = {"提案書": "PP"}

    def company_of(self, text):  # noqa: ANN001
        return "医療法人社団 恒一会 かえで総合病院" if "恒一会" in text or "かえで" in text else None


_PROJ = "医療法人社団 恒一会 かえで総合病院"
_ITEMS7 = [f"✖  item{i}" for i in range(1, 8)]


def _rows():
    return [
        # 対象: 提案書.pptx — ノート「スコープ対象外」直下 ✖ 7本。
        {"doc_id": f"プロジェクト/{_PROJ}/00.提案/提案書.pptx", "project": _PROJ,
         "category": "proposal", "doc_name": "提案書.pptx", "ext": "pptx",
         "notes_slide_count": 1, "scope_excluded_count": 7,
         "notes": [{"note": "ppt/notesSlides/notesSlide1.xml", "items": _ITEMS7, "count": 7}],
         "items": _ITEMS7},
        # 撹乱: 同案件の別 pptx(スコープ対象外なし) — doc-kind『提案書』束縛 + count>0 で除外。
        {"doc_id": f"プロジェクト/{_PROJ}/00.提案/概要.pptx", "project": _PROJ,
         "category": "proposal", "doc_name": "概要.pptx", "ext": "pptx",
         "notes_slide_count": 0, "scope_excluded_count": 0, "notes": [], "items": []},
        # 撹乱: 別案件の提案書(スコープ対象外3本) — 案件束縛で除外。
        {"doc_id": "プロジェクト/白峰信用リスク評価株式会社/00.提案/提案書.pptx",
         "project": "白峰信用リスク評価株式会社", "category": "proposal",
         "doc_name": "提案書.pptx", "ext": "pptx", "notes_slide_count": 1,
         "scope_excluded_count": 3, "notes": [], "items": ["✖ x", "✖ y", "✖ z"]},
    ]


@pytest.fixture
def store(monkeypatch):
    monkeypatch.setattr(ps, "load", lambda path=None: _rows())
    monkeypatch.setattr(pl, "_glossary", lambda: _FakeGlossary())
    monkeypatch.setenv("RAG_PPTX_NOTE_SCOPE", "1")
    monkeypatch.setenv("RAG_FACT_LAYER", "1")
    return None


_Q27 = "恒一会 かえで総合病院の提案書において、スコープ対象外としている項目はいくつありますか。"


def test_idx27_scope_count(store):
    res = pl.resolve(_Q27)
    assert res is not None and res["value"] == "7"
    assert res["evidence"]["doc_name"] == "提案書.pptx"
    assert res["evidence"]["scope_excluded_count"] == 7


def test_bare_integer_answer(store):
    # 裸整数 "7"(項目/個 等の接尾なし・gold と byte 一致)。
    assert pl.resolve(_Q27)["value"] == "7"


def test_project_binding_excludes_other_case(store):
    # 別案件(白峰)の同名 提案書.pptx(scope=3)は 恒一会 束縛で除外される。
    assert pl.resolve(_Q27)["evidence"]["doc_id"].split("/")[1] == _PROJ


def test_defer_when_company_absent(store):
    # どの案件も名指さない → 案件を一意化できず defer(rows 全体へ広げない)。
    assert pl.resolve("提案書でスコープ対象外の項目はいくつありますか。") is None


def test_defer_when_no_count_ask(store):
    # 個数を問うていない(内容質問)→ 発火しない。
    assert pl.resolve("恒一会 かえで総合病院の提案書のスコープ対象外の内容を教えてください。") is None


def test_defer_when_no_scope_cue(store):
    # スコープ対象外 cue がない → 発火しない。
    assert pl.resolve("恒一会 かえで総合病院の提案書のスライドはいくつありますか。") is None


def test_defer_when_count_zero(monkeypatch, store):
    # 対象案件の提案書だが scope=0 のみ → 一意な scope>0 文書が無く defer。
    monkeypatch.setattr(ps, "load", lambda path=None: [
        {"doc_id": f"プロジェクト/{_PROJ}/00.提案/提案書.pptx", "project": _PROJ,
         "category": "proposal", "doc_name": "提案書.pptx", "ext": "pptx",
         "notes_slide_count": 0, "scope_excluded_count": 0, "notes": [], "items": []}])
    assert pl.resolve(_Q27) is None


# --------------------------------------------------------------------------- OFF byte-identical
def test_off_is_inert(monkeypatch):
    monkeypatch.setattr(ps, "load", lambda path=None: _rows())
    monkeypatch.setattr(pl, "_glossary", lambda: _FakeGlossary())
    monkeypatch.delenv("RAG_PPTX_NOTE_SCOPE", raising=False)
    assert pl.enabled() is False
    assert pl.resolve(_Q27) is None
    assert pl.tool() is None


def test_off_lane_not_in_fact_layer(monkeypatch):
    monkeypatch.setattr(ps, "load", lambda path=None: _rows())
    monkeypatch.setattr(pl, "_glossary", lambda: _FakeGlossary())
    monkeypatch.setenv("RAG_FACT_LAYER", "1")
    monkeypatch.delenv("RAG_PPTX_NOTE_SCOPE", raising=False)
    assert fl.resolve(_Q27, "fact_lookup") is None
    assert pl.PPTX_NOTE_SCOPE_LOOKUP not in {t[0] for t in fl.tools()}


# --------------------------------------------------------------------------- fact_layer wiring (ON)
def test_fact_layer_routes_and_publishes_tool(store):
    res = fl.resolve(_Q27, "fact_lookup")
    assert res is not None and _contract.is_contract(res) and res["value"] == "7"
    assert pl.PPTX_NOTE_SCOPE_LOOKUP in {t[0] for t in fl.tools()}


# --------------------------------------------------------------------------- investigator tool contract
def test_tool_lookup_by_project(store):
    name, desc, schema, handler = pl.tool()
    assert name == pl.PPTX_NOTE_SCOPE_LOOKUP
    out = handler("恒一会 かえで総合病院")
    assert _contract.is_contract(out)
    docs = out["value"]
    target = [d for d in docs if d["doc_id"].split("/")[1] == _PROJ and d["scope_excluded_count"] == 7]
    assert target and target[0]["doc_name"] == "提案書.pptx"


# --------------------------------------------------------------------------- real pptx record assertion
def _target_ref():
    import unicodedata

    from src.rag import corpus
    for r in corpus.walk():
        if r.ext == "pptx" and "恒一会" in unicodedata.normalize("NFC", r.rel) \
                and "提案書" in unicodedata.normalize("NFC", r.name):
            return r
    return None


def test_real_proposal_record_counts_seven():
    """恒一会 かえで総合病院 提案書.pptx の実レコード: ノート「スコープ対象外」✖項目 = 7。"""
    ref = _target_ref()
    if ref is None:
        pytest.skip("corpus 未配置(恒一会 提案書.pptx なし)")
    rec = ps.compute_doc(ref)
    assert rec is not None
    assert rec["scope_excluded_count"] == 7
    assert all(it.lstrip().startswith("✖") for it in rec["items"])


def test_real_proposal_lane_resolves(tmp_path, monkeypatch):
    """実 pptx から build したストア + 実 Glossary で、決定論レーンが idx27 を 7 に確定。"""
    from src.rag import corpus
    if _target_ref() is None:
        pytest.skip("corpus 未配置(恒一会 提案書.pptx なし)")
    store_path = tmp_path / "pptx_note_scope_store.jsonl"
    ps.build([r for r in corpus.walk() if r.ext == "pptx"], out=store_path, write_report=False)
    built = ps.load(store_path)
    monkeypatch.setattr(ps, "load", lambda path=None: built)
    monkeypatch.setenv("RAG_PPTX_NOTE_SCOPE", "1")
    res = pl.resolve(_Q27)  # 実 Glossary(社内用語集.docx)で 案件解決
    assert res is not None and res["value"] == "7"
