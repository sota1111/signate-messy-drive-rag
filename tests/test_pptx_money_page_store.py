"""SOT-2705 — pptx 金額提示ページ事実ストア + serve-path 決定論レーンの offline テスト.

2 層で検証する:
* **合成ストア**(``pptx_money_page_store.load`` を monkeypatch)+ **偽 Glossary**(``_glossary`` を monkeypatch)で、
  ネットワーク/LLM/実コーパス非依存に、決定論レーンの束縛(京ソ→案件・PP_final→提案書_final の略称解決)・
  金額提示スライドの一意判定・精度優先の deferral・RAG_PPTX_MONEY_PAGE 既定 OFF の byte-identical・ツール contract。
* **実 pptx レコード断言**(提案書_final.pptx): スライド13 の money_token_count 優位 ∧ visible_page_number=13。
  python-pptx のみ(soffice/genai 非依存)。コーパス未配置環境では skip。
"""
from __future__ import annotations

import pytest

from src.rag.agent import fact_layer as fl
from src.rag.agent import pptx_money_page_lane as pl
from src.rag.index import pptx_money_page_store as ps
from src.rag.tools import contract as _contract


# =========================================================================== store-level regex helpers
def test_money_token_count():
    texts = ["契約金額（税抜）\n¥5,250,000", "消費税額 ¥525,000", "本文に金額は無い", "5,775,000円"]
    assert ps._money_token_count(texts) == 3  # ¥5,250,000 / ¥525,000 / 5,775,000円


def test_money_token_count_zero_on_plain_numbers():
    # 通貨記号/単位のない裸数字(スライド番号・箇条番号)は金額トークンにしない。
    assert ps._money_token_count(["1", "2. 目的", "01 02 03"]) == 0


# =========================================================================== synthetic store + fake glossary
class _FakeGlossary:
    formal_to_abbrev = {"提案書": "PP", "最終報告": "FR"}

    def company_of(self, text):  # noqa: ANN001
        return "京橋信用ソリューションズ株式会社" if "京ソ" in text else None


_PROJ = "京橋信用ソリューションズ株式会社"


def _rows():
    return [
        # 対象: 提案書_final.pptx — スライド13『8. 費用見積』が金額提示ページ(可視頁13)。
        {"doc_id": f"プロジェクト/{_PROJ}/00.提案/提案書_final.pptx", "project": _PROJ,
         "category": "proposal", "doc_name": "提案書_final.pptx", "ext": "pptx", "slide_count": 3,
         "slides": [
             {"slide_index": 1, "title": "データ分析プロジェクト提案書", "money_token_count": 0,
              "has_pricing_table": False, "visible_page_number": 1},
             {"slide_index": 13, "title": "8. 費用見積", "money_token_count": 9,
              "has_pricing_table": False, "visible_page_number": 13},
             {"slide_index": 18, "title": "ご検討のほど", "money_token_count": 1,
              "has_pricing_table": False, "visible_page_number": 18}]},
        # 撹乱: 同案件の別提案書(v1) — PP_final では名指されない。
        {"doc_id": f"プロジェクト/{_PROJ}/00.提案/提案書_v1.pptx", "project": _PROJ,
         "category": "proposal", "doc_name": "提案書_v1.pptx", "ext": "pptx", "slide_count": 1,
         "slides": [{"slide_index": 5, "title": "費用見積", "money_token_count": 4,
                     "has_pricing_table": True, "visible_page_number": 5}]},
        # 撹乱: 別案件の pptx — 案件束縛で除外される。
        {"doc_id": "プロジェクト/白峰信用リスク評価株式会社/00.提案/提案書_final.pptx",
         "project": "白峰信用リスク評価株式会社", "category": "proposal",
         "doc_name": "提案書_final.pptx", "ext": "pptx", "slide_count": 1,
         "slides": [{"slide_index": 7, "title": "8. 費用見積", "money_token_count": 12,
                     "has_pricing_table": False, "visible_page_number": 7}]},
    ]


@pytest.fixture
def store(monkeypatch):
    monkeypatch.setattr(ps, "load", lambda path=None: _rows())
    monkeypatch.setattr(pl, "_glossary", lambda: _FakeGlossary())
    monkeypatch.setenv("RAG_PPTX_MONEY_PAGE", "1")
    monkeypatch.setenv("RAG_FACT_LAYER", "1")
    return None


_Q59 = "京ソのPP_final.pptxにおいて、この案件にかかる金額の提示がまとまっているのは何ページですか。"


def test_idx59_money_page_via_alias(store):
    # 京ソ→京橋(company_of) + PP_final→提案書_final(略称展開) で一意束縛、スライド13→可視頁13。
    res = pl.resolve(_Q59)
    assert res is not None and res["value"] == "13ページ"
    assert res["evidence"]["doc_name"] == "提案書_final.pptx"
    assert res["evidence"]["slide_index"] == 13
    assert res["evidence"]["visible_page_number"] == 13


def test_bare_answer_no_parenthetical(store):
    # cycle4 の wrong=括弧付加 の教訓: 裸形式 "13ページ" のみ(括弧付加情報なし)。
    assert pl.resolve(_Q59)["value"] == "13ページ"


def test_project_binding_excludes_other_case(store):
    # 別案件(白峰)の同名 提案書_final.pptx(費用見積 money=12)は 京ソ 束縛で除外される。
    assert pl.resolve(_Q59)["evidence"]["doc_id"].split("/")[1] == _PROJ


def test_selection_prefers_price_title_slide(store):
    # 価格タイトル(費用見積) ∧ 金額トークンありのスライドが一意 → それを採る(money argmax とも一致)。
    res = pl.resolve(_Q59)
    assert res["evidence"]["slide_title"] == "8. 費用見積"


def test_defer_when_document_ambiguous(store):
    # ファイル名(PP_final/提案書_final)を名指さない → 案件内で pptx を一意化できず defer。
    assert pl.resolve("京ソの金額の提示は何ページですか。") is None


def test_defer_when_no_money_cue(store):
    # 金額語がない(ページのみ)→ このレーンは発火しない。
    assert pl.resolve("京ソのPP_final.pptxのスケジュールは何ページですか。") is None


def test_defer_when_no_page_ask(store):
    # ページを問うていない(金額の内容質問)→ 発火しない。
    assert pl.resolve("京ソのPP_final.pptxの契約金額はいくらですか。") is None


# --------------------------------------------------------------------------- OFF byte-identical
def test_off_is_inert(monkeypatch):
    monkeypatch.setattr(ps, "load", lambda path=None: _rows())
    monkeypatch.setattr(pl, "_glossary", lambda: _FakeGlossary())
    monkeypatch.delenv("RAG_PPTX_MONEY_PAGE", raising=False)
    assert pl.enabled() is False
    assert pl.resolve(_Q59) is None
    assert pl.tool() is None


def test_off_lane_not_in_fact_layer(monkeypatch):
    monkeypatch.setattr(ps, "load", lambda path=None: _rows())
    monkeypatch.setattr(pl, "_glossary", lambda: _FakeGlossary())
    monkeypatch.setenv("RAG_FACT_LAYER", "1")
    monkeypatch.delenv("RAG_PPTX_MONEY_PAGE", raising=False)
    assert fl.resolve(_Q59, "fact_lookup") is None
    assert pl.PPTX_MONEY_PAGE_LOOKUP not in {t[0] for t in fl.tools()}


# --------------------------------------------------------------------------- fact_layer wiring (ON)
def test_fact_layer_routes_and_publishes_tool(store):
    res = fl.resolve(_Q59, "fact_lookup")
    assert res is not None and _contract.is_contract(res) and res["value"] == "13ページ"
    assert pl.PPTX_MONEY_PAGE_LOOKUP in {t[0] for t in fl.tools()}


# --------------------------------------------------------------------------- investigator tool contract
def test_tool_lookup_by_alias(store):
    name, desc, schema, handler = pl.tool()
    assert name == pl.PPTX_MONEY_PAGE_LOOKUP
    out = handler("PP_final.pptx")
    assert _contract.is_contract(out)
    docs = out["value"]
    names = {d["doc_name"] for d in docs}
    assert "提案書_final.pptx" in names
    # 対象文書のスライド13 が money 優位。
    target = [d for d in docs if d["doc_id"].split("/")[1] == _PROJ][0]
    s13 = [s for s in target["slides"] if s["slide_index"] == 13][0]
    assert s13["money_token_count"] == 9


# --------------------------------------------------------------------------- real pptx record assertion
def test_real_proposal_final_record():
    """提案書_final.pptx の実レコード: スライド13 が money_token_count 単独最大 ∧ visible_page_number=13。"""
    from src.rag import corpus
    refs = [r for r in corpus.walk()
            if r.name == "提案書_final.pptx" and "京橋" in r.project and r.ext == "pptx"]
    if not refs:
        pytest.skip("corpus 未配置(提案書_final.pptx なし)")
    rec = ps.compute_doc(refs[0])
    assert rec is not None
    by_idx = {s["slide_index"]: s for s in rec["slides"]}
    assert 13 in by_idx
    s13 = by_idx[13]
    top = max(s["money_token_count"] for s in rec["slides"])
    argmax = [s["slide_index"] for s in rec["slides"] if s["money_token_count"] == top]
    assert argmax == [13]  # money トークン密度の単独最大がスライド13
    assert s13["visible_page_number"] == 13
    assert pl._PRICE_TITLE.search(pl._norm(s13["title"]))  # タイトルに価格キーワード(費用/見積)


def test_real_proposal_final_lane_resolves(tmp_path, monkeypatch):
    """実 pptx から build したストア + 実 Glossary で、決定論レーンが idx59 を 13ページ に確定(略称解決込み)。"""
    from src.rag import corpus
    refs = [r for r in corpus.walk() if r.ext == "pptx"]
    if not any(r.name == "提案書_final.pptx" and "京橋" in r.project for r in refs):
        pytest.skip("corpus 未配置(提案書_final.pptx なし)")
    store_path = tmp_path / "pptx_money_page_store.jsonl"
    ps.build(refs, out=store_path, write_report=False)
    built = ps.load(store_path)
    monkeypatch.setattr(ps, "load", lambda path=None: built)
    monkeypatch.setenv("RAG_PPTX_MONEY_PAGE", "1")
    res = pl.resolve(_Q59)  # 実 Glossary(社内用語集.docx)で 京ソ/PP を解決
    assert res is not None and res["value"] == "13ページ"
