"""SOT-2705 — pptx 金額提示ページ事実ストアの serve-path 配線(決定論レーン + investigator ツール).

:mod:`src.rag.index.pptx_money_page_store` が build 時に質問非依存で焼いた
「pptx×スライド→{title, money_token_count, has_pricing_table, visible_page_number}」を、answer path の
**決定論直答レーン**として接続する。heading_page_lane と同じ精度優先の規律:

* 対象 pptx を一意に束縛でき、かつその文書内で「金額提示スライド」が(タイトルの費用/見積/価格キーワード
  or 金額トークン密度の一意 argmax で)ちょうど 1 枚に確定する時だけ ``{value,...}`` を返し、少しでも曖昧なら
  ``None`` を返して LLM ループへフォールバック(回答数を減らさない・wrong を増やさない)。
* ページは可視頁番号テキスト優先、無ければ印字ページ規約(物理 1-based)。裸形式(``13ページ``)で返し括弧付加を
  しない(cycle4 の idx59 wrong=括弧付加情報 の教訓)。
* 別名解決は既存 Glossary(社内用語集.docx)で決定論に行う: ``京ソ``→京橋信用ソリューションズ株式会社、
  ``PP``→提案書。よって ``PP_final.pptx``→``提案書_final.pptx`` をハードコードなしで束縛する。
* ``RAG_PPTX_MONEY_PAGE`` 既定 OFF ⇒ :func:`resolve`/:func:`tool` は None ⇒ ツール集合・serve path は
  byte-identical。serve 中 genai 呼び出しは 0(事前計算済みを読むだけ)。

対象(cycle10 abstain):
* idx59 — 京ソ 提案書_final.pptx『8. 費用見積』(スライド13)の金額提示ページ → gold=13ページ。
"""
from __future__ import annotations

import re
import unicodedata
from typing import Any, Callable

from src.rag.index import pptx_money_page_store as _store
from src.rag.tools import contract as _contract

PPTX_MONEY_PAGE_LOOKUP = "pptx_money_page_lookup"
_STR = {"type": "string"}

# 金額を問う語 と ページ位置を問う語 の共起で発火(どちらか欠けたら発火しない)。
_MONEY_CUE = re.compile(r"金額|費用|見積|価格|コスト|料金|単価")
_PAGE_ASK = re.compile(r"何ページ|ページ番号|ページ目|ページですか|ページでしょうか|ページに|ページか")
# 金額提示スライドのタイトルに現れる価格キーワード(第一シグナル)。
_PRICE_TITLE = re.compile(r"費用|見積|価格|金額|単価|コスト")
# 会社法人格(案件名突合キーの正規化: 接頭・接尾どちらも除去)。
_CORP = r"(?:株式会社|医療法人社団|一般社団法人|一般財団法人|有限会社|合同会社|合資会社)"
_CORP_PREFIX = re.compile(rf"^{_CORP}\s*")
_CORP_SUFFIX = re.compile(rf"\s*{_CORP}$")


def enabled() -> bool:
    return _store.enabled()


def _norm(text: Any) -> str:
    return re.sub(r"\s+", "", unicodedata.normalize("NFKC", str(text or ""))).lower()


def _owner_key(value: Any) -> str:
    s = unicodedata.normalize("NFKC", str(value or ""))
    s = _CORP_PREFIX.sub("", s)
    s = _CORP_SUFFIX.sub("", s)
    return re.sub(r"[\s　]", "", s).lower()


def _glossary():
    """社内用語集 Glossary(会社別名 / 正式名称↔社内略称)。読めなければ None(fail-open)。"""
    try:
        from src.rag.extract import glossary
        return glossary.load()
    except Exception:  # noqa: BLE001
        return None


# --------------------------------------------------------------------------- document resolution
def _stem_variants(doc_name: str, glossary) -> set[str]:
    """doc のファイル名ステムと、その正式名称→社内略称置換版(例 提案書_final→PP_final)の正規化キー集合。"""
    stem = str(doc_name or "").rsplit(".", 1)[0]
    variants = {_norm(stem)}
    if glossary is not None:
        for formal, abbrev in getattr(glossary, "formal_to_abbrev", {}).items():
            if formal and abbrev and formal in stem:
                variants.add(_norm(stem.replace(formal, abbrev)))
    return {v for v in variants if len(v) >= 3}


def _project_rows(question: str, qn: str, rows: list[dict[str, Any]], glossary) -> list[dict[str, Any]]:
    """質問が名指す案件の pptx 行に絞る。Glossary の company_of 優先、無ければ owner-key 最長一致で一意化。"""
    company = None
    if glossary is not None:
        try:
            company = glossary.company_of(question)
        except Exception:  # noqa: BLE001
            company = None
    if company:
        ck = _owner_key(company)
        hit = [r for r in rows if _owner_key(r.get("project", "")) == ck]
        if hit:
            return hit
    # owner-key 直接一致(最長)で一意な案件へ。
    proj_len: dict[str, int] = {}
    for r in rows:
        key = _owner_key(r.get("project", ""))
        if key and key in qn:
            proj_len[key] = len(key)
    if not proj_len:
        return []
    top = max(proj_len.values())
    winners = {k for k, v in proj_len.items() if v == top}
    if len(winners) != 1:
        return []
    win = next(iter(winners))
    return [r for r in rows if _owner_key(r.get("project", "")) == win]


def _resolve_doc(question: str, qn: str, rows: list[dict[str, Any]], glossary) -> "dict[str, Any] | None":
    """質問が一意に名指す pptx 文書レコードを返す(ファイル名ステム/略称展開で一意、曖昧なら None)。"""
    pool = _project_rows(question, qn, rows, glossary)
    pool = pool if pool else rows
    matched = [r for r in pool
               if any(v in qn for v in _stem_variants(str(r.get("doc_name", "")), glossary))]
    if len(matched) == 1:
        return matched[0]
    return None  # 0 件 or 複数 → 曖昧回避


# --------------------------------------------------------------------------- money page selection
def _money_page_slide(doc: dict[str, Any]) -> "dict[str, Any] | None":
    """文書内で「金額提示ページ」に当たるスライドを一意に確定(できなければ None)。

    第一シグナル: タイトルに費用/見積/価格キーワードを持ち かつ 金額トークンを含むスライドがちょうど 1 枚。
    それが無い/複数の時は、金額トークン密度の厳密単独 argmax(>0)を採る。どちらでも一意化できなければ None。
    """
    slides = [s for s in doc.get("slides", []) if isinstance(s, dict)]
    if not slides:
        return None
    money_slides = [s for s in slides if int(s.get("money_token_count", 0) or 0) > 0]
    if not money_slides:
        return None
    # (1) 価格タイトル ∧ 金額トークンあり が一意。
    titled = [s for s in money_slides if _PRICE_TITLE.search(_norm(s.get("title", "")))]
    if len(titled) == 1:
        return titled[0]
    # (2) 金額トークン密度の厳密単独 argmax。
    top = max(int(s.get("money_token_count", 0) or 0) for s in money_slides)
    arg = [s for s in money_slides if int(s.get("money_token_count", 0) or 0) == top]
    if len(arg) == 1:
        return arg[0]
    return None


def _page_of(slide: dict[str, Any]) -> "int | None":
    """スライドの回答ページ: 可視頁番号優先、無ければ印字ページ規約(物理 1-based=slide_index)。"""
    vp = slide.get("visible_page_number")
    if isinstance(vp, int) and vp > 0:
        return vp
    si = slide.get("slide_index")
    return si if isinstance(si, int) and si > 0 else None


def _result(page: int, *, doc: dict[str, Any], slide: dict[str, Any]) -> dict[str, Any]:
    ev = {"store": "pptx_money_page", "provenance": "precomputed (question-independent)",
          "doc_id": doc.get("doc_id"), "doc_name": doc.get("doc_name"),
          "slide_index": slide.get("slide_index"), "money_token_count": slide.get("money_token_count"),
          "slide_title": slide.get("title"), "printed_page": page,
          "visible_page_number": slide.get("visible_page_number"), "slide_count": doc.get("slide_count")}
    method = {"engine": "pptx_money_page", "contract": "fact_lookup",
              "selection": "money_presentation_slide_to_printed_page",
              "naturalize": False, "verified_operand": True, "confidence": 1.0}
    return _contract.ensure_contract({"value": f"{page}ページ", "evidence": ev, "method": method})


def resolve(question: str) -> "dict[str, Any] | None":
    """pptx 金額提示ページの質問を決定論的に束縛、できなければ ``None``(fail-open)。"""
    if not enabled():
        return None
    try:
        qn = _norm(question)
        if not (_MONEY_CUE.search(qn) and _PAGE_ASK.search(qn)):
            return None
        rows = _store.load()
        if not rows:
            return None
        glossary = _glossary()
        doc = _resolve_doc(question, qn, rows, glossary)
        if not doc:
            return None
        slide = _money_page_slide(doc)
        if not slide:
            return None
        page = _page_of(slide)
        if page is None:
            return None
        return _result(page, doc=doc, slide=slide)
    except Exception:  # noqa: BLE001 — a broken lane must fall back, never break the answer path
        return None


# --------------------------------------------------------------------------- investigator tool (補助)
def _pptx_money_page_lookup(document: str) -> dict[str, Any]:
    """pptx 金額提示ページストアを引く: document(ファイル名/略称/案件名の一部)で pptx を特定し、各スライドの
    タイトル・金額トークン数・価格表有無・可視頁番号を返す。file_grep 反復不要で「金額提示は何ページ」型に答える
    材料になる(金額トークン密度が一意に最大のスライド=金額提示ページ)。"""
    rows = _store.load()
    glossary = _glossary()
    dq = _norm(document)
    matched: list[dict[str, Any]] = []
    for r in rows:
        proj = _owner_key(r.get("project", ""))
        if any(v in dq or dq in v for v in _stem_variants(str(r.get("doc_name", "")), glossary)) \
                or (_owner_key(document) and _owner_key(document) in proj):
            matched.append(r)
    if not matched:
        return _contract.make(None, engine="pptx_money_page",
                              evidence={"applicable": True, "document": document, "found": False},
                              note="該当 pptx なし")
    out: list[dict[str, Any]] = []
    for r in matched:
        out.append({"doc_id": r.get("doc_id"), "doc_name": r.get("doc_name"),
                    "slide_count": r.get("slide_count"), "slides": r.get("slides", [])})
    return _contract.make(out, engine="pptx_money_page",
                          evidence={"applicable": True, "document": document, "docs": len(out)},
                          scheme="pptx-money-page-lookup")


def tool() -> "tuple[str, str, dict[str, Any], Callable[..., Any]] | None":
    """pptx 金額提示ページ lookup ツール(OFF の時 None ⇒ ツール集合/MCP surface は byte-identical)。"""
    if not enabled():
        return None
    return (
        PPTX_MONEY_PAGE_LOOKUP,
        "pptx 金額提示ページストア: 全 pptx の各スライドのタイトル/金額トークン数/価格表有無/可視頁番号を"
        "事前計算済みで引く。document にファイル名/略称(PP_final 等)/案件名の一部を渡す。"
        "「◯◯の金額提示は何ページ」型は file_grep を反復せず本ツールを使う(金額トークンが一意に集中する"
        "スライドが金額提示ページ、可視頁番号がそのページ番号)。",
        {"type": "object", "properties": {"document": _STR}, "required": ["document"]},
        _pptx_money_page_lookup,
    )
