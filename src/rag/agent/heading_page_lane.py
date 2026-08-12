"""SOT-2668 — 見出し→印字ページ locator ストアの serve-path 配線（決定論レーン + investigator ツール）.

:mod:`src.rag.index.heading_page_store` が build 時に質問非依存で焼いた「文書×見出し→印字ページ番号」を、
answer path の **決定論直答レーン**として接続する。fact_layer / visual_lane と同じ精度優先の規律:

* 文書を一意に束縛でき、かつ質問が名指す見出しがちょうど 1 ページに対応する時だけ ``{value,...}`` を返し、
  少しでも曖昧なら ``None`` を返して LLM ループへフォールバック（回答数を減らさない・wrong を増やさない）。
* Sonnet は fact-layer ツールを確実には選ばないため（SOT-2649）、回収の主経路は **決定論レーン**。ツール
  :func:`tool` は補助（別 idx / 別モデル）として同時に公開する。
* ``RAG_HEADING_PAGE_STORE`` 既定 OFF ⇒ :func:`resolve` は None・:func:`tool` は None ⇒ ツール集合・関数
  スキーマ・serve path は byte-identical。serve 中 genai 呼び出しは 0（事前計算済みを読むだけ）。

対象（cycle5 クラスタC5）:
* idx12 — docx「WBS観点の進捗状況」の見出しは何ページ（印字p2, LibreOffice レンダ→フッタ番号）。
* idx18 — 会議ID:M04 の会議録PDFで「進捗サマリ」が記載されているページ番号（印字p2, OCR ページマーカ）。
"""
from __future__ import annotations

import re
import unicodedata
from typing import Any, Callable

from src.rag.index import heading_page_store as _store
from src.rag.tools import contract as _contract

HEADING_PAGE_LOOKUP = "heading_page_lookup"
_STR = {"type": "string"}

# 「何ページ / ページ番号 / ページ目 / 何ページ目」= ページ位置を問う質問。
_PAGE_CUE = re.compile(r"ページ")
_PAGE_ASK = re.compile(r"何ページ|ページ番号|ページ目|ページですか|ページでしょうか|ページに")
# 見出し/節/記載 = locator 意図（company 名の偶発一致を避けるための第二条件）。
_LOCATOR_CUE = re.compile(r"見出し|セクション|節|章|記載され|載っている|書かれている|ある")
# 文書が自称する識別コード（会議ID:M04 → M04）を質問側から拾う。
_ID_IN_Q = re.compile(r"(?:会議|資料|報告|文書|ドキュメント)?ID\s*[:：]?\s*([A-Za-z]-?\d{1,3})")
# 会社法人格（案件名突合キーの正規化: 接頭・接尾どちらも除去）。
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


def _norm_id(value: Any) -> str:
    return re.sub(r"[\s\-_]", "", unicodedata.normalize("NFKC", str(value or ""))).upper()


# --------------------------------------------------------------------------- document resolution
def _resolve_doc(qn: str, question_raw: str, rows: list[dict[str, Any]]) -> "dict[str, Any] | None":
    """質問が一意に名指す文書レコードを返す（ファイル名 → 自称ID+案件 → 案件+種別 の順、曖昧なら None）。"""
    # (1) ファイル名/ステム直指定（"報告資料_2025-07-08.docx" 等）。
    named = [r for r in rows
             if (stem := _norm(str(r.get("doc_name", "")).rsplit(".", 1)[0])) and len(stem) >= 4
             and stem in qn]
    if len(named) == 1:
        return named[0]
    if len(named) > 1:
        return None  # 同名ステム複数 → 曖昧回避

    # (2) 自称ID（会議ID:M04）+ 案件名。ID は文書の doc_ids と区切り非依存で突合。
    id_codes = {_norm_id(m.group(1)) for m in _ID_IN_Q.finditer(question_raw)}
    if id_codes:
        proj_hits = _project_docs(qn, rows)
        pool = proj_hits if proj_hits else rows
        id_match = [r for r in pool
                    if id_codes & {_norm_id(c) for c in r.get("doc_ids", [])}]
        if len(id_match) == 1:
            return id_match[0]
        if len(id_match) > 1:
            return None

    return None


def _project_docs(qn: str, rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """質問が名指す案件（最長 owner-key 一致で一意）に属する文書。曖昧/不在なら []。"""
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


# --------------------------------------------------------------------------- heading → page
def _heading_page(qn: str, doc: dict[str, Any]) -> "tuple[str, int] | None":
    """質問が名指す見出しに一致する記録を、ちょうど 1 ページに対応する時だけ返す。

    doc の各見出し（番号剥がし・正規化）が質問の部分文字列であるものを候補に、最長の見出しを採る。最長候補が
    ちょうど 1 ページに解決する時のみ ``(見出し, 印字ページ)`` を返す（複数ページ/最長タイでページ相違 → None）。
    """
    cands: list[tuple[str, int, str]] = []  # (heading_text, page, key)
    for h in doc.get("headings", []):
        text = str(h.get("text", ""))
        key = _norm(_store.strip_heading_number(text)) or _norm(text)
        if len(key) < 2 or key not in qn:
            continue
        page = h.get("page")
        if isinstance(page, int):
            cands.append((text, page, key))
    if not cands:
        return None
    longest = max(len(k) for (_t, _p, k) in cands)
    top = [(t, p, k) for (t, p, k) in cands if len(k) == longest]
    pages = {p for (_t, p, _k) in top}
    if len(pages) != 1:
        return None  # 最長見出しが複数ページ or 別見出しがタイ → 曖昧回避
    text, page = top[0][0], top[0][1]
    return text, page


def _result(value: Any, *, doc: dict[str, Any], heading: str, page: int) -> dict[str, Any]:
    ev = {"store": "heading_page", "provenance": "precomputed (question-independent)",
          "doc_id": doc.get("doc_id"), "doc_name": doc.get("doc_name"),
          "heading": heading, "printed_page": page, "page_count": doc.get("page_count")}
    method = {"engine": "heading_page", "contract": "fact_lookup", "selection": "heading_to_printed_page",
              "naturalize": False, "verified_operand": True, "confidence": 1.0}
    return _contract.ensure_contract({"value": value, "evidence": ev, "method": method})


def resolve(question: str) -> "dict[str, Any] | None":
    """見出し→印字ページの質問を決定論的に束縛、できなければ ``None``（fail-open）。"""
    if not enabled():
        return None
    try:
        qn = _norm(question)
        if not (_PAGE_CUE.search(qn) and _PAGE_ASK.search(qn) and _LOCATOR_CUE.search(qn)):
            return None
        rows = _store.load()
        if not rows:
            return None
        doc = _resolve_doc(qn, question, rows)
        if not doc:
            return None
        hp = _heading_page(qn, doc)
        if not hp:
            return None
        heading, page = hp
        return _result(f"{page}ページ", doc=doc, heading=heading, page=page)
    except Exception:  # noqa: BLE001 — a broken lane must fall back, never break the answer path
        return None


# --------------------------------------------------------------------------- investigator tool (補助)
def _heading_page_lookup(document: str, heading: str = "") -> dict[str, Any]:
    """見出し→印字ページ ストアを引く: document（ファイル名/案件名/自称ID の一部）で文書を特定し、その全見出しと
    印字ページ番号を返す。heading 指定でその見出しに一致するページ番号へ絞る。file_grep 反復不要で
    「◯◯の見出しは何ページ」「◯◯が記載されているページ」に答える材料になる。"""
    rows = _store.load()
    dq = _norm(document)
    idq = {_norm_id(m.group(1)) for m in _ID_IN_Q.finditer(document)}
    matched: list[dict[str, Any]] = []
    for r in rows:
        stem = _norm(str(r.get("doc_name", "")).rsplit(".", 1)[0])
        proj = _owner_key(r.get("project", ""))
        ids = {_norm_id(c) for c in r.get("doc_ids", [])}
        if (dq and (dq in stem or stem in dq or _owner_key(document) and _owner_key(document) in proj)) \
                or (idq and idq & ids):
            matched.append(r)
    if not matched:
        return _contract.make(None, engine="heading_page",
                              evidence={"applicable": True, "document": document, "found": False},
                              note="該当文書なし")
    hq = _norm(_store.strip_heading_number(heading)) if heading else ""
    out: list[dict[str, Any]] = []
    for r in matched:
        heads = r.get("headings", [])
        if hq:
            heads = [h for h in heads if hq in _norm(_store.strip_heading_number(str(h.get("text", ""))))]
        out.append({"doc_id": r.get("doc_id"), "doc_name": r.get("doc_name"),
                    "doc_ids": r.get("doc_ids"), "page_count": r.get("page_count"),
                    "headings": heads})
    return _contract.make(out, engine="heading_page",
                          evidence={"applicable": True, "document": document,
                                    "heading": heading or None, "docs": len(out)},
                          scheme="heading-page-lookup")


def tool() -> "tuple[str, str, dict[str, Any], Callable[..., Any]] | None":
    """heading→page lookup ツール（OFF の時 None ⇒ ツール集合/MCP surface は byte-identical）。"""
    if not enabled():
        return None
    return (
        HEADING_PAGE_LOOKUP,
        "見出し→印字ページ ストア: docx/pdf の全見出し→印字ページ番号(表紙除外のフッタ番号)を事前計算済みで引く。"
        "document にファイル名/案件名/自称ID(会議ID:M04 等)の一部を、heading に見出し語を渡す。"
        "「◯◯の見出しは何ページ」「◯◯が記載されているページ番号」は file_grep を反復せず本ツールを使う。",
        {"type": "object", "properties": {"document": _STR, "heading": _STR},
         "required": ["document"]},
        _heading_page_lookup,
    )
