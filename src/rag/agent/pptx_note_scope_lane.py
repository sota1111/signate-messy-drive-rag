"""SOT-2714 — pptx 発表者ノート「スコープ対象外」✖項目カウント事実ストアの serve-path 配線
（決定論直答レーン + investigator ツール）.

:mod:`src.rag.index.pptx_note_scope_store` が build 時に質問非依存で焼いた
「pptx→{scope_excluded_count, items[]}」を、answer path の**決定論直答レーン**として接続する。
pptx_money_page_lane と同じ精度優先の規律:

* 「スコープ対象外」を数え上げる質問（``スコープ対象外`` cue ∧ 個数を問う cue）で、対象 pptx を案件 + doc-kind
  （提案書）で一意に束縛でき、その文書の ``scope_excluded_count > 0`` の時だけ **裸の整数**（``7``）を返す。
  少しでも曖昧（案件が一意に定まらない / 候補文書が複数 / count=0）なら ``None`` を返して LLM ループへ
  フォールバック（回答数を減らさない・wrong を増やさない）。
* 別名解決は既存 Glossary（社内用語集.docx）の company_of / 略称展開を優先。
* ``RAG_PPTX_NOTE_SCOPE`` 既定 OFF ⇒ :func:`resolve`/:func:`tool` は None ⇒ ツール集合・serve path は
  byte-identical。serve 中 genai 呼び出しは 0（事前計算済みを読むだけ）。

対象（cycle11 abstain）:
* idx27 — 恒一会 かえで総合病院 提案書.pptx の発表者ノート「スコープ対象外」直下 ✖段落 7本 → gold=7。
"""
from __future__ import annotations

import re
import unicodedata
from typing import Any, Callable

from src.rag.index import pptx_note_scope_store as _store
from src.rag.tools import contract as _contract

PPTX_NOTE_SCOPE_LOOKUP = "pptx_note_scope_lookup"
_STR = {"type": "string"}

# 「スコープ対象外」を数える質問の共起シグナル（どちらか欠けたら発火しない）。
_SCOPE_CUE = re.compile(r"スコープ対象外|対象外(?:と|に|の|項目)")
_COUNT_ASK = re.compile(r"いくつ|何(?:項目|個|件|つ)|幾つ|数は|何ですか")
# doc-kind（提案書）: 対象は pptx 提案書ノート。質問がこの語を含む時に doc-kind バインドへ使う。
_PROPOSAL_KIND = re.compile(r"提案書|提案資料")
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


def _glossary():
    """社内用語集 Glossary（会社別名 / 正式名称↔社内略称）。読めなければ None（fail-open）。"""
    try:
        from src.rag.extract import glossary
        return glossary.load()
    except Exception:  # noqa: BLE001
        return None


# --------------------------------------------------------------------------- document resolution
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
    # owner-key 直接一致（最長）で一意な案件へ。案件名の一部（法人格除去後 stem）が質問に現れるものを拾う。
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
    """質問が一意に名指す pptx 文書レコード（scope>0）を返す。曖昧なら None。"""
    pool = _project_rows(question, qn, rows, glossary)
    if not pool:
        return None  # 案件が一意に定まらない → 曖昧回避（rows 全体には広げない）
    # doc-kind（提案書）が質問にあるなら pptx 提案書へ寄せる。
    if _PROPOSAL_KIND.search(qn):
        titled = [r for r in pool if _PROPOSAL_KIND.search(_norm(r.get("doc_name", "")))]
        if titled:
            pool = titled
    scoped = [r for r in pool if int(r.get("scope_excluded_count", 0) or 0) > 0]
    if len(scoped) == 1:
        return scoped[0]
    return None  # 0 件 or 複数 → 曖昧回避


def _result(count: int, *, doc: dict[str, Any]) -> dict[str, Any]:
    ev = {"store": "pptx_note_scope", "provenance": "precomputed (question-independent)",
          "doc_id": doc.get("doc_id"), "doc_name": doc.get("doc_name"),
          "project": doc.get("project"), "scope_excluded_count": count,
          "notes_slide_count": doc.get("notes_slide_count"), "items": doc.get("items", [])}
    method = {"engine": "pptx_note_scope", "contract": "fact_lookup",
              "selection": "notes_scope_excluded_xmark_paragraph_count",
              "naturalize": False, "verified_operand": True, "confidence": 1.0}
    return _contract.ensure_contract({"value": str(count), "evidence": ev, "method": method})


def resolve(question: str) -> "dict[str, Any] | None":
    """pptx ノート「スコープ対象外」項目数の質問を決定論的に束縛、できなければ ``None``（fail-open）。"""
    if not enabled():
        return None
    try:
        qn = _norm(question)
        if not (_SCOPE_CUE.search(qn) and _COUNT_ASK.search(qn)):
            return None
        rows = _store.load()
        if not rows:
            return None
        glossary = _glossary()
        doc = _resolve_doc(question, qn, rows, glossary)
        if not doc:
            return None
        count = int(doc.get("scope_excluded_count", 0) or 0)
        if count <= 0:
            return None
        return _result(count, doc=doc)
    except Exception:  # noqa: BLE001 — a broken lane must fall back, never break the answer path
        return None


# --------------------------------------------------------------------------- investigator tool (補助)
def _pptx_note_scope_lookup(document: str) -> dict[str, Any]:
    """pptx ノート「スコープ対象外」ストアを引く: document（ファイル名/案件名の一部）で pptx を特定し、
    各文書の発表者ノート「スコープ対象外」見出し直下の ✖項目数と項目テキストを返す。file_grep 反復不要で
    「◯◯の提案書でスコープ対象外はいくつ」型に答える材料になる。"""
    rows = _store.load()
    dq = _norm(document)
    dk = _owner_key(document)
    matched: list[dict[str, Any]] = []
    for r in rows:
        proj = _owner_key(r.get("project", ""))
        name = _norm(r.get("doc_name", ""))
        if (dq and (dq in name or name in dq)) or (dk and dk in proj):
            matched.append(r)
    if not matched:
        return _contract.make(None, engine="pptx_note_scope",
                              evidence={"applicable": True, "document": document, "found": False},
                              note="該当 pptx なし")
    out = [{"doc_id": r.get("doc_id"), "doc_name": r.get("doc_name"), "project": r.get("project"),
            "scope_excluded_count": r.get("scope_excluded_count"), "items": r.get("items", [])}
           for r in matched]
    return _contract.make(out, engine="pptx_note_scope",
                          evidence={"applicable": True, "document": document, "docs": len(out)},
                          scheme="pptx-note-scope-lookup")


def tool() -> "tuple[str, str, dict[str, Any], Callable[..., Any]] | None":
    """pptx ノート スコープ対象外 lookup ツール（OFF の時 None ⇒ ツール集合/MCP surface は byte-identical）。"""
    if not enabled():
        return None
    return (
        PPTX_NOTE_SCOPE_LOOKUP,
        "pptx 発表者ノート「スコープ対象外」項目ストア: 全 pptx の発表者ノートで見出し『スコープ対象外』の"
        "直下に列挙された ✖ 項目段落の数と本文を事前計算済みで引く。document にファイル名/案件名の一部を渡す。"
        "「◯◯の提案書でスコープ対象外の項目はいくつ」型は file_grep を反復せず本ツールを使う"
        "（段落単位で数えるので読点併記は 1 項目）。",
        {"type": "object", "properties": {"document": _STR}, "required": ["document"]},
        _pptx_note_scope_lookup,
    )
