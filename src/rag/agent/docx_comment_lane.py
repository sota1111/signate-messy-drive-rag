"""docx コメントアンカーストアの serve-path 配線（決定論 lookup レーン, SOT-2711 / idx49）.

:mod:`src.rag.index.docx_comment_store` が build 時に質問非依存で焼いた「全 docx のコメント（本文・
アンカー逐語・位置）」を、コメント抽出型の質問（「コメントがついている部分をそのまま抽出」等）への
**決定論直答レーン** として answer path に接続する。

規律（format_facts_lane と同一の精度優先）:

* 質問が project ∧ doc-kind で選んだ文書群のコメントアンカーが **一意** に絞れた時だけ anchor 逐語を返す。
  複数一致・ゼロ一致は ``None`` を返して従来経路へ（無理な回答化をしない）。
* 「すべて/全て抽出」型は文書順に「、」連結で列挙。
* 値は anchor_text の byte-exact 逐語（gold 非依存）。
* ``RAG_DOCX_COMMENT_ANCHOR`` 既定 OFF ⇒ :func:`resolve` は None ⇒ serve path は byte-identical。
"""
from __future__ import annotations

import re
from typing import Any

from src.rag.agent import format_facts_lane as _ffl
from src.rag.index import docx_comment_store as _store
from src.rag.tools import contract as _contract

# コメント抽出型の質問キュー（NFKC 正規化・空白除去済みの質問に対して照合）。
_COMMENT_RE = re.compile(r"コメント|注釈|吹き出し|コメ")
# 「コメントがついている/付与された部分・箇所」等、注釈対象の本文逐語を求める意図。
_INTENT_RE = re.compile(
    r"ついて|付いて|つけられ|付けられ|付与|されている|されてる|箇所|部分|抽出|そのまま|逐語|抜き出|列挙")
# 列挙要求（format_facts_lane と同一規則）。
_ENUM_RE = _ffl._ENUM_RE


def enabled() -> bool:
    return _store.enabled()


def _result(value: Any, *, doc: dict[str, Any], comment: dict[str, Any],
            selection: str, n_hits: int) -> dict[str, Any]:
    ev = {"store": "docx_comment_anchor", "provenance": "precomputed (question-independent)",
          "case": doc.get("project"), "doc_id": doc.get("rel"), "doc_kind": doc.get("doc_kind"),
          "comment_id": comment.get("id"), "comment_author": comment.get("author"),
          "comment_text": comment.get("comment_text"), "locator": comment.get("loc"),
          "n_hits": n_hits}
    method = {"engine": "docx_comment_anchor", "contract": "simple_lookup",
              "selection": selection, "naturalize": False, "verified_operand": True,
              "confidence": 1.0}
    return {"value": value, "evidence": ev, "method": method}


def resolve(question: str) -> "dict[str, Any] | None":
    """コメント抽出型の質問を docx コメントストアで一意/列挙束縛できる時だけ contract を返す（fail-open）。"""
    if not enabled():
        return None
    try:
        rows = _store.load()
        if not rows:
            return None
        qn = _ffl._norm(question)
        if not (_COMMENT_RE.search(qn) and _INTENT_RE.search(qn)):
            return None  # コメント抽出意図が読めない質問はこのレーンの対象外
        project = _ffl._bind_project(qn, rows)
        if project is None:
            return None
        docs = [r for r in rows if r.get("project") == project]
        docs = _ffl._select_docs(qn, docs)
        if not docs:
            return None
        # 選択文書のコメントアンカーを文書順に収集（アンカー逐語が空のコメントは対象外）。
        collected: list[tuple[dict[str, Any], dict[str, Any], str]] = []
        seen: set[str] = set()
        for d in docs:
            for c in d.get("comments", []):
                text = str(c.get("anchor_text", "")).strip()
                if not text or text in seen:
                    continue
                seen.add(text)
                collected.append((d, c, text))
        if not collected:
            return None
        if _ENUM_RE.search(qn):
            value = "、".join(t for _, _, t in collected)
            doc, comment, _ = collected[0]
            res = _result(value, doc=doc, comment=comment,
                          selection="comment_anchor_enumerate", n_hits=len(collected))
        else:
            if len(collected) != 1:
                return None  # 複数の異なるアンカー ⇒ 曖昧 ⇒ 従来経路へ
            doc, comment, value = collected[0]
            res = _result(value, doc=doc, comment=comment,
                          selection="comment_anchor", n_hits=1)
        if _contract.is_contract(res) and res.get("value") is not None:
            return _contract.ensure_contract(res)
    except Exception:  # noqa: BLE001 — 壊れたレーンは fall back、答えパスを壊さない
        return None
    return None
