"""SOT-2677 — 文書テーブル/全文チャンクの serve 配線（cycle6 K1 の lookup ツール）.

:mod:`src.rag.index.doc_reach_store` が build 時に質問非依存で焼いた per-doc の table store と切り詰め
なし full-text から、cycle6 で「read_office 結果が investigator の 8000 字境界で切り詰められ、末尾の表/
長文へ到達できず予算切れ棄権」だった文書リーチ欠落クラスタを **投資者が呼べる lookup ツール** として
接続する。決定論の直答はしない（idx8/50/99 は表を読んで質問固有の値を答える型で、単一値の厳密束縛には
落ちない）ので resolve() は持たず、ツール到達のみを提供する:

* **doc_table_lookup(case/file, keyword)** — 案件/ファイルの全表を出自(表/スライド/ページ番号)付きで返す。
  keyword があれば caption/セルに一致する表へ絞る。read_office の 8000 字切り詰めを受けない（表を直接返す）。
* **doc_fulltext_search(case/file, keyword)** — 切り詰めなし full-text から keyword 近傍窓を返す。keyword が
  無ければ末尾(8000 字より後ろ)窓を返す。read_office で末尾が落ちる長文の後半へ到達させる。

規律: OFF（``RAG_DOC_REACH_STORE`` 既定）⇒ :func:`tool` は None（ツール集合/関数スキーマ/MCP surface は
byte-identical）。serve 中の追加 LLM 呼び出しは 0（事前計算済みを読むだけ）。曖昧な case は None ではなく
keyword 横断検索（bounded）へフォールバックし、リーチを最大化する（回答数を減らさない）。
"""
from __future__ import annotations

from typing import Any, Callable

from src.rag.index import doc_reach_store as _drs
from src.rag.tools import contract as _contract

DOC_TABLE_LOOKUP = "doc_table_lookup"
DOC_FULLTEXT_SEARCH = "doc_fulltext_search"
_STR = {"type": "string"}

# tool 出力が investigator の _jsonable(max_str=8000/max_items=60) を超えないよう控えめに束ねる。
_MAX_TABLES_OUT = 6
_MAX_ROWS_OUT = 60
_MAX_WINDOWS_OUT = 6
_WINDOW_RADIUS = 600           # keyword 前後の抜粋半径
_TAIL_FROM = 7000              # keyword 無し時に返す「末尾」窓の開始オフセット（8000 切り詰めの手前から）


def enabled() -> bool:
    return _drs.enabled()


# --------------------------------------------------------------------------- doc selection
def _bind_project(text: str) -> str | None:
    try:
        from src.rag.extract import glossary
        return glossary.load().company_of(text)
    except Exception:  # noqa: BLE001
        return None


def _select_docs(store: dict[str, Any], *, case: str, file: str) -> list[dict[str, Any]]:
    """case（会社名）/ file（rel/ファイル名部分一致）で対象文書を選ぶ。両方空なら全文書。"""
    docs: list[dict[str, Any]] = store.get("docs") or []
    if file:
        key = _drs.norm(file)
        hit = [d for d in docs if key in _drs.norm(d.get("rel")) or key in _drs.norm(d.get("name"))]
        if hit:
            return hit
    project = _bind_project(case) if case else None
    if project:
        scoped = store.get("by_project", {}).get(project) or []
        if scoped:
            return scoped
    if case:
        key = _drs.norm(case)
        scoped = [d for d in docs if key in _drs.norm(d.get("project"))
                  or key in _drs.norm(d.get("rel"))]
        if scoped:
            return scoped
    return docs


# --------------------------------------------------------------------------- table lookup
def _table_matches(table: dict[str, Any], key: str) -> bool:
    if not key:
        return True
    if key in _drs.norm(table.get("caption")):
        return True
    for row in table.get("rows") or []:
        if any(key in _drs.norm(c) for c in row):
            return True
    return False


def _table_lookup(case: str = "", keyword: str = "", file: str = "") -> dict[str, Any]:
    store = _drs.load()
    docs = _select_docs(store, case=case, file=file)
    key = _drs.norm(keyword)
    out: list[dict[str, Any]] = []
    for d in docs:
        for t in d.get("tables") or []:
            if not _table_matches(t, key):
                continue
            out.append({
                "file": d.get("rel"), "project": d.get("project"), "locus": t.get("locus"),
                "caption": t.get("caption"), "rows": (t.get("rows") or [])[:_MAX_ROWS_OUT],
            })
            if len(out) >= _MAX_TABLES_OUT:
                break
        if len(out) >= _MAX_TABLES_OUT:
            break
    found = bool(out)
    return _contract.make(
        {"tables": out, "count": len(out)}, engine="doc_reach_store",
        evidence={"applicable": True, "case": case or None, "file": file or None,
                  "keyword": keyword or None, "found": found,
                  "docs_scanned": len(docs)},
        scheme="doc-table", note=None if found else "一致する表なし")


# --------------------------------------------------------------------------- fulltext search
def _windows(text: str, key: str) -> list[dict[str, Any]]:
    """keyword 近傍窓（重複しない）を返す。key 空なら末尾窓（8000 切り詰めの向こう側）を返す。"""
    if not text:
        return []
    if not key:
        if len(text) <= _TAIL_FROM:
            return [{"offset": 0, "excerpt": text}]
        return [{"offset": _TAIL_FROM, "excerpt": text[_TAIL_FROM:_TAIL_FROM + 2 * _WINDOW_RADIUS + 4000]}]
    # norm() は空白除去でオフセットがずれるため、検索は素直な lower() で行う。
    hay = text.lower()
    needle = key
    wins: list[dict[str, Any]] = []
    start = 0
    while len(wins) < _MAX_WINDOWS_OUT:
        pos = hay.find(needle, start)
        if pos < 0:
            break
        a = max(0, pos - _WINDOW_RADIUS)
        b = min(len(text), pos + len(needle) + _WINDOW_RADIUS)
        wins.append({"offset": a, "excerpt": text[a:b]})
        start = b  # 次の窓は現窓の終端から（重複回避）
    return wins


def _fulltext_search(case: str = "", keyword: str = "", file: str = "") -> dict[str, Any]:
    store = _drs.load()
    docs = _select_docs(store, case=case, file=file)
    # keyword は空白除去・NFKC したものを lower 検索に使う（full_text 側は lower のみ）。
    key = _drs.norm(keyword)
    out: list[dict[str, Any]] = []
    for d in docs:
        wins = _windows(d.get("full_text") or "", key)
        if not wins:
            continue
        out.append({"file": d.get("rel"), "project": d.get("project"),
                    "n_chars": d.get("n_chars"), "windows": wins})
        if len(out) >= _MAX_WINDOWS_OUT:
            break
    found = any(doc["windows"] for doc in out)
    return _contract.make(
        {"documents": out, "count": len(out)}, engine="doc_reach_store",
        evidence={"applicable": True, "case": case or None, "file": file or None,
                  "keyword": keyword or None, "found": found, "docs_scanned": len(docs)},
        scheme="doc-fulltext", note=None if found else "一致箇所なし")


# --------------------------------------------------------------------------- tool wiring
def tool() -> "list[tuple[str, str, dict[str, Any], Callable[..., Any]]]":
    """RAG_DOC_REACH_STORE OFF ⇒ [] ⇒ ツール集合/関数スキーマ/MCP surface は byte-identical。"""
    if not enabled():
        return []
    return [
        (
            DOC_TABLE_LOOKUP,
            "文書テーブル参照: docx/pptx/pdf の全表を出自(表番号/スライド番号/ページ番号)と直近見出し付きで"
            "返す。case(会社名の一部)か file(ファイル名の一部)で対象を絞り、keyword を渡すと見出し/セルが一致"
            "する表へ絞れる。read_office は長文の末尾表を 8000 字境界で切り詰めるが、本ツールは表を直接返すので"
            "切り詰められない。『◯◯の比較表/ランキング表/一覧表の値』は read_office 反復より本ツールで引く。",
            {"type": "object", "properties": {"case": _STR, "keyword": _STR, "file": _STR},
             "required": []},
            _table_lookup,
        ),
        (
            DOC_FULLTEXT_SEARCH,
            "文書全文(切り詰めなし)検索: docx/pptx/pdf の full-text から keyword 近傍窓(offset+抜粋)を返す。"
            "keyword を省くと末尾(先頭 8000 字より後ろ)窓を返す。read_office の出力が 8000 字で切れて文書後半・"
            "末尾の記述に到達できない時、本ツールで keyword を渡してその箇所へ直接到達する。case/file で対象を"
            "絞れる（省略時は全文書横断・bounded）。",
            {"type": "object", "properties": {"case": _STR, "keyword": _STR, "file": _STR},
             "required": []},
            _fulltext_search,
        ),
    ]
