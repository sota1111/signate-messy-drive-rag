"""複合書式ファクトストアの serve-path 配線（決定論 lookup レーン, SOT-2703 / idx16/71）.

:mod:`src.rag.index.format_facts_store` が build 時に質問非依存で焼いた「全 docx の書式付き run
（highlight/font_color/bold/underline/italic）」を、書式属性抽出型の質問（「黄色ハイライトかつ赤字」
「太字、下線、イタリックのすべて」等）に対する **決定論直答レーン** として answer path に接続する。

規律（report_attr_lane / format_series_lane と同一の精度優先）:

* 質問が指定した属性集合 ∧ project ∧ doc-kind でストアを引き、**一意** に絞れた時だけ ``{value, ...}``
  を返す。複数一致・ゼロ一致は ``None`` を返して従来経路へ（無理な回答化をしない, 回答数を減らさない）。
* 値は run テキストの逐語（gold 非依存）。
* ``RAG_FORMAT_FACTS`` 既定 OFF ⇒ :func:`resolve` は None ⇒ serve path は byte-identical。
  RAG_FACT_LAYER の下位レーンとして fact_layer.resolve の末尾に後置される（投資者ツールサーフェスは非改変）。
"""
from __future__ import annotations

import re
import unicodedata
from typing import Any, Callable

from src.rag.index import format_facts_store as _store
from src.rag.tools import contract as _contract

# --- 属性キュー（NFKC 正規化・空白除去済みの質問に対して照合） ------------------------------------
# ハイライト色: 「黄色ハイライト」「オレンジのマーカー」等。色語 + (色) + ハイライト/マーカー。
_COLOR_WORDS = {
    "黄": "黄", "黄色": "黄", "オレンジ": "オレンジ", "橙": "オレンジ",
    "赤": "赤", "青": "青", "水色": "水色", "緑": "緑", "紫": "紫",
    "ピンク": "ピンク", "黄緑": "黄緑",
}
_HL_RE = re.compile(
    r"(黄色|黄緑|黄|オレンジ|橙|赤|青|水色|緑|紫|ピンク)(?:色)?(?:で|の|に)?"
    r"(?:ハイライト|マーカー|でマーク)")
# フォント色: 「赤字」「赤色の文字」「青字」等。
_FONT_RE = re.compile(
    r"(黄|オレンジ|橙|赤|青|水色|緑|紫|ピンク|黄緑)(?:色)?(?:の)?(?:字|文字)")
# 太字/下線/イタリック（font_emphasis の語彙に合わせる）。
_BOLD_RE = re.compile(r"太字|ボールド|bold")
_UNDERLINE_RE = re.compile(r"下線|アンダーライン|underline")
_ITALIC_RE = re.compile(r"イタリック|斜体|italic|oblique")

# doc-kind 語彙（質問が使う自然表現）。ストア側 _KIND_FOLDERS + 自己同定 _SELFID_PHRASES と対応。
_KIND_CUES = (
    "中間報告", "最終報告", "会議録", "議事録", "報告資料", "報告書",
    "提案書", "契約書", "見積書", "仕様書",
)

_CORP_AFFIX = re.compile(
    r"(株式会社|医療法人社団|医療法人|一般社団法人|一般財団法人|有限会社|合同会社|合資会社|合名会社)")

# 列挙要求（「…をすべて抽出」「全て列挙」等）。すべて/全て/全部 の直後が抽出系動詞の時だけ列挙モード。
# idx11「太字…イタリックの **すべてに該当** する箇所を抽出」は すべて の直後が「に該当」なので非列挙
# （＝装飾の連言、単発 lookup）。この隣接規則で列挙(idx3)と連言(idx11/71)を機械分離する。
_ENUM_RE = re.compile(r"(?:すべて|全て|全部|全ての)(?:を|の)?(?:抽出|列挙|挙げ|書き出|抜き出|列記|洗い出)")
# 日付除外フィルタ（「日付以外」「日付を除」）。
_DATE_EXCL_RE = re.compile(r"日付以外|日付を除|日付は除|日付除")
# 日付らしい span（西暦/和暦の年月日。区切りは - / . 年月日）。列挙時の除外に使う（gold 非依存の構造規則）。
_DATE_RE = re.compile(
    r"^(?:令和|平成|昭和|r|h|s)?\s*\d{1,4}\s*[-/.年]\s*\d{1,2}\s*[-/.月]\s*\d{1,2}\s*日?$")


def enabled() -> bool:
    return _store.enabled()


def _norm(text: Any) -> str:
    return unicodedata.normalize("NFKC", str(text or "")).replace(" ", "").replace("　", "").lower()


# --------------------------------------------------------------------------- 属性パース
def _required_attrs(qn: str) -> dict[str, Any]:
    """質問から要求書式属性を抽出（highlight 色 / font_color 色 / bold / underline / italic）。"""
    req: dict[str, Any] = {}
    m = _HL_RE.search(qn)
    if m:
        req["highlight"] = _COLOR_WORDS.get(m.group(1), m.group(1))
    m = _FONT_RE.search(qn)
    if m:
        req["font_color"] = _COLOR_WORDS.get(m.group(1), m.group(1))
    if _BOLD_RE.search(qn):
        req["bold"] = True
    if _UNDERLINE_RE.search(qn):
        req["underline"] = True
    if _ITALIC_RE.search(qn):
        req["italic"] = True
    return req


def _run_satisfies(attrs: dict[str, Any], req: dict[str, Any]) -> bool:
    """run が要求属性を **すべて** 満たすか（色は family-aware, bool は True 要求）。"""
    from src.rag.tools.format_events import _color_matches
    for key, want in req.items():
        if key in ("highlight", "font_color"):
            if not _color_matches(attrs.get(key), want):
                return False
        else:  # bold / underline / italic
            if not attrs.get(key):
                return False
    return True


# --------------------------------------------------------------------------- project / doc-kind バインド
def _case_tokens(case_id: str) -> list[str]:
    """案件名の識別トークン（法人格接頭/接尾除去 stem ＋ 生 ＋ 4 文字接頭）。"""
    raw = _norm(case_id)
    stem = _norm(_CORP_AFFIX.sub("", str(case_id or "")))
    toks: list[str] = []
    for t in (stem, raw, stem[:4] if len(stem) >= 4 else ""):
        if t and t not in toks:
            toks.append(t)
    return toks


def _bind_project(qn: str, rows: list[dict[str, Any]]) -> "str | None":
    """質問がただ一つの案件を名指す時だけその project を返す（複数一致/不一致は None = 精度優先）。"""
    best_by_proj: dict[str, int] = {}
    for r in rows:
        proj = r.get("project", "")
        best = 0
        for t in _case_tokens(proj):
            if t and t in qn:
                best = max(best, len(t))
        if best:
            best_by_proj[proj] = max(best_by_proj.get(proj, 0), best)
    if not best_by_proj:
        return None
    top = max(best_by_proj.values())
    winners = [p for p, b in best_by_proj.items() if b == top]
    return winners[0] if len(winners) == 1 else None


def _kind_cues(qn: str) -> list[str]:
    return [k for k in _KIND_CUES if k in qn]


def _select_docs(qn: str, docs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """doc-kind キューがあれば被覆最大の文書に絞る（無ければ案件内全 docx）。"""
    cues = _kind_cues(qn)
    if not cues:
        return docs
    scored: list[tuple[int, dict[str, Any]]] = []
    for d in docs:
        kinds = [_norm(k) for k in d.get("doc_kind", [])]
        cov = sum(1 for c in cues if _norm(c) in kinds)
        scored.append((cov, d))
    top = max((s for s, _ in scored), default=0)
    if top == 0:
        return []  # doc-kind を名指したのに一致文書なし ⇒ 従来経路へ
    return [d for s, d in scored if s == top]


# --------------------------------------------------------------------------- 強調span 収集
def _is_date(text: str) -> bool:
    return bool(_DATE_RE.match(unicodedata.normalize("NFKC", str(text or "")).strip()))


def _emph_source(doc: dict[str, Any]) -> list[dict[str, Any]]:
    """強調span の供給元。``emph_spans``（merged, docx/pptx/pdf）優先、無ければ ``runs`` から装飾ありを流用
    （旧スキーマ/合成レコード向けの fail-open）。"""
    spans = doc.get("emph_spans")
    if spans:
        return spans
    return [r for r in doc.get("runs", [])
            if any(r.get("attrs", {}).get(k) for k in ("bold", "underline", "italic"))]


def _collect_emph(docs: list[dict[str, Any]], req: dict[str, Any], *,
                  exclude_dates: bool) -> list[tuple[dict[str, Any], dict[str, Any], str]]:
    """選択文書の強調span から要求装飾を満たすものを文書順で収集（テキスト重複は初出のみ）。"""
    seen: set[str] = set()
    out: list[tuple[dict[str, Any], dict[str, Any], str]] = []
    for d in docs:
        for sp in _emph_source(d):
            if not _run_satisfies(sp.get("attrs", {}), req):
                continue
            text = str(sp.get("text", "")).strip()
            if not text:
                continue
            if exclude_dates and _is_date(text):
                continue
            if text not in seen:
                seen.add(text)
                out.append((d, sp, text))
    return out


# --------------------------------------------------------------------------- 直答
def _result(value: Any, *, selection: str = "composite_format",
            evidence: dict[str, Any]) -> dict[str, Any]:
    ev = {"store": "format_facts", "provenance": "precomputed (question-independent)", **evidence}
    method = {"engine": "format_facts", "contract": "simple_lookup", "selection": selection,
              "naturalize": False, "verified_operand": True, "confidence": 1.0}
    return {"value": value, "evidence": ev, "method": method}


def _resolve_color(qn: str, docs: list[dict[str, Any]],
                   req: dict[str, Any]) -> "dict[str, Any] | None":
    """色述語（highlight/font_color）を含む質問: docx の色付き run から一意束縛（idx16 型）。"""
    hits: list[tuple[dict[str, Any], dict[str, Any]]] = []
    for d in docs:
        for run in d.get("runs", []):
            if _run_satisfies(run.get("attrs", {}), req):
                hits.append((d, run))
    if not hits:
        return None
    distinct: dict[str, tuple[dict[str, Any], dict[str, Any]]] = {}
    for d, run in hits:
        distinct.setdefault(run.get("text", ""), (d, run))
    if len(distinct) != 1:
        return None  # 複数の異なる値 ⇒ 曖昧 ⇒ 従来経路へ
    value = next(iter(distinct))
    if not str(value).strip():
        return None
    doc, run = distinct[value]
    return _result(value, evidence={
        "case": doc.get("project"), "doc_id": doc.get("rel"), "doc_kind": doc.get("doc_kind"),
        "locator": run.get("loc"), "attrs": run.get("attrs"), "required": req, "n_hits": len(hits)})


def _resolve_emphasis(qn: str, docs: list[dict[str, Any]],
                      req: dict[str, Any]) -> "dict[str, Any] | None":
    """装飾述語（太字/下線/イタリックのみ）の質問: merged 強調span から列挙 or 一意束縛。

    * 列挙モード（「…をすべて抽出」）: 要求装飾を満たす span を文書順に「、」連結（日付除外つき）。
    * 単発モード（idx11 PDF / idx71 docx の B∧U∧I 等）: 相異なる値が 1 つに絞れた時だけ逐語直答。
    """
    exclude_dates = bool(_DATE_EXCL_RE.search(qn))
    collected = _collect_emph(docs, req, exclude_dates=exclude_dates)
    if not collected:
        return None
    if _ENUM_RE.search(qn):
        value = "、".join(t for _, _, t in collected)
        doc, sp, _ = collected[0]
        return _result(value, selection="composite_format_enumerate", evidence={
            "case": doc.get("project"), "doc_id": doc.get("rel"), "doc_kind": doc.get("doc_kind"),
            "locator": sp.get("loc"), "required": req, "exclude_dates": exclude_dates,
            "n_items": len(collected)})
    distinct: dict[str, tuple[dict[str, Any], dict[str, Any]]] = {}
    for d, sp, t in collected:
        distinct.setdefault(t, (d, sp))
    if len(distinct) != 1:
        return None
    value = next(iter(distinct))
    doc, sp = distinct[value]
    return _result(value, evidence={
        "case": doc.get("project"), "doc_id": doc.get("rel"), "doc_kind": doc.get("doc_kind"),
        "locator": sp.get("loc"), "attrs": sp.get("attrs"), "required": req,
        "n_hits": len(collected)})


def resolve(question: str) -> "dict[str, Any] | None":
    """書式属性抽出型の質問を複合書式ストアで一意に束縛できる時だけ contract を返す（fail-open）。"""
    if not enabled():
        return None
    try:
        rows = _store.load()
        if not rows:
            return None
        qn = _norm(question)
        req = _required_attrs(qn)
        if not req:
            return None  # 書式述語が読めない質問はこのレーンの対象外
        project = _bind_project(qn, rows)
        if project is None:
            return None
        docs = [r for r in rows if r.get("project") == project]
        docs = _select_docs(qn, docs)
        if not docs:
            return None
        wants_color = ("highlight" in req) or ("font_color" in req)
        res = (_resolve_color(qn, docs, req) if wants_color
               else _resolve_emphasis(qn, docs, req))
        if res is not None and _contract.is_contract(res) and res.get("value") is not None:
            return _contract.ensure_contract(res)
    except Exception:  # noqa: BLE001 — 壊れたレーンは fall back、答えパスを壊さない
        return None
    return None
