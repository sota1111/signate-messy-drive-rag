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


# --------------------------------------------------------------------------- 直答
def _result(value: Any, *, evidence: dict[str, Any]) -> dict[str, Any]:
    ev = {"store": "format_facts", "provenance": "precomputed (question-independent)", **evidence}
    method = {"engine": "format_facts", "contract": "simple_lookup", "selection": "composite_format",
              "naturalize": False, "verified_operand": True, "confidence": 1.0}
    return {"value": value, "evidence": ev, "method": method}


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
        # 全該当文書の書式付き run から要求属性を満たすものを収集。
        hits: list[tuple[dict[str, Any], dict[str, Any]]] = []
        for d in docs:
            for run in d.get("runs", []):
                if _run_satisfies(run.get("attrs", {}), req):
                    hits.append((d, run))
        if not hits:
            return None
        distinct = {}
        for d, run in hits:
            distinct.setdefault(run.get("text", ""), (d, run))
        if len(distinct) != 1:
            return None  # 複数の異なる値 ⇒ 曖昧 ⇒ 従来経路へ（無理な回答化をしない）
        value = next(iter(distinct))
        doc, run = distinct[value]
        if not str(value).strip():
            return None
        res = _result(
            value,
            evidence={"case": doc.get("project"), "doc_id": doc.get("rel"),
                      "doc_kind": doc.get("doc_kind"), "locator": run.get("loc"),
                      "attrs": run.get("attrs"), "required": req,
                      "n_hits": len(hits)})
        if _contract.is_contract(res) and res.get("value") is not None:
            return _contract.ensure_contract(res)
    except Exception:  # noqa: BLE001 — 壊れたレーンは fall back、答えパスを壊さない
        return None
    return None
