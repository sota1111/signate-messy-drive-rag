"""SOT-2653 — xlsx 可視事実ストアの serve-path 配線（決定論レーン + investigator ツール）.

:mod:`src.rag.index.visual_store` が build 時に質問非依存で焼いた xlsx の可視事実（チャート系列→カラム
対応 / 静的・条件付きハイライトの全数収蔵 / 色族要約 / 行全体・列全体ハイライト / 相関シート判定）を、
answer path の **決定論直答レーン**として接続する。fact_layer / action_row_lane と同じ精度優先の規律:

* 一意に束縛できる時だけ ``{value, evidence, method}`` を返し、少しでも曖昧なら ``None`` を返して LLM
  ループへフォールバック（回答数を減らさない・wrong を増やさない, SOT-2601 の教訓）。
* Sonnet は fact-layer ツールを確実には選ばないため（SOT-2649 の教訓）、回収の主経路は **決定論レーン**。
  ツール :func:`tool` は補助（別 idx / 別モデル）として同時に公開する。
* ``RAG_VISUAL_STORE`` 既定 OFF ⇒ :func:`resolve` は None・:func:`tool` は None を返し、ツール集合・関数
  スキーマ・serve path は byte-identical。serve 中 genai 呼び出しは 0（事前計算済みを読むだけ）。

対象（cycle4 クラスタB）:
* idx39 — 「Sheet1のグラフ1はどのカラムを可視化したものか」→ chartEx の値系列参照→カラムヘッダ。
* idx65 — 「相関係数シートで黄色ハイライトのセルの条件」→ 可視な相関シートの当色 CF ルールを自然化。
* idx82 — 「WBSでオレンジ行のタスクIDをすべて」→ 行全体ハイライトの ID 列値を列挙。
* idx97 — 「黄色ハイライトが交差する2セルの値の差の絶対値」→ 列全体×行全体ハイライトの交差セル差。

idx42/77（黄セルの pivot 抽出条件）は束縛が曖昧なため決定論化せず、ツール/Evidence として材料提供に留める。
"""
from __future__ import annotations

import re
import unicodedata
from typing import Any, Callable

from src.rag.index import visual_store as _store
from src.rag.tools import contract as _contract

VISUAL_LOOKUP = "visual_lookup"
_STR = {"type": "string"}

# --- question cues (NFKC + whitespace-stripped + case-folded) ------------------------------------
_GRAPH_NO = re.compile(r"(?:グラフ|ちゃーと|ちゃると|chart|graph)([0-9]+)")
_CHART_CUE = re.compile(r"グラフ|ちゃーと|chart|graph")
_VISUALIZE_CUE = re.compile(r"可視化|かし化|表(し|して)|プロット|描(い|画)")
_CORR_CUE = re.compile(r"相関")
_HILITE_CUE = re.compile(r"はいらいと|ハイライト|色|塗")  # NFKC lowercases ハイライト→はいらいと? no; keep literal too
_COND_CUE = re.compile(r"条件")
_ROW_CUE = re.compile(r"行")
_ALL_CUE = re.compile(r"すべて|全て|全部|漏れなく")
_INTERSECT_CUE = re.compile(r"交差|交わ|重な")
_DIFF_CUE = re.compile(r"差")

# Colour words → the coarse family the store emits (a minimal, portable vocabulary; no corpus fact).
_COLOR_WORDS = {
    "黄": "黄", "黄色": "黄", "きいろ": "黄", "yellow": "黄",
    "オレンジ": "オレンジ", "オレンジ色": "オレンジ", "橙": "オレンジ", "橙色": "オレンジ", "orange": "オレンジ",
    "赤": "赤", "赤色": "赤", "あか": "赤", "red": "赤",
    "緑": "緑", "緑色": "緑", "みどり": "緑", "green": "緑",
    "青": "青", "青色": "青", "あお": "青", "blue": "青",
    "水色": "水色", "みずいろ": "水色",
    "紫": "紫", "紫色": "紫", "ピンク": "ピンク",
}
# A colour word ONLY when it sits in a highlight/colour context — so a company name like「青葉」/「青潮」
# (青 = blue) never reads as a colour filter. The lookahead demands 色/ハイライト/マーカー/塗/系 or a
# particle+セル/行/列 immediately after the colour token (precision-first).
_COLOR_CTX = re.compile(
    r"(黄色|オレンジ色|橙色|赤色|緑色|青色|水色|紫色|黄|オレンジ|橙|赤|緑|青|紫|ピンク|"
    r"yellow|orange|red|green|blue)"
    r"(?=色|ハイライト|マーカー|塗|系|(?:に|で|の)(?:ハイライト|マーカー|色|塗|なって|の?セル|行|列))")


def enabled() -> bool:
    return _store.enabled()


def _norm(text: Any) -> str:
    return unicodedata.normalize("NFKC", str(text or "")).replace(" ", "").replace("　", "").lower()


def _result(value: Any, *, selection: str, evidence: dict[str, Any]) -> dict[str, Any]:
    ev = {"store": "visual", "provenance": "precomputed (question-independent)", **evidence}
    method = {"engine": "visual", "contract": "visual_fact", "selection": selection,
              "naturalize": False, "verified_operand": True, "confidence": 1.0}
    return _contract.ensure_contract({"value": value, "evidence": ev, "method": method})


def _wanted_color(question_raw: str) -> "str | None":
    """The colour family the question names in a highlight context (first such mention), or ``None``."""
    m = _COLOR_CTX.search(question_raw)
    if not m:
        return None
    w = m.group(1)
    if w in _COLOR_WORDS:
        return _COLOR_WORDS[w]
    return _COLOR_WORDS.get(w[:-1]) if w.endswith("色") else None


def _docs(qn: str, question_raw: str) -> list[dict[str, Any]]:
    """Visual records the question names (unique project by longest owner-key match); filtered by filename."""
    rows = _store.load()
    proj_len: dict[str, int] = {}
    for r in rows:
        key = _store.owner_key(r.get("project", ""))
        if key and key in qn:
            proj_len[key] = len(key)
    if not proj_len:
        return []
    top = max(proj_len.values())
    winners = {k for k, v in proj_len.items() if v == top}
    if len(winners) != 1:
        return []  # 複数案件を同時に名指す質問は曖昧回避
    win = next(iter(winners))
    docs = [r for r in rows if _store.owner_key(r.get("project", "")) == win]
    # 質問がファイル名を名指すなら絞る（train.xlsx / スケジュール.xlsx …）。
    named = [d for d in docs if _norm(d.get("doc_name")) and _norm(d.get("doc_name")).rsplit(".", 1)[0] in qn]
    return named or docs


def _sheet_named(qn: str, sheets: dict[str, Any]) -> "str | None":
    """A sheet name the question literally mentions (e.g. WBS / Sheet1), else None."""
    hit = [name for name in sheets if _norm(name) and _norm(name) in qn]
    return hit[0] if len(hit) == 1 else None


# --------------------------------------------------------------------------- (idx39) chart column
def _chart_column_lane(qn: str, question_raw: str, docs: list[dict[str, Any]]) -> "dict[str, Any] | None":
    """Which column a named chart visualises → the value series' column header (idx39)."""
    if not (_CHART_CUE.search(qn) and (_VISUALIZE_CUE.search(qn) or "からむ" in qn or "かし" in qn
                                       or "どのから" in qn or "から" in qn or "列" in qn)):
        return None
    m = _GRAPH_NO.search(qn)
    charts = [(d, c) for d in docs for c in d.get("charts", [])]
    if not charts:
        return None
    if m:
        num = m.group(1)
        matched = [(d, c) for (d, c) in charts if c.get("title") and re.search(rf"{num}\b|{num}$", _norm(c["title"]))]
    else:
        matched = charts
    if len(matched) != 1:
        return None
    doc, chart = matched[0]
    headers = [col.get("header") for col in chart.get("columns", []) if col.get("header")]
    if len(headers) != 1:
        return None
    return _result(headers[0], selection="chart_value_column",
                   evidence={"doc_id": doc.get("doc_id"), "chart": chart.get("title"),
                             "member": chart.get("member"), "columns": chart.get("columns")})


# --------------------------------------------------------------------------- (idx65) correlation cond
def _correlation_condition_lane(qn: str, question_raw: str, docs: list[dict[str, Any]]) -> "dict[str, Any] | None":
    """The condition of a coloured highlight on the (displayed) correlation sheet, naturalised (idx65)."""
    if not (_CORR_CUE.search(qn) and _COND_CUE.search(qn) and ("はいらいと" in qn or "ハイライト" in qn or "塗" in qn)):
        return None
    color = _wanted_color(question_raw)
    if not color:
        return None
    hits: list[tuple[dict[str, Any], str, dict[str, Any]]] = []
    for d in docs:
        for sname, s in d.get("sheets", {}).items():
            if s.get("state") != "visible" or not s.get("is_correlation"):
                continue
            for cf in s.get("cf", []):
                if cf.get("color") == color:
                    hits.append((d, sname, cf))
    if len(hits) != 1:
        return None
    doc, sname, cf = hits[0]
    natural = _naturalize_corr_condition(cf)
    if not natural:
        return None
    return _result(natural, selection="correlation_highlight_condition",
                   evidence={"doc_id": doc.get("doc_id"), "sheet": sname, "color": color,
                             "operator": cf.get("operator"), "formula": cf.get("formula"),
                             "raw_condition": cf.get("condition")})


def _naturalize_corr_condition(cf: dict[str, Any]) -> "str | None":
    """Phrase a correlation-sheet CF rule as ``相関係数が … のセル`` (derived from operator + threshold)."""
    op = cf.get("operator")
    formula = cf.get("formula") or []
    if not formula:
        return None
    thr = str(formula[0]).strip()
    phrase = {"lessThan": f"{thr}未満（{thr}より小さい）",
              "lessThanOrEqual": f"{thr}以下",
              "greaterThan": f"{thr}を超える（{thr}より大きい）",
              "greaterThanOrEqual": f"{thr}以上",
              "equal": f"{thr}に等しい", "notEqual": f"{thr}に等しくない"}.get(op or "")
    if op in ("between", "notBetween") and len(formula) >= 2:
        lo, hi = str(formula[0]).strip(), str(formula[1]).strip()
        phrase = f"{lo}〜{hi}の範囲" + ("外" if op == "notBetween" else "")
    if not phrase:
        return None
    return f"相関係数が{phrase}のセル"


# --------------------------------------------------------------------------- (idx82) highlighted-row IDs
_ID_HEADER_RE = re.compile(r"タスク?id|task ?id|wbs|id")


def _highlight_row_ids_lane(qn: str, question_raw: str, docs: list[dict[str, Any]]) -> "dict[str, Any] | None":
    """All ID-column values of the coloured, fully-highlighted rows on a named sheet (idx82)."""
    if not (("はいらいと" in qn or "ハイライト" in qn or "塗" in qn) and _ROW_CUE.search(qn)
            and _ALL_CUE.search(qn) and "id" in qn):
        return None
    color = _wanted_color(question_raw)
    if not color:
        return None
    # Which ID column does the question ask for? Prefer タスクID, else any *ID* header.
    wants_task = "たすくid" in qn or "タスクid" in qn
    candidates: list[tuple[dict[str, Any], str, list[Any]]] = []
    for d in docs:
        sheets = d.get("sheets", {})
        sname = _sheet_named(qn, sheets)
        target_sheets = [sname] if sname else list(sheets)
        for sn in target_sheets:
            s = sheets.get(sn) or {}
            rows = [rh for rh in s.get("row_highlights", []) if rh.get("color") == color]
            if not rows:
                continue
            ids = _row_id_values(rows, prefer_task=wants_task)
            if ids is not None:
                candidates.append((d, sn, ids))
    if len(candidates) != 1:
        return None
    doc, sname, ids = candidates[0]
    if not ids:
        return None
    return _result("、".join(str(x) for x in ids), selection="highlighted_row_ids",
                   evidence={"doc_id": doc.get("doc_id"), "sheet": sname, "color": color,
                             "count": len(ids)})


def _row_id_values(rows: list[dict[str, Any]], *, prefer_task: bool) -> "list[Any] | None":
    """The ID-column value of each highlighted row, in row order (``None`` if no consistent ID column)."""
    def id_col_index(cells: list[dict[str, Any]]) -> "int | None":
        # Prefer a header that literally contains タスクID / ID.
        best = None
        for i, cell in enumerate(cells):
            h = _norm(cell.get("header"))
            if not h:
                continue
            if prefer_task and ("たすくid" in h or "タスクid" in h):
                return i
            if _ID_HEADER_RE.search(h) and best is None:
                best = i
        return best
    out: list[Any] = []
    for rh in sorted(rows, key=lambda r: r.get("row", 0)):
        cells = rh.get("cells") or []
        idx = id_col_index(cells)
        if idx is None:
            return None
        val = cells[idx].get("value")
        if val in (None, ""):
            return None
        out.append(val)
    return out or None


# --------------------------------------------------------------------------- (idx97) intersect diff
def _intersect_diff_lane(qn: str, question_raw: str, docs: list[dict[str, Any]]) -> "dict[str, Any] | None":
    """Absolute difference of the two cells where a highlighted column meets highlighted rows (idx97)."""
    if not (("はいらいと" in qn or "ハイライト" in qn or "塗" in qn) and _INTERSECT_CUE.search(qn)
            and _DIFF_CUE.search(qn)):
        return None
    color = _wanted_color(question_raw)
    results: list[tuple[dict[str, Any], str, list[tuple[str, float]], float]] = []
    for d in docs:
        for sname, s in d.get("sheets", {}).items():
            inter = _intersections(s, color)
            if inter is None:
                continue
            coords, diff = inter
            results.append((d, sname, coords, diff))
    if len(results) != 1:
        return None
    doc, sname, coords, diff = results[0]
    return _result(_fmt_num(diff), selection="highlight_intersection_diff",
                   evidence={"doc_id": doc.get("doc_id"), "sheet": sname,
                             "color": color, "cells": [{"cell": c, "value": v} for c, v in coords]})


def _intersections(sheet: dict[str, Any], color: "str | None") -> "tuple[list[tuple[str, float]], float] | None":
    """The (col-highlight × row-highlight) intersection cells; ``(coords, |diff|)`` iff exactly two numeric."""
    cols = [c for c in sheet.get("col_highlights", []) if color is None or c.get("color") == color]
    rows = [r for r in sheet.get("row_highlights", []) if color is None or r.get("color") == color]
    if not cols or not rows:
        return None
    static = {(it["row"], it["column"]): it for it in sheet.get("static", [])}
    coords: list[tuple[str, float]] = []
    for c in cols:
        for r in rows:
            cell = static.get((r["row"], c["col"]))
            if cell is None:
                continue
            num = _num(cell.get("value"))
            if num is None:
                return None
            coords.append((cell.get("cell"), num))
    if len(coords) != 2:
        return None
    diff = abs(coords[0][1] - coords[1][1])
    return coords, diff


def _num(text: Any) -> "float | None":
    try:
        return float(str(text).replace(",", "").strip())
    except (TypeError, ValueError):
        return None


def _fmt_num(x: float) -> Any:
    return int(x) if float(x).is_integer() else round(x, 6)


# --------------------------------------------------------------------------- dispatch
_LANES = (_chart_column_lane, _correlation_condition_lane, _highlight_row_ids_lane, _intersect_diff_lane)


def resolve(question: str) -> "dict[str, Any] | None":
    """Bind a visual question to the store deterministically, else ``None`` (fail-open)."""
    if not enabled():
        return None
    try:
        rows = _store.load()
        if not rows:
            return None
        qn = _norm(question)
        docs = _docs(qn, question)
        if not docs:
            return None
        for lane in _LANES:
            res = lane(qn, question, docs)
            if res is not None and _contract.is_contract(res) and res.get("value") is not None:
                return _contract.ensure_contract(res)
    except Exception:  # noqa: BLE001 — a broken lane must fall back, never break the answer path
        return None
    return None


# --------------------------------------------------------------------------- investigator tool (補助)
def _visual_lookup(project: str, sheet: str = "", kind: str = "") -> dict[str, Any]:
    """visual ストアを引く: project の xlsx 可視事実（チャート/ハイライト/CF/色族要約/行・列ハイライト）を
    doc×sheet で返す。sheet 指定でそのシートに絞る。kind='chart'/'highlight'/'cf' で種別を絞る。"""
    docs = _store.docs_for_project(project)
    if not docs:
        return _contract.make(None, engine="visual",
                              evidence={"applicable": True, "project": project, "found": False},
                              note="該当案件なし")
    sn = _norm(sheet)
    out: list[dict[str, Any]] = []
    for d in docs:
        entry: dict[str, Any] = {"doc_id": d.get("doc_id"), "doc_name": d.get("doc_name")}
        if kind in ("", "chart"):
            entry["charts"] = d.get("charts", [])
        sheets = {}
        for name, s in d.get("sheets", {}).items():
            if sn and sn not in _norm(name):
                continue
            view = {"state": s.get("state"), "is_correlation": s.get("is_correlation"),
                    "color_summary": s.get("color_summary")}
            if kind in ("", "highlight"):
                view["static"] = s.get("static")
                view["row_highlights"] = [{k: v for k, v in rh.items() if k != "cells"}
                                          for rh in s.get("row_highlights", [])]
                view["col_highlights"] = s.get("col_highlights")
            if kind in ("", "cf"):
                view["cf"] = s.get("cf")
            sheets[name] = view
        if sheets:
            entry["sheets"] = sheets
        out.append(entry)
    return _contract.make(out, engine="visual",
                          evidence={"applicable": True, "project": project, "sheet": sheet or None,
                                    "kind": kind or None, "docs": len(out)},
                          scheme="visual-lookup")


def tool() -> "tuple[str, str, dict[str, Any], Callable[..., Any]] | None":
    """visual lookup ツール（OFF の時 None ⇒ ツール集合/MCP surface は byte-identical）。"""
    if not enabled():
        return None
    return (
        VISUAL_LOOKUP,
        "xlsx 可視事実ストア: チャートの系列→カラムヘッダ対応、静的/条件付き書式ハイライトの全セル(座標/値/"
        "色族/行ラベル/列ヘッダ)、色族ごとの値min/max/範囲、行全体・列全体ハイライト、相関シート判定を事前計算済みで"
        "引く。project に案件名(一部可)、sheet でシート、kind='chart'/'highlight'/'cf' で種別を絞る。"
        "「どのカラムを可視化」「黄色ハイライトの条件」「オレンジ行のID」「交差2セルの差」等は file_grep 不要。",
        {"type": "object", "properties": {"project": _STR, "sheet": _STR, "kind": _STR},
         "required": ["project"]},
        _visual_lookup,
    )
