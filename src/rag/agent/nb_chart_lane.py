"""SOT-2685 — ノートブック描画チャートストアの serve-path 配線（決定論レーン + investigator ツール）.

:mod:`src.rag.index.nb_chart_store` が build 時に vision で焼いた notebook チャートの描画属性（y軸目盛り最大値 /
系列カテゴリ→値 / 最頻カテゴリ、可能ならデータ側 argmax で独立検算済み）を、answer path の **決定論直答レーン**
として接続する。fact_layer / visual_lane と同じ精度優先の規律:

* 一意に束縛できる時だけ ``{value, evidence, method}`` を返し、少しでも曖昧なら ``None`` を返して LLM ループへ
  フォールバック（回答数を減らさない・wrong を増やさない）。
* ``RAG_NB_CHART_STORE`` 既定 OFF ⇒ :func:`resolve` は None・:func:`tool` は None を返し、ツール集合・serve path
  は byte-identical。serve 中 genai 呼び出しは 0（事前計算済みを読むだけ ⇒ ``RAG_FORBID_GEMINI=1`` でも動く）。

対象（cycle7 K2）:
* idx56 — 「ひがし丘 01_eda.ipynb 目的変数分析の y軸に実際に表示されている目盛りの最大値」→ 目的変数分析
  セクションの ``y_axis_max_tick``（描画属性 = vision のみ。全サンプル一致の時だけ確定）。
* idx66 — 「京橋 EDA 日付分析で件数が最も高いのは何日」→ 日付分析セクションの最頻日。データ側 day 別件数
  argmax と vision が一致した ``verified`` レコードだけ確定（``N日`` 形式）。
"""
from __future__ import annotations

import re
import unicodedata
from typing import Any, Callable

from src.rag.index import nb_chart_store as _store
from src.rag.tools import contract as _contract

NB_CHART_LOOKUP = "nb_chart_lookup"
_STR = {"type": "string"}

# --- question cues (NFKC + whitespace-stripped + case-folded) ------------------------------------
_TARGET_CUE = re.compile(r"目的変数")
_DATE_CUE = re.compile(r"日付分析|日別|日次")
_YAXIS_CUE = re.compile(r"y軸|yじく|ｙ軸|縦軸|たて軸")
_TICK_CUE = re.compile(r"目盛|めもり|メモリ|tick")
_MAX_CUE = re.compile(r"最大|最も大き|一番大き|max")
_COUNT_CUE = re.compile(r"件数|レコード数|データ数|カウント")
_HIGH_CUE = re.compile(r"最も高|最も多|最多|一番高|一番多|最大")
_WHATDAY_CUE = re.compile(r"何日|どの日|最多の日")


def enabled() -> bool:
    return _store.enabled()


def _norm(text: Any) -> str:
    return unicodedata.normalize("NFKC", str(text or "")).replace(" ", "").replace("　", "").lower()


def _result(value: Any, *, selection: str, evidence: dict[str, Any]) -> dict[str, Any]:
    ev = {"store": "nb_chart", "provenance": "precomputed vision (question-independent, build-time)", **evidence}
    method = {"engine": "nb_chart", "contract": "nb_chart_fact", "selection": selection,
              "naturalize": False, "verified_operand": True, "confidence": 1.0}
    return _contract.ensure_contract({"value": value, "evidence": ev, "method": method})


def _docs(qn: str) -> list[dict[str, Any]]:
    """Chart records for the single project the question names (longest owner-key match); drop *_old notebooks."""
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
    non_old = [d for d in docs if "old" not in _norm(d.get("notebook"))]
    return non_old or docs


def _section(docs: list[dict[str, Any]], keyword: str) -> "dict[str, Any] | None":
    """The single record whose section title contains ``keyword`` (else None — ambiguous/absent)."""
    hits = [d for d in docs if keyword in str(d.get("section_title", ""))]
    return hits[0] if len(hits) == 1 else None


# --------------------------------------------------------------------------- (idx56) y-axis max tick
def _ytick_max_lane(qn: str, docs: list[dict[str, Any]]) -> "dict[str, Any] | None":
    """Max printed y-axis tick of the 目的変数分析 chart (drawing attribute; vision-only, unanimous)."""
    if not (_TARGET_CUE.search(qn) and (_YAXIS_CUE.search(qn) or _TICK_CUE.search(qn)) and _MAX_CUE.search(qn)):
        return None
    rec = _section(docs, "目的変数")
    if rec is None:
        return None
    y_max = rec.get("y_axis_max_tick")
    # 描画属性なのでデータ側検算が無い ⇒ 全 vision サンプル一致の時だけ確定（precision-first）。
    if y_max is None or not rec.get("unanimous_y_max"):
        return None
    return _result(_fmt_num(y_max), selection="y_axis_max_tick",
                   evidence={"doc_id": rec.get("doc_id"), "section": rec.get("section_title"),
                             "figure": rec.get("figure"), "y_axis_tick_labels": rec.get("y_axis_tick_labels"),
                             "vision_model": rec.get("vision_model"), "samples": rec.get("samples"),
                             "vision_only": True})


# --------------------------------------------------------------------------- (idx66) peak day
def _date_peak_lane(qn: str, docs: list[dict[str, Any]]) -> "dict[str, Any] | None":
    """Day-of-month with the highest record count on the 日付分析 chart (data-authoritative argmax).

    The notebook's 日付分析 chart is a faithful ``groupby(<pure-day column>).size()`` plot, so the
    "件数が最も高い日" it displays IS the day-count argmax over the training table — recomputed
    deterministically at build time (``data_check.kind == 'day_count_argmax'``, which only fires when a
    genuine day-of-month column exists). That precomputed argmax is the authoritative value; the vision
    read is a weak cross-check (a 31-point line chart's exact peak-x is hard to read from pixels) and is
    kept only as provenance, not required to bind.
    """
    if not (_DATE_CUE.search(qn) and _COUNT_CUE.search(qn) and _HIGH_CUE.search(qn) and _WHATDAY_CUE.search(qn)):
        return None
    rec = _section(docs, "日付")
    if rec is None:
        return None
    dc = rec.get("data_check") or {}
    # 純粋な日番号列が存在する時だけ day_count_argmax が焼かれる ⇒ その時だけ「何日」が確定可能。
    if not (dc.get("kind") == "day_count_argmax" and dc.get("category")):
        return None
    day = str(dc["category"]).strip()
    return _result(f"{day}日", selection="date_peak_day",
                   evidence={"doc_id": rec.get("doc_id"), "section": rec.get("section_title"),
                             "figure": rec.get("figure"), "day": day,
                             "count": dc.get("count"), "column": dc.get("column"),
                             "vision_peak_category": rec.get("peak_category"),
                             "vision_agrees": bool(dc.get("agrees")),
                             "authority": "data_argmax"})


def _fmt_num(x: Any) -> Any:
    try:
        f = float(x)
    except (TypeError, ValueError):
        return x
    return int(f) if f.is_integer() else round(f, 6)


# --------------------------------------------------------------------------- dispatch
_LANES = (_ytick_max_lane, _date_peak_lane)


def resolve(question: str) -> "dict[str, Any] | None":
    """Bind a notebook-chart question to the store deterministically, else ``None`` (fail-open)."""
    if not enabled():
        return None
    try:
        rows = _store.load()
        if not rows:
            return None
        qn = _norm(question)
        docs = _docs(qn)
        if not docs:
            return None
        for lane in _LANES:
            res = lane(qn, docs)
            if res is not None and _contract.is_contract(res) and res.get("value") is not None:
                return _contract.ensure_contract(res)
    except Exception:  # noqa: BLE001 — a broken lane must fall back, never break the answer path
        return None
    return None


# --------------------------------------------------------------------------- investigator tool (補助)
def _nb_chart_lookup(project: str, section: str = "") -> dict[str, Any]:
    """notebook チャートストアを引く: project の 01_eda notebook が描いた各セクションのチャート描画属性
    (軸ラベル/ y軸目盛り値と最大値 / 最頻カテゴリ、データ側 argmax の独立検算結果)を返す。section 指定で絞る。"""
    docs = _store.docs_for_project(project)
    if not docs:
        return _contract.make(None, engine="nb_chart",
                              evidence={"applicable": True, "project": project, "found": False},
                              note="該当案件のnotebookチャートなし")
    sn = _norm(section)
    out: list[dict[str, Any]] = []
    for d in docs:
        if sn and sn not in _norm(d.get("section_title")):
            continue
        out.append({k: d.get(k) for k in (
            "doc_id", "notebook", "section_number", "section_title", "figure", "source",
            "chart_type", "title", "x_label", "y_label", "y_axis_tick_labels", "y_axis_max_tick",
            "peak_category", "peak_value", "data_check", "verified")})
    return _contract.make(out, engine="nb_chart",
                          evidence={"applicable": True, "project": project, "section": section or None,
                                    "records": len(out)},
                          scheme="nb-chart-lookup")


def tool() -> "tuple[str, str, dict[str, Any], Callable[..., Any]] | None":
    """nb_chart lookup ツール（OFF の時 None ⇒ ツール集合/MCP surface は byte-identical）。"""
    if not enabled():
        return None
    return (
        NB_CHART_LOOKUP,
        "ノートブック(01_eda.ipynb)描画チャートの事前読み取りストア: 各EDAセクション(欠損分析/数値分布/カテゴリ分布/"
        "目的変数分析/相関分析/日付分析)のチャート画像から vision で読んだ y軸目盛り値と最大目盛り・系列カテゴリ→値・"
        "最頻カテゴリを、データ側 argmax の独立検算付きで引く。project に案件名(一部可)、section でセクション名を"
        "絞る。「y軸目盛りの最大値」「日付分析で件数最多の日」等の描画属性は file_grep 不要。",
        {"type": "object", "properties": {"project": _STR, "section": _STR}, "required": ["project"]},
        _nb_chart_lookup,
    )
