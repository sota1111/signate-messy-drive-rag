"""SOT-2699 — 段階メトリクス フル精度 F1 差 直答レーン（cycle9, idx36）.

:mod:`src.rag.index.analysis_xref_store` の ``staged_metrics`` に SOT-2699 で焼いた **中間 vs 最終 F1 の
全精度差**（``f1_stage_abs_diff``）を読み、idx36 を **決定論直答** する:

    「<案件> において、中間報告時点のF1スコア実測値と最終報告時点のF1スコア実測値の差を絶対値で。」
    例: 恒一会 かえで総合病院 = |metrics.json f1_macro 0.8291582445227382 −
        05.会議/報告資料(interim) f1_macro 0.7329671168078127| = 0.09619112771492555。

**なぜ別レーン/別フラグ**: :mod:`analysis_xref_lane` は idx36 を DELIBERATELY 未配線のまま残す
（中間段階のフル精度が焼けていない前提の honest abstain, SOT-2687）。本レーンはフル精度が **実際に焼けた
案件だけ** 発火する — store が ``f1_stage_abs_diff`` を verified で持つ時のみ。丸め値しか無い案件では
store がそのキーを焼かないので、本レーンも None（honest abstain 維持）。judge はフル精度一致を要求する
（SOT-2687）ので、値は ``repr`` でフル精度のまま返す。

規律（precision-first）: 案件を一意束縛でき、質問が中間×最終×F1×差の型で、store が verified なフル精度差を
持つ時だけ ``{value, evidence, method}`` を返し、少しでも曖昧なら None（LLM ループへ）。
``RAG_STAGED_METRICS`` 既定 OFF ⇒ :func:`resolve`/:func:`tool` は None（serve path は byte-identical）。
"""
from __future__ import annotations

import os
import re
import unicodedata
from typing import Any, Callable

from src.rag.index import analysis_xref_store as _store
from src.rag.tools import contract as _contract

STAGED_METRICS_LOOKUP = "staged_metrics_lookup"
_STR = {"type": "string"}
_ON = {"1", "true", "yes", "on"}

# idx36 型の cue（NFKC 正規化・空白除去・lower した質問文で判定）。
_INTERIM_CUE = re.compile(r"中間報告|中間段階|interim")
_FINAL_CUE = re.compile(r"最終報告|最終段階|final")
_F1_CUE = re.compile(r"f1")
_DIFF_CUE = re.compile(r"差")
_ABS_CUE = re.compile(r"絶対値|絶対")


def enabled() -> bool:
    """serve レーン／ツールを有効にするか。既定 OFF（``RAG_STAGED_METRICS``）⇒ byte-identical。"""
    return os.getenv("RAG_STAGED_METRICS", "0").strip().lower() in _ON


def _norm(text: Any) -> str:
    return unicodedata.normalize("NFKC", str(text or "")).replace(" ", "").replace("　", "").lower()


def _result(value: Any, *, selection: str, evidence: dict[str, Any]) -> dict[str, Any]:
    ev = {"store": "analysis_xref_store.staged_metrics",
          "provenance": "precomputed (question-independent, dual-verified full precision)", **evidence}
    method = {"engine": "staged_metrics", "contract": "numeric", "selection": selection,
              "naturalize": False, "verified_operand": True, "confidence": 1.0}
    return _contract.ensure_contract({"value": value, "evidence": ev, "method": method})


def _bind_case(question: str, rows: list[dict[str, Any]]) -> dict[str, Any] | None:
    """用語集 company_of で canonical 会社名（案件フォルダ名）へ一意束縛。曖昧/未収録 ⇒ None。"""
    try:
        from src.rag.extract import glossary
        company = glossary.load().company_of(question)
    except Exception:  # noqa: BLE001
        company = None
    if not company:
        return None
    for rec in rows:
        if rec.get("project") == company:
            return rec
    return None


def _f1_stage_diff_lane(q: str, rec: dict[str, Any]):
    """idx36: 中間 vs 最終 F1 の全精度差（store が verified で焼いている時だけ）。"""
    if not (_INTERIM_CUE.search(q) and _FINAL_CUE.search(q) and _F1_CUE.search(q)
            and _DIFF_CUE.search(q) and _ABS_CUE.search(q)):
        return None
    sm = rec.get("staged_metrics") or {}
    diff = sm.get("f1_stage_abs_diff")
    if diff is None or not sm.get("f1_stage_diff_verified"):
        return None  # フル精度が焼けていない案件は honest abstain（丸めで近似しない, SOT-2687）
    interim = sm.get("interim_f1") or {}
    final_f1 = sm.get("final_f1_macro")
    # serve 側二重検算: repr(store 値) が abs(final - interim) と一致することを確認（fail-closed）。
    try:
        recomputed = abs(float(final_f1) - float(interim.get("value")))
    except (TypeError, ValueError):
        return None
    if repr(recomputed) != repr(diff):
        return None
    value = repr(diff)  # フル精度のまま（judge はフル精度一致を要求, SOT-2687）
    return _result(value, selection="stage_metric_f1_absolute_diff_full_precision",
                   evidence={"case": rec.get("project"),
                             "final_f1": final_f1, "final_source": sm.get("final_f1_source"),
                             "intermediate_f1": interim.get("value"),
                             "intermediate_source": interim.get("source_rel"),
                             "intermediate_stage": "interim", "intermediate_date": interim.get("date"),
                             "formula": "abs(final_f1 - intermediate_f1)"})


# --------------------------------------------------------------------------- serve entry
def resolve(question: str) -> "dict[str, Any] | None":
    """idx36 の決定論直答（束縛できれば contract、曖昧/未焼きなら None）。OFF なら常に None。"""
    if not enabled():
        return None
    try:
        rows = _store.load()
    except Exception:  # noqa: BLE001 — 壊れたストアは fall back、答えパスを壊さない
        return None
    if not rows:
        return None
    try:
        rec = _bind_case(question, rows)
        if rec is None:
            return None
        result = _f1_stage_diff_lane(_norm(question), rec)
    except Exception:  # noqa: BLE001
        return None
    if result is None or not _contract.is_contract(result):
        return None
    normalized = _contract.ensure_contract(result)
    return normalized if normalized.get("value") is not None else None


# --------------------------------------------------------------------------- investigator tool (補助)
def _tool_handler(question: str = "", case: str = "") -> dict[str, Any]:
    res = resolve(question or "")
    if res is not None:
        return res
    rows = _store.load()
    rec = _bind_case(case or question, rows)
    if rec is None:
        return _contract.make(None, engine="staged_metrics",
                              evidence={"applicable": True, "bound": False},
                              note="案件を一意束縛できず（用語集 company_of 未解決）")
    sm = rec.get("staged_metrics") or {}
    value = {"case": rec.get("project"), "final_f1_macro": sm.get("final_f1_macro"),
             "interim_f1": sm.get("interim_f1"), "f1_stage_abs_diff": sm.get("f1_stage_abs_diff"),
             "reports": [{"rel": r.get("rel"), "stage": r.get("stage"), "date": r.get("date")}
                         for r in (sm.get("reports") or [])]}
    return _contract.make(value, engine="staged_metrics",
                          evidence={"applicable": True, "case": rec.get("project"), "found": True},
                          scheme="staged-metrics-lookup")


def tool() -> "tuple[str, str, dict[str, Any], Callable[..., Any]] | None":
    """RAG_STAGED_METRICS OFF ⇒ None ⇒ ツール集合/関数スキーマ/MCP surface は byte-identical。"""
    if not enabled():
        return None
    return (
        STAGED_METRICS_LOOKUP,
        "段階メトリクス フル精度: 案件の 05.会議/報告資料(中間/最終)から抽出したフル精度メトリクス値と、"
        "中間 vs 最終 F1 の全精度差 f1_stage_abs_diff を出典付きで返す。『中間報告時点のF1と最終報告時点の"
        "F1の差(絶対値)』を丸めずに引く。",
        {"type": "object", "properties": {"question": _STR, "case": _STR}, "required": []},
        _tool_handler,
    )
