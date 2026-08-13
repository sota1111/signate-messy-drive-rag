"""SOT-2692 — 計画・スケジュール表カバレッジの serve 配線（決定論直答レーン + investigator ツール）.

:mod:`src.rag.index.plan_coverage_store` が build 時に質問非依存で焼いた per-case の派生メトリクス／週次
スケジュールから、cycle6/8 で「証拠は到達可能・経路が深いだけ」で予算切れ abstain だった 2 型を **決定論
直答レーン** として接続する:

* **idx79 — 担当者別 1タスク当たり想定工数の最大**: ``plan_metrics.max_hours_per_task``（想定工数 ÷ 担当
  タスク数を全数計算した厳密単独最大）を ``氏名、比率(小数第2位)`` で返す。
* **idx88 — 提案書 第N週の実施項目**: ``weekly_schedule.weeks[N]``（ガント週グリッドの resolved バーから
  事前確定した フェーズ名）を返す。

規律（fact_layer / 他レーンと同じ）: 一意・厳密に束縛できる時だけ ``{value, evidence, method}`` を返し、少し
でも曖昧なら ``None`` を返して LLM ループへフォールバック（回答数を減らさない・wrong を増やさない）。
``RAG_PLAN_COVERAGE`` 既定 OFF ⇒ :func:`resolve` は None・:func:`tool` は None（ツール集合/スキーマ/serve
path は byte-identical）。serve 中の追加 LLM 呼び出しは 0（事前計算済みを読むだけ）。
"""
from __future__ import annotations

import re
import unicodedata
from typing import Any, Callable

from src.rag.index import plan_coverage_store as _pcs
from src.rag.tools import contract as _contract

PLAN_COVERAGE_LOOKUP = "plan_coverage_lookup"
_STR = {"type": "string"}

# --- クエリ意図の cue（NFKC 正規化・空白除去した質問文に対して判定） -------------------------------
_PER_TASK_CUE = re.compile(r"1タスク当たり|1タスク当たり|1タスクあたり|1タスク当り|タスク当たり|タスクあたり")
_HOURS_CUE = re.compile(r"想定工数|工数")
_MAX_CUE = re.compile(r"最も大きい|最大|一番大きい|最も多い")
_WEEK_NUM = re.compile(r"第\s*0*(\d+)\s*週")
_WEEK_ITEM_CUE = re.compile(r"実施|項目|スケジュール|行う|やること|何")

# 計画パスワードのヒント文（社内管理を確認…）は company_of を 社内管理共通 に誤束縛させるので除去する。
_PASSWORD_HINT = re.compile(r"ファイルに鍵がかかっている場合は社内管理を確認してください。?")


def enabled() -> bool:
    return _pcs.enabled()


def _norm(text: Any) -> str:
    return unicodedata.normalize("NFKC", str(text or "")).replace(" ", "").replace("　", "").lower()


def _result(value: Any, *, contract_type: str, selection: str, evidence: dict[str, Any]) -> dict[str, Any]:
    ev = {"store": "plan_coverage_store", "provenance": "precomputed (deterministic xlsx/pptx scan)", **evidence}
    method = {"engine": "plan_coverage_store", "contract": contract_type, "selection": selection,
              "naturalize": False, "verified_operand": True, "confidence": 1.0}
    return _contract.ensure_contract({"value": value, "evidence": ev, "method": method})


def _bind_case(question: str, rows: list[dict[str, Any]]) -> dict[str, Any] | None:
    """glossary.company_of で canonical 会社名へ束縛（計画パスワードヒントは除去）。曖昧/未収録 ⇒ None。"""
    stripped = _PASSWORD_HINT.sub("", question)
    try:
        from src.rag.extract import glossary
        company = glossary.load().company_of(stripped)
    except Exception:  # noqa: BLE001
        company = None
    if not company:
        return None
    for rec in rows:
        if rec.get("project") == company:
            return rec
    return None


def _fmt_ratio(value: Any) -> str | None:
    try:
        return f"{float(value):.2f}"
    except (TypeError, ValueError):
        return None


# --------------------------------------------------------------------------- sub-lanes
def _hours_per_task_lane(q: str, rec: dict[str, Any]):
    """idx79: 担当者別 想定工数 ÷ 担当タスク数 の最大（氏名、比率）。"""
    if not (_PER_TASK_CUE.search(q) or (_HOURS_CUE.search(q) and "÷" in q)):
        return None
    if not (_HOURS_CUE.search(q) and _MAX_CUE.search(q)):
        return None
    metrics = rec.get("plan_metrics") or {}
    top = metrics.get("max_hours_per_task")
    if not top or top.get("name") is None or top.get("hours_per_task") is None:
        return None
    ratio = _fmt_ratio(top["hours_per_task"])
    if ratio is None:
        return None
    value = f"{top['name']}、{ratio}"
    return _result(value, contract_type="fact_lookup", selection="max_hours_per_task",
                   evidence={"case": rec.get("project"), "name": top["name"], "role": top.get("role"),
                             "hours": top.get("hours"), "task_count": top.get("task_count"),
                             "hours_per_task": top.get("hours_per_task"),
                             "source": metrics.get("source")})


def _week_item_lane(q: str, rec: dict[str, Any]):
    """idx88: 提案書 第N週の実施項目（ガント週グリッド resolved）。"""
    m = _WEEK_NUM.search(q)
    if not m or not _WEEK_ITEM_CUE.search(q):
        return None
    weekly = rec.get("weekly_schedule") or {}
    weeks = weekly.get("weeks") or {}
    items = weeks.get(str(int(m.group(1))))
    if not items:
        return None
    value = "、".join(items)
    return _result(value, contract_type="document_extract", selection="weekly_schedule_item",
                   evidence={"case": rec.get("project"), "week": int(m.group(1)),
                             "items": items, "source": weekly.get("source")})


# --------------------------------------------------------------------------- serve entry
def resolve(question: str) -> "dict[str, Any] | None":
    """計画・スケジュールカバレッジの決定論直答（束縛できれば contract、曖昧なら None）。OFF なら常に None。"""
    if not enabled():
        return None
    try:
        rows = _pcs.load()
    except Exception:  # noqa: BLE001 — 壊れたストアは fall back、答えパスを壊さない
        return None
    if not rows:
        return None
    try:
        rec = _bind_case(question, rows)
        if rec is None:
            return None
        q = _norm(question)
        for lane in (_hours_per_task_lane, _week_item_lane):
            result = lane(q, rec)
            if result is not None:
                break
        else:
            result = None
    except Exception:  # noqa: BLE001
        return None
    if result is None or not _contract.is_contract(result):
        return None
    normalized = _contract.ensure_contract(result)
    return normalized if normalized.get("value") is not None else None


# --------------------------------------------------------------------------- investigator tool (補助)
def _tool_handler(case: str = "", query: str = "") -> dict[str, Any]:
    """計画カバレッジの lookup（補助ツール）。case でレコードを束縛して事前計算値を返す。"""
    rows = _pcs.load()
    rec = _bind_case(case or query, rows)
    if rec is None and case:
        try:
            from src.rag.extract import glossary
            company = glossary.load().company_of(case) or case
        except Exception:  # noqa: BLE001
            company = case
        rec = next((r for r in rows if r.get("project") == company
                    or company in (r.get("project") or "")), None)
    if rec is None:
        return _contract.make(None, engine="plan_coverage_store",
                              evidence={"applicable": True, "case": case, "found": False},
                              note="該当案件なし")
    metrics = rec.get("plan_metrics") or {}
    weekly = rec.get("weekly_schedule") or {}
    value = {
        "project": rec.get("project"),
        "max_hours_per_task": metrics.get("max_hours_per_task"),
        "people": metrics.get("people"),
        "weeks": weekly.get("weeks"),
    }
    return _contract.make(value, engine="plan_coverage_store",
                          evidence={"applicable": True, "case": rec.get("project"), "found": True},
                          scheme="plan-schedule-coverage")


def tool() -> "tuple[str, str, dict[str, Any], Callable[..., Any]] | None":
    """RAG_PLAN_COVERAGE OFF ⇒ None ⇒ ツール集合/関数スキーマ/MCP surface は byte-identical。"""
    if not enabled():
        return None
    return (
        PLAN_COVERAGE_LOOKUP,
        "計画・スケジュール表カバレッジ: case を渡すと、その案件の事前計算値を返す — "
        "(1) 担当者別 想定工数 ÷ 担当タスク数（計画xlsxのリソース配分×WBS担当者を突合、暗号化は復号済）と "
        "その最大 max_hours_per_task, (2) 提案書スケジュール案の 第N週 → 実施項目 weeks(ガント週グリッド)。"
        "工数派生の argmax・第N週の項目を file_grep 反復せず本ツールで引く。",
        {"type": "object", "properties": {"case": _STR, "query": _STR}, "required": []},
        _tool_handler,
    )
