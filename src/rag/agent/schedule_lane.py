"""SOT-2680 — スケジュール/ID/体制クロス参照の serve 配線（決定論直答レーン + investigator ツール）.

:mod:`src.rag.index.schedule_store` が build 時に質問非依存で焼いた per-case のスケジュール構造から、cycle6 K4 で
「クロス参照クエリ形が無く compute 前に予算切れ」だった 4 型を **決定論直答レーン** として接続する:

* **idx92 — ID 種別×件数の合計**: 案件の非マークダウン出典に発行済みのマイルストーンID/タスクID/アクションID を
  種別ごとに distinct 集計し、その合計を返す（`id_counts.total`）。
* **idx96 — チェックポイント N → タスクID**: `checkpoints[CP{N}].related_task_ids`（CP→直接T ∪ CP→MS→関連T の
  事前確定チェーン）を返す。
* **idx94 — マイルストーン N × 役職 → タスクID**: `ms_tasks[MS{N}]` を案件スコープの roster（name→role）で役職
  フィルタして返す。**役割マッピングは案件ごとに異なる**（roster は必ずその案件の提案書由来）。
* **idx72 — 役職 → 担当タスク数**: roster で役職の氏名を解決し、その担当タスク数を返す。

規律（fact_layer / 他レーンと同じ）: 一意・厳密に束縛できる時だけ ``{value, evidence, method}`` を返し、少しでも
曖昧なら ``None`` を返して LLM ループへフォールバック（回答数を減らさない・wrong を増やさない, SOT-2601）。
``RAG_SCHEDULE_STORE`` 既定 OFF ⇒ :func:`resolve` は None・:func:`tool` は None（ツール集合/スキーマ/serve path は
byte-identical）。serve 中の追加 LLM 呼び出しは 0（事前計算済みを読むだけ）。
"""
from __future__ import annotations

import re
import unicodedata
from typing import Any, Callable

from src.rag.index import schedule_store as _ss
from src.rag.tools import contract as _contract

SCHEDULE_LOOKUP = "schedule_lookup"
_STR = {"type": "string"}

# --- クエリ意図の cue（NFKC 正規化・空白除去した質問文に対して判定） -------------------------------
_ID_KIND_MS = re.compile(r"マイルストーン|ms")
_ID_KIND_TASK = re.compile(r"タスク")
_ID_KIND_ACTION = re.compile(r"アクション")
_COUNT_CUE = re.compile(r"合計|いくつ|何個|何件|幾つ|総数|何種類|数は")
_CHECKPOINT_NUM = re.compile(r"チェックポイント\s*0*(\d+)")
_MS_NUM = re.compile(r"ms\s*0*(\d+)|マイルストーン\s*0*(\d+)")
_TASK_CUE = re.compile(r"タスク")
_ID_TOKEN = re.compile(r"id")


def enabled() -> bool:
    return _ss.enabled()


def _norm(text: Any) -> str:
    return unicodedata.normalize("NFKC", str(text or "")).replace(" ", "").replace("　", "").lower()


def _result(value: Any, *, contract_type: str, selection: str, evidence: dict[str, Any]) -> dict[str, Any]:
    ev = {"store": "schedule_store", "provenance": "precomputed (deterministic xlsx/pptx scan)", **evidence}
    method = {"engine": "schedule_store", "contract": contract_type, "selection": selection,
              "naturalize": False, "verified_operand": True, "confidence": 1.0}
    return _contract.ensure_contract({"value": value, "evidence": ev, "method": method})


# --------------------------------------------------------------------------- case binding
def _bind_case(question: str, rows: list[dict[str, Any]]) -> dict[str, Any] | None:
    """glossary.company_of で canonical 会社名へ束縛し、その 1 レコードを返す（曖昧/未収録 ⇒ None）。"""
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


def _matched_role(q: str, roster: list[dict[str, Any]]) -> "tuple[str, list[str]] | None":
    """質問に現れる roster 役職のうち最長のものを選び、その役職の name_key 群を返す。"""
    roles = sorted({e.get("role", "") for e in roster if e.get("role")}, key=len, reverse=True)
    for role in roles:
        if role and _norm(role) in q:
            keys = [e["name_key"] for e in roster if e.get("role") == role and e.get("name_key")]
            if keys:
                return role, keys
    return None


def _join_ids(ids: list[str]) -> str:
    return "、".join(ids)


# --------------------------------------------------------------------------- sub-lanes
def _id_count_lane(q: str, rec: dict[str, Any]):
    """idx92: マイルストーン/タスク/アクションの 3 種別 ID の distinct 合計。"""
    if not (_ID_KIND_MS.search(q) and _ID_KIND_TASK.search(q) and _ID_KIND_ACTION.search(q)):
        return None
    if not (_COUNT_CUE.search(q) and _ID_TOKEN.search(q)):
        return None
    counts = rec.get("id_counts") or {}
    total = counts.get("total")
    if not isinstance(total, int) or total <= 0:
        return None
    # 3 種別すべてに 1 件以上の裏付けがある時だけ確定（アクション 0 の案件は誤射しない）。
    if not (counts.get("task") and counts.get("milestone") and counts.get("action")):
        return None
    return _result(total, contract_type="numeric", selection="id_kind_total",
                   evidence={"case": rec.get("project"), "counts": counts,
                             "schedule_file": rec.get("schedule_file")})


def _checkpoint_lane(q: str, rec: dict[str, Any]):
    """idx96: チェックポイント N → 関連タスクID（事前確定チェーン）。"""
    m = _CHECKPOINT_NUM.search(q)
    if not m or not _TASK_CUE.search(q):
        return None
    cpid = f"CP{int(m.group(1))}"
    cp = next((c for c in rec.get("checkpoints") or [] if c.get("id") == cpid), None)
    if cp is None:
        return None
    tasks = cp.get("related_task_ids") or []
    if not tasks:
        return None
    return _result(_join_ids(tasks), contract_type="fact_lookup", selection="checkpoint_tasks",
                   evidence={"case": rec.get("project"), "checkpoint": cpid,
                             "content": cp.get("content"), "task_ids": tasks})


def _milestone_role_lane(q: str, rec: dict[str, Any]):
    """idx94: マイルストーン N に紐づくタスクのうち役職フィルタで残るタスクID。"""
    m = _MS_NUM.search(q)
    if not m or not _TASK_CUE.search(q):
        return None
    matched = _matched_role(q, rec.get("roster") or [])
    if matched is None:
        return None
    role, keys = matched
    msid = f"MS{int(m.group(1) or m.group(2))}"
    ms_tasks = (rec.get("ms_tasks") or {}).get(msid) or []
    if not ms_tasks:
        return None
    by_id = {t["id"]: t for t in rec.get("tasks") or []}
    hit = [tid for tid in ms_tasks
           if any(k in keys for k in (by_id.get(tid, {}).get("owner_keys") or []))]
    if not hit:
        return None
    return _result(_join_ids(hit), contract_type="fact_lookup", selection="milestone_role_tasks",
                   evidence={"case": rec.get("project"), "milestone": msid, "role": role,
                             "role_names": [e["name"] for e in rec.get("roster") or []
                                            if e.get("role") == role],
                             "milestone_tasks": ms_tasks, "task_ids": hit})


def _role_task_count_lane(q: str, rec: dict[str, Any]):
    """idx72: 役職の担当タスク数（MS/CP 指定が無い純粋な役職→件数）。"""
    if _MS_NUM.search(q) or _CHECKPOINT_NUM.search(q):
        return None
    if not (_TASK_CUE.search(q) and _COUNT_CUE.search(q)):
        return None
    matched = _matched_role(q, rec.get("roster") or [])
    if matched is None:
        return None
    role, keys = matched
    count = sum(1 for t in rec.get("tasks") or []
                if any(k in keys for k in (t.get("owner_keys") or [])))
    if count <= 0:
        return None
    return _result(count, contract_type="numeric", selection="role_task_count",
                   evidence={"case": rec.get("project"), "role": role,
                             "role_names": [e["name"] for e in rec.get("roster") or []
                                            if e.get("role") == role]})


# --------------------------------------------------------------------------- serve entry
def resolve(question: str) -> "dict[str, Any] | None":
    """スケジュールクロス参照の決定論直答（束縛できれば contract、曖昧なら None）。OFF なら常に None。"""
    if not enabled():
        return None
    try:
        rows = _ss.load()
    except Exception:  # noqa: BLE001 — 壊れたストアは fall back、答えパスを壊さない
        return None
    if not rows:
        return None
    try:
        rec = _bind_case(question, rows)
        if rec is None:
            return None
        q = _norm(question)
        for lane in (_id_count_lane, _checkpoint_lane, _milestone_role_lane, _role_task_count_lane):
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
    """スケジュールクロス参照の lookup（補助ツール）。case でレコードを束縛し query の意図で分岐。"""
    rows = _ss.load()
    rec = None
    if case:
        try:
            from src.rag.extract import glossary
            company = glossary.load().company_of(case) or case
        except Exception:  # noqa: BLE001
            company = case
        rec = next((r for r in rows if r.get("project") == company
                    or company in (r.get("project") or "")), None)
    if rec is None:
        rec = _bind_case(case or query, rows)
    if rec is None:
        return _contract.make(None, engine="schedule_store",
                              evidence={"applicable": True, "case": case, "found": False},
                              note="該当案件なし")
    value = {
        "project": rec.get("project"),
        "id_counts": rec.get("id_counts"),
        "ms_tasks": rec.get("ms_tasks"),
        "checkpoints": [{"id": c.get("id"), "content": c.get("content"),
                         "related_task_ids": c.get("related_task_ids")}
                        for c in rec.get("checkpoints") or []],
        "roster": rec.get("roster"),
        "action_ids": rec.get("action_ids"),
    }
    return _contract.make(value, engine="schedule_store",
                          evidence={"applicable": True, "case": rec.get("project"), "found": True,
                                    "schedule_file": rec.get("schedule_file")},
                          scheme="schedule-cross-reference")


def tool() -> "tuple[str, str, dict[str, Any], Callable[..., Any]] | None":
    """RAG_SCHEDULE_STORE OFF ⇒ None ⇒ ツール集合/関数スキーマ/MCP surface は byte-identical。"""
    if not enabled():
        return None
    return (
        SCHEDULE_LOOKUP,
        "スケジュール/ID/体制クロス参照: case を渡すと、その案件のスケジュール構造を事前計算値で返す — "
        "(1) ID 種別×件数(マイルストーン/タスク/アクションの distinct 合計), (2) マイルストーン→タスクの逆引き "
        "ms_tasks, (3) チェックポイント→関連タスクID(CP→MS→タスクの事前確定チェーン), (4) 案件別 氏名→役職 "
        "roster(提案書体制由来・案件スコープ厳守)。ID 集計・役職での絞り込み・CP/MS からのタスク列挙を "
        "file_grep 反復せず本ツールで引く。",
        {"type": "object", "properties": {"case": _STR, "query": _STR}, "required": []},
        _tool_handler,
    )
