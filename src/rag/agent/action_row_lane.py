"""SOT-2652 — action-row（AI/タスク行）ストアの serve-path 配線（決定論レーン + investigator ツール）.

:mod:`src.rag.index.action_row_store` が build 時に質問非依存で焼いた文書×行動行
（アクション行テーブル / 優先タスク一覧 / 優先フォロー注記）を、simple_lookup contract の
**決定論直答レーン**として answer path に接続する。fact_layer / report_attr_lane と同じ精度優先の規律:

* 一意に束縛できる時だけ ``{value, evidence, method}`` を返し、少しでも曖昧なら ``None`` を返して
  LLM ループへフォールバック（回答数を減らさない・wrong を増やさない, SOT-2601 の教訓）。
* Sonnet は fact-layer ツールを確実には選ばないため（SOT-2649 の教訓）、回収の主経路は **決定論レーン**。
  ツール :func:`tool` は補助（別 idx / 別モデル）として同時に公開する。
* ``RAG_ACTION_ROW_STORE`` 既定 OFF ⇒ :func:`resolve` は None・:func:`tool` は None を返し、
  ツール集合・関数スキーマ・serve path は byte-identical。serve 中 genai 呼び出しは 0（事前計算済みを読むだけ）。

対象（cycle4 クラスタA）: idx20（優先タスクの担当2名条件での行選択）/ idx70（Open 優先フォロー AI と
会議録の完了記録の突合）を決定論で回収。idx34/idx93 等はストアを Evidence/ツールとして提供し、束縛できない
問いは棄権のまま（無理な回答化はしない）。
"""
from __future__ import annotations

import re
import unicodedata
from typing import Any, Callable

from src.rag.index import action_row_store as _store
from src.rag.tools import contract as _contract

ACTION_ROW_LOOKUP = "action_row_lookup"
_STR = {"type": "string"}

# --- question anchors (NFKC + whitespace-stripped + case-folded) --------------------------------
_TASK_CUE = re.compile(r"優先タスク|タスク")
_ASSIGNEE_CUE = re.compile(r"担当")
_FOLLOW_CUE = re.compile(r"優先フォロー")
_MEETING_CUE = re.compile(r"会議録")
_DONE_CUE = re.compile(r"完了")
_MMDD = re.compile(r"(\d{1,2})月(\d{1,2})日")
_YMD = re.compile(r"(\d{4})-(\d{1,2})-(\d{1,2})")


def enabled() -> bool:
    return _store.enabled()


def _norm(text: Any) -> str:
    return unicodedata.normalize("NFKC", str(text or "")).replace(" ", "").replace("　", "").lower()


def _result(value: Any, *, selection: str, evidence: dict[str, Any]) -> dict[str, Any]:
    ev = {"store": "action_row", "provenance": "precomputed (question-independent)", **evidence}
    method = {"engine": "action_row", "contract": "simple_lookup", "selection": selection,
              "naturalize": False, "verified_operand": True, "confidence": 1.0}
    return _contract.ensure_contract({"value": value, "evidence": ev, "method": method})


def _project_docs(qn: str, rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """質問が名指す案件の文書レコード群（案件トークンが質問に出現するもの）。曖昧なら全件は返さない。"""
    hits: list[tuple[dict[str, Any], int]] = []
    seen_proj: dict[str, int] = {}
    for r in rows:
        key = _store._owner_key(r.get("project", ""))
        if not key:
            continue
        if key in qn:
            seen_proj[key] = len(key)
    if not seen_proj:
        return []
    # 最長一致の案件のみ（複数案件を同時に名指す質問は対象外＝曖昧回避）。
    top = max(seen_proj.values())
    winners = {k for k, v in seen_proj.items() if v == top}
    if len(winners) != 1:
        return []
    win = next(iter(winners))
    return [r for r in rows if _store._owner_key(r.get("project", "")) == win]


def _wanted_date(qn: str) -> "tuple[int, int] | None":
    """質問中の月日（5月27日 / 2025-05-27）→ (month, day)。無ければ None。"""
    m = _MMDD.search(qn)
    if m:
        return int(m.group(1)), int(m.group(2))
    y = _YMD.search(qn)
    if y:
        return int(y.group(2)), int(y.group(3))
    return None


def _date_matches(doc_date: str | None, md: "tuple[int, int] | None") -> bool:
    if not doc_date or md is None:
        return md is None  # 日付指定が無い質問は日付で絞らない
    dm = _YMD.match(doc_date)
    return bool(dm) and (int(dm.group(2)), int(dm.group(3))) == md


# --------------------------------------------------------------------------- deterministic lanes
def _priority_task_owner_lane(qn: str, docs: list[dict[str, Any]]) -> "dict[str, Any] | None":
    """質問が名指す担当者集合とちょうど一致する優先タスクの内容（原文）を返す（idx20 型）。"""
    if not (_TASK_CUE.search(qn) and _ASSIGNEE_CUE.search(qn)):
        return None
    md = _wanted_date(qn)
    tasks: list[dict[str, Any]] = []
    for d in docs:
        if md is not None and not _date_matches(d.get("date"), md):
            continue
        tasks.extend(d.get("priority_tasks") or [])
    if not tasks:
        return None
    # 全タスクの担当キー全集合のうち、質問に literal 出現するものが名指し集合。
    all_owner_keys = {k for t in tasks for k in (t.get("owner_keys") or [])}
    named = {k for k in all_owner_keys if k and k in qn}
    if len(named) < 2:  # 「2人が担当」= 複数担当条件（単独担当は本レーン対象外）
        return None
    matched = [t for t in tasks if set(t.get("owner_keys") or []) == named]
    if len(matched) != 1:
        return None  # 一致タスクが 0 or 複数 ⇒ 曖昧回避で defer
    t = matched[0]
    content = t.get("content")
    if not content:
        return None
    return _result(content, selection="priority_task_by_assignee_set",
                   evidence={"owners": t.get("owners"), "named_owner_keys": sorted(named),
                             "due": t.get("due"), "source": t.get("source")})


def _open_followup_not_completed_lane(qn: str, docs: list[dict[str, Any]]) -> "dict[str, Any] | None":
    """報告の Open 優先フォローIDのうち、同日会議録で完了になっていないIDを返す（idx70 型）。"""
    if not (_FOLLOW_CUE.search(qn) and _MEETING_CUE.search(qn) and _DONE_CUE.search(qn)):
        return None
    md = _wanted_date(qn)
    # 1) 報告資料の優先フォロー集合（日付一致・status_in_report==open を優先）。
    follow_docs = [d for d in docs if d.get("priority_follow") and _date_matches(d.get("date"), md)]
    if len(follow_docs) != 1:
        return None
    follows = follow_docs[0]["priority_follow"]
    follow_open = [f for f in follows if f.get("status_in_report") == "open"] or follows
    follow_keys = {f["id_key"]: f["id"] for f in follow_open}
    if not follow_keys:
        return None
    # 2) 同日会議録の行動行 status（id_key -> status_kind）。
    meets = [d for d in docs if "会議録" in _norm_name(d) and _date_matches(d.get("date"), md)]
    if len(meets) != 1:
        return None
    status_by_id: dict[str, str] = {}
    for a in meets[0].get("action_rows") or []:
        status_by_id[a["id_key"]] = a.get("status_kind", "unknown")
    # 3) 完了になっていない = 会議録に present かつ status != done。
    not_done = [(k, orig) for k, orig in follow_keys.items()
                if k in status_by_id and status_by_id[k] != "done"]
    if len(not_done) != 1:
        return None  # 0 or 複数 ⇒ defer
    key, orig = not_done[0]
    return _result(orig, selection="open_followup_not_completed_in_minutes",
                   evidence={"report_doc": follow_docs[0].get("doc_id"),
                             "minutes_doc": meets[0].get("doc_id"),
                             "followup_ids": sorted(follow_keys.values()),
                             "minutes_status": {orig: status_by_id.get(key)}})


def _norm_name(doc: dict[str, Any]) -> str:
    return _norm(doc.get("doc_name") or doc.get("doc_id"))


_LANES = (_priority_task_owner_lane, _open_followup_not_completed_lane)


def resolve(question: str) -> "dict[str, Any] | None":
    """simple_lookup 質問を action-row ストアで一意に束縛できる時だけ contract を返す（fail-open）。"""
    if not enabled():
        return None
    try:
        rows = _store.load()
        if not rows:
            return None
        qn = _norm(question)
        docs = _project_docs(qn, rows)
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
def _action_row_lookup(project: str, action_id: str = "") -> dict[str, Any]:
    """action-row ストアを引く: project で案件文書群を、action_id 指定時はそのIDの全出現（内容/担当/期日/
    状態/出典）を返す。空 action_id なら案件のアクション行/優先タスク/優先フォローの一覧を返す。"""
    rows = _store.docs_for_project(project)
    if not rows:
        return _contract.make(None, engine="action_row",
                              evidence={"applicable": True, "project": project, "found": False},
                              note="該当案件なし")
    aid = unicodedata.normalize("NFKC", str(action_id or "")).strip()
    if aid:
        key = _store.norm_id(aid)
        hits = []
        for r in rows:
            for a in r.get("action_rows") or []:
                if a.get("id_key") == key:
                    hits.append({"id": a.get("id"), "content": a.get("content"),
                                 "owner": a.get("owner"), "due": a.get("due"),
                                 "status": a.get("status"), "source": a.get("source")})
            for f in r.get("priority_follow") or []:
                if f.get("id_key") == key:
                    hits.append({"id": f.get("id"), "priority_follow": True,
                                 "status_in_report": f.get("status_in_report"),
                                 "source": f.get("source")})
        return _contract.make(hits or None, engine="action_row",
                              evidence={"applicable": True, "project": project, "action_id": aid,
                                        "found": bool(hits)}, scheme="action-row-lookup")
    summary = [{"doc_id": r.get("doc_id"), "date": r.get("date"),
                "action_rows": len(r.get("action_rows") or []),
                "priority_tasks": [t.get("content") for t in (r.get("priority_tasks") or [])],
                "priority_follow": [f.get("id") for f in (r.get("priority_follow") or [])]}
               for r in rows]
    return _contract.make(summary, engine="action_row",
                          evidence={"applicable": True, "project": project, "found": True},
                          scheme="action-row-index")


def tool() -> "tuple[str, str, dict[str, Any], Callable[..., Any]] | None":
    """action-row lookup ツール（OFF の時 None ⇒ ツール集合/MCP surface は byte-identical）。"""
    if not enabled():
        return None
    return (
        ACTION_ROW_LOOKUP,
        "action-row（AI/タスク行）ストア: スキャンPDFの会議録アクション表・報告資料の優先タスク/優先フォローを"
        "事前計算済みで引く。project に案件名(一部可)、action_id に 'AI-05'/'A10'(区切り非依存)を渡すと"
        "内容(原文)/担当/期日/状態/出典を全出現返す(空なら案件の行一覧)。スキャンPDF本文は事前OCR済みなので"
        "file_grep 反復は不要。",
        {"type": "object", "properties": {"project": _STR, "action_id": _STR}, "required": ["project"]},
        _action_row_lookup,
    )
