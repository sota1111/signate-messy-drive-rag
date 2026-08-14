"""SOT-2710 (cycle11) — xlsx スケジュール/プラン既存ストアの自動発火 決定論直答レーン.

cycle10 で Sonnet(claude-mcp) LLM 経路（churn リスク）に乗っていた xlsx スケジュール/プラン系 5 idx を、
**既存ストアの serve 自動発火**で決定論昇格する（SOT-2698「証拠は在るが到達不能=自動レーン追加だけで安価
回収」の再適用）。証拠は質問非依存に全数事前計算済みのストアから読むだけ（serve 中 genai 呼び出し 0）:

* **idx2** — 案件のスケジュール xlsx でオレンジにハイライトされた行のタスク名を全列挙。
  ← :mod:`src.rag.index.visual_store` の ``row_highlights``（行単位ハイライト事実）。
* **idx41** — 案件 PLAN で指定担当者が含まれるタスク数。← :mod:`src.rag.index.plan_coverage_store` の
  ``plan_metrics.people[].task_count``（担当者×担当タスク数の全数集計）。
* **idx75** — 提案書のスケジュール案でフェーズ（モデル構築 等）の実施週。
  ← :mod:`src.rag.index.schedule_plan_store` の ``gantt_phase_weeks``（塗り潰しバー割付）。
* **idx89** — スケジュール xlsx のフェーズ No.N で最後に開始するタスク名（フェーズ内 max 開始日）。
  ← ``schedule_plan_store.schedule_rows``。
* **idx90** — スケジュール xlsx のバッファ工数の合計（種別＝バッファ行の工数総和）。
  ← ``schedule_plan_store.buffer_hours_total``。

規律（fact_layer / 他レーンと同じ）: 案件を glossary.company_of で一意束縛でき、属性が一意・厳密に確定する
時だけ ``{value, evidence, method}`` を返し、少しでも曖昧なら ``None`` を返して LLM ループへフォールバック
（回答数を減らさない・wrong を増やさない）。``RAG_SCHEDULE_PLAN_LOOKUP`` 既定 OFF ⇒ :func:`resolve` は None・
:func:`tool` は None（ツール集合/スキーマ/serve path は byte-identical）。gold 値ハードコードなし。
"""
from __future__ import annotations

import re
import unicodedata
from typing import Any, Callable

from src.rag.index import schedule_plan_store as _store
from src.rag.tools import contract as _contract

SCHEDULE_PLAN_LOOKUP = "schedule_plan_lookup"
_STR = {"type": "string"}

# 計画パスワードのヒント文（company_of を社内管理へ誤束縛させるため除去）。
_PASSWORD_HINT = re.compile(r"ファイルに鍵がかかっている場合は社内管理を確認してください。?")

# --- 意図 cue（NFKC・空白除去した質問文に対して判定） -----------------------------------------------
_HILITE_CUE = re.compile(r"ハイライト")
_ROW_CUE = re.compile(r"行")
_TASKNAME_CUE = re.compile(r"タスク名")
_ALL_CUE = re.compile(r"すべて|全て|全部")
_COLORS = ("オレンジ", "黄色", "黄", "赤", "青", "水色", "緑", "紫", "ピンク", "灰")

_ASSIGNEE_CUE = re.compile(r"担当")
_TASKCOUNT_CUE = re.compile(r"タスク")
_COUNT_ASK = re.compile(r"いくつ|何個|何件|数|幾つ")
_PERSON = re.compile(r"([一-龥ぁ-んァ-ヶ]{1,4})さん")

_WEEK_ASK = re.compile(r"第?何週|第[0-9]+週")
_WEEK_DO = re.compile(r"実施|行う|着手|実行")

_PHASE_NO = re.compile(r"フェーズ(?:No|ナンバー|番号)?[.．]?\s*0*([0-9]+)")
_LAST_START = re.compile(r"最後に(?:開始|着手|始ま|スタート)")

_BUFFER_CUE = re.compile(r"バッファ")
_HOURS_CUE = re.compile(r"工数")
_SUM_CUE = re.compile(r"合計|総|全体|総和")


def enabled() -> bool:
    return _store.enabled()


def _norm(text: Any) -> str:
    return re.sub(r"\s+", "", unicodedata.normalize("NFKC", str(text or ""))).lower()


def _company_of(question: str) -> "str | None":
    stripped = _PASSWORD_HINT.sub("", question)
    try:
        from src.rag.extract import glossary
        return glossary.load().company_of(stripped) or None
    except Exception:  # noqa: BLE001
        return None


def _name_key(value: Any) -> str:
    return unicodedata.normalize("NFKC", str(value or "")).replace(" ", "").replace("　", "")


def _result(value: Any, *, contract_type: str, selection: str, evidence: dict[str, Any]) -> dict[str, Any]:
    ev = {"provenance": "precomputed (question-independent)", **evidence}
    method = {"engine": "schedule_plan_lookup", "contract": contract_type, "selection": selection,
              "naturalize": False, "verified_operand": True, "confidence": 1.0}
    return _contract.ensure_contract({"value": value, "evidence": ev, "method": method})


# --------------------------------------------------------------------------- sub-lanes
def _highlight_rows_lane(question: str, qn: str, company: str):
    """idx2: オレンジにハイライトされた行のタスク名を全列挙（visual_store の row_highlights）。"""
    if not (_HILITE_CUE.search(qn) and _ROW_CUE.search(qn) and _TASKNAME_CUE.search(qn)):
        return None
    # 質問が名指す色（row_highlights の color と一致する語）を一意に決定。
    color = next((c for c in _COLORS if _norm(c) in qn), None)
    if color is None:
        return None
    try:
        from src.rag.index import visual_store as _vs
        rows = _vs.load()
    except Exception:  # noqa: BLE001
        return None
    ckey = _name_key(company)
    docs = [r for r in rows if _name_key(r.get("project", "")) == ckey]
    # 質問がファイル名（スケジュール_r2 等）を名指す場合はそのステムで一意化。
    named = [r for r in docs if _norm(str(r.get("doc_name", "")).rsplit(".", 1)[0]) in qn]
    pool = named if named else docs
    if len(pool) != 1:
        return None
    doc = pool[0]
    task_names: list[str] = []
    matched_color = _name_key(color)
    for sh in (doc.get("sheets") or {}).values():
        for rh in sh.get("row_highlights", []):
            if _name_key(rh.get("color", "")) != matched_color:
                continue
            name = next((c.get("value") for c in rh.get("cells", [])
                         if _TASKNAME_CUE.search(_norm(c.get("header", "")))), None)
            if name:
                task_names.append(str(name).strip())
    if not task_names:
        return None
    value = "、".join(task_names)
    return _result(value, contract_type="enumeration", selection="highlight_row_task_names",
                   evidence={"store": "visual_store", "case": company, "doc": doc.get("doc_name"),
                             "color": color, "count": len(task_names), "task_names": task_names})


def _assignee_task_count_lane(question: str, qn: str, company: str):
    """idx41: 指定担当者が含まれるタスク数（plan_coverage_store の people task_count）。"""
    if not (_ASSIGNEE_CUE.search(qn) and _TASKCOUNT_CUE.search(qn) and _COUNT_ASK.search(qn)):
        return None
    m = _PERSON.search(question)
    if not m:
        return None
    surname = _name_key(m.group(1))
    try:
        from src.rag.index import plan_coverage_store as _pcs
        rows = _pcs.load()
    except Exception:  # noqa: BLE001
        return None
    rec = next((r for r in rows if r.get("project") == company), None)
    if rec is None:
        return None
    people = (rec.get("plan_metrics") or {}).get("people") or []
    hits = [p for p in people if _name_key(p.get("name_key") or p.get("name", "")).startswith(surname)]
    if len(hits) != 1:
        return None
    count = hits[0].get("task_count")
    if not isinstance(count, int):
        return None
    return _result(str(count), contract_type="fact_lookup", selection="assignee_task_count",
                   evidence={"store": "plan_coverage_store", "case": company,
                             "assignee": hits[0].get("name"), "task_count": count})


def _phase_week_lane(question: str, qn: str, rec: dict[str, Any]):
    """idx75: フェーズの実施週（gantt_phase_weeks）。フェーズ名が質問に一意で現れる時だけ確定。"""
    if not (_WEEK_ASK.search(qn) and _WEEK_DO.search(qn)):
        return None
    weeks = rec.get("gantt_phase_weeks") or {}
    if not weeks:
        return None
    hits = [(ph, wk) for ph, wk in weeks.items() if _norm(ph) and _norm(ph) in qn]
    if len(hits) != 1:
        return None
    phase, wk = hits[0]
    return _result(f"第{int(wk)}週目", contract_type="fact_lookup", selection="gantt_phase_week",
                   evidence={"store": "schedule_plan_store", "case": rec.get("project"),
                             "phase": phase, "week": int(wk)})


def _phase_last_task_lane(question: str, qn: str, rec: dict[str, Any]):
    """idx89: フェーズ No.N で最後に開始するタスク名（フェーズ内 max 開始日, tie は行順）。"""
    m = _PHASE_NO.search(question) or _PHASE_NO.search(unicodedata.normalize("NFKC", question))
    if not m or not _LAST_START.search(qn) or not _TASKNAME_CUE.search(qn):
        return None
    target = str(int(m.group(1)))
    phase_rows = [r for r in rec.get("schedule_rows", []) if str(r.get("phase_no", "")) == target]
    phase_rows = [r for r in phase_rows if r.get("name") and r.get("start_date")]
    if not phase_rows:
        return None
    # 最後に開始 = 開始日 max。同日タイは行順（元 xlsx の並び）で後方を採る。
    best = None
    for i, r in enumerate(phase_rows):
        keyed = (r.get("start_date", ""), i)
        if best is None or keyed >= best[0]:
            best = (keyed, r)
    if best is None:
        return None
    # タイ検証: 最遅開始日が複数タスクにまたがっても、行順最後を一意採用（決定論）。
    name = best[1]["name"]
    return _result(name, contract_type="fact_lookup", selection="phase_last_start_task",
                   evidence={"store": "schedule_plan_store", "case": rec.get("project"),
                             "phase_no": target, "task_id": best[1].get("id"),
                             "start_date": best[1].get("start_date"), "name": name})


def _buffer_hours_lane(question: str, qn: str, rec: dict[str, Any]):
    """idx90: バッファ工数の合計（buffer_hours_total を裸の '<n>時間' で返す）。"""
    if not (_BUFFER_CUE.search(qn) and _HOURS_CUE.search(qn) and _SUM_CUE.search(qn)):
        return None
    total = rec.get("buffer_hours_total")
    if total is None:
        return None
    n = int(total) if float(total).is_integer() else total
    return _result(f"{n}時間", contract_type="fact_lookup", selection="buffer_hours_total",
                   evidence={"store": "schedule_plan_store", "case": rec.get("project"),
                             "buffer_hours_total": total})


# --------------------------------------------------------------------------- serve entry
def resolve(question: str) -> "dict[str, Any] | None":
    """xlsx スケジュール/プランの決定論直答（束縛できれば contract、曖昧なら None）。OFF なら常に None。"""
    if not enabled():
        return None
    try:
        company = _company_of(question)
        if not company:
            return None
        qn = _norm(question)
        # 案件記録（新ストア）を先に引く（idx75/89/90 が使う）。
        rec = _store.case_record(company)
        result = None
        # visual_store / plan_coverage_store 由来（案件記録に依存しない）。
        for lane in (_highlight_rows_lane, _assignee_task_count_lane):
            result = lane(question, qn, company)
            if result is not None:
                break
        if result is None and rec is not None:
            for lane in (_phase_week_lane, _phase_last_task_lane, _buffer_hours_lane):
                result = lane(question, qn, rec)
                if result is not None:
                    break
    except Exception:  # noqa: BLE001 — 壊れたレーンは fall back、答えパスを壊さない
        return None
    if result is None or not _contract.is_contract(result):
        return None
    normalized = _contract.ensure_contract(result)
    return normalized if normalized.get("value") is not None else None


# --------------------------------------------------------------------------- investigator tool (補助)
def _tool_handler(case: str = "", query: str = "") -> dict[str, Any]:
    """スケジュール/プラン派生値の lookup（補助ツール）。case で案件記録を束縛して事前計算値を返す。"""
    company = _company_of(case or query) or case
    rec = _store.case_record(company)
    if rec is None:
        return _contract.make(None, engine="schedule_plan_lookup",
                              evidence={"applicable": True, "case": case, "found": False},
                              note="該当案件なし")
    value = {
        "project": rec.get("project"),
        "primary_sheet": rec.get("primary_sheet"),
        "buffer_hours_total": rec.get("buffer_hours_total"),
        "gantt_phase_weeks": rec.get("gantt_phase_weeks"),
        "schedule_rows": rec.get("schedule_rows"),
    }
    return _contract.make(value, engine="schedule_plan_lookup",
                          evidence={"applicable": True, "case": rec.get("project"), "found": True},
                          scheme="schedule-plan-lookup")


def tool() -> "tuple[str, str, dict[str, Any], Callable[..., Any]] | None":
    """RAG_SCHEDULE_PLAN_LOOKUP OFF ⇒ None ⇒ ツール集合/関数スキーマ/MCP surface は byte-identical。"""
    if not enabled():
        return None
    return (
        SCHEDULE_PLAN_LOOKUP,
        "xlsx スケジュール/プラン派生: case を渡すと、その案件の事前計算値を返す — スケジュール xlsx の行"
        "（フェーズNo/開始日/工数/種別）、バッファ工数合計、提案書ガントのフェーズ別実施週。フェーズ内で最後に"
        "開始するタスク名・バッファ工数合計・フェーズの実施週を file_grep 反復せず本ツールで引く。",
        {"type": "object", "properties": {"case": _STR, "query": _STR}, "required": []},
        _tool_handler,
    )
