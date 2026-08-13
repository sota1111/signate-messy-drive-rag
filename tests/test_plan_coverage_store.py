"""SOT-2692 — 計画・スケジュール表カバレッジ ストア + serve レーンの offline テスト（LLM/network 不要）。

決定論の束縛規律を合成ストア行で固定する: OFF ⇒ None（byte-identical）、idx79=担当者別 想定工数÷担当
タスク数の厳密単独最大（氏名、比率）、idx88=提案書 第N週の実施項目。曖昧束縛（同率首位/該当なし週）は defer。
純関数（工数派生の argmax / 週バケット化）も単体で固定する。
"""
from __future__ import annotations

from src.rag.index import plan_coverage_store as S
from src.rag.agent import plan_coverage_lane as L


# --------------------------------------------------------------------------- pure helpers
def test_name_key_and_float():
    assert S.name_key("池田 直哉") == "池田直哉"
    assert S._to_float("14") == 14.0
    assert S._to_float("52.0時間") == 52.0
    assert S._to_float("") is None
    assert S._to_float("なし") is None


# --------------------------------------------------------------------------- lane binding
def _kaede():
    return {
        "project": "医療法人社団 恒一会 かえで総合病院",
        "plan_metrics": {
            "source": "…/スケジュール.xlsx",
            "people": [
                {"name": "佐藤 健一", "role": "PM", "hours": 32.0, "task_count": 10, "hours_per_task": 3.2},
                {"name": "池田 直哉", "role": "QA", "hours": 14.0, "task_count": 2, "hours_per_task": 7.0},
            ],
            "max_hours_per_task": {"name": "池田 直哉", "role": "QA", "hours": 14.0,
                                   "task_count": 2, "hours_per_task": 7.0},
        },
        "weekly_schedule": None,
    }


def _minamino():
    return {
        "project": "医療法人社団 蒼樹会 みなみ野女性医療センター",
        "plan_metrics": None,
        "weekly_schedule": {
            "source": "…/提案書.pptx",
            "weeks": {"2": ["データ理解・品質診断"], "5": ["解釈・業務示唆整理"], "6": ["最終化・報告"]},
            "items": [],
        },
    }


def _rows():
    return [_kaede(), _minamino()]


def _q79():
    return ("恒一会 かえで総合病院の計画フォルダ内において、データアステル側の担当者のうち、"
            "1タスク当たりの想定工数（想定工数 ÷ 担当タスク数）が最も大きい人のフルネームと、"
            "その1タスク当たりの想定工数を小数第2位で答えてください。"
            "ファイルに鍵がかかっている場合は社内管理を確認してください。")


def _q88():
    return "蒼樹会 みなみ野女性医療センターの提案書内のスケジュール案において、第5週目に実施することになっている項目は何ですか。"


def test_off_is_none(monkeypatch):
    monkeypatch.delenv("RAG_PLAN_COVERAGE", raising=False)
    assert L.resolve(_q79()) is None
    assert L.tool() is None


def test_hours_per_task_argmax(monkeypatch):
    monkeypatch.setenv("RAG_PLAN_COVERAGE", "1")
    monkeypatch.setattr(L._pcs, "load", lambda path=None: _rows())
    res = L.resolve(_q79())
    assert res is not None
    assert res["value"] == "池田 直哉、7.00"


def test_week_item_lookup(monkeypatch):
    monkeypatch.setenv("RAG_PLAN_COVERAGE", "1")
    monkeypatch.setattr(L._pcs, "load", lambda path=None: _rows())
    res = L.resolve(_q88())
    assert res is not None
    assert res["value"] == "解釈・業務示唆整理"


def test_missing_week_defers(monkeypatch):
    monkeypatch.setenv("RAG_PLAN_COVERAGE", "1")
    monkeypatch.setattr(L._pcs, "load", lambda path=None: _rows())
    q = "蒼樹会 みなみ野女性医療センターの提案書内のスケジュール案において、第4週目に実施することになっている項目は何ですか。"
    assert L.resolve(q) is None  # 第4週は resolved バーなし ⇒ defer


def test_password_hint_does_not_hijack_binding(monkeypatch):
    # "社内管理を確認" が company_of を 社内管理共通 に誤束縛させないことを固定。
    monkeypatch.setenv("RAG_PLAN_COVERAGE", "1")
    monkeypatch.setattr(L._pcs, "load", lambda path=None: _rows())
    rec = L._bind_case(_q79(), _rows())
    assert rec is not None
    assert "かえで" in rec["project"]


def test_argmax_tie_defers(monkeypatch):
    monkeypatch.setenv("RAG_PLAN_COVERAGE", "1")
    tie = {
        "source": "x", "people": [
            {"name": "A A", "hours": 10.0, "task_count": 2, "hours_per_task": 5.0},
            {"name": "B B", "hours": 5.0, "task_count": 1, "hours_per_task": 5.0},
        ], "max_hours_per_task": None,
    }
    rec = {"project": "医療法人社団 恒一会 かえで総合病院", "plan_metrics": tie, "weekly_schedule": None}
    monkeypatch.setattr(L._pcs, "load", lambda path=None: [rec])
    assert L.resolve(_q79()) is None
