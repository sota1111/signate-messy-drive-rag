"""SOT-2652 — action-row（AI/タスク行）ストア + serve-path 決定論レーンの offline テスト.

ネットワーク/LLM/実コーパス非依存。合成 OCR テキストで抽出器（パイプ表 / 優先タスク / 優先フォロー）を、
合成ストア（``action_row_store.load`` を monkeypatch）で 2 本の決定論レーン（idx20 担当集合 / idx70 Open
優先フォロー × 会議録完了突合）と精度優先の deferral、RAG_ACTION_ROW_STORE 既定 OFF の byte-identical 挙動、
ツール contract を検証する。
"""
from __future__ import annotations

import pytest

from src.rag.agent import action_row_lane as arl
from src.rag.agent import fact_layer as fl
from src.rag.index import action_row_store as ars
from src.rag.tools import contract as _contract


# =========================================================================== extractor unit tests
_REPORT_TEXT = (
    "[ページ5]\n"
    "6. 次回までの実施事項\n"
    "優先タスク(責任者・期限)\n"
    "データ分析計画書初版作成 (担当：渡辺遥/藤田) →期限：2025-08-21\n"
    "(WBS T03 / MS2 に合わせる)\n"
    "2. 分析環境準備・データ読込確認 (担当：斎藤悠斗) →期限：2025-\n"
    "08-21 (WBS T04 / MS2)\n"
    "3. データ棚卸し・品質確認 (担当：斎藤悠斗、渡辺遥) →期限：2025-08-27\n"
    "[ページ6]\n"
    "4. EDA (欠損・カテゴリ分布) (担当：渡辺\n遥) →期限：2025-08-29\n"
)

_FOLLOW_TEXT = (
    "[ページ1]\n"
    "オープンアクション (抜粋)\n"
    "優先フォロー: AI-03(分析方針レビュー資料、期日 2025-05-26、status: Open)、 "
    "AI-07 (EDA 初版、期日 2025-05-21、status: Open)、AI-05(着手金請求・支払確認、"
    "期日 2025-05-20、status: Open)\n"
    "3. 主要な分析結果\n"
    "総レコード数: 7,352\n"
)

_MINUTES_TEXT = (
    "[ページ2]\n"
    "ID | Action | Owner | Due Date | Status\n"
    "AI-03 | 分析方針レビュー資料作成 | 山本彩乃 | 2025-05-26 | Closed(会議資料として提出)\n"
    "AI-05 | 着手金請求書の発行と支払確認 | 伊藤 翔太 | 2025-05-20 | Open(期日超過:即時フォロー)\n"
    "AI-07 | データプロファイリング初版 | 斎藤悠斗/山本彩乃 | 2025-05-21 | Closed (EDA 出力確認)\n"
)


def test_priority_task_extraction():
    tasks = ars._priority_tasks(_REPORT_TEXT, "doc.pdf")
    by_content = {t["content"]: t for t in tasks}
    assert "分析計画書初版作成" in by_content
    t = by_content["分析計画書初版作成"]
    assert t["owners"] == ["渡辺遥", "藤田"] and t["due"] == "2025-08-21"
    # 内容から先頭の箇条番号は除去される。
    assert "分析環境準備・データ読込確認" in by_content
    # 担当が行またぎ（渡辺\n遥）でも 1 名にまとまる。
    eda = by_content["EDA (欠損・カテゴリ分布)"]
    assert eda["owner_keys"] == [ars._owner_key("渡辺遥")]


def test_priority_follow_extraction_stops_at_section():
    follow = ars._priority_follow(_FOLLOW_TEXT, "doc.pdf")
    ids = [f["id_key"] for f in follow]
    # 3 件すべて（日付中の "26、" で早期停止しない）。
    assert ids == ["ai03", "ai07", "ai05"]
    assert all(f["status_in_report"] == "open" for f in follow)


def test_pipe_row_extraction_status_kind():
    rows = ars._pipe_rows(_MINUTES_TEXT, "min.pdf")
    by_id = {r["id_key"]: r for r in rows}
    assert by_id["ai03"]["status_kind"] == "done"
    assert by_id["ai05"]["status_kind"] == "open"
    assert by_id["ai07"]["status_kind"] == "done"
    assert by_id["ai05"]["owner"] == "伊藤 翔太" and by_id["ai05"]["due"] == "2025-05-20"


def test_norm_id_and_owner_key():
    assert ars.norm_id("AI-05") == ars.norm_id("ai05") == "ai05"
    assert ars.norm_id("A10") == "a10"
    # 法人格は接頭・接尾どちらも除去。
    assert ars._owner_key("白峰信用リスク評価株式会社") == ars._owner_key("白峰信用リスク評価")
    assert ars._owner_key("医療法人社団 蒼樹会 みなみ野") == ars._owner_key("蒼樹会みなみ野")


def test_compute_doc_and_date(monkeypatch):
    class R:
        rel = "プロジェクト/株式会社東都人材プラットフォーム/05.会議/報告資料/報告資料_2025-08-18.pdf"
        project = "株式会社東都人材プラットフォーム"
        category = "meeting"
        name = "報告資料_2025-08-18.pdf"
        ext = "pdf"
        path = "/x"
    monkeypatch.setattr(ars, "_doc_text", lambda ref: _REPORT_TEXT)
    rec = ars.compute_doc(R())
    assert rec is not None and rec["date"] == "2025-08-18"
    assert rec["priority_tasks"]


# =========================================================================== synthetic store + lanes
def _rows():
    east = "プロジェクト/株式会社東都人材プラットフォーム/05.会議/報告資料/報告資料_2025-08-18.pdf"
    rep = "プロジェクト/白峰信用リスク評価株式会社/05.会議/報告資料/報告資料_2025-05-27.pdf"
    minutes = "プロジェクト/白峰信用リスク評価株式会社/05.会議/会議録/会議録_2025-05-27.pdf"
    return [
        {"doc_id": east, "project": "株式会社東都人材プラットフォーム", "category": "meeting",
         "date": "2025-08-18", "doc_name": "報告資料_2025-08-18.pdf",
         "priority_tasks": [
             {"content": "分析計画書初版作成", "owners": ["渡辺遥", "藤田"],
              "owner_keys": [ars._owner_key("渡辺遥"), ars._owner_key("藤田")],
              "due": "2025-08-21", "source": {"doc_id": east, "page": 5}},
             {"content": "データ棚卸し・品質確認", "owners": ["斎藤悠斗", "渡辺遥"],
              "owner_keys": [ars._owner_key("斎藤悠斗"), ars._owner_key("渡辺遥")],
              "due": "2025-08-27", "source": {"doc_id": east, "page": 5}}]},
        {"doc_id": rep, "project": "白峰信用リスク評価株式会社", "category": "meeting",
         "date": "2025-05-27", "doc_name": "報告資料_2025-05-27.pdf",
         "priority_follow": [
             {"id": "AI-03", "id_key": "ai03", "status_in_report": "open", "source": {"doc_id": rep}},
             {"id": "AI-07", "id_key": "ai07", "status_in_report": "open", "source": {"doc_id": rep}},
             {"id": "AI-05", "id_key": "ai05", "status_in_report": "open", "source": {"doc_id": rep}}]},
        {"doc_id": minutes, "project": "白峰信用リスク評価株式会社", "category": "meeting",
         "date": "2025-05-27", "doc_name": "会議録_2025-05-27.pdf",
         "action_rows": [
             {"id": "AI-03", "id_key": "ai03", "content": "分析方針レビュー資料作成",
              "owner": "山本彩乃", "due": "2025-05-26", "status": "Closed", "status_kind": "done",
              "source": {"doc_id": minutes, "page": 2}},
             {"id": "AI-05", "id_key": "ai05", "content": "着手金請求書の発行と支払確認",
              "owner": "伊藤 翔太", "due": "2025-05-20", "status": "Open", "status_kind": "open",
              "source": {"doc_id": minutes, "page": 2}},
             {"id": "AI-07", "id_key": "ai07", "content": "データプロファイリング初版",
              "owner": "斎藤悠斗", "due": "2025-05-21", "status": "Closed", "status_kind": "done",
              "source": {"doc_id": minutes, "page": 2}}]},
    ]


@pytest.fixture
def store(monkeypatch):
    monkeypatch.setattr(ars, "load", lambda path=None: _rows())
    monkeypatch.setenv("RAG_ACTION_ROW_STORE", "1")
    monkeypatch.setenv("RAG_FACT_LAYER", "1")
    return None


# --------------------------------------------------------------------------- OFF byte-identical
def test_off_is_inert(monkeypatch):
    monkeypatch.delenv("RAG_ACTION_ROW_STORE", raising=False)
    assert arl.enabled() is False
    assert arl.resolve("東都人材プラットフォームで渡辺遥と藤田彩が担当の優先タスクは?") is None
    assert arl.tool() is None


def test_off_lane_not_in_fact_layer(monkeypatch):
    monkeypatch.setattr(ars, "load", lambda path=None: _rows())
    monkeypatch.setenv("RAG_FACT_LAYER", "1")
    monkeypatch.delenv("RAG_ACTION_ROW_STORE", raising=False)
    assert fl.resolve("東都人材プラットフォームで渡辺遥と藤田彩が担当の優先タスクは?", "simple_lookup") is None
    assert arl.ACTION_ROW_LOOKUP not in {t[0] for t in fl.tools()}


# --------------------------------------------------------------------------- (idx20) priority task by assignee set
def test_priority_task_owner_lane(store):
    r = arl.resolve("東都人材プラットフォームの報告資料_2025-08-18.pdf で、渡辺遥と藤田彩の2人が"
                    "担当となっている優先タスクを抽出してください。")
    assert r is not None and r["value"] == "分析計画書初版作成"
    assert r["method"]["selection"] == "priority_task_by_assignee_set"


def test_priority_task_via_fact_layer(store):
    r = fl.resolve("東都人材プラットフォームで渡辺遥と藤田彩の2人が担当となっている優先タスクは?",
                   "simple_lookup")
    assert r is not None and r["value"] == "分析計画書初版作成"


def test_priority_task_defers_single_owner(store):
    # 単独担当者しか名指さない ⇒ 「2人」条件が立たず defer（精度優先）。
    assert arl.resolve("東都人材プラットフォームで渡辺遥が担当の優先タスクは?") is None


def test_priority_task_defers_wrong_owner_pair(store):
    # 名指し集合とちょうど一致するタスクが無い ⇒ defer。
    assert arl.resolve("東都人材プラットフォームで佐藤健一と藤田彩が担当の優先タスクは?") is None


# --------------------------------------------------------------------------- (idx70) open follow-up not completed
def test_open_followup_not_completed_lane(store):
    r = arl.resolve("白峰信用リスク評価の5月27日の報告資料で Open として優先フォロー対象に挙げられている"
                    "アクションIDの中で、会議録において完了となっていないIDを上げてください。")
    assert r is not None and r["value"] == "AI-05"
    assert r["method"]["selection"] == "open_followup_not_completed_in_minutes"


def test_open_followup_defers_on_date_mismatch(store):
    # 会議録/報告が無い日付を指定 ⇒ 束縛できず defer。
    assert arl.resolve("白峰信用リスク評価の6月17日の報告資料で優先フォロー対象で会議録で完了でないID?") is None


# --------------------------------------------------------------------------- tool contract
def test_tool_lookup_by_action_id(store):
    t = arl.tool()
    assert t is not None and t[0] == arl.ACTION_ROW_LOOKUP
    out = t[3](project="白峰信用リスク評価", action_id="AI-05")
    assert _contract.is_contract(out)
    assert any(h.get("status") == "Open" for h in out["value"])


def test_tool_unknown_project(store):
    out = arl.tool()[3](project="存在しない会社", action_id="AI-01")
    assert out["value"] is None


# --------------------------------------------------------------------------- store schema roundtrip
def test_store_load_schema_roundtrip(tmp_path):
    p = tmp_path / "action_row_store.jsonl"
    ars.write_store(_rows(), path=p)
    ars.reset_cache()
    rows = ars.load(path=p)
    assert len(rows) == 3
    docs = ars.docs_for_project("白峰信用リスク評価", path=p)
    assert len(docs) == 2
