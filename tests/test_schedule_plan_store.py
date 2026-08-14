"""SOT-2710 — xlsx スケジュール/プラン自動発火 決定論レーン + 派生ストアの offline テスト.

3 層で検証する:
* **ストア helper 単体**(工数パース / 開始日 ISO 化 / 主シート選定 / タスク行正規化 / ガント塗り潰しバー週割付)。
* **合成ストア**(schedule_plan/visual/plan_coverage の ``load`` を monkeypatch)+ **偽 company_of** で、
  5 型（オレンジ行タスク名 idx2 / 担当者タスク数 idx41 / フェーズ実施週 idx75 / フェーズ内最終開始タスク
  idx89 / バッファ工数合計 idx90）の決定論束縛・精度優先の deferral・RAG_SCHEDULE_PLAN_LOOKUP 既定 OFF の
  byte-identical・ツール contract を検証。ネットワーク/LLM/実コーパス非依存。
* **実コーパス断言**(配置時のみ・未配置は skip): 青潮 バッファ合計 8h / みなみ野 ガント モデル構築=4週。
"""
from __future__ import annotations

import pytest

from src.rag.agent import schedule_plan_lane as L
from src.rag.index import schedule_plan_store as S


# =========================================================================== store helpers (unit)
def test_to_hours_variants():
    assert S._to_hours(2) == 2.0
    assert S._to_hours("2") == 2.0
    assert S._to_hours("2h") == 2.0
    assert S._to_hours("2時間") == 2.0
    assert S._to_hours("") is None
    assert S._to_hours(None) is None
    assert S._to_hours("なし") is None


def test_iso_date_from_string_and_none():
    assert S._iso_date("2025-11-11 00:00:00") == "2025-11-11"
    assert S._iso_date("2025/11/1") == "2025-11-01"
    assert S._iso_date(None) == ""


def test_parse_sheet_rows_carry_down_phase_and_buffer():
    rows = [
        ["タスクID", "フェーズNo.", "フェーズ名", "種別", "タスク名", "開始日", "工数(h)"],
        ["T01", "1", "立上げ", "タスク", "契約発効", "2025-10-01", None],
        ["T02", "", "", "タスク", "キックオフ", "2025-10-01", None],
        ["B01", "", "", "バッファ", "リスクバッファ", "2025-10-05", 2],
        ["T03", "2", "EDA", "タスク", "品質確認", "2025-10-02", None],
    ]
    parsed = S._parse_sheet_rows(rows)
    assert parsed is not None
    # フェーズNo/フェーズ名は結合セル起因の空欄を直近上方でキャリーダウン。
    assert [r["phase_no"] for r in parsed] == ["1", "1", "1", "2"]
    assert parsed[1]["phase_name"] == "立上げ"
    assert parsed[2]["kind"] == "バッファ" and parsed[2]["hours"] == 2.0


def test_parse_sheet_rows_no_header_returns_none():
    assert S._parse_sheet_rows([["氏名", "役職"], ["佐藤", "PM"]]) is None


# =========================================================================== synthetic stores + fake bind
_AOBM = "株式会社青葉バイオメディカル機器"
_KYOBASHI = "京橋信用ソリューションズ株式会社"
_AOSHIO = "株式会社青潮モビリティサービス"
_MINAMINO = "医療法人社団 蒼樹会 みなみ野女性医療センター"
_AOMINE = "株式会社青嶺不動産アセットマネジメント"


def _sched_rows():
    return [
        {"project": _KYOBASHI, "primary_sheet": "WBSタスク一覧", "buffer_hours_total": None,
         "gantt_phase_weeks": {}, "schedule_rows": [
             {"id": "T23", "kind": "", "phase_no": "6", "phase_name": "最終成果物化",
              "name": "最終モデル確定・再評価", "start_date": "2025-11-01", "hours": None},
             {"id": "T26", "kind": "", "phase_no": "6", "phase_name": "最終成果物化",
              "name": "品質保証レビュー・リスクバッファ", "start_date": "2025-11-08", "hours": None},
             {"id": "T27", "kind": "", "phase_no": "6", "phase_name": "最終成果物化",
              "name": "最終報告・成果物提出・検収会", "start_date": "2025-11-11", "hours": None},
             {"id": "T28", "kind": "", "phase_no": "7", "phase_name": "検収後",
              "name": "検収結果反映", "start_date": "2025-11-12", "hours": None}]},
        {"project": _AOSHIO, "primary_sheet": "WBSスケジュール_rev", "buffer_hours_total": 8.0,
         "gantt_phase_weeks": {}, "schedule_rows": []},
        {"project": _MINAMINO, "primary_sheet": None, "buffer_hours_total": None,
         "gantt_phase_weeks": {"データ理解・品質診断": 2, "前処理設計": 3, "モデル構築": 4,
                               "解釈・業務示唆整理": 5}, "schedule_rows": []},
    ]


def _visual_rows():
    def cell(header, value):
        return {"header": header, "value": value}
    return [{"doc_name": "スケジュール_r2.xlsx", "project": _AOMINE, "sheets": {"スケジュール": {
        "row_highlights": [
            {"color": "オレンジ", "row": 2, "cells": [cell("タスク名", "プロジェクトキックオフ実施")]},
            {"color": "オレンジ", "row": 12, "cells": [cell("タスク名", "中間報告会実施")]},
            {"color": "オレンジ", "row": 21, "cells": [cell("タスク名", "最終報告会実施")]},
            {"color": "赤", "row": 5, "cells": [cell("タスク名", "無関係タスク")]}]}}}]


def _plan_rows():
    return [{"project": _AOBM, "plan_metrics": {"people": [
        {"name": "加藤 大輔", "name_key": "加藤大輔", "task_count": 11},
        {"name": "渡辺 遥", "name_key": "渡辺遥", "task_count": 4}]}}]


_COMPANY = {
    "青嶺": _AOMINE, "AOBM": _AOBM, "MINAMINO": _MINAMINO,
    "京橋": _KYOBASHI, "青潮": _AOSHIO,
}


@pytest.fixture
def store(monkeypatch):
    monkeypatch.setattr(S, "load", lambda path=None: _sched_rows())
    monkeypatch.setattr(L._store, "case_record",
                        lambda project, **k: next((r for r in _sched_rows() if r["project"] == project), None))

    def _fake_company(question):
        for token, name in _COMPANY.items():
            if token in question:
                return name
        return None

    monkeypatch.setattr(L, "_company_of", _fake_company)

    import src.rag.index.visual_store as vs
    import src.rag.index.plan_coverage_store as pcs
    monkeypatch.setattr(vs, "load", lambda path=None: _visual_rows())
    monkeypatch.setattr(pcs, "load", lambda path=None: _plan_rows())
    monkeypatch.setenv("RAG_SCHEDULE_PLAN_LOOKUP", "1")
    monkeypatch.setenv("RAG_FACT_LAYER", "1")
    return None


_Q2 = "青嶺不動産アセットマネジメントのスケジュール_r2.xlsxにおいて、オレンジにハイライトされている行のタスク名をすべて答えてください。"
_Q41 = "AOBMのPLANにおいて、加藤さんが担当者に含まれるタスクIDはいくつありますか。"
_Q75 = "MINAMINOのPP内のPL案において、モデル構築は第何週に実施することになっていますか。"
_Q89 = "京橋信用ソリューションズのスケジュール.xlsxにおいて、フェーズNo6にて最後に開始するタスク名は何ですか。"
_Q90 = "青潮モビリティサービスのスケジュール.xlsxにおいて、バッファとして使用した工数の合計は何時間ですか。"


def test_idx2_highlight_rows(store):
    r = L.resolve(_Q2)
    assert r is not None
    assert r["value"] == "プロジェクトキックオフ実施、中間報告会実施、最終報告会実施"
    assert r["evidence"]["color"] == "オレンジ"


def test_idx41_assignee_task_count(store):
    r = L.resolve(_Q41)
    assert r is not None and r["value"] == "11"  # 裸整数
    assert r["evidence"]["assignee"] == "加藤 大輔"


def test_idx75_phase_week(store):
    r = L.resolve(_Q75)
    assert r is not None and r["value"] == "第4週目"
    assert r["evidence"]["phase"] == "モデル構築"


def test_idx89_phase_last_start_task(store):
    r = L.resolve(_Q89)
    assert r is not None and r["value"] == "最終報告・成果物提出・検収会"
    assert r["evidence"]["task_id"] == "T27"  # phase6 内 max 開始日


def test_idx90_buffer_hours_total(store):
    r = L.resolve(_Q90)
    assert r is not None and r["value"] == "8時間"  # 8.0 -> 整数 "8時間"


# --------------------------------------------------------------------------- deferral (精度優先)
def test_defer_when_company_absent(store):
    assert L.resolve("スケジュールでバッファ工数の合計は何時間ですか。") is None


def test_defer_idx2_wrong_color_absent(store):
    # 質問が色語を含まない → 発火しない。
    assert L.resolve("青嶺不動産アセットマネジメントのスケジュール_r2.xlsxのハイライト行のタスク名をすべて。") is None


def test_defer_idx75_phase_not_in_question(store):
    # フェーズ語がガントに無い → 一意化できず defer。
    assert L.resolve("MINAMINOのPPで存在しないフェーズは第何週に実施しますか。") is None


def test_defer_idx89_unknown_phase_no(store):
    # フェーズNo9 は行が無い → defer。
    assert L.resolve("京橋信用ソリューションズのスケジュール.xlsxでフェーズNo9にて最後に開始するタスク名は何ですか。") is None


# --------------------------------------------------------------------------- OFF byte-identical
def test_off_resolve_and_tool_are_none(monkeypatch):
    monkeypatch.delenv("RAG_SCHEDULE_PLAN_LOOKUP", raising=False)
    assert L.resolve(_Q90) is None
    assert L.tool() is None


def test_fact_layer_off_excludes_tool():
    from src.rag.agent import fact_layer
    # レイヤ OFF ⇒ tools() は空（surface byte-identical）。
    assert fact_layer.tools() == []


# =========================================================================== real corpus (skip if absent)
def _corpus_or_skip():
    try:
        from src.rag import corpus
        refs = corpus.walk()
    except Exception:
        pytest.skip("corpus unavailable")
    if not refs:
        pytest.skip("corpus empty")
    return refs


def test_real_aoshio_buffer_and_minamino_gantt():
    refs = _corpus_or_skip()
    aoshio = S.build_case(_AOSHIO, refs)
    minamino = S.build_case(_MINAMINO, refs)
    if aoshio is None or minamino is None:
        pytest.skip("target projects absent")
    assert aoshio["buffer_hours_total"] == 8.0
    assert minamino["gantt_phase_weeks"].get("モデル構築") == 4
