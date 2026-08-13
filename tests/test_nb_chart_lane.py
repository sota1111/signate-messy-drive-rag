"""SOT-2685 — notebook chart-image serve レーン（cycle7 K2）の offline テスト（ネットワーク/LLM 不要）。

決定論束縛の規律を合成ストアで固定する: OFF ⇒ None（byte-identical）、idx56=目的変数分析の y軸目盛り最大値
(全 vision サンプル一致の時のみ)、idx66=日付分析の件数最多日(データ側 day 別件数 argmax = 権威)。曖昧な案件
名指し / 描画属性未確定 / 非該当質問は defer。
"""
from __future__ import annotations

import pytest

from src.rag.agent import nb_chart_lane as L

_HIGASHI = "医療法人社団 蒼泉会 ひがし丘総合病院"
_KYOBASHI = "京橋信用ソリューションズ株式会社"


def _rec(project, num, title, **kw):
    base = {"doc_id": f"…/{project}/…/01_eda.ipynb", "project": project, "notebook": "01_eda.ipynb",
            "section_number": num, "section_title": title, "figure": None, "source": "figure_file",
            "y_axis_max_tick": None, "y_axis_tick_labels": None, "unanimous_y_max": False,
            "peak_category": None, "unanimous_peak": False, "data_check": None,
            "verified": False, "samples": 3, "vision_model": "gemini-2.5-flash"}
    base.update(kw)
    return base


@pytest.fixture()
def synth(monkeypatch):
    rows = [
        _rec(_HIGASHI, "5", "目的変数分析", figure="target_distribution.png",
             y_axis_max_tick=1200, y_axis_tick_labels=[0, 200, 400, 600, 800, 1000, 1200],
             unanimous_y_max=True, peak_category="0",
             data_check={"kind": "target_count_argmax", "column": "charges", "category": "0",
                         "count": 1256, "agrees": True}, verified=True),
        _rec(_HIGASHI, "7", "日付分析", figure="date_feature_trend.png",
             data_check={"kind": "day_count_argmax", "column": "day", "category": "12",
                         "count": 900, "agrees": True}, verified=True, peak_category="12"),
        _rec(_KYOBASHI, "5", "目的変数分析", figure="target_distribution.png",
             y_axis_max_tick=800, unanimous_y_max=True, peak_category="0"),
        _rec(_KYOBASHI, "7", "日付分析", figure="figure_06.png",
             peak_category="0",  # vision misread the peak x on a dense line chart …
             data_check={"kind": "day_count_argmax", "column": "day", "category": "20",
                         "count": 1612, "agrees": False}),  # … but data argmax = 20 (authoritative)
    ]
    monkeypatch.setattr(L._store, "load", lambda path=None: rows)
    return rows


_Q56 = "蒼泉会 ひがし丘総合病院の01_eda.ipynbにおける目的変数分析の可視化において、y軸に実際に表示されている目盛りの最大値は何ですか。"
_Q66 = "京橋信用ソリューションズのEDAの日付分析の可視化において、件数が最も高いのは何日ですか。"


def test_default_off_returns_none(monkeypatch, synth):
    monkeypatch.delenv("RAG_NB_CHART_STORE", raising=False)
    assert L.enabled() is False
    assert L.resolve(_Q56) is None and L.resolve(_Q66) is None
    assert L.tool() is None


def test_idx56_y_axis_max_tick(monkeypatch, synth):
    monkeypatch.setenv("RAG_NB_CHART_STORE", "1")
    r = L.resolve(_Q56)
    assert r is not None and r["value"] == 1200
    assert r["method"]["selection"] == "y_axis_max_tick"
    assert r["evidence"]["vision_only"] is True


def test_idx66_peak_day_uses_data_argmax_despite_vision_disagreement(monkeypatch, synth):
    monkeypatch.setenv("RAG_NB_CHART_STORE", "1")
    r = L.resolve(_Q66)
    assert r is not None and r["value"] == "20日"  # data argmax, not the vision peak_category "0"
    assert r["method"]["selection"] == "date_peak_day"
    assert r["evidence"]["authority"] == "data_argmax"
    assert r["evidence"]["vision_agrees"] is False


def test_y_tick_defers_when_vision_not_unanimous(monkeypatch, synth):
    monkeypatch.setenv("RAG_NB_CHART_STORE", "1")
    synth[0]["unanimous_y_max"] = False  # 描画属性は全サンプル一致でなければ確定しない
    assert L.resolve(_Q56) is None


def test_date_defers_without_pure_day_column(monkeypatch, synth):
    monkeypatch.setenv("RAG_NB_CHART_STORE", "1")
    synth[3]["data_check"] = {"kind": "monthly", "category": "2025-01"}  # 日番号列でない ⇒ 「何日」不成立
    assert L.resolve(_Q66) is None


def test_non_target_questions_defer(monkeypatch, synth):
    monkeypatch.setenv("RAG_NB_CHART_STORE", "1")
    assert L.resolve("京橋信用ソリューションズの相関分析で最も相関が高い組み合わせは何ですか。") is None
    assert L.resolve("ひがし丘総合病院の train.csv の行数は何行ですか。") is None


def test_ambiguous_when_no_project_named(monkeypatch, synth):
    monkeypatch.setenv("RAG_NB_CHART_STORE", "1")
    # 案件名を含まない質問は案件を一意に束縛できない ⇒ defer
    assert L.resolve("日付分析の可視化で件数が最も高いのは何日ですか。") is None


def test_tool_shape_when_enabled(monkeypatch, synth):
    monkeypatch.setenv("RAG_NB_CHART_STORE", "1")
    t = L.tool()
    assert t is not None
    name, desc, schema, fn = t
    assert name == "nb_chart_lookup"
    assert schema["required"] == ["project"]
    out = fn(project="京橋信用ソリューションズ", section="日付")
    assert out["value"] and out["value"][0]["section_title"] == "日付分析"
