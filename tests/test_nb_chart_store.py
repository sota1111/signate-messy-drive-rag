"""SOT-2685 — notebook chart-image store (cycle7 K2) の offline テスト（ネットワーク/LLM 不要）。

build 時の vision コールには触れず、決定論部分だけを固定する: フラグ既定 OFF、notebook セクション抽出
(``## N. 見出し`` ↔ savefig/埋め込み画像)、純粋日番号列判定、カテゴリ等価、vision JSON パース、write→load
往復と schema-mismatch フェイルオープン。
"""
from __future__ import annotations

import json

from src.rag.index import nb_chart_store as S


# --------------------------------------------------------------------------- flags default OFF
def test_flags_default_off(monkeypatch):
    monkeypatch.delenv("RAG_NB_CHART_STORE", raising=False)
    monkeypatch.delenv("RAG_NB_CHART_STORE_BUILD", raising=False)
    assert S.enabled() is False
    assert S.build_enabled() is False
    monkeypatch.setenv("RAG_NB_CHART_STORE", "1")
    assert S.enabled() is True
    monkeypatch.setenv("RAG_NB_CHART_STORE_BUILD", "on")
    assert S.build_enabled() is True


# --------------------------------------------------------------------------- owner key / category equality
def test_owner_key_absorbs_corp_and_spacing():
    assert S.owner_key("京橋信用ソリューションズ株式会社") == S.owner_key("京橋信用ソリューションズ")
    assert S.owner_key("医療法人社団 蒼泉会 ひがし丘総合病院") == "蒼泉会ひがし丘総合病院"


def test_cat_eq_numeric_and_string():
    assert S._cat_eq("20", 20) is True
    assert S._cat_eq("20", "20.0") is True
    assert S._cat_eq("0", "0.0") is True
    assert S._cat_eq("20", "18") is False
    assert S._cat_eq(None, "20") is False


# --------------------------------------------------------------------------- notebook parsing
def _cell(kind, source, images=0):
    outputs = [{"output_type": "display_data", "data": {"image/png": "aGVsbG8="}} for _ in range(images)]
    return {"cell_type": kind, "source": source, "outputs": outputs}


def test_notebook_sections_maps_heading_to_savefig_and_embedded():
    nb = {"cells": [
        _cell("markdown", ["## 5. 目的変数分析"]),
        _cell("code", ["plt.hist(x)\n", "plt.savefig(FIG_DIR / 'target_distribution.png', dpi=150)\n"], images=1),
        _cell("markdown", ["## 7. 日付分析"]),
        _cell("code", ["ax.plot(a)\n", "plt.savefig(FIG_DIR / \"figure_06.png\")\n"]),
        _cell("markdown", ["最も相関が高いのは…"]),  # 見出しでない markdown はセクションを変えない
        _cell("code", ["print('no chart here')\n"]),  # 画像も savefig も無い ⇒ セクション化しない
    ]}
    secs = S._notebook_sections(nb)
    assert len(secs) == 2
    assert secs[0]["number"] == "5" and secs[0]["title"] == "目的変数分析"
    assert secs[0]["savefigs"] == ["target_distribution.png"]
    assert len(secs[0]["embedded"]) == 1
    assert secs[1]["number"] == "7" and secs[1]["title"] == "日付分析"
    assert secs[1]["savefigs"] == ["figure_06.png"]


def test_pure_day_column_prefers_day_and_rejects_non_day():
    import pandas as pd
    df = pd.DataFrame({"day": [1, 15, 31, 20], "age": [40, 55, 70, 33], "label": ["a", "b", "c", "d"]})
    assert S._pure_day_column(df) == "day"
    df2 = pd.DataFrame({"age": [40, 55, 70], "label": ["a", "b", "c"]})  # age > 31 ⇒ not a day column
    assert S._pure_day_column(df2) is None


# --------------------------------------------------------------------------- vision JSON parsing
def test_parse_vision_json_strips_fences_and_prose():
    assert S._parse_vision_json('```json\n{"y_axis_max_tick": 1200}\n```')["y_axis_max_tick"] == 1200
    assert S._parse_vision_json('前置き {"peak_category": "20"} 後置き')["peak_category"] == "20"
    assert S._parse_vision_json("not json at all") is None
    assert S._parse_vision_json("") is None


# --------------------------------------------------------------------------- write / load roundtrip
def _rec(doc, num, title, **kw):
    base = {"record_key": f"{doc}#{num}#sha", "doc_id": doc, "project": "京橋信用ソリューションズ株式会社",
            "notebook": "01_eda.ipynb", "section_number": num, "section_title": title,
            "samples": 3, "y_axis_max_tick": None, "peak_category": None}
    base.update(kw)
    return base


def test_write_load_roundtrip_and_schema_guard(tmp_path):
    p = tmp_path / "nb.jsonl"
    recs = [_rec("d/京橋", "7", "日付分析", y_axis_max_tick=1600),
            _rec("d/京橋", "5", "目的変数分析", y_axis_max_tick=1200)]
    S.write_store(recs, p)
    S.reset_cache()
    loaded = S.load(p)
    assert [r["section_number"] for r in loaded] == ["5", "7"]  # (doc_id, section) sorted
    # docs_for_project 前置きの corp/spacing 吸収
    S.reset_cache()
    assert len(S.docs_for_project("京橋信用ソリューションズ", path=p)) == 2

    # schema mismatch ⇒ フェイルオープンで空
    bad = tmp_path / "bad.jsonl"
    bad.write_text(json.dumps({"schema": "nb-chart-store", "version": 999}) + "\n"
                   + json.dumps(_rec("x", "1", "t")) + "\n", encoding="utf-8")
    S.reset_cache()
    assert S.load(bad) == []

    # 完全欠落 ⇒ 空（回帰ゼロ）
    S.reset_cache()
    assert S.load(tmp_path / "missing.jsonl") == []
