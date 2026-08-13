"""SOT-2699 — 統計表 rank/ratio 事前計算ストアの決定論ヘルパの offline テスト（LLM/corpus 不要）。"""
from __future__ import annotations

from src.rag.index import derived_ranking_store as S


def test_parse_numeric_accepts_plain_and_percent_rejects_units():
    assert S.parse_numeric("18.2") == 18.2
    assert S.parse_numeric("7.3") == 7.3
    assert S.parse_numeric("10.5%") == 10.5
    assert S.parse_numeric("1,234") == 1234.0
    assert S.parse_numeric("-3.5") == -3.5
    # 単位語・混在は数値ではない（fail-closed）。
    assert S.parse_numeric("5億3,700万人") is None
    assert S.parse_numeric("1位") is None
    assert S.parse_numeric("") is None
    assert S.parse_numeric("約12") is None


def test_metric_core_strips_units_and_parens():
    assert S.metric_core("死亡率（%）") == "死亡率"
    assert S.metric_core("平均年収 (万円)") == "平均年収"
    assert S.metric_core("死亡率") == "死亡率"


def _death_table():
    # みなみ野 糖尿病統計の死亡率ランキング表（高い側/低い側が同一 header『死亡率（%）』で分割）。
    return {
        "table_id": 3, "locus": "表3", "caption": "都道府県別 糖尿病死亡率",
        "rows": [
            ["順位", "死亡率が高い都道府県（ワースト）", "死亡率（%）", "死亡率が低い都道府県（ベスト）", "死亡率（%）"],
            ["1位", "青森県", "18.2", "神奈川県", "7.2"],
            ["2位", "秋田県", "16.3", "愛知県", "7.22"],
            ["3位", "香川県", "16.1", "東京都", "7.28"],
            ["4位", "鹿児島県", "15.0", "滋賀県", "7.3"],
            ["5位", "徳島県", "14.9", "奈良県", "8.0"],
        ],
    }


def test_table_series_merges_same_header_columns_and_sorts():
    series = S._table_series(_death_table())
    # 『死亡率』系列がちょうど1つ（高い側 col + 低い側 col を統合 = 10 値）。
    death = [s for s in series if s["metric_key"] == "死亡率"]
    assert len(death) == 1
    s = death[0]
    assert s["n"] == 10
    # 昇順/降順が値で正しくソートされ、label が対応都道府県に紐づく。
    assert s["sorted_desc"][0]["value"] == 18.2
    assert s["sorted_desc"][0]["label"] == "青森県"
    assert s["sorted_asc"][0]["value"] == 7.2
    assert s["sorted_asc"][3]["value"] == 7.3  # 4番目に低い
    assert s["sorted_asc"][3]["label"] == "滋賀県"


def test_table_series_skips_non_numeric_and_short_columns():
    # 数値列が無い / 2 値未満の表からは系列を作らない。
    t = {"rows": [["名前", "所属"], ["青森", "東北"], ["滋賀", "近畿"]]}
    assert S._table_series(t) == []
    t2 = {"rows": [["値"], ["10"]]}  # 1 値のみ
    assert S._table_series(t2) == []


def test_build_and_load_roundtrip(tmp_path, monkeypatch):
    docs = {"docs": [{"project": "テスト案件", "rel": "x/統計.docx", "name": "統計.docx",
                      "tables": [_death_table()]}]}
    monkeypatch.setattr(S.doc_reach_store, "load", lambda *a, **k: docs)
    out = tmp_path / "derived_ranking_store.jsonl"
    summary = S.build(out=out, write_report=False)
    assert summary["projects"] == 1
    data = S.load(out)
    series = data["by_project"]["テスト案件"]
    death = [s for s in series if s["metric_key"] == "死亡率"]
    assert len(death) == 1 and death[0]["n"] == 10


def test_load_schema_mismatch_returns_empty(tmp_path):
    p = tmp_path / "bad.jsonl"
    p.write_text('{"schema": "derived_ranking_store", "version": 999}\n', encoding="utf-8")
    assert S.load(p) == {"by_project": {}}
