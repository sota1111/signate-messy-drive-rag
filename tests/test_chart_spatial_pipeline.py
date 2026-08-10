"""SOT-2609 (Wave A4) — offline tests for the deterministic ``chart_read`` / ``spatial`` pipeline.

Network-free and corpus-free: the strict chart reader (``chart_numcache.read_chart_values``), the seating
directory lookup (``seating_chart.seating_lookup``), file resolution (``_find_named_file`` /
``_resolve_project`` / ``_resolve_train_ref``) and per-series color resolution (``_series_color_names``) are
monkeypatched with canned values, so the pipeline's own recognizer / dispatch / color-family / rounding /
A3-composition logic runs (not a stub). The invariants under test: the pipeline self-registers ``chart_read``
and ``spatial`` and composes seating列挙 in front of A3's ``full_enumeration`` WITHOUT dropping A3 (idx44 型);
each recognizer grounds only its tight shape (idx10/33/58 型) and falls back (``None``) on ambiguity; a
color that pins ≠1 series falls back; OFF ⇒ byte-identical; and it wires through ``det_pipeline.resolve`` +
``formatting.format_contract`` end-to-end (number / 「、」-joined list — template-first, no LLM naturalize).
"""
from __future__ import annotations

import pytest

from src.rag.agent import det_pipeline as dp
from src.rag.agent import formatting
from src.rag.agent.pipelines import chart_spatial as cs
from src.rag.agent.pipelines import enumeration as en


# --------------------------------------------------------------------------- registry cleanup fixture
@pytest.fixture(autouse=True)
def _restore_registry():
    saved = dict(dp._REGISTRY)
    saved_a3 = cs._A3_FULL_ENUMERATION
    try:
        yield
    finally:
        dp._REGISTRY.clear()
        dp._REGISTRY.update(saved)
        cs._A3_FULL_ENUMERATION = saved_a3


# --------------------------------------------------------------------------- registration + A3 composition
def test_registers_chart_read_and_spatial_and_composes_full_enum():
    cs.register(replace=True)
    contracts = dp.registered_contracts()
    assert {"chart_read", "spatial", "full_enumeration"} <= set(contracts)
    assert dp._REGISTRY["chart_read"] is cs.chart_read_pipeline
    assert dp._REGISTRY["spatial"] is cs.spatial_pipeline
    # full_enumeration is fronted by the composition, and A3's own pipeline is captured (not dropped).
    assert dp._REGISTRY["full_enumeration"] is cs._composed_full_enumeration
    assert cs._A3_FULL_ENUMERATION is en.full_enumeration_pipeline


def test_composed_full_enum_delegates_to_a3_when_not_seating(monkeypatch):
    sentinel = {"value": ["A3"], "evidence": {}, "method": {"shape": "a3"}}
    monkeypatch.setattr(cs, "_A3_FULL_ENUMERATION", lambda q, *, profile=None: sentinel)
    monkeypatch.setattr(cs, "_seating_answer", lambda q: None)  # non-seating ⇒ seating recognizer is silent
    out = cs._composed_full_enumeration("契約期間が重なる案件を主略称ですべて挙げてください。")
    assert out is sentinel  # A3 behavior preserved byte-identically


def test_composed_full_enum_prefers_seating_enum(monkeypatch):
    seat = {"value": ["鈴木", "藤田"], "evidence": {}, "method": {"shape": "seating_relation_lookup"}}
    monkeypatch.setattr(cs, "_A3_FULL_ENUMERATION",
                        lambda q, *, profile=None: pytest.fail("A3 must not be reached for a seating enum"))
    monkeypatch.setattr(cs, "_seating_answer", lambda q: seat)
    out = cs._composed_full_enumeration("佐藤さんから見て右側に座っている人の名前をすべて挙げてください。")
    assert out is seat


# --------------------------------------------------------------------------- chart shape 1: histogram max
def _wire_hist(monkeypatch, *, result_count=958):
    monkeypatch.setattr(cs, "_resolve_project", lambda q: "医療法人社団 恒一会 かえで総合病院")
    monkeypatch.setattr(cs, "_find_named_file", lambda q, p: object())  # a resolvable ref stand-in
    captured = {}

    def fake_read(ref, **kwargs):
        captured.update(kwargs)
        return {"value": {"result": result_count}, "evidence": {"file": "train.xlsx",
                "source_column": "AG_ratio", "bin_width": 0.01, "n_rows": 3000}, "method": {}}

    monkeypatch.setattr(cs, "_read_chart_values", fake_read)
    return captured


def test_histogram_max_count_grounds(monkeypatch):
    captured = _wire_hist(monkeypatch)
    q = "恒一会 かえで総合病院のtrain.xlsxにおいて、AG_ratioのヒストグラムで最も多いカウント数はいくつですか。"
    out = cs.chart_read_pipeline(q)
    assert out is not None
    assert out["value"] == 958
    assert captured["column"] == "AG_ratio"
    assert captured["operation"] == "histogram_max_count"
    assert out["method"]["shape"] == "histogram_max_count"
    assert out["method"]["vision_used"] is False


def test_histogram_without_max_cue_falls_back(monkeypatch):
    _wire_hist(monkeypatch)
    q = "train.xlsxにおいて、AG_ratioのヒストグラムを表示してください。"  # no 最も多い/最大 count cue
    assert cs.chart_read_pipeline(q) is None


def test_histogram_non_int_result_falls_back(monkeypatch):
    _wire_hist(monkeypatch, result_count="oops")  # a non-int strict result ⇒ never committed
    q = "train.xlsxにおいて、AG_ratioのヒストグラムで最も多いカウント数はいくつですか。"
    assert cs.chart_read_pipeline(q) is None


# --------------------------------------------------------------------------- chart shape 2: colored point read
def _wire_point(monkeypatch, colors, series):
    monkeypatch.setattr(cs, "_resolve_project", lambda q: "株式会社青潮モビリティサービス")
    monkeypatch.setattr(cs, "_find_named_file", lambda q, p: _Ref())
    charts = [{"member": "c1", "series": [{"name": "s1", "values": [0.0]}]},
              {"member": "c2", "series": series}]
    monkeypatch.setattr(cs, "_read_chart_values",
                        lambda ref, **k: {"value": {"charts": charts}, "evidence": {}, "method": {}})
    monkeypatch.setattr(cs, "_series_color_names", lambda ref, member: colors)


class _Ref:
    rel = "proj/基礎分析.docx"


def test_chart_point_read_grounds_with_color_family_and_rounding(monkeypatch):
    # chart2: cnt=オレンジ, windspeed=水色 (accent1). 青 query ⊇ 水色 ⇒ windspeed uniquely; x=3 → values[2].
    series = [{"name": "平均 / cnt", "values": [143.8, 145.9, 147.5]},
              {"name": "平均 / windspeed", "values": [0.1818, 0.18390, 0.1869637479541717]}]
    _wire_point(monkeypatch, ["オレンジ", "水色"], series)
    q = "基礎分析.docxのグラフ2で、x=3のときの青色の折れ線のyの値を小数第5位で答えてください。"
    out = cs.chart_read_pipeline(q)
    assert out is not None
    assert out["value"] == 0.18696  # exact numCache value rounded to 5 places
    assert out["evidence"]["series_name"] == "平均 / windspeed"
    assert out["evidence"]["category_index_1based"] == 3


def test_chart_point_read_ambiguous_color_falls_back(monkeypatch):
    # Two series both classify blue-family ⇒ color cannot pin one series ⇒ fallback (never guess).
    series = [{"name": "s1", "values": [1.0, 2.0, 3.0]}, {"name": "s2", "values": [4.0, 5.0, 6.0]}]
    _wire_point(monkeypatch, ["青", "水色"], series)
    q = "基礎分析.docxのグラフ2で、x=3のときの青色の折れ線のyの値を答えてください。"
    assert cs.chart_read_pipeline(q) is None


def test_chart_point_read_chart_index_out_of_range_falls_back(monkeypatch):
    series = [{"name": "s", "values": [1.0, 2.0, 3.0]}]
    _wire_point(monkeypatch, ["水色"], series)
    q = "基礎分析.docxのグラフ5で、x=1のときの青色の折れ線のyの値を答えてください。"  # only 2 charts exist
    assert cs.chart_read_pipeline(q) is None


def test_chart_point_read_no_color_falls_back(monkeypatch):
    series = [{"name": "s", "values": [1.0, 2.0, 3.0]}]
    _wire_point(monkeypatch, ["水色"], series)
    q = "基礎分析.docxのグラフ2で、x=2のときの折れ線のyの値を答えてください。"  # no color word ⇒ not this shape
    assert cs.chart_read_pipeline(q) is None


# --------------------------------------------------------------------------- spatial: seating
def _wire_seating(monkeypatch, value, *, expect_field=None, expect_name=None):
    captured = {}

    def fake_lookup(**kwargs):
        captured.update(kwargs)
        ev = {"relation": "opposite", "neighbour": {"ext": "7102"}, "origin": "reviewed"}
        return {"value": value, "evidence": ev, "method": {"scheme": "spatial"}}

    monkeypatch.setattr(cs, "_seating_lookup", fake_lookup)
    return captured


def test_seating_opposite_ext_grounds(monkeypatch):
    captured = _wire_seating(monkeypatch, "7102")
    q = "社内管理フォルダにあるFMにおいて、井上さんの向かいに座っている方のEXTを教えてください。"
    out = cs.spatial_pipeline(q)
    assert out is not None
    assert out["value"] == "7102"
    assert captured["name"] == "井上"
    assert captured["field"] == "ext"  # 「EXT」 ask ⇒ ext
    assert out["method"]["shape"] == "seating_relation_lookup"


def test_seating_right_side_name_list_grounds(monkeypatch):
    captured = _wire_seating(monkeypatch, ["鈴木", "藤田"])
    q = "IMにあるFMにおいて、佐藤さんから見て右側に座っている人の名前をすべて挙げてください。"
    out = cs.spatial_pipeline(q)
    assert out is not None
    assert out["value"] == ["鈴木", "藤田"]
    assert captured["name"] == "佐藤"
    assert captured["field"] == "name"  # 「名前」 ask ⇒ name


def test_seating_abstain_value_falls_back(monkeypatch):
    _wire_seating(monkeypatch, None)  # seating_lookup abstains (ambiguous / not found) ⇒ fallback
    q = "田中さんの向かいに座っている方のEXTを教えてください。"
    assert cs.spatial_pipeline(q) is None


def test_non_seating_question_falls_back(monkeypatch):
    monkeypatch.setattr(cs, "_seating_lookup",
                        lambda **k: pytest.fail("seating_lookup must not be called without a seating cue"))
    assert cs.spatial_pipeline("train.xlsxのAG_ratioの平均はいくつですか。") is None


# --------------------------------------------------------------------------- OFF byte-identical + e2e format
def test_resolve_off_returns_none(monkeypatch):
    monkeypatch.delenv("RAG_DET_PIPELINE_ROUTER", raising=False)
    assert dp.resolve("井上さんの向かいに座っている方のEXTを教えてください。", "spatial") is None


def test_end_to_end_seating_list_joined(monkeypatch):
    _wire_seating(monkeypatch, ["鈴木", "藤田"])
    q = "佐藤さんから見て右側に座っている人の名前をすべて挙げてください。"
    contract = dp.resolve(q, "spatial", force=True)
    assert contract is not None
    formatted = formatting.format_contract(contract, q, contract_type="spatial", force=True)
    assert formatted["value"] == "鈴木、藤田"  # list → 「、」-joined, template-first
    assert formatted["method"]["formatting"]["template_only"] is True


def test_end_to_end_histogram_number(monkeypatch):
    _wire_hist(monkeypatch, result_count=958)
    q = "train.xlsxにおいて、AG_ratioのヒストグラムで最も多いカウント数はいくつですか。"
    contract = dp.resolve(q, "chart_read", force=True)
    formatted = formatting.format_contract(contract, q, contract_type="chart_read", force=True)
    assert formatted["value"] == "958"
    assert formatted["method"]["formatting"]["template_only"] is True
