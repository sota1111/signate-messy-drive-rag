"""SOT-2655 — 報告数値属性ストア + serve-path 決定論レーンの offline テスト（ネットワーク/LLM/実コーパス非依存）.

合成ストア（``report_attr_store.load`` を monkeypatch）に対して、3 本の決定論レーン
（最良モデル param / フェーズ工数合計 / 高影響特徴量×相関最大）、精度優先の deferral、
RAG_REPORT_ATTR_STORE 既定 OFF の byte-identical 挙動、ツール contract を検証する。
"""
from __future__ import annotations

import pytest

from src.rag.agent import fact_layer as fl
from src.rag.agent import report_attr_lane as ral
from src.rag.index import report_attr_store as ras
from src.rag.tools import contract as _contract


# --------------------------------------------------------------------------- synthetic store
def _rows():
    src = {"doc_id": "プロジェクト/株式会社青潮モビリティサービス/06.報告書/…_最終報告.pdf"}
    train = {"doc_id": "…/03.データ/train.csv", "target": "Outcome"}
    return [
        {"case_id": "株式会社青潮モビリティサービス",
         "report_doc": src["doc_id"],
         "model": {"model_type": "hist_gradient_boosting", "target_column": "cnt",
                   "metrics": {"rmse": 46.98, "r2": 0.845},
                   "params": {"max_depth": {"value": 6, "source": {"doc_id": "…/metrics.json"}},
                              "learning_rate": {"value": 0.05, "source": {"doc_id": "…/metrics.json"}},
                              "max_iter": {"value": 400, "source": {"doc_id": "…/metrics.json"}}}},
         "phases": [
             {"name": "フェーズA", "key": "A", "hours": {"low": 20, "high": 30, "raw": "20-30h"}, "source": src},
             {"name": "フェーズB", "key": "B", "hours": {"low": 60, "high": 100, "raw": "60-100h"}, "source": src},
             {"name": "フェーズC", "key": "C", "hours": None, "source": src}]},
        {"case_id": "医療法人社団 蒼樹会 みなみ野女性医療センター",
         "report_doc": "…/みなみ野…_最終報告.pdf",
         "features": {"target_column": "Outcome",
                      "high_impact_features": {"value": ["Glucose", "BMI"], "source": src},
                      "target_correlations": {
                          "Glucose": {"value": 0.0647, "abs": 0.0647, "source": train},
                          "BMI": {"value": 0.2444, "abs": 0.2444, "source": train}}}},
    ]


@pytest.fixture
def store(monkeypatch):
    monkeypatch.setattr(ras, "load", lambda path=None: _rows())
    monkeypatch.setenv("RAG_REPORT_ATTR_STORE", "1")
    monkeypatch.setenv("RAG_FACT_LAYER", "1")
    return None


# --------------------------------------------------------------------------- OFF byte-identical
def test_off_is_inert(monkeypatch):
    monkeypatch.delenv("RAG_REPORT_ATTR_STORE", raising=False)
    assert ral.enabled() is False
    assert ral.resolve("青潮モビリティサービスの最良モデルのmax_depthは?") is None
    assert ral.tool() is None


def test_off_lane_not_in_fact_layer_resolve(monkeypatch):
    # RAG_FACT_LAYER on but report store OFF ⇒ the report lane never fires (byte-identical fact layer).
    monkeypatch.setattr(ras, "load", lambda path=None: _rows())
    monkeypatch.setenv("RAG_FACT_LAYER", "1")
    monkeypatch.delenv("RAG_REPORT_ATTR_STORE", raising=False)
    assert fl.resolve("青潮モビリティサービスの最良モデルのmax_depthはいくつ?", "numeric") is None


def test_off_tool_not_in_fact_layer_tools(monkeypatch):
    monkeypatch.setenv("RAG_FACT_LAYER", "1")
    monkeypatch.delenv("RAG_REPORT_ATTR_STORE", raising=False)
    assert ral.REPORT_ATTR_LOOKUP not in {t[0] for t in fl.tools()}


# --------------------------------------------------------------------------- (idx5) model hyperparameter
def test_model_param_lane_max_depth(store):
    r = ral.resolve("青潮モビリティサービスの最終報告にて最良モデルとしているモデルのパラメータである"
                    "max_depthはいくらに設定されていますか。")
    assert r is not None and r["value"] == 6
    assert r["method"]["verified_operand"] is True and r["evidence"]["param"] == "max_depth"


def test_model_param_lane_via_fact_layer(store):
    r = fl.resolve("青潮モビリティサービスの最良モデルのパラメータmax_depthは?", "numeric")
    assert r is not None and r["value"] == 6


def test_model_param_defers_without_param_name(store):
    # モデル への言及はあるが具体パラメータ名が無い ⇒ defer（精度優先）
    assert ral.resolve("青潮モビリティサービスの最良モデルは何ですか?") is None


def test_model_param_defers_on_two_params(store):
    # 2 つのパラメータ名が同時に現れたら一意に選べない ⇒ defer
    assert ral.resolve("青潮モビリティサービスのモデルの max_depth と learning_rate は?") is None


# --------------------------------------------------------------------------- (idx64) phase effort sum
def test_phase_sum_lane_A_plus_B(store):
    r = ral.resolve("青潮モビリティサービスの最終報告PDFにおいて、将来のフェーズAとフェーズBを"
                    "実施した場合の想定工数は合計で何時間ですか。")
    assert r is not None and r["value"] == "80〜130時間"
    assert r["evidence"]["sum_low"] == 80 and r["evidence"]["sum_high"] == 130


def test_phase_sum_defers_without_sum_cue(store):
    # 「合計」等の集計キューが無ければ defer（単一フェーズ lookup は別経路）
    assert ral.resolve("青潮モビリティサービスのフェーズAの想定工数は何時間?") is None


def test_phase_sum_defers_on_missing_phase_hours(store):
    # フェーズC は工数 None ⇒ A+C は完全被覆が立たず defer
    assert ral.resolve("青潮モビリティサービスのフェーズAとフェーズCの工数合計は何時間?") is None


# --------------------------------------------------------------------------- (idx28) feature corr max
def test_feature_corr_lane_bmi(store):
    r = ral.resolve("蒼樹会 みなみ野女性医療センターの分析結果として予測に影響が高いと報告されている"
                    "特徴量の中で、最もターゲットとの相関が高い特徴量を答えてください。")
    assert r is not None and r["value"] == "BMI"
    assert set(r["evidence"]["high_impact_features"]) == {"Glucose", "BMI"}


def test_feature_corr_defers_without_corr_cue(store):
    # 相関の言及が無い（影響最大だけ）なら defer（別の意味の問い）
    assert ral.resolve("みなみ野女性医療センターで最も影響が高い特徴量は?") is None


# --------------------------------------------------------------------------- tool contract
def test_tool_lookup_contract(store):
    t = ral.tool()
    assert t is not None and t[0] == ral.REPORT_ATTR_LOOKUP
    handler = t[3]
    out = handler(case="青潮モビリティサービス", attribute="max_depth")
    assert _contract.is_contract(out) and out["value"] == 6
    phases = handler(case="青潮モビリティサービス", attribute="phases")
    assert isinstance(phases["value"], list) and len(phases["value"]) == 3
    corr = handler(case="みなみ野女性医療センター", attribute="correlations")
    assert corr["value"]["BMI"] == 0.2444


def test_tool_unknown_case(store):
    out = ral.tool()[3](case="存在しない会社", attribute="max_depth")
    assert out["value"] is None


# --------------------------------------------------------------------------- store build contract (schema)
def test_store_load_schema_roundtrip(tmp_path):
    p = tmp_path / "report_attr_store.jsonl"
    ras.write_store(_rows(), path=p)
    ras.reset_cache()
    rows = ras.load(path=p)
    assert len(rows) == 2
    rec = ras.case_record("青潮モビリティサービス", path=p)
    assert rec is not None and rec["model"]["params"]["max_depth"]["value"] == 6
