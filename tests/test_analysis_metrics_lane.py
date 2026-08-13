"""SOT-2698 — 分析出力メタデータ レーン（idx32 metrics enum / idx61 config hyperparam）の offline テスト。

合成ストア＋glossary スタブで決定論束縛の規律を固定する: 両 OFF ⇒ None（byte-identical）、束縛できない
案件は defer、質問が名指ししたハイパラだけを質問中の出現順で ``名前=値`` で返す。
"""
from __future__ import annotations

import pytest

from src.rag.agent import analysis_metrics_lane as L

_Q32 = ("青嶺不動産アセットマネジメントの分析出力 metrics.json の feature_selection.selected_columns に"
        "含まれている列のうち、分析コードで生成された数値交互作用特徴量の列名をすべて答えてください。")
_Q61 = ("京橋信用ソリューションズの分析コードにおいて、今回の学習で勾配ブースティング法のモデルに"
        "実際に渡される n_estimators、learning_rate、random_state はそれぞれいくつですか。設定ファイルに"
        "明示されていない値がある場合も、実行時にコード上で適用される値を含めて答えてください。")

_SEIREI = "株式会社青嶺不動産アセットマネジメント"
_KYOBASHI = "京橋信用ソリューションズ株式会社"


def _enum_rows():
    return [{
        "project": _SEIREI, "metrics_rel": "青嶺/04.分析/analysis_outputs/metrics.json",
        "selected_columns": ["BOROUGH", "BLOCK", "BOROUGH__x__BLOCK", "BLOCK__x__LOT"],
        "interaction_columns": ["BOROUGH__x__BLOCK", "BLOCK__x__LOT"],
    }]


def _raw_store():
    return {"cases_by_project": {_KYOBASHI: {
        "project": _KYOBASHI,
        "applied_hyperparams": {
            "model_type": "gradient_boosting",
            "applied": {"n_estimators": 300, "min_samples_leaf": 1, "learning_rate": 0.1,
                        "alpha": 1.0, "c": 1.0, "max_depth": 3, "max_iter": 1000, "random_state": 42},
            "config_rel": "京橋/configs/project_config.json",
            "modeling_rel": "京橋/src/modeling.py",
            "provenance": "適用値 = config.model_params があればそれ…",
        },
    }}}


class _G:
    def company_of(self, q):
        if "青嶺" in q:
            return _SEIREI
        if "京橋" in q:
            return _KYOBASHI
        return None


@pytest.fixture()
def wired(monkeypatch):
    monkeypatch.setenv("RAG_ANALYSIS_METRICS_ENUM", "1")
    monkeypatch.setenv("RAG_ANALYSIS_CONFIG_HYPERPARAM", "1")
    import src.rag.index.analysis_metrics_enum_store as enum_store
    import src.rag.index.raw_artifact_store as raw_store
    import src.rag.extract.glossary as g
    monkeypatch.setattr(enum_store, "load", lambda *a, **k: _enum_rows())
    monkeypatch.setattr(raw_store, "load", lambda *a, **k: _raw_store())
    monkeypatch.setattr(g, "load", lambda *a, **k: _G())
    return L


def test_off_is_none(monkeypatch):
    monkeypatch.delenv("RAG_ANALYSIS_METRICS_ENUM", raising=False)
    monkeypatch.delenv("RAG_ANALYSIS_CONFIG_HYPERPARAM", raising=False)
    assert L.enabled() is False
    assert L.resolve(_Q32) is None  # OFF ⇒ byte-identical fallback
    assert L.resolve(_Q61) is None
    assert L.tool() is None


def test_idx32_interaction_columns(wired):
    r = wired.resolve(_Q32)
    assert r is not None
    assert r["value"] == "BOROUGH__x__BLOCK、BLOCK__x__LOT"
    assert r["method"]["contract"] == "enum_set"


def test_idx61_applied_hyperparams_question_order(wired):
    r = wired.resolve(_Q61)
    assert r is not None
    # 質問中の出現順 (n_estimators → learning_rate → random_state)、値はストア由来。
    assert r["value"] == "n_estimators=300、learning_rate=0.1、random_state=42"
    assert r["method"]["contract"] == "config_hyperparam"
    # 質問が名指ししていない alpha/c/max_depth などは含めない。
    assert "alpha" not in r["value"] and "max_depth" not in r["value"]


def test_idx61_short_key_no_false_match(wired):
    # 'c' (単一文字キー) が Japanese 文中に誤爆しないこと。
    r = wired.resolve(_Q61)
    assert "c=1.0" not in r["value"]


def test_metrics_enum_flag_independent(monkeypatch):
    # metrics enum だけ ON、hyperparam は OFF ⇒ idx32 は発火、idx61 は defer。
    monkeypatch.setenv("RAG_ANALYSIS_METRICS_ENUM", "1")
    monkeypatch.delenv("RAG_ANALYSIS_CONFIG_HYPERPARAM", raising=False)
    import src.rag.index.analysis_metrics_enum_store as enum_store
    import src.rag.extract.glossary as g
    monkeypatch.setattr(enum_store, "load", lambda *a, **k: _enum_rows())
    monkeypatch.setattr(g, "load", lambda *a, **k: _G())
    assert L.resolve(_Q32) is not None
    assert L.resolve(_Q61) is None


def test_unbound_case_defers(wired):
    assert wired.resolve("未収録の会社の交互作用特徴量列名を metrics.json からすべて答えてください") is None


def test_missing_interaction_defers(monkeypatch):
    monkeypatch.setenv("RAG_ANALYSIS_METRICS_ENUM", "1")
    import src.rag.index.analysis_metrics_enum_store as enum_store
    import src.rag.extract.glossary as g
    rows = [{"project": _SEIREI, "selected_columns": ["A", "B"], "interaction_columns": []}]
    monkeypatch.setattr(enum_store, "load", lambda *a, **k: rows)
    monkeypatch.setattr(g, "load", lambda *a, **k: _G())
    assert L.resolve(_Q32) is None  # 交互作用列なし → defer
