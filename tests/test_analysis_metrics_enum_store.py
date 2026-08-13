"""SOT-2698 — 分析出力 metrics.json enum ストア（idx32）の offline テスト（LLM/corpus 不要）。

合成 FileRef で build を回し、``feature_selection.selected_columns`` から ``__x__`` 交互作用列の部分集合
が焼けること、汎用 enum フィールドが順序保存で焼けること、OFF フラグ判定を固定する。
"""
from __future__ import annotations

import json
from types import SimpleNamespace

from src.rag.index import analysis_metrics_enum_store as S


def _write_metrics(tmp_path, project, rel_name, payload):
    p = tmp_path / rel_name
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
    return SimpleNamespace(project=project, rel=f"{project}/04.分析/{rel_name}", name="metrics.json",
                           ext="json", category="analysis", path=p)


_SEIREI = "株式会社青嶺不動産アセットマネジメント"
_METRICS = {
    "task_type": "regression",
    "feature_selection": {
        "selected_columns": [
            "BOROUGH", "BLOCK", "LOT", "ZIP CODE",
            "BOROUGH__x__BLOCK", "BOROUGH__x__LOT", "BOROUGH__x__ZIP CODE",
            "BLOCK__x__LOT", "BLOCK__x__ZIP CODE", "LOT__x__ZIP CODE",
        ],
        "excluded_columns": [
            {"column": "id", "reason": "identifier_like_name"},
            {"column": "NEIGHBORHOOD", "reason": "high_cardinality_categorical"},
        ],
    },
    "ordered_feature_columns": ["BOROUGH", "BLOCK", "LOT"],
}


def test_build_bakes_interaction_subset(tmp_path):
    ref = _write_metrics(tmp_path, _SEIREI, "analysis_outputs/metrics.json", _METRICS)
    out = tmp_path / "store.jsonl"
    S.build([ref], out=out, write_report=False)
    rows = S.load(out)
    assert len(rows) == 1
    rec = rows[0]
    assert rec["project"] == _SEIREI
    # 交互作用列 = __x__ を含む部分集合、順序保存。
    assert rec["interaction_columns"] == [
        "BOROUGH__x__BLOCK", "BOROUGH__x__LOT", "BOROUGH__x__ZIP CODE",
        "BLOCK__x__LOT", "BLOCK__x__ZIP CODE", "LOT__x__ZIP CODE",
    ]
    assert rec["selected_columns"][0] == "BOROUGH"
    # excluded は dict → column 名へ正規化。
    assert rec["excluded_columns"] == ["id", "NEIGHBORHOOD"]
    # ordered_feature_columns / enum_fields も焼ける（汎用網羅抽出）。
    assert rec["ordered_feature_columns"] == ["BOROUGH", "BLOCK", "LOT"]
    assert "feature_selection.selected_columns" not in rec["enum_fields"]  # 直下 list のみ
    assert rec["enum_fields"]["ordered_feature_columns"] == ["BOROUGH", "BLOCK", "LOT"]


def test_prefers_metrics_with_feature_selection(tmp_path):
    # feature_selection の無い metrics.json は fallback にしかならず、ある方を選ぶ。
    ref_no_fs = _write_metrics(tmp_path, _SEIREI, "analysis_project/metrics.json", {"task_type": "x"})
    ref_fs = _write_metrics(tmp_path, _SEIREI, "analysis_outputs/metrics.json", _METRICS)
    out = tmp_path / "store.jsonl"
    S.build([ref_no_fs, ref_fs], out=out, write_report=False)
    rows = S.load(out)
    assert rows and rows[0].get("interaction_columns")
    assert "analysis_outputs" in rows[0]["metrics_rel"]


def test_no_interaction_columns_still_bakes_selected(tmp_path):
    payload = {"feature_selection": {"selected_columns": ["A", "B", "C"]}}
    ref = _write_metrics(tmp_path, "P", "analysis_outputs/metrics.json", payload)
    out = tmp_path / "store.jsonl"
    S.build([ref], out=out, write_report=False)
    rows = S.load(out)
    assert rows[0]["selected_columns"] == ["A", "B", "C"]
    assert rows[0]["interaction_columns"] == []  # 交互作用列なし → 空（レーンは None で defer）


def test_enabled_flag(monkeypatch):
    monkeypatch.delenv("RAG_ANALYSIS_METRICS_ENUM", raising=False)
    assert S.enabled() is False
    monkeypatch.setenv("RAG_ANALYSIS_METRICS_ENUM", "1")
    assert S.enabled() is True


def test_load_schema_mismatch_returns_empty(tmp_path):
    out = tmp_path / "store.jsonl"
    out.write_text(json.dumps({"schema": "other", "version": 99}) + "\n", encoding="utf-8")
    assert S.load(out) == []
