"""SOT-2678 — 分析成果物 raw ファイルストア + lookup レーンの offline テスト（ネット/LLM 不要）。

固定する不変量:
* OFF（``RAG_RAW_ARTIFACT_STORE`` unset）⇒ :func:`raw_artifact_lane.tool` は [] ⇒ tool 集合/MCP surface byte-identical。
* fact_layer.tools() は OFF で raw-artifact ツールを一切足さない（surface byte-identical）。
* ON ⇒ artifact_grep がコード/設定/CSV の中身へ行番号+近傍付きで到達（idx32）、
  analysis_artifact_lookup が適用ハイパラ（idx61）と leaderboard 最良順行（idx62）を返す。
* config.model_params 空でも modeling.py の to_int/to_float コード上デフォルトを適用ハイパラに解決する。
* leaderboard のメトリクス方向（誤差系=昇順 / それ以外=降順）。
"""
from __future__ import annotations

import pytest

from src.rag.agent import raw_artifact_lane as L
from src.rag.index import raw_artifact_store as S


# --------------------------------------------------------------------------- synth store
_MODELING_SRC = (
    "n_estimators = to_int(model_params.get(\"n_estimators\"), 300)\n"
    "learning_rate = to_float(model_params.get(\"learning_rate\"), 0.1)\n"
    "min_samples_leaf = to_int(model_params.get(\"min_samples_leaf\"), 1)\n"
    "model = GradientBoostingRegressor(n_estimators=500, learning_rate=0.3, random_state=random_state)\n"
)
_FEATURES_SRC = (
    "def augment_numeric_interactions(df, enabled, max_numeric_features):\n"
    "    selected_columns = pick_top(df, max_numeric_features)\n"
    "    for a, b in combinations(selected_columns, 2):\n"
    "        df[f'{a}_x_{b}'] = df[a] * df[b]\n"
)


def _file(project, rel, ext, text, *, parsed=None):
    rec = {"project": project, "rel": rel, "name": rel.split("/")[-1], "ext": ext,
           "category": "analysis", "n_chars": len(text), "n_lines": text.count("\n") + 1,
           "text": text}
    if parsed is not None:
        rec["json" if ext == "json" else "csv"] = parsed
    return rec


@pytest.fixture()
def synth_store(monkeypatch):
    kyobashi = "京橋信用ソリューションズ株式会社"
    aoba = "青葉与信マネジメント株式会社"
    aomine = "株式会社青嶺不動産アセットマネジメント"
    files = [
        _file(kyobashi, "プロジェクト/京橋/04.分析/analysis_project/src/modeling.py", "py", _MODELING_SRC),
        _file(aomine, "プロジェクト/青嶺/04.分析/analysis_project/src/features.py", "py", _FEATURES_SRC),
    ]
    lb_rows = [
        {"trial_index": "10", "model_type": "extra_trees", "n_estimators": "500",
         "task_type": "classification", "primary_metric": "f1_macro", "primary_value": "0.60266642"},
        {"trial_index": "6", "model_type": "extra_trees", "n_estimators": "300",
         "task_type": "classification", "primary_metric": "f1_macro", "primary_value": "0.59534963"},
        {"trial_index": "1", "model_type": "random_forest", "n_estimators": "300",
         "task_type": "classification", "primary_metric": "f1_macro", "primary_value": "0.58929290"},
    ]
    cases = [
        {"_kind": "case", "project": kyobashi,
         "artifacts": [{"rel": files[0]["rel"], "name": "modeling.py", "ext": "py"}],
         "config": {"rel": "cfg", "value": {"model_type": "gradient_boosting", "model_params": {},
                                            "random_state": 42}},
         "metrics": {"rel": "m", "value": {"task_type": "classification"}},
         "applied_hyperparams": S.resolve_hyperparams(
             {"model_type": "gradient_boosting", "model_params": {}, "random_state": 42},
             {"task_type": "classification"}, _MODELING_SRC)},
        {"_kind": "case", "project": aoba,
         "artifacts": [], "leaderboard": S._leaderboard_rows(
             _file(aoba, "プロジェクト/青葉/04.分析/leaderboard.csv", "csv", "x",
                   parsed={"header": list(lb_rows[0].keys()), "rows": lb_rows, "n_rows": 3}))},
    ]
    data = {
        "files": files,
        "cases": cases,
        "files_by_project": {kyobashi: [files[0]], aomine: [files[1]]},
        "cases_by_project": {kyobashi: cases[0], aoba: cases[1]},
    }
    monkeypatch.setattr(S, "load", lambda path=None: data)
    monkeypatch.setattr(L, "_bind_project", lambda text: (
        kyobashi if "京橋" in text else aoba if "青葉" in text else aomine if "青嶺" in text else None))
    return data


# --------------------------------------------------------------------------- OFF byte-identical
def test_default_off_tool_empty(monkeypatch):
    monkeypatch.delenv("RAG_RAW_ARTIFACT_STORE", raising=False)
    assert S.enabled() is False
    assert L.enabled() is False
    assert L.tool() == []


def test_fact_layer_off_excludes_raw_artifact(monkeypatch):
    from src.rag.agent import fact_layer
    monkeypatch.setenv("RAG_FACT_LAYER", "1")
    monkeypatch.delenv("RAG_RAW_ARTIFACT_STORE", raising=False)
    names = {t[0] for t in fact_layer.tools()}
    assert "artifact_grep" not in names and "analysis_artifact_lookup" not in names


def test_fact_layer_on_includes_raw_artifact(monkeypatch):
    from src.rag.agent import fact_layer
    monkeypatch.setenv("RAG_FACT_LAYER", "1")
    monkeypatch.setenv("RAG_RAW_ARTIFACT_STORE", "1")
    names = {t[0] for t in fact_layer.tools()}
    assert {"artifact_grep", "analysis_artifact_lookup"} <= names


# --------------------------------------------------------------------------- artifact_grep (idx32 reach)
def test_grep_reaches_code_lines(monkeypatch, synth_store):
    monkeypatch.setenv("RAG_RAW_ARTIFACT_STORE", "1")
    r = L._artifact_grep(case="青嶺不動産", keyword="selected_columns", ext="py")
    assert r["value"]["count"] == 1
    f = r["value"]["files"][0]
    assert "features.py" in f["file"]
    assert any("selected_columns" in "".join(h["context"]) for h in f["hits"])
    assert r["evidence"]["found"] is True


def test_grep_no_match(monkeypatch, synth_store):
    monkeypatch.setenv("RAG_RAW_ARTIFACT_STORE", "1")
    r = L._artifact_grep(case="青嶺不動産", keyword="語XYZ存在しない")
    assert r["value"]["count"] == 0 and r["evidence"]["found"] is False


def test_grep_head_window_without_keyword(monkeypatch, synth_store):
    monkeypatch.setenv("RAG_RAW_ARTIFACT_STORE", "1")
    r = L._artifact_grep(file="modeling.py")
    f = r["value"]["files"][0]
    assert f["hits"][0]["line_no"] == 1 and f["hits"][0]["context"]


# --------------------------------------------------------------------------- hyperparams (idx61)
def test_hyperparams_applies_code_defaults(monkeypatch, synth_store):
    monkeypatch.setenv("RAG_RAW_ARTIFACT_STORE", "1")
    r = L._lookup(case="京橋信用ソリューションズ", kind="hyperparams")
    applied = r["value"]["applied"]
    assert applied["n_estimators"] == 300      # model_params 空 → コード上デフォルト
    assert applied["learning_rate"] == 0.1
    assert applied["random_state"] == 42        # config 由来
    assert r["value"]["task_type"] == "classification"
    # 回帰 GB 分岐のリテラル(500/0.3)は透明性のため code_literals に保持（applied には混入しない）。
    assert "n_estimators" in r["value"]["code_literals"]


def test_resolve_hyperparams_unit():
    res = S.resolve_hyperparams(
        {"model_type": "random_forest", "model_params": {"n_estimators": 700}, "random_state": 77},
        {"task_type": "regression"}, _MODELING_SRC)
    assert res["applied"]["n_estimators"] == 700   # model_params が code default を上書き
    assert res["applied"]["random_state"] == 77
    assert res["model_type"] == "random_forest" and res["task_type"] == "regression"


# --------------------------------------------------------------------------- leaderboard (idx62)
def test_leaderboard_top2_setting_diff(monkeypatch, synth_store):
    monkeypatch.setenv("RAG_RAW_ARTIFACT_STORE", "1")
    r = L._lookup(case="青葉与信", kind="leaderboard")
    rows = r["value"]["rows"]
    assert rows[0]["trial_index"] == "10" and rows[0]["n_estimators"] == "500"
    assert rows[1]["trial_index"] == "6" and rows[1]["n_estimators"] == "300"
    # 上位2件の設定差分 = n_estimators（500 vs 300）
    assert rows[0]["n_estimators"] != rows[1]["n_estimators"]
    assert r["value"]["top2_comparison"]["setting_differences"] == {
        "n_estimators": {"rank1": "500", "rank2": "300"}
    }


def test_leaderboard_error_metric_ascending():
    """rmse など誤差系メトリクスは昇順（最良=最小が先頭）。"""
    rec = {"rel": "lb.csv", "csv": {"header": ["trial_index", "primary_metric", "primary_value"],
           "rows": [{"trial_index": "a", "primary_metric": "rmse", "primary_value": "900"},
                    {"trial_index": "b", "primary_metric": "rmse", "primary_value": "100"}]}}
    lb = S._leaderboard_rows(rec)
    assert lb["lower_is_better"] is True
    assert lb["rows"][0]["trial_index"] == "b"   # 最小 rmse が先頭


def test_lookup_unknown_case(monkeypatch, synth_store):
    monkeypatch.setenv("RAG_RAW_ARTIFACT_STORE", "1")
    r = L._lookup(case="存在しない会社", kind="hyperparams")
    assert r["value"] is None and r["evidence"]["found"] is False


# --------------------------------------------------------------------------- store build (structural)
def test_build_file_parses_json_and_csv(tmp_path, monkeypatch):
    from src.rag.corpus import FileRef
    p = tmp_path / "metrics.json"
    p.write_text('{"task_type": "regression", "rmse": 1.5}', encoding="utf-8")
    ref = FileRef(path=p, project="P", category="analysis", rel="P/metrics.json",
                  name="metrics.json", ext="json")
    rec = S.build_file(ref)
    assert rec is not None and rec["json"]["task_type"] == "regression"

    c = tmp_path / "leaderboard.csv"
    c.write_text("﻿trial,primary_metric,primary_value\n1,f1_macro,0.6\n", encoding="utf-8")
    cref = FileRef(path=c, project="P", category="analysis", rel="P/leaderboard.csv",
                   name="leaderboard.csv", ext="csv")
    crec = S.build_file(cref)
    assert crec is not None and crec["csv"]["rows"][0]["trial"] == "1"   # BOM 除去済み
    assert crec["csv"]["header"][0] == "trial"
