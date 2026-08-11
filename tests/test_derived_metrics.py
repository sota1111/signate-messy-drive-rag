"""SOT-2645 — 派生メトリクス事前計算ストアの offline テスト（ネットワーク/LLM 不要）。

重い office 抽出を伴う実コーパスビルドは ``test_real_corpus_*`` に隔離し（コーパス/依存が無い環境では
skip）、標準統計・分位点・相関・ヒストグラム・比/差・OLS 予測・F1 閾値スイープ・独立検算(fail-closed)・
スキーマ/読み出し・書き出し決定論の純ロジックは合成 CSV / 合成配列で検証する。
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from config import settings
from src.rag.corpus import FileRef
from src.rag.index import derived_metrics as dm


def _ref(path: Path, project: str, ext: str = "csv") -> FileRef:
    rel = f"プロジェクト/{project}/03.データ/train.{ext}"
    return FileRef(path=path, project=project, category="data", rel=rel, name=f"train.{ext}", ext=ext)


# --------------------------------------------------------------------------- pure numeric primitives
def test_pure_stats_matches_known_values():
    st = dm._pure_stats([1.0, 2.0, 3.0, 4.0])
    assert st["count"] == 4 and st["min"] == 1.0 and st["max"] == 4.0
    assert st["mean"] == 2.5 and st["sum"] == 10.0
    # 母標準偏差(ddof=0): sqrt(mean((x-2.5)^2)) = sqrt(1.25)
    assert abs(st["std"] - 1.25 ** 0.5) < 1e-12


def test_pure_percentile_linear_interpolation():
    xs = [0.0, 1.0, 2.0, 3.0, 4.0]
    assert dm._pure_percentile(xs, 50) == 2.0
    assert dm._pure_percentile(xs, 90) == pytest.approx(3.6)   # linear interp: rank=3.6
    assert dm._pure_percentile([5.0], 90) == 5.0               # 単一要素


def test_pure_pearson_perfect_and_constant():
    assert dm._pure_pearson([1, 2, 3], [2, 4, 6]) == pytest.approx(1.0)
    assert dm._pure_pearson([1, 2, 3], [3, 2, 1]) == pytest.approx(-1.0)
    import math
    assert math.isnan(dm._pure_pearson([1, 1, 1], [1, 2, 3]))  # 定数列 → nan（欠測を偽装しない）


def test_f1_from_counts():
    assert dm._f1_from_counts(0, 0, 0) == 0.0
    assert dm._f1_from_counts(2, 0, 0) == 1.0
    assert dm._f1_from_counts(1, 1, 1) == pytest.approx(0.5)   # 2*1/(2+1+1)


# --------------------------------------------------------------------------- dual verification (fail-closed)
def test_verifier_commits_on_match_drops_on_mismatch():
    vf = dm._Verifier()
    assert vf.commit("a", 1.0, 1.0) == 1.0                     # 一致 → 保存
    assert vf.commit("b", 1.0, 2.0) is None                    # 不一致 → 棄却（fail-closed）
    assert vf.computed == 2 and len(vf.mismatches) == 1
    assert vf.mismatches[0]["metric"] == "b"


def test_verifier_nan_pair_is_match():
    vf = dm._Verifier()
    assert vf.commit("nan", float("nan"), float("nan")) is None  # 両 nan → round が None を返す
    assert len(vf.mismatches) == 0                               # 一致扱い（棄却ではない）


# --------------------------------------------------------------------------- threshold sweep (idx57 型)
def test_threshold_sweep_exact_best_f1():
    import numpy as np
    # 完全分離: scores が y と単調 → 最良閾値で F1=1.0
    scores = [0.1, 0.2, 0.8, 0.9]
    y = [0, 0, 1, 1]
    sw = dm._sweep_thresholds(np, scores, y)
    assert sw is not None
    assert sw["best_f1"] == pytest.approx(1.0)
    assert sw["confusion_at_best"] == {"tp": 2, "fp": 0, "fn": 0}
    assert sw["positives"] == 2 and sw["negatives"] == 2


def test_threshold_sweep_none_when_not_binary():
    import numpy as np
    assert dm._sweep_thresholds(np, [0.1, 0.2, 0.3], [0, 1, 2]) is None   # 3クラス → 棄権
    assert dm._sweep_thresholds(np, [0.1, 0.2], [1, 1]) is None           # 片側のみ → 棄権


# --------------------------------------------------------------------------- OLS end-to-end (idx63 型)
def test_compute_case_ols_prediction_and_stats(tmp_path: Path):
    # y = 2*x1 + 3 に近い線形。id=0 の予測が OLS で復元されることを確認。
    csv = tmp_path / "train.csv"
    csv.write_text(
        "id,x1,y\n"
        "0,1,5\n1,2,7\n2,3,9\n3,4,11\n4,5,13\n5,6,15\n",
        encoding="utf-8")
    ref = _ref(csv, "案件LIN")
    rec = dm.compute_case(ref, [ref])
    assert rec is not None
    d = rec.to_dict()
    assert d["row_count"] == 6
    assert set(d["numeric_columns"]) == {"id", "x1", "y"}
    # 列統計は独立検算を全通過（mismatch ゼロ）
    assert d["verification"]["mismatches_dropped"] == 0
    # y は連続値（多クラス）→ 二値でないので sweep は None、しかし OLS 予測は出る
    model = d["model"]
    assert model["present"] is True and model["target"] == "y"
    assert model["features"] == ["x1"]          # id 的な列は特徴量から除外
    # y = 2 x1 + 3 → id=0 (x1=1) の予測 = 5
    assert model["prediction_id0"]["prediction"] == pytest.approx(5.0, abs=1e-6)
    assert model["coefficients"]["x1"] == pytest.approx(2.0, abs=1e-6)
    # coef_source は OLS フィットである旨を honest に明示（gold 由来ではない）
    assert "ols_fit" in model["coef_source"]["basis"]


def test_compute_case_binary_target_has_sweep(tmp_path: Path):
    csv = tmp_path / "train.csv"
    rows = ["id,x,target"]
    for i in range(20):
        rows.append(f"{i},{i},{1 if i >= 10 else 0}")
    csv.write_text("\n".join(rows) + "\n", encoding="utf-8")
    ref = _ref(csv, "案件BIN")
    rec = dm.compute_case(ref, [ref])
    d = rec.to_dict()
    assert d["model"]["present"] is True
    sw = d["model"]["threshold_sweep"]
    assert sw is not None and sw["best_f1"] == pytest.approx(1.0)   # 線形分離 → F1=1


def test_compute_case_no_numeric_columns_returns_none(tmp_path: Path):
    csv = tmp_path / "train.csv"
    csv.write_text("name,note\nfoo,bar\nbaz,qux\n", encoding="utf-8")
    ref = _ref(csv, "案件TXT")
    assert dm.compute_case(ref, [ref]) is None


# --------------------------------------------------------------------------- write / load / determinism
@pytest.fixture()
def built_store(tmp_path: Path) -> Path:
    csv = tmp_path / "train.csv"
    csv.write_text("id,x,target\n0,1,0\n1,2,0\n2,3,1\n3,4,1\n", encoding="utf-8")
    ref = _ref(csv, "案件A")
    rec = dm.compute_case(ref, [ref])
    out = tmp_path / "derived.jsonl"
    dm.write_store([rec], out)
    return out


def test_write_is_deterministic_and_schema_headed(built_store: Path):
    first = built_store.read_bytes()
    header = json.loads(first.decode().splitlines()[0])
    assert header == {"schema": dm.SCHEMA, "version": dm.SCHEMA_VERSION}
    rows = dm.load(built_store)
    assert len(rows) == 1 and rows[0]["case_id"] == "案件A"
    # 同じレコードを書き直すと byte-identical（決定論）
    from src.rag.index.derived_metrics import CaseMetrics
    recs = [CaseMetrics(**{k: rows[0][k] for k in (
        "case_id", "train_file", "row_count", "numeric_columns", "column_stats",
        "correlations", "histograms", "ratios", "model", "verification", "sources")})]
    dm.write_store(recs, built_store)
    assert built_store.read_bytes() == first


def test_case_metrics_read_api_zero_regression(built_store: Path):
    assert dm.case_metrics("案件A", path=built_store)["case_id"] == "案件A"
    assert dm.case_metrics("案件", path=built_store)["case_id"] == "案件A"   # 部分一致
    assert dm.case_metrics("案件Z", path=built_store) is None               # 未知案件
    assert dm.case_metrics("案件A", path=Path("/nonexistent.jsonl")) is None  # 無アーティファクト


# --------------------------------------------------------------------------- opt-in default OFF
def test_enabled_defaults_off(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.delenv("RAG_DERIVED_METRICS", raising=False)
    assert dm.enabled() is False
    monkeypatch.setenv("RAG_DERIVED_METRICS", "1")
    assert dm.enabled() is True


# --------------------------------------------------------------------------- hard-core coverage（診断・honest）
def test_derived_hard_core_coverage_is_honest(tmp_path: Path):
    # 青葉与信 相当の合成: model + sweep が立つと idx57/idx63 が covered=True になる
    csv = tmp_path / "train.csv"
    rows = ["id,x,loan_status"]
    for i in range(30):
        rows.append(f"{i},{i},{1 if i >= 20 else 0}")
    csv.write_text("\n".join(rows) + "\n", encoding="utf-8")
    ref = _ref(csv, "青葉与信マネジメント株式会社")
    rec = dm.compute_case(ref, [ref])
    cov = dm.derived_hard_core_coverage([rec])
    # idx57/idx63 は青葉与信にマッチしモデルメトリクスで covered
    assert cov["idx57"]["matched_case"] == "青葉与信マネジメント株式会社"
    assert cov["idx57"]["fully_covered"] is True
    assert cov["idx63"]["fully_covered"] is True
    # idx40/idx50/idx97 は本ストア対象外を honest に明示（欠測を偽装しない）
    for i in ("idx40", "idx50", "idx97"):
        assert cov[i]["solvable_from_derived_store"] is False and cov[i]["note"]


# --------------------------------------------------------------------------- 実コーパスビルド（重い・隔離）
@pytest.mark.skipif(not settings.CORPUS_DIR.exists(), reason="corpus not present")
def test_real_corpus_build_is_deterministic(tmp_path: Path):
    out1, out2 = tmp_path / "dm1.jsonl", tmp_path / "dm2.jsonl"
    s1 = dm.build(out=out1, write_report=False)
    s2 = dm.build(out=out2, write_report=False)
    assert out1.read_bytes() == out2.read_bytes()          # 2回ビルドで byte-identical
    assert s1["cases"] == s2["cases"] >= 0
    # 独立検算の不一致は保存されない（fail-closed）
    assert s1["report"]["verification_summary"]["mismatches_dropped"] == 0


@pytest.mark.skipif(not settings.CORPUS_DIR.exists(), reason="corpus not present")
def test_real_corpus_derived_hard_core_idx57_idx63_covered():
    """回帰基準: 実データで idx57(best_f1) / idx63(id=0 予測) が gold と小数第5位で一致する。"""
    from src.rag.corpus import walk
    refs = walk()
    ref = dm._train_ref("青葉与信マネジメント株式会社", refs)
    if ref is None:
        pytest.skip("青葉与信 train not present")
    rec = dm.compute_case(ref, refs)
    model = rec.to_dict()["model"]
    assert model["present"] is True
    assert round(model["prediction_id0"]["prediction"], 5) == 0.15002        # idx63 gold
    assert round(model["threshold_sweep"]["best_f1"], 5) == 0.42396          # idx57 gold
