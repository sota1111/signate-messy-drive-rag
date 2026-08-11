"""SOT-2646 — 版ペア差分事前確定ストアの offline テスト（ネットワーク/LLM 不要）。

重い office 抽出を伴う実コーパスビルドは ``test_real_corpus_build_is_deterministic`` に隔離し（コーパス/依存が
無い環境では skip）、ペア列挙の和集合・重複排除、変更属性の決定論付与、除外フィルタ、スキーマ/読み出し、書き出し
決定論、カバレッジ診断の純ロジックは合成 ``VersionPair`` / ``RankedChange`` で検証する。
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from config import settings
from src.rag import diffpair
from src.rag.corpus import FileRef
from src.rag.index import diff_store as ds


def _ref(rel: str, project: str = "案件A", ext: str = "pptx") -> FileRef:
    return FileRef(path=Path(rel), project=project, category="", rel=rel,
                   name=rel.rsplit("/", 1)[-1], ext=ext)


def _pair(old_rel: str, new_rel: str, base: str = "提案書", basis: str = "rev-suffix",
          project: str = "案件A", ext: str = "pptx") -> diffpair.VersionPair:
    return diffpair.VersionPair(_ref(old_rel, project, ext), _ref(new_rel, project, ext), base, basis)


def _rc(before: str, after: str, kind: str = "modify", label: str = "") -> diffpair.RankedChange:
    """Classify a synthetic change through the real deterministic classifier (no LLM)."""
    return diffpair.classify_change(diffpair.Change(label, before, after, kind))


# --------------------------------------------------------------------------- 変更属性（決定論）
def test_change_attributes_status_transition():
    rc = _rc("未着手", "完了", "modify", label="ステータス")
    attrs = ds.change_attributes(rc)
    assert "status_transition" in attrs and "modification" in attrs


def test_change_attributes_status_transition_absorbs_notation_variants():
    # 表記ゆれ（空白・大小・ToDo）も _norm 経由で吸収される
    assert "status_transition" in ds.change_attributes(_rc("未 着手", "完 了"))
    assert "status_transition" in ds.change_attributes(_rc("ToDo", "Done"))
    # 状態語でない値の変更は status_transition にしない
    assert "status_transition" not in ds.change_attributes(_rc("山田 太郎", "佐藤 花子"))


def test_change_attributes_assignee_and_numeric_and_formatting():
    # 担当ラベル文脈 + person → assignee_change
    a = ds.change_attributes(_rc("池田 直哉", "小林 直樹", label="QAレビューア"))
    assert "assignee_change" in a and "person_name_change" in a
    # 数値/割合変更
    n = ds.change_attributes(_rc("売上 100万円", "売上 120万円"))
    assert "numeric_change" in n and "money_change" in n
    # 書式のみ（ページ番号）は formatting_only（BOILERPLATE）
    f = ds.change_attributes(_rc("", "2", "add"))
    assert "formatting_only" in f


def test_change_attributes_are_deterministic_and_deduped():
    rc = _rc("interest_rate", "interest_rate 修正", "modify", label="列名")
    a1, a2 = ds.change_attributes(rc), ds.change_attributes(rc)
    assert a1 == a2 and len(a1) == len(set(a1))


# --------------------------------------------------------------------------- ペア列挙（和集合・重複排除・source）
def test_enumerate_pairs_unions_and_dedups_with_sources(monkeypatch: pytest.MonkeyPatch):
    common = _pair("dir/提案書_v1.pptx", "dir/提案書_v3.pptx")
    only_registry = _pair("d/報告_old.pptx", "d/報告.pptx", base="報告", basis="registry-family")
    monkeypatch.setattr(diffpair, "find_pairs", lambda *a, **k: [common])
    monkeypatch.setattr(diffpair, "_registry_family_pairs", lambda *a, **k: [common, only_registry])
    pairs = ds.enumerate_pairs()
    by_new = {p.new.rel: srcs for p, srcs in pairs}
    assert len(pairs) == 2                               # 重複排除される
    assert by_new["dir/提案書_v3.pptx"] == ["filename", "registry-family"]  # 両列挙器由来
    assert by_new["d/報告.pptx"] == ["registry-family"]  # registry のみ（_old×無印 latest 補完）


def test_enumerate_pairs_survives_one_enumerator_failing(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(diffpair, "find_pairs", lambda *a, **k: (_ for _ in ()).throw(RuntimeError("boom")))
    monkeypatch.setattr(diffpair, "_registry_family_pairs", lambda *a, **k: [_pair("a/x_v1.pptx", "a/x_v2.pptx")])
    pairs = ds.enumerate_pairs()
    assert len(pairs) == 1 and pairs[0][1] == ["registry-family"]


# --------------------------------------------------------------------------- pair_record（全 candidate 保持）
def test_pair_record_keeps_all_ranked_candidates_with_attributes(monkeypatch: pytest.MonkeyPatch):
    pair = _pair("d/s_r1.xlsx", "d/s_r2.xlsx", base="s", ext="xlsx")
    ranked = [
        _rc("池田 直哉", "小林 直樹", label="担当"),
        _rc("未着手", "完了"),
        _rc("", "版 v2", "add"),
    ]
    monkeypatch.setattr(diffpair, "rank_changes", lambda p: ranked)
    rec = ds.pair_record(pair, ["filename"])
    assert rec["alignment_ok"] is True and rec["change_count"] == 3
    assert [c["rank"] for c in rec["changes"]] == [0, 1, 2]     # 順位付き
    assert "assignee_change" in rec["changes"][0]["attributes"]
    assert "status_transition" in rec["changes"][1]["attributes"]


def test_pair_record_alignment_failure_is_not_faked(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(diffpair, "rank_changes", lambda p: None)   # 片方読めない / 構造抽出不可
    rec = ds.pair_record(_pair("a/x_v1.pptx", "a/x_v2.pptx"), ["filename"])
    assert rec["alignment_ok"] is False and rec["changes"] == [] and rec["change_count"] == 0


# --------------------------------------------------------------------------- 除外フィルタ（質問側条件）
def test_filter_changes_excludes_and_requires():
    record = {"changes": [
        {"rank": 0, "attributes": ["modification", "assignee_change", "person_name_change"]},
        {"rank": 1, "attributes": ["modification", "status_transition"]},
        {"rank": 2, "attributes": ["addition", "formatting_only"]},
    ]}
    # idx95「未着手→完了 を除く」→ status_transition を除外
    kept = ds.filter_changes(record, exclude_attributes={"status_transition"})
    assert [c["rank"] for c in kept] == [0, 2]
    # 担当者変更だけ要求
    req = ds.filter_changes(record, require_attributes={"assignee_change"})
    assert [c["rank"] for c in req] == [0]


# --------------------------------------------------------------------------- スキーマ / 読み出し / 決定論
@pytest.fixture()
def store(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    out = tmp_path / "diff_store.jsonl"
    monkeypatch.setattr(ds, "default_out_path", lambda: out)
    monkeypatch.setattr(diffpair, "find_pairs", lambda *a, **k: [_pair("d/提案書_v1.pptx", "d/提案書_v3.pptx")])
    monkeypatch.setattr(diffpair, "_registry_family_pairs", lambda *a, **k: [])
    monkeypatch.setattr(diffpair, "rank_changes", lambda p: [_rc("池田 直哉", "小林 直樹", label="担当")])
    ds.reset_cache()
    ds.build(write_report=False)
    return out


def test_write_store_is_reproducible_and_schema_headed(store: Path):
    first = store.read_bytes()
    header = json.loads(store.read_text(encoding="utf-8").splitlines()[0])
    assert header == {"schema": ds.SCHEMA, "version": ds.SCHEMA_VERSION}
    rows = ds.load(store)
    assert rows and rows[0]["new_rel"] == "d/提案書_v3.pptx" and rows[0]["change_count"] == 1
    # 同じレコードを書き直すと byte-identical（決定論）
    ds.write_store(rows, store)
    assert store.read_bytes() == first


def test_lookup_pair_and_zero_regression(store: Path):
    assert [r["new_rel"] for r in ds.lookup_pair(new_rel="提案書_v3", path=store)] == ["d/提案書_v3.pptx"]
    assert ds.lookup_pair(new_rel="提案書_v3", project="別案件", path=store) == []
    # 未知ペア / 無アーティファクト → 空（劣化なし）
    assert ds.lookup_pair(new_rel="存在しない", path=store) == []
    assert ds.lookup_pair(new_rel="提案書", path=Path("/nonexistent.jsonl")) == []


# --------------------------------------------------------------------------- カバレッジ表（診断・honest）
def test_version_hard_core_coverage_is_honest():
    records = [{
        "old_rel": "プロジェクト/白峰信用リスク評価株式会社/00.提案/提案書old.pptx",
        "new_rel": "プロジェクト/白峰信用リスク評価株式会社/00.提案/提案書.pptx",
        "sources": ["registry-family"], "alignment_ok": True, "change_count": 1,
    }]
    cov = ds.version_hard_core_coverage(records)
    # idx0 は白峰 _old×無印 latest ペアが列挙され diff 確定
    assert cov["idx0"]["enumerated"] is True and cov["idx0"]["alignment_ok"] is True
    assert cov["idx0"]["sources"] == ["registry-family"]
    # idx22 は .ipynb 版ペアで office diff 対象外 → 未列挙を honest に報告（欠測を偽装しない）
    assert cov["idx22"]["enumerated"] is False and cov["idx22"]["note"]
    assert all(cov[i]["gist"] and cov[i]["note"] for i in ("idx0", "idx1", "idx14", "idx22", "idx95"))


# --------------------------------------------------------------------------- opt-in default OFF
def test_enabled_defaults_off(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.delenv("RAG_DIFF_STORE", raising=False)
    assert ds.enabled() is False
    monkeypatch.setenv("RAG_DIFF_STORE", "1")
    assert ds.enabled() is True


# --------------------------------------------------------------------------- 実コーパスビルド（重い・隔離）
@pytest.mark.skipif(not settings.CORPUS_DIR.exists(), reason="corpus not present")
def test_real_corpus_build_is_deterministic(tmp_path: Path):
    try:
        import pptx  # noqa: F401 — office extraction deps required for the real build
    except Exception:
        pytest.skip("office extraction deps not installed")
    out1, out2 = tmp_path / "d1.jsonl", tmp_path / "d2.jsonl"
    s1 = ds.build(out=out1, write_report=False)
    s2 = ds.build(out=out2, write_report=False)
    assert out1.read_bytes() == out2.read_bytes()          # 2回ビルドで byte-identical
    assert s1["pairs"] == s2["pairs"] >= 0
    assert s1["report"]["certificate"]["question_independent"] is True
    # 全 version-family ペアが列挙され、idx1/14/95 の版ペアが diff 確定していること（idx22 は ipynb で対象外）
    cov = s1["report"]["version_hard_core_coverage"]
    for idx in ("idx1", "idx14", "idx95"):
        assert cov[idx]["enumerated"] and cov[idx]["alignment_ok"], idx
