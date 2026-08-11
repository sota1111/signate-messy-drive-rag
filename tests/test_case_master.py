"""SOT-2643 — 案件マスタテーブル事前構築の offline テスト（ネットワーク/LLM 不要）。

重い office 抽出を伴う実コーパスビルドは ``test_real_corpus_build_is_deterministic`` に隔離し（コーパス/
依存が無い環境では skip）、決裁基準ルール導出・precision-first 抽出・スキーマ/読み出し・カバレッジ表・
書き出し決定論の純ロジックは合成データで検証する。
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from config import settings
from src.rag.index import case_master as cm


# --------------------------------------------------------------------------- 決裁基準ルール導出
def test_derive_approval_follows_decree_ladder_medical_and_tm():
    # §2 通常帯（非医療・固定）
    assert cm.derive_approval(2_999_999, False, False)["approval_level"] == "主任承認"
    assert cm.derive_approval(3_000_000, False, False)["apr_code"] == "APR-M1"   # 課長承認
    assert cm.derive_approval(5_000_000, False, False)["apr_code"] == "APR-M2"   # 部長承認
    assert cm.derive_approval(8_000_000, False, False)["apr_code"] == "APR-M3"   # 本部長承認
    # §3.1 医療案件は1段階上（課長→部長）
    assert cm.derive_approval(3_960_000, True, False)["apr_code"] == "APR-M2"
    # §3.2 t&m は最低でも部長承認（金額が主任帯でも）
    assert cm.derive_approval(1_000_000, False, True)["apr_code"] == "APR-M2"
    # 医療 かつ t&m: 1段階上げが本部長に達したら本部長（capは3）
    assert cm.derive_approval(8_000_000, True, True)["apr_code"] == "APR-M3"
    # 主任承認は APR-M 帯の下 → コードなし
    assert cm.derive_approval(1_000_000, False, False)["apr_code"] is None
    # 金額不明は棄権（None）
    assert cm.derive_approval(None, True, True) is None


# --------------------------------------------------------------------------- precision-first 抽出
class _Ref:
    def __init__(self, rel):
        self.rel = rel


def test_first_unique_int_commits_only_on_single_distinct_value():
    pats = [r"最終請求金額（税込）\s*[：:|]?\s*[¥￥]?\s*([0-9][0-9,]*)\s*(?:円|JPY)?"]
    # 同一値が複数行 → 一意なので commit（コンマ書式も許容）
    texts = [(_Ref("プロジェクト/A/06.報告/FR.pptx"),
              "最終請求金額（税込）：3,850,000円\n最終請求金額（税込） | 3,850,000 JPY")]
    got = cm._first_unique_int(texts, pats)
    assert got is not None and got[0] == 3_850_000
    assert got[1]["doc_id"] == "プロジェクト/A/06.報告/FR.pptx" and "snippet" in got[1]
    # 異なる値が2つ → 曖昧なので棄権（None）
    ambiguous = [(_Ref("x"), "最終請求金額（税込）：1,000円\n最終請求金額（税込）：2,000円")]
    assert cm._first_unique_int(ambiguous, pats) is None
    # 該当なし → None
    assert cm._first_unique_int([(_Ref("x"), "無関係テキスト")], pats) is None


# --------------------------------------------------------------------------- cell / source contract
def test_cell_and_missing_shape():
    c = cm._cell(42, cm._source("doc.md", snippet="line", basis="b"))
    assert c["value"] == 42 and c["source"]["doc_id"] == "doc.md"
    m = cm._missing("理由")
    assert m["value"] is None and m["source"] is None and m["reason"] == "理由"
    # source は doc_id 必須、余分な媒体ロケータのみ付く
    s = cm._source("d.xlsx", sheet="Sheet1", cell="B2")
    assert s == {"doc_id": "d.xlsx", "sheet": "Sheet1", "cell": "B2"}


# --------------------------------------------------------------------------- スキーマ / 読み出し / フィルタ
@pytest.fixture()
def table(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    out = tmp_path / "case_master.jsonl"
    monkeypatch.setattr(cm, "default_out_path", lambda: out)
    recs = [
        cm.CaseRecord("案件A", "AAA", {
            "status": cm._cell("完了", cm._source("A/FR.pptx")),
            "apr_code": cm._cell("APR-M2", cm._source(cm._DECREE_DOC_ID)),
            "train_rows": cm._cell(12000, cm._source("A/train.csv")),
        }),
        cm.CaseRecord("案件B", "BBB", {
            "status": cm._cell("完了", cm._source("B/FR.pptx")),
            "apr_code": cm._cell("APR-M1", cm._source(cm._DECREE_DOC_ID)),
            "train_rows": cm._missing("表なし"),
        }),
    ]
    cm.reset_cache()
    cm.write_table(recs, out)
    return out


def test_write_table_is_reproducible_and_schema_headed(table: Path):
    first = table.read_bytes()
    header = json.loads(table.read_text(encoding="utf-8").splitlines()[0])
    assert header == {"schema": cm.SCHEMA, "version": cm.SCHEMA_VERSION}
    rows = cm.load(table)
    assert [r["case_id"] for r in rows] == ["案件A", "案件B"]  # case_id ソート済み
    # 同じレコードを書き直すと byte-identical（決定論）
    recs = [cm.CaseRecord(r["case_id"], r["abbrev"], r["attributes"]) for r in rows]
    cm.write_table(recs, table)
    assert table.read_bytes() == first


def test_filter_cases_minimal_read_api(table: Path):
    hits = cm.filter_cases(path=table, status="完了", apr_code="APR-M2")
    assert [r["case_id"] for r in hits] == ["案件A"]
    # 不一致 / 未知属性 → 空（劣化なし）
    assert cm.filter_cases(path=table, status="進行中") == []
    assert cm.filter_cases(path=table, nonexistent="x") == []


# --------------------------------------------------------------------------- カバレッジ表（診断）
def test_enum_hard_core_coverage_reports_gaps():
    recs = [
        cm.CaseRecord("案件A", "AAA", {
            "status": cm._cell("完了", cm._source("x")),
            "apr_code": cm._cell("APR-M2", cm._source("x")),
            "contract_amount_incl_tax": cm._cell(5_000_000, cm._source("x")),
            "proposal_amount_incl_tax": cm._missing("抽出不能"),
            "fr_amount_incl_tax": cm._missing("抽出不能"),
            "train_rows": cm._cell(12000, cm._source("x")),
        }),
    ]
    cov = cm.enum_hard_core_coverage(recs)
    # idx38 は apr_code + contract_amount が全案件で埋まれば解ける
    assert cov["idx38"]["solvable_from_case_master"] is True
    # idx67 は proposal/fr が欠測 → 解けない（配線 issue の前提情報）
    assert cov["idx67"]["solvable_from_case_master"] is False
    assert cov["idx67"]["attribute_coverage"]["fr_amount_incl_tax"]["fully_covered"] is False
    # idx32/45 は案件マスタ標準属性の対象外 → need 空・非solvable
    assert cov["idx32"]["required_attributes"] == [] and cov["idx32"]["solvable_from_case_master"] is False


# --------------------------------------------------------------------------- opt-in default OFF
def test_enabled_defaults_off(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.delenv("RAG_CASE_MASTER", raising=False)
    assert cm.enabled() is False
    monkeypatch.setenv("RAG_CASE_MASTER", "1")
    assert cm.enabled() is True


# --------------------------------------------------------------------------- 実コーパスビルド（重い・隔離）
@pytest.mark.skipif(not settings.CORPUS_DIR.exists(), reason="corpus not present")
def test_real_corpus_build_is_deterministic_and_universe_certified(tmp_path: Path):
    try:
        import docx  # noqa: F401 — office extraction deps required for the real build
    except Exception:
        pytest.skip("office extraction deps not installed")
    out1, out2 = tmp_path / "cm1.jsonl", tmp_path / "cm2.jsonl"
    s1 = cm.build(out=out1, write_report=False)
    s2 = cm.build(out=out2, write_report=False)
    assert out1.read_bytes() == out2.read_bytes()          # 2回ビルドで byte-identical
    cert = s1["report"]["certificate"]
    assert cert["row_count_equals_universe"] is True         # 行数 = universe（completeness certificate）
    assert cert["case_count"] == s1["cases"] >= 1
