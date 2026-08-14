"""SOT-2712 — version_diff direct-commit の対象クラス拡大（サブフラグ既定 OFF）の決定論ユニットテスト.

対象:
  * ``RAG_VDIFF_DC_NOCHANGE``  — 実質変更ゼロ（全 LAYOUT/SURFACE）ペア ⇒ 裸形式「該当なし」（idx9）。
  * ``RAG_VDIFF_DC_COLRENAME`` — schema_name_change 群 ⇒ 列名変更クラスの逐語コミット（idx14）。

不変条件（回帰ゼロ）:
  * 全サブフラグ OFF ⇒ resolve() は None（SOT-2706 と byte-identical）。
  * ``RAG_VDIFF_DIRECT_COMMIT`` 自体 OFF ⇒ サブフラグに関係なく None。
  * gold 文言はハードコードしない（summary は old/new 実物から導出）。

diff_store の対象ペアが未ビルドの環境では skip（欠測を偽装しない）。LLM・Gemini・ネットワークは一切不要。
"""
from __future__ import annotations

import pytest

from src.rag.agent import vdiff_direct_lane as L
from src.rag.index import diff_store

# 実コーパスの対象ペアに束縛する質問（質問文はストアの版種別/ファイル名トークンで一意に解決される）。
Q_NOCHANGE = "青葉与信マネジメントの最終報告資料の最新版になる際に修正されたもののうち、案件遂行に関連する変更を挙げてください。"
Q_COLRENAME = "青葉与信マネジメントの提案書_v1.pptxから提案書_v3.pptxに修正されたもののうち、案件遂行に関連する変更を挙げてください。"


def _has_pair(pred) -> bool:
    try:
        return any(pred(r) for r in diff_store.load())
    except Exception:  # noqa: BLE001
        return False


_HAS_NOCHANGE_PAIR = _has_pair(
    lambda r: "06.報告書" in r.get("old_rel", "") and "青葉" in r.get("project", "")
    and not any(c.get("intent") == "SUBSTANTIVE" for c in r.get("changes", []))
)
_HAS_COLRENAME_PAIR = _has_pair(
    lambda r: "提案書_v1" in r.get("old_rel", "") and "青葉" in r.get("project", "")
    and any("schema_name_change" in (c.get("attributes") or []) for c in r.get("changes", []))
)


@pytest.fixture
def _flags_on(monkeypatch):
    monkeypatch.setenv("RAG_VDIFF_DIRECT_COMMIT", "1")
    monkeypatch.setenv("RAG_VDIFF_DC_NOCHANGE", "1")
    monkeypatch.setenv("RAG_VDIFF_DC_COLRENAME", "1")


@pytest.mark.skipif(not _HAS_NOCHANGE_PAIR, reason="no zero-substantive diff pair in the store")
def test_no_change_verdict_commits_bare_gairanashi(_flags_on):
    res = L.resolve(Q_NOCHANGE)
    assert res is not None
    assert res["value"] == "該当なし"  # 裸形式（括弧・接頭辞なし）
    assert res["method"]["selection"] == "no_change_verdict_direct_commit"
    assert res["evidence"]["substantive_change"] is False


@pytest.mark.skipif(not _HAS_COLRENAME_PAIR, reason="no schema_name_change diff pair in the store")
def test_column_rename_class_commits_derived_summary(_flags_on):
    res = L.resolve(Q_COLRENAME)
    assert res is not None
    assert res["method"]["selection"] == "column_rename_class_direct_commit"
    val = res["value"]
    # 実データから導出した列名（interest_rate 等）が列挙され、変換記述は「アンダースコア表記」。
    assert "interest_rate" in val and "loan_status" in val
    assert "アンダースコア" in val
    # gold 文言のハードコードでないこと（列名は old/new 実物由来 = evidence にも残る）。
    assert res["evidence"]["renamed_columns"]


@pytest.mark.skipif(not (_HAS_NOCHANGE_PAIR and _HAS_COLRENAME_PAIR),
                    reason="target pairs absent")
def test_subflags_off_is_byte_identical_none(monkeypatch):
    """サブフラグ OFF（base direct-commit のみ）⇒ 拡張は発火しない（None = 従来経路へ委譲）。"""
    monkeypatch.setenv("RAG_VDIFF_DIRECT_COMMIT", "1")
    monkeypatch.setenv("RAG_VDIFF_DC_NOCHANGE", "0")
    monkeypatch.setenv("RAG_VDIFF_DC_COLRENAME", "0")
    assert L.resolve(Q_NOCHANGE) is None
    assert L.resolve(Q_COLRENAME) is None


def test_base_flag_off_disables_everything(monkeypatch):
    monkeypatch.setenv("RAG_VDIFF_DIRECT_COMMIT", "0")
    monkeypatch.setenv("RAG_VDIFF_DC_NOCHANGE", "1")
    monkeypatch.setenv("RAG_VDIFF_DC_COLRENAME", "1")
    assert L.resolve(Q_NOCHANGE) is None
    assert L.resolve(Q_COLRENAME) is None
