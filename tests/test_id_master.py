"""SOT-2644 — IDマスタ逆引き表事前構築の offline テスト（ネットワーク/LLM 不要）。

重い office 抽出を伴う実コーパスビルドは ``test_real_corpus_build_is_deterministic`` に隔離し（コーパス/
依存が無い環境では skip）、ID 抽出・種別分類（列ヘッダ/ラベル/接頭辞）・原文/隣接列格納・スキーマ/読み出し・
書き出し決定論の純ロジックは合成抽出テキストで検証する。
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from config import settings
from src.rag.corpus import FileRef
from src.rag.index import id_master as idm


def _ref(rel: str, project: str = "案件A", ext: str = "xlsx") -> FileRef:
    return FileRef(path=Path(rel), project=project, category="", rel=rel,
                   name=rel.rsplit("/", 1)[-1], ext=ext)


# --------------------------------------------------------------------------- ID 抽出パターン
def test_id_candidate_matches_id_codes_not_noise():
    def norms(s: str) -> list[str]:
        return [idm.id_norm(m.group("pre"), m.group("num")) for m in idm._ID_CAND.finditer(s)]
    # 正: 各 ID 体系（ハイフンあり/なし）
    assert norms("T09 と A10、M-01、WBS-01、CP1、AI-3") == ["T09", "A10", "M01", "WBS01", "CP1", "AI3"]
    # 負: 合成コードの後半（APR-M2 の "M2"）を拾わない（"-" 直後を弾く）
    assert norms("承認は APR-M2 が必要") == []
    # 負: 5文字以上の英字 + 数字（Sheet1 の "Sheet" は 5 文字 → 非マッチ）/ 数字のみ(2025)
    assert norms("Sheet1 の 2025 年") == []
    # 負: メトリクス名などの英数字連続（AUC-ROC は数字末尾なし / F1-macro は数字の直後がハイフンで失格）
    assert norms("AUC-ROC / F1-macro") == []
    # 正: 独立トークンの F1（後続が区切り）は候補になる（種別は文脈で決まる）
    assert norms("課題 F1 を参照") == ["F1"]


# --------------------------------------------------------------------------- 種別分類（決定論）
def test_kind_resolution_header_over_label_over_prefix():
    # 列ヘッダ優先: ヘッダに「タスクID」→ prefix が M でも task
    assert idm._resolve_kind("M", "行内容", "タスクID") == ("task", "column_header")
    # 行内ラベル: ヘッダ無しで行に「マイルストーン」→ milestone
    assert idm._resolve_kind("X", "マイルストーン M01 完了", None) == ("milestone", "line_label")
    # 接頭辞: ラベル無し、既知接頭辞 CP → checkpoint（行にラベル語なし）
    assert idm._resolve_kind("CP", "CP1 レビュー実施", None) == ("checkpoint", "prefix")
    # 未知接頭辞かつラベル無し → unknown / pattern_only（欠測を偽装しない）
    assert idm._resolve_kind("ZZ", "ZZ01 なにか", None) == ("unknown", "pattern_only")


# --------------------------------------------------------------------------- xlsx 行: 原文 + 隣接列 + ヘッダ
def test_scan_doc_captures_row_text_and_adjacent_columns():
    text = (
        "[シート: タスク一覧]  範囲 A1:D3\n"
        "タスクID | 内容 | 担当者 | 期日\n"
        "T09 | 前処理パイプライン構築 | 山田 | 2025-07-01\n"
        "T10 | モデル学習 | 佐藤 | 2025-07-15\n"
    )
    occ = idm.scan_doc(_ref("プロジェクト/案件A/02.計画/wbs.xlsx"), text)
    by_id = {o.id_norm: o for o in occ}
    assert set(by_id) == {"T09", "T10"}
    t09 = by_id["T09"]
    assert t09.id_kind == "task" and t09.kind_basis == "column_header"     # 列ヘッダ「タスクID」で分類
    assert t09.sheet == "タスク一覧" and t09.column_index == 1
    assert t09.column_header == "タスクID"
    # 原文行全文 + 隣接列（内容/担当/期日）を保持 —「そのまま抜き出して」に応えられる粒度
    assert "前処理パイプライン構築" in t09.line and "山田" in t09.line
    assert t09.columns == ("T09", "前処理パイプライン構築", "山田", "2025-07-01")


def test_scan_doc_does_not_use_an_adjacent_cell_label_for_kind():
    text = (
        "[シート: マイルストーン]\n"
        "ID | 関連会議 | 説明\n"
        "MS1 | M01 キックオフ | KPIとスコープを確定\n"
    )
    occ = idm.scan_doc(_ref("案件/plan.xlsx"), text)
    by_id = {o.id_norm: o for o in occ}
    assert by_id["M01"].id_kind == "milestone"
    assert by_id["M01"].kind_basis == "prefix"


def test_scan_doc_tracks_slide_and_page_locators():
    text = "[スライド6]\nマイルストーン M03 レビュー完了\n[ページ4]\nアクションアイテム A12 実施\n"
    occ = idm.scan_doc(_ref("プロジェクト/案件A/05.会議/議事.pptx", ext="pptx"), text)
    by_id = {o.id_norm: o for o in occ}
    assert by_id["M03"].slide == 6 and by_id["M03"].id_kind == "milestone"
    assert by_id["A12"].page == 4 and by_id["A12"].id_kind == "action_item"


def test_scan_doc_preserves_cross_document_same_id():
    a = idm.scan_doc(_ref("プロジェクト/X/plan.xlsx", project="X"), "M01 | 要件定義\n")
    b = idm.scan_doc(_ref("プロジェクト/Y/plan.xlsx", project="Y"), "M01 | 設計\n")
    # 同名 ID を doc_id で区別して両方保持（文書横断の取りこぼしゼロ）
    assert a[0].id_norm == b[0].id_norm == "M01"
    assert a[0].doc_id != b[0].doc_id and a[0].project == "X" and b[0].project == "Y"


# --------------------------------------------------------------------------- スキーマ / 読み出し / 決定論
@pytest.fixture()
def master(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    out = tmp_path / "id_master.jsonl"
    monkeypatch.setattr(idm, "default_out_path", lambda: out)
    text = ("[シート: WBS]  範囲 A1:C3\n"
            "WBS | 工程 | 担当\n"
            "WBS-01 | 基本設計 | 田中\n")
    occ = idm.scan_doc(_ref("プロジェクト/案件A/02.計画/wbs.xlsx"), text)
    idm.reset_cache()
    idm.write_master(occ, out)
    return out


def test_write_master_is_reproducible_and_schema_headed(master: Path):
    first = master.read_bytes()
    header = json.loads(master.read_text(encoding="utf-8").splitlines()[0])
    assert header == {"schema": idm.SCHEMA, "version": idm.SCHEMA_VERSION}
    rows = idm.load(master)
    assert rows and rows[0]["id_norm"] == "WBS01" and rows[0]["id_kind"] == "wbs"
    # 同じ occurrences を書き直すと byte-identical（決定論）
    from src.rag.index.id_master import IdOccurrence
    recs = [IdOccurrence(**{**r, "columns": tuple(r["columns"])}) for r in rows]
    idm.write_master(recs, master)
    assert master.read_bytes() == first


def test_lookup_is_separator_insensitive_and_zero_regression(master: Path):
    assert [r["id_norm"] for r in idm.lookup("WBS-01", path=master)] == ["WBS01"]
    assert [r["id_norm"] for r in idm.lookup("wbs01", path=master)] == ["WBS01"]  # 区切り/大小無視
    # 種別/案件フィルタ
    assert idm.lookup("WBS01", id_kind="task", path=master) == []
    # 未知 ID / 無アーティファクト → 空（劣化なし）
    assert idm.lookup("T99", path=master) == []
    assert idm.lookup("WBS01", path=Path("/nonexistent.jsonl")) == []


# --------------------------------------------------------------------------- カバレッジ表（診断・honest）
def test_fact_hard_core_coverage_is_honest():
    occ = idm.scan_doc(_ref("プロジェクト/青潮モビリティサービス/03.データ/train.xlsx",
                            project="青潮モビリティサービス"),
                       "[シート: Sheet1]\nタスクID | 内容\nT01 | 前処理\n")
    cov = idm.fact_hard_core_coverage(occ)
    # idx39 は青潮文書に doc_hint 一致 → ID を抽出した件数を報告
    assert cov["idx39"]["ids_in_matched_docs"] >= 1
    assert cov["idx39"]["matched_docs"] and "青潮" in cov["idx39"]["matched_docs"][0]
    # 3問はいずれも ID→内容 逆引き型ではない事実を明示（欠測を偽装しない）
    assert all(cov[i]["id_lookup_shaped"] is False and cov[i]["note"] for i in ("idx39", "idx48", "idx98"))


# --------------------------------------------------------------------------- opt-in default OFF
def test_enabled_defaults_off(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.delenv("RAG_ID_MASTER", raising=False)
    assert idm.enabled() is False
    monkeypatch.setenv("RAG_ID_MASTER", "1")
    assert idm.enabled() is True


def test_build_skips_non_document_sources(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    refs = [_ref("docs/tasks.xlsx"), _ref("analysis/uv.lock", ext="lock")]

    class _Doc:
        text = "タスクID | 内容\nT01 | 前処理"

    seen: list[str] = []

    def fake_extract(ref: FileRef, *, caption_images: bool):
        assert caption_images is False
        seen.append(ref.rel)
        return _Doc()

    monkeypatch.setattr("src.rag.extract.extract", fake_extract)
    summary = idm.build(refs, out=tmp_path / "master.jsonl", write_report=False)
    assert seen == ["docs/tasks.xlsx"]
    assert summary["occurrences"] == 1
    assert summary["report"]["by_source_ext"] == {"xlsx": 1}


# --------------------------------------------------------------------------- 実コーパスビルド（重い・隔離）
@pytest.mark.skipif(not settings.CORPUS_DIR.exists(), reason="corpus not present")
def test_real_corpus_build_is_deterministic(tmp_path: Path):
    try:
        import docx  # noqa: F401 — office extraction deps required for the real build
    except Exception:
        pytest.skip("office extraction deps not installed")
    out1, out2 = tmp_path / "idm1.jsonl", tmp_path / "idm2.jsonl"
    s1 = idm.build(out=out1, write_report=False)
    s2 = idm.build(out=out2, write_report=False)
    assert out1.read_bytes() == out2.read_bytes()          # 2回ビルドで byte-identical
    assert s1["occurrences"] == s2["occurrences"] >= 0
    assert s1["report"]["certificate"]["question_independent"] is True
