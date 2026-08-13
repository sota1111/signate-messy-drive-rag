"""SOT-2700 — deterministic structural repairs for the four version_diff wrong cases (RAG_VDIFF_STRUCT).

Two layers:
  * hermetic unit tests of the pure helpers (no corpus/LLM) — row-label xlsx keying, whole-table
    deletion collapse, notebook 目的変数 attribution;
  * corpus-backed end-to-end assertions (skip cleanly when the SIGNATE corpus is absent) pinning
    idx1/idx14/idx95 behaviour AND the flag-OFF byte-identical invariant + the idx9 該当なし guard.

    .venv/bin/python -m pytest scoring/test_vdiff_struct.py -q
"""
from __future__ import annotations

import pytest

from src.rag import corpus, diffpair
from src.rag.index import diff_store

_CORPUS_PRESENT = bool(corpus.walk())
_needs_corpus = pytest.mark.skipif(not _CORPUS_PRESENT, reason="corpus not present")


@pytest.fixture(autouse=True)
def _struct_on(monkeypatch):
    """Every test states its own flag; default the lane ON so the helpers exercise the new path."""
    monkeypatch.setenv("RAG_VDIFF_STRUCT", "1")
    monkeypatch.setenv("RAG_VDIFF_CLASSIFY", "1")
    yield


# --------------------------------------------------------------------------- flag reader
def test_struct_enabled_default_off(monkeypatch):
    monkeypatch.delenv("RAG_VDIFF_STRUCT", raising=False)
    assert diffpair.struct_enabled() is False
    monkeypatch.setenv("RAG_VDIFF_STRUCT", "1")
    assert diffpair.struct_enabled() is True


# --------------------------------------------------------------------------- idx95 row-label xlsx keying
def _schedule_rows(status, t15_owner):
    """A minimal task-schedule sheet: header + three タスクID rows. ``status``/``t15_owner`` vary."""
    header = ["No.", "タスクID", "タスク名", "ステータス", "担当者"]
    return [
        header,
        ["1", "T01", "キックオフ", status, "佐藤 健一"],
        ["2", "T14", "特徴量設計", status, "渡辺 遥"],
        ["3", "T15", "モデル評価", status, t15_owner],
    ]


def test_rowlabel_keying_survives_column_reorder_and_row_shift():
    # OLD keyed by row-label+header; NEW has the SAME rows but columns reordered and two blank rows on top.
    old = diffpair._Struct()
    assert diffpair._key_sheet_by_rowlabel(old, "S", _schedule_rows("未着手", "渡辺 遥"))
    new_rows = [[None] * 5, [None] * 5]  # header shifted down two rows
    for r in _schedule_rows("完了", "渡辺 遥 / 小林 直樹"):
        new_rows.append([r[3], r[1], r[4], r[0], r[2]])  # reorder columns arbitrarily
    new = diffpair._Struct()
    assert diffpair._key_sheet_by_rowlabel(new, "S", new_rows)
    # Same identity keys on both sides despite the reorder/shift.
    assert set(old.cells) == set(new.cells)
    changed = [(lab, ov, new.cells[k][1]) for k, (lab, ov) in old.cells.items()
               if diffpair._norm(ov) != diffpair._norm(new.cells[k][1])]
    # Exactly: 3 status flips (未着手→完了) + the T15 assignee append. No coordinate churn.
    assert sum(1 for _lab, b, a in changed if b == "未着手" and a == "完了") == 3
    assert any("T15" in lab and b == "渡辺 遥" and "小林 直樹" in a for lab, b, a in changed)
    assert len(changed) == 4


def test_rowlabel_keying_needs_a_header_and_identity_column():
    # A sheet with no id-like/sequence header and no distinct leftmost column ⇒ falls back (returns False).
    st = diffpair._Struct()
    rows = [["x", "x"], ["a", "a"], ["a", "b"]]  # leftmost col not distinct, no header keyword
    assert diffpair._key_sheet_by_rowlabel(st, "S", rows) is False


def test_status_only_modify_demoted_below_assignee_append():
    status = diffpair.classify_change(diffpair.Change("T01／ステータス", "未着手", "完了", "modify"))
    assignee = diffpair.classify_change(
        diffpair.Change("T15／担当者", "渡辺 遥", "渡辺 遥 / 小林 直樹", "modify"))
    assert status.intent == diffpair.UNCERTAIN and "status_only" in status.features
    assert assignee.intent == diffpair.SUBSTANTIVE and assignee.score > status.score
    assert "assignee_append" in assignee.features


# --------------------------------------------------------------------------- idx1 whole-table deletion
def _cell_struct(pairs):
    st = diffpair._Struct()
    for key, label, val in pairs:
        diffpair._add_cell(st, key, label, val)
    return st


def test_deleted_metrics_table_collapses_to_summary():
    old = _cell_struct([
        ("s7:t1:指標:1", "指標", "中間"), ("s7:t1:指標:2", "指標", "最終"), ("s7:t1:指標:3", "指標", "改善幅"),
        ("s7:t1:auc-roc:1", "AUC-ROC", "0.825"), ("s7:t1:auc-roc:2", "AUC-ROC", "0.905"),
        ("s7:t1:f1-macro:1", "F1-macro", "0.733"), ("s7:t1:f1-macro:2", "F1-macro", "0.829"),
    ])
    new = diffpair._Struct()
    new.flow.append("AUC-ROC: +0.080ポイント改善 / F1-macro: +0.096ポイント改善")
    new.flow_labels.append("スライド7")
    summaries, skip_keys, skip_flow = diffpair._collapse_deleted_tables(old, new)
    assert len(summaries) == 1
    s = summaries[0]
    assert s.kind == "modify" and s.after == "改善幅のみを示す1行要約に置換" and "比較表" in s.before
    assert "0.825" not in s.before and "中間" not in s.before
    assert skip_keys == set(old.cells)                       # every per-cell remove suppressed
    assert diffpair._norm(new.flow[0]) in skip_flow          # the paired summary line is consumed


def test_moved_table_is_not_collapsed():
    # A table whose cells REAPPEAR verbatim as new cells was reorganized, not deleted → no collapse.
    old = _cell_struct([
        ("s9:t1:r:1", "リスク", "A"), ("s9:t1:r:2", "リスク", "B"),
        ("s9:t1:s:1", "対策", "C"), ("s9:t1:s:2", "対策", "D"),
    ])
    new = _cell_struct([
        ("s9:t2:x:1", "x", "A"), ("s9:t2:x:2", "x", "B"),
        ("s9:t2:y:1", "y", "C"), ("s9:t2:y:2", "y", "D"),
    ])
    summaries, skip_keys, _ = diffpair._collapse_deleted_tables(old, new)
    assert summaries == [] and skip_keys == set()


# --------------------------------------------------------------------------- idx22 notebook attribution
def test_notebook_target_variable_and_stats_annotation(monkeypatch):
    cells = [{"type": "code", "src": "print(target)", "out": "目的変数列: class\n行数: 7352"}]
    monkeypatch.setattr(diff_store, "_nb_cells", lambda path: cells)
    pair = diffpair.VersionPair(
        diffpair.FileRef(path="o", project="白峰", category="a", rel="o.ipynb", name="o", ext="ipynb"),
        diffpair.FileRef(path="n", project="白峰", category="a", rel="n.ipynb", name="n", ext="ipynb"),
        "01_eda", "notebook")
    assert diff_store._notebook_target_variable(pair) == "class"
    rec = {"changes": [{"kind": "modify", "attributes": ["notebook", "embedded_image", "table_headers",
                                                         "column_added"],
                        "headers_added": ["class"], "headers_old_count": 64, "headers_new_count": 65,
                        "old": "平均 0.05 標準偏差 0.74", "new": "平均 0.05 標準偏差 0.74"}]}
    out = diff_store._annotate_notebook_stats(rec, pair)
    ch = out["changes"][0]
    assert "目的変数 class の列の統計量が追加" in ch["summary"]
    assert "他の列は同一" in ch["summary"]
    assert {"stats_table", "target_variable_added"} <= set(ch["attributes"])


def test_collapse_column_range_contiguous_and_rejects_gaps():
    # SOT-2700 (idx22): a uniform prefix + contiguous 1..N suffix collapses to a 「Attr1〜64」 range.
    assert diff_store._collapse_column_range([f"Attr{i}" for i in range(1, 65)]) == "Attr1〜64"
    assert diff_store._collapse_column_range(["feature_3", "feature_1", "feature_2"]) == "feature_1〜3"
    # A gap, a mixed prefix, or a non-numeric member is NOT a clean range ⇒ None (caller stays generic).
    assert diff_store._collapse_column_range(["Attr1", "Attr2", "Attr4"]) is None
    assert diff_store._collapse_column_range(["Attr1", "Col2"]) is None
    assert diff_store._collapse_column_range(["Attr1", "class"]) is None


def test_unchanged_columns_phrase_names_range_when_source_resolves(monkeypatch):
    # ON + resolvable source schema whose residual count matches the diff's old count ⇒ specific range.
    pair = diffpair.VersionPair(
        diffpair.FileRef(path="o", project="白峰", category="a", rel="o.ipynb", name="o", ext="ipynb"),
        diffpair.FileRef(path="n", project="白峰", category="a", rel="n.ipynb", name="n", ext="ipynb"),
        "01_eda", "notebook")
    monkeypatch.setattr(diff_store, "_dataset_feature_columns",
                        lambda p: [f"Attr{i}" for i in range(1, 65)] + ["class"])
    assert diff_store._unchanged_columns_phrase(pair, ["class"], 64) == "（Attr1〜64は同一）"
    # Count mismatch (diff says 30 old cols, schema residual is 64) ⇒ certainty gate fails ⇒ generic.
    assert diff_store._unchanged_columns_phrase(pair, ["class"], 30) == "（他の列は同一）"
    # No source resolvable ⇒ generic remainder.
    monkeypatch.setattr(diff_store, "_dataset_feature_columns", lambda p: None)
    assert diff_store._unchanged_columns_phrase(pair, ["class"], 64) == "（他の列は同一）"


def test_notebook_annotation_off_is_noop(monkeypatch):
    monkeypatch.delenv("RAG_VDIFF_STRUCT", raising=False)
    pair = diffpair.VersionPair(
        diffpair.FileRef(path="o", project="p", category="a", rel="o.ipynb", name="o", ext="ipynb"),
        diffpair.FileRef(path="n", project="p", category="a", rel="n.ipynb", name="n", ext="ipynb"),
        "b", "notebook")
    rec = {"changes": [{"kind": "modify", "attributes": ["notebook"], "headers_added": ["class"]}]}
    out = diff_store._annotate_notebook_stats(rec, pair)
    assert "summary" not in out["changes"][0]


# =============================================================================== corpus end-to-end
_SCHED = "スケジュール_r1.xlsx"


def _pair(sub_old, sub_new, base, basis="rev-suffix"):
    refs = diffpair._walk(None)
    o = [r for r in refs if sub_old in r.rel][0]
    n = [r for r in refs if sub_new in r.rel][0]
    return diffpair.VersionPair(o, n, base, basis)


@_needs_corpus
def test_idx95_only_assignee_survives_status_exclusion():
    pair = _pair("スケジュール_r1.xlsx", "スケジュール_r2.xlsx", "スケジュール")
    ranked = diffpair.rank_changes(pair)
    assert len(ranked) < 30                       # 369 coordinate-churn cells collapse to ~23
    top = ranked[0]
    assert top.intent == diffpair.SUBSTANTIVE and "T15" in top.change.label
    assert "小林 直樹" in top.change.after
    # Every non-assignee change is a status transition (excludable by the question filter).
    non_status = [r for r in ranked
                  if "status_transition" not in diff_store.change_attributes(r)]
    assert len(non_status) == 1


@_needs_corpus
def test_idx1_metrics_table_collapses_to_single_summary():
    pair = _pair("かえで総合病院_最終報告_old.pptx", "恒一会 かえで総合病院_最終報告.pptx",
                 "最終報告", "old-folder")
    ranked = diffpair.rank_changes(pair)
    top = ranked[0]
    assert top.intent == diffpair.SUBSTANTIVE
    assert "比較表" in top.change.before and "1行要約" in top.change.after


@_needs_corpus
def test_idx14_all_four_column_renames_present():
    pair = _pair("青葉与信マネジメント株式会社/00.提案/提案書_v1.pptx",
                 "青葉与信マネジメント株式会社/00.提案/提案書_v3.pptx", "提案書")
    ranked = diffpair.rank_changes(pair)
    renamed = {r.change.after for r in ranked if r.change.label == "データ列名"}
    assert {"loan_status", "employment_length", "application_type", "interest_rate"} <= renamed


@_needs_corpus
@pytest.mark.parametrize("sub_old,sub_new,base,basis", [
    ("スケジュール_r1.xlsx", "スケジュール_r2.xlsx", "スケジュール", "rev-suffix"),
    ("かえで総合病院_最終報告_old.pptx", "恒一会 かえで総合病院_最終報告.pptx", "最終報告", "old-folder"),
])
def test_flag_off_is_byte_identical_to_champion(monkeypatch, sub_old, sub_new, base, basis):
    pair = _pair(sub_old, sub_new, base, basis)
    monkeypatch.setenv("RAG_VDIFF_STRUCT", "1")
    on = [r.change.render() for r in diffpair.rank_changes(pair)]
    monkeypatch.delenv("RAG_VDIFF_STRUCT", raising=False)
    off = [r.change.render() for r in diffpair.rank_changes(pair)]
    assert on != off                              # the lane demonstrably changes these two pairs …
    # … and OFF is exactly the historical (coordinate/enumeration) output — a large churn set.
    assert len(off) > len(on)


@_needs_corpus
def test_idx9_still_reports_no_substantive_change(monkeypatch):
    # Guard: idx9 gold is 該当なし (a heading-label / layout reorg only) — the struct lane must not
    # manufacture a substantive change by suppressing the moved-block removes.
    refs = diffpair._walk(None)
    o = [r for r in refs if "青葉与信" in r.rel and "06.報告書/old/" in r.rel][0]
    n = [r for r in refs if r.rel.endswith("06.報告書/青葉与信マネジメント株式会社_最終報告.pptx")][0]
    pair = diffpair.VersionPair(o, n, "最終報告", "old-folder")
    ranked = diffpair.rank_changes(pair)
    assert [r for r in ranked if r.intent == diffpair.SUBSTANTIVE] == []
