"""SOT-2706 — version_diff record-summary direct-commit lane + office table-frame summary annotation.

Two layers (mirrors scoring/test_vdiff_struct.py):
  * hermetic unit tests of the pure helpers (no corpus/LLM) — deleted-table semantic-frame extraction,
    flag-gated office summary annotation, and the serve-lane's verbatim commit + precision guards;
  * corpus-backed end-to-end assertions (skip cleanly when the SIGNATE corpus is absent) pinning the
    idx1 かえで frame and the flag-OFF byte-identical invariant.

    .venv/bin/python -m pytest scoring/test_vdiff_direct_commit.py -q
"""
from __future__ import annotations

import pytest

from src.rag import corpus, diffpair
from src.rag.agent import vdiff_direct_lane
from src.rag.index import diff_store

_CORPUS_PRESENT = bool(corpus.walk())
_needs_corpus = pytest.mark.skipif(not _CORPUS_PRESENT, reason="corpus not present")


# --------------------------------------------------------------------------- flag reader
def test_direct_commit_enabled_default_off(monkeypatch):
    monkeypatch.delenv("RAG_VDIFF_DIRECT_COMMIT", raising=False)
    assert vdiff_direct_lane.enabled() is False
    assert diff_store._direct_commit_build_enabled() is False
    monkeypatch.setenv("RAG_VDIFF_DIRECT_COMMIT", "1")
    assert vdiff_direct_lane.enabled() is True
    assert diff_store._direct_commit_build_enabled() is True


# --------------------------------------------------------------------------- semantic-frame extraction
def _cell_struct(pairs, flow=()):
    st = diffpair._Struct()
    for key, label, val in pairs:
        diffpair._add_cell(st, key, label, val)
    for txt, lab in flow:
        st.flow.append(txt)
        st.flow_labels.append(lab)
    return st


def _kaede_like_pair(monkeypatch):
    """A stub pptx pair whose slide-7 metrics comparison table is wholly deleted (no corpus)."""
    old = _cell_struct(
        [("s7:t1:指標:1", "指標", "中間 (T04 linear)"),
         ("s7:t1:指標:2", "指標", "最終 (hist_gradient_boosting)"),
         ("s7:t1:指標:3", "指標", "改善幅"),
         ("s7:t1:auc-roc:1", "AUC-ROC", "0.825"), ("s7:t1:auc-roc:2", "AUC-ROC", "0.905"),
         ("s7:t1:f1-macro:1", "F1-macro", "0.733"), ("s7:t1:f1-macro:2", "F1-macro", "0.829"),
         ("s7:t1:accuracy:1", "Accuracy", "0.736"), ("s7:t1:accuracy:2", "Accuracy", "0.833")],
        flow=[("6. 最終モデル性能指標と中間段階との比較", "スライド7"),
              ("中間段階 vs 最終モデル性能比較", "スライド7")])
    new = diffpair._Struct()
    new.flow.append("非線形モデル採用により全指標で+0.08〜+0.10ポイント改善")
    new.flow_labels.append("スライド7")

    class _Ref:
        ext = "pptx"
        path = "x"

    pair = diffpair.VersionPair(_Ref(), _Ref(), "最終報告", "registry-family")
    monkeypatch.setattr(diffpair, "_struct", lambda ref: old if ref is pair.old else new)
    return pair


def test_collapsed_table_frames_extracts_title_columns_metrics(monkeypatch):
    pair = _kaede_like_pair(monkeypatch)
    frames = diffpair.collapsed_table_frames(pair)
    assert set(frames) == {"スライド7"}
    fr = frames["スライド7"]
    # slide heading, leading section number stripped
    assert fr["title"] == "最終モデル性能指標と中間段階との比較"
    # column headers = header-row values, model annotation parenthetical stripped (中間/最終/改善幅)
    assert fr["columns"] == ["中間", "最終", "改善幅"]
    assert fr["metrics"] == ["AUC-ROC", "F1-macro", "Accuracy"]
    assert fr["header_label"] == "指標"


# --------------------------------------------------------------------------- office summary annotation
def test_office_summary_annotation_flag_gated(monkeypatch):
    pair = _kaede_like_pair(monkeypatch)
    monkeypatch.setenv("RAG_VDIFF_STRUCT", "1")
    record = {
        "ext": "pptx",
        "changes": [{"kind": "modify", "intent": "SUBSTANTIVE",
                     "old": "指標の比較表（AUC-ROC・F1-macro・Accuracy）",
                     "new": "改善幅のみを示す1行要約に置換", "structural_location": "スライド7",
                     "attributes": ["modification", "substantive"]}],
    }

    monkeypatch.delenv("RAG_VDIFF_DIRECT_COMMIT", raising=False)
    out = diff_store._annotate_office_table_summary(dict(record, changes=[dict(record["changes"][0])]), pair)
    assert out["changes"][0].get("summary") is None  # OFF ⇒ untouched (byte-identical)

    monkeypatch.setenv("RAG_VDIFF_DIRECT_COMMIT", "1")
    out = diff_store._annotate_office_table_summary(dict(record, changes=[dict(record["changes"][0])]), pair)
    summary = out["changes"][0]["summary"]
    # the semantic frame — what-vs-what + row/column headings — derived from the OLD slide, no gold value
    assert "最終モデル性能指標と中間段階との比較" in summary
    assert "中間" in summary and "最終" in summary
    assert "AUC-ROC" in summary and "F1-macro" in summary and "Accuracy" in summary
    assert "改善幅のみを示す1行要約に置換" in summary
    assert "table_frame_summary" in out["changes"][0]["attributes"]


def test_office_summary_skips_non_pptx_and_non_collapse(monkeypatch):
    pair = _kaede_like_pair(monkeypatch)
    monkeypatch.setenv("RAG_VDIFF_STRUCT", "1")
    monkeypatch.setenv("RAG_VDIFF_DIRECT_COMMIT", "1")
    # xlsx record is skipped (ext guard)
    xrec = {"ext": "xlsx", "changes": [{"kind": "modify", "intent": "SUBSTANTIVE",
                                        "old": "指標の比較表（x）", "structural_location": "スライド7"}]}
    assert diff_store._annotate_office_table_summary(xrec, pair)["changes"][0].get("summary") is None
    # a normal per-cell modify (no 比較表 in old) is left alone
    prec = {"ext": "pptx", "changes": [{"kind": "modify", "intent": "SUBSTANTIVE",
                                        "old": "担当者：田中", "new": "担当者：鈴木",
                                        "structural_location": "スライド7"}]}
    assert diff_store._annotate_office_table_summary(prec, pair)["changes"][0].get("summary") is None


# --------------------------------------------------------------------------- serve-lane direct commit
_NOTEBOOK_SUMMARY = "記述統計（基本統計量）の表に、目的変数 class の列の統計量が追加された（Attr1〜64は同一）"


def _store_rows():
    return [
        {"project": "白峰信用リスク評価株式会社", "ext": "ipynb", "basis": "notebook",
         "old_name": "01_eda_old.ipynb", "new_name": "01_eda.ipynb",
         "old_rel": "…/01_eda_old.ipynb", "new_rel": "…/01_eda.ipynb",
         "changes": [{"rank": 0, "kind": "modify", "intent": "SUBSTANTIVE",
                      "structural_location": "cell 8 (embedded image)", "summary": _NOTEBOOK_SUMMARY}]},
        {"project": "医療法人社団 恒一会 かえで総合病院", "ext": "pptx", "basis": "registry-family",
         "old_name": "…_最終報告_old.pptx", "new_name": "…_最終報告.pptx",
         "old_rel": "…/_最終報告_old.pptx", "new_rel": "…/_最終報告.pptx",
         "changes": [{"rank": 0, "kind": "modify", "intent": "SUBSTANTIVE", "structural_location": "スライド7",
                      "summary": "スライド7にあった「…比較」の比較表（…）が削除され、要約に置換"},
                     {"rank": 1, "kind": "add", "intent": "LAYOUT_METADATA", "summary": None}]},
    ]


@pytest.fixture
def _stub_store(monkeypatch):
    monkeypatch.setattr(diff_store, "load", lambda *a, **k: _store_rows())
    yield


def test_direct_commit_off_returns_none(monkeypatch, _stub_store):
    monkeypatch.delenv("RAG_VDIFF_DIRECT_COMMIT", raising=False)
    q = "白峰信用リスク評価の01_eda_old.ipynbから01_eda.ipynbへの変更内容は何ですか。"
    assert vdiff_direct_lane.resolve(q) is None


def test_direct_commit_notebook_summary_verbatim(monkeypatch, _stub_store):
    """idx22-型: notebook 逐語 — 括弧部（Attr1〜64は同一）を脱落させずそのまま commit する。"""
    monkeypatch.setenv("RAG_VDIFF_DIRECT_COMMIT", "1")
    q = "白峰信用リスク評価の01_eda_old.ipynbから01_eda.ipynbへの変更内容は何ですか。"
    res = vdiff_direct_lane.resolve(q)
    assert res is not None
    assert res["value"] == _NOTEBOOK_SUMMARY               # verbatim, parenthetical preserved
    assert res["method"]["naturalize"] is False
    assert res["method"]["selection"] == "record_summary_direct_commit"


def test_direct_commit_pptx_frame_by_project_stem(monkeypatch, _stub_store):
    """idx1-型: 法人格接頭辞を落とした project stem で一意束縛し、意味枠 summary を commit。"""
    monkeypatch.setenv("RAG_VDIFF_DIRECT_COMMIT", "1")
    q = "恒一会 かえで総合病院の最終報告書old版と最新版を比較したとき、実質的な変更を挙げてください。"
    res = vdiff_direct_lane.resolve(q)
    assert res is not None and "比較表" in res["value"]


def test_direct_commit_requires_change_intent(monkeypatch, _stub_store):
    """変更を問わない質問（版ペアに束縛できても）は commit しない（precision-first）。"""
    monkeypatch.setenv("RAG_VDIFF_DIRECT_COMMIT", "1")
    q = "白峰信用リスク評価の01_eda_old.ipynbと01_eda.ipynbの目的変数の平均値はいくつですか。"
    assert vdiff_direct_lane.resolve(q) is None


def test_direct_commit_ambiguous_pair_defers(monkeypatch, _stub_store):
    """project も明示ファイル名も一意に束縛できない質問は None（従来経路へ）。"""
    monkeypatch.setenv("RAG_VDIFF_DIRECT_COMMIT", "1")
    assert vdiff_direct_lane.resolve("最新版で変わった点を挙げてください。") is None


def test_direct_commit_non_substantive_rank0_defers(monkeypatch):
    monkeypatch.setenv("RAG_VDIFF_DIRECT_COMMIT", "1")
    rows = [{"project": "某社株式会社", "ext": "pptx", "old_name": "a_old.pptx", "new_name": "a.pptx",
             "changes": [{"rank": 0, "kind": "add", "intent": "LAYOUT_METADATA", "summary": "x"}]}]
    monkeypatch.setattr(diff_store, "load", lambda *a, **k: rows)
    assert vdiff_direct_lane.resolve("某社のa_old.pptxからa.pptxへの変更点は？") is None


def test_direct_commit_multiple_summaries_defers(monkeypatch):
    """rank0 以外にも summary 付き SUBSTANTIVE がある（多重変更）と曖昧なので commit しない。"""
    monkeypatch.setenv("RAG_VDIFF_DIRECT_COMMIT", "1")
    rows = [{"project": "某社株式会社", "ext": "pptx", "old_name": "a_old.pptx", "new_name": "a.pptx",
             "changes": [{"rank": 0, "kind": "modify", "intent": "SUBSTANTIVE", "summary": "s0"},
                         {"rank": 1, "kind": "modify", "intent": "SUBSTANTIVE", "summary": "s1"}]}]
    monkeypatch.setattr(diff_store, "load", lambda *a, **k: rows)
    assert vdiff_direct_lane.resolve("某社のa_old.pptxからa.pptxへの変更点は？") is None


# --------------------------------------------------------------------------- corpus-backed (idx1 かえで)
@_needs_corpus
def test_idx1_kaede_pair_gets_frame_summary(monkeypatch):
    monkeypatch.setenv("RAG_VDIFF_STRUCT", "1")
    monkeypatch.setenv("RAG_VDIFF_CLASSIFY", "1")
    pairs = [p for p in diffpair._registry_family_pairs() if "恒一会" in (p.new.rel or "")]
    assert pairs, "かえで pair not enumerated"
    frames = diffpair.collapsed_table_frames(pairs[0])
    fr = frames.get("スライド7")
    assert fr is not None
    assert "中間段階" in fr["title"] and "最終モデル" in fr["title"]
    assert "中間" in fr["columns"] and "最終" in fr["columns"]


@_needs_corpus
def test_flag_off_office_build_has_no_frame_summary(monkeypatch, tmp_path):
    """RAG_VDIFF_DIRECT_COMMIT OFF ⇒ office records carry no table_frame_summary (store byte-identical)."""
    monkeypatch.setenv("RAG_VDIFF_STRUCT", "1")
    monkeypatch.setenv("RAG_VDIFF_CLASSIFY", "1")
    monkeypatch.delenv("RAG_VDIFF_DIRECT_COMMIT", raising=False)
    pairs = diff_store.enumerate_pairs()
    for pair, sources in pairs:
        rec = diff_store._annotate_office_table_summary(diff_store.pair_record(pair, sources), pair)
        for ch in rec.get("changes", []):
            assert "summary" not in ch or "table_frame_summary" not in (ch.get("attributes") or [])
