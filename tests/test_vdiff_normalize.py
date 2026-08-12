"""SOT-2681 (cycle6 K5) — version_diff semantic normalization (list-append→追加).

Network-free / corpus-free. Covers the pure list-append helpers in :mod:`src.rag.diffpair` and the
flag-gated committed-modify rendering in the deterministic ``version_diff`` pipeline. Invariants:

* ``is_list_append`` distinguishes an *append* (old ⊊ new by membership, idx95) from a *replacement*
  (idx74: old fully swapped out) and from a no-op.
* With ``RAG_VDIFF_NORMALIZE`` OFF (default) the committed modify rendering is byte-identical to before;
  ON, a list-append modify renders 「…に<追加項目>を追加」 while a plain replacement is unchanged.
* The Sonnet-backend system suffix binds the normalization contract only when the flag is on.
"""
from __future__ import annotations

from types import SimpleNamespace

import pytest

from src.rag import diffpair
from src.rag.agent import det_pipeline as dp
from src.rag.agent.pipelines import version_diff as vd
from src.rag.llm_providers import claude_mcp


# --------------------------------------------------------------------------- pure helpers
@pytest.mark.parametrize("before,after,expected", [
    ("渡辺 遥", "渡辺 遥 / 小林 直樹", True),          # idx95: 担当者 append
    ("A、B", "A、B、C", True),                          # 、-joined append
    ("鈴木", "鈴木・田中・佐藤", True),                 # ・-joined, multi append
    ("藤田 彩", "井上 里奈", False),                    # idx74: pure replacement (old vanished)
    ("A / B", "A / C", False),                          # one member swapped ⇒ replacement
    ("渡辺 遥 / 小林 直樹", "渡辺 遥", False),          # removal, not append
    ("同じ", "同じ", False),                            # no change
    ("", "小林 直樹", False),                           # no old members ⇒ not an append
])
def test_is_list_append(before, after, expected):
    assert diffpair.is_list_append(before, after) is expected


def test_appended_items_are_verbatim_new_members():
    assert diffpair.appended_items("渡辺 遥", "渡辺 遥 / 小林 直樹") == ["小林 直樹"]
    assert diffpair.appended_items("A、B", "A、B、C、D") == ["C", "D"]
    # order preserved as they appear in ``after``; existing members dropped
    assert diffpair.appended_items("B", "A、B、C") == ["A", "C"]


def test_single_space_inside_name_is_not_a_separator():
    # 姓 名 with a single internal space stays one item (not split into 渡辺 / 遥).
    assert diffpair._list_items("渡辺 遥") == ["渡辺 遥"]


# --------------------------------------------------------------------------- pipeline wiring
@pytest.fixture(autouse=True)
def _restore_registry():
    saved = dict(dp._REGISTRY)
    try:
        yield
    finally:
        dp._REGISTRY.clear()
        dp._REGISTRY.update(saved)


def _pair():
    old = SimpleNamespace(rel="proj/スケジュール_r1.xlsx", ext="xlsx", stem="スケジュール_r1")
    new = SimpleNamespace(rel="proj/スケジュール_r2.xlsx", ext="xlsx", stem="スケジュール_r2")
    return SimpleNamespace(old=old, new=new, base="スケジュール", basis="rev-suffix")


def _wire(monkeypatch, change):
    ranked = [diffpair.RankedChange(change=change, intent=diffpair.SUBSTANTIVE, score=0.92,
                                    location=change.label)]
    monkeypatch.setattr(diffpair, "is_diff_question", lambda q: True)
    monkeypatch.setattr(diffpair, "_resolve_pair_for_render", lambda q: _pair())
    monkeypatch.setattr(diffpair, "rank_changes", lambda p: ranked)


def test_list_append_modify_renders_addition_when_flag_on(monkeypatch):
    monkeypatch.setenv("RAG_VDIFF_NORMALIZE", "1")
    change = diffpair.Change(label="担当者", before="渡辺 遥", after="渡辺 遥 / 小林 直樹", kind="modify")
    _wire(monkeypatch, change)
    out = vd.pipeline("担当者の変更点は?")
    assert out is not None
    assert out["value"] == "担当者に小林 直樹を追加"
    assert out["method"]["normalization"] == "list_append_to_add"
    assert out["evidence"]["normalized_as"] == "list_append"


def test_list_append_modify_is_byte_identical_when_flag_off(monkeypatch):
    monkeypatch.delenv("RAG_VDIFF_NORMALIZE", raising=False)
    change = diffpair.Change(label="担当者", before="渡辺 遥", after="渡辺 遥 / 小林 直樹", kind="modify")
    _wire(monkeypatch, change)
    out = vd.pipeline("担当者の変更点は?")
    # OFF ⇒ historical 変更 rendering, no normalization keys.
    assert out["value"] == "担当者が渡辺 遥から渡辺 遥 / 小林 直樹に変更"
    assert "normalization" not in out["method"]


def test_pure_replacement_still_renders_change_even_when_flag_on(monkeypatch):
    # idx74 sentinel shape: a genuine replacement is NOT a list-append ⇒ unchanged 変更 rendering.
    monkeypatch.setenv("RAG_VDIFF_NORMALIZE", "1")
    change = diffpair.Change(label="担当者", before="藤田 彩", after="井上 里奈", kind="modify")
    _wire(monkeypatch, change)
    out = vd.pipeline("担当者の変更点は?")
    assert out["value"] == "担当者が藤田 彩から井上 里奈に変更"
    assert "normalization" not in out["method"]


# --------------------------------------------------------------------------- prompt contract
def test_contract_absent_by_default(monkeypatch):
    monkeypatch.delenv("RAG_VDIFF_NORMALIZE", raising=False)
    assert "版差分の意味正規化契約" not in claude_mcp._harness_system_suffix()


def test_contract_appended_when_flag_on(monkeypatch):
    monkeypatch.setenv("RAG_VDIFF_NORMALIZE", "1")
    suffix = claude_mcp._harness_system_suffix()
    assert "版差分の意味正規化契約" in suffix
    assert "を追加" in suffix and "に置換" in suffix
