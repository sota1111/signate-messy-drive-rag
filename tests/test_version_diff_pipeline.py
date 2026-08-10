"""SOT-2605 (Wave A1) — offline tests for the deterministic ``version_diff`` pipeline.

All network-free and corpus-free: :mod:`src.rag.diffpair` is monkeypatched so no Office file is read.
The invariants under test are: the pipeline self-registers for ``version_diff``; it grounds a value
**only** for a single substantive ``modify`` (idx74-shaped) and otherwise returns ``None`` so the router
falls back to the LLM loop (idx1/idx22/idx95-shaped — 複数変更 / 非対応形式 / 全シート再整列); it recovers
the row/role label from the preceding shared paragraph; and it wires through ``det_pipeline.resolve`` +
``formatting.format_contract`` end-to-end.
"""
from __future__ import annotations

from types import SimpleNamespace

import pytest

from src.rag import diffpair
from src.rag.agent import det_pipeline as dp
from src.rag.agent import formatting
from src.rag.agent.pipelines import version_diff as vd


# --------------------------------------------------------------------------- registry cleanup fixture
@pytest.fixture(autouse=True)
def _restore_registry():
    saved = dict(dp._REGISTRY)
    try:
        yield
    finally:
        dp._REGISTRY.clear()
        dp._REGISTRY.update(saved)


# --------------------------------------------------------------------------- fake structural-diff helpers
def _pair(basis="explicit"):
    old = SimpleNamespace(rel="proj/提案書_v1.pptx", ext="pptx", stem="提案書_v1")
    new = SimpleNamespace(rel="proj/提案書_v2.pptx", ext="pptx", stem="提案書_v2")
    return SimpleNamespace(old=old, new=new, base="提案書", basis=basis)


def _ranked(change, intent=diffpair.SUBSTANTIVE, score=0.92):
    return diffpair.RankedChange(change=change, intent=intent, score=score, location=change.label)


def _wire(monkeypatch, *, is_diff=True, pair=None, ranked=None, struct=None):
    monkeypatch.setattr(diffpair, "is_diff_question", lambda q: is_diff)
    monkeypatch.setattr(diffpair, "_resolve_pair_for_render", lambda q: pair)
    monkeypatch.setattr(diffpair, "rank_changes", lambda p: ranked)
    if struct is not None:
        monkeypatch.setattr(diffpair, "_struct", struct)


# --------------------------------------------------------------------------- registration
def test_pipeline_registers_for_version_diff():
    vd.register(replace=True)
    assert "version_diff" in dp.registered_contracts()
    assert dp._REGISTRY["version_diff"] is vd.pipeline


# --------------------------------------------------------------------------- grounding (idx74-shaped)
def test_single_substantive_modify_grounds_value_with_recovered_label(monkeypatch):
    change = diffpair.Change(label="", before="藤田 彩", after="井上 里奈", kind="modify")
    so = diffpair._Struct(flow=["ビジネスアナリスト", "藤田 彩", "業務要件整理"],
                          flow_labels=["s9", "s9", "s9"])
    sn = diffpair._Struct(flow=["ビジネスアナリスト", "井上 里奈", "業務要件整理"],
                          flow_labels=["s9", "s9", "s9"])
    _wire(monkeypatch, pair=_pair(), ranked=[_ranked(change)],
          struct=lambda ref: so if ref.stem.endswith("v1") else sn)

    out = vd.pipeline("q")
    assert out is not None
    # value is deterministic gold-shaped prose: recovered role label + verbatim before→after (SOT-2619).
    assert out["value"] == "ビジネスアナリストが藤田 彩から井上 里奈に変更"
    assert out["evidence"]["before"] == "藤田 彩"
    assert out["evidence"]["after"] == "井上 里奈"
    assert out["evidence"]["structural_location"] == "ビジネスアナリスト"
    assert out["method"]["naturalize"] is False  # deterministic prose ⇒ Stage3 renders verbatim, no LLM


def test_keyed_cell_label_is_used_verbatim(monkeypatch):
    # A change that already carries a cell label needs no struct read for context.
    change = diffpair.Change(label="担当者", before="A", after="B", kind="modify")
    _wire(monkeypatch, pair=_pair(), ranked=[_ranked(change)],
          struct=lambda ref: (_ for _ in ()).throw(AssertionError("must not read structs")))
    out = vd.pipeline("q")
    assert out["value"] == "担当者がAからBに変更"


# --------------------------------------------------------------------------- fallbacks (idx1/22/95-shaped)
def test_not_a_diff_question_falls_back(monkeypatch):
    _wire(monkeypatch, is_diff=False)
    assert vd.pipeline("q") is None


def test_no_resolvable_pair_falls_back(monkeypatch):
    _wire(monkeypatch, pair=None)
    assert vd.pipeline("q") is None


def test_multiple_substantive_changes_fall_back(monkeypatch):
    # idx1/idx95: several substantive edits ⇒ locus is ambiguous ⇒ LLM fallback.
    a = diffpair.Change(label="AUC", before="0.9", after="", kind="remove")
    b = diffpair.Change(label="F1", before="0.7", after="", kind="remove")
    ranked = [_ranked(a, score=0.72), _ranked(b, score=0.72)]
    _wire(monkeypatch, pair=_pair(), ranked=ranked)
    assert vd.pipeline("q") is None


def test_lone_non_modify_change_falls_back(monkeypatch):
    # A single substantive *add/delete* (block reorganization) is not a locatable in-place edit.
    change = diffpair.Change(label="s6", before="", after="新セクション", kind="add")
    _wire(monkeypatch, pair=_pair(), ranked=[_ranked(change)])
    assert vd.pipeline("q") is None


def test_non_substantive_singleton_falls_back(monkeypatch):
    change = diffpair.Change(label="footer", before="v1", after="v2", kind="modify")
    _wire(monkeypatch, pair=_pair(), ranked=[_ranked(change, intent=diffpair.BOILERPLATE, score=0.10)])
    assert vd.pipeline("q") is None


def test_empty_ranked_falls_back(monkeypatch):
    _wire(monkeypatch, pair=_pair(), ranked=None)
    assert vd.pipeline("q") is None


def test_read_error_falls_back(monkeypatch):
    def _boom(q):
        raise RuntimeError("corpus unreadable")
    monkeypatch.setattr(diffpair, "is_diff_question", _boom)
    assert vd.pipeline("q") is None  # never raises into the answer path


def test_label_recovery_skips_when_no_shared_preceding_header(monkeypatch):
    change = diffpair.Change(label="", before="旧", after="新", kind="modify")
    # preceding paragraphs differ between versions ⇒ no trustworthy shared header ⇒ bare render.
    so = diffpair._Struct(flow=["古い見出し", "旧"], flow_labels=["s1", "s1"])
    sn = diffpair._Struct(flow=["新しい見出し", "新"], flow_labels=["s1", "s1"])
    _wire(monkeypatch, pair=_pair(), ranked=[_ranked(change)],
          struct=lambda ref: so if ref.stem.endswith("v1") else sn)
    out = vd.pipeline("q")
    assert out["value"] == "旧から新に変更"  # no shared header ⇒ label-free prose
    assert out["evidence"]["structural_location"] == ""


# --------------------------------------------------------------------------- router + formatting e2e
def test_resolve_wraps_pipeline_when_forced(monkeypatch):
    change = diffpair.Change(label="担当", before="X", after="Y", kind="modify")
    _wire(monkeypatch, pair=_pair(), ranked=[_ranked(change)])
    out = dp.resolve("q", "version_diff", force=True)
    assert out is not None and out["value"] == "担当がXからYに変更"


def test_resolve_off_returns_none_even_with_registered_pipeline(monkeypatch):
    monkeypatch.delenv("RAG_DET_PIPELINE_ROUTER", raising=False)
    change = diffpair.Change(label="担当", before="X", after="Y", kind="modify")
    _wire(monkeypatch, pair=_pair(), ranked=[_ranked(change)])
    # flag OFF ⇒ champion serve path byte-identical: the pipeline is never consulted.
    assert dp.resolve("q", "version_diff") is None


def test_end_to_end_through_formatting_is_deterministic_no_llm(monkeypatch):
    # SOT-2619: version_diff now hands Stage3 gold-shaped prose with naturalize=False, so the formatting
    # layer renders it verbatim and NEVER calls the naturalizer (idx74 の preamble/箇条書き 混入を排除).
    change = diffpair.Change(label="ビジネスアナリスト", before="藤田 彩", after="井上 里奈", kind="modify")
    _wire(monkeypatch, pair=_pair(), ranked=[_ranked(change)])
    contract = dp.resolve("q", "version_diff", force=True)

    def _must_not_be_called(value, q):  # a naturalizer that fails loudly if the layer calls the LLM
        raise AssertionError("version_diff must not invoke the LLM naturalizer any more")

    formatted = formatting.format_contract(
        contract, "q", contract_type="version_diff",
        naturalizer=_must_not_be_called, force=True)
    assert formatted["value"] == "ビジネスアナリストが藤田 彩から井上 里奈に変更"
    assert formatted["method"]["formatting"]["template_only"] is True
    assert "llm_naturalized" not in formatted["method"]["formatting"]["rules"]
