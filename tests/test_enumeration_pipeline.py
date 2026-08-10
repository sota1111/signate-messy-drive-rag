"""SOT-2608 (Wave A3) — offline tests for the deterministic ``enum`` / ``cross_aggregate`` pipeline.

Network-free and corpus-free: ``canonical_route.resolve_project`` / ``corpus.walk`` / the cross-corpus
``corpus_aggregate`` are monkeypatched, and the schedule sheet is a tiny real ``.xlsx`` written to
``tmp_path`` (so the actual pandas read/filter/closure logic runs, not a stub). The invariants under test:
the pipeline self-registers for **both** ``full_enumeration`` and ``cross_aggregate``; it grounds a value
only for its three tight shapes (idx19/26/86 型) confirmed by closure; it never commits an empty
enumeration as 「該当なし」 (SOT-2600); it falls back (``None``) on every ambiguous / unrecognised question
(idx96/38/67/87/70 型); OFF ⇒ byte-identical; and it wires through ``det_pipeline.resolve`` +
``formatting.format_contract`` end-to-end (both are template-first — no LLM naturalize call).
"""
from __future__ import annotations

import importlib

import pandas as pd
import pytest

from src.rag.agent import det_pipeline as dp
from src.rag.agent import formatting
from src.rag.agent.pipelines import enumeration as en
from src.rag.corpus import FileRef


# --------------------------------------------------------------------------- registry cleanup fixture
@pytest.fixture(autouse=True)
def _restore_registry():
    saved = dict(dp._REGISTRY)
    try:
        yield
    finally:
        dp._REGISTRY.clear()
        dp._REGISTRY.update(saved)


# --------------------------------------------------------------------------- schedule fixture
def _write_schedule(tmp_path):
    """A tiny real schedule .xlsx (header row + タスクID/開始日/終了日) mirroring the corpus layout."""
    rows = [
        ["No.", "フェーズ", "タスクID", "タスク名", "担当者", "開始日", "終了日"],
        [1, "P1", "T01", "kickoff", "A", "2025-08-06", "2025-08-06"],
        [2, "P1", "T02", "scope", "A", "2025-08-06", "2025-08-07"],
        [3, "P1", "T03", "受領", "B", "2025-08-06", "2025-08-08"],
        [4, "P1", "T04", "整理", "C", "2025-08-07", "2025-08-11"],   # end in window
        [5, "P2", "T05", "点検", "B", "2025-08-11", "2025-08-15"],   # start in window
        [6, "P3", "T06", "報告", "D", "2025-09-08", "2025-09-11"],   # start in window
        [7, "P4", "T07", "後続", "D", "2025-09-11", "2025-09-12"],   # both out of window
    ]
    df = pd.DataFrame(rows)
    path = tmp_path / "スケジュール_r2.xlsx"
    df.to_excel(path, header=False, index=False)
    return path


def _wire_schedule(monkeypatch, path, *, project="青嶺不動産アセットマネジメント"):
    cr = importlib.import_module("src.rag.tools.canonical_route")
    monkeypatch.setattr(cr, "resolve_project", lambda q, p, **k: project)
    ref = FileRef(path=str(path), project=project, category="plan",
                  rel=f"proj/02.計画/{path.name}", name=path.name, ext="xlsx")
    monkeypatch.setattr("src.rag.corpus.walk", lambda *a, **k: [ref])
    return ref


# --------------------------------------------------------------------------- registration
def test_pipeline_registers_for_both_contracts():
    en.register(replace=True)
    contracts = dp.registered_contracts()
    assert "full_enumeration" in contracts and "cross_aggregate" in contracts
    assert dp._REGISTRY["full_enumeration"] is en.full_enumeration_pipeline
    assert dp._REGISTRY["cross_aggregate"] is en.cross_aggregate_pipeline


# --------------------------------------------------------------------------- shape 1: schedule window enum
def test_schedule_window_enum_grounds_and_orders(monkeypatch, tmp_path):
    path = _write_schedule(tmp_path)
    _wire_schedule(monkeypatch, path)
    q = ("青嶺不動産アセットマネジメントのスケジュール_r2.xlsxにおいて、2025-08-11から2025-09-09の間に"
         "開始日または終了日が設定されているタスクIDをすべて挙げてください。")
    out = en.full_enumeration_pipeline(q)
    assert out is not None
    # T04 (end 08-11), T05 (start 08-11), T06 (start 09-08) — in ID-ascending order; T01-T03/T07 excluded.
    assert out["value"] == ["T04", "T05", "T06"]
    assert out["evidence"]["closure"] == "four_conditions_satisfied"
    assert out["method"]["shape"] == "schedule_window_task_enum"


def test_schedule_window_empty_scan_falls_back_not_none_answer(monkeypatch, tmp_path):
    # A window matching no task row must NOT be committed as 「該当なし」 (SOT-2600) — it falls back.
    path = _write_schedule(tmp_path)
    _wire_schedule(monkeypatch, path)
    q = ("青嶺不動産アセットマネジメントのスケジュール_r2.xlsxにおいて、2030-01-01から2030-02-01の間に"
         "開始日または終了日が設定されているタスクIDをすべて挙げてください。")
    assert en.full_enumeration_pipeline(q) is None


def test_schedule_enum_without_window_falls_back(monkeypatch, tmp_path):
    path = _write_schedule(tmp_path)
    _wire_schedule(monkeypatch, path)
    q = "青嶺不動産アセットマネジメントのスケジュール_r2.xlsxのタスクIDをすべて挙げてください。"
    assert en.full_enumeration_pipeline(q) is None  # no date window ⇒ not this shape


def test_schedule_enum_no_named_file_falls_back(monkeypatch, tmp_path):
    path = _write_schedule(tmp_path)
    _wire_schedule(monkeypatch, path)
    q = ("青嶺不動産アセットマネジメントで、2025-08-11から2025-09-09の間に開始日または終了日が"
         "設定されているタスクIDをすべて挙げてください。")
    assert en.full_enumeration_pipeline(q) is None  # no *.xlsx named ⇒ ambiguous ⇒ fallback


# --------------------------------------------------------------------------- shape 2: period overlap filter
def _wire_corpus_aggregate(monkeypatch, fn):
    mod = importlib.import_module("src.rag.tools.corpus_aggregate")
    monkeypatch.setattr(mod, "corpus_aggregate", fn)


def test_period_overlap_filter_grounds(monkeypatch):
    captured = {}

    def fake_agg(**kwargs):
        captured.update(kwargs)
        return {"value": ["TOTO", "AOMINE"], "evidence": {"matches": [], "projects": 9},
                "method": {}}

    _wire_corpus_aggregate(monkeypatch, fake_agg)
    q = ("2025-08-15 から 2025-09-07 の間に契約期間が重なっている案件の中で、契約期間が 40日 を"
         "超えている案件を、主略称ですべて挙げてください。")
    out = en.cross_aggregate_pipeline(q)
    assert out is not None
    assert out["value"] == ["TOTO", "AOMINE"]
    assert captured["op"] == "filter"
    assert captured["overlap_start"] == "2025-08-15" and captured["overlap_end"] == "2025-09-07"
    assert captured["min_days"] == 40  # 「40日を超えている」 → strict > 40


def test_period_filter_empty_result_falls_back(monkeypatch):
    _wire_corpus_aggregate(monkeypatch, lambda **k: {"value": [], "evidence": {}, "method": {}})
    q = ("2025-08-15 から 2025-09-07 の間に契約期間が重なっている案件を主略称ですべて挙げてください。")
    assert en.cross_aggregate_pipeline(q) is None  # empty filter never committed as 該当なし


def test_min_days_ijou_translates_to_strict_gt(monkeypatch):
    captured = {}

    def fake_agg(**kwargs):
        captured.update(kwargs)
        return {"value": ["X"], "evidence": {}, "method": {}}

    _wire_corpus_aggregate(monkeypatch, fake_agg)
    q = "2025-08-15 から 2025-09-07 の間に契約期間が重なる案件で、契約期間が 40日 以上の案件を主略称で挙げて。"
    en.cross_aggregate_pipeline(q)
    assert captured["min_days"] == 39  # 「40日以上」 (>=40) against corpus_aggregate's strict > ⇒ 39


# --------------------------------------------------------------------------- shape 3: staff population count
def test_staff_population_count_grounds(monkeypatch):
    _wire_corpus_aggregate(
        monkeypatch,
        lambda **k: {"value": {"count": 19, "people": ["a"] * 19},
                     "evidence": {"document_types": ["PP"], "projects": ["p"]},
                     "method": {"closure": "four_conditions_satisfied"}})
    q = ("各案件のPP・契約書・PLAN・FRにおいて、DA側の実施体制として役割付きで記載されている人物は"
         "全部で何人ですか。")
    out = en.cross_aggregate_pipeline(q)
    assert out is not None
    assert out["value"] == 19
    assert out["method"]["shape"] == "staff_population_count"


def test_staff_population_unclosed_falls_back(monkeypatch):
    # corpus_aggregate returns a None value when its own closure is not satisfied ⇒ fallback (no guess).
    _wire_corpus_aggregate(monkeypatch, lambda **k: {"value": None, "evidence": {}, "method": {}})
    q = ("各案件のPP・契約書・PLAN・FRにおいて、DA側の実施体制として役割付きで記載されている人物は"
         "全部で何人ですか。")
    assert en.cross_aggregate_pipeline(q) is None


# --------------------------------------------------------------------------- fallbacks (idx96/38/67/87/70-shaped)
def test_semantic_relation_question_falls_back(monkeypatch):
    # idx96 型: 「チェックポイント2として設定されている内容に関連する」 — semantic, not a supported shape.
    _wire_corpus_aggregate(monkeypatch, lambda **k: {"value": ["x"], "evidence": {}, "method": {}})
    q = "青葉与信マネジメントのチェックポイント2として設定されている内容に関連するタスクIDを教えてください。"
    assert en.full_enumeration_pipeline(q) is None


def test_apr_conditioned_aggregate_falls_back(monkeypatch):
    # idx38 型: APR-master cross-reference — no recognizer fires (no dates + no staff-roster shape).
    _wire_corpus_aggregate(monkeypatch, lambda **k: {"value": ["x"], "evidence": {}, "method": {}})
    q = "社内管理のAPRに照らして、APR-M3が必要な案件を主略称ですべて挙げ、それらの契約金額の合計を答えてください。"
    assert en.full_enumeration_pipeline(q) is None


# --------------------------------------------------------------------------- router + formatting e2e
def test_resolve_off_returns_none(monkeypatch):
    monkeypatch.delenv("RAG_DET_PIPELINE_ROUTER", raising=False)
    # flag OFF ⇒ champion serve path byte-identical: the pipeline is never consulted.
    assert dp.resolve("各案件のPP・契約書・PLAN・FRの役割付き人物は全部で何人ですか。", "cross_aggregate") is None


def test_end_to_end_staff_count_template_only(monkeypatch):
    _wire_corpus_aggregate(
        monkeypatch,
        lambda **k: {"value": {"count": 19, "people": []}, "evidence": {}, "method": {}})
    q = ("各案件のPP・契約書・PLAN・FRにおいて、DA側の実施体制として役割付きで記載されている人物は"
         "全部で何人ですか。")
    contract = dp.resolve(q, "cross_aggregate", force=True)
    assert contract is not None
    formatted = formatting.format_contract(contract, q, contract_type="cross_aggregate", force=True)
    assert formatted["value"] == "19"  # renders template-first
    assert formatted["method"]["formatting"]["template_only"] is True


def test_end_to_end_schedule_enum_template_only(monkeypatch, tmp_path):
    path = _write_schedule(tmp_path)
    _wire_schedule(monkeypatch, path)
    q = ("青嶺不動産アセットマネジメントのスケジュール_r2.xlsxにおいて、2025-08-11から2025-09-09の間に"
         "開始日または終了日が設定されているタスクIDをすべて挙げてください。")
    contract = dp.resolve(q, "full_enumeration", force=True)
    formatted = formatting.format_contract(contract, q, contract_type="full_enumeration", force=True)
    assert formatted["value"] == "T04、T05、T06"
    assert formatted["method"]["formatting"]["template_only"] is True
