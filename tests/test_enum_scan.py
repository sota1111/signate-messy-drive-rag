"""SOT-2587 — tests for the symbolic exhaustive-scan ENUM lane + completeness certificate."""
from __future__ import annotations

import re

import pytest

from src.rag.agent import enum_scan as es
from src.rag.agent import enumeration as _enum
from src.rag.agent import failure_taxonomy as _tax


# --------------------------------------------------------------------------- fixtures
def _reg(doc_id, case_id="案件A", ext="xlsx", caps=("openpyxl",), warnings=(), title=None):
    """A minimal registry row (only the fields the scan reads)."""
    return {
        "doc_id": doc_id,
        "case_id": case_id,
        "full_path": doc_id,
        "normalized_filename": doc_id.rsplit("/", 1)[-1].lower(),
        "filename_aliases": [doc_id.rsplit("/", 1)[-1].lower()],
        "extension": ext,
        "title": title or doc_id.rsplit("/", 1)[-1].rsplit(".", 1)[0],
        "parser_capabilities": list(caps),
        "extraction_warnings": list(warnings),
        "named_entities": [],
    }


def _ev(rel, typ, value, case="案件A"):
    return {"type": typ, "value": value, "key": value.lower(), "project": case,
            "file": rel.rsplit("/", 1)[-1], "rel": rel, "sheet": "", "cell": "", "paragraph": ""}


REGISTRY = [
    _reg("案件A/train.xlsx"),
    _reg("案件A/schedule.xlsx"),
    _reg("案件A/scan.png", ext="png", caps=("vision",)),          # image-only → unsupported
    _reg("案件A/locked.xlsx", warnings=("encrypted",)),           # encrypted → unsupported
    _reg("案件B/other.xlsx", case_id="案件B"),
]

INDEX = [
    _ev("案件A/train.xlsx", "person", "佐藤"),
    _ev("案件A/train.xlsx", "person", "鈴木"),
    _ev("案件A/schedule.xlsx", "person", "藤田"),
    _ev("案件A/schedule.xlsx", "column", "T04"),
    _ev("案件A/schedule.xlsx", "column", "T05"),
    _ev("案件B/other.xlsx", "person", "田中", case="案件B"),        # out of scope for 案件A
]


# --------------------------------------------------------------------------- flag / import
def test_disabled_by_default(monkeypatch):
    monkeypatch.delenv("RAG_ENUM_SCAN", raising=False)
    assert es.enabled() is False


def test_enabled_reads_flag(monkeypatch):
    monkeypatch.setenv("RAG_ENUM_SCAN", "1")
    assert es.enabled() is True


# --------------------------------------------------------------------------- universe resolution
def test_universe_project_scope_counts_and_unsupported():
    u = es.resolve_universe("案件Aの担当者をすべて挙げてください", project="案件A",
                            registry_rows=REGISTRY)
    assert u.documents_total == 5
    assert u.documents_applicable == 4                     # 4 in 案件A
    assert u.unsupported_documents == 2                    # png(vision) + encrypted
    assert u.documents_scanned == 2
    assert u.complete is False
    assert set(u.unsupported_ids) == {"案件A/scan.png", "案件A/locked.xlsx"}


def test_universe_explicit_file_scope():
    u = es.resolve_universe("schedule.xlsx のタスクIDをすべて挙げて", registry_rows=REGISTRY)
    assert u.scope == "explicit_file"
    assert u.applicable_ids == ("案件A/schedule.xlsx",)
    assert u.complete is True                              # single scannable file, fully covered


def test_universe_complete_when_all_scannable():
    rows = [_reg("案件C/a.xlsx", case_id="案件C"), _reg("案件C/b.xlsx", case_id="案件C")]
    u = es.resolve_universe("案件Cのメンバーをすべて挙げて", project="案件C", registry_rows=rows)
    assert u.complete is True
    assert u.to_dict() == {"documents_total": 2, "documents_applicable": 2,
                           "documents_scanned": 2, "unsupported_documents": 0}


# --------------------------------------------------------------------------- scan + certificate
def test_scan_person_population_matches_scanned_docs_only():
    res = es.scan("案件Aの担当者をすべて挙げてください", project="案件A",
                  registry_rows=REGISTRY, index_rows=INDEX)
    assert res.population_kind == _enum.PERSON
    # 佐藤/鈴木/藤田 come from scannable 案件A docs; 田中 (案件B) is out of scope.
    assert set(res.matched) == {"佐藤", "鈴木", "藤田"}
    cert = res.certificate()
    assert set(cert) == {"matched", "universe", "complete"}
    assert cert["universe"]["documents_scanned"] == 2
    assert cert["complete"] is False


def test_scan_predicate_filters_elements():
    res = es.scan("schedule.xlsx のタスクIDをすべて挙げて", predicate=r"^T0[45]$",
                  population_kind=_enum.TASK, registry_rows=REGISTRY, index_rows=INDEX)
    assert set(res.matched) == {"T04", "T05"}
    assert res.complete is True


def test_scan_document_population_enumerates_universe_filenames():
    rows = [_reg("案件D/報告1.docx", case_id="案件D", ext="docx"),
            _reg("案件D/報告2.docx", case_id="案件D", ext="docx")]
    res = es.scan("案件Dのファイルをすべて挙げて", population_kind=_enum.DOCUMENT,
                  project="案件D", registry_rows=rows, index_rows=[])
    assert set(res.matched) == {"報告1", "報告2"}
    assert res.complete is True


# --------------------------------------------------------------------------- the idx16-type guard
def test_guard_forbids_none_when_unsupported_and_empty():
    # A predicate nothing matches, over a universe with unsupported docs → guard must fire.
    res = es.scan("案件Aの担当者をすべて挙げてください", predicate="存在しない値XYZ",
                  project="案件A", registry_rows=REGISTRY, index_rows=INDEX)
    assert res.matched == ()
    assert res.guard_triggered is True
    assert es.forbids_none_answer(res) is True
    assert res.may_answer_none is False
    assert res.state_code == _tax.PARSER_CAPABILITY_MISS


def test_guard_not_triggered_when_matches_exist_despite_unsupported():
    res = es.scan("案件Aの担当者をすべて挙げてください", project="案件A",
                  registry_rows=REGISTRY, index_rows=INDEX)
    assert res.matched
    assert res.guard_triggered is False
    assert res.may_answer_none is True


def test_evidence_based_empty_forbids_none_even_when_complete():
    # SOT-2600 (idx70): an evidence-based enumeration (person/task/…) whose single-path index scan matched
    # nothing must NOT be concluded as "該当なし", even over a fully-scannable (complete) universe — index
    # absence is not proof of non-existence. The guard fires and codes EVIDENCE_INCOMPLETE (not the
    # parser-miss code, since no document was unreadable).
    rows = [_reg("案件E/a.xlsx", case_id="案件E")]
    res = es.scan("案件Eの担当者をすべて挙げて", predicate="不一致", project="案件E",
                  registry_rows=rows, index_rows=[])
    assert res.matched == ()
    assert res.universe.complete is True         # no unsupported docs
    assert res.negative_certifiable is False     # …but a person population is not self-certifying
    assert res.guard_triggered is True
    assert es.forbids_none_answer(res) is True
    assert res.may_answer_none is False
    assert res.state_code == _tax.EVIDENCE_INCOMPLETE


def test_idx70_task_id_enumeration_empty_is_not_none_answer():
    # idx70-shape: a TASK-ID enumeration filtered by a predicate that matches no persisted row. Even though
    # the universe is fully scannable, the empty result must degrade (may_answer_none=False), never assert
    # "該当なし" — the real answer (AI-05) lives in evidence the single index path did not surface.
    rows = [_reg("白峰信用リスク評価/報告_0527.docx", case_id="白峰信用リスク評価", ext="docx")]
    res = es.scan("白峰信用リスク評価の未完了のアクションIDを挙げて", predicate=r"^AI-\d+$",
                  project="白峰信用リスク評価", population_kind=_enum.TASK,
                  registry_rows=rows, index_rows=[])
    assert res.matched == ()
    assert res.population_kind == _enum.TASK
    assert res.may_answer_none is False
    assert res.state_code == _tax.EVIDENCE_INCOMPLETE


def test_document_population_empty_over_complete_universe_may_answer_none():
    # The one self-certifying case: a *document* enumeration whose members ARE the scanned universe. Over a
    # complete universe, an empty match genuinely certifies "該当なし" — the guard must NOT fire (otherwise
    # a legitimate document non-existence answer would be wrongly suppressed).
    rows = [_reg("案件D/報告1.docx", case_id="案件D", ext="docx")]
    res = es.scan("案件Dの Excel ファイルをすべて挙げて", predicate=r"\.xlsx$",
                  project="案件D", population_kind=_enum.DOCUMENT, registry_rows=rows, index_rows=[])
    assert res.matched == ()
    assert res.universe.complete is True
    assert res.negative_certifiable is True
    assert res.guard_triggered is False
    assert res.may_answer_none is True
    assert res.state_code == ""


# --------------------------------------------------------------------------- agent tool
def test_tool_returns_certificate_shape(monkeypatch):
    # Route through the real artifacts-free path by injecting rows via scan()'s kwargs is not possible
    # through the tool; instead stub the module-level loaders so the tool exercises the artifact seam.
    monkeypatch.setattr(es._dr, "load", lambda *a, **k: REGISTRY)
    monkeypatch.setattr(es._ev, "load", lambda *a, **k: INDEX)
    monkeypatch.setattr(es._dr, "Resolver", lambda rows: _StubResolver())
    out = es.enum_scan_tool("案件Aの担当者をすべて挙げてください", project="案件A")
    assert "matched" in out and "universe" in out and "complete" in out
    assert out["universe"]["documents_total"] == 5
    assert out["may_answer_none"] in (True, False)


class _StubResolver:
    def resolve_explicit(self, question, project=None):
        return []


def test_tool_empty_question_is_error():
    assert "error" in es.enum_scan_tool("")


def test_tool_never_raises(monkeypatch):
    def _boom(*a, **k):
        raise RuntimeError("boom")
    monkeypatch.setattr(es, "scan", _boom)
    out = es.enum_scan_tool("案件Aの担当者をすべて挙げて")
    assert "error" in out


# --------------------------------------------------------------------------- directive (no corpus fact)
def test_directive_names_tool_and_guard():
    d = es.enum_lane_directive("案件Aの担当者をすべて挙げてください")
    assert es.TOOL_NAME in d
    assert "completeness certificate" in d
    assert "該当なし" in d              # the guard rule is stated
    assert "全数走査" in d


# --------------------------------------------------------------------------- investigator wiring (OFF byte-identical)
def test_tool_absent_when_flag_off(monkeypatch):
    import importlib
    from src.rag.tools.profile import CorpusProfile
    monkeypatch.delenv("RAG_ENUM_SCAN", raising=False)
    inv = importlib.import_module("src.rag.agent.investigator")
    names = {t.name for t in inv.build_generic_tools(CorpusProfile())}
    assert es.TOOL_NAME not in names          # champion tool set is byte-identical (no enum_scan tool)


def test_tool_present_when_flag_on(monkeypatch):
    import importlib
    from src.rag.tools.profile import CorpusProfile
    monkeypatch.setenv("RAG_ENUM_SCAN", "1")
    inv = importlib.import_module("src.rag.agent.investigator")
    names = {t.name for t in inv.build_generic_tools(CorpusProfile())}
    assert es.TOOL_NAME in names


# --------------------------------------------------------------------------- offline diagnostics
def test_measure_enum_computes_four_metrics():
    cases = [
        es.EnumCase(index=1, question="案件Aの担当者をすべて挙げてください", project="案件A",
                    gold_universe=("案件A/train.xlsx", "案件A/schedule.xlsx"),
                    gold_set=("佐藤", "鈴木", "藤田"), population_kind=_enum.PERSON),
        es.EnumCase(index=2, question="schedule.xlsx のT04,T05をすべて挙げて",
                    gold_universe=("案件A/schedule.xlsx",), gold_set=("T04", "T05"),
                    predicate=r"^T0[45]$", population_kind=_enum.TASK),
    ]
    m = es.measure_enum(cases, registry_rows=REGISTRY, index_rows=INDEX)
    assert m["n_cases"] == 2
    assert m["universe_resolution_accuracy"] == 1.0        # both gold universes are in scope
    assert 0.0 <= m["exhaustive_scan_completion"] <= 1.0
    assert m["set_precision"] == 1.0 and m["set_recall"] == 1.0
    assert m["graded_set_cases"] == 2
