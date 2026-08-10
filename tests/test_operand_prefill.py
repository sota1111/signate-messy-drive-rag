"""SOT-2616 — tests for NUMERIC operand candidate prefill + PoT bound-operand handoff.

Network-free / corpus-free: the structure store and registry resolution are stubbed, so these exercise
the enumeration, ranking, catalog rendering, the PoT ``select``→catalog binding, and the Evidence Packet
OFF-byte-identical / ON-injection contract without any live document read.
"""
from __future__ import annotations

from src.rag.agent import evidence_packet as ep
from src.rag.agent import operand_prefill as opf
from src.rag.agent import pot_lane as pl
from src.rag.agent import query_router as qr
from src.rag.index import structure_store as store


# --------------------------------------------------------------------------- flag / defaults
def test_disabled_by_default():
    assert opf.enabled() is False


def test_max_candidates_default_and_override(monkeypatch):
    monkeypatch.delenv("RAG_OPERAND_PREFILL_MAX", raising=False)
    assert opf.max_candidates() == 12
    monkeypatch.setenv("RAG_OPERAND_PREFILL_MAX", "3")
    assert opf.max_candidates() == 3
    monkeypatch.setenv("RAG_OPERAND_PREFILL_MAX", "junk")
    assert opf.max_candidates() == 12  # bad value falls back to default


# --------------------------------------------------------------------------- structure store reader
_FAKE_STORE = {
    "files": {
        "案件A/train.xlsx": {
            "highlights": [
                {"value": "73,260円", "evidence": {"file": "案件A/train.xlsx", "sheet": "Sheet1",
                                                    "cell": "C5", "column": 3,
                                                    "group": {"項目": "契約単価"}},
                 "method": {"color": "yellow"}},
                {"value": "対象外", "evidence": {"file": "案件A/train.xlsx", "sheet": "Sheet1",
                                                 "cell": "A5"}, "method": {"color": "yellow"}},
                {"value": 11.2, "evidence": {"file": "案件A/train.xlsx", "sheet": "実績",
                                             "cell": "D9"}, "method": {"color": "blue"}},
            ]
        }
    }
}


def test_stored_numeric_cells_filters_non_numeric(monkeypatch):
    monkeypatch.setattr(store, "load", lambda path=None: _FAKE_STORE)
    cells = store.stored_numeric_cells("案件A/train.xlsx")
    values = [c["value"] for c in cells]
    assert "73,260円" in values
    assert 11.2 in values
    assert "対象外" not in values  # label, not an operand
    top = next(c for c in cells if c["value"] == "73,260円")
    assert top["number"] == 73260.0
    assert top["sheet"] == "Sheet1" and top["cell"] == "C5"


def test_stored_numeric_cells_absent_doc_returns_empty(monkeypatch):
    monkeypatch.setattr(store, "load", lambda path=None: _FAKE_STORE)
    assert store.stored_numeric_cells("案件Z/missing.xlsx") == []


# --------------------------------------------------------------------------- enumeration / ranking
def test_enumerate_from_highlights(monkeypatch):
    monkeypatch.setattr(store, "load", lambda path=None: _FAKE_STORE)
    monkeypatch.setattr(opf, "_row_candidates", lambda rel: [])  # highlight-only for this test
    cands = opf.enumerate_candidates("契約単価はいくら", ["案件A/train.xlsx"])
    assert cands, "expected at least one candidate"
    ids = [c.id for c in cands]
    assert ids == [f"op{i}" for i in range(1, len(cands) + 1)]  # stable dense ids
    top = cands[0]
    # "契約単価" overlaps the group label of the 73,260 cell → it ranks first.
    assert top.value == "73,260円"
    assert top.source == "案件A/train.xlsx:Sheet1!C5"
    assert top.origin == "highlight"


def test_enumerate_empty_without_docs(monkeypatch):
    monkeypatch.setattr(store, "load", lambda path=None: _FAKE_STORE)
    monkeypatch.setattr(opf, "_row_candidates", lambda rel: [])
    assert opf.enumerate_candidates("契約単価", []) == []


def test_enumerate_dedups_by_source(monkeypatch):
    dup = {"files": {"d/a.xlsx": {"highlights": [
        {"value": 5, "evidence": {"file": "d/a.xlsx", "sheet": "S", "cell": "B2"}, "method": {}},
        {"value": 5, "evidence": {"file": "d/a.xlsx", "sheet": "S", "cell": "B2"}, "method": {}},
    ]}}}
    monkeypatch.setattr(store, "load", lambda path=None: dup)
    monkeypatch.setattr(opf, "_row_candidates", lambda rel: [])
    cands = opf.enumerate_candidates("値", ["d/a.xlsx"])
    assert len(cands) == 1


def test_enumerate_respects_cap(monkeypatch):
    many = {"files": {"d/a.xlsx": {"highlights": [
        {"value": i, "evidence": {"file": "d/a.xlsx", "sheet": "S", "cell": f"B{i}"}, "method": {}}
        for i in range(1, 30)
    ]}}}
    monkeypatch.setattr(store, "load", lambda path=None: many)
    monkeypatch.setattr(opf, "_row_candidates", lambda rel: [])
    cands = opf.enumerate_candidates("値", ["d/a.xlsx"], limit=5)
    assert len(cands) == 5


def test_catalog_directive_lists_ids_and_sources(monkeypatch):
    monkeypatch.setattr(store, "load", lambda path=None: _FAKE_STORE)
    monkeypatch.setattr(opf, "_row_candidates", lambda rel: [])
    catalog = opf.build_catalog("契約単価", ["案件A/train.xlsx"])
    directive = opf.candidates_directive(catalog)
    assert "operand 候補" in directive
    assert "op1" in directive
    assert "verify_formula" in directive and "select" in directive
    assert "案件A/train.xlsx:Sheet1!C5" in directive


def test_empty_catalog_directive_is_blank():
    assert opf.candidates_directive([]) == ""


# --------------------------------------------------------------------------- PoT bound-operand handoff
def test_resolve_operand_selections_binds_from_catalog():
    catalog = [{"id": "op1", "value": "73,260", "unit": "円",
                "source": "案件A/train.xlsx:Sheet1!C5"}]
    candidates = [{"operands": [{"name": "base", "select": "op1"}],
                   "formula": {"ref": "base"}, "result_unit": "円"}]
    out = pl.resolve_operand_selections(candidates, catalog)
    op = out[0]["operands"][0]
    assert op["value"] == "73,260"
    assert op["unit"] == "円"
    assert op["source"] == "案件A/train.xlsx:Sheet1!C5"
    assert op["name"] == "base"  # role name preserved
    # inputs not mutated
    assert "value" not in candidates[0]["operands"][0]


def test_resolve_operand_selections_unknown_id_raises():
    import pytest
    with pytest.raises(pl.CatalogError):
        pl.resolve_operand_selections(
            [{"operands": [{"name": "x", "select": "opZ"}], "formula": {"ref": "x"}}],
            [{"id": "op1", "value": 1, "source": "d:s!a1"}])


def test_resolve_operand_selections_literal_passthrough():
    # An operand with an explicit value and no select is untouched even when a catalog is present.
    candidates = [{"operands": [{"name": "x", "value": 10, "source": "s"}], "formula": {"ref": "x"}}]
    out = pl.resolve_operand_selections(candidates, [{"id": "op1", "value": 99, "source": "z"}])
    assert out[0]["operands"][0]["value"] == 10


def test_verify_formula_catalog_commit_binds_selected_value():
    catalog = [
        {"id": "op1", "value": "73,260", "unit": "円", "source": "案件A/train.xlsx:Sheet1!C5"},
        {"id": "op2", "value": "2,000", "unit": "円", "source": "案件A/train.xlsx:Sheet1!C6"},
    ]
    candidates = [{
        "operands": [{"name": "base", "select": "op1"}, {"name": "add", "select": "op2"}],
        "formula": {"op": "ADD", "args": [{"ref": "base"}, {"ref": "add"}]},
        "result_unit": "円",
    }]
    out = pl.verify_formula(candidates, catalog=catalog)
    assert out["status"] == pl.COMMIT
    assert out["value"] == "75,260円"  # 73,260 + 2,000, bound from the catalog verbatim


def test_verify_formula_without_catalog_unchanged():
    # The literal-operand contract still works with no catalog (byte-compatible superset).
    out = pl.verify_formula([{
        "operands": [{"name": "a", "value": 3, "source": "s"},
                     {"name": "b", "value": 4, "source": "s"}],
        "formula": {"op": "MUL", "args": [{"ref": "a"}, {"ref": "b"}]},
    }])
    assert out["status"] == pl.COMMIT
    assert out["value"] == "12"


def test_verify_formula_catalog_unknown_id_is_error_not_raise():
    out = pl.verify_formula(
        [{"operands": [{"name": "x", "select": "opX"}], "formula": {"ref": "x"}}],
        catalog=[{"id": "op1", "value": 1, "source": "s"}])
    assert "error" in out


# --------------------------------------------------------------------------- Evidence Packet integration
def test_packet_off_is_byte_identical(monkeypatch):
    monkeypatch.delenv("RAG_OPERAND_PREFILL", raising=False)
    packet, directive = ep.build_directive("京橋のtrain.xlsxの平均年齢を教えて")
    assert "operand_candidates" not in packet.evidence
    assert "operand 候補" not in directive


def test_packet_embeds_catalog_when_enabled(monkeypatch):
    monkeypatch.setenv("RAG_OPERAND_PREFILL", "1")
    fake_catalog = [{"id": "op1", "value": "73,260", "unit": "円", "label": "契約単価",
                     "source": "案件A/train.xlsx:Sheet1!C5", "sheet": "Sheet1", "cell": "C5",
                     "doc": "案件A/train.xlsx", "origin": "highlight"}]
    # Stub resolution (so a doc is present) and enumeration (corpus-free).
    monkeypatch.setattr(ep, "_resolve_documents",
                        lambda *a, **k: (ep.ResolvedDocument("案件A/train.xlsx",
                                                             "案件A/train.xlsx", "alias", 0.9, "案件A"),))
    monkeypatch.setattr(opf, "build_catalog", lambda q, docs, **k: fake_catalog)
    packet, directive = ep.build_directive("契約単価に2000円足すといくら")
    assert packet.route == qr.NUMERIC
    assert packet.evidence.get("operand_candidates") == fake_catalog
    assert "operand 候補" in directive
    assert "op1" in directive
    # required slots (incl. operands) stay missing — the catalog is candidates, not a filled answer.
    assert "operands" in packet.missing


def test_packet_no_catalog_when_empty(monkeypatch):
    monkeypatch.setenv("RAG_OPERAND_PREFILL", "1")
    monkeypatch.setattr(ep, "_resolve_documents",
                        lambda *a, **k: (ep.ResolvedDocument("d/a.xlsx", "d/a.xlsx", "alias", 0.9, ""),))
    monkeypatch.setattr(opf, "build_catalog", lambda q, docs, **k: [])
    packet, directive = ep.build_directive("契約単価に2000円足すといくら")
    assert "operand_candidates" not in packet.evidence
    assert "operand 候補" not in directive
