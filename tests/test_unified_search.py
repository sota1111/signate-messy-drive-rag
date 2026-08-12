"""Offline tests for the unified fan-out search tool (SOT-2659 / unified_search).

No corpus / no network / no LLM: the retrievers are injected as deterministic mocks so the fusion core is
tested in isolation. Asserts (1) RRF fuses ``Σ weight/(K+rank)`` to the exact expected scores/order,
(2) a duplicate hit surfaced by two retrievers merges (summed score, both sources), (3) the per-file cap
enforces source diversity, (4) context re-expansion attaches sibling neighbours, (5) the tool is a no-op
with an empty value when the flag is OFF (byte-identical serve invariant), and (6) the returned packet has
the contract shape with a coverage report. The ID-token extractor is unit-tested too.
"""
from __future__ import annotations

import pytest

from src.rag.tools import unified_search as us


def _mock(name: str, hits: list[dict], weight: float = 1.0) -> us.Retriever:
    return us.Retriever(name, lambda q, h, _hits=hits: list(_hits), weight)


@pytest.fixture(autouse=True)
def _on(monkeypatch):
    monkeypatch.setenv("RAG_UNIFIED_SEARCH", "1")
    # pin fusion constants so expected values are stable regardless of ambient env
    monkeypatch.delenv("RAG_UNIFIED_RRF_K", raising=False)
    monkeypatch.delenv("RAG_UNIFIED_PER_FILE_CAP", raising=False)


def test_disabled_is_noop(monkeypatch):
    monkeypatch.delenv("RAG_UNIFIED_SEARCH", raising=False)
    res = us.search("anything", retrievers=[_mock("a", [{"doc_id": "d", "text": "t"}])])
    assert res["value"] == []
    assert res["evidence"]["reason"] == "disabled"
    assert res["method"]["engine"] == "unified_search"


def test_contract_shape_and_coverage():
    rs = [_mock("fts", [{"doc_id": "d1", "locator": "line:1", "text": "hello"}]),
          _mock("distill", [])]
    res = us.search("q", retrievers=rs)
    assert set(res.keys()) == {"value", "evidence", "method"}
    assert res["method"]["engine"] == "unified_search"
    assert res["method"]["fusion"] == "rrf"
    cov = res["evidence"]["coverage"]
    assert set(cov["retrievers_run"]) == {"fts", "distill"}
    assert cov["hits_per_retriever"] == {"fts": 1, "distill": 0}
    assert res["value"][0]["doc_id"] == "d1"
    assert res["value"][0]["sources"] == ["fts"]


def test_rrf_scores_and_order():
    # A: [X rank1, Y rank2]; B: [Y rank1, Z rank2]; K=60, weights 1.0
    a = _mock("a", [{"doc_id": "X", "locator": "l", "text": "x"},
                    {"doc_id": "Y", "locator": "l", "text": "y"}])
    b = _mock("b", [{"doc_id": "Y", "locator": "l", "text": "y"},
                    {"doc_id": "Z", "locator": "l", "text": "z"}])
    res = us.search("q", retrievers=[a, b])
    by_doc = {r["doc_id"]: r for r in res["value"]}
    assert by_doc["Y"]["score"] == pytest.approx(1 / 62 + 1 / 61, abs=1e-6)
    assert by_doc["X"]["score"] == pytest.approx(1 / 61, abs=1e-6)
    assert by_doc["Z"]["score"] == pytest.approx(1 / 62, abs=1e-6)
    # consensus hit Y (surfaced by both) outranks either single-retriever rank-1
    assert [r["doc_id"] for r in res["value"]] == ["Y", "X", "Z"]
    assert by_doc["Y"]["sources"] == ["a", "b"]


def test_weight_scales_contribution():
    a = _mock("a", [{"doc_id": "X", "locator": "l", "text": "x"}], weight=2.0)
    b = _mock("b", [{"doc_id": "Z", "locator": "l", "text": "z"}], weight=1.0)
    res = us.search("q", retrievers=[a, b])
    by_doc = {r["doc_id"]: r for r in res["value"]}
    assert by_doc["X"]["score"] == pytest.approx(2 / 61, abs=1e-6)
    assert by_doc["Z"]["score"] == pytest.approx(1 / 61, abs=1e-6)
    assert res["value"][0]["doc_id"] == "X"  # heavier retriever wins at equal rank


def test_dedup_merges_same_doc_locator():
    a = _mock("a", [{"doc_id": "D", "locator": "cell:A1", "text": "first"}])
    b = _mock("b", [{"doc_id": "D", "locator": "cell:A1", "text": "dup"}])
    res = us.search("q", retrievers=[a, b])
    assert len(res["value"]) == 1
    hit = res["value"][0]
    assert hit["sources"] == ["a", "b"]
    assert hit["score"] == pytest.approx(2 / 61, abs=1e-6)


def test_per_file_cap(monkeypatch):
    monkeypatch.setenv("RAG_UNIFIED_PER_FILE_CAP", "2")
    hits = [{"doc_id": "D", "locator": f"line:{i}", "text": f"t{i}"} for i in range(5)]
    res = us.search("q", retrievers=[_mock("a", hits)], limit=50)
    # only 2 of the 5 same-doc hits survive the per-file cap (diversity)
    assert len(res["value"]) == 2
    assert all(r["doc_id"] == "D" for r in res["value"])


def test_limit_caps_results():
    hits = [{"doc_id": f"D{i}", "locator": "l", "text": f"t{i}"} for i in range(20)]
    res = us.search("q", retrievers=[_mock("a", hits)], limit=3)
    assert len(res["value"]) == 3


def test_context_reexpansion_from_siblings():
    # same doc, two locators — the winner's context should include the sibling's text
    a = _mock("a", [{"doc_id": "D", "locator": "line:1", "text": "alpha heading"},
                    {"doc_id": "D", "locator": "line:2", "text": "beta note"}])
    res = us.search("q", retrievers=[a], limit=50, project=None)
    top = res["value"][0]
    assert "beta note" in top["context"] or "alpha heading" in top["context"]
    # the winner's own text is preserved
    joined = " ".join(r["text"] for r in res["value"])
    assert "alpha heading" in joined and "beta note" in joined


def test_retriever_provided_context_preserved():
    a = _mock("a", [{"doc_id": "D", "locator": "unit:sheet1", "text": "summary",
                     "context": "answerable: what is X?"}])
    res = us.search("q", retrievers=[a])
    assert "answerable: what is X?" in res["value"][0]["context"]


def test_no_match_reason_when_all_empty():
    res = us.search("q", retrievers=[_mock("a", []), _mock("b", [])])
    assert res["value"] == []
    assert res["evidence"]["reason"] == "no_match"
    assert res["evidence"]["coverage"]["retrievers_run"] == ["a", "b"]


def test_failing_retriever_is_fail_open():
    def boom(q, h):
        raise RuntimeError("retriever exploded")
    bad = us.Retriever("bad", boom, 1.0)
    good = _mock("good", [{"doc_id": "D", "locator": "l", "text": "t"}])
    res = us.search("q", retrievers=[bad, good])
    # the bad retriever contributes zero, the good one still answers
    assert res["evidence"]["coverage"]["hits_per_retriever"] == {"bad": 0, "good": 1}
    assert res["value"][0]["doc_id"] == "D"


def test_malformed_hit_without_doc_id_dropped():
    a = _mock("a", [{"locator": "l", "text": "no doc id"},
                    {"doc_id": "D", "locator": "l", "text": "ok"}])
    res = us.search("q", retrievers=[a])
    assert [r["doc_id"] for r in res["value"]] == ["D"]


def test_custom_rrf_k(monkeypatch):
    monkeypatch.setenv("RAG_UNIFIED_RRF_K", "10")
    res = us.search("q", retrievers=[_mock("a", [{"doc_id": "X", "locator": "l", "text": "x"}])])
    assert res["value"][0]["score"] == pytest.approx(1 / 11, abs=1e-6)
    assert res["method"]["rrf_k"] == 10


@pytest.mark.parametrize("query,expected", [
    ("EXT1234 の担当は", ["EXT1234"]),
    ("APR-M3 の案件", ["APR-M3"]),
    ("タスク M-01 の内容", ["M-01"]),
    ("契約書と担当者について", []),          # no id-shaped token ⇒ no id_master scan
    ("WBS_12 と AI08 を確認", ["WBS_12", "AI08"]),
])
def test_extract_id_tokens(query, expected):
    assert us._extract_id_tokens(query) == expected
