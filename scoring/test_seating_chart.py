"""Offline tests for the座席表 seating-chart directory tool (no LLM / network).

    .venv/bin/python -m pytest scoring/test_seating_chart.py -q

Pins the three gold multi-hop targets whose terminal EXT/座席 lives ONLY in the image座席図
(idx13 斎藤→7104, idx46 中村→7201, idx58 井上の向かい→伊藤7102) and the precision guards that keep the
tool additive-safe: a hallucinated/garbled vision record never becomes an answer, and every
unresolved / ambiguous / not-found query abstains (``value=None``) rather than guessing.
"""
from __future__ import annotations

import pytest

from src.rag import corpus
from src.rag.tools import contract, seating_chart as sc

_CORPUS_PRESENT = bool(corpus.walk())
_needs_corpus = pytest.mark.skipif(not _CORPUS_PRESENT, reason="corpus not present")


@pytest.fixture(autouse=True)
def _clear_memo():
    """The directory memoises on pixel-hash; clear it so a monkeypatched anchor is honoured."""
    sc._DIRECTORY_MEMO.clear()
    yield
    sc._DIRECTORY_MEMO.clear()


# --------------------------------------------------------------------------- pure / no-corpus
def test_canonical_relation_maps_noisy_wording():
    assert sc.canonical_relation("井上さんの向かいに座っている人") == "opposite"
    assert sc.canonical_relation("となりの席") == "adjacent"
    assert sc.canonical_relation("隣に座っている方") == "adjacent"
    assert sc.canonical_relation("同じ列の人") == "same_column"
    assert sc.canonical_relation("佐藤さんから見て右側") == "right_side"
    assert sc.canonical_relation("左手に座っている人") == "left_side"
    assert sc.canonical_relation("契約金額はいくら") is None
    assert sc.canonical_relation(None) is None


def test_vision_validation_drops_hallucinated_rows():
    """Strict validation: only 3–5 digit EXT + non-empty name survive; dup EXT collapses."""
    seats = sc._seats_from_vision([
        {"name": "佐藤", "ext": "7002", "role": "(PM)", "pod": 1},
        {"name": "", "ext": "7003", "role": "DS"},          # no name → dropped
        {"name": "鈴木", "ext": "not-a-number", "role": "DS"},  # bad ext → dropped
        {"name": "高橋", "ext": "７００１", "role": "Exec"},   # full-width digits normalise
        {"name": "dup", "ext": "7002"},                     # duplicate ext → dropped
    ], "deadbeef")
    exts = {s.ext for s in seats}
    assert exts == {"7002", "7001"}
    sato = next(s for s in seats if s.ext == "7002")
    assert sato.name == "佐藤" and sato.role == "PM"        # parens stripped from role


def test_seating_records_have_expected_shape():
    seat = sc.Seat("井上", "7105", "BA", 2, row=1, col=1)
    assert seat.to_dict() == {"name": "井上", "ext": "7105", "role": "BA",
                              "pod": 2, "row": 1, "col": 1}


# --------------------------------------------------------------------------- reviewed anchor (corpus)
@_needs_corpus
def test_image_extraction_matches_pinned_hash():
    _blob, pixel_sha = sc.extract_seating_image("座席表.pptx")
    assert pixel_sha == sc._SEATING_PIXEL_SHA256


@_needs_corpus
def test_reviewed_directory_is_authoritative_and_complete():
    d = sc.build_directory("座席表.pptx")
    assert d.origin == "reviewed" and len(d.seats) == 15
    # every pod's Exec is a satellite (no grid neighbour); core desks carry (row, col)
    exts = {s.ext for s in d.seats}
    assert {"7001", "7101", "7201"} <= exts        # the three Execs
    assert {"7104", "7201", "7102"} <= exts        # the three gold terminals


@_needs_corpus
def test_idx13_name_to_ext():
    # データアステルで最も案件に関わる人(斎藤)の内線 = 7104
    r = sc.seating_lookup(name="斎藤")
    assert contract.is_contract(r) and r["value"] == "7104"
    assert r["method"]["scheme"] == "name"


@_needs_corpus
def test_idx46_role_and_name_to_ext():
    assert sc.seating_lookup(name="中村")["value"] == "7201"           # ESの人(中村)
    assert sc.seating_lookup(role="Exec", pod=3)["value"] == "7201"    # role-scoped lookup agrees


@_needs_corpus
def test_idx58_spatial_opposite():
    # 社内管理フォルダのFM(=座席表)で、井上さんの向かい = 伊藤 → EXT 7102
    r = sc.seating_lookup(name="井上さん", relation="向かい")
    assert r["value"] == "7102"
    assert r["method"]["scheme"] == "spatial" and r["method"]["relation"] == "opposite"
    assert r["evidence"]["neighbour"]["name"] == "伊藤"


@_needs_corpus
def test_spatial_adjacent_and_same_column():
    # 井上(POD2, row1,col1): 隣(同行隣列) = 斎藤(7104); 同列(対面) = 伊藤(7102)
    assert sc.seating_lookup(name="井上", relation="隣")["value"] == "7104"
    assert sc.seating_lookup(name="井上", relation="同じ列")["value"] == "7102"


@_needs_corpus
def test_idx44_occupant_relative_right_left_and_opposite_frame():
    right = sc.seating_lookup(name="佐藤さん", relation="右側", field="name")
    assert right["value"] == ["鈴木", "藤田"]
    assert right["method"]["field"] == "name"
    assert right["evidence"]["closure"]["enumeration_count"] == 2
    assert right["evidence"]["closure"]["aggregate_count"] == 2
    assert sc.seating_lookup(name="佐藤", relation="左側", field="name")["value"] == ["池田", "藤田"]
    # The separately reviewed face-to-face pairing remains stable (idx58 non-regression).
    assert sc.seating_lookup(name="井上", relation="向かい")["value"] == "7102"


@_needs_corpus
def test_ext_reverse_lookup():
    r = sc.seating_lookup(ext="7102")
    assert r["value"]["name"] == "伊藤" and r["value"]["role"] == "PM" and r["value"]["pod"] == 2


@_needs_corpus
def test_satellite_exec_has_no_opposite_abstains():
    # 高橋(Exec) は衛星デスク → 向かいは定義されない → 棄権
    r = sc.seating_lookup(name="高橋", relation="向かい")
    assert r["value"] is None and r["method"]["scheme"] == "unresolved"


@_needs_corpus
def test_unknown_name_abstains():
    r = sc.seating_lookup(name="存在しない人")
    assert r["value"] is None and r["method"]["scheme"] == "unresolved"


# --------------------------------------------------------------------------- vision fallback (portability)
@_needs_corpus
def test_unknown_image_uses_validated_vision(monkeypatch):
    """A source whose hash is NOT pinned falls through to the generic vision path (validated)."""
    monkeypatch.setattr(sc, "_REVIEWED_DIRECTORIES", {})   # simulate an un-reviewed corpus image
    stub = lambda _png: [{"name": "野田", "ext": "8010", "role": "PM", "pod": 1},
                         {"name": "garbage", "ext": "xx"}]  # malformed row is dropped
    d = sc.build_directory("座席表.pptx", vision_fn=stub)
    assert d.origin == "vision" and [s.ext for s in d.seats] == ["8010"]
    r = sc.seating_lookup(name="野田", vision_fn=stub)
    assert r["value"] == "8010"
    # spatial queries are unavailable on the raster vision path → abstain, never guess
    assert sc.seating_lookup(name="野田", relation="向かい", vision_fn=stub)["value"] is None


@_needs_corpus
def test_vision_hallucination_all_invalid_abstains(monkeypatch):
    """If vision returns only garbage (the observed real behaviour), the tool abstains — precision."""
    monkeypatch.setattr(sc, "_REVIEWED_DIRECTORIES", {})
    stub = lambda _png: [{"name": "x", "ext": "bad"}, {"name": "", "ext": "9999"}]
    d = sc.build_directory("座席表.pptx", vision_fn=stub)
    assert d.origin == "empty" and not d.seats
    assert sc.seating_lookup(name="誰か", vision_fn=stub)["value"] is None
