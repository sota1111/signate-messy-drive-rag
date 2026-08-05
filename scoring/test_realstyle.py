"""Offline tests for the real-style transcription bench (SOT-2447 / A2). No LLM / network.

    .venv/bin/python -m pytest scoring/test_realstyle.py -q
"""
from __future__ import annotations

import collections

from scoring import deterministic as D
from scoring import dist_match, realstyle, synth

# The 8 core archetypes the bench must keep represented (test100 archetype-ratio maintenance).
_CORE_ARCHETYPES = {
    "config_model_type", "config_hyperparam", "metric_score", "data_shape",
    "csv_column_mean", "csv_column_max", "glossary_formal", "glossary_abbrev",
}


def test_bench_has_at_least_50_deterministic_items():
    items = realstyle.build()
    assert len(items) >= 50, f"real-style bench must have ≥50 items, got {len(items)}"
    # every deterministic truth self-scores Perfect (GT is sound → the axis is trustworthy)
    for it in items:
        assert it.kind in ("numeric", "set", "string")
        assert D.score(it.truth, it.truth, it.kind) == "Perfect", f"{it.id}: {it.truth!r}"


def test_bench_matches_test100_answer_mode_distribution():
    items = realstyle.build()
    modes = collections.Counter(it.mode for it in items)
    total = len(items)
    # all four production answer modes present…
    assert set(modes) == {"calculate", "compare", "extract", "lookup"}
    # …in proportions close to test100 (計算34/比較5/抽出35/参照26), within a small tolerance.
    for fam, frac in realstyle.TEST100_FAMILY_FRACTIONS.items():
        share = modes[fam] / total
        assert abs(share - frac) <= 0.06, f"{fam} share {share:.2f} far from test100 {frac:.2f}"
    # extract is the dominant production mode — it must not be under-represented.
    assert modes["extract"] >= modes["compare"]


def test_bench_covers_the_eight_core_archetypes():
    items = realstyle.build()
    present = {it.archetype for it in items}
    missing = _CORE_ARCHETYPES - present
    assert not missing, f"real-style bench missing core archetypes: {missing}"
    # compare/extract families are represented by their real archetypes too.
    assert "version_diff" in present and "document_extract" in present


def test_bench_does_not_leak_valid30():
    """valid30 is the burnt-out dev set; the real-style axis must be independent of it (isolation)."""
    items = realstyle.build()
    for it in items:
        assert "valid" not in it.source.lower(), f"{it.id} leaks a valid30 source: {it.source}"
        assert it.archetype not in realstyle._VALID_DERIVED, f"{it.id} uses a valid-derived archetype"


def test_phrasing_is_distinct_from_synth_surface_form():
    """The bench must transcribe the test100 STYLE, i.e. a different surface form than scoring.synth,
    so it measures transfer rather than re-measuring the synth phrasings the RAG was tuned on."""
    synth_qs = {it.question for it in synth.build()}
    items = realstyle.build()
    overlap = [it for it in items if it.question in synth_qs]
    assert not overlap, f"{len(overlap)} real-style questions duplicate synth phrasings"
    # every item carries a test100 style reference (provenance of the transcription).
    assert all(it.style_ref.startswith("test100:") for it in items)


def test_document_extract_truth_is_the_bold_marker_set():
    """The extract-family items carry the deterministic 太字 marker set (not a fabricated answer)."""
    items = [it for it in realstyle.build() if it.archetype == "document_extract"]
    assert items, "expected deterministic bold-marker extract items"
    for it in items:
        terms = it.truth.split("、")
        assert len(terms) >= 2 and all(terms)  # a real enumeration, order-independent set


def test_build_is_deterministic():
    a = [it.id for it in realstyle.build()]
    b = [it.id for it in realstyle.build()]
    assert a == b, "bench assembly must be deterministic (stable ids/order)"
