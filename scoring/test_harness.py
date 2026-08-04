"""Offline tests for the deterministic self-improvement harness (no LLM / network).

    .venv/bin/python -m pytest scoring/test_harness.py -q
"""
from __future__ import annotations

import collections

from scoring import deterministic as D
from scoring import synth
from src.rag import archetype


# ------------------------------- deterministic scorer -----------------------------------------
def test_numeric_perfect_and_comma():
    assert D.score("5775000", "5,775,000", "numeric") == "Perfect"
    assert D.score("5,775,000円", "5775000", "numeric") == "Perfect"


def test_numeric_rounding_acceptable_and_incorrect():
    assert D.score("0.72243", "0.7224", "numeric") == "Acceptable"  # equal at the GT's precision
    assert D.score("0.95", "0.8999", "numeric") == "Incorrect"


def test_abstain_is_missing_not_incorrect():
    assert D.score("わかりません", "42", "numeric") == "Missing"
    assert D.score("", "x", "string") == "Missing"


def test_set_order_independent():
    assert D.score("A、B、C", "C、A、B", "set") == "Perfect"
    assert D.score("A、B", "A、B、C", "set") == "Incorrect"


def test_string_containment():
    assert D.score("hist_gradient_boosting", "hist_gradient_boosting", "string") == "Perfect"
    assert D.score("dog", "cat", "string") == "Incorrect"


# ------------------------------- archetype classifier -----------------------------------------
def test_classify_known_archetypes():
    assert archetype.classify("青葉のproject_config.jsonの model_type は何ですか。") == "config_model_type"
    assert archetype.classify("metrics.json における accuracy の値を答えてください。") == "metric_score"
    assert archetype.classify("社内用語集で「PP」の正式名称は何ですか。") == "glossary_formal"
    assert archetype.classify("この契約書の太字箇所をすべて抜き出してください。") == "unknown"
    assert archetype.classify(
        "青嶺の提案書について、oldフォルダ内の旧版と最新版を比較し、変更された箇所を変更前と変更後で答えてください。"
    ) == "version_diff"


def test_kind_of():
    assert archetype.kind_of("config_hyperparam") == "numeric"
    assert archetype.kind_of("glossary_formal") == "string"
    assert archetype.kind_of("nonexistent") == "string"


# ------------------------------- synthetic benchmark ------------------------------------------
# The 8 core archetypes are bulk-generated (≥10 each); version_diff is corpus-limited (one item per
# real version pair) so it is validated separately below.
_CORE_ARCHETYPES = {
    "config_model_type", "config_hyperparam", "metric_score", "data_shape",
    "csv_column_mean", "csv_column_max", "glossary_formal", "glossary_abbrev",
}


def test_synth_builds_and_self_scores_perfect():
    items = synth.build()
    assert len(items) >= 100, f"expected at least 100 benchmark items, got {len(items)}"
    counts = collections.Counter(it.archetype for it in items)
    core = {a: n for a, n in counts.items() if a in _CORE_ARCHETYPES}
    assert set(core) == _CORE_ARCHETYPES, f"expected the 8 core archetypes, got {counts}"
    assert min(core.values()) >= 10, f"every core archetype needs at least 10 items: {counts}"
    # every programmatically-extracted truth must score Perfect against itself (rubric alignment)
    for it in items:
        assert D.score(it.truth, it.truth, it.kind) == "Perfect", f"{it.id}: {it.truth!r}"
    # archetype labels are all known comparators
    for it in items:
        assert it.kind in ("numeric", "set", "string")


def test_synth_version_diff_present_and_labelled():
    items = synth.build()
    vd = [it for it in items if it.archetype == "version_diff"]
    assert len(vd) >= 3, f"expected several version_diff benchmark items, got {len(vd)}"
    for it in vd:
        assert it.kind == "string"
        assert "→" in it.truth  # a "変更前 → 変更後" rendering
        assert archetype.classify(it.question) == "version_diff"


# ------------------------------- additive trust gate ------------------------------------------
def test_trust_gate_blocks_untrusted_without_llm(monkeypatch):
    from src.rag import generate

    # a question that classifies to an archetype we mark untrusted
    q = "青葉の train.csv の balance 列の平均値を小数第2位まで答えてください。"
    assert archetype.classify(q) == "csv_column_mean"
    monkeypatch.setattr(generate, "_load_trust",
                        lambda: {"csv_column_mean": {"trust": False}})
    res = generate.answer_question(q)  # must NOT reach retrieval / LLM
    assert res["answer"] == generate.settings.ABSTAIN
    assert res["confidence"] == "untrusted-archetype"


def test_trust_gate_noop_when_map_missing(monkeypatch):
    from src.rag import generate

    monkeypatch.setattr(generate, "_load_trust", lambda: {})
    # unknown / unmapped archetype is never blocked
    assert generate._trust_blocks("この契約書の太字箇所を抜き出してください。") is False
