"""Offline tests for the deterministic self-improvement harness (no LLM / network).

    .venv/bin/python -m pytest scoring/test_harness.py -q
"""
from __future__ import annotations

import collections
import json

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
    assert archetype.classify("この契約書の太字箇所をすべて抜き出してください。") == "enum_set"
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
    assert len(items) >= 250, f"expected at least 250 benchmark items, got {len(items)}"
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


# ------------------------------- hold-out split & trust decision (SOT-2424) --------------------
def test_split_rows_by_sealed_company(monkeypatch):
    from scoring import selfimprove

    monkeypatch.setenv("SEAL_COMPANIES", "株式会社青葉バイオメディカル機器")
    rows = [
        {"company": "株式会社青葉バイオメディカル機器", "points": 1.0},
        {"company": "京橋信用ソリューションズ株式会社", "points": 1.0},
        {"company": "横断", "points": 0.0},
    ]
    dev, hold = selfimprove.split_rows(rows)
    assert [r["company"] for r in hold] == ["株式会社青葉バイオメディカル機器"]
    assert {r["company"] for r in dev} == {"京橋信用ソリューションズ株式会社", "横断"}


def test_decide_trust_judges_on_holdout_not_dev():
    from scoring import selfimprove

    # dev looks great but the sealed hold-out is unreliable → NOT validated, and trust=False (evidence)
    dev = {"pivot_condition": {"precision": 1.0, "committed": 8, "coverage": 1.0,
                               "mean_score": 1.0, "n": 8,
                               "verdicts": {"Perfect": 8}}}
    hold = {"pivot_condition": {"precision": 0.4, "committed": 5, "coverage": 1.0,
                                "mean_score": 0.4, "n": 5, "verdicts": {"Perfect": 2}}}
    decided = selfimprove.decide_trust(dev, hold, threshold=0.8, min_committed=5, holdout_min=3)
    e = decided["pivot_condition"]
    assert e["holdout_validated"] is False
    assert e["trust"] is False and e["trust_basis"] == "holdout"


def test_decide_trust_validates_when_holdout_clears_threshold():
    from scoring import selfimprove

    dev = {"version_diff": {"precision": 1.0, "committed": 6, "coverage": 1.0,
                            "mean_score": 1.0, "n": 6, "verdicts": {"Perfect": 6}}}
    hold = {"version_diff": {"precision": 1.0, "committed": 4, "coverage": 1.0,
                             "mean_score": 1.0, "n": 4, "verdicts": {"Perfect": 4}}}
    decided = selfimprove.decide_trust(dev, hold, threshold=0.8, min_committed=5, holdout_min=3)
    e = decided["version_diff"]
    assert e["holdout_validated"] is True and e["trust"] is True


def test_decide_trust_insufficient_holdout_is_additive_safe():
    from scoring import selfimprove

    # only dev data (e.g. glossary/横断, never in the hold-out): not commit-validated, but trust stays
    # True when dev clears the bar so the additive abstain gate never blocks it.
    dev = {"glossary_formal": {"precision": 1.0, "committed": 9, "coverage": 0.6,
                               "mean_score": 0.6, "n": 15, "verdicts": {"Perfect": 9}}}
    decided = selfimprove.decide_trust(dev, {}, threshold=0.8, min_committed=5, holdout_min=3)
    e = decided["glossary_formal"]
    assert e["holdout_validated"] is False
    assert e["trust"] is True and e["trust_basis"] == "dev"


# ------------------------------- overfit detector (SOT-2424) -----------------------------------
def test_overfit_detects_state_4_valid_up_holdout_down():
    from scoring import overfit_check

    # the #4 regression: valid30 proxy +0.5833 while the real hold-out slice went −0.1
    v = overfit_check.assess(dev_gain=0.5833, holdout_gain=-0.1)
    assert v.overfit is True and v.label == "OVERFIT_SUSPECTED"


def test_overfit_ok_when_both_slices_rise_together():
    from scoring import overfit_check

    v = overfit_check.assess(dev_gain=0.10, holdout_gain=0.09)
    assert v.overfit is False


def test_overfit_ignores_noise_level_dev_change():
    from scoring import overfit_check

    v = overfit_check.assess(dev_gain=0.005, holdout_gain=-0.20)
    assert v.overfit is False  # no material dev gain → nothing to adopt/judge


# ------------------------------- hard-module advisory gate (SOT-2424) --------------------------
def test_hard_module_commits_only_when_holdout_validated(monkeypatch):
    from src.rag import generate

    q = "全案件で支払った税込金額をもとに、消費税額の総額を計算してください。"
    assert generate.compute.is_compute_question(q)

    # any LLM/retrieval reach is a bug in this path
    def _boom(*a, **k):
        raise AssertionError("must not call the LLM/retrieval on a validated hard commit")
    monkeypatch.setattr(generate.llm, "generate", _boom)
    monkeypatch.setattr(generate.retrieve, "get", _boom)

    monkeypatch.setattr(generate, "_load_trust",
                        lambda: {"cross_aggregate": {"holdout_validated": True}})
    res = generate.answer_question(q)
    assert res["verified"] is True and res["confidence"] == "high"
    assert res["answer"] and res["answer"] != generate.settings.ABSTAIN


def test_hard_module_advisory_when_not_validated(monkeypatch):
    from src.rag import generate

    q = "全案件で支払った税込金額をもとに、消費税額の総額を計算してください。"

    # unvalidated → the compute answer must NOT be committed directly; it flows through retrieval+LLM
    # as an advisory hint, and the (stubbed) verify pass abstains on unsupported hints.
    captured = {}

    class _Retriever:
        def retrieve(self, question, k=16):
            return [{"rel": "x", "text": "根拠", "kind": "text"}]
    monkeypatch.setattr(generate.retrieve, "get", lambda: _Retriever())
    monkeypatch.setattr(generate, "_gather_images", lambda *a, **k: [])

    def _fake_llm(prompt, **kw):
        captured["prompt"] = captured.get("prompt", "") + prompt
        # first pass draft: low confidence → abstain path
        return json.dumps({"answer": "", "confidence": "low"})
    monkeypatch.setattr(generate.llm, "generate", _fake_llm)
    monkeypatch.setattr(generate, "_load_trust",
                        lambda: {"cross_aggregate": {"holdout_validated": False}})

    res = generate.answer_question(q)
    # the deterministic compute candidate was injected as an advisory hint (not committed)
    assert "未検証の自動抽出候補" in captured["prompt"]
    assert res["answer"] == generate.settings.ABSTAIN
    assert res["verified"] is False
