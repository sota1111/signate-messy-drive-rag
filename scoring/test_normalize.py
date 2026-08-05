"""Tests for the meaning-preserving answer normalization layer (SOT-2448 / B2).

Two things must hold: (1) the transform only ever removes redundant formatting/prose while keeping
every value/identifier — the safety contract that makes it a zero-new-risk score lever; and (2)
applying it to a correct answer keeps that answer correct under the deterministic scorer, so it
cannot regress the sealed hold-out or the real-style generalization axes (the two-axis adoption
gate). The second property is the structural proof that this change clears both gates.
"""
from __future__ import annotations

import pytest

from scoring import deterministic
from src.rag import normalize as N


# --------------------------------- safety contract: abstentions -------------------------------
@pytest.mark.parametrize("a", ["わかりません", "該当なし", "不明", "N/A", "", "なし"])
def test_abstentions_are_never_normalized(a):
    assert N.normalize_answer(a) == a


# --------------------------------- meaning-preservation gate ----------------------------------
def test_preserves_meaning_accepts_pure_reformat():
    assert N.preserves_meaning("100 万ドル超", "100万ドル超")


def test_preserves_meaning_rejects_dropped_number():
    assert not N.preserves_meaning("合計 14,355,000円", "合計円")


def test_preserves_meaning_rejects_dropped_identifier():
    assert not N.preserves_meaning("T04、T05、T06", "T04、T05")


def test_preserves_meaning_rejects_new_content():
    # A candidate that introduces a character absent from the original is fabrication → rejected.
    assert not N.preserves_meaning("追加対応", "追加対応です")


def test_preserves_meaning_rejects_empty():
    assert not N.preserves_meaning("6", "")


def test_numbers_normalize_thousands_separator():
    assert N.numbers("25,000") == N.numbers("25000") == ["25000"]


# ------------------------------------ concrete transforms -------------------------------------
def test_digit_unit_space_is_removed():
    # The one committed test100 answer this changes (idx48): "100 万" → "100万" (JP never spaces a
    # number from its unit); the range and every value are preserved.
    out = N.normalize_answer("100 万ドル超 - 500 万ドル以下")
    assert out == "100万ドル超 - 500万ドル以下"
    assert N.preserves_meaning("100 万ドル超 - 500 万ドル以下", out)


def test_trailing_polite_copula_is_stripped():
    assert N.normalize_answer("追加対応です") == "追加対応"
    assert N.normalize_answer("有効と考えられます") == "有効"


def test_leading_answer_label_is_stripped():
    assert N.normalize_answer("回答: Age") == "Age"
    assert N.normalize_answer("答え：AOMINE") == "AOMINE"


def test_redundant_whitespace_is_collapsed():
    assert N.normalize_answer("A01、  A02、 A03") == "A01、A02、A03"


def test_already_concise_value_answers_are_unchanged():
    for a in ["6", "0.90527", "Age", "AOMINE", "T04、T05、T06",
              "n_estimators: 300, learning_rate: 0.1, random_state: random_state"]:
        assert N.normalize_answer(a) == a


def test_extraction_list_with_embedded_clause_is_preserved():
    # idx3: a bold-text extraction whose items include a full clause. The clause is a *required*
    # extraction, not verbosity, so normalization must NOT drop it — identity is the safe outcome.
    a = "time_and_materials、実績工数に基づき、案件完了後に最終成果物の検収を経て一括精算する。、30、分単位、25,000、円／時間"
    out = N.normalize_answer(a)
    assert "実績工数に基づき" in out
    assert N.numbers(out) == N.numbers(a)
    assert N.identifiers(out) == N.identifiers(a)


def test_env_toggle_disables_normalization(monkeypatch):
    monkeypatch.setenv(N._ENV_FLAG, "0")
    assert N.normalize_answer("100 万ドル超") == "100 万ドル超"
    monkeypatch.setenv(N._ENV_FLAG, "1")
    assert N.normalize_answer("100 万ドル超") == "100万ドル超"


# ============ structural two-axis non-regression proof (sealed hold-out × real-style) ==========
def _deterministic_gt_items():
    """Every deterministic-GT (truth, kind) pair backing the two generalization axes: the sealed
    hold-out / synth bench (``scoring.synth``) and the real-style transfer bench
    (``scoring.realstyle``). Both are LLM-free machine-extracted GT."""
    from scoring import realstyle, synth
    items = [(it.truth, it.kind) for it in synth.build()]
    items += [(it.truth, it.kind) for it in realstyle.build()]
    return items


def test_normalization_preserves_deterministic_correctness():
    """Applying the normalizer to a correct answer keeps it Perfect under the deterministic scorer.

    Since every committed answer is only ever *reformatted* (values preserved), a truth stays a
    perfect match after normalization — so neither generalization axis can drop. This is the gate
    proof: real-style gain ≥ 0 and hold-out gain ≥ 0, hence the two-axis adoption gate cannot block
    on account of this change.
    """
    items = _deterministic_gt_items()
    assert len(items) >= 50, "expected the full deterministic-GT pool"
    regressions = []
    for truth, kind in items:
        norm = N.normalize_answer(truth)
        if not N.preserves_meaning(truth, norm):
            regressions.append(("meaning", kind, truth, norm))
            continue
        if deterministic.score(norm, truth, kind) != "Perfect":
            regressions.append(("score", kind, truth, norm))
    assert not regressions, f"normalization regressed {len(regressions)} deterministic truths: {regressions[:5]}"
