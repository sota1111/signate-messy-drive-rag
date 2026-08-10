"""SOT-2588 — tests for the version-diff block-alignment + edit-intent lane (RAG_DIFF_ALIGN, opt-in).

    .venv/bin/python -m pytest scoring/test_diff_align.py -q

The classification / ranking tests are pure (no corpus). The registry version-family resolution and the
xlsx precision guard read the SIGNATE corpus, so they skip cleanly when it is absent (lean CI image).
"""
from __future__ import annotations

import pytest

from src.rag import corpus, diffpair
from src.rag.diffpair import (
    BOILERPLATE, LAYOUT_METADATA, SUBSTANTIVE, UNCERTAIN, Change, _INTENT_RANK,
    classify_change,
)

_CORPUS_PRESENT = bool(corpus.walk())
_needs_corpus = pytest.mark.skipif(not _CORPUS_PRESENT, reason="corpus not present")

# idx1 — old版 paired with the *unversioned* latest via the registry family (not the filename rules).
_Q_IDX1 = ("恒一会 かえで総合病院の最終報告書old版と最新版を比較したとき、案件遂行に関連する"
           "実質的な変更を挙げてください。")
# idx95 — a schedule xlsx r1→r2 pair (whole-sheet churn) that must stay an abstention.
_Q_IDX95 = ("青嶺不動産アセットマネジメントのスケジュール_r1.xlsxとスケジュール_r2.xlsxを比較したとき、"
            "案件遂行に関連する変更を挙げてください。")


# ---------------------------------- opt-in / byte-identical OFF ----------------------------------
def test_align_disabled_by_default(monkeypatch):
    monkeypatch.delenv("RAG_DIFF_ALIGN", raising=False)
    assert diffpair.align_enabled() is False


def test_align_enabled_truthy(monkeypatch):
    for v in ("1", "true", "on", "YES"):
        monkeypatch.setenv("RAG_DIFF_ALIGN", v)
        assert diffpair.align_enabled() is True
    monkeypatch.setenv("RAG_DIFF_ALIGN", "0")
    assert diffpair.align_enabled() is False


# ------------------------------------- edit-intent classification -------------------------------
def test_money_modify_is_substantive_high():
    c = Change("第5条/契約金額", "契約金額は8,000万円とする。", "契約金額は8,500万円とする。", "modify")
    rc = classify_change(c)
    assert rc.intent == SUBSTANTIVE
    assert rc.score >= 0.9  # an in-place value MODIFY is the strongest signal
    assert "money" in rc.features


def test_person_change_is_substantive():
    # 担当者 / 契約当事者 change carries no numeric marker but is a first-class substantive edit (idx74).
    rc = classify_change(Change("担当者", "藤田 彩", "井上 里奈", "modify"))
    assert rc.intent == SUBSTANTIVE
    assert "person_name" in rc.features


def test_footer_and_pagination_are_boilerplate():
    footer = classify_change(Change("", "Copyright 2024 ACME Inc.", "Copyright 2025 ACME Inc.", "modify"))
    assert footer.intent == BOILERPLATE
    page = classify_change(Change("", "1", "2", "modify"))  # bare page/bullet number
    assert page.intent == BOILERPLATE


def test_added_summary_section_is_layout_not_substantive():
    # An added executive-summary re-presents existing figures (idx1): demote below the real edit even
    # though it restates money/dates.
    c = Change("スライド2", "", "エグゼクティブサマリ 契約期間 2025年9月2日 最終請求金額 3,850,000円", "add")
    rc = classify_change(c)
    assert rc.intent == LAYOUT_METADATA
    assert "summary_representation" in rc.features


def test_moved_block_is_layout_metadata():
    moved = {diffpair._norm("STEP 4 方針合意")}
    add = classify_change(Change("スライド8", "", "STEP 4 方針合意", "add"), moved)
    assert add.intent == LAYOUT_METADATA
    assert "moved_block" in add.features


def test_identifier_change_is_substantive():
    # idx14: データ列名をアンダースコア表記へ (loan status → loan_status) is a schema edit.
    rc = classify_change(Change("列名", "loan status", "loan_status", "modify"))
    assert rc.intent == SUBSTANTIVE
    assert "identifier" in rc.features


def test_boilerplate_ranks_below_substantive():
    # Acceptance criterion #2: footer/version/pagination sort strictly below a substantive change.
    subst = classify_change(Change("金額", "100万円", "120万円", "modify"))
    boiler = classify_change(Change("", "1", "2", "modify"))
    assert _INTENT_RANK[subst.intent] > _INTENT_RANK[boiler.intent]
    assert subst.score > boiler.score


# ------------------------------------- registry version-family (idx1) ---------------------------
@_needs_corpus
def test_off_path_does_not_resolve_idx1_family(monkeypatch):
    # OFF: the filename rules alone cannot pair `_old` with the unversioned latest → abstain (unchanged).
    monkeypatch.delenv("RAG_DIFF_ALIGN", raising=False)
    assert diffpair.resolve_pair(_Q_IDX1) is None
    assert diffpair.answer_question(_Q_IDX1) is None


@_needs_corpus
def test_registry_family_resolves_idx1_and_surfaces_substantive(monkeypatch):
    monkeypatch.setenv("RAG_DIFF_ALIGN", "1")
    pair = diffpair._resolve_pair_for_render(_Q_IDX1)
    assert pair is not None
    assert pair.basis == "registry-family"
    ranked = diffpair.rank_changes(pair)
    assert ranked and ranked[0].intent == SUBSTANTIVE
    ans = diffpair.answer_question(_Q_IDX1)
    # the substantive change is the deleted performance-comparison table (AUC-ROC / F1 values), not the
    # added executive summary the champion mis-selected.
    assert ans and ("AUC-ROC" in ans or "F1" in ans)


@_needs_corpus
def test_xlsx_sheet_churn_stays_abstention(monkeypatch):
    # idx95: a schedule xlsx r1→r2 pair is whole-sheet realignment — the lane must still abstain (0 > -1).
    monkeypatch.setenv("RAG_DIFF_ALIGN", "1")
    assert diffpair.answer_question(_Q_IDX95) is None


@_needs_corpus
def test_ranked_candidates_are_prioritised(monkeypatch):
    monkeypatch.setenv("RAG_DIFF_ALIGN", "1")
    cands = diffpair.ranked_candidates(_Q_IDX1)
    assert cands
    scores = [c["score"] for c in cands]
    assert scores == sorted(scores, reverse=True)  # returned already ranked, best first
    assert {"intent", "score", "old", "new", "structural_location", "reason_features"} <= set(cands[0])
