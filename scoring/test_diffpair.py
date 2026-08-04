"""Offline tests for the version-pair structural differ (no LLM / network; needs the corpus).

    .venv/bin/python -m pytest scoring/test_diffpair.py -q

These pin the behaviour behind valid idx9 (青嶺の提案書 old→最新: QAレビューア 池田 直哉 → 小林 直樹)
and the guards that keep the feature *additive-safe* (non-diff questions are never routed here;
unresolvable diffs abstain rather than emit garbage).
"""
from __future__ import annotations

import pytest

from src.rag import corpus, diffpair

# The tests below read the SIGNATE corpus; skip cleanly when it is absent (e.g. lean CI image).
_CORPUS_PRESENT = bool(corpus.walk())
_needs_corpus = pytest.mark.skipif(not _CORPUS_PRESENT, reason="corpus not present")

_Q_IDX9 = ("青嶺不動産アセットマネジメントの提案書について、oldフォルダ内の旧版と提案フォルダ直下の"
           "最新版を比較し、変更された箇所を変更前と変更後で答えてください。")


# ------------------------------- question detection (no corpus needed) ------------------------
def test_is_diff_question_true_for_version_diff():
    assert diffpair.is_diff_question(_Q_IDX9) is True
    assert diffpair.is_diff_question(
        "京橋の提案書の旧版と最新版を比較して変更点を教えてください。") is True


def test_is_diff_question_false_for_ordinary_questions():
    # ordinary retrieval questions must NOT be routed to the differ
    assert diffpair.is_diff_question("京橋信用ソリューションズの契約金額（税込）はいくらですか。") is False
    assert diffpair.is_diff_question("青嶺の modeling.py の model_type は何ですか。") is False
    assert diffpair.is_diff_question("train.csv の balance 列の平均値を答えてください。") is False


# ------------------------------- the idx9 target ----------------------------------------------
@_needs_corpus
def test_idx9_reports_qa_reviewer_change():
    ans = diffpair.answer_question(_Q_IDX9)
    assert ans is not None
    assert "池田 直哉" in ans and "小林 直樹" in ans and "→" in ans
    assert "QAレビューア" in ans


@_needs_corpus
def test_idx9_routes_through_generate_without_llm(monkeypatch):
    """generate.answer_question must answer idx9 from the structural diff, never calling the LLM."""
    from src.rag import generate

    def _boom(*a, **k):  # any LLM/retrieval use would mean the diff route didn't fire
        raise AssertionError("LLM/retrieval must not be reached for a resolved version-diff question")

    monkeypatch.setattr(generate.llm, "generate", _boom)
    monkeypatch.setattr(generate.retrieve, "get", _boom)
    res = generate.answer_question(_Q_IDX9)
    assert "池田 直哉" in res["answer"] and "小林 直樹" in res["answer"]
    assert res["confidence"] == "high"


# ------------------------------- cosmetic vs substantive --------------------------------------
def _diff_with(old_struct, new_struct):
    """Run structural_diff over two synthetic _Struct sides (distinct FileRef sentinels)."""
    import src.rag.diffpair as M

    s_old, s_new = object(), object()
    pair = M.VersionPair(old=s_old, new=s_new, base="x", basis="test")  # type: ignore[arg-type]
    orig = M._struct
    M._struct = lambda ref: old_struct if ref is s_old else new_struct
    try:
        return M.structural_diff(pair)
    finally:
        M._struct = orig


def test_cosmetic_only_change_is_dropped():
    # whitespace / full-width vs half-width only → not a substantive change
    old = diffpair._Struct(cells={"k": ("氏名", "山田 太郎")}, flow=["ＡＢＣ 123"])
    new = diffpair._Struct(cells={"k": ("氏名", "山田　太郎")}, flow=["ABC123"])
    changes = _diff_with(old, new)
    assert changes == [], f"cosmetic-only diff must be empty, got {[c.render() for c in changes]}"


def test_substantive_change_is_reported():
    old = diffpair._Struct(cells={"k": ("QAレビューア", "池田 直哉")}, flow=[])
    new = diffpair._Struct(cells={"k": ("QAレビューア", "小林 直樹")}, flow=[])
    changes = _diff_with(old, new)
    assert [c.render() for c in changes] == ["QAレビューア：池田 直哉 → 小林 直樹"]


# ------------------------------- explicitly-named version endpoints ---------------------------
@_needs_corpus
def test_explicit_endpoints_pick_the_named_versions():
    # v1→v2 and v1→v3 of the same proposal must diff the *named* endpoints, not a corpus-wide pair.
    a_12 = diffpair.answer_question(
        "青葉与信マネジメントの提案書_v1.pptxから提案書_v2.pptxに修正された変更を挙げてください。")
    a_13 = diffpair.answer_question(
        "青葉与信マネジメントの提案書_v1.pptxから提案書_v3.pptxに修正された変更を挙げてください。")
    assert a_12 and a_13 and a_12 != a_13


@_needs_corpus
def test_noisy_realigned_xlsx_diff_abstains():
    # スケジュール_r1→r2 realigns (row insertions) → too many changes → abstain rather than emit garbage
    assert diffpair.answer_question(
        "青嶺不動産アセットマネジメントのスケジュール_r1.xlsxとスケジュール_r2.xlsxを比較し変更点を挙げてください。"
    ) is None


# ------------------------------- abstention on unresolvable ------------------------------------
def test_unresolved_diff_question_returns_none():
    # a diff question naming no locatable company/document → no pair → None (caller abstains)
    assert diffpair.answer_question(
        "存在しない会社の幻の資料について旧版と最新版を比較し変更点を答えてください。") is None
