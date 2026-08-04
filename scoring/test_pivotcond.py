"""Offline tests for the PivotTable / AutoFilter condition reader (no LLM / network; needs corpus).

    .venv/bin/python -m pytest scoring/test_pivotcond.py -q

These pin the behaviour behind valid idx6 / idx11 / idx21 (the PivotTable and applied-AutoFilter
extraction-condition questions) and the additive-safe guards (ordinary questions are never routed
here; an unresolvable condition question abstains rather than emit garbage).
"""
from __future__ import annotations

import pytest

from src.rag import archetype, corpus, pivotcond

_CORPUS_PRESENT = bool(corpus.walk())
_needs_corpus = pytest.mark.skipif(not _CORPUS_PRESENT, reason="corpus not present")

_Q_IDX6 = ("恒一会 かえで総合病院のtrain.xlsx内の PivotTable で集計されている表から、"
           "ALPの平均が最も高いものの抽出条件と集計内容を答えてください。")
_Q_IDX11 = ("東都人材プラットフォームのtrain.xlsxにおいて、trainシートでフィルターで"
            "抽出されている条件を教えてください。")
_Q_IDX21 = ("青葉バイオメディカル機器のtrain.xlsxのPivotシートにおいて、"
            "平均月収が最も高い層の抽出条件を答えてください。")


# ------------------------------- question detection (no corpus needed) ------------------------
def test_is_pivot_condition_question_true():
    assert pivotcond.is_pivot_condition_question(_Q_IDX6) is True
    assert pivotcond.is_pivot_condition_question(_Q_IDX11) is True
    assert pivotcond.is_pivot_condition_question(_Q_IDX21) is True


def test_is_pivot_condition_question_false_for_ordinary():
    for q in (
        "京橋信用ソリューションズの契約金額（税込）はいくらですか。",
        "青葉の train.csv の balance 列の平均値を小数第2位まで答えてください。",
        "青嶺の modeling.py の model_type は何ですか。",
        "青嶺の提案書について、旧版と最新版を比較し変更点を答えてください。",
    ):
        assert pivotcond.is_pivot_condition_question(q) is False


def test_classify_routes_to_pivot_condition():
    assert archetype.classify(_Q_IDX6) == "pivot_condition"
    assert archetype.classify(_Q_IDX11) == "pivot_condition"
    assert archetype.classify(_Q_IDX21) == "pivot_condition"
    assert archetype.kind_of("pivot_condition") == "string"


# ------------------------------- the idx6 / idx11 / idx21 targets -----------------------------
@_needs_corpus
def test_idx6_pivot_condition_with_aggregation():
    ans = pivotcond.answer_question(_Q_IDX6)
    assert ans is not None
    for tok in ("Gender=Male", "disease=1", "Age=68"):
        assert tok in ans
    assert "平均" in ans and "ALP" in ans  # 集計内容 requested → measure name present


@_needs_corpus
def test_idx11_autofilter_condition():
    ans = pivotcond.answer_question(_Q_IDX11)
    assert ans == "Gender=Male、Country=India、target=2"


@_needs_corpus
def test_idx21_pivot_condition_conditions_only():
    ans = pivotcond.answer_question(_Q_IDX21)
    assert ans is not None
    for tok in ("Attrition=No", "Gender=Female", "MaritalStatus=Single",
                "EducationField=Human Resources"):
        assert tok in ans
    # no aggregation requested (抽出条件 only) → no "で抽出されたデータに対する" suffix
    assert "で抽出されたデータに対する" not in ans


# ------------------------------- routes through generate without LLM --------------------------
@_needs_corpus
def test_idx6_routes_through_generate_without_llm(monkeypatch):
    from src.rag import generate

    def _boom(*a, **k):
        raise AssertionError("LLM/retrieval must not be reached for a resolved pivot-condition question")

    monkeypatch.setattr(generate.llm, "generate", _boom)
    monkeypatch.setattr(generate.retrieve, "get", _boom)
    res = generate.answer_question(_Q_IDX6)
    assert "Gender=Male" in res["answer"] and "Age=68" in res["answer"]
    assert res["confidence"] == "high"


# ------------------------------- abstention on unresolvable -----------------------------------
def test_unresolved_condition_question_returns_none():
    # a condition question naming no locatable company/workbook → abstain
    assert pivotcond.answer_question(
        "存在しない会社のピボットテーブルで最も高い層の抽出条件を答えてください。") is None
