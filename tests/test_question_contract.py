"""SOT-2493 — offline tests for the question-contract classifier.

Everything here runs network-free:

* the deterministic layer classifies representative questions of all nine contracts;
* the hybrid ``flash`` arbiter is exercised with an injected fake (and asserted NOT to be called on a
  confident deterministic hit);
* the production :func:`flash_classify` wrapper is driven with ``llm.generate`` monkeypatched;
* the acceptance metric — agreement with the gold-100 ``archetype`` column — is measured deterministically
  and asserted ``≥0.90`` (受け入れ条件①), with every mismatch surfaced for review.
"""
from __future__ import annotations

import csv
from pathlib import Path

import pytest

from config import settings
from src.rag.agent import question_contract as qc
from src.rag.agent.question_contract import (
    CHART_READ,
    CONTRACT_ARCHETYPES,
    CONTRACT_COMPLETION,
    CONTRACT_LABELS,
    CONTRACT_ROUTE,
    CONTRACTS,
    CROSS_AGGREGATE,
    FORMAT_CHECK,
    FULL_ENUMERATION,
    MULTI_HOP,
    NUMERIC,
    SIMPLE_LOOKUP,
    SPATIAL,
    VERSION_DIFF,
    QuestionContract,
    agreement_rate,
    classify,
    flash_classify,
    numeric_requirements,
    validate_numeric_answer,
)

GOLD_CSV = settings.ARTIFACTS_DIR / "gold_100_review.csv"


# --------------------------------------------------------------------------- taxonomy invariants
def test_every_contract_has_completion_route_and_archetypes() -> None:
    """Each of the nine contracts declares a label, non-empty completion template, route, archetype set."""
    assert len(CONTRACTS) == 9 and len(set(CONTRACTS)) == 9
    for c in CONTRACTS:
        assert CONTRACT_LABELS[c]
        assert CONTRACT_COMPLETION[c] and all(isinstance(s, str) and s for s in CONTRACT_COMPLETION[c])
        assert CONTRACT_ROUTE[c]
        assert isinstance(CONTRACT_ARCHETYPES[c], frozenset) and CONTRACT_ARCHETYPES[c]


# --------------------------------------------------------------------------- deterministic classification
# One representative question per contract; each must be resolved deterministically (no flash).
_REPRESENTATIVE: list[tuple[str, str]] = [
    (VERSION_DIFF, "白峰信用リスク評価の提案書old版と最新版を比較して、変更点を挙げてください。"),
    (FULL_ENUMERATION, "契約書に登場する担当者名をすべて挙げてください。"),
    (CROSS_AGGREGATE, "全プロジェクトの契約総額を計算してください。"),
    (CHART_READ, "基礎分析.docxのグラフ1で、x=3のときのyの値を答えてください。"),
    (SPATIAL, "井上さんの向かいに座っている方のEXTを教えてください。"),
    (FORMAT_CHECK, "契約書において、太字で記載されている箇所をすべて抽出してください。"),
    (MULTI_HOP, "もっとも多くの案件にかかわっている人の内線番号を教えてください。"),
    (NUMERIC, "着手金と検収金の差額はいくらですか。"),
    (SIMPLE_LOOKUP, "この契約書における甲の正式名称は何ですか。"),
]


@pytest.mark.parametrize("expected,question", _REPRESENTATIVE)
def test_representative_questions_classified_deterministically(expected: str, question: str) -> None:
    res = classify(question)
    assert res.contract == expected, f"{question!r} -> {res.contract} (want {expected})"
    assert res.method == "deterministic"
    assert res.completion_conditions == CONTRACT_COMPLETION[expected]
    assert res.route == CONTRACT_ROUTE[expected]
    assert 0.0 < res.confidence <= 1.0


def test_result_is_immutable_and_serialisable() -> None:
    res = classify("着手金と検収金の差額はいくらですか。")
    with pytest.raises(Exception):
        res.contract = "x"  # type: ignore[misc]  # frozen dataclass
    d = res.to_dict()
    assert d["contract"] == NUMERIC and d["label"] == CONTRACT_LABELS[NUMERIC]
    assert d["completion_conditions"] == list(CONTRACT_COMPLETION[NUMERIC])


def test_classification_is_deterministic() -> None:
    q = "全プロジェクトの契約総額を計算してください。"
    assert classify(q).contract == classify(q).contract == CROSS_AGGREGATE


def test_idx30_quantity_contract_pins_denominator_unit_and_rounding() -> None:
    q = ("青葉与信マネジメントの分析対象データにおいて、標準化されたloan_amntが0未満の行のうち、"
         "purpose=credit_cardに該当し、かつloan_amntがpurpose=credit_card全体の平均を上回る行の"
         "割合は何%ですか。小数第2位まで答えてください。")
    req = numeric_requirements(q)
    assert req.ratio is True
    assert req.denominator_scope == "標準化されたloan_amntが0未満の行"
    assert req.denominator_fields == ("loan_amnt",)
    assert req.denominator_operators == ("lt",)
    assert req.unit == "%" and req.decimal_places == 2

    wrong = [
        {"code": "len(df[df['purpose'] == 'credit_card'])", "output": 3053},
        {"code": "round(129 / 3053 * 100, 2)", "output": 4.23},
    ]
    rejected = validate_numeric_answer(q, "4.23%", wrong)
    assert not rejected.passed
    assert any("対象列" in issue for issue in rejected.issues)
    assert any("比較" in issue for issue in rejected.issues)

    correct = [
        {"code": "len(df[((df['loan_amnt'] - 1582.99) / 830.19 < 0)])", "output": 10938},
        {"code": "round(129 / 10938 * 100, 2)", "output": 1.18},
    ]
    assert validate_numeric_answer(q, "1.18%", correct).passed


def test_pivot_highlight_question_is_format_contract() -> None:
    q = "基礎分析.pptxで黄色ハイライトされている数値の抽出条件と集計内容を答えてください。"
    assert classify(q).contract == FORMAT_CHECK


def test_regulation_content_contract_requires_fallback_general_rule() -> None:
    q = "契約条件において、稼働が上限を超えた場合の精算方法に関する規定内容を答えてください。"
    result = classify(q)
    assert result.contract == SIMPLE_LOOKUP
    assert any("一般規定" in condition and "単価" in condition and "上限" in condition
               for condition in result.completion_conditions)

    incomplete = qc.validate_regulation_answer(q, "超過時の特別な精算規定は存在しません。")
    assert not incomplete.passed and incomplete.applicable
    assert set(incomplete.missing) == {"単価", "税処理", "課金単位", "丸め", "精算周期", "上限"}

    cycle3_wording = qc.validate_regulation_answer(
        q, "200時間を超えた場合でも特別な精算方法は規定されていません。時間単価は30,000円です。")
    assert not cycle3_wording.passed and cycle3_wording.applicable
    assert "税処理" in cycle3_wording.missing and "上限" in cycle3_wording.missing

    complete = qc.validate_regulation_answer(
        q,
        "特別な精算規定は存在しません。一般規定として時間単価30,000円に消費税を加算し、15分単位で切上げ、"
        "月次精算し、上限はありません。",
    )
    assert complete.passed and complete.applicable and not complete.missing


def test_gantt_week_question_is_chart_read_with_geometry_completion() -> None:
    q = "提案書.pptxでモデル改善の実行予定スケジュールは案件開始から第何週目ですか。"
    result = classify(q)
    assert result.contract == CHART_READ
    assert any("left/width" in condition for condition in result.completion_conditions)
    assert qc.is_gantt_week_question(q)


# --------------------------------------------------------------------------- hybrid flash arbitration
def _ambiguous_question() -> str:
    """A bare fact-lookup with no structural cue → the flash-arbitration path."""
    res = classify("この契約書における甲の正式名称は何ですか。")
    # glossary/formal cue makes the above concrete; find a truly bare one from gold instead.
    rows = _load_gold()
    for r in rows:
        cand = classify(r["question"])
        if cand.confidence == qc._CONF_WEAK and cand.method == "deterministic":
            return r["question"]
    pytest.skip("no ambiguous gold question found")


def test_flash_consulted_only_when_ambiguous() -> None:
    calls: list[str] = []

    def arbiter(q: str) -> str:
        calls.append(q)
        return MULTI_HOP

    # Confident deterministic hit → flash must NOT be called.
    res = classify("全プロジェクトの契約総額を計算してください。", flash=arbiter)
    assert res.method == "deterministic" and not calls

    # Ambiguous question → flash IS called and its verdict is used.
    res2 = classify(_ambiguous_question(), flash=arbiter)
    assert calls, "flash arbiter was not consulted on an ambiguous question"
    assert res2.method == "flash" and res2.contract == MULTI_HOP


def test_ambiguous_without_flash_falls_back_to_simple_lookup() -> None:
    res = classify(_ambiguous_question())  # no flash injected
    assert res.contract == SIMPLE_LOOKUP and res.method == "deterministic"


def test_flash_bad_verdict_ignored() -> None:
    res = classify(_ambiguous_question(), flash=lambda q: "not_a_contract")
    assert res.contract == SIMPLE_LOOKUP  # invalid code ignored → deterministic default


# --------------------------------------------------------------------------- production flash wrapper
def test_flash_classify_parses_and_validates() -> None:
    # generate is injected so this never touches the live Gemini client / SDK.
    assert flash_classify("q", generate=lambda *a, **k: '{"contract": "numeric"}') == NUMERIC
    assert flash_classify("q", generate=lambda *a, **k: "version_diff") == VERSION_DIFF  # bare code
    assert flash_classify("q", generate=lambda *a, **k: '{"contract": "bogus"}') is None  # OOV → None

    def boom(*a, **k):
        raise RuntimeError("network down")

    assert flash_classify("q", generate=boom) is None  # unreachable model → None


# --------------------------------------------------------------------------- acceptance metric
def _load_gold() -> list[dict[str, str]]:
    if not GOLD_CSV.exists():
        pytest.skip(f"gold csv not present: {GOLD_CSV}")
    with GOLD_CSV.open(encoding="utf-8-sig") as fh:
        return list(csv.DictReader(fh))


def test_agreement_with_gold_archetypes_meets_target() -> None:
    """受け入れ条件①: 分類一致率 ≥90% against the gold-100 archetype column; mismatches are recorded."""
    rows = _load_gold()
    report = agreement_rate(rows)  # flash=None → fully deterministic, network-free
    assert report.total >= 30, "expected a populated gold set"
    assert report.rate >= 0.90, (
        f"agreement {report.rate:.1%} < 90%; mismatches="
        + "; ".join(f"{m['contract']}!={m['gold_archetype']} ({m['question'][:30]})"
                    for m in report.mismatches)
    )
    # Every recorded mismatch must be a real (question, gold, contract) triple for human review.
    for m in report.mismatches:
        assert m["question"] and m["gold_archetype"] and m["contract"]


def test_refinements_are_consistent_specialisations() -> None:
    """The specificity gains (spatial/chart/format/multi_hop) must each be consistent with their gold archetype."""
    rows = _load_gold()
    report = agreement_rate(rows)
    assert report.refinements, "expected at least one contract to refine a coarser archetype"
    for r in report.refinements:
        contract, gold = r["contract"], r["gold_archetype"]
        assert gold in CONTRACT_ARCHETYPES[contract] or gold == "unknown"
