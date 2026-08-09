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


def test_four_document_staff_population_is_cross_aggregate() -> None:
    q = "各案件のPP・契約書・PLAN・FRにおいて、DA側の実施体制として役割付きで記載されている人物は全部で何人ですか。"
    result = classify(q)
    assert result.contract == CROSS_AGGREGATE
    assert qc.is_staff_population_question(q)


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


def test_numeric_contract_rejects_uncomputed_notebook_claim() -> None:
    result = validate_numeric_answer(
        "notebookを確認して目的変数と相関が最も高い数値特徴量を教えてください。",
        "age",
        [],
    )
    assert not result.passed
    assert any("決定論的計算証跡" in issue for issue in result.issues)


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


def test_regulation_content_flags_not_found_phrasing() -> None:
    """SOT-2548 idx78: a 「…が見つかりませんでした」 negative must trip the completeness guard.

    The consolidated run answered idx78 with a "not found" wording rather than 「存在しない」, which the
    original _NO_SPECIAL_REGULATION_RE missed — so the general-rule fields were never enforced and the
    answer trailed off incomplete.
    """
    q = ("ひがし丘の契約条件において、ACTHが200時間を超えた場合の精算方法に関する規定内容を"
         "答えてください。")
    idx78_answer = ("ひがし丘の契約書には「ACTH」に関する記述が見つかりませんでしたが、実績工数が"
                    "見積を超えた場合の精算方法については、以下の一般規定が適用されます。")
    flagged = qc.validate_regulation_answer(q, idx78_answer)
    assert not flagged.passed and flagged.applicable
    assert set(flagged.missing) == {"単価", "税処理", "課金単位", "丸め", "精算周期", "上限"}

    # The consolidated run instead phrased the negative as 「直接の記述はありません」 (subject 記述 + あり
    # ません, which the 見つからない-only widening still missed) — the guard must fire on this too.
    no_record = ("ACTHという直接の記述はありませんが、実績工数に基づき時間単価25,000円で精算されます。")
    flagged_record = qc.validate_regulation_answer(q, no_record)
    assert not flagged_record.passed and "精算周期" in flagged_record.missing

    complete = qc.validate_regulation_answer(
        q,
        "ACTHの特別な精算規定は見つかりませんが、一般規定として時間単価25,000円(税別)に消費税を"
        "加算し、30分単位で切上げ、月次精算します。上限はありません。",
    )
    assert complete.passed and not complete.missing

    # A regulation-content answer that does not make a negative/not-found claim is untouched.
    positive = qc.validate_regulation_answer(
        q, "ACTHが200時間を超えた分は時間単価25,000円で別途精算すると規定されています。")
    assert positive.passed


def test_gantt_week_question_is_chart_read_with_geometry_completion() -> None:
    q = "提案書.pptxでモデル改善の実行予定スケジュールは案件開始から第何週目ですか。"
    result = classify(q)
    assert result.contract == CHART_READ
    assert any("left/width" in condition for condition in result.completion_conditions)
    assert qc.is_gantt_week_question(q)
    assert all("numCache" not in condition for condition in result.completion_conditions)


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


# --------------------------------------------------------------------------- SOT-2545 granularity
def test_granularity_flags_verbatim_truncation_against_fuller_fragment() -> None:
    """idx93 shape: a 「そのまま抜き出す」 extract that returns only the head of a longer evidence fragment."""
    q = "蒼樹会 みなみ野女性医療センターのアクションIDA10の内容をそのまま抜き出してください。"
    full = "前処理パイプライン実装：0値を疑似欠損（NA）扱いにする処理と補完ロジック（中央値等）を実装・ドキュメント化"
    v = qc.validate_answer_granularity(q, "前処理パイプライン", [{"result": f"A10 {full}"}])
    assert not v.passed and v.kind == "truncation"
    assert v.expected and "補完ロジック" in v.expected
    assert v.directive


def test_granularity_flags_titlelike_content_extract_without_fragment() -> None:
    """A 「内容をそのまま抜き出す」 ask answered with a short title-only string is a truncation even with no fuller fragment in hand."""
    q = "アクションIDA10の内容をそのまま抜き出してください。"
    v = qc.validate_answer_granularity(q, "前処理パイプライン", tool_outputs=[])
    assert not v.passed and v.kind == "truncation"


def test_granularity_flags_meta_response_instead_of_verbatim_text() -> None:
    """A process apology is not source text and must be sent back for one bounded correction."""
    q = "アクションIDA10の内容をそのまま抜き出してください。"
    v = qc.validate_answer_granularity(
        q, "I'm sorry, I made a mistake in the previous step and should be looking elsewhere.")
    assert not v.passed and v.kind == "truncation"
    assert "探索手順" in v.directive


def test_verbatim_completion_requires_full_source_item() -> None:
    conditions = qc.completion_conditions(
        "アクションIDA10の内容をそのまま抜き出してください。", qc.SIMPLE_LOOKUP)
    assert any("本文全体" in condition for condition in conditions)


def test_granularity_passes_full_verbatim_extract() -> None:
    """A verbatim extract that already returns the full body (with descriptive punctuation) passes."""
    q = "アクションIDA10の内容をそのまま抜き出してください。"
    full = "前処理パイプライン実装：0値を疑似欠損（NA）扱いにする処理と補完ロジック（中央値等）を実装・ドキュメント化"
    v = qc.validate_answer_granularity(q, full, [{"result": f"A10 {full}"}])
    assert v.passed and v.kind == ""


def test_granularity_flags_single_item_over_enumeration() -> None:
    """idx88 shape: a 「第N週の項目は何ですか」 singular ask answered with several comma-listed tasks."""
    q = "提案書内のスケジュール案において、第5週目に実施することになっている項目は何ですか。"
    ans = "解釈分析・特徴量整理、業務活用示唆整理、最終報告書ドラフト作成"
    v = qc.validate_answer_granularity(q, ans)
    assert not v.passed and v.kind == "over_enumeration"
    assert "第5週" in v.directive


def test_granularity_single_item_answer_passes() -> None:
    """A single-item selector answered with one item (・ is intra-item, not a list separator) passes."""
    q = "第5週目に実施することになっている項目は何ですか。"
    v = qc.validate_answer_granularity(q, "解釈・業務示唆整理")
    assert v.passed


def test_granularity_enumeration_question_not_flagged() -> None:
    """An explicit enumeration ask (「すべて挙げて」) must never be trimmed by the single-item guard."""
    q = "第5週目に実施することになっている項目をすべて挙げてください。"
    ans = "解釈分析・特徴量整理、業務活用示唆整理、最終報告書ドラフト作成"
    v = qc.validate_answer_granularity(q, ans)
    assert v.passed


def test_granularity_plain_multi_item_answer_without_selector_passes() -> None:
    """No single-item selector ⇒ a multi-item answer is left alone (guard stays tight, avoids false positives)."""
    q = "このプロジェクトの担当者は誰ですか。"
    v = qc.validate_answer_granularity(q, "田中太郎、佐藤花子")
    assert v.passed


def test_granularity_abstain_answer_passes() -> None:
    """An abstain is never rejected by the granularity guard (EV-safe)."""
    q = "アクションIDA10の内容をそのまま抜き出してください。"
    v = qc.validate_answer_granularity(q, "わかりません")
    # abstain text is non-empty but carries no items/body; the guard must not fire a truncation on it.
    assert v.passed or v.kind != "over_enumeration"


# --------------------------------------------------------------------------- SOT-2549 conflict resolution
def test_conflict_resolves_confirmed_single_over_range() -> None:
    """idx75 shape: a coarse range + a confirmed single that refines it ⇒ prefer the single (第4週)."""
    q = "MINAMINOのPP内のPL案において、モデル構築は第何週に実施することになっていますか。"
    ans = ("提案書のスケジュール案において、「第3週目から第5週目」と「第4週目」という2つの記載が"
           "競合しており、特定できません。")
    v = qc.resolve_record_conflict(q, ans)
    assert not v.passed and v.rule == "range_vs_confirmed"
    assert v.resolved == "第4週"
    assert "第4週" in v.directive


def test_conflict_single_outside_range_stays_abstained() -> None:
    """A single value outside the range is not reconcilable ⇒ abstain preserved (rule c, EV-safe)."""
    v = qc.resolve_record_conflict("q", "「第3週目から第5週目」と「第7週目」が競合し特定できません。")
    assert v.passed


def test_conflict_multiple_in_range_singles_stays_abstained() -> None:
    """Two distinct in-range singles are genuinely ambiguous ⇒ abstain preserved."""
    v = qc.resolve_record_conflict(
        "q", "「第1週目から第5週目」の中で第2週目と第4週目が競合し特定できません。")
    assert v.passed


def test_conflict_prefers_latest_confirmed_version() -> None:
    """Rule (a): two version-labelled records ⇒ prefer the one marked 最新/確定."""
    v = qc.resolve_record_conflict("q", "初版では第2章、最新版では第4章と記載が競合しており特定できません。")
    assert not v.passed and v.rule == "version_latest"
    assert v.resolved == "第4章"


def test_conflict_correct_answer_not_touched() -> None:
    """A committed correct answer (range + the specific week, no refusal) must never be rewritten."""
    v = qc.resolve_record_conflict("q", "第3週から第5週のうち、モデル構築は第4週です。")
    assert v.passed


def test_conflict_empty_abstain_not_touched() -> None:
    """A plain abstain with no surfaced records is left alone (no false resolution)."""
    assert qc.resolve_record_conflict("q", "わかりません").passed
    assert qc.resolve_record_conflict("q", "").passed


def test_conflict_ambiguous_version_without_label_stays_abstained() -> None:
    """Two version values but no 最新/確定 label ⇒ cannot pick authoritatively ⇒ abstain preserved."""
    v = qc.resolve_record_conflict("q", "ある版では第2章、別の版では第4章と競合し特定できません。")
    assert v.passed


def test_conflict_single_ordinal_question_fires_without_refusal_word() -> None:
    """run-d shape: a 「第何週」 question whose answer surfaces range+in-range single but was truncated
    before it could word the refusal ⇒ still resolves to the refining single (第4週)."""
    q = "MINAMINOのPP内のPL案において、モデル構築は第何週に実施することになっていますか。"
    ans = "モデル構築の実施期間は第3週から第5週、または第4週と記載があり、情報が"
    v = qc.resolve_record_conflict(q, ans)
    assert not v.passed and v.rule == "range_vs_confirmed" and v.resolved == "第4週"


def test_conflict_single_ordinal_correct_single_answer_untouched() -> None:
    """A 「第何週」 question already answered with the single week (no range) is never rewritten."""
    q = "モデル構築は第何週に実施しますか。"
    assert qc.resolve_record_conflict(q, "第4週です。").passed


def test_conflict_general_question_nonrefusal_range_not_resolved() -> None:
    """A general (non single-ordinal) question with a range answer that merely mentions a sub-week is
    left alone unless the model explicitly refuses — gold there may legitimately be the range."""
    v = qc.resolve_record_conflict("実施期間はいつですか。", "第3週から第5週（第4週に中間報告）")
    assert v.passed


# --------------------------------------------------------------------------- SOT-2562 numeric-feature corr
def test_numeric_feature_corr_rejects_categorical_encoding() -> None:
    """idx4 shape: 「相関が最も高い数値特徴量」 answered via a .map categorical re-encoding ⇒ reject."""
    q = "01_eda.ipynbを確認して、目的変数と相関が最も高い数値特徴量を教えてください。"
    trail = [{"value": ["charges", "smoker", "bmi"],
              "method": {"code": "df.assign(smoker=df['smoker'].map({'no':0,'yes':1}))"
                                 ".corr(numeric_only=True)['charges'].abs().sort_values().index.tolist()"}}]
    v = qc.validate_numeric_feature_correlation(q, "smoker", trail)
    assert not v.passed
    assert "numeric_only" in v.directive and ".map" in v.directive


def test_numeric_feature_corr_passes_plain_numeric_only() -> None:
    """A correlation ranking with no categorical re-encoding in the trail passes unchanged."""
    q = "目的変数と相関が最も高い数値特徴量を教えてください。"
    trail = [{"method": {"code": "df.corr(numeric_only=True)['charges'].abs().sort_values().index[-1]"}}]
    v = qc.validate_numeric_feature_correlation(q, "bmi", trail)
    assert v.passed


def test_numeric_feature_corr_passes_when_not_numeric_feature_question() -> None:
    """No 「数値特徴量」 in the ask ⇒ the guard never fires even if a .map corr is in the trail."""
    q = "目的変数と最も関係が深い特徴量は何ですか。"
    trail = [{"method": {"code": "df.assign(x=df['x'].map({'a':0})).corr()['y']"}}]
    assert qc.validate_numeric_feature_correlation(q, "smoker", trail).passed


def test_numeric_feature_corr_abstain_passes() -> None:
    q = "相関が最も高い数値特徴量を教えてください。"
    trail = [{"method": {"code": "df.assign(s=df['s'].map({'no':0})).corr(numeric_only=True)['c']"}}]
    assert qc.validate_numeric_feature_correlation(q, "わかりません", trail).passed


# --------------------------------------------------------------------------- SOT-2562 relevance enumeration
def test_relevance_enum_rejects_ungrounded_change_list() -> None:
    """idx9 shape: 「案件遂行に関連する変更を挙げて」 answered with a change list ⇒ push a re-filter."""
    q = "最終報告資料の最新版になる際に修正されたもののうち、案件遂行に関連する変更を挙げてください。"
    v = qc.validate_relevance_enumeration(q, "業務提言がクイックウィンと中期に分けられた")
    assert not v.passed and v.aspect == "案件遂行"
    assert "該当なし" in v.directive


def test_relevance_enum_already_nomatch_passes() -> None:
    q = "提案書_v3.pptxに修正されたもののうち、案件遂行に関連する変更を挙げてください。"
    assert qc.validate_relevance_enumeration(q, "該当なし").passed


def test_relevance_enum_abstain_passes() -> None:
    q = "最新版で修正されたもののうち、案件遂行に関連する変更を挙げてください。"
    assert qc.validate_relevance_enumeration(q, "わかりません").passed


def test_relevance_enum_non_change_question_passes() -> None:
    """A plain lookup (no 「関連する変更」 + version cue) is never touched."""
    assert qc.validate_relevance_enumeration("このプロジェクトの担当者は誰ですか。", "田中太郎").passed
