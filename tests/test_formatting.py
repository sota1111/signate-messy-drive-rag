"""SOT-2604 (Stage3) — offline tests for the deterministic-value → gold-format naturalization layer.

All network-free: the template path is pure, and the one LLM naturalize call is exercised only through an
injected stub. The invariants under test are the Stage3 contract: template-first (no LLM for
数値/列挙/週/「該当なし」), value facts never invented (SOT-2544 記号↔文章形の同義 / SOT-2545 粒度トリム・truncation
補完 are evidence-bound), additive (a valid non-blank contract in ⇒ a valid contract out), and gated OFF
with the router (identity no-op unless RAG_DET_PIPELINE_ROUTER / force).
"""
from __future__ import annotations

import pytest

from src.rag.agent import det_pipeline as dp
from src.rag.agent import formatting as fmt
from src.rag.agent import investigator as inv
from src.rag.agent import question_contract as qc
from src.rag.agent.investigator import ABSTAIN, SUBMIT_ANSWER, Call, Step, Usage
from src.rag.tools import contract as _contract


def _ct(value, **evidence):
    """Build a valid {value, evidence, method} contract for the layer under test."""
    return _contract.make(value, engine="det", evidence=evidence or {"file": "x"})


def _ct_nat(value, **evidence):
    """Contract that explicitly opts into the one short LLM naturalize call (method.naturalize)."""
    return _contract.make(value, engine="det", evidence=evidence or {"file": "x"}, naturalize=True)


# --------------------------------------------------------------------------- flag / no-op
def test_enabled_shares_router_flag(monkeypatch):
    monkeypatch.delenv("RAG_DET_PIPELINE_ROUTER", raising=False)
    assert fmt.enabled() is False
    monkeypatch.setenv("RAG_DET_PIPELINE_ROUTER", "1")
    assert fmt.enabled() is True


def test_disabled_is_identity_noop(monkeypatch):
    monkeypatch.delenv("RAG_DET_PIPELINE_ROUTER", raising=False)
    c = _ct("該当する項目はありません")  # a none-form that WOULD canonicalize when enabled
    out = fmt.format_contract(c, "何ですか")
    assert out is not None
    assert out["value"] == "該当する項目はありません"  # untouched when gated off


def test_malformed_contract_returns_none():
    assert fmt.format_contract({"not": "a contract"}, "q", force=True) is None


def test_blank_value_returns_none_for_fallback():
    # 決定論値が空 ⇒ 整形せず上位(棄権/LLMループ)へ返す
    assert fmt.format_contract(_ct(""), "q", force=True) is None
    assert fmt.format_contract(_ct([]), "q", force=True) is None
    assert fmt.format_contract(_ct(None), "q", force=True) is None


# --------------------------------------------------------------------------- template-first (no LLM)
def test_string_value_passthrough_verbatim():
    # a structured-notation string is preserved byte-for-byte (値改変ゼロ) and never sent to an LLM.
    calls = []
    out = fmt.format_contract(_ct("n_estimators（1位=500、2位=300）"), "スコア差の設定差分は",
                              contract_type=qc.NUMERIC, naturalizer=lambda *a: calls.append(a) or "X",
                              force=True)
    assert out["value"] == "n_estimators（1位=500、2位=300）"
    assert calls == []                                   # template-first: LLM untouched
    assert out["method"]["formatting"]["template_only"] is True


def test_number_value_renders_without_trailing_zero():
    assert fmt.format_contract(_ct(500.0), "平均は", contract_type=qc.NUMERIC,
                               force=True)["value"] == "500"
    assert fmt.format_contract(_ct(12.3), "平均は", contract_type=qc.NUMERIC,
                               force=True)["value"] == "12.3"


def test_list_value_renders_gold_enumeration():
    out = fmt.format_contract(_ct(["A", "B", "C"]), "すべて挙げて",
                              contract_type=qc.FULL_ENUMERATION, force=True)
    assert out["value"] == "A、B、C"


def test_single_item_list_renders_bare():
    out = fmt.format_contract(_ct(["解釈・業務示唆整理"]), "第5週の項目は",
                              contract_type=qc.SIMPLE_LOOKUP, force=True)
    assert out["value"] == "解釈・業務示唆整理"


# --------------------------------------------------------------------------- SOT-2544 none-form synonym
@pytest.mark.parametrize("raw", ["存在しません", "ありません", "なし", "該当する項目はありません", "N/A"])
def test_none_forms_canonicalize(raw):
    out = fmt.format_contract(_ct(raw), "未達成の項目を挙げて", contract_type=qc.FULL_ENUMERATION,
                              force=True)
    assert out["value"] == "該当なし"
    assert "none_canonical" in out["method"]["formatting"]["rules"] or raw == "該当なし"


def test_substring_none_is_not_collapsed():
    # 「なし」 as a substring of a real answer must NOT be collapsed to 該当なし.
    out = fmt.format_contract(_ct("課題なし体制の構築"), "体制は", contract_type=qc.SIMPLE_LOOKUP,
                              force=True)
    assert out["value"] == "課題なし体制の構築"


# --------------------------------------------------------------------------- SOT-2545 granularity repair
def test_over_enumeration_trim_is_evidence_driven():
    # single-item ask + evidence-designated item ⇒ trimmed to that one item (no guessing).
    out = fmt.format_contract(
        _ct(["データ収集・前処理", "解釈・業務示唆整理", "報告書作成"],
            selected="解釈・業務示唆整理", week=5),
        "第5週目に実施する項目は何ですか", contract_type=qc.SIMPLE_LOOKUP, force=True)
    assert out["value"] == "解釈・業務示唆整理"
    assert "over_enumeration_trimmed" in out["method"]["formatting"]["rules"]


def test_over_enumeration_not_trimmed_without_selector():
    # a genuine enumeration ask keeps the full list (never trims).
    out = fmt.format_contract(_ct(["A", "B", "C"], selected="B"),
                              "すべて挙げてください", contract_type=qc.FULL_ENUMERATION, force=True)
    assert out["value"] == "A、B、C"


def test_over_enumeration_not_trimmed_without_designation():
    # single-selector but no evidence-designated item ⇒ leave the list (don't fabricate a selection).
    out = fmt.format_contract(_ct(["A", "B", "C"]), "第5週目の項目は",
                              contract_type=qc.SIMPLE_LOOKUP, force=True)
    assert out["value"] == "A、B、C"


def test_truncation_completed_from_fulltext_evidence():
    full = "前処理パイプライン実装：0値を疑似欠損（NA）扱いにする処理と補完ロジック（中央値等）を実装・ドキュメント化"
    out = fmt.format_contract(
        _ct("前処理パイプライン", full_text=full),
        "アクションIDA10の内容をそのまま抜き出してください", contract_type=qc.SIMPLE_LOOKUP, force=True)
    assert out["value"] == full
    assert "truncation_completed" in out["method"]["formatting"]["rules"]


def test_truncation_not_completed_without_fuller_evidence():
    # no fuller fragment in evidence ⇒ leave the value (never invent the body).
    out = fmt.format_contract(_ct("前処理パイプライン"), "内容をそのまま抜き出して",
                              contract_type=qc.SIMPLE_LOOKUP, force=True)
    assert out["value"] == "前処理パイプライン"


# --------------------------------------------------------------------------- one short LLM naturalize
def test_llm_naturalize_only_for_freetext_raw_structure():
    seen = {}

    def stub(value_text, question):
        seen["value_text"] = value_text
        seen["question"] = question
        return "本レポートの主眼は解釈と業務示唆の整理です。"

    out = fmt.format_contract(
        _ct_nat(["解釈", "業務示唆", "整理"]), "所見をまとめると",
        contract_type=qc.CROSS_AGGREGATE, naturalizer=stub, force=True)
    # the naturalizer received the DETERMINISTIC value text (facts to preserve), not an empty string.
    assert seen["value_text"] == "解釈、業務示唆、整理"
    assert out["value"] == "本レポートの主眼は解釈と業務示唆の整理です。"
    assert "llm_naturalized" in out["method"]["formatting"]["rules"]
    assert out["method"]["formatting"]["template_only"] is False


def test_llm_naturalize_skipped_for_clean_string_freetext():
    calls = []
    out = fmt.format_contract(_ct("既に整った回答文です。"), "説明して",
                              contract_type=qc.SIMPLE_LOOKUP,
                              naturalizer=lambda *a: calls.append(a) or "改変", force=True)
    assert out["value"] == "既に整った回答文です。"  # clean string ⇒ template-first, no LLM
    assert calls == []


def test_llm_failure_degrades_to_template_text():
    def boom(value_text, question):
        raise RuntimeError("naturalizer down")

    out = fmt.format_contract(_ct_nat({"設定": "n_estimators"}), "所見は",
                              contract_type=qc.CROSS_AGGREGATE, naturalizer=boom, force=True)
    # LLM failed ⇒ keep the deterministic template text; answer is never dropped.
    assert out["value"] == "設定=n_estimators"


def test_method_confidence_is_preserved():
    c = _contract.make("500", engine="pandas", evidence={"file": "t.xlsx"}, confidence=0.95)
    out = fmt.format_contract(c, "平均は", contract_type=qc.NUMERIC, force=True)
    assert out["method"]["confidence"] == pytest.approx(0.95)
    assert out["method"]["engine"] == "pandas"


# --------------------------------------------------------------------------- focused A1/A2 (idx62/85/88/93)
# Each focus case feeds a plausible deterministic contract for that question and asserts the layer yields
# the gold answer with the facts preserved. gold100 is not run here (Wave 末 で一括).
def test_idx62_score_diff_structured_notation_preserved():
    gold = "n_estimators（1位=500、2位=300）"
    out = fmt.format_contract(_ct(gold, file="report.pptx"),
                              "モデル比較で上位2件のスコア差を生んでいる設定差分は何ですか。",
                              contract_type=qc.NUMERIC, force=True)
    assert out["value"] == gold            # 値改変ゼロ


def test_idx85_none_answer_survives_as_real_answer():
    # 「該当なし」 is a REAL answer under the rubric — it must pass through, never be dropped/abstained.
    out = fmt.format_contract(_ct("該当なし", certified_absent=True),
                              "設定されたKPIとして未達成とされている項目を挙げてください。",
                              contract_type=qc.FULL_ENUMERATION, force=True)
    assert out is not None
    assert out["value"] == "該当なし"


def test_idx88_week5_single_item_trim():
    out = fmt.format_contract(
        _ct(["データ収集・前処理", "解釈・業務示唆整理", "報告書作成"], selected="解釈・業務示唆整理"),
        "スケジュール案において、第5週目に実施することになっている項目は何ですか。",
        contract_type=qc.SIMPLE_LOOKUP, force=True)
    assert out["value"] == "解釈・業務示唆整理"


def test_idx93_verbatim_truncation_completion():
    gold = "前処理パイプライン実装：0値を疑似欠損（NA）扱いにする処理と補完ロジック（中央値等）を実装・ドキュメント化"
    out = fmt.format_contract(
        _ct("前処理パイプライン", full_text=gold),
        "アクションIDA10の内容をそのまま抜き出してください。",
        contract_type=qc.SIMPLE_LOOKUP, force=True)
    assert out["value"] == gold


# --------------------------------------------------------------------------- SOT-2617 derived 書式契約
# unit/rounding/verbosity contracts for the derived residual (idx6/64/65 class). Question-cue-driven,
# value-preserving, gated behind RAG_DERIVED_FORMAT_CONTRACTS (default OFF ⇒ byte-identical).
@pytest.fixture
def _derived_on(monkeypatch):
    monkeypatch.setenv("RAG_DERIVED_FORMAT_CONTRACTS", "1")


def _fc(value, question, **kw):
    return fmt.format_contract(_ct(value), question, force=True, **kw)


def test_derived_contracts_flag_default_off(monkeypatch):
    monkeypatch.delenv("RAG_DERIVED_FORMAT_CONTRACTS", raising=False)
    assert fmt.derived_contracts_enabled() is False
    monkeypatch.setenv("RAG_DERIVED_FORMAT_CONTRACTS", "1")
    assert fmt.derived_contracts_enabled() is True


def test_idx6_unit_currency_counter_to_yen(_derived_on):
    # 差額はいくら (currency ask) answered with a generic counter 「0件」 ⇒ 「0円」; the number 0 is unchanged.
    out = _fc("0件", "税込み見込み金額と最終請求金額の差額はいくらですか。", contract_type=qc.NUMERIC)
    assert out["value"] == "0円"
    assert "unit_currency" in out["method"]["formatting"]["rules"]


def test_idx6_unit_off_is_byte_identical(monkeypatch):
    monkeypatch.delenv("RAG_DERIVED_FORMAT_CONTRACTS", raising=False)
    out = _fc("0件", "差額はいくらですか。", contract_type=qc.NUMERIC)
    assert out["value"] == "0件"  # flag off ⇒ untouched
    assert out["method"]["formatting"]["rules"] == []


def test_unit_currency_not_applied_to_count_question(_derived_on):
    # an explicit count ask (いくつ) keeps 件 — currency contract must not fire.
    out = _fc("7件", "スコープ対象外としている項目はいくつありますか。", contract_type=qc.NUMERIC)
    assert out["value"] == "7件"


def test_unit_currency_leaves_real_unit_intact(_derived_on):
    # a genuine domain unit (時間) is never rewritten to 円 even under a currency-ish ask.
    out = _fc("5時間", "作業費用は何時間ぶんですか。", contract_type=qc.NUMERIC)
    assert out["value"] == "5時間"


def test_idx64_verbosity_summary_quantity_extracted(_derived_on):
    verbose = ("フェーズA（20〜30時間想定）とフェーズB（60〜100時間想定）をあわせた合計想定工数は、"
               "80〜130時間（最小80時間、最大130時間）です。")
    out = _fc(verbose, "フェーズAとフェーズBを実施した場合の想定工数は合計で何時間ですか。",
              contract_type=qc.NUMERIC)
    assert out["value"] == "80〜130時間"
    assert "verbosity_summary" in out["method"]["formatting"]["rules"]


def test_verbosity_not_trimmed_when_ambiguous(_derived_on):
    # multiple distinct spans and no 合計/あわせ marker ⇒ no guessing which one is the answer.
    v = "AとBは10時間、Cは20時間かかります。"
    out = _fc(v, "各項目は何時間ですか。", contract_type=qc.NUMERIC)
    assert out["value"] == v


def test_idx65_trailing_count_note_stripped(_derived_on):
    # a condition ask trailing a redundant 「（該当件数: N件）」 tally ⇒ drop the parenthetical (冗長除去).
    out = _fc("セルの値 < -0.9（該当件数: 14件）", "黄色ハイライトになっているセルの条件を答えてください。",
              contract_type=qc.NUMERIC)
    assert out["value"] == "セルの値 < -0.9"
    assert "verbosity_count_note" in out["method"]["formatting"]["rules"]


def test_count_question_summary_extracts_the_count(_derived_on):
    # for a 「何件」 ask the count IS the answer — the summary rule reduces prose to that count (not the
    # condition-strip rule, which is gated away from count asks).
    out = _fc("該当セル（14件）", "条件に一致するセルは何件ありますか。", contract_type=qc.NUMERIC)
    assert out["value"] == "14件"
    assert "verbosity_count_note" not in out["method"]["formatting"]["rules"]


def test_rounding_contract_honors_decimal_places(_derived_on):
    out = _fc("12.34567", "平均値を小数第2位まで四捨五入して答えてください。", contract_type=qc.NUMERIC)
    assert out["value"] == "12.35"
    assert "rounding" in out["method"]["formatting"]["rules"]


def test_rounding_contract_to_integer(_derived_on):
    out = _fc("3.6", "件数を整数で答えて。", contract_type=qc.NUMERIC)
    assert out["value"] == "4"


def test_rounding_noop_without_directive(_derived_on):
    out = _fc("12.34567", "平均値はいくつですか。", contract_type=qc.NUMERIC)
    assert out["value"] == "12.34567"  # no precision directive ⇒ value untouched


def test_derived_contracts_do_not_hardcode_gold(_derived_on):
    # the contract is class-general: a *different* currency answer flows through the same 件→円 rule.
    out = _fc("1250件", "請求金額はいくらですか。", contract_type=qc.NUMERIC)
    assert out["value"] == "1250円"


# --------------------------------------------------------------------------- investigator wiring
@pytest.fixture(autouse=True)
def _restore_registry():
    saved = dict(dp._REGISTRY)
    try:
        yield
    finally:
        dp._REGISTRY.clear()
        dp._REGISTRY.update(saved)


class _ScriptedModel:
    def __init__(self, steps):
        self._steps = list(steps)
        self._i = 0
        self.model_name = "fake-model"

    def next(self, _tool_responses):
        if self._i >= len(self._steps):
            return Step(function_calls=(), final_text=ABSTAIN, usage=Usage(1, 1))
        step = self._steps[self._i]
        self._i += 1
        return step


def _numeric_question():
    return "京橋信用ソリューションズの train.xlsx の loan_amnt の平均は。"


def test_wiring_formats_det_answer_none_form(monkeypatch):
    # flag ON + a grounded pipeline returning a none-form ⇒ committed answer is the canonical 「該当なし」.
    monkeypatch.setenv("RAG_DET_PIPELINE_ROUTER", "1")
    dp.register("numeric",
                lambda q, *, profile=None: _contract.make("存在しません", engine="pandas",
                                                          evidence={"file": "train.xlsx"}),
                replace=True)  # override the real Wave A2 pipeline for this wiring test

    def fake_factory(question, tools, *, model=None, system=None):
        raise AssertionError("LLM loop must not be entered when the router grounds an answer")

    monkeypatch.setattr(inv, "gemini_model_factory", fake_factory)
    res = inv.answer_question(_numeric_question(), ledger=False, calc_ledger=False,
                              research=False, enumeration=False)
    assert res.model == "deterministic"
    assert res.answer.answer == "該当なし"
    assert res.tool_calls == ["det_pipeline:numeric"]


def test_wiring_blank_det_value_falls_back_to_loop(monkeypatch):
    # a pipeline that grounds a value which formatting empties ⇒ fall back to the LLM loop (回答数維持).
    # (whitespace value passes resolve's `is None` guard but is blank to the formatter.)
    monkeypatch.setenv("RAG_DET_PIPELINE_ROUTER", "1")
    dp.register("numeric",
                lambda q, *, profile=None: _contract.make("   ", engine="pandas",
                                                          evidence={"file": "train.xlsx"}),
                replace=True)  # override the real Wave A2 pipeline for this wiring test

    def fake_factory(question, tools, *, model=None, system=None):
        return _ScriptedModel([Step(
            function_calls=(Call(SUBMIT_ANSWER, {"answer": "フォールバック値", "confidence": 0.8}),),
            usage=Usage(10, 5))])

    monkeypatch.setattr(inv, "gemini_model_factory", fake_factory)
    res = inv.answer_question(_numeric_question(), ledger=False, calc_ledger=False,
                              research=False, enumeration=False)
    assert res.model != "deterministic"
    assert res.answer.answer == "フォールバック値"
    assert res.tool_calls != ["det_pipeline:numeric"]


# --------------------------------------------------------------------------- SOT-2682 小数指定問の単位strip
_DEC_Q = ("恒一会 かえで総合病院の計画フォルダ内において、データアステル側の担当者のうち、1タスク当たりの想定"
          "工数（想定工数 ÷ 担当タスク数）が最も大きい人のフルネームと、その1タスク当たりの想定工数を小数第2位"
          "で答えてください。")


def test_decimal_unit_strip_idx79_case():
    # idx79: name + decimal + composite rate unit ⇒ unit dropped, value (name + number) preserved.
    out, fired = fmt.strip_decimal_spec_unit(_DEC_Q, "池田 直哉、7.00時間/タスク")
    assert out == "池田 直哉、7.00"
    assert fired == ["decimal_spec_unit_strip"]


def test_decimal_unit_strip_percent_and_bai():
    # a 小数第N位 rate/ratio ask answered with %/倍 ⇒ the trailing unit is dropped (gold is bare).
    assert fmt.strip_decimal_spec_unit("上昇率を小数第2位まで答えてください", "2.21%")[0] == "2.21"
    assert fmt.strip_decimal_spec_unit("何倍か小数第2位まで求めてください", "2.49倍")[0] == "2.49"


def test_decimal_unit_strip_noop_when_already_bare():
    # gold-shaped bare decimals are a no-op (no unit to strip).
    for v in ("1.18", "6.088138 ~ 6.288138", "0.42396", "池田 直哉、7.00"):
        out, fired = fmt.strip_decimal_spec_unit(_DEC_Q, v)
        assert out == v and fired == []


def test_decimal_unit_strip_requires_decimal_spec_question():
    # no 小数第N位 directive ⇒ never fires, even if the answer carries a number+unit.
    out, fired = fmt.strip_decimal_spec_unit("想定工数を答えてください", "7.00時間/タスク")
    assert out == "7.00時間/タスク" and fired == []


def test_decimal_unit_strip_integer_answer_untouched():
    # integer answers (no decimal point) are out of scope — idx27「5」 must not be touched.
    out, fired = fmt.strip_decimal_spec_unit("項目はいくつありますか。小数第2位で答えてください", "5")
    assert out == "5" and fired == []


def test_decimal_unit_strip_preserves_numeric_tokens():
    # the numeric token is never altered — only the trailing unit is removed.
    out, _ = fmt.strip_decimal_spec_unit(_DEC_Q, "3.14ラジアン")
    assert out == "3.14"


def test_decimal_unit_strip_skips_multiline_and_paren_prose():
    # multi-line prose is skipped; a trailing parenthetical is left to the paren-strip contract.
    assert fmt.strip_decimal_spec_unit(_DEC_Q, "7.00時間\n(根拠…)")[1] == []
    # a trailing balanced paren (not a bare unit token) is not a unit ⇒ no strip here.
    assert fmt.strip_decimal_spec_unit(_DEC_Q, "7.00（時間/タスク）")[1] == []


def test_decimal_unit_strip_flag_default_off():
    import os
    prev = os.environ.pop("RAG_DECIMAL_UNIT_STRIP", None)
    try:
        assert fmt.decimal_unit_strip_enabled() is False
        os.environ["RAG_DECIMAL_UNIT_STRIP"] = "1"
        assert fmt.decimal_unit_strip_enabled() is True
    finally:
        os.environ.pop("RAG_DECIMAL_UNIT_STRIP", None)
        if prev is not None:
            os.environ["RAG_DECIMAL_UNIT_STRIP"] = prev


# ------------------------------------------------ SOT-2688 (cycle7 K5, idx29) ビン範囲の区間記法→チルダ書式
_BIN_Q = ("恒一会 かえで総合病院のtrain.xlsx内のTPのヒストグラムで、3番目にカウント数が多いビンの"
          "範囲を小数第6位までで答えてください。")


def test_bin_range_tilde_idx29_case():
    out, fired = fmt.naturalize_bin_range(_BIN_Q, "(6.088138, 6.288138]")
    assert out == "6.088138 ~ 6.288138"
    assert fired == ["bin_range_tilde"]


def test_bin_range_all_bracket_variants():
    for raw in ["(6.088138, 6.288138]", "[6.088138, 6.288138)", "(6.088138,6.288138)",
                "[6.088138, 6.288138]", "(6.088138、6.288138]"]:
        out, fired = fmt.naturalize_bin_range(_BIN_Q, raw)
        assert out == "6.088138 ~ 6.288138" and fired == ["bin_range_tilde"]


def test_bin_range_preserves_numeric_tokens():
    out, _ = fmt.naturalize_bin_range(_BIN_Q, "(-1.5, 2.0]")
    assert out == "-1.5 ~ 2.0"


def test_bin_range_requires_range_question():
    # 範囲/ビンを問わない問いは触らない（座標等の巻き添え防止）。
    out, fired = fmt.naturalize_bin_range("値を答えてください", "(6.088138, 6.288138]")
    assert out == "(6.088138, 6.288138]" and fired == []


def test_bin_range_noop_on_non_interval_answer():
    for v in ["6.088138 ~ 6.288138", "6.288138", "範囲は (6.0, 6.2] です", ""]:
        out, fired = fmt.naturalize_bin_range(_BIN_Q, v)
        assert out == v and fired == []
