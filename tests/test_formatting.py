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
                                                          evidence={"file": "train.xlsx"}))

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
                                                          evidence={"file": "train.xlsx"}))

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
