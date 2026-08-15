"""SOT-2720 — 小数第N位 決定論丸め契約 (RAG_DECIMAL_PRECISION) ＋ Gemini 該当なし裸形式 fold (RAG_NONE_BARE_FOLD).

いずれも default OFF・値保存の書式契約で、純 Gemini serve 経路の serve-boundary hook として適用される
(既存の RAG_CURRENCY_DIFF_UNIT / RAG_PAGE_COUNT_BARE と同型)。

レバーA (小数精度丸め): 設問が「小数第N位(まで)」を指定するとき数値回答を N 桁へ ROUND_HALF_UP。
  idx57 (0.42395962→0.42396) / idx63 (0.15001822→0.15002) を gold 文字列一致。
  【回帰ガード】精度指定が無い数値は絶対に丸めない — idx36=0.09619112771492555 (17桁) / idx68「小数で」/ idx16/35。

レバーB (該当なし裸形式): 列挙/挙示型設問への冗長 no-items 回答を裸「該当なし」へ畳む。
  idx85「未達成…はありません（全6項目達成）」→「該当なし」。既に正答の idx9/38/85 は idempotent。
  【回帰ガード】回答が存在する設問を誤って該当なしにしない (非存在結論と判定できるときのみ)。
"""
from __future__ import annotations

import pytest

from src.rag.agent import formatting
from src.rag.agent import investigator


# =============================================================== レバーA: 小数精度丸め
def _round(question, value):
    return formatting.apply_decimal_precision(question, value)


Q57 = ("青葉のTXにて算出された回帰係数を用いて全データの予測値を計算し、正解データに対する F1 スコアが最大となる"
       "ように閾値を設定したときの F1 スコアを答えてください。小数第5位まで求めてください。")
Q63 = ("青葉与信マネジメントのtrain.xlsxにて算出された回帰係数を使ってid=0を予測した場合の予測値はいくらになりますか。"
       "小数第5位まで求めてください。")


@pytest.mark.parametrize("val,expected", [("1", True), ("on", True), ("0", False), ("", False)])
def test_flag(monkeypatch, val, expected):
    monkeypatch.setenv("RAG_DECIMAL_PRECISION", val)
    assert formatting.decimal_precision_enabled() is expected


def test_flag_default_off():
    assert formatting.decimal_precision_enabled() is False


def test_idx57_round_to_5():
    # 0.42395962 → 小数第5位 → 0.42396 (6桁目8で切り上げ)。
    out, rules = _round(Q57, "0.42395962")
    assert (out, rules) == ("0.42396", ["decimal_precision"])


def test_idx63_round_to_5():
    # 0.15001822 → 小数第5位 → 0.15002 (6桁目8で切り上げ)。
    out, rules = _round(Q63, "0.15001822")
    assert (out, rules) == ("0.15002", ["decimal_precision"])


def test_round_half_up_boundary():
    q = "…小数第2位まで答えてください。"
    assert _round(q, "1.005")[0] == "1.01"  # ROUND_HALF_UP (banker's だと 1.00)


def test_already_at_precision_is_noop():
    # 既に N 桁の回答 (idx33/54 型) は丸めても値不変 ⇒ no-op。
    q = "…小数第5位で答えてください。"
    assert _round(q, "0.18696") == ("0.18696", [])
    assert _round(q, "0.50534") == ("0.50534", [])


def test_unit_preserved():
    q = "…割合は何%ですか。小数第2位まで答えてください。"
    out, rules = _round(q, "12.3456%")
    assert (out, rules) == ("12.35%", ["decimal_precision"])


def test_comma_grouping_value_unchanged_is_noop():
    # 値が丸めで変わらないときは桁区切りなど表層を保持して no-op (回帰安全)。
    q = "…小数第2位まで答えてください。"
    assert _round(q, "1,234.50") == ("1,234.50", [])


# ---- 【最重要】回帰ガード: 精度指定が無い数値は絶対に丸めない -------------------------
Q36 = "恒一会 かえで総合病院案件において、中間報告時点のF1スコア実測値と最終報告時点のF1スコア実測値の差を絶対値で答えてください。"
Q68 = "東都人材プラットフォームのデータサイエンス市場の未来予測.pdfにおいて、投資実装係数の計算式が記載されているページの数値情報を式に代入し、投資実装係数を小数で答えてください。"
Q35 = "京橋信用ソリューションズの…F1スコアにてgradient_boostingに次ぐ順位のモデルの Accuracy はいくつですか。"


def test_idx36_full_precision_untouched():
    # 設問に「小数第N位」が無い ⇒ 17桁 full precision を絶対に丸めない。
    assert _round(Q36, "0.09619112771492555") == ("0.09619112771492555", [])


def test_idx68_kosuu_de_no_place_spec_untouched():
    # 「小数で」は桁指定ではない ⇒ 発火しない (idx68 gold=1.3986 維持)。
    assert _round(Q68, "1.3986") == ("1.3986", [])
    assert _round(Q68, "1.39860001") == ("1.39860001", [])


def test_idx35_no_spec_untouched():
    assert _round(Q35, "0.90527") == ("0.90527", [])
    assert _round(Q35, "0.905271234") == ("0.905271234", [])


def test_non_numeric_answer_noop():
    q = "…小数第5位まで求めてください。"
    assert _round(q, "該当なし") == ("該当なし", [])


def test_multiline_answer_noop():
    q = "…小数第5位まで求めてください。"
    assert _round(q, "0.42395962\n補足") == ("0.42395962\n補足", [])


def test_empty_noop():
    assert _round(Q57, "") == ("", [])


# =============================================================== レバーB: 該当なし裸形式 fold
def _fold(question, value):
    return formatting.fold_none_bare(question, value)


Q85 = "青葉バイオメディカル機器の最終報告において、設定されたKPIとして未達成とされている項目を挙げてください。"
Q9 = "青葉与信マネジメントの最終報告資料の最新版になる際に修正されたもののうち、案件遂行に関連する変更を挙げてください。"


@pytest.mark.parametrize("val,expected", [("1", True), ("on", True), ("0", False), ("", False)])
def test_fold_flag(monkeypatch, val, expected):
    monkeypatch.setenv("RAG_NONE_BARE_FOLD", val)
    assert formatting.none_bare_fold_enabled() is expected


def test_fold_flag_default_off():
    assert formatting.none_bare_fold_enabled() is False


def test_idx85_verbose_no_items_with_paren_note():
    # 「未達成…はありません（全6項目達成）」→ 裸「該当なし」: 末尾括弧注記を落とし非存在結論を検出。
    out, rules = _fold(Q85, "未達成とされている項目はありません（全6項目達成）")
    assert (out, rules) == ("該当なし", ["none_bare_fold"])


def test_idx85_variants():
    for v in ["未達成の項目は存在しません", "該当する項目はありません", "未達成とされている項目はございません（6項目すべて達成）",
              "未達成とされている項目はありません（全6項目のKPIにおいてすべて「達成」と評価されています）。"]:
        out, rules = _fold(Q85, v)
        assert out == "該当なし", v
        assert rules == ["none_bare_fold"], v


def test_idx85_all_achieved_certificate():
    v = "設定されたKPI（データ理解、要因把握、モデル評価、説明可能性、実務接続、ガバナンスの全6項目）はすべて「達成」と評価されています。"
    assert _fold(Q85, v) == ("該当なし", ["none_bare_fold"])


def test_already_bare_none_is_noop():
    # 既に裸「該当なし」(idx9/38/85 の正答) は idempotent (fired=False)。
    assert _fold(Q85, "該当なし") == ("該当なし", [])
    assert _fold(Q9, "該当なし") == ("該当なし", [])


def test_canonicalizes_plain_none_form():
    # 素の none 形は「該当なし」へ正規化。
    out, rules = _fold(Q9, "なし")
    assert (out, rules) == ("該当なし", ["none_bare_fold"])


# ---- 回帰ガード: 回答が存在する設問を誤って該当なしにしない ----------------------
def test_real_enumeration_answer_not_folded():
    # 実項目を読点で列挙した回答は畳まない (複数項目列挙 = 該当あり)。
    v = "見出しラベルの追加、スライドの分割、体裁の再構成"
    assert _fold(Q9, v) == (v, [])


def test_real_single_item_answer_not_folded():
    # 非存在結論で終わらない実回答は畳まない。
    v = "データ収集の自動化が未達成"
    assert _fold(Q85, v) == (v, [])


def test_non_enumeration_question_noop():
    # 列挙/挙示型でない設問は対象外 (該当なし系文言でも触らない)。
    q = "契約金額はいくらですか。"
    assert _fold(q, "該当する金額はありません") == ("該当する金額はありません", [])


def test_multiline_noop():
    assert _fold(Q85, "未達成項目はありません\n詳細あり") == ("未達成項目はありません\n詳細あり", [])


def test_empty_noop_fold():
    assert _fold(Q85, "") == ("", [])


def test_serve_boundary_formats_deterministic_answer(monkeypatch):
    """The final contract must also cover deterministic early-return answers (idx57/63 route)."""
    monkeypatch.setenv("RAG_DECIMAL_PRECISION", "1")
    inv = investigator.Investigation(
        question=Q57,
        answer=investigator.Answer(answer="0.42395962", confidence=1.0),
        iterations=1,
        tool_calls=["det_pipeline:numeric"],
        usage=investigator.Usage(),
        model="deterministic",
        elapsed_s=0.0,
        stop_reason="answered",
    )
    out = investigator._apply_sot2720_formatting(inv, Q57)
    assert out.answer.answer == "0.42396"
    assert out.interventions["decimal_precision"]["fired"] is True
