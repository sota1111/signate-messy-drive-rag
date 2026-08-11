"""SOT-2656 — 値保存回答正規化 (RAG_FORMAT_VALUE_NORM, default OFF, value-preserving).

cycle4 クラスタE (docs/ai/sonnet_cycle_analysis/cycle4.md): the value is correct but wrapped in
non-value framing — a full sentence (idx6), an approximation prefix (idx8/36), or a redundant
counter (idx41/92). Pins the recoveries AND the fail-closed boundaries (value-bearing counts /
prose / non-scalar asks / approx-before-non-digit are never altered), plus the numeric-token
preservation guard and OFF-time byte-identity.
"""
from __future__ import annotations

import pytest

from src.rag.agent import formatting


def _norm(question, value):
    return formatting.normalize_value_answer(question, value)


def _strip_then_norm(question, value):
    """The production composition: SOT-2650 paren strip → SOT-2656 value-norm."""
    v, _ = formatting.strip_trailing_parenthetical(question, value)
    return formatting.normalize_value_answer(question, v)


# --------------------------------------------------------------------------- flag
@pytest.mark.parametrize("val,expected", [("1", True), ("on", True), ("0", False), ("", False)])
def test_flag(monkeypatch, val, expected):
    monkeypatch.setenv("RAG_FORMAT_VALUE_NORM", val)
    assert formatting.value_norm_enabled() is expected


def test_flag_default_off():
    assert formatting.value_norm_enabled() is False


# --------------------------------------------------------------------------- sentence frame (idx6)
def test_sentence_frame_currency():
    # idx6: 「差額は0円です（提案時と同額のため）」→ 「0円」 — the paren is a reason (not annotation-keyed),
    # so the whole recovery happens in value-norm's sentence-frame rule.
    q = "提案時と現在で差額はいくらですか"
    out, rules = _norm(q, "差額は0円です（提案時と同額のため）")
    assert out == "0円"
    assert rules == ["sentence_frame"]


def test_sentence_frame_requires_scalar_ask():
    # a non-scalar ask does not license the frame collapse (fail-closed).
    out, rules = _norm("内容を説明してください", "対象は3種類です")
    assert (out, rules) == ("対象は3種類です", [])


def test_sentence_frame_requires_digit_core():
    # prose core with no digit ⇒ not a scalar value ⇒ left alone.
    out, rules = _norm("担当はいくらですか", "担当は田中です")
    assert (out, rules) == ("担当は田中です", [])


# --------------------------------------------------------------------------- approximation prefix (idx8/36)
def test_approx_prefix_currency():
    out, rules = _norm("差はいくらですか", "約14,744ドル")
    assert out == "14,744ドル"
    assert rules == ["approx_prefix"]


@pytest.mark.parametrize("prefix", ["約", "およそ", "おおよそ", "ほぼ", "概ね", "おおむね"])
def test_approx_prefix_variants(prefix):
    out, rules = _norm("金額はいくらですか", f"{prefix}22,000円")
    assert out == "22,000円"
    assert rules == ["approx_prefix"]


def test_approx_prefix_not_before_digit_kept():
    # 「約款」 — 約 not followed by a digit ⇒ never stripped.
    out, rules = _norm("何が記載されていますか", "約款の変更点")
    assert (out, rules) == ("約款の変更点", [])


# --------------------------------------------------------------------------- counter suffix (idx41/92)
def test_count_suffix_bare_count_ask():
    out, rules = _norm("タスクIDはいくつありますか", "11件")
    assert out == "11"
    assert rules == ["count_suffix"]


def test_count_suffix_with_trailing_paren():
    # idx92 shape reaching the rule with its breakdown paren still attached.
    out, rules = _norm("いくつありますか", "49件（内訳: A20/B29）")
    assert out == "49"
    assert rules == ["count_suffix"]


def test_count_suffix_requires_count_ask():
    # not a count ask ⇒ 「N件」 is a value, kept.
    out, rules = _norm("対象は何ですか", "11件")
    assert (out, rules) == ("11件", [])


def test_count_suffix_only_whole_value():
    # a counter embedded in a longer phrase is not a bare count ⇒ untouched.
    out, rules = _norm("いくつありますか", "11件のタスク")
    assert (out, rules) == ("11件のタスク", [])


# --------------------------------------------------------------------------- production composition
@pytest.mark.parametrize("question,value,expected", [
    # idx8: paren strip removes the 根拠 group, value-norm removes 約.
    ("MLエンジニアとデータエンジニアの差はいくらですか",
     "約14,744ドル（MLエンジニア 約140,000ドル − データエンジニア 125,256ドル）", "14,744ドル"),
    # idx41: paren strip removes タスクID列挙, value-norm removes 件.
    ("加藤さんが担当者に含まれるタスクIDはいくつありますか",
     "11件(タスクID: T01, T02, T05)", "11"),
    # idx92: 内訳 paren is NOT annotation-keyed so paren-strip skips it; count_suffix removes both.
    ("疑似欠損の対象はいくつありますか", "49件（内訳: 前処理20/本処理29）", "49"),
    # idx4 / idx59 / idx88 already recovered by the paren strip alone — value-norm is a no-op on them.
    ("目的変数と相関が最も高い数値特徴量を教えてください", "bmi(相関係数 約0.171)", "bmi"),
    ("金額の提示がまとまっているのは何ページですか", "13ページ（スライド13「8. 費用見積」）", "13ページ"),
])
def test_strip_then_norm_targets(question, value, expected):
    out, _ = _strip_then_norm(question, value)
    assert out == expected


# --------------------------------------------------------------------------- value-preservation guard
@pytest.mark.parametrize("question,value", [
    ("差額はいくらですか", "約14,744ドル（MLエンジニア 約140,000ドル − データエンジニア 125,256ドル）"),
    ("いくつありますか", "49件（内訳: 20/29）"),
    ("差額はいくらですか", "差額は0円です（同額）"),
])
def test_numeric_tokens_never_invented(question, value):
    out, rules = _strip_then_norm(question, value)
    src_nums = formatting._num_tokens(value)
    for tok in formatting._num_tokens(out):
        assert tok in src_nums  # every output number came from the input — none fabricated


# --------------------------------------------------------------------------- untouched / abstain / blank
def test_plain_values_unchanged():
    assert _norm("いくつですか", "6") == ("6", [])
    assert _norm("質問", "") == ("", [])
    assert _norm("差額はいくらですか", "0円") == ("0円", [])


def test_multiline_untouched():
    out, rules = _norm("いくつですか", "1件\n2件")
    assert rules == []


def test_ready_forms_pass_through():
    # idx12/idx18 form 「2ページ目」 must never be converted (双方向変換禁止) — no rule targets it.
    out, rules = _norm("何ページですか", "2ページ目")
    assert (out, rules) == ("2ページ目", [])


# --------------------------------------------------------------------------- OFF-time byte-identity (serve path)
def test_commit_gate_value_norm_off_by_default(monkeypatch):
    from src.rag.agent import commit_gate

    monkeypatch.delenv("RAG_FORMAT_VALUE_NORM", raising=False)
    monkeypatch.delenv("RAG_FORMAT_STRIP_PAREN", raising=False)
    out, _ = commit_gate._apply_formatting("差額はいくらですか", "numeric", "約14,744ドル", None, None)
    assert out == "約14,744ドル"


def test_commit_gate_value_norm_applies(monkeypatch):
    from src.rag.agent import commit_gate

    monkeypatch.setenv("RAG_FORMAT_VALUE_NORM", "1")
    monkeypatch.delenv("RAG_FORMAT_STRIP_PAREN", raising=False)
    out, tel = commit_gate._apply_formatting("差額はいくらですか", "numeric", "約14,744ドル", None, None)
    assert out == "14,744ドル"
    assert tel["applied"] is True and "approx_prefix" in tel["rules"]
