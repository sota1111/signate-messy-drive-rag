"""SOT-2719 — 「ページ数」型のみ bare 番号へ決定論整形 (RAG_PAGE_COUNT_BARE, default OFF, value-preserving).

idx84「…記載されているページ数を教えてください」の gold は `5` (bare) だが、Gemini serve 経路は
「5ページ（スライド6）」等の単位＋provenance 注記付きで返し judge 不一致になる framing churn。
本ルールは設問が「ページ数」型のときに限り、回答内に既にある印字ページ番号 N を bare 化する（数値は不変）。

【最重要】回帰ガード: gold のページ形式は型ごとに違う (idx84=5 / idx12=2ページ / idx59=13ページ /
idx18=2ページ目)。発火は「ページ数」トークンを問う設問のみに厳密ゲートし、「何ページ」「ページ番号」型は
絶対に触らない。gold100 全走査で「ページ数」を含む設問は idx84 のみなので、12/18/59 は構造的に無改変。
"""
from __future__ import annotations

import pytest

from src.rag.agent import formatting


def _bare(question, value):
    return formatting.apply_page_count_bare(question, value)


# --------------------------------------------------------------------------- flag
@pytest.mark.parametrize("val,expected", [("1", True), ("on", True), ("0", False), ("", False)])
def test_flag(monkeypatch, val, expected):
    monkeypatch.setenv("RAG_PAGE_COUNT_BARE", val)
    assert formatting.page_count_bare_enabled() is expected


def test_flag_default_off():
    assert formatting.page_count_bare_enabled() is False


# --------------------------------------------------------------------------- idx84 recovery (target)
Q84 = "東都人材プラットフォームの最終報告書で分析結果が記載されている中で、モデル毎のF1スコアがランキング形式で記載されているページ数を教えてください。"


def test_idx84_strip_unit_and_slide_annotation():
    # 「5ページ（スライド6）」→ gold 「5」(bare): 単位「ページ」と provenance 注記「（スライド6）」を剥がす。
    out, rules = _bare(Q84, "5ページ（スライド6）")
    assert out == "5"
    assert rules == ["page_count_bare"]


def test_idx84_strip_unit_only():
    out, rules = _bare(Q84, "5ページ")
    assert (out, rules) == ("5", ["page_count_bare"])


def test_idx84_fullwidth_digit():
    out, rules = _bare(Q84, "５ページ目")
    assert (out, rules) == ("5", ["page_count_bare"])


def test_idx84_already_bare_is_noop():
    # 既に bare の正解は無改変（fired=False）。
    out, rules = _bare(Q84, "5")
    assert (out, rules) == ("5", [])


def test_idx84_labeled_self_report_framing():
    # churn variant B「スライド6（文書に記載のページ番号: 5）」→ gold「5」: 回答が自己申告した印字ページ番号 5 を
    # 「ページ番号:」に係留して採る。先頭の「スライド6」の 6 は「ページ」非係留なので決して拾わない。
    out, rules = _bare(Q84, "スライド6（文書に記載のページ番号: 5）")
    assert (out, rules) == ("5", ["page_count_bare"])


def test_idx84_labeled_fullwidth_colon():
    out, rules = _bare(Q84, "スライド6（ページ：5）")
    assert (out, rules) == ("5", ["page_count_bare"])


def test_idx84_slide_only_no_guess():
    # 印字ページ番号をどこにも述べない（スライド番号だけ）回答は N を推測しない（値保存・fail-closed）。
    out, rules = _bare(Q84, "スライド6")
    assert (out, rules) == ("スライド6", [])


# --------------------------------------------------------------------------- 回帰ガード (idx12/18/59)
def test_idx12_nan_page_kept():
    # idx12「…何ページですか」gold=2ページ ⇒ 発火せず現状維持。
    q = "蒼泉会 ひがし丘総合病院の報告資料_2025-07-08.docxにおいて、WBS観点の進捗状況の見出しがあるのは何ページですか。"
    assert _bare(q, "2ページ") == ("2ページ", [])


def test_idx59_nan_page_kept():
    # idx59「…何ページですか」gold=13ページ ⇒ 発火せず現状維持。
    q = "京ソのPP_final.pptxにおいて、この案件にかかる金額の提示がまとまっているのは何ページですか。"
    assert _bare(q, "13ページ") == ("13ページ", [])


def test_idx18_page_number_kept():
    # idx18「…ページ番号を答えてください」gold=2ページ目 ⇒ 発火せず現状維持（ページ目保持）。
    q = "白峰信用リスク評価の会議ID：M04の会議録にて、進捗サマリが記載されているページ番号を答えてください。"
    assert _bare(q, "2ページ目") == ("2ページ目", [])


def test_page_number_ask_even_with_page_count_word_is_kept():
    # 防御的: 「何ページ」/「ページ番号」型は、仮に「ページ数」文字列が同居しても除外側が勝つ。
    q = "総ページ数のうち、進捗サマリは何ページですか。"
    assert _bare(q, "3ページ") == ("3ページ", [])


# --------------------------------------------------------------------------- fail-closed / value-preservation
def test_non_page_count_question_noop():
    out, rules = _bare("差額はいくらですか。", "5ページ")
    assert (out, rules) == ("5ページ", [])


def test_abstain_like_multiline_noop():
    out, rules = _bare(Q84, "5ページ\n補足あり")
    assert rules == []


def test_empty_noop():
    assert _bare(Q84, "") == ("", [])
