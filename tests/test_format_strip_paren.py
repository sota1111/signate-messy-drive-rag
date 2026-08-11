"""SOT-2650 — 括弧内付加情報の書式契約 (RAG_FORMAT_STRIP_PAREN, default OFF, value-preserving).

Pins the wrong-class recoveries (idx4/8/12/16/41/52/59/84/87/88 shapes) AND the fail-closed
boundaries: gold answers that legitimately end with a parenthetical (idx49 task name, idx62 value
detail, idx78 prose) must never be altered.
"""
from __future__ import annotations

import pytest

from src.rag.agent import formatting


def _strip(question, value):
    return formatting.strip_trailing_parenthetical(question, value)


# --------------------------------------------------------------------------- flag
@pytest.mark.parametrize("val,expected", [("1", True), ("on", True), ("0", False), ("", False)])
def test_flag(monkeypatch, val, expected):
    monkeypatch.setenv("RAG_FORMAT_STRIP_PAREN", val)
    assert formatting.strip_paren_enabled() is expected


def test_flag_default_off():
    assert formatting.strip_paren_enabled() is False


# --------------------------------------------------------------------------- annotation shapes stripped
@pytest.mark.parametrize("question,value,expected", [
    # idx87: corporate-name expansion
    ("完了案件のうち…該当する案件を主略称で挙げてください",
     "AYM(青葉与信マネジメント株式会社)", "AYM"),
    # idx88: attribution
    ("スケジュール案において、第5週目に実施することになっている項目は何ですか",
     "解釈・業務示唆整理（担当：松本・鈴木）", "解釈・業務示唆整理"),
    # idx84: slide locator
    ("モデル毎のF1スコアがランキング形式で記載されているページ数を教えてください",
     "5（スライド6、文書内ページ番号5）", "5"),
    # idx59: locator with nested quotes
    ("金額の提示がまとまっているのは何ページですか",
     "13ページ（スライド13「8. 費用見積」）", "13ページ"),
    # idx4: metric detail
    ("目的変数と相関が最も高い数値特徴量を教えてください", "bmi(相関係数 約0.171)", "bmi"),
    # alternate week notation repeats the value rather than adding a second fact
    ("どの週からどの週に変更されましたか", "第5週目から第6週目（W5〜W6）", "第5週目から第6週目"),
    # idx41: enumeration listing
    ("加藤さんが担当者に含まれるタスクIDはいくつありますか",
     "11件(タスクID: T01, T02, T05)", "11件"),
    # idx12: heading locator
    ("WBS観点の進捗状況の見出しがあるのは何ページですか",
     "2ページ目（見出し「2. 進捗状況」の中の小見出し）", "2ページ目"),
    # idx8: metric detail (約 + digits)
    ("MLエンジニアとデータエンジニアの差はいくらですか",
     "14,744ドル（MLエンジニア 約140,000ドル − データエンジニア 125,256ドル）", "14,744ドル"),
])
def test_annotation_paren_stripped(question, value, expected):
    out, rules = _strip(question, value)
    assert out == expected
    assert "strip_trailing_paren" in rules


def test_question_echo_qualifier_stripped():
    # idx52: 「別契約」 echoes the question's own qualifier — annotation, not value
    q = "データアステル側の役割として「別契約」と明記されているものを抽出してください"
    out, rules = _strip(q, "監視ダッシュボード構築（別契約）")
    assert out == "監視ダッシュボード構築"


def test_sentence_tail_preserved():
    out, _ = _strip("差額はいくらですか", "差額は0円です（提案時と同額のため）。".replace("提案時と同額のため", "スライド3参照"))
    assert out == "差額は0円です。"


def test_quote_unwrap_after_paren_strip():
    # idx16: verbatim ask — meta-commentary paren (部分) strips, then the 「」 wrap unwraps
    q = "黄色ハイライトかつ赤字となっている部分を抜き出してください"
    out, rules = _strip(q, "「0.589」（F1 (macro): 0.58929290 という記載のうち「0.589」の部分）")
    assert out == "0.589"
    assert rules == ["strip_trailing_paren", "unwrap_quotes"]


def test_verbatim_quote_with_paragraph_locator_stripped():
    q = "契約書において、太字で記載されている部分を抽出してください"
    out, rules = _strip(q, "「契約締結日兼効力発生日：2025-10-01」（契約書.docx 第114段落）")
    assert out == "契約締結日兼効力発生日：2025-10-01"
    assert rules == ["strip_trailing_paren", "unwrap_quotes"]


# --------------------------------------------------------------------------- fail-closed keeps
@pytest.mark.parametrize("question,value", [
    # idx49-shape gold: task name whose paren detail is part of the value — no annotation cue
    ("M02までに完了しているタスクを挙げてください", "WBS・進捗管理台帳確定（タスク割振・ガント更新）"),
    # idx62-shape gold: value detail (settings values) — no annotation cue
    ("上位2件のスコア差を生んでいる設定差分は何ですか", "n_estimators（1位=500、2位=300）"),
    # idx22-shape gold: supplementary but value-bearing comparison — no annotation cue
    ("内容として変わっている点は何ですか", "classの列の統計量が追加された（Attr1〜64は同一）"),
    # fully parenthesized answer — the group IS the value
    ("ビンの範囲を答えてください", "（6.088138, 6.288138）"),
    # interval notation ending with ] — not a paren group
    ("ビンの範囲を答えてください", "(6.088138, 6.288138]"),
    # mid-string parenthetical only — never touched
    ("アクションIDA10の内容を教えてください", "0値を疑似欠損（NA）扱いにする処理を実装"),
    # unbalanced trailing closer — left alone
    ("設定は何ですか", "value）"),
])
def test_value_bearing_paren_kept(question, value):
    out, rules = _strip(question, value)
    assert out == value
    assert rules == []


def test_verbatim_ask_keeps_source_parens():
    # そのまま抜き出し: a source-text paren (no meta-commentary cue) must survive
    q = "アクションIDA10の内容をそのまま抜き出してください"
    out, rules = _strip(q, "補完ロジック（中央値等）")
    assert out == "補完ロジック（中央値等）"
    assert rules == []


def test_multiline_answer_untouched():
    out, rules = _strip("変更を挙げてください", "1. 変更A（スライド2）\n2. 変更B（スライド3）")
    assert rules == []
    assert "変更A（スライド2）" in out


def test_multiple_trailing_annotation_groups():
    out, rules = _strip("何ページですか", "5（スライド6）（文書内ページ番号5）")
    assert out == "5"


def test_abstain_and_plain_values_unchanged():
    assert _strip("質問", "わかりません")[1] == []
    assert _strip("質問", "6")[1] == []
    assert _strip("質問", "")[1] == []


# --------------------------------------------------------------------------- commit-gate composition
def test_commit_gate_formatting_applies_strip(monkeypatch):
    from src.rag.agent import commit_gate

    monkeypatch.setenv("RAG_FORMAT_STRIP_PAREN", "1")
    out, tel = commit_gate._apply_formatting(
        "記載されているページ数を教えてください", "fact_lookup", "5（スライド6）", None, None)
    assert out == "5"
    assert tel["applied"] is True and "strip_trailing_paren" in tel["rules"]


def test_commit_gate_formatting_off_by_default(monkeypatch):
    from src.rag.agent import commit_gate

    monkeypatch.delenv("RAG_FORMAT_STRIP_PAREN", raising=False)
    out, _ = commit_gate._apply_formatting(
        "記載されているページ数を教えてください", "fact_lookup", "5（スライド6）", None, None)
    assert out == "5（スライド6）"
