"""SOT-2631 (G1, PLAN SOT-2602) — tests for the highlight-extraction procedure port.

Network-free / corpus-free: the detectors are pure regex over the question text, so these pin the
default-OFF contract, the per-archetype firing on the G1 target questions (and the same-class existing
MATCH idx7/42 the focused gate uses as regression checks), and — critically — that NO detector fires on
the mandatory focused-gate sentinels. The most important negative is sentinel #10
「黄色ハイライトかつ赤字…抜き出してください」, a champion MATCH textually adjacent to idx17: the idx17 detector
requires a calculation token the extraction-only sentinel lacks, so it must not fire. The target texts and
the ten sentinel texts are the real production questions.
"""
from __future__ import annotations

import re

import pytest

from src.rag.agent import g1_highlight_port as g1

# The G1 target questions (SOT-2631 / dossier), verbatim from data/questions/questions_test.csv.
# idx15/80/17 are the abstain-recovery targets; idx7/42 are same-class EXISTING champion MATCH answers the
# focused gate adds to --target as regression checks — the directive must fire on them too (same手順) and
# not break them.
TARGETS = {
    15: "東都人材プラットフォームのtrain.xlsxにおいて、Sheet1の黄色にハイライトされたセルの抽出条件と"
        "集計内容を答えてください。",
    80: "東都人材プラットフォームのtrain.xlsxにおいて、Sheet2の黄色にハイライトされたセルの抽出条件と"
        "集計内容を答えてください。",
    17: "AYMのMMにおいて、黄色ハイライトかつREDになっている数値を対象に、最初のMMから最後のMMまでの"
        "上昇率を計算してください。上昇率は （最後の値 - 最初の値） / 最初の値 × 100 で求め、"
        "小数第2位まで答えてください。",
    7: "青潮モビリティサービスの基礎分析.pptxにおいて、黄色ハイライトされている数値に対応するデータの"
       "抽出条件と集計内容を答えてください。",
    42: "蒼泉会 ひがし丘総合病院のtrain.xlsxのSheet1において、黄色ハイライトされている数値に対応する"
        "データの抽出条件と集計内容を答えてください。",
}

# The single archetype each target must (and only) match.
TARGET_ARCHETYPE = {
    15: "highlight_condition",
    80: "highlight_condition",
    7: "highlight_condition",
    42: "highlight_condition",
    17: "highlight_color_calc",
}

# The ten mandatory focused-gate sentinels (scripts/focused_sentinels.json), verbatim. A detector that
# fires on any of these would perturb an existing-MATCH answer under the gate — forbidden. Sentinel #10 is
# the critical near-miss for idx17 (黄∧赤 but extraction-only, no calc).
SENTINELS = [
    "白峰信用リスク評価の提案書old.pptxから提案書.pptxへの更新内容のうち、案件遂行に関連する実質的な"
    "変更を挙げてください。",
    "青嶺不動産アセットマネジメントのスケジュール_r2.xlsxにおいて、オレンジにハイライトされている行の"
    "タスク名をすべて答えてください。",
    "社内管理フォルダにあるFMにおいて、井上さんの向かいに座っている方のEXTを教えてください。",
    "青嶺不動産アセットマネジメントのスケジュール_r2.xlsxにおいて、2025-08-11から2025-09-09の間に開始日"
    "または終了日が設定されているタスクIDをすべて挙げてください。",
    "青葉与信マネジメントの分析対象データにおいて、標準化されたloan_amntが0未満の行のうち、"
    "purpose=credit_cardに該当し、かつloan_amntがpurpose=credit_cardの中央値より小さい行数を答えてください。",
    "青潮モビリティサービスの基礎分析.docxのグラフ2で、x=3のときの青色の折れ線のyの値を小数第5位で"
    "答えてください。",
    "IMにあるFMにおいて、佐藤さんから見て右側に座っている人の名前をすべて挙げてください。",
    "恒一会 かえで総合病院のtrain.xlsxにおいて、AG_ratioのヒストグラムで最も多いカウント数はいくつですか。",
    "青嶺不動産アセットマネジメントの報告資料の中で、太字、下線、イタリックのすべてに該当する箇所を"
    "抽出してください。",
    "青葉与信マネジメントの中間報告資料にて、黄色ハイライトかつ赤字となっている部分を抜き出してください。",
]


# --------------------------------------------------------------------------- flag / defaults
def test_disabled_by_default(monkeypatch):
    monkeypatch.delenv("RAG_G1_HIGHLIGHT_PORT", raising=False)
    assert g1.enabled() is False


@pytest.mark.parametrize("val,expect", [("1", True), ("true", True), ("on", True),
                                        ("0", False), ("", False), ("no", False)])
def test_enabled_reads_flag(monkeypatch, val, expect):
    monkeypatch.setenv("RAG_G1_HIGHLIGHT_PORT", val)
    assert g1.enabled() is expect


# --------------------------------------------------------------------------- per-target firing
@pytest.mark.parametrize("idx", sorted(TARGETS))
def test_each_target_fires_its_archetype(idx):
    directive, tel = g1.port_directive(TARGETS[idx])
    assert directive is not None, f"target idx{idx} produced no directive"
    assert tel["fired"] is True
    assert TARGET_ARCHETYPE[idx] in tel["archetypes"], (
        f"idx{idx} expected {TARGET_ARCHETYPE[idx]}, got {tel['archetypes']}")


def test_targets_fire_exactly_one_archetype():
    # The two detectors are mutually exclusive: no target matches both.
    for idx in TARGETS:
        _, tel = g1.port_directive(TARGETS[idx])
        assert len(tel["archetypes"]) == 1, f"idx{idx} fired {tel['archetypes']}"


def test_idx17_routes_calc_through_pot_lane():
    # idx17 arithmetic must instruct the PoT lane (verify_formula) — no 暗算 commit.
    directive, _ = g1.port_directive(TARGETS[17])
    assert "verify_formula" in directive


def test_condition_directive_mentions_reverse_lookup_and_selfverify():
    directive, _ = g1.port_directive(TARGETS[15])
    assert "逆引き" in directive
    assert "自己検証" in directive


# --------------------------------------------------------------------------- sentinel safety
@pytest.mark.parametrize("q", SENTINELS)
def test_no_detector_fires_on_sentinels(q):
    directive, tel = g1.port_directive(q)
    assert directive is None, f"a G1 detector fired on a sentinel: {tel['archetypes']}"
    assert tel["fired"] is False


def test_idx17_detector_excludes_extraction_only_sentinel():
    # The critical near-miss: sentinel #10 shares 「黄色ハイライトかつ赤字」 with idx17 but is extraction-only.
    sentinel10 = SENTINELS[9]
    assert "黄色ハイライトかつ赤字" in sentinel10          # shares the composite phrasing …
    assert g1.is_highlight_color_calc(sentinel10) is False  # … but no calc token ⇒ must not fire.
    assert g1.is_highlight_color_calc(TARGETS[17]) is True


# --------------------------------------------------------------------------- procedure-only / no-fact
def test_directives_are_procedure_text_only():
    # Every directive is non-empty procedure guidance; none embeds a resolved value literal (an "=<number>"
    # assignment or a currency amount) — the port carries 手順 only, never a corpus fact / answer.
    for i in TARGETS:
        directive, _ = g1.port_directive(TARGETS[i])
        assert directive and len(directive) > 20
        assert not re.search(r"=\s*\d", directive)   # no "children=3" style answer
        assert "円" not in directive                  # no currency amount


def test_non_matching_question_is_idle():
    # A plain question matches no archetype: no directive, telemetry records the ON-but-idle state.
    directive, tel = g1.port_directive("このデータセットの行数はいくつですか。")
    assert directive is None
    assert tel["fired"] is False
    assert tel["archetypes"] == []
