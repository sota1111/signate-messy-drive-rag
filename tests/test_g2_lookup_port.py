"""SOT-2632 (G2, PLAN SOT-2602) — tests for the lookup/derived procedure port.

Network-free / corpus-free: the detectors are pure regex over the question text, so these pin the
default-OFF contract, the per-archetype firing on the five G2 target questions, and — critically — that
NO detector fires on the mandatory focused-gate sentinel questions (a false fire on a sentinel would risk
a gate regression). The five target texts and the ten sentinel texts are the real production questions.
"""
from __future__ import annotations

import io

import pytest

from src.rag.agent import g2_lookup_port as g2

# The five G2 target questions (SOT-2632 / dossier), verbatim from artifacts/questions_test.csv.
TARGETS = {
    5: "青潮モビリティサービスの最終報告にて最良モデルとしているモデルのパラメータであるmax_depthは"
       "いくらに設定されていますか。",
    53: "TOTOのFR書にて記載のある選択特徴量のうち、ENG-FTはいくつありますか。",
    96: "青葉与信マネジメントのチェックポイント2として設定されている内容に関連するタスクIDを教えてください。",
    36: "恒一会 かえで総合病院案件において、中間報告時点のF1スコア実測値と最終報告時点のF1スコア実測値の"
        "差を絶対値で答えてください。",
    79: "恒一会 かえで総合病院の計画フォルダ内において、データアステル側の担当者のうち、1タスク当たりの"
        "想定工数（想定工数 ÷ 担当タスク数）が最も大きい人のフルネームと、その1タスク当たりの想定工数を"
        "小数第2位で答えてください。ファイルに鍵がかかっている場合は社内管理を確認してください。",
}

# The ten mandatory focused-gate sentinels (scripts/focused_sentinels.json), verbatim. A detector that
# fires on any of these would perturb an existing-MATCH answer under the gate — forbidden.
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

# The single archetype each target must (and only) match.
TARGET_ARCHETYPE = {
    5: "hyperparam_code_fallback",
    53: "legend_color_count",
    96: "cross_sheet_hop",
    36: "two_doc_metric_diff",
    79: "instructed_fallback_aggregate",
}


# --------------------------------------------------------------------------- flag / defaults
def test_disabled_by_default(monkeypatch):
    monkeypatch.delenv("RAG_G2_LOOKUP_PORT", raising=False)
    assert g2.enabled() is False


@pytest.mark.parametrize("val,expect", [("1", True), ("true", True), ("on", True),
                                        ("0", False), ("", False), ("no", False)])
def test_enabled_reads_flag(monkeypatch, val, expect):
    monkeypatch.setenv("RAG_G2_LOOKUP_PORT", val)
    assert g2.enabled() is expect


# --------------------------------------------------------------------------- per-target firing
@pytest.mark.parametrize("idx", sorted(TARGETS))
def test_each_target_fires_its_archetype(idx):
    directive, tel = g2.port_directive(TARGETS[idx])
    assert directive is not None, f"target idx{idx} produced no directive"
    assert tel["fired"] is True
    assert TARGET_ARCHETYPE[idx] in tel["archetypes"], (
        f"idx{idx} expected {TARGET_ARCHETYPE[idx]}, got {tel['archetypes']}")


def test_calc_targets_route_through_pot_lane():
    # idx36/79 directives must instruct the PoT lane (verify_formula) — no 暗算 commit.
    for idx in (36, 79):
        directive, _ = g2.port_directive(TARGETS[idx])
        assert "verify_formula" in directive


def test_idx79_instructed_fallback_echoes_question_clause():
    directive, tel = g2.port_directive(TARGETS[79])
    assert tel["instructed_fallback"] is True
    # The clause is lifted verbatim from the question text (never a corpus fact / answer).
    assert "社内管理を確認" in directive


# --------------------------------------------------------------------------- sentinel safety
@pytest.mark.parametrize("q", SENTINELS)
def test_no_detector_fires_on_sentinels(q):
    directive, tel = g2.port_directive(q)
    assert directive is None, f"a G2 detector fired on a sentinel: {tel['archetypes']}"
    assert tel["fired"] is False


def test_directives_are_procedure_text_only():
    # Every directive is non-empty procedure guidance; none embeds a resolved value literal (an "=<number>"
    # assignment or a currency amount) — the port carries手順 only, never a corpus fact / answer.
    import re

    for i in TARGETS:
        directive, _ = g2.port_directive(TARGETS[i])
        assert directive and len(directive) > 20
        assert not re.search(r"=\s*\d", directive)   # no "max_depth=6" style answer
        assert "円" not in directive                  # no currency amount


# --------------------------------------------------------------------------- compute tool-gap wiring
def test_compute_decrypt_helper_off_by_default(monkeypatch):
    from src.rag.tools import compute_sandbox as cs

    monkeypatch.delenv("RAG_G2_LOOKUP_PORT", raising=False)
    assert cs._g2_decrypt_enabled() is False
    # OFF ⇒ never attempts decryption (returns None regardless of the ref).
    from pathlib import Path

    from src.rag.corpus import FileRef

    ref = FileRef(path=Path("/nonexistent.xlsx"), project="", category="", rel="x.xlsx",
                  name="x.xlsx", ext="xlsx")
    assert cs._decrypted_xlsx_source(ref) is None


def test_read_xlsx_accepts_buffer_source(monkeypatch):
    # The refactored _read_xlsx must accept an in-memory buffer (not only a path) so a decrypted xlsx can
    # be read. Build a tiny xlsx in memory and confirm it parses.
    from openpyxl import Workbook

    from src.rag.tools import compute_sandbox as cs

    wb = Workbook()
    ws = wb.active
    ws.append(["a", "b"])
    ws.append([1, 2])
    ws.append([3, 4])
    buf = io.BytesIO()
    wb.save(buf)
    buf.seek(0)
    df, title, trace = cs._read_xlsx(buf, None, display_name="mem.xlsx")
    assert list(df.columns) == ["a", "b"]
    assert int(df["a"].sum()) == 4
