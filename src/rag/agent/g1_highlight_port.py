"""SOT-2631 (G1, PLAN SOT-2602) — port Sonnet's highlight-extraction success procedures to the flash path.

The Sonnet dev gold100 (SOT-2628 / SOT-2630 dossier ``docs/ai/sonnet_trace_dossier.md``) reached three
highlight-extraction-condition questions the flash champion abstained on (idx 15 / 80 / 17). Every one
is solvable with the *same* tool set (highlight_extract / read_office / compute); the gap is 誘導・手順
(which cell to reverse-lookup, that the pivot labels reconstruct the extraction condition, that the
aggregate can self-verify with compute, and — for idx17 — that the arithmetic must go through the PoT
lane), not model capability. This module ports each question's **procedure** (not its answer) as an
advisory HINT appended to the generation agent's preamble.

Design invariants (mirroring the sibling opt-ins — evidence_packet / condition_prefill / pot_lane /
g2_lookup_port):

* **Opt-in, byte-identical OFF.**  :func:`enabled` (``RAG_G1_HIGHLIGHT_PORT``) gates the whole thing; the
  default-OFF serve path never builds a directive and injects nothing, so the champion answer path is
  byte-identical.
* **No firing-condition relaxation, no gold values.**  Detection is regex over the *question text only*.
  A question that matches no G1 archetype gets **no** directive and follows the ordinary path (the
  SOT-2601 net12 failure axis — 発火条件の緩和 — is deliberately avoided). The directives carry procedure
  guidance only; they never inject a corpus fact, a document's contents, or the answer.
* **Precision-preserving.**  The directives are advisory (appended to the prompt; final tool choice stays
  with the model) and each is scoped tightly to its archetype's question shape so it never fires on an
  unrelated question. Critically, the mandatory focused-gate sentinels do NOT match any detector — most
  importantly sentinel #10 「黄色ハイライトかつ赤字…抜き出してください」 (a champion MATCH that is textually
  adjacent to idx17): the idx17 detector requires a *calculation* token (上昇率／変化率／×100 …), which the
  extraction-only sentinel lacks, so it does not fire on it. The idx17 arithmetic is routed through the
  PoT lane (``verify_formula``: binder → 制限AST → Decimal → 独立検算, EXEC_MATCH only commit) — never 暗算.

Each detector corresponds to one dossier archetype:

  * :func:`is_highlight_condition`        idx15 / idx80 (同型既存MATCH idx7 / idx42) — 黄色ハイライトセル/
    数値の「抽出条件と集計内容」を pivot ラベル逆引き＋compute 自己検証で復元.
  * :func:`is_highlight_color_calc`       idx17 — 黄(ハイライト)∧RED(赤字) の複合色条件に該当する数値の
    上昇率計算（複合特定→PoT lane 検算）.
"""
from __future__ import annotations

import os
import re
from typing import Any

_ON = {"1", "true", "yes", "on"}


def enabled() -> bool:
    """True when the G1 highlight-extraction procedure port is active (default OFF — opt-in).

    Gated by ``RAG_G1_HIGHLIGHT_PORT`` exactly like the sibling opt-ins (RAG_EVIDENCE_PACKET /
    RAG_CONDITION_PREIR / RAG_POT_HARD_LANE / RAG_G2_LOOKUP_PORT): default-OFF means the serve path
    injects nothing extra, so the champion answer path is byte-identical.
    """
    return os.getenv("RAG_G1_HIGHLIGHT_PORT", "0").strip().lower() in _ON


# --------------------------------------------------------------------------- archetype detectors
# A highlight-colour token. Kept coarse (黄色/黄) so it matches idx15/80/17 without demanding an exact
# phrasing; the archetype-specific companion tokens below keep each detector off unrelated questions.
_HIGHLIGHT = re.compile(r"ハイライト|highlight", re.I)
_YELLOW = re.compile(r"黄色|黄")
# The extraction-condition archetype names its deliverable literally: 「抽出条件と集計内容」 (idx7/15/42/80 all
# use this exact pair). Requiring the 抽出条件 token keeps this off other highlight questions (e.g. the
# sentinel 「オレンジにハイライトされている行のタスク名」 — タスク名, not 抽出条件).
_EXTRACTION_CONDITION = re.compile(r"抽出条件")


def is_highlight_condition(question: str) -> bool:
    """idx15 / idx80 (同型既存MATCH idx7 / idx42) — a yellow-highlighted cell's 抽出条件と集計内容."""
    q = question or ""
    return bool(_YELLOW.search(q) and _HIGHLIGHT.search(q) and _EXTRACTION_CONDITION.search(q))


# The composite colour condition 「黄(ハイライト) ∧ RED(赤字)」 (idx17). "RED" appears verbatim in idx17
# (no ASCII \b: it is preceded by a Japanese char 「つRED」 where \b never matches); "赤字" is the equivalent
# Japanese wording used by the adjacent sentinel #10.
_RED = re.compile(r"RED|赤字|赤色|赤文字|赤", re.I)
# A calculation signal. idx17 asks for an 上昇率 (…÷…×100); the textually-adjacent champion-MATCH sentinel
# 「黄色ハイライトかつ赤字…抜き出してください」 is extraction-only and carries NONE of these, so requiring a
# calc token here is what keeps the idx17 detector from firing on that sentinel.
_CALC = re.compile(r"上昇率|変化率|増加率|減少率|伸び率|率を|割合|×\s*100|÷|計算し")


def is_highlight_color_calc(question: str) -> bool:
    """idx17 — a rate/ratio over numbers that satisfy the composite 黄(highlight)∧RED(赤字) condition."""
    q = question or ""
    return bool(_YELLOW.search(q) and _HIGHLIGHT.search(q) and _RED.search(q) and _CALC.search(q))


def _archetypes(question: str) -> list[str]:
    """The list of G1 archetype keys the question matches (0..n; in practice at most one).

    The two detectors are mutually exclusive by construction: ``highlight_condition`` requires 抽出条件
    (a phrase absent from idx17) and ``highlight_color_calc`` requires both RED and a calc token (both
    absent from idx7/15/42/80), so a question fires at most one archetype.
    """
    fired: list[str] = []
    if is_highlight_condition(question):
        fired.append("highlight_condition")
    if is_highlight_color_calc(question):
        fired.append("highlight_color_calc")
    return fired


# --------------------------------------------------------------------------- directives (procedure only)
_H_CONDITION = (
    "【手順ヒント: ハイライトセルの抽出条件と集計内容の復元】まず highlight_extract(color=黄) で該当ファイルの"
    "黄色ハイライトセル/数値を特定する。次に read_office/compute でそのセルが属する pivot(outline)の行ラベル・"
    "列ラベルを、結合セルの前方補完(空欄は直上/直左の見出しを継承)で逆引きし、抽出条件(どのグループ・どの集計軸か)"
    "を復元する。抽出条件が確定したら compute で元データに同条件の集計を適用して該当セル値と一致することを"
    "自己検証してから、抽出条件＋集計内容を根拠付きで回答する。ドメイン語意(例: コード 0/1/2 が価格帯 低/中/高 等)は"
    "提案書/会議録/最終報告を file_grep で横断裏取りして補う。値の推測やハイライト無視はしない。")
_H_COLOR_CALC = (
    "【手順ヒント: 複合色条件(黄∧赤)の数値の上昇率】対象は『黄色ハイライト かつ 赤字(RED)』の両条件を同時に"
    "満たす数値のみ。highlight_extract の色情報(背景ハイライト色とフォント色)を突き合わせて複合条件を満たす"
    "数値だけを時系列順(最初→最後)に抽出する。片方の条件しか満たさない数値は対象外。上昇率は暗算で commit せず、"
    "最初の値と最後の値を source 付きに束縛し、利用可能なら verify_formula(PoT lane: (最後-最初)/最初×100)で"
    "EXEC_MATCH のみを commit する(利用不可なら compute で再検算する)。小数第2位まで整形する。")


_DIRECTIVE_BY_KEY = {
    "highlight_condition": _H_CONDITION,
    "highlight_color_calc": _H_COLOR_CALC,
}


def port_directive(question: str, *, contract: str | None = None) -> tuple[str | None, dict[str, Any]]:
    """Build the advisory G1 procedure directive for ``question`` (or ``None`` when nothing matches).

    Returns ``(directive_or_None, telemetry)``.  ``telemetry`` is the SOT-2629-style per-question
    intervention record: ``{"archetypes": [...], "fired": bool}`` — a key is recorded by the caller only
    when the flag is ON, and ``fired`` is False when the flag was ON but the question matched no archetype
    (an ON-but-idle intervention, distinguishable from OFF).  Never injects a corpus fact or the answer —
    procedure guidance only.
    """
    fired = _archetypes(question or "")
    tel: dict[str, Any] = {
        "archetypes": list(fired),
        "fired": bool(fired),
    }
    if not fired:
        return None, tel
    directive = "\n\n".join(_DIRECTIVE_BY_KEY[k] for k in fired)
    return directive, tel
