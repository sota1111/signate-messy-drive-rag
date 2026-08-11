"""SOT-2632 (G2, PLAN SOT-2602) — port Sonnet's lookup/derived success procedures to the flash path.

The Sonnet dev gold100 (SOT-2628 / SOT-2630 dossier ``docs/ai/sonnet_trace_dossier.md``) reached five
lookup/derived questions the flash champion abstained on (BUDGET_EXHAUSTED / 探索不発): idx 5 / 53 / 96 /
36 / 79. Every one is solvable with the *same* tool set — the gap is 誘導・手順 (which document, which
extra hop, how to combine), not model capability. This module ports each question's **procedure** (not
its answer) as an advisory HINT appended to the generation agent's preamble, plus a companion
deterministic tool-gap fix in :mod:`src.rag.tools.compute_sandbox` (compute could not open a decrypted
in-memory xlsx — idx79's true blocker).

Design invariants (mirroring the sibling opt-ins — evidence_packet / condition_prefill / pot_lane):

* **Opt-in, byte-identical OFF.**  :func:`enabled` (``RAG_G2_LOOKUP_PORT``) gates the whole thing; the
  default-OFF serve path never builds a directive and injects nothing, so the champion answer path is
  byte-identical.
* **No firing-condition relaxation, no gold values.**  Detection is regex over the *question text only*.
  A question that matches no G2 archetype gets **no** directive and follows the ordinary path (the
  SOT-2601 net12 failure axis — 発火条件の緩和 — is deliberately avoided). The directives carry procedure
  guidance only; they never inject a corpus fact, a document's contents, or the answer.
* **Precision-preserving.**  The directives are advisory (appended to the prompt; final tool choice stays
  with the model) and each is scoped tightly to its archetype's question shape so it never fires on an
  unrelated question (the mandatory focused-gate sentinels do not match any detector). The two
  calculation archetypes (idx36 / idx79) route their arithmetic through the PoT lane
  (``verify_formula``: binder → 制限AST → Decimal → 独立検算, EXEC_MATCH only commit) — never 暗算 commit.

Each detector corresponds to one dossier archetype:

  * :func:`is_hyperparam_code_fallback`   idx5  — hyperparameter of the best model, 報告書に無ければ
    modeling ソースの既定値へフォールバック.
  * :func:`is_legend_color_count`         idx53 — count selected features by 凡例(色分け)→カテゴリ写像,
    分類数の総和＝総数 を整合検証.
  * :func:`is_cross_sheet_hop`            idx96 — CP→MS→タスク の複数シート横断参照.
  * :func:`is_two_doc_metric_diff`        idx36 — 中間/最終 の2文書から同一指標を抽出し差の絶対値（PoT lane）.
  * :func:`is_instructed_fallback_aggregate` idx79 — 鍵付きファイルの指示付きフォールバック＋担当者別集計
    （PoT lane）.
"""
from __future__ import annotations

import os
import re
from typing import Any

_ON = {"1", "true", "yes", "on"}


def enabled() -> bool:
    """True when the G2 lookup/derived procedure port is active (default OFF — opt-in).

    Gated by ``RAG_G2_LOOKUP_PORT`` exactly like the sibling opt-ins (RAG_EVIDENCE_PACKET /
    RAG_CONDITION_PREIR / RAG_POT_HARD_LANE): default-OFF means the serve path injects nothing extra, so
    the champion answer path is byte-identical.
    """
    return os.getenv("RAG_G2_LOOKUP_PORT", "0").strip().lower() in _ON


# --------------------------------------------------------------------------- archetype detectors
# A model hyperparameter name (the question names the parameter literally — idx5 "max_depth"). Kept to
# canonical sklearn / GBM hyperparameter tokens so an ordinary numeric question cannot match.
_HYPERPARAM = re.compile(
    r"max_depth|n_estimators|learning_rate|max_leaf_nodes|min_samples_(?:leaf|split)|min_child_weight"
    r"|num_leaves|subsample|colsample|reg_(?:alpha|lambda)|n_neighbors|hidden_layer_sizes|max_iter"
    r"|n_clusters|random_state", re.I)
_BEST_MODEL = re.compile(r"最良モデル|ベストモデル|最も(?:良い|精度|高い)|採用(?:した|している)?モデル|最終報告")


def is_hyperparam_code_fallback(question: str) -> bool:
    """idx5 — a hyperparameter of the best/adopted model (value may live only in the modeling code)."""
    q = question or ""
    return bool(_HYPERPARAM.search(q)) and (bool(_BEST_MODEL.search(q)) or "パラメータ" in q)


_FEATURE = re.compile(r"選択特徴量|特徴量")
_COUNT = re.compile(r"いくつ|個数|何個|何件|数は")
# A feature *category* tag the legend colour-codes (idx53 "ENG-FT"). Requiring a category token keeps this
# off plain "特徴量はいくつ" questions that have no legend classification.
_FEATURE_CATEGORY = re.compile(r"ENG[-‐]?FT|ENG\b|エンジニアリング特徴量|エンジニアリング|原特徴量|派生特徴量")


def is_legend_color_count(question: str) -> bool:
    """idx53 — count the selected features that fall in one legend-colour category."""
    q = question or ""
    return bool(_FEATURE.search(q) and _COUNT.search(q) and _FEATURE_CATEGORY.search(q))


_CHECKPOINT = re.compile(r"チェックポイント|checkpoint|\bCP\d|マイルストーン")
_TASK_REF = re.compile(r"タスク(?:ID|Id|ＩＤ)?|タスク")
_RELATED = re.compile(r"関連|関する|紐づく|紐付く|対応(?:する|した)")


def is_cross_sheet_hop(question: str) -> bool:
    """idx96 — a checkpoint→milestone→task cross-sheet hop in a schedule workbook."""
    q = question or ""
    return bool(_CHECKPOINT.search(q) and _TASK_REF.search(q) and _RELATED.search(q))


_INTERIM = re.compile(r"中間報告|中間時点|中間")
_FINAL = re.compile(r"最終報告|最終時点|最終")
_DIFF = re.compile(r"差(?:を|は|の)?|差分")
_METRIC = re.compile(r"F1|Ｆ1|スコア|精度|実測値|Accuracy|AUC|Macro")


def is_two_doc_metric_diff(question: str) -> bool:
    """idx36 — the absolute difference of the same metric between the interim and final reports."""
    q = question or ""
    return bool(_INTERIM.search(q) and _FINAL.search(q) and _DIFF.search(q) and _METRIC.search(q))


_EFFORT = re.compile(r"想定工数|工数")
_ASSIGNEE = re.compile(r"担当者|担当")
_PER_TASK = re.compile(r"÷|割|当たり|あたり|per|１タスク|1タスク")
_MAX = re.compile(r"最も(?:大きい|多い|高い)|最大")
# The generic "instructed fallback" clause: <lock/encryption> ... 場合 ... を確認/参照 (idx79 names 社内管理).
_INSTRUCTED_FALLBACK = re.compile(
    r"(?:鍵|パスワード|暗号)[^。]*?場合[^。]*?(?:を確認|を参照|に確認|を見て|を調べ)")


def instructed_fallback_clause(question: str) -> str | None:
    """Return the question's own «…鍵…場合…確認» instruction clause verbatim (or None).

    Generic: the clause is lifted straight from the question text (never a corpus fact), so the directive
    can tell the agent to follow *the question's* alternate-location instruction on a decrypt failure.
    """
    m = _INSTRUCTED_FALLBACK.search(question or "")
    return m.group(0) if m else None


def is_instructed_fallback_aggregate(question: str) -> bool:
    """idx79 — a per-assignee effort aggregate (想定工数÷担当タスク数 の最大) on a locked planning folder."""
    q = question or ""
    return bool(_EFFORT.search(q) and _ASSIGNEE.search(q) and _PER_TASK.search(q) and _MAX.search(q))


# --------------------------------------------------------------------------- directives (procedure only)
_H_HYPERPARAM = (
    "【手順ヒント: モデルのハイパーパラメータ照会】まず該当案件の報告書本文・付録・会議録を精読し、対象の"
    "ハイパーパラメータ値の明記を探す。本文に数値の記載が無い場合に限り、canonical_route→read_office で当該"
    "案件の modeling コード(例 src/modeling.py)や設定/上書き config を開き、そのモデル構築部の該当パラメータの"
    "設定値(無ければ既定値)を根拠付きで採用する。上書き config があればそれを優先し、値の推測はしない。")
_H_LEGEND = (
    "【手順ヒント: 凡例色による特徴量の分類カウント】該当スライド/資料を read_office で読み、選択特徴量の凡例"
    "(色分け=カテゴリ)に基づき各特徴量をカテゴリへ写像して数える。回答前に『各カテゴリの分類数の総和＝選択"
    "特徴量の総数』が成立することを必ず整合検証し、不一致なら再分類してから回答する。")
_H_CROSS_SHEET = (
    "【手順ヒント: スケジュールの複数シート横断参照】schedule/スケジュール系 xlsx は単一シートで打ち切らず"
    "複数シートを跨いで解決する。チェックポイント(CP)はマイルストーン(MS)を介してタスクへ紐づくことがある"
    "ため、CP→MS→タスク の多段参照を最後までたどってから回答する。")
_H_TWO_DOC_DIFF = (
    "【手順ヒント: 2文書間の指標差】中間報告と最終報告の双方から同一指標(F1 等)の実測値をそれぞれ根拠付きで"
    "抽出し、差の絶対値を求める。本文に改善量の記載があれば照合して検算する。差の計算は暗算で commit せず、"
    "利用可能なら verify_formula(PoT lane)で2つの operand を source 付きで束縛し EXEC_MATCH のみを commit する"
    "(利用不可なら compute で再検算する)。")
_H_INSTRUCTED = (
    "【手順ヒント: 指示付きフォールバック＋担当者別集計】質問文が『…の場合は…を確認』の代替所在の指示を含む"
    "場合、暗号化/鍵付きファイルの decrypt を試み、失敗したらその指示に従って代替の所在(例: 社内管理)を"
    "確認して読み取る。担当者別のタスク数は確定した表から集計し(担当実績0件の担当者は除外)、想定工数÷担当"
    "タスク数 を算出する。この除算も利用可能なら verify_formula(PoT lane)で束縛し EXEC_MATCH のみ commit する。")


def _archetypes(question: str) -> list[str]:
    """The list of G2 archetype keys the question matches (0..n; in practice at most one)."""
    fired: list[str] = []
    if is_hyperparam_code_fallback(question):
        fired.append("hyperparam_code_fallback")
    if is_legend_color_count(question):
        fired.append("legend_color_count")
    if is_cross_sheet_hop(question):
        fired.append("cross_sheet_hop")
    if is_two_doc_metric_diff(question):
        fired.append("two_doc_metric_diff")
    if is_instructed_fallback_aggregate(question):
        fired.append("instructed_fallback_aggregate")
    return fired


_DIRECTIVE_BY_KEY = {
    "hyperparam_code_fallback": _H_HYPERPARAM,
    "legend_color_count": _H_LEGEND,
    "cross_sheet_hop": _H_CROSS_SHEET,
    "two_doc_metric_diff": _H_TWO_DOC_DIFF,
    "instructed_fallback_aggregate": _H_INSTRUCTED,
}


def port_directive(question: str, *, contract: str | None = None) -> tuple[str | None, dict[str, Any]]:
    """Build the advisory G2 procedure directive for ``question`` (or ``None`` when nothing matches).

    Returns ``(directive_or_None, telemetry)``.  ``telemetry`` is the SOT-2629-style per-question
    intervention record: ``{"archetypes": [...], "fired": bool, "instructed_fallback": bool}`` — a key is
    recorded by the caller only when the flag is ON, and ``fired`` is False when the flag was ON but the
    question matched no archetype (an ON-but-idle intervention, distinguishable from OFF).  Never injects a
    corpus fact or the answer — procedure guidance only.
    """
    fired = _archetypes(question or "")
    tel: dict[str, Any] = {
        "archetypes": list(fired),
        "fired": bool(fired),
        "instructed_fallback": False,
    }
    if not fired:
        return None, tel
    parts = [_DIRECTIVE_BY_KEY[k] for k in fired]
    # For the instructed-fallback archetype, echo the question's own alternate-location instruction so the
    # directive is concrete (the clause is from the question text, not a corpus fact).
    if "instructed_fallback_aggregate" in fired:
        clause = instructed_fallback_clause(question)
        if clause:
            tel["instructed_fallback"] = True
            parts.append(f"(質問文の指示: 「{clause}」に従うこと。)")
    directive = "\n\n".join(parts)
    return directive, tel
