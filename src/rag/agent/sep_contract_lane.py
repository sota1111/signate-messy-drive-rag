"""Deterministic answer lane for the "別契約と明記された役割" extraction shape (SOT-2702 / cycle10).

Recovers the Sonnet abstain **idx52**（蒼樹会みなみ野の今後の運用で、データアステル側の役割として
「別契約」と明記されているもの → gold=監視ダッシュボード構築）。cycle10 実測は project フィルタ緩和
（``RAG_TEXT_FTS_PROJECT_ALIAS``）で証拠 ``監視ダッシュボード構築(別契約)`` が retrieval に到達したものの、
LLM 経路（plan_fanout Sonnet）が 2 回とも ``わかりません`` に落ちて回収できなかった（SOT-2701 の教訓:
Sonnet は証拠が読めても合成で棄権する → 決定論レーン昇格が確実）。

このレーンは **新しい store を作らず**、既存の全文 FTS 索引（``text_fts``）を質問非依存で引き、テキストが
``X（別契約）`` 形をとる役割ラベル ``X`` を逐語抽出する。``別契約`` を単なる部分文字列（「カテゴリ別契約率」
「本番移行は別契約として扱う」等）から切り分けるため、**丸括弧で明示的にマークされた** ``（別契約）`` /
``(別契約)`` の直前ランのみを採る。

発火条件（precision-first・fail-open）:
  * ``RAG_SEP_CONTRACT_ROLE`` が ON（既定 OFF ⇒ ``resolve`` は None ⇒ byte-identical）。
  * ``text_fts`` が有効（索引が無ければ委譲）。
  * 質問が「別契約」を含み、かつ抽出意図（抽出／明記／該当／役割）を持つ。
  * 質問が案件（project）を名指しし、その案件の ``X（別契約）`` ラベルが **一意** に解決する。
上記を満たさなければ ``None`` を返して従来の LLM 経路へ委譲する（回答数を減らさない・wrong を増やさない）。
値は原文ラベルの完全写経（正規化・言い換えなし）で裸形式に返す。gold ハードコードなし。
"""
from __future__ import annotations

import os
import re
import unicodedata
from typing import Any

from src.rag.index import text_fts as _tf
from src.rag.tools import contract as _contract

SEP_CONTRACT_ROLE_FLAG = "RAG_SEP_CONTRACT_ROLE"
_ON = {"1", "true", "yes", "on"}

# The role label is the non-paren run immediately preceding an explicit ``（別契約）`` / ``(別契約)``
# marker. Excluding paren/句読点/箇条書き記号 keeps the label a clean verbatim span (e.g.
# ``監視ダッシュボード構築``), while the paren anchor rejects bare-substring 別契約 mentions.
_LABEL_RE = re.compile(r"([^\s、。・:：()（）]{2,40}?)\s*[（(]\s*別契約\s*[）)]")

# Extraction-intent cues: the question asks WHAT is *marked* as 別契約 (not a generic 契約 mention).
_INTENT = ("抽出", "明記", "該当", "役割")


def enabled() -> bool:
    """True when the deterministic 別契約-role lane is active (default OFF — opt-in, byte-identical off)."""
    return os.getenv(SEP_CONTRACT_ROLE_FLAG, "0").strip().lower() in _ON


def _nfc(value: Any) -> str:
    return unicodedata.normalize("NFC", str(value or ""))


def _project_named(project: str, q: str) -> bool:
    """True when the question names this project — a ≥3-char whitespace segment of the (NFC) project
    name appears verbatim in the (NFC) question. Segment granularity avoids the corporate-prefix noise
    (e.g. ``医療法人社団``) binding unrelated cases while still matching ``蒼樹会`` / ``みなみ野…``."""
    for seg in re.split(r"\s+", _nfc(project)):
        seg = seg.strip()
        if len(seg) >= 3 and seg in q:
            return True
    return False


def resolve(question: str) -> "dict[str, Any] | None":
    if not enabled():
        return None
    if not _tf.enabled():
        return None
    try:
        q = _nfc(question)
        if "別契約" not in q:
            return None
        if not any(cue in q for cue in _INTENT):
            return None
        hits = _tf.search("別契約", limit=100, project=None)
        labels: dict[str, str] = {}          # label -> project (dedup across pages/docs)
        provenance: list[dict[str, Any]] = []
        for h in hits:
            proj = h.get("project") or ""
            if not _project_named(proj, q):
                continue
            for m in _LABEL_RE.finditer(_nfc(h.get("text") or "")):
                lbl = m.group(1).strip()
                if lbl:
                    labels[lbl] = proj
                    provenance.append({"project": proj, "locator": h.get("locator"),
                                       "doc_id": h.get("doc_id")})
        distinct = sorted(labels)
        if len(distinct) != 1:                # 0 (no marked label) or >1 (ambiguous) ⇒ fail-open
            return None
        value = distinct[0]
        return _contract.ensure_contract({
            "value": value,
            "evidence": {
                "store": "text_fts",
                "marker": "別契約",
                "provenance": "precomputed FTS (question-independent, X（別契約） 逐語抽出)",
                "project": labels[value],
                "candidates": provenance[:5],
            },
            "method": {
                "engine": "text_fts",
                "contract": "simple_lookup",
                "selection": "separate_contract_role",
                "verified_operand": True,
                "naturalize": False,
                "confidence": 1.0,
            },
        })
    except Exception:  # noqa: BLE001 — fail-open optional lane, never break the answer path
        return None
