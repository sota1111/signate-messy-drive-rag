"""書式付き数値系列の上昇率レーン（SOT-2693 / cycle8 C4, idx17）.

cycle8 abstain の「文書横断の書式付き数値系列カバレッジ欠落」（idx17）を、build 時に焼いた
:mod:`src.rag.index.format_series_store` を serve 時に **質問非依存に決定論束縛** して回収する。
新しい build は足さず、事前計算済みの成果物を読むだけ（Gemini 呼び出しゼロ・$0）。

* **idx17 上昇率**（AYM＝青葉与信マネジメントの MM＝会議録系列）— 系列の各文書で 黄色ハイライト×赤字
  の数値（対象）を日付順に並べ、最初 → 最後の上昇率 ``(最後 − 最初) / 最初 × 100`` を小数第2位で返す。
  例: 会議録 2025-04-29 (0.589) → 2025-05-27 (0.602) ⇒ ``(0.602 − 0.589) / 0.589 × 100 = 2.21``。

precision-first: 系列・対象値が一意に束縛できる時だけ確定し、少しでも曖昧なら ``None`` → LLM ループへ委譲。
``RAG_FORMAT_SERIES`` 既定 OFF ⇒ :func:`resolve` は None を返し serve path は byte-identical。
RAG_FACT_LAYER の下位レーンとして fact_layer.resolve の末尾に後置される（投資者ツールサーフェスは非改変）。
"""
from __future__ import annotations

import os
import re
import unicodedata
from typing import Any

from src.rag.index import format_series_store as _store
from src.rag.tools import contract as _contract

_ON = {"1", "true", "yes", "on"}

# 質問が指す系列種別（コーパスの文書種別語 → ストアの series_type）。社内用語集の略号
# （会議録=MM / 報告資料=RP）に対応。gold ではなくコーパス語彙。
_SERIES_TOKENS: tuple[tuple[str, str], ...] = (
    ("会議録", "会議録"),
    ("議事録", "会議録"),
    ("報告資料", "報告資料"),
)
# ascii 略号は語境界で厳密照合（短いので誤マッチ防止）。
_ABBREV_TOKENS: tuple[tuple[re.Pattern[str], str], ...] = (
    (re.compile(r"(?<![A-Za-z])MM(?![A-Za-z])"), "会議録"),
    (re.compile(r"(?<![A-Za-z])RP(?![A-Za-z])"), "報告資料"),
)

# 上昇率/変化率を問う語（少なくとも1つ必須）。
_RATE_CUES = ("上昇率", "変化率", "増加率", "伸び率")
_DECIMALS_RE = re.compile(r"小数第\s*([0-9０-９]+)\s*位")


def enabled() -> bool:
    """True when the serve path should consult the formatted-value series lane (default OFF)."""
    return os.getenv("RAG_FORMAT_SERIES", "0").strip().lower() in _ON


def _norm(value: Any) -> str:
    return unicodedata.normalize("NFKC", str(value or "")).replace(" ", "").replace("　", "").lower()


def _bind_project(question: str) -> str | None:
    try:
        from src.rag.extract import glossary
        return glossary.load().company_of(question)
    except Exception:  # noqa: BLE001
        return None


def _series_type(question: str) -> str | None:
    """質問から系列種別を一意に決める（複数該当は曖昧 ⇒ None）。"""
    hits: set[str] = set()
    qn = unicodedata.normalize("NFKC", str(question or ""))
    for tok, stype in _SERIES_TOKENS:
        if tok in qn:
            hits.add(stype)
    for rx, stype in _ABBREV_TOKENS:
        if rx.search(qn):
            hits.add(stype)
    return next(iter(hits)) if len(hits) == 1 else None


def _decimals(question: str) -> int:
    m = _DECIMALS_RE.search(question)
    if not m:
        return 2
    return int(unicodedata.normalize("NFKC", m.group(1)))


def _single_event_value(doc: dict[str, Any]) -> float | None:
    """文書の対象値が一意（黄×赤 数値がちょうど1つ）ならその値、そうでなければ None。"""
    events = doc.get("events") or []
    vals = [e.get("value") for e in events if isinstance(e.get("value"), (int, float))]
    return float(vals[0]) if len(vals) == 1 else None


def resolve(question: str) -> dict[str, Any] | None:
    """系列の上昇率を焼き込みストアで一意束縛できる時だけ contract を返す（fail-open・OFF なら None）。"""
    if not enabled():
        return None
    try:
        qn = _norm(question)
        # 上昇率/変化率 を問うている？
        if not any(_norm(c) in qn for c in _RATE_CUES):
            return None
        # 黄色ハイライト × 赤（RED）の複合述語が明示されている？
        if "黄" not in qn or not ("赤" in qn or "red" in qn):
            return None
        stype = _series_type(question)
        if stype is None:
            return None
        project = _bind_project(question)
        if not project:
            return None
        rec = _store.series_for(project, stype)
        if not rec:
            return None
        # 対象値（黄×赤 数値が一意）を持つ文書を日付順に。ちょうど1つの系列として first/last を取る。
        used = [(d, _single_event_value(d)) for d in (rec.get("docs") or [])]
        used = [(d, v) for d, v in used if v is not None]
        if len(used) < 2:
            return None
        first_doc, first_val = used[0]
        last_doc, last_val = used[-1]
        if first_val == 0:
            return None
        decimals = _decimals(question)
        rise = (last_val - first_val) / first_val * 100.0
        text = f"{round(rise, decimals):.{decimals}f}"
        evidence = {
            "store": "format_series_store", "provenance": "precomputed (question-independent)",
            "project": rec.get("project"), "series_type": stype,
            "first": {"date": first_doc.get("date"), "doc": first_doc.get("doc_name"), "value": first_val},
            "last": {"date": last_doc.get("date"), "doc": last_doc.get("doc_name"), "value": last_val},
            "n_series_docs": len(used), "decimals": decimals,
            "formula": "(last - first) / first * 100",
        }
        method = {"engine": "format_series_store", "contract": "deterministic",
                  "selection": "yellow_red_series_rise", "naturalize": False,
                  "verified_operand": True, "confidence": 1.0}
        result = {"value": text, "evidence": evidence, "method": method}
        if _contract.is_contract(result) and result.get("value") is not None:
            return _contract.ensure_contract(result)
    except Exception:  # noqa: BLE001 — a broken lane must fall back, never break the answer path
        return None
    return None
