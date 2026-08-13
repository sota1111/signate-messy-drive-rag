"""税率表 帯別差分レーン（SOT-2693 / cycle8 C4, idx48）.

cycle8 abstain の「PDF 価格帯×税率表への浅到達欠落」（idx48）を、build 時に焼いた
:mod:`src.rag.index.rate_table_store`（帯別 ``|新 − 現行|`` ＋ argmin/argmax 事前計算済み）を serve 時に
**質問非依存に決定論束縛** して回収する。新しい build は足さず、事前計算済みの成果物を読むだけ
（Gemini 呼び出しゼロ・$0）。

* **idx48**（青嶺不動産アセットマネジメント NY 不動産調査 PDF）— マンション税の価格帯別 現行/新税率表で、
  現行税率からの **絶対値の増加が最も小さい価格帯**（argmin）を返す。例: 100万ドル超-500万ドル以下。

precision-first: 表・argmin が一意に束縛できる時だけ確定し、少しでも曖昧なら ``None`` → LLM ループへ委譲。
``RAG_RATE_TABLE`` 既定 OFF ⇒ :func:`resolve` は None を返し serve path は byte-identical。
RAG_FACT_LAYER の下位レーンとして fact_layer.resolve の末尾に後置される（投資者ツールサーフェスは非改変）。
"""
from __future__ import annotations

import os
import re
import unicodedata
from typing import Any

from src.rag.index import rate_table_store as _store
from src.rag.tools import contract as _contract

_ON = {"1", "true", "yes", "on"}

# 税率表の帯別差分（増加）を問う語。
_TAX_CUES = ("税率", "税")
_BAND_CUES = ("価格帯", "帯")
_INCREASE_CUES = ("増加", "上昇", "引き上げ", "引上げ", "上がり", "増える", "差")
_MIN_CUES = ("最も小さい", "最小", "一番小さい", "最も少ない", "小さい")
_MAX_CUES = ("最も大きい", "最大", "一番大きい", "最も多い", "大きい")
# PDF ファイル名の手がかり（拡張子つき固有名を質問が含むとき優先束縛）。
_PDF_NAME_RE = re.compile(r"([^\s「」『』（）()]+\.pdf)", re.IGNORECASE)


def enabled() -> bool:
    """True when the serve path should consult the rate-table band-diff lane (default OFF)."""
    return os.getenv("RAG_RATE_TABLE", "0").strip().lower() in _ON


def _norm(value: Any) -> str:
    return unicodedata.normalize("NFKC", str(value or "")).replace(" ", "").replace("　", "").lower()


def _bind_project(question: str) -> str | None:
    try:
        from src.rag.extract import glossary
        return glossary.load().company_of(question)
    except Exception:  # noqa: BLE001
        return None


def _select_records(question: str) -> list[dict[str, Any]]:
    """質問から対象 PDF レコードを選ぶ（ファイル名 > 案件名）。"""
    qn = unicodedata.normalize("NFKC", str(question or ""))
    m = _PDF_NAME_RE.search(qn)
    if m:
        hit = _store.docs_for(m.group(1).rsplit(".", 1)[0])
        if hit:
            return hit
    project = _bind_project(question)
    if project:
        hit = _store.docs_for(project)
        if hit:
            return hit
    return []


def resolve(question: str) -> dict[str, Any] | None:
    """税率表の帯別 argmin/argmax を焼き込みストアで一意束縛できる時だけ contract を返す（OFF なら None）。"""
    if not enabled():
        return None
    try:
        qn = _norm(question)
        if not any(_norm(c) in qn for c in _TAX_CUES):
            return None
        if not any(_norm(c) in qn for c in _BAND_CUES):
            return None
        if not any(_norm(c) in qn for c in _INCREASE_CUES):
            return None
        want_min = any(_norm(c) in qn for c in _MIN_CUES)
        want_max = any(_norm(c) in qn for c in _MAX_CUES)
        # 最小/最大のどちらか一意に決まらなければ defer。
        if want_min == want_max:
            return None
        records = _select_records(question)
        # 率2列の税率表をちょうど1つ持つレコードに一意束縛。
        tables = [t for r in records for t in (r.get("rate_tables") or [])]
        if len(tables) != 1:
            return None
        table = tables[0]
        key_band = "argmin_band" if want_min else "argmax_band"
        key_uni = "argmin_unique" if want_min else "argmax_unique"
        if not table.get(key_uni):
            return None  # 同点は決定論的に一意でない ⇒ defer
        band = table.get(key_band)
        if not band:
            return None
        which = "argmin" if want_min else "argmax"
        evidence = {
            "store": "rate_table_store", "provenance": "precomputed (question-independent)",
            "selection": which, "caption": table.get("caption"),
            "n_bands": table.get("n_bands"),
            "bands": [{"band": b.get("band"), "current": b.get("current_repr"),
                       "new": b.get("new_repr"), "abs_increase": round(b.get("abs_increase", 0.0), 4)}
                      for b in (table.get("bands") or [])],
            "criterion": "|新税率 − 現行税率| の " + ("最小" if want_min else "最大"),
        }
        method = {"engine": "rate_table_store", "contract": "deterministic",
                  "selection": f"rate_band_{which}", "naturalize": False,
                  "verified_operand": True, "confidence": 1.0}
        result = {"value": band, "evidence": evidence, "method": method}
        if _contract.is_contract(result) and result.get("value") is not None:
            return _contract.ensure_contract(result)
    except Exception:  # noqa: BLE001 — a broken lane must fall back, never break the answer path
        return None
    return None
