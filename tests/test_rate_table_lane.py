"""SOT-2693 — 税率表 帯別差分レーン（cycle8 C4, idx48）の offline テスト（LLM/corpus 不要）。

決定論束縛の規律を合成ストアで固定する: OFF ⇒ None（byte-identical）、idx48=価格帯×現行/新税率表の
|新−現行| argmin、同点は defer、最大/最小の指定が一意でなければ defer。
"""
from __future__ import annotations

import pytest

from src.rag.agent import rate_table_lane as L

_Q48 = ("青嶺不動産アセットマネジメントのニューヨーク不動産市場の最新動向調査.pdfにおいて、提案されている"
        "マンション税の新税率のうち、現行税率からの絶対値の増加が最も小さい価格帯はどこですか。")


def _tax_bands():
    rows = [
        ("50万ドル超 - 100万ドル以下", 0.0, 1.425),   # diff 1.425
        ("100万ドル超 - 500万ドル以下", 1.25, 1.425),  # diff 0.175 → unique argmin
        ("500万ドル超 - 1,000万ドル以下", 2.25, 3.9),  # diff 1.65
        ("2,500万ドル超", 3.9, 6.0),                    # diff 2.10 → unique argmax
    ]
    bands = [{"band": b, "current_repr": c, "new_repr": n, "abs_increase": abs(n - c)}
             for b, c, n in rows]
    diffs = [x["abs_increase"] for x in bands]
    return {
        "caption": "物件価格帯 現行税率 提案されている新税率", "n_bands": len(bands), "bands": bands,
        "argmin_band": bands[min(range(len(bands)), key=lambda i: diffs[i])]["band"],
        "argmin_unique": sum(1 for d in diffs if abs(d - min(diffs)) < 1e-9) == 1,
        "argmax_band": bands[max(range(len(bands)), key=lambda i: diffs[i])]["band"],
        "argmax_unique": sum(1 for d in diffs if abs(d - max(diffs)) < 1e-9) == 1,
    }


def _record():
    return {"doc_id": "x/ニューヨーク不動産市場の最新動向調査.pdf", "project": "青嶺",
            "doc_name": "ニューヨーク不動産市場の最新動向調査.pdf", "rate_tables": [_tax_bands()]}


@pytest.fixture()
def wired(monkeypatch):
    monkeypatch.setenv("RAG_RATE_TABLE", "1")
    monkeypatch.setattr(L._store, "docs_for", lambda hint, path=None: [_record()])
    return L


def test_off_is_none(monkeypatch):
    monkeypatch.delenv("RAG_RATE_TABLE", raising=False)
    monkeypatch.setattr(L._store, "docs_for", lambda *a, **k: [_record()])
    assert L.resolve(_Q48) is None  # OFF ⇒ byte-identical fallback


def test_idx48_argmin(wired):
    res = wired.resolve(_Q48)
    assert res is not None
    assert res["value"] == "100万ドル超 - 500万ドル以下"
    assert res["evidence"]["selection"] == "argmin"


def test_argmax_variant(wired):
    q = _Q48.replace("最も小さい", "最も大きい")
    res = wired.resolve(q)
    assert res is not None and res["evidence"]["selection"] == "argmax"


def test_non_unique_argmin_defers(monkeypatch):
    monkeypatch.setenv("RAG_RATE_TABLE", "1")
    rec = _record()
    rec["rate_tables"][0]["argmin_unique"] = False
    monkeypatch.setattr(L._store, "docs_for", lambda *a, **k: [rec])
    assert L.resolve(_Q48) is None  # 同点 ⇒ 決定論的に一意でない ⇒ defer


def test_requires_increase_and_band_cue(wired):
    q = "青嶺不動産アセットマネジメントの新税率は何％ですか。"  # 帯/増加キュー無し
    assert wired.resolve(q) is None


def test_ambiguous_min_max_defers(wired):
    q = _Q48.replace("最も小さい", "小さいか大きい")  # 最小と最大の双方が該当
    assert wired.resolve(q) is None
