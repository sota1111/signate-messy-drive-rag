"""SOT-2693 — 書式付き数値系列の上昇率レーン（cycle8 C4, idx17）の offline テスト（LLM/corpus 不要）。

決定論束縛の規律を合成ストアで固定する: OFF ⇒ None（byte-identical）、idx17=会議録(MM)系列の 黄×赤 数値を
日付順に並べた上昇率、上昇率キュー/黄×赤述語/系列種別/案件束縛が揃わなければ defer。
"""
from __future__ import annotations

import pytest

from src.rag.agent import format_series_lane as L

_AYM = "青葉与信マネジメント株式会社"

_Q17 = ("AYMのMMにおいて、黄色ハイライトかつREDになっている数値を対象に、最初のMMから最後のMMまでの"
        "上昇率を計算してください。上昇率は （最後の値 - 最初の値） / 最初の値 × 100 で求め、"
        "小数第2位まで答えてください。")


def _series():
    return {
        "project": _AYM, "series_type": "会議録",
        "docs": [
            {"date": "2025-04-09", "doc_name": "会議録_2025-04-09.pdf", "events": []},
            {"date": "2025-04-29", "doc_name": "会議録_2025-04-29.pdf",
             "events": [{"raw": "0.589", "value": 0.589, "loc": "ページ2"}]},
            {"date": "2025-05-27", "doc_name": "会議録_2025-05-27.pdf",
             "events": [{"raw": "0.602", "value": 0.602, "loc": "ページ2"}]},
        ],
    }


@pytest.fixture()
def wired(monkeypatch):
    monkeypatch.setenv("RAG_FORMAT_SERIES", "1")
    monkeypatch.setattr(L, "_bind_project", lambda q: _AYM)
    monkeypatch.setattr(L._store, "series_for",
                        lambda project, stype, path=None: _series() if stype == "会議録" else None)
    return L


def test_off_is_none(monkeypatch):
    monkeypatch.delenv("RAG_FORMAT_SERIES", raising=False)
    monkeypatch.setattr(L, "_bind_project", lambda q: _AYM)
    monkeypatch.setattr(L._store, "series_for", lambda *a, **k: _series())
    assert L.resolve(_Q17) is None  # OFF ⇒ byte-identical fallback


def test_idx17_rise(wired):
    res = wired.resolve(_Q17)
    assert res is not None
    assert res["value"] == "2.21"  # (0.602 - 0.589) / 0.589 * 100 = 2.2071 → 2.21
    ev = res["evidence"]
    assert ev["first"]["value"] == 0.589 and ev["last"]["value"] == 0.602
    assert ev["series_type"] == "会議録"


def test_requires_rate_cue(wired):
    q = "AYMのMMで黄色ハイライトかつREDの数値を抜き出してください。"  # 上昇率キュー無し
    assert wired.resolve(q) is None


def test_requires_highlight_predicate(wired):
    q = "AYMのMMにおける数値の上昇率を小数第2位まで求めてください。"  # 黄×赤 述語無し
    assert wired.resolve(q) is None


def test_requires_series_binding(monkeypatch, wired):
    # 系列種別トークン（MM/会議録）を含まない ⇒ defer
    q = "AYMにおいて黄色ハイライトかつREDの数値の上昇率を小数第2位まで求めてください。"
    assert wired.resolve(q) is None


def test_single_doc_defers(monkeypatch):
    monkeypatch.setenv("RAG_FORMAT_SERIES", "1")
    monkeypatch.setattr(L, "_bind_project", lambda q: _AYM)
    one = {"project": _AYM, "series_type": "会議録",
           "docs": [{"date": "2025-04-29", "doc_name": "x.pdf",
                     "events": [{"raw": "0.589", "value": 0.589}]}]}
    monkeypatch.setattr(L._store, "series_for", lambda *a, **k: one)
    assert L.resolve(_Q17) is None  # 対象文書が1つ ⇒ 上昇率を組めない


def test_decimals_from_question(wired):
    q = _Q17.replace("小数第2位", "小数第4位")
    res = wired.resolve(q)
    assert res is not None and res["value"] == f"{(0.602 - 0.589) / 0.589 * 100:.4f}"
