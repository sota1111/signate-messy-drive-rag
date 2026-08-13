"""SOT-2693 — 税率表ストア（cycle8 C4）の決定論ヘルパの offline テスト（LLM/corpus 不要）。"""
from __future__ import annotations

from types import SimpleNamespace

from src.rag.index import rate_table_store as S


def test_norm_band_wrap_fix():
    # PDF 折返し由来の空白（"以 下"）を除去し、範囲 ' - ' 区切りは保持（gold 書式）。
    assert S._norm_band("100 万ドル超 - 500 万ドル以 下") == "100万ドル超 - 500万ドル以下"
    assert S._norm_band("2,500 万ドル超") == "2,500万ドル超"
    assert S._norm_band("1,000 万ドル超 - 1,500 万ド ル以下") == "1,000万ドル超 - 1,500万ドル以下"


def test_parse_rate_single_and_range():
    assert S._parse_rate("1.425%") == {"raw": "1.425%", "value": 1.425}
    r = S._parse_rate("1.00% - 1.50%")
    assert r["low"] == 1.0 and r["high"] == 1.5 and r["mid"] == 1.25
    assert S._parse_rate("なし") is None
    assert S._rate_repr({"value": 3.9}) == 3.9
    assert S._rate_repr({"low": 1.0, "high": 1.5, "mid": 1.25}) == 1.25


def test_header_columns():
    hc = S._header_columns(["物件価格帯", "現行税率", "提案されている新税率"])
    assert hc == (0, 1, 2)
    assert S._header_columns(["名前", "住所"]) is None  # no rate columns


def test_finalize_argmin_unique():
    bands = [
        {"band": "A", "current": {"value": 0.0}, "new": {"value": 1.425}},
        {"band": "B", "current": {"low": 1.0, "high": 1.5, "mid": 1.25}, "new": {"value": 1.425}},
        {"band": "C", "current": {"value": 2.25}, "new": {"value": 3.675}},
    ]
    ref = SimpleNamespace(rel="x", project="p", name="n")
    t = S._finalize([{**b, "page": 1} for b in bands], "cap", ref)
    assert t["argmin_band"] == "B" and t["argmin_unique"] is True
    # A と C は同点(1.425) ⇒ argmax は非一意
    assert t["argmax_unique"] is False


def test_is_band_cell():
    assert S._is_band_cell("100万ドル超 - 500万ドル以下")
    assert not S._is_band_cell("現行税率")
