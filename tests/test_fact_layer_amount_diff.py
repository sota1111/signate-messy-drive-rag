"""SOT-2650 — amount-difference enumeration lane (idx67 型) boundary tests.

「完了案件のうち APR-M2 該当で提案時金額と FR 時金額が異なる案件を略称ですべて」 is deterministic over the
case master ONLY under a full-coverage certificate (both amounts filled on every filtered-universe
member). These tests pin the fire shape, the certificate deferral, the empty-set answer, and the
precision-first extra-predicate deferrals — all against synthetic stores (no artifacts / network).
"""
from __future__ import annotations

import pytest

from src.rag.agent import fact_layer as fl

Q67 = ("完了案件のうち、社内管理のAPRでAPR-M2に該当する案件の中で、"
       "提案時金額とFR時の金額が異なる案件を案件略称ですべて挙げてください。")


def _cell(v, src="doc/提案書.pptx"):
    return {"value": v, "source": {"doc_id": src}}


def _rows(*, drop_amount_for=(), apr_none_for=(), amounts=None):
    """10-case synthetic store: 3 differing (AOSHIO/AOMINE/AOBM), rest equal; AYM is APR-M1."""
    base = {
        "KSS": (5775000, 5775000, "APR-M2"), "KAEDE": (3850000, 3850000, "APR-M2"),
        "MINAMINO": (3960000, 3960000, "APR-M2"), "SOHK": (4675000, 4675000, "APR-M2"),
        "TOTO": (4675000, 4675000, "APR-M2"), "AOMINE": (4675000, 5073750, "APR-M2"),
        "AOSHIO": (4675000, 5245000, "APR-M2"), "AOBM": (3740000, 3443000, "APR-M2"),
        "SHR": (7480000, 7480000, "APR-M2"), "AYM": (4620000, 4620000, "APR-M1"),
    }
    if amounts:
        base.update(amounts)
    rows = []
    for ab, (prop, fr, apr) in base.items():
        attrs = {"abbrev": _cell(ab), "status": _cell("完了"),
                 "apr_code": (_cell(apr) if ab not in apr_none_for
                              else {"value": None, "source": None, "reason": "x"}),
                 "proposal_amount_incl_tax": _cell(prop), "fr_amount_incl_tax": _cell(fr)}
        if ab in drop_amount_for:
            attrs["fr_amount_incl_tax"] = {"value": None, "source": None, "reason": "抽出できない"}
        rows.append({"case_id": f"案件{ab}", "abbrev": ab, "attributes": attrs})
    return rows


@pytest.fixture
def stores(monkeypatch):
    from src.rag.index import case_master
    monkeypatch.setenv("RAG_FACT_LAYER", "1")

    def install(rows):
        monkeypatch.setattr(case_master, "load", lambda path=None: rows)
    return install


def test_idx67_shape_fires_and_enumerates(stores):
    stores(_rows())
    out = fl.resolve(Q67, "full_enumeration")
    assert out is not None
    assert out["value"] == "AOMINE、AOSHIO、AOBM"
    ev = out["evidence"]
    assert ev["filter"] == {"apr_code": "APR-M2", "status": "完了"}
    assert ev["filtered"] == 9 and ev["matched"] == 3
    assert out["method"]["selection"] == "amount_diff_enumeration"
    assert out["method"]["verified_operand"] is True


def test_missing_amount_in_universe_defers(stores):
    # SHR's FR amount unfilled ⇒ the filtered universe is not fully covered ⇒ defer (never partial)
    stores(_rows(drop_amount_for={"SHR"}))
    assert fl.resolve(Q67, "full_enumeration") is None


def test_missing_apr_anywhere_defers(stores):
    stores(_rows(apr_none_for={"AYM"}))
    assert fl.resolve(Q67, "full_enumeration") is None


def test_all_equal_amounts_answers_none(stores):
    stores(_rows(amounts={"AOMINE": (4675000, 4675000, "APR-M2"),
                          "AOSHIO": (4675000, 4675000, "APR-M2"),
                          "AOBM": (3740000, 3740000, "APR-M2")}))
    out = fl.resolve(Q67, "full_enumeration")
    assert out is not None and out["value"] == "該当なし"


def test_extra_predicate_defers(stores):
    stores(_rows())
    q = Q67.replace("案件の中で", "医療案件の中で")
    assert fl.resolve(q, "full_enumeration") is None


def test_other_agg_shape_still_defers(stores):
    # 差-aggregation phrasing without the amount-diff shape must not fire this lane
    stores(_rows())
    q = "APR-M2に該当する案件の契約金額の差をすべて挙げてください。"
    assert fl.resolve(q, "full_enumeration") is None


def test_complement_phrasing_defers(stores):
    stores(_rows())
    q = Q67.replace("該当する案件", "該当する案件以外")
    assert fl.resolve(q, "full_enumeration") is None


def test_off_is_inert(monkeypatch, stores):
    stores(_rows())
    monkeypatch.delenv("RAG_FACT_LAYER", raising=False)
    assert fl.resolve(Q67, "full_enumeration") is None
