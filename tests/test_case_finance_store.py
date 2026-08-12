from pathlib import Path

from src.rag.agent import case_finance_lane as lane
from src.rag.index import case_finance_store as store


def _cell(value):
    return {"value": value, "source": {"doc_id": "fixture"}}


def _rows():
    def row(case, abbrev, *, est, act, rate=25000, settlement="検収基準精算", payments=()):
        return {"case_id": case, "abbrev": abbrev, "operands": {
            "contract_type": _cell("time_and_materials"), "settlement_type": _cell(settlement),
            "estimated_effort_hours": _cell(est), "actual_effort_hours": _cell(act),
            "effort_variance": _cell(abs(est - act)), "time_rate_excl_tax": _cell(rate),
            "tax_rate": _cell(0.1), "rounding_unit_hours": _cell(0.5),
            "estimate_amount_incl_tax": _cell(3_740_000),
            "confirmed_amount_incl_tax": _cell(3_443_000),
            "payment_schedule": _cell(list(payments)),
        }}
    return [
        row("株式会社青葉バイオメディカル機器", "AOBM", est=170, act=156.5,
            payments=({"year": 2025, "month": 8, "amount_incl_tax": 3_740_000},)),
        row("株式会社青嶺不動産アセットマネジメント", "AOMINE", est=170, act=184.5,
            payments=({"year": 2025, "month": 9, "amount_incl_tax": 4_675_000},)),
        row("案件C", "C", est=100, act=101,
            payments=({"year": 2025, "month": 10, "amount_incl_tax": 11_412_500},)),
    ]


def test_default_off(monkeypatch):
    monkeypatch.delenv("RAG_CASE_FINANCE_STORE", raising=False)
    assert store.enabled() is False
    assert lane.resolve("AOBMの見込金額と確定金額") is None
    assert lane.tool() is None


def test_store_roundtrip_is_sorted(tmp_path: Path):
    records = [store.CaseFinance(x["case_id"], x["abbrev"], x["operands"]) for x in reversed(_rows())]
    path = tmp_path / "finance.jsonl"
    assert store.write_store(records, path) == {"cases": 3}
    loaded = store.load(path)
    assert [x["case_id"] for x in loaded] == sorted(x["case_id"] for x in _rows())


def test_deterministic_lanes(monkeypatch):
    monkeypatch.setenv("RAG_CASE_FINANCE_STORE", "1")
    monkeypatch.setattr(store, "load", lambda path=None: _rows())
    questions = [
        "AOBMにおいて、見込金額（税込）と確定金額（税込）の差を、ESTHとACTHの差で割った1時間あたりの減少金額を計算してください。",
        "2026年7月1日時点で存在する案件について、支払月ごとの精算総額が多い月を上位3つ、総額とあわせて答えてください。",
        "事後精算案件のうち、提案時の見積工数と最終報告で報告されている実績工数の乖離が最も大きい案件を主略称で挙げてください。",
        "AOMINEの契約条件において、契約単価が現状よりも2,000円高く、実績工数が11.2時間少なかった場合、税込請求金額は、実際の税込請求金額と比べていくら変動しますか。",
    ]
    assert [lane.resolve(q)["value"] for q in questions] == [
        "22,000円",
        "2025年10月:11,412,500円、2025年9月:4,675,000円、2025年8月:3,740,000円",
        "AOMINE",
        "79,200円増加",
    ]


def test_non_matching_questions_defer(monkeypatch):
    monkeypatch.setenv("RAG_CASE_FINANCE_STORE", "1")
    monkeypatch.setattr(store, "load", lambda path=None: _rows())
    assert lane.resolve("AOMINEの実績工数を教えて") is None
    assert lane.resolve("RATEの変更日はいつですか") is None
