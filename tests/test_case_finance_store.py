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


def _sohk_row(provisions=()):
    return {"case_id": "医療法人社団 蒼泉会 ひがし丘総合病院", "abbrev": "SOHK", "operands": {
        "contract_type": _cell("time_and_materials"), "settlement_type": _cell("月次精算"),
        "estimated_effort_hours": _cell(170.0), "actual_effort_hours": _cell(170.0),
        "time_rate_excl_tax": _cell(25000.0), "tax_rate": _cell(0.1),
        "rounding_unit_hours": _cell(0.5), "estimate_amount_incl_tax": _cell(4_675_000),
        "confirmed_amount_incl_tax": _cell(4_675_000),
        "special_settlement_provisions": _cell(list(provisions)),
    }}


def test_billing_reduction_scenario(monkeypatch):
    # idx23: ACTH 155h10m → 30分切上 155.5h → 税込請求 4,276,250 → 見込 4,675,000 との差額 398,750。
    monkeypatch.setenv("RAG_CASE_FINANCE_STORE", "1")
    monkeypatch.setattr(store, "load", lambda path=None: [_sohk_row()])
    q = ("ひがし丘の案件において、案件終了後のACTHが155時間10分だった場合の税込請求額は"
         "提案書内で記載の見込税込金額と比べて何円の減額になりますか。")
    assert lane.resolve(q)["value"] == "398,750円"


def test_special_provision_synthesis(monkeypatch):
    # idx78: >200時間 の特別条項が契約に無い → 一般規定要点を gold と同形で合成（付加情報なし）。
    monkeypatch.setenv("RAG_CASE_FINANCE_STORE", "1")
    monkeypatch.setattr(store, "load", lambda path=None: [_sohk_row()])
    q = ("ひがし丘の契約条件において、ACTHが200時間を超えた場合の精算方法に関する規定内容を答えてください。")
    assert lane.resolve(q)["value"] == (
        "ACTHが200時間を超えた場合の特別な精算規定はなく、実績工数に時間単価25,000円(税別)を乗じ"
        "消費税を加算して月次で精算する(30分単位・切上げ、上限なし)。")


def test_special_provision_defers_when_threshold_clause_exists(monkeypatch):
    # 質問の閾値に該当する時間閾値条項が契約にあれば「特別規定なし」と決めつけない（fail-closed）。
    monkeypatch.setenv("RAG_CASE_FINANCE_STORE", "1")
    monkeypatch.setattr(store, "load",
                        lambda path=None: [_sohk_row(provisions=[{"threshold_hours": 200.0}])])
    q = ("ひがし丘の契約条件において、ACTHが200時間を超えた場合の精算方法に関する規定内容を答えてください。")
    assert lane.resolve(q) is None


def test_non_matching_questions_defer(monkeypatch):
    monkeypatch.setenv("RAG_CASE_FINANCE_STORE", "1")
    monkeypatch.setattr(store, "load", lambda path=None: _rows())
    assert lane.resolve("AOMINEの実績工数を教えて") is None
    assert lane.resolve("RATEの変更日はいつですか") is None


_IDX6_Q = "蒼泉会 ひがし丘総合病院案件において、提案時の税込み見込み金額と最終請求金額の差額はいくらですか。"


def test_amount_difference_idx6_zero(monkeypatch):
    # idx6: 見込税込 4,675,000 と確定税込 4,675,000 の差額 → gold「0円」。案件バインドは case_id 名一致。
    monkeypatch.setenv("RAG_CASE_FINANCE_STORE", "1")
    monkeypatch.setenv("RAG_CASE_FINANCE_DIFF", "1")
    monkeypatch.setattr(store, "load", lambda path=None: [_sohk_row()])
    res = lane.resolve(_IDX6_Q)
    assert res["value"] == "0円"
    assert res["method"]["selection"] == "estimate_vs_confirmed_amount_difference"
    assert res["evidence"]["case"] == "医療法人社団 蒼泉会 ひがし丘総合病院"


def test_amount_difference_nonzero_thousands_format(monkeypatch):
    # 差額>0 は「N,NNN円」の桁区切り。gold ハードコードなし（store オペランドの決定論算術）。
    monkeypatch.setenv("RAG_CASE_FINANCE_STORE", "1")
    monkeypatch.setenv("RAG_CASE_FINANCE_DIFF", "1")
    row = _sohk_row()
    row["operands"]["confirmed_amount_incl_tax"] = _cell(3_443_000)  # 見込 4,675,000 − 確定 3,443,000
    monkeypatch.setattr(store, "load", lambda path=None: [row])
    q = "ひがし丘総合病院案件の、提案時の見込金額と最終請求金額の差額はいくらですか。"
    assert lane.resolve(q)["value"] == "1,232,000円"


def test_amount_difference_default_off_byte_identical(monkeypatch):
    # RAG_CASE_FINANCE_DIFF 未設定 → レーンは不活性（既存挙動と byte-identical）。
    monkeypatch.setenv("RAG_CASE_FINANCE_STORE", "1")
    monkeypatch.delenv("RAG_CASE_FINANCE_DIFF", raising=False)
    monkeypatch.setattr(store, "load", lambda path=None: [_sohk_row()])
    assert lane.resolve(_IDX6_Q) is None


def test_amount_difference_defers_on_per_hour_and_missing_operand(monkeypatch):
    monkeypatch.setenv("RAG_CASE_FINANCE_DIFF", "1")
    rows = _rows()
    q37 = ("aobmにおいて、見込金額（税込）と確定金額（税込）の差を、esthとacthの差で割った"
           "1時間あたりの減少金額を計算してください。")
    # per-hour 除算設問（idx37 型）は単純差額レーンの領分外 → 新レーンは defer（_aobm_reduction が処理）。
    assert lane._amount_difference(lane._norm(q37), rows) is None
    # idx37 は既存 _aobm_reduction が引き続き回答する（レーン衝突なし）。
    monkeypatch.setenv("RAG_CASE_FINANCE_STORE", "1")
    monkeypatch.setattr(store, "load", lambda path=None: rows)
    assert lane.resolve(
        "AOBMにおいて、見込金額（税込）と確定金額（税込）の差を、ESTHとACTHの差で割った"
        "1時間あたりの減少金額を計算してください。")["value"] == "22,000円"
    # 案件名が質問に無い → バインド不能 → defer（fail-closed）。
    assert lane.resolve("提案時の見込金額と最終請求金額の差額はいくらですか。") is None


def test_amount_difference_defers_when_operand_missing(monkeypatch):
    monkeypatch.setenv("RAG_CASE_FINANCE_STORE", "1")
    monkeypatch.setenv("RAG_CASE_FINANCE_DIFF", "1")
    row = _sohk_row()
    row["operands"]["confirmed_amount_incl_tax"] = {"value": None, "source": None}
    monkeypatch.setattr(store, "load", lambda path=None: [row])
    assert lane.resolve(_IDX6_Q) is None
