from pathlib import Path

from src.rag.agent import contact_master_lane as lane
from src.rag.index import contact_master_store as store


# --- signature-block parsing (unit): 会社名/部署名/主担当者/役職 を甲=client・乙=vendor 別に抽出する。
_AOBM_CONTRACT = """データ分析業務委託契約書
株式会社青葉バイオメディカル機器（以下「甲」という。）と株式会社データアステル（以下「乙」という。）は、
1. 当事者
（1）甲
会社名：株式会社青葉バイオメディカル機器
部署名：人事本部 人材戦略部
主担当者：山田 太一
役職：人材戦略部長
（2）乙
会社名：株式会社データアステル
部署名：データサイエンス部
エグゼクティブスポンサー：中村 誠
2. 目的
13. 署名欄
甲
株式会社青葉バイオメディカル機器
人事本部 人材戦略部
人材戦略部長　山田 太一
"""


def test_party_fields_extracts_signature_block_role():
    fields = store._party_fields(_AOBM_CONTRACT)
    # 甲(client)側は 役職：人材戦略部長 を構造化フィールドとして持つ（会議録の展開形ではない権威ソース）。
    assert fields["client"]["role"][0] == "人材戦略部長"
    assert fields["client"]["department"][0] == "人事本部 人材戦略部"
    assert fields["client"]["contact"][0] == "山田 太一"
    # 乙(vendor)側は 会社名/部署名 のみ（役職/主担当者 の構造化フィールドは無い）。
    assert fields["vendor"]["company"][0] == "株式会社データアステル"
    assert "role" not in fields["vendor"]
    # 本文中の「（以下「甲」という。）」はマーカー扱いしない（長い行は $ 終端で除外）。


def test_party_fields_ignores_labels_before_party_marker():
    text = "会社名：無所属\n（1）甲\n役職：部長\n"
    fields = store._party_fields(text)
    assert fields["client"].get("role", ("",))[0] == "部長"
    assert "company" not in fields["client"]  # party マーカー前の 会社名 は無帰属


def _rows():
    def cell(v):
        return {"value": v, "source": {"doc_id": "fixture"}}

    def missing():
        return {"value": None, "source": None, "reason": "no field"}

    return [
        {"case_id": "株式会社青葉バイオメディカル機器", "abbrev": "AOBM", "operands": {
            "client_company": cell("株式会社青葉バイオメディカル機器"),
            "client_department": cell("人事本部 人材戦略部"),
            "client_contact": cell("山田 太一"),
            "client_role": cell("人材戦略部長"),
            "vendor_company": cell("株式会社データアステル"),
            "vendor_department": cell("データサイエンス部"),
            "vendor_contact": missing(), "vendor_role": missing(),
        }},
        {"case_id": "株式会社青嶺不動産アセットマネジメント", "abbrev": "AOMINE", "operands": {
            "client_company": missing(), "client_department": missing(),
            "client_contact": missing(), "client_role": missing(),
            "vendor_company": missing(), "vendor_department": missing(),
            "vendor_contact": missing(), "vendor_role": missing(),
        }},
    ]


def test_default_off(monkeypatch):
    monkeypatch.delenv("RAG_CONTACT_MASTER", raising=False)
    assert store.enabled() is False
    assert lane.resolve("青葉バイオメディカル機器のクライアントの主担当者の役職は何ですか。") is None
    assert lane.tool() is None


def test_store_roundtrip_is_sorted(tmp_path: Path):
    records = [store.ContactMaster(x["case_id"], x["abbrev"], x["operands"]) for x in reversed(_rows())]
    path = tmp_path / "contact.jsonl"
    assert store.write_store(records, path) == {"cases": 2}
    loaded = store.load(path)
    assert [x["case_id"] for x in loaded] == sorted(x["case_id"] for x in _rows())


def test_idx21_client_role_lane(monkeypatch):
    # idx21: 会議録の展開形ではなく契約書署名欄の 役職：人材戦略部長 を裸形式で返す。案件名は接頭辞
    # 「株式会社」を落としても束縛できる。
    monkeypatch.setenv("RAG_CONTACT_MASTER", "1")
    monkeypatch.setattr(store, "load", lambda path=None: _rows())
    res = lane.resolve("青葉バイオメディカル機器のクライアントの主担当者の役職は何ですか。")
    assert res["value"] == "人材戦略部長"
    assert res["method"]["selection"] == "party_contact_role"
    assert res["evidence"]["case"] == "株式会社青葉バイオメディカル機器"


def test_client_contact_and_department_lanes(monkeypatch):
    monkeypatch.setenv("RAG_CONTACT_MASTER", "1")
    monkeypatch.setattr(store, "load", lambda path=None: _rows())
    assert lane.resolve("青葉バイオメディカル機器のクライアントの主担当者は誰ですか。")["value"] == "山田 太一"
    assert lane.resolve("青葉バイオメディカル機器のクライアントの部署名は何ですか。")["value"] == "人事本部 人材戦略部"


def test_defers_without_explicit_party(monkeypatch):
    # 甲/乙(クライアント/受託) を明示しない質問は曖昧 ⇒ 委譲（precision-first）。
    monkeypatch.setenv("RAG_CONTACT_MASTER", "1")
    monkeypatch.setattr(store, "load", lambda path=None: _rows())
    assert lane.resolve("青葉バイオメディカル機器の主担当者の役職は何ですか。") is None


def test_defers_when_field_absent(monkeypatch):
    # 該当フィールドが store に無い（vendor 役職）/案件が束縛できない ⇒ 委譲（fail-closed）。
    monkeypatch.setenv("RAG_CONTACT_MASTER", "1")
    monkeypatch.setattr(store, "load", lambda path=None: _rows())
    assert lane.resolve("青葉バイオメディカル機器の受託者の役職は何ですか。") is None
    assert lane.resolve("クライアントの主担当者の役職は何ですか。") is None


def test_defers_on_ambiguous_case_bind(monkeypatch):
    # 同名接頭辞で複数案件が並ぶ（曖昧）ときは束縛しない。
    monkeypatch.setenv("RAG_CONTACT_MASTER", "1")
    rows = _rows()
    rows.append({"case_id": "株式会社青葉与信", "abbrev": "AOYO", "operands": {
        "client_role": {"value": "経理部長", "source": {"doc_id": "f"}}}})
    monkeypatch.setattr(store, "load", lambda path=None: rows)
    # 「青葉」だけでは 2 案件に一致するが、"青葉バイオメディカル機器" のフル社名一致で一意化される。
    assert lane.resolve("青葉のクライアントの主担当者の役職は？") is None
