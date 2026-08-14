"""SOT-2713 — master-join store + lane (人物×案件×役割×座席 全数結合 決定論 lookup)。

質問非依存の結合テーブルを fixture で与え、idx13/43/46 の 3 質問形が決定論で直答し、fail-closed
（argmax 同率タイ・座席非一意・案件別名の曖昧）で棄権することを検証する（gold ハードコードなし）。
"""
from pathlib import Path

from src.rag.agent import master_join_lane as lane
from src.rag.index import master_join_store as store


def _case(case_id, abbrev, aliases, *, deposit=None, fixed=True, staff=None, client=None):
    return store.CaseRow(case_id=case_id, abbrev=abbrev, aliases=aliases,
                         deposit_incl_tax=deposit, fixed=fixed, staff=staff or {}, client_contact=client)


def _person(name, count, cases, seat):
    return store.PersonRow(name=name, case_count=count, cases=cases, seat=seat)


def _tables():
    cases = [
        _case("白峰信用リスク評価株式会社", "SHR", ["SHR", "白峰"], deposit=2_992_000,
              staff={"ES": "中村 誠", "DE": "斎藤 悠斗"}, client="白峰 花子"),
        _case("株式会社東都人材プラットフォーム", "TOTO", ["TOTO", "東都", "人材PF"],
              deposit=1_000_000, staff={"ES": "山田 直樹", "DE": "斎藤 悠斗"}, client="石川 直樹"),
        _case("株式会社青葉バイオメディカル機器", "AOBM", ["AOBM", "青葉バイオ", "青葉"],
              deposit=500_000, staff={"ES": "中村 誠", "DE": "斎藤 悠斗"}, client="山田 太一"),
        _case("青葉与信マネジメント株式会社", "AYM", ["AYM", "青葉与信", "青葉"],
              deposit=None, staff={"PM": "伊藤 翔太"}, client=None),
    ]
    persons = [
        _person("斎藤 悠斗", 3, [{"case_id": "白峰", "role_code": "DE"}],
                {"ext": "7104", "role": "DE", "pod": 2, "name": "斎藤"}),
        _person("中村 誠", 2, [{"case_id": "白峰", "role_code": "ES"}],
                {"ext": "7201", "role": "Exec", "pod": 3, "name": "中村"}),
        _person("山田 直樹", 1, [{"case_id": "東都", "role_code": "ES"}],
                {"ext": "7101", "role": "Exec", "pod": 2, "name": "山田"}),
        _person("松本 真央", 2, [{"case_id": "白峰", "role_code": "BA"}], None),  # 姓が座席表に無い
    ]
    return {"cases": [c.to_dict() for c in cases], "persons": [p.to_dict() for p in persons]}


# --------------------------------------------------------------------------- OFF / roundtrip
def test_default_off(monkeypatch):
    monkeypatch.delenv("RAG_MASTER_JOIN_LOOKUP", raising=False)
    assert store.enabled() is False
    assert lane.resolve("データアステル社の中でもっとも多くの案件にかかわっている人の内線番号を教えてください。") is None
    assert lane.tool() is None


def test_store_roundtrip_is_sorted(tmp_path: Path):
    cases = [_case("株式会社Z", "Z", ["Z"]), _case("株式会社A", "A", ["A"])]
    persons = [_person("鈴木 一郎", 1, [], None), _person("阿部 花子", 2, [], None)]
    path = tmp_path / "mj.jsonl"
    assert store.write_store(cases, persons, path) == {"cases": 2, "persons": 2}
    loaded = store.load(path)
    assert [c["case_id"] for c in loaded["cases"]] == ["株式会社A", "株式会社Z"]
    assert [p["name"] for p in loaded["persons"]] == ["鈴木 一郎", "阿部 花子"] or \
        sorted(p["name"] for p in loaded["persons"]) == ["鈴木 一郎", "阿部 花子"]


# --------------------------------------------------------------------------- idx13 / idx46 / idx43
def test_idx13_most_cases_person_ext(monkeypatch):
    monkeypatch.setenv("RAG_MASTER_JOIN_LOOKUP", "1")
    monkeypatch.setattr(store, "load", lambda path=None: _tables())
    res = lane.resolve("データアステル社の中でもっとも多くの案件にかかわっている人の内線番号を教えてください。")
    assert res["value"] == "7104"
    assert res["method"]["selection"] == "most_cases_person_ext"
    assert res["evidence"]["person"] == "斎藤 悠斗"


def test_idx46_max_deposit_role_ext(monkeypatch):
    monkeypatch.setenv("RAG_MASTER_JOIN_LOOKUP", "1")
    monkeypatch.setattr(store, "load", lambda path=None: _tables())
    res = lane.resolve("着手金が最も高い案件について、その案件のESの内線番号を教えてください。")
    assert res["value"] == "7201"  # 白峰(最高着手金) ES=中村 誠 → 7201
    assert res["method"]["selection"] == "max_deposit_role_ext"
    assert res["evidence"]["role"] == "ES"


def test_idx43_client_contact_fullname(monkeypatch):
    monkeypatch.setenv("RAG_MASTER_JOIN_LOOKUP", "1")
    monkeypatch.setattr(store, "load", lambda path=None: _tables())
    res = lane.resolve("東都のCTにおいて、甲側の主担当者をフルネームで教えてください。")
    assert res["value"] == "石川 直樹"
    assert res["method"]["selection"] == "client_contact_fullname"
    assert res["evidence"]["abbrev"] == "TOTO"


# --------------------------------------------------------------------------- fail-closed
def test_tie_on_most_cases_defers(monkeypatch):
    monkeypatch.setenv("RAG_MASTER_JOIN_LOOKUP", "1")
    t = _tables()
    t["persons"][1]["case_count"] = 3  # 斎藤 と同率タイ
    monkeypatch.setattr(store, "load", lambda path=None: t)
    assert lane.resolve("データアステル社の中でもっとも多くの案件にかかわっている人の内線番号を教えてください。") is None


def test_unseated_top_person_defers(monkeypatch):
    monkeypatch.setenv("RAG_MASTER_JOIN_LOOKUP", "1")
    t = _tables()
    t["persons"][0]["seat"] = None  # 最多関与者の座席が引けない
    monkeypatch.setattr(store, "load", lambda path=None: t)
    assert lane.resolve("データアステル社の中でもっとも多くの案件にかかわっている人の内線番号を教えてください。") is None


def test_ambiguous_alias_bind_defers(monkeypatch):
    # 「青葉」は AOBM/AYM 双方の別名 ⇒ 案件が一意に定まらず棄権（fail-closed）。
    monkeypatch.setenv("RAG_MASTER_JOIN_LOOKUP", "1")
    monkeypatch.setattr(store, "load", lambda path=None: _tables())
    assert lane.resolve("青葉のクライアントの主担当者をフルネームで教えてください。") is None


def test_unknown_role_defers(monkeypatch):
    # 役割を一意に判定できない（役割語なし）⇒ 棄権。
    monkeypatch.setenv("RAG_MASTER_JOIN_LOOKUP", "1")
    monkeypatch.setattr(store, "load", lambda path=None: _tables())
    assert lane.resolve("着手金が最も高い案件の内線番号を教えてください。") is None


def test_guards_do_not_fire(monkeypatch):
    monkeypatch.setenv("RAG_MASTER_JOIN_LOOKUP", "1")
    monkeypatch.setattr(store, "load", lambda path=None: _tables())
    # idx21: 役職 質問（フルネーム要求なし）は client_contact レーンを発火しない。
    assert lane.resolve("青葉バイオメディカル機器のクライアントの主担当者の役職は何ですか。") is None
    # idx31: 着手金でなく契約金額 → deposit レーン不発。
    assert lane.resolve("固定金額契約の中で、分析データ1行あたりの契約金額が最も高い案件を答えてください。") is None


def test_role_detection_unique():
    assert lane._detect_role(lane._norm("その案件のESの内線番号")) == "ES"
    assert lane._detect_role(lane._norm("データエンジニアの内線")) == "DE"
    assert lane._detect_role(lane._norm("内線番号を教えて")) is None
