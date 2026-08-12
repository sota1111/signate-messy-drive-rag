"""SOT-2680 — スケジュール/ID/体制クロス参照ストア + serve レーンの offline テスト（LLM/network 不要）。

決定論の束縛規律を合成ストア行で固定する: OFF ⇒ None（byte-identical）、idx92=ID種別合計、
idx96=CP→タスク（MS チェーン展開）、idx94=MS×役職→タスク、idx72=役職→タスク数。曖昧束縛/番兵は defer。
純関数（expand_refs / name_key / チェックポイント解決）も単体で固定する。
"""
from __future__ import annotations

import pytest

from src.rag.index import schedule_store as S
from src.rag.agent import schedule_lane as L


# --------------------------------------------------------------------------- pure helpers
def test_expand_refs_range_and_list():
    assert S.expand_refs("T05〜T08") == ["T05", "T06", "T07", "T08"]
    assert S.expand_refs("T05, T06") == ["T05", "T06"]
    assert S.expand_refs("MS2 / T12") == ["MS2", "T12"]
    assert S.expand_refs("T05～T08") == ["T05", "T06", "T07", "T08"]  # full-width tilde
    assert S.expand_refs(None) == []


def test_name_key_strips_space():
    assert S.name_key("松本 真央") == "松本真央"
    assert S.name_key("斎藤　悠斗") == "斎藤悠斗"


def test_resolve_checkpoint_chains_via_milestone():
    milestones = {"MS2": {"id": "MS2", "rel_tasks": ["T05", "T06", "T07", "T08"]}}
    cp = {"id": "CP2", "rel": "MS2"}
    assert S._resolve_checkpoint_tasks(cp, milestones) == ["T05", "T06", "T07", "T08"]
    # direct task refs ∪ MS-derived, de-duped, order preserved
    cp2 = {"id": "CP9", "rel": "T01, MS2"}
    assert S._resolve_checkpoint_tasks(cp2, milestones) == ["T01", "T05", "T06", "T07", "T08"]


# --------------------------------------------------------------------------- lane binding
def _kaede():
    return {
        "project": "医療法人社団 恒一会 かえで総合病院",
        "schedule_file": "…/スケジュール.xlsx",
        "tasks": [{"id": f"T{n:02d}", "name": "", "owners": [], "owner_keys": [], "status": ""}
                  for n in range(1, 24)],
        "milestones": [{"id": f"MS{n}", "rel_tasks": []} for n in range(1, 8)],
        "ms_tasks": {},
        "checkpoints": [],
        "roster": [],
        "action_ids": [f"A{n:02d}" for n in range(1, 20)],
        "id_counts": {"task": 23, "milestone": 7, "action": 19, "total": 49},
    }


def _minamino():
    return {
        "project": "医療法人社団 蒼樹会 みなみ野女性医療センター",
        "tasks": [
            {"id": "T07", "owner_keys": ["鈴木美咲", "岡田佑樹"]},
            {"id": "T08", "owner_keys": ["鈴木美咲"]},
            {"id": "T09", "owner_keys": ["松本真央", "鈴木美咲"]},
        ],
        "milestones": [],
        "ms_tasks": {"MS3": ["T07", "T08", "T09"]},
        "checkpoints": [],
        "roster": [{"role": "ビジネスアナリスト", "name": "松本 真央", "name_key": "松本真央"},
                   {"role": "データエンジニア", "name": "岡田 佑樹", "name_key": "岡田佑樹"}],
        "action_ids": [],
        "id_counts": {"task": 3, "milestone": 1, "action": 0, "total": 4},
    }


def _aoba():
    return {
        "project": "青葉与信マネジメント株式会社",
        "tasks": [],
        "milestones": [{"id": "MS2", "rel_tasks": ["T05", "T06", "T07", "T08"]}],
        "ms_tasks": {"MS2": ["T05", "T06", "T07", "T08"]},
        "checkpoints": [{"id": "CP2", "content": "データ理解完了",
                         "related_task_ids": ["T05", "T06", "T07", "T08"]}],
        "roster": [],
        "action_ids": [],
        "id_counts": {"task": 23, "milestone": 10, "action": 5, "total": 38},
    }


def _kss():
    return {
        "project": "京橋信用ソリューションズ株式会社",
        "tasks": [{"id": t, "owner_keys": ["斎藤悠斗"]} for t in ("T03", "T07", "T11", "T12", "T25")]
                 + [{"id": "T01", "owner_keys": ["佐藤健一"]}],
        "milestones": [],
        "ms_tasks": {},
        "checkpoints": [],
        "roster": [{"role": "データエンジニア", "name": "斎藤 悠斗", "name_key": "斎藤悠斗"}],
        "action_ids": [],
        "id_counts": {"task": 6, "milestone": 0, "action": 0, "total": 6},
    }


@pytest.fixture()
def synth(monkeypatch):
    rows = [_kaede(), _minamino(), _aoba(), _kss()]
    monkeypatch.setattr(L._ss, "load", lambda path=None: rows)
    return rows


Q92 = "恒一会 かえで総合病院案件において、マイルストーンID、タスクID、アクションIDの3種類のIDは合計でいくつ発行されていますか。マークダウンファイル以外から算出してください。"
Q94 = "蒼樹会 みなみ野女性医療センターのスケジュール.xlsxにおいて、MS3に紐づくタスクのうち、ビジネスアナリストが関わっているタスクIDを答えてください。"
Q96 = "青葉与信マネジメントのチェックポイント2として設定されている内容に関連するタスクIDを教えてください。"
Q72 = "KSSにおいて、データエンジニアが担当するタスクIDはいくつありますか。"
# sentinel-ish schedule questions that must NOT fire
Q_SENT2 = "青嶺不動産アセットマネジメントのスケジュール_r2.xlsxにおいて、オレンジにハイライトされている行のタスク名をすべて答えてください。"
Q_SENT90 = "青潮モビリティサービスのスケジュール.xlsxにおいて、バッファとして使用した工数の合計は何時間ですか。"
Q_SENT21 = "青葉バイオメディカル機器のクライアントの主担当者の役職は何ですか。"


def test_default_off_returns_none(monkeypatch, synth):
    monkeypatch.delenv("RAG_SCHEDULE_STORE", raising=False)
    assert L.enabled() is False
    for q in (Q92, Q94, Q96, Q72):
        assert L.resolve(q) is None
    assert L.tool() is None


def test_idx92_id_kind_total(monkeypatch, synth):
    monkeypatch.setenv("RAG_SCHEDULE_STORE", "1")
    r = L.resolve(Q92)
    assert r is not None and r["value"] == 49
    assert r["method"]["selection"] == "id_kind_total"


def test_idx94_milestone_role_tasks(monkeypatch, synth):
    monkeypatch.setenv("RAG_SCHEDULE_STORE", "1")
    r = L.resolve(Q94)
    assert r is not None and r["value"] == "T09"
    assert r["method"]["selection"] == "milestone_role_tasks"
    assert r["evidence"]["role"] == "ビジネスアナリスト"


def test_idx96_checkpoint_tasks(monkeypatch, synth):
    monkeypatch.setenv("RAG_SCHEDULE_STORE", "1")
    r = L.resolve(Q96)
    assert r is not None and r["value"] == "T05、T06、T07、T08"
    assert r["method"]["selection"] == "checkpoint_tasks"


def test_idx72_role_task_count(monkeypatch, synth):
    monkeypatch.setenv("RAG_SCHEDULE_STORE", "1")
    r = L.resolve(Q72)
    assert r is not None and r["value"] == 5
    assert r["method"]["selection"] == "role_task_count"


def test_sentinels_do_not_fire(monkeypatch, synth):
    monkeypatch.setenv("RAG_SCHEDULE_STORE", "1")
    for q in (Q_SENT2, Q_SENT90, Q_SENT21):
        assert L.resolve(q) is None


def test_role_scope_is_case_local(monkeypatch, synth):
    """役割は案件スコープ厳守: みなみ野 BA=松本真央。KSS の roster を流用しない。"""
    monkeypatch.setenv("RAG_SCHEDULE_STORE", "1")
    # KSS には BA roster が無い ⇒ みなみ野向けの BA クエリを KSS 文脈で誤射しない
    q_kss_ba = "KSSにおいて、MS3に紐づくタスクのうちビジネスアナリストが関わるタスクIDを答えてください。"
    assert L.resolve(q_kss_ba) is None


def test_tool_surface_off_none(monkeypatch, synth):
    monkeypatch.setenv("RAG_SCHEDULE_STORE", "1")
    t = L.tool()
    assert t is not None and t[0] == "schedule_lookup"
