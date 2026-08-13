"""SOT-2699 — 統計表 rank/ratio serve レーンの offline テスト（ネットワーク/LLM 不要）。

決定論束縛の規律を合成ストアで固定する: OFF ⇒ None（byte-identical）、idx99=最高÷4番目に低い の比を
丸め指定つきで直答、案件名の一部（『女性』）を metric と誤認しない、曖昧束縛は defer。
"""
from __future__ import annotations

import pytest

from src.rag.agent import derived_ranking_lane as L


def _death_series():
    entries = [
        {"label": "青森県", "value": 18.2}, {"label": "秋田県", "value": 16.3},
        {"label": "香川県", "value": 16.1}, {"label": "鹿児島県", "value": 15.0},
        {"label": "徳島県", "value": 14.9}, {"label": "神奈川県", "value": 7.2},
        {"label": "愛知県", "value": 7.22}, {"label": "東京都", "value": 7.28},
        {"label": "滋賀県", "value": 7.3}, {"label": "奈良県", "value": 8.0},
    ]
    return {
        "rel": "x/糖尿病統計情報.docx", "locus": "表3", "caption": "都道府県別死亡率",
        "metric_key": "死亡率", "header": "死亡率（%）", "unit": "%", "n": len(entries),
        "entries": entries,
        "sorted_asc": sorted(entries, key=lambda e: e["value"]),
        "sorted_desc": sorted(entries, key=lambda e: e["value"], reverse=True),
    }


def _female_series():
    # 案件名『みなみ野女性医療センター』の一部と一致してしまう罠系列（metric='女性'）。
    entries = [{"label": "A", "value": 1.0}, {"label": "B", "value": 2.0}]
    return {"rel": "x/統計.docx", "caption": "性別内訳", "metric_key": "女性", "header": "女性",
            "n": 2, "entries": entries,
            "sorted_asc": entries, "sorted_desc": entries[::-1]}


@pytest.fixture()
def wired(monkeypatch):
    monkeypatch.setenv("RAG_DERIVED_RANKING", "1")
    by_project = {"医療法人社団 蒼樹会 みなみ野女性医療センター": [_death_series(), _female_series()]}
    monkeypatch.setattr(L._store, "load", lambda *a, **k: {"by_project": by_project})
    return L


_Q99 = ("蒼樹会 みなみ野女性医療センターの糖尿病統計情報調査結果において、死亡率が最も高い都道府県の"
        "死亡率は、4番目に低い都道府県の死亡率の何倍ですか。小数第2位まで求めてください。")


def test_off_is_none(monkeypatch):
    monkeypatch.delenv("RAG_DERIVED_RANKING", raising=False)
    by_project = {"医療法人社団 蒼樹会 みなみ野女性医療センター": [_death_series()]}
    monkeypatch.setattr(L._store, "load", lambda *a, **k: {"by_project": by_project})
    assert L.enabled() is False
    assert L.resolve(_Q99) is None  # OFF ⇒ byte-identical fallback
    assert L.tool() is None


def test_idx99_max_over_nth_lowest_ratio(wired):
    r = wired.resolve(_Q99)
    assert r is not None
    assert r["value"] == "2.49"  # 18.2 / 7.3 = 2.4931… → 小数第2位
    ev = r["evidence"]
    assert ev["numerator"]["value"] == 18.2 and ev["numerator"]["label"] == "青森県"
    assert ev["denominator"]["value"] == 7.3 and ev["denominator"]["label"] == "滋賀県"


def test_metric_bind_ignores_case_name_token(wired):
    # 『女性』系列は case 名の一部なので metric とは誤認しない（助詞隣接 = 『死亡率が』のみ拾う）。
    ser = wired._bind_series(
        [_death_series(), _female_series()], wired._norm(_Q99))
    assert ser is not None and ser["metric_key"] == "死亡率"


def test_defer_without_round_spec(wired):
    q = "みなみ野女性医療センターで死亡率が最も高い県の死亡率は4番目に低い県の死亡率の何倍ですか。"
    assert wired.resolve(q) is None  # 丸め指定なし ⇒ format 曖昧 ⇒ defer


def test_defer_on_ambiguous_case(monkeypatch):
    monkeypatch.setenv("RAG_DERIVED_RANKING", "1")
    # 2 案件が同一セグメント『みなみ野女性医療センター』で両方束縛しうる ⇒ defer。
    by_project = {"蒼樹会 みなみ野女性医療センター": [_death_series()],
                  "分院 みなみ野女性医療センター": [_death_series()]}
    monkeypatch.setattr(L._store, "load", lambda *a, **k: {"by_project": by_project})
    assert L.resolve(_Q99) is None
