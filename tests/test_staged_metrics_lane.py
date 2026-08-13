"""SOT-2699 — 段階メトリクス フル精度 F1 差 serve レーンの offline テスト（LLM/corpus 不要）。

OFF ⇒ None（byte-identical）、idx36 = |最終 − 中間| をフル精度で直答、フル精度が焼けていない案件は
honest abstain（None）、二重検算不一致は defer。
"""
from __future__ import annotations

import pytest

from src.rag.agent import staged_metrics_lane as L

_Q36 = ("恒一会 かえで総合病院案件において、中間報告時点のF1スコア実測値と最終報告時点のF1スコア実測値の"
        "差を絶対値で答えてください。")

_FINAL = 0.8291582445227382
_INTERIM = 0.7329671168078127
_GOLD = "0.09619112771492555"


def _rows(*, verified=True, interim=_INTERIM):
    sm = {"final_f1_macro": _FINAL, "final_f1_source": "metrics.json:f1_macro"}
    if verified:
        sm["interim_f1"] = {"value": interim, "raw": repr(interim), "source_rel":
                            "y/05.会議/報告資料/報告資料_2025-09-16.docx", "date": "2025-09-16"}
        sm["f1_stage_abs_diff"] = abs(_FINAL - interim)
        sm["f1_stage_diff_verified"] = True
    else:
        sm["intermediate_full_precision_available"] = False
    return [{"project": "医療法人社団 恒一会 かえで総合病院", "staged_metrics": sm}]


@pytest.fixture()
def wired(monkeypatch):
    monkeypatch.setenv("RAG_STAGED_METRICS", "1")
    monkeypatch.setattr(L._store, "load", lambda *a, **k: _rows())

    class _G:
        def company_of(self, q):
            return "医療法人社団 恒一会 かえで総合病院" if "恒一会" in q or "かえで" in q else None

    import src.rag.extract.glossary as g
    monkeypatch.setattr(g, "load", lambda *a, **k: _G())
    return L


def test_off_is_none(monkeypatch):
    monkeypatch.delenv("RAG_STAGED_METRICS", raising=False)
    monkeypatch.setattr(L._store, "load", lambda *a, **k: _rows())
    assert L.enabled() is False
    assert L.resolve(_Q36) is None  # OFF ⇒ byte-identical fallback
    assert L.tool() is None


def test_idx36_full_precision_diff(wired):
    r = wired.resolve(_Q36)
    assert r is not None
    assert r["value"] == _GOLD  # フル精度（judge がフル精度一致を要求, SOT-2687）
    assert r["method"]["engine"] == "staged_metrics"


def test_honest_abstain_when_not_verified(monkeypatch):
    monkeypatch.setenv("RAG_STAGED_METRICS", "1")
    monkeypatch.setattr(L._store, "load", lambda *a, **k: _rows(verified=False))

    class _G:
        def company_of(self, q):
            return "医療法人社団 恒一会 かえで総合病院"

    import src.rag.extract.glossary as g
    monkeypatch.setattr(g, "load", lambda *a, **k: _G())
    # フル精度が焼けていない ⇒ 丸めで近似せず None（honest abstain）。
    assert L.resolve(_Q36) is None


def test_defer_when_double_check_mismatches(monkeypatch):
    monkeypatch.setenv("RAG_STAGED_METRICS", "1")
    rows = _rows()
    rows[0]["staged_metrics"]["f1_stage_abs_diff"] = 0.5  # store 値と再計算が食い違う
    monkeypatch.setattr(L._store, "load", lambda *a, **k: rows)

    class _G:
        def company_of(self, q):
            return "医療法人社団 恒一会 かえで総合病院"

    import src.rag.extract.glossary as g
    monkeypatch.setattr(g, "load", lambda *a, **k: _G())
    assert L.resolve(_Q36) is None  # 二重検算不一致 ⇒ fail-closed


def test_unrelated_question_defers(wired):
    assert wired.resolve("かえで総合病院の契約金額はいくらですか。") is None
