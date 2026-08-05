"""Network-free regressions for the 関門2汎化 promotion gate + ledger attribution (SOT-2478).

The promotion gate adopts a single diff ONLY when it does not regress the 関門2 汎化 (gate2 sealed
= 未知案件への転移) score. Gold match rate is recorded for attribution but never promotes on its own.
Every case here is built from plain gate2 report dicts (the shape scoring.gate2.GateReport.to_dict()
produces), so nothing touches the LLM / network.
"""
import json

import pytest

from scoring import ledger, selfimprove as S


def _gate2(seen: float, sealed: float | None, *, usable: bool | None = None) -> dict:
    """A minimal gate2 report dict: seen/sealed slice scores + overall + usable flag."""
    scores = [seen] + ([sealed] if sealed is not None else [])
    overall = round(sum(scores) / len(scores), 4) if scores else None
    return {
        "n": len(scores),
        "overall_score": overall,
        "seen": {"n": 1, "score": seen},
        "sealed": {"n": (1 if sealed is not None else 0), "score": sealed},
        "generalization_gap": None if sealed is None else round(seen - sealed, 4),
        "usable": (sealed is not None) if usable is None else usable,
    }


# --------------------------------------------------------------------- generalization score helper
def test_generalization_score_reads_sealed_slice():
    assert S.gate2_generalization_score(_gate2(0.9, 0.7)) == pytest.approx(0.7)
    assert S.gate2_generalization_score(_gate2(0.9, None)) is None  # no sealed slice → undefined
    assert S.gate2_generalization_score(None) is None


# ------------------------------------------------------------------------------- promotion decision
def test_non_regressing_diff_is_promoted():
    champ = _gate2(0.80, 0.60)
    cand = _gate2(0.85, 0.62)  # 汎化 improved
    d = S.decide_promotion("bump-retrieval-k", champ, cand)
    assert d.promote is True
    assert d.basis == "gate2_generalization_non_regression"
    assert d.gen_delta == pytest.approx(0.02)


def test_flat_generalization_is_non_regression():
    """Byte-identical 汎化 (Δ≈0) counts as 非劣化 within the float tolerance and is promoted."""
    champ = _gate2(0.80, 0.60)
    cand = _gate2(0.90, 0.60)  # seen rose, sealed unchanged
    d = S.decide_promotion("cosmetic-refactor", champ, cand)
    assert d.promote is True
    assert d.gen_delta == pytest.approx(0.0)


def test_generalization_regressing_diff_is_rejected():
    """核心: a diff that drops the 関門2 汎化 score is automatically 却下 (検証内容)."""
    champ = _gate2(0.80, 0.60)
    cand = _gate2(0.95, 0.50)  # seen up but 汎化 DOWN = overfit
    d = S.decide_promotion("overfit-prompt", champ, cand)
    assert d.promote is False
    assert d.gen_delta == pytest.approx(-0.10)
    assert any("劣化" in r for r in d.reasons)


def test_gold_improvement_alone_never_promotes():
    """ゴールド一致率だけでは昇格させない: gold up but 汎化 regressed → still 却下."""
    champ = _gate2(0.80, 0.60)
    cand = _gate2(0.80, 0.55)  # 汎化 regressed
    d = S.decide_promotion("gold-tuned", champ, cand,
                           gold_champion=0.70, gold_candidate=0.90)  # gold +0.20
    assert d.promote is False
    assert d.gold_delta == pytest.approx(0.20)
    assert any("ゴールド" in r and "昇格させない" in r for r in d.reasons)


def test_unusable_report_cannot_promote():
    """No sealed slice → 汎化 undefined → cannot judge → 却下 (even if seen improved)."""
    champ = _gate2(0.80, None)  # usable False
    cand = _gate2(0.99, None)
    d = S.decide_promotion("no-sealed-fixture", champ, cand, gold_champion=0.5, gold_candidate=0.9)
    assert d.usable is False
    assert d.promote is False
    assert any("汎化スコアが未定義" in r for r in d.reasons)


def test_decision_is_json_serializable():
    d = S.decide_promotion("x", _gate2(0.8, 0.6), _gate2(0.8, 0.6))
    json.dumps(d.to_dict(), ensure_ascii=False)  # must not raise
    assert set(d.to_dict()) >= {"diff", "promote", "basis", "usable", "champion_gen",
                                "candidate_gen", "gen_delta", "gold_delta", "reasons"}


# --------------------------------------------------------------------------- ledger attribution
def test_record_promotion_round_trip_and_upsert(tmp_path):
    path = tmp_path / "promotions.jsonl"
    d = S.decide_promotion("diff-A", _gate2(0.8, 0.60), _gate2(0.8, 0.65))
    row = ledger.record_promotion(d.to_dict(), date="2026-08-05", commit="abc1234",
                                  notes="first", path=path)
    assert row["diff"] == "diff-A" and row["promote"] is True
    assert row["date"] == "2026-08-05" and row["commit"] == "abc1234"
    assert ledger.load_promotions(path) == [row]

    # A different diff appends a second row.
    d2 = S.decide_promotion("diff-B", _gate2(0.8, 0.60), _gate2(0.8, 0.50))
    ledger.record_promotion(d2.to_dict(), date="2026-08-05", path=path)
    assert [r["diff"] for r in ledger.load_promotions(path)] == ["diff-A", "diff-B"]

    # Re-recording the SAME diff upserts in place (one single diff → one row).
    ledger.record_promotion(d.to_dict(), date="2026-08-06", commit="def5678", path=path)
    rows = ledger.load_promotions(path)
    assert sum(r["diff"] == "diff-A" for r in rows) == 1
    assert next(r for r in rows if r["diff"] == "diff-A")["commit"] == "def5678"


def test_promote_and_record_writes_attribution(tmp_path):
    path = tmp_path / "promotions.jsonl"
    decision, row = S.promote_and_record(
        "regressing-diff", _gate2(0.8, 0.60), _gate2(0.95, 0.40),
        date="2026-08-05", commit="c0ffee", gold_champion=0.6, gold_candidate=0.8, path=path)
    assert decision.promote is False          # 汎化 dropped 0.60 → 0.40
    assert row["promote"] is False and row["gold_delta"] == pytest.approx(0.20)
    assert ledger.load_promotions(path)[0]["diff"] == "regressing-diff"
