"""Network-free regressions for the 関門2 generalization gate (SOT-2476).

The report machinery is exercised with a stub answerer + stub judge, so no LLM / network is touched:
the answerer returns a canned string per question and the judge returns a fixture verdict per pair.
"""
import json
import unicodedata

import pytest

from scoring import gate2 as G


def _qa(skill, company, sealed, answer="a", question=None):
    return G.QA(question=question or f"q-{company}-{skill}", answer=answer,
                skill=skill, company=company, sealed=sealed)


def _judge(points_by_gt):
    """A Judge returning (mean, results) with a fixture verdict/points keyed by ground truth."""
    def judge(pairs):
        results = []
        for pred, truth in pairs:
            pts = points_by_gt[truth]
            verdict = {1.0: "Perfect", 0.5: "Acceptable", 0.0: "Missing", -1.0: "Incorrect"}.get(
                pts, "Perfect")
            results.append({"judged": verdict, "points": pts, "pred": pred, "truth": truth})
        mean = sum(r["points"] for r in results) / len(results) if results else 0.0
        return mean, results
    return judge


def test_evaluate_partitions_seen_and_sealed():
    holdout = [
        _qa("bold", "SeenCo", False, answer="s1"),
        _qa("csv_agg", "SeenCo", False, answer="s2"),
        _qa("bold", "SealedCo", True, answer="x1"),
    ]
    # seen: 1.0 and 0.0 -> mean 0.5 ; sealed: -1.0 -> mean -1.0
    judge = _judge({"s1": 1.0, "s2": 0.0, "x1": -1.0})
    report = G.evaluate(holdout, answerer=lambda q: "pred", judge=judge, workers=1)
    d = report.to_dict()

    assert d["n"] == 3
    assert d["seen"] == {"n": 2, "score": 0.5}
    assert d["sealed"] == {"n": 1, "score": -1.0}
    assert d["generalization_gap"] == pytest.approx(1.5)  # seen - sealed = 0.5 - (-1.0)
    assert d["usable"] is True
    assert d["by_skill"]["bold"]["n"] == 2
    assert d["verdicts"] == {"Incorrect": 1, "Missing": 1, "Perfect": 1}


def test_answerer_is_called_with_each_question():
    holdout = [_qa("bold", "SeenCo", False, answer="a", question="Q1"),
               _qa("bold", "SealedCo", True, answer="b", question="Q2")]
    asked = []

    def answerer(q):
        asked.append(q)
        return "pred"

    G.evaluate(holdout, answerer=answerer, judge=_judge({"a": 1.0, "b": 1.0}), workers=1)
    assert sorted(asked) == ["Q1", "Q2"]


def test_usable_false_without_a_sealed_slice():
    holdout = [_qa("bold", "SeenCo", False, answer="a")]
    d = G.evaluate(holdout, answerer=lambda q: "p", judge=_judge({"a": 1.0}), workers=1).to_dict()
    assert d["usable"] is False
    assert d["sealed"]["n"] == 0
    assert d["sealed"]["score"] is None
    assert d["generalization_gap"] is None  # undefined without both slices


def test_report_json_serializable():
    holdout = [_qa("bold", "SeenCo", False, answer="a"),
               _qa("bold", "SealedCo", True, answer="b")]
    d = G.evaluate(holdout, answerer=lambda q: "p",
                   judge=_judge({"a": 1.0, "b": 1.0}), workers=1).to_dict()
    json.dumps(d, ensure_ascii=False)  # must not raise
    assert set(d) >= {"overall_score", "seen", "sealed", "generalization_gap", "usable", "isolation"}


# --------------------------------------------------------------------------- isolation invariant
def test_isolation_raises_on_nfc_nfd_leak():
    """The SAME company folder in NFC vs NFD must not silently split into both slices."""
    name = "バイオ社"  # "バ" = ハ + combining dakuten, so NFC != NFD
    nfd = unicodedata.normalize("NFD", name)
    nfc = unicodedata.normalize("NFC", name)
    assert nfd != nfc  # precondition: the name actually differs by normalization form
    sealed = {G.nfc(name)}
    holdout = [_qa("bold", nfc, True, answer="a"), _qa("bold", nfd, False, answer="b")]
    # After NFC-normalization both map to one company carrying two different sealed labels.
    with pytest.raises(G.IsolationError):
        G.check_isolation(holdout, sealed)


def test_isolation_raises_when_sealed_company_in_seen_slice():
    sealed = {G.nfc("SealedCo")}
    holdout = [_qa("bold", "SealedCo", False, answer="a")]  # mislabeled seen
    with pytest.raises(G.IsolationError):
        G.check_isolation(holdout, sealed)


def test_isolation_ok_returns_diagnostics():
    sealed = {G.nfc("SealedCo")}
    holdout = [_qa("bold", "SeenCo", False), _qa("bold", "SealedCo", True)]
    diag = G.check_isolation(holdout, sealed)
    assert diag["seen_companies"] == ["SeenCo"]
    assert diag["sealed_present"] == ["SealedCo"]


def test_evaluate_enforces_isolation():
    """evaluate() must refuse a holdout whose sealed labels are inconsistent (calls check_isolation)."""
    sealed_name = next(iter(G.sealed_companies()), None) or "SealedCo"
    holdout = [_qa("bold", sealed_name, False, answer="a")]  # sealed company mislabeled as seen
    with pytest.raises(G.IsolationError):
        G.evaluate(holdout, answerer=lambda q: "p", judge=_judge({"a": 1.0}), workers=1)
