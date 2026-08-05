"""Network-free regressions for the gold-100 offline runner (SOT-2472).

The report logic is exercised with a stub judge so no LLM / network is touched: the stub returns
whatever verdict the fixture asks for, keyed by the ground-truth string.
"""
import csv
import json

from config import settings
from scoring import gold_offline as GO


def _stub_judge(verdict_by_truth):
    """A Judge that returns the fixture's verdict for each pair's ground_truth."""
    def judge(pairs):
        return [verdict_by_truth[truth] for _pred, truth in pairs]
    return judge


def _preds(*rows):
    return {r["index"] if isinstance(r["index"], int) else int(r["index"]): r for r in rows}


def test_load_gold_headerless(tmp_path):
    p = tmp_path / "gold.csv"
    with p.open("w", encoding="utf-8", newline="") as fh:
        w = csv.writer(fh)
        w.writerow([0, "bmi"])
        w.writerow([1, "time_and_materials、25,000円／時間"])  # embedded comma must survive quoting
    gold = GO.load_gold(p)
    assert gold == {0: "bmi", 1: "time_and_materials、25,000円／時間"}


def test_load_predictions_jsonl_and_csv(tmp_path):
    j = tmp_path / "p.details.jsonl"
    j.write_text(json.dumps({"index": 0, "answer": "x", "cost_usd": "0.5"}) + "\n"
                 + json.dumps({"index": 1, "answer": "y"}) + "\n", encoding="utf-8")
    rows = GO.load_predictions(j)
    assert rows[0]["answer"] == "x" and rows[0]["cost_usd"] == "0.5"

    c = tmp_path / "p.csv"
    with c.open("w", encoding="utf-8", newline="") as fh:
        csv.writer(fh).writerows([[0, "a"], [1, "b"]])  # official headerless
    rows = GO.load_predictions(c)
    assert rows[0]["answer"] == "a" and rows[1]["answer"] == "b"


def test_match_wrong_abstain_buckets():
    gold = {0: "bmi", 1: "42", 2: "青"}
    preds = _preds(
        {"index": 0, "question": "略称は？", "answer": "bmi", "cost_usd": 0.01},
        {"index": 1, "question": "何行？", "answer": "99", "cost_usd": 0.02},
        {"index": 2, "question": "色は？", "answer": settings.ABSTAIN, "cost_usd": 0.03},
    )
    judge = _stub_judge({"bmi": "Perfect", "42": "Incorrect"})  # abstain never reaches the judge
    rep = GO.evaluate(preds, gold, judge=judge)
    d = rep.to_dict()
    assert d["match"]["count"] == 1
    assert d["wrong"]["count"] == 1
    assert d["abstain"]["count"] == 1
    assert d["cost_usd"] == 0.06


def test_sentinel_abstain_skips_the_judge():
    gold = {0: "x"}
    preds = _preds({"index": 0, "question": "q", "answer": settings.ABSTAIN})

    def exploding_judge(pairs):
        raise AssertionError("judge must not be called for a sentinel abstention")

    rep = GO.evaluate(preds, gold, judge=exploding_judge)
    assert rep.to_dict()["abstain"]["count"] == 1


def test_empty_answer_is_abstain_not_wrong():
    gold = {0: "x"}
    preds = _preds({"index": 0, "question": "q", "answer": "   "})
    rep = GO.evaluate(preds, gold, judge=_stub_judge({}))
    assert rep.to_dict()["abstain"]["count"] == 1 and rep.to_dict()["wrong"]["count"] == 0


def test_missing_verdict_from_judge_counts_as_abstain():
    # A non-sentinel answer the judge deems "Missing" (e.g. "見つかりません") is still an abstention.
    gold = {0: "x"}
    preds = _preds({"index": 0, "question": "q", "answer": "見つかりませんでした"})
    rep = GO.evaluate(preds, gold, judge=_stub_judge({"x": "Missing"}))
    assert rep.to_dict()["abstain"]["count"] == 1


def test_baseline_manual_rate_gate():
    gold = {i: "x" for i in range(30)}
    # 26 correct, 4 wrong -> exactly the manual 26/30 pace -> meets.
    preds = _preds(*[
        {"index": i, "question": "q", "answer": "x" if i < 26 else "z"} for i in range(30)
    ])
    verdicts = {"x": "Perfect"}
    judge = _stub_judge({"x": "Perfect"})

    # override: the 4 "z" answers still map by truth "x"; make them wrong explicitly
    def judge2(pairs):
        return ["Perfect" if pred == "x" else "Incorrect" for pred, _t in pairs]

    rep = GO.evaluate(preds, gold, judge=judge2)
    d = rep.to_dict()
    assert d["match"]["count"] == 26
    assert d["baseline"]["meets"] is True
    # one fewer correct -> below the manual pace
    preds[25]["answer"] = "z"
    rep2 = GO.evaluate(preds, gold, judge=judge2)
    assert rep2.to_dict()["baseline"]["meets"] is False


def test_by_type_breakdown_uses_archetype():
    gold = {0: "略", 1: "12"}
    preds = _preds(
        {"index": 0, "question": "用語集の略称は何ですか", "answer": "略"},
        {"index": 1, "question": "train_rows は何行ですか", "answer": "12"},
    )
    rep = GO.evaluate(preds, gold, judge=_stub_judge({"略": "Perfect", "12": "Perfect"}))
    by_type = rep.to_dict()["by_type"]
    assert "glossary_abbrev" in by_type
    assert "data_shape" in by_type
    assert by_type["glossary_abbrev"]["match"] == 1


def test_wrong_to_abstain_share():
    gold = {0: "a", 1: "b", 2: "c", 3: "d"}
    preds = _preds(
        {"index": 0, "question": "q", "answer": "a"},                # match
        {"index": 1, "question": "q", "answer": settings.ABSTAIN},   # abstain
        {"index": 2, "question": "q", "answer": settings.ABSTAIN},   # abstain
        {"index": 3, "question": "q", "answer": "wrong"},            # wrong
    )
    rep = GO.evaluate(preds, gold, judge=_stub_judge({"a": "Perfect", "d": "Incorrect"}))
    w2a = rep.to_dict()["wrong_to_abstain"]
    assert w2a["non_match"] == 3
    assert w2a["abstained"] == 2
    assert w2a["incorrect"] == 1
    assert w2a["abstain_share"] == round(2 / 3, 4)


def test_conversion_counts_wrong_to_abstain():
    # baseline: idx0 Incorrect, idx1 Perfect, idx2 Incorrect
    # primary : idx0 abstain (dropped!), idx1 Perfect, idx2 Incorrect (still wrong)
    gold = {0: "a", 1: "b", 2: "c"}
    base = GO.evaluate(
        _preds({"index": 0, "answer": "x", "question": "q"},
               {"index": 1, "answer": "b", "question": "q"},
               {"index": 2, "answer": "y", "question": "q"}),
        gold, judge=_stub_judge({"a": "Incorrect", "b": "Perfect", "c": "Incorrect"}))
    primary = GO.evaluate(
        _preds({"index": 0, "answer": settings.ABSTAIN, "question": "q"},
               {"index": 1, "answer": "b", "question": "q"},
               {"index": 2, "answer": "y", "question": "q"}),
        gold, judge=_stub_judge({"b": "Perfect", "c": "Incorrect"}))
    conv = GO.compare_conversion(base, primary)
    assert conv["wrong_to_abstain"] == 1
    assert conv["wrong_to_wrong"] == 1
    assert conv["match_to_match"] == 1
    assert conv["shared"] == 3


def test_indices_subset_limits_scoring():
    gold = {0: "a", 1: "b", 2: "c"}
    preds = _preds(
        {"index": 0, "answer": "a", "question": "q"},
        {"index": 1, "answer": "b", "question": "q"},
        {"index": 2, "answer": "c", "question": "q"},
    )
    rep = GO.evaluate(preds, gold, judge=_stub_judge({"a": "Perfect", "b": "Perfect"}),
                      indices=[0, 1])
    assert rep.n == 2


def test_only_shared_indices_are_scored():
    gold = {0: "a", 5: "z"}  # 5 has no prediction
    preds = _preds({"index": 0, "answer": "a", "question": "q"},
                   {"index": 9, "answer": "?", "question": "q"})  # 9 not in gold
    rep = GO.evaluate(preds, gold, judge=_stub_judge({"a": "Perfect"}))
    assert rep.n == 1 and rep.items[0].index == 0


def test_render_is_printable():
    gold = {0: "a"}
    preds = _preds({"index": 0, "answer": "a", "question": "用語集の略称は"})
    rep = GO.evaluate(preds, gold, judge=_stub_judge({"a": "Perfect"}))
    text = rep.render()
    assert "match" in text and "by type" in text
