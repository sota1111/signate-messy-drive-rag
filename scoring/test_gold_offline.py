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


def test_default_judge_requests_three_vote_majority(monkeypatch):
    from scoring import crag

    seen = {}

    def fake_score_pairs(pairs, **kwargs):
        seen.update(kwargs)
        return 1.0, [{"judged": "Perfect"} for _ in pairs]

    monkeypatch.setattr(crag, "score_pairs", fake_score_pairs)
    assert GO.default_judge([("answer", "gold")]) == ["Perfect"]
    assert seen == {"votes": 3, "majority": True}


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


# --------------------------------------------------------------------------- SOT-2497: abstain codes

def _ledger_line(question, state_code, kind, detail):
    return json.dumps({
        "recorded_at": "2026-08-06T00:00:00+00:00",
        "question": question,
        "state_code": state_code,
        "evidence_obligation": f"{detail} 調査メモ: なし",
        "missing": [{"kind": kind, "detail": detail, "signals": {}}],
        "explored_paths": ["find_files"],
        "stop_reason": "answered", "iterations": 1, "tool_calls": ["find_files"],
        "model": "gemini-2.5-pro",
    }, ensure_ascii=False)


def test_abstain_state_codes_aggregate_by_code_and_archetype(tmp_path):
    # Two abstains with ledger codes (one retrieval miss, one budget), one uncoded abstain (no ledger
    # record), plus a match — the block must split棄権 by cause, metrics only.
    gold = {0: "a", 1: "b", 2: "c", 3: "d"}
    q0, q1, q2 = "用語集の略称は何ですか", "train_rows は何行ですか", "契約金額はいくらですか"
    preds = _preds(
        {"index": 0, "question": q0, "answer": settings.ABSTAIN},   # abstain, coded NOT_RETRIEVED
        {"index": 1, "question": q1, "answer": settings.ABSTAIN},   # abstain, coded BUDGET_EXHAUSTED
        {"index": 2, "question": q2, "answer": settings.ABSTAIN},   # abstain, NOT in ledger -> uncoded
        {"index": 3, "question": "略称は？", "answer": "d"},        # match
    )
    rep = GO.evaluate(preds, gold, judge=_stub_judge({"d": "Perfect"}))

    ledger = tmp_path / "abstain_ledger.jsonl"
    ledger.write_text(
        _ledger_line(q0, "NOT_RETRIEVED", "retrieval", "関連文書を検索で特定できなかった") + "\n"
        + _ledger_line(q1, "BUDGET_EXHAUSTED", "budget", "反復上限で打ち切った") + "\n",
        encoding="utf-8")
    GO.attach_abstain_state_codes(rep, ledger)

    block = rep.to_dict()["abstain_state_codes"]
    assert block["n_abstain"] == 3
    assert block["coded"] == 2 and block["uncoded"] == 1
    assert block["by_code"]["NOT_RETRIEVED"] == 1
    assert block["by_code"]["BUDGET_EXHAUSTED"] == 1
    assert block["by_code"]["EVIDENCE_INCOMPLETE"] == 0  # all seven codes present, zeros included
    # archetype split is nested under each code
    assert block["by_code_archetype"]["NOT_RETRIEVED"] == {"glossary_abbrev": 1}
    assert block["by_code_archetype"]["BUDGET_EXHAUSTED"] == {"data_shape": 1}
    # metrics only — no question text leaks into the block
    assert q0 not in json.dumps(block, ensure_ascii=False)


def test_abstain_codes_latest_ledger_record_wins(tmp_path):
    gold = {0: "a"}
    q = "行数は？"
    preds = _preds({"index": 0, "question": q, "answer": settings.ABSTAIN})
    rep = GO.evaluate(preds, gold, judge=_stub_judge({}))
    ledger = tmp_path / "l.jsonl"
    ledger.write_text(
        _ledger_line(q, "NOT_RETRIEVED", "retrieval", "old") + "\n"
        + _ledger_line(q, "EVIDENCE_INCOMPLETE", "completeness", "根拠が部分的") + "\n",
        encoding="utf-8")
    GO.attach_abstain_state_codes(rep, ledger)
    assert rep.abstain_codes[0]["state_code"] == "EVIDENCE_INCOMPLETE"  # last line wins
    assert rep.abstain_codes[0]["obligation"] == "根拠が部分的"


def test_abstain_codes_missing_ledger_is_all_uncoded(tmp_path):
    from src.rag.agent.abstain_ledger import STATE_CODES

    gold = {0: "a"}
    preds = _preds({"index": 0, "question": "q", "answer": settings.ABSTAIN})
    rep = GO.evaluate(preds, gold, judge=_stub_judge({}))
    GO.attach_abstain_state_codes(rep, tmp_path / "does_not_exist.jsonl")  # fail-open
    block = rep.to_dict()["abstain_state_codes"]
    assert block["n_abstain"] == 1 and block["coded"] == 0 and block["uncoded"] == 1
    assert rep.abstain_codes == {}
    # every state code is present as a zero so history diffs stay stable
    assert set(block["by_code"].values()) == {0}
    assert set(block["by_code"]) == set(STATE_CODES)
