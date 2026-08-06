"""Network-free regressions for the gold-100 review表 generator (SOT-2491).

Builds a :class:`~scoring.gold_offline.Report` with a stub judge (no LLM/network), then checks that
``gold_review.generate`` writes md/csv with one row per scored item and appends (never overwrites)
one metrics-only line per run to the history jsonl.
"""
import csv
import json

from config import settings
from scoring import gold_offline as GO
from scoring import gold_review as GR


def _stub_judge(verdict_by_pred):
    def judge(pairs):
        return [verdict_by_pred[pred] for pred, _truth in pairs]
    return judge


def _report_100():
    """100-item report: cycle match / wrong / sentinel-abstain."""
    gold = {i: f"gold-{i}" for i in range(100)}
    preds = {}
    verdict_by_pred = {}
    for i in range(100):
        r = i % 3
        if r == 0:
            ans = f"ok-{i}"
            verdict_by_pred[ans] = "Perfect"
        elif r == 1:
            ans = f"bad-{i}"
            verdict_by_pred[ans] = "Incorrect"
        else:
            ans = settings.ABSTAIN  # sentinel abstain never reaches the judge
        preds[i] = {"index": i, "question": f"質問{i}?", "answer": ans, "cost_usd": 0.001}
    report = GO.evaluate(preds, gold, judge=_stub_judge(verdict_by_pred))
    return report


def test_generate_writes_md_csv_100_rows(tmp_path):
    report = _report_100()
    md = tmp_path / "gold_100_review.md"
    csv_path = tmp_path / "gold_100_review.csv"
    history = tmp_path / "history.jsonl"
    details = {0: {"reason": "確信ありのため回答", "confidence": 0.9, "status": "answered"}}

    entry = GR.generate(report, details=details, gen="investigator",
                        out_md=md, out_csv=csv_path, history_path=history)

    assert md.exists() and csv_path.exists()
    # csv: header + 100 data rows, columns exactly the spec order.
    with csv_path.open(encoding="utf-8-sig", newline="") as fh:
        rows = list(csv.DictReader(fh))
    assert len(rows) == 100
    assert list(rows[0].keys()) == GR._CSV_COLUMNS
    # status buckets reflect the verdicts (0->MATCH, 1->WRONG, 2->ABSTAIN).
    by_index = {int(r["index"]): r for r in rows}
    assert by_index[0]["status"] == "MATCH"
    assert by_index[1]["status"] == "WRONG"
    assert by_index[2]["status"] == "ABSTAIN"
    # details enrich reason/confidence for the matching index only.
    assert by_index[0]["reason"] == "確信ありのため回答"
    assert by_index[0]["confidence"] == "0.90"
    assert by_index[1]["reason"] == "" and by_index[1]["confidence"] == ""
    # difficulty carries the system-behaviour suffix.
    assert "システム正答" in by_index[0]["difficulty"]
    assert "システム過信誤り" in by_index[1]["difficulty"]
    assert "システム棄権" in by_index[2]["difficulty"]

    # md: 100 data rows (table rows minus the header + separator).
    body = [ln for ln in md.read_text(encoding="utf-8").splitlines() if ln.startswith("| ")]
    assert len(body) == 100 + 1  # +1 header row (separator uses `|---`, not `| `)
    assert "Private" in md.read_text(encoding="utf-8")

    # history: metrics-only, no question/answer leakage.
    assert entry["n"] == 100 and entry["match"] == 34  # indices 0,3,...,99 -> 34 matches
    line = history.read_text(encoding="utf-8").strip()
    rec = json.loads(line)
    assert rec["gen"] == "investigator" and rec["n"] == 100
    assert "by_type" in rec
    blob = json.dumps(rec, ensure_ascii=False)
    assert "質問" not in blob and "gold-" not in blob  # no prompt/answer content in history


def test_history_appends_not_overwrites(tmp_path):
    report = _report_100()
    md = tmp_path / "r.md"
    csv_path = tmp_path / "r.csv"
    history = tmp_path / "history.jsonl"

    GR.generate(report, gen="investigator", out_md=md, out_csv=csv_path, history_path=history)
    GR.generate(report, gen="resolve", out_md=md, out_csv=csv_path, history_path=history)

    lines = [ln for ln in history.read_text(encoding="utf-8").splitlines() if ln.strip()]
    assert len(lines) == 2
    assert json.loads(lines[0])["gen"] == "investigator"
    assert json.loads(lines[1])["gen"] == "resolve"


def test_load_details_tolerates_investigator_format(tmp_path):
    # investigator details have confidence but no status/reason; loader must not choke.
    p = tmp_path / "p.details.jsonl"
    p.write_text(
        json.dumps({"index": 0, "answer": "x", "confidence": "high"}) + "\n"
        + json.dumps({"index": 1, "answer": "y", "status": "abstained", "reason": "根拠不足",
                      "confidence": 0.0}) + "\n"
        + "\n",  # blank line ignored
        encoding="utf-8")
    d = GR.load_details(p)
    assert d[0]["confidence"] == "high" and d[0]["reason"] == ""
    assert d[1]["reason"] == "根拠不足" and d[1]["status"] == "abstained"
    # a non-jsonl / missing path returns {} instead of raising.
    assert GR.load_details(tmp_path / "nope.csv") == {}
    assert GR.load_details(None) == {}
