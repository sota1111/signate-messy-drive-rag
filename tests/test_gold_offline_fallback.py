"""SOT-2660 — fallback依存率 (raw-file dependency KPI) aggregation in the gold-offline report.

Network-free: builds ``Item``s directly with SOT-2660 ``raw_file_access`` telemetry and checks the
``_fallback_dependency_block`` / ``Report.to_dict`` computation.
"""
from __future__ import annotations

from scoring.gold_offline import Item, Report, _fallback_dependency_block


def _rfa(used, tools=None, blocked=None, db_only=False):
    return {"raw_file_access": {"used": used, "tools": tools or {}, "blocked": blocked or {},
                                "db_only": db_only}}


def _item(index, verdict, rfa):
    return Item(index=index, question="q", answer="a", gold="g", archetype="lookup",
                verdict=verdict, interventions=rfa)


def test_fallback_rate_counts_only_raw_dependent_matches():
    items = [
        _item(1, "Perfect", _rfa(True, {"read_office": 1})),   # correct, raw-dependent
        _item(2, "Acceptable", _rfa(False)),                    # correct, DB-only
        _item(5, "Perfect", _rfa(True, {"file_grep": 2})),     # correct, raw-dependent
        _item(9, "Incorrect", _rfa(True, {"compute": 1})),     # wrong — excluded from the rate
        _item(3, "Missing", _rfa(False)),                       # abstain — excluded
    ]
    blk = _fallback_dependency_block(items)
    assert blk["measured"] is True
    assert blk["match_total"] == 3
    assert blk["match_raw_dependent"] == 2
    assert blk["fallback_rate"] == round(2 / 3, 4)
    assert blk["db_only_coverage"] == 1
    assert blk["raw_dependent_match_idx"] == [1, 5]
    assert blk["raw_file_tool_calls"] == {"file_grep": 2, "read_office": 1, "compute": 1}


def test_db_only_diagnostic_zero_fallback_and_blocked_tally():
    """A RAG_DB_ONLY run: correct answers used zero raw files, refusals are tallied under blocked."""
    items = [
        _item(1, "Perfect", _rfa(False, blocked={"read_office": 1}, db_only=True)),
        _item(2, "Missing", _rfa(False, blocked={"file_grep": 3}, db_only=True)),
    ]
    blk = _fallback_dependency_block(items)
    assert blk["db_only"] is True
    assert blk["fallback_rate"] == 0.0
    assert blk["db_only_coverage"] == 1
    assert blk["blocked_total"] == 4
    assert blk["raw_dependent_match_idx"] == []


def test_fallback_block_fail_open_on_legacy_rows():
    """Rows without raw_file_access telemetry (legacy details.jsonl) → not measured, never raises."""
    items = [Item(index=1, question="q", answer="a", gold="g", archetype="lookup",
                  verdict="Perfect", interventions={})]
    assert _fallback_dependency_block(items) == {"measured": False}


def test_report_to_dict_and_render_include_fallback_dependency():
    items = [
        _item(1, "Perfect", _rfa(True, {"read_office": 1})),
        _item(2, "Perfect", _rfa(False)),
    ]
    d = Report(items=items).to_dict()
    assert d["fallback_dependency"]["fallback_rate"] == 0.5
    assert d["fallback_dependency"]["raw_dependent_match_idx"] == [1]
    text = Report(items=items).render()
    assert "fallback依存率" in text
