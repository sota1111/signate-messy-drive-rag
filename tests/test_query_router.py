"""SOT-2584 — tests for the typed query router (7 routes + slots + budget + primary lane)."""
from __future__ import annotations

from src.rag.agent import query_router as qr
from src.rag.agent import question_contract as qc


def test_all_seven_routes_have_slots_and_budget():
    """Every route declares evidence slots and a budget contract (6+1 完備)."""
    assert set(qr.ROUTES) == {
        qr.LOOKUP, qr.ENUM, qr.NUMERIC, qr.VERSION_DIFF, qr.FORMAT, qr.PIVOT, qr.EXISTENCE}
    for route in qr.ROUTES:
        assert qr.ROUTE_SLOTS[route], route
        assert route in qr.ROUTE_BUDGET, route
        assert route in qr.ROUTE_LABELS, route


def test_numeric_route_maps_to_compute_hard_lane():
    d = qr.classify_route("京橋のtrain.xlsxの平均年齢を教えて")
    assert d.route == qr.NUMERIC
    assert d.primary_lane == "compute"
    assert d.has_hard_lane
    assert d.evidence_slots == ("operands", "unit", "operation", "rounding")


def test_version_diff_route():
    d = qr.classify_route("旧版と最新版の変更点は何ですか")
    assert d.route == qr.VERSION_DIFF
    assert d.primary_lane == "version_diff"
    assert "version_pair" in d.evidence_slots


def test_enum_route_zero_fallback_budget():
    d = qr.classify_route("該当する項目をすべて挙げてください")
    assert d.route == qr.ENUM
    # ENUM's budget contract forbids free exploration fallback (research: scan + fallback 0).
    assert d.budget.tool_budget_remaining == 0


def test_format_route():
    d = qr.classify_route("太字かつ下線の箇所を抽出してください")
    assert d.route == qr.FORMAT
    assert d.primary_lane == "highlight_extract"


def test_pivot_route_overrides_format():
    q = "ピボットテーブルで行フィールドが部署のとき対象列の集計値は何ですか"
    assert qr.classify_route(q).route == qr.PIVOT


def test_existence_route_refines_lookup():
    d = qr.classify_route("かえで病院案件に特別な精算規定は存在しますか")
    assert d.route == qr.EXISTENCE
    assert d.budget.tool_budget_remaining == 0
    assert d.evidence_slots == ("exhaustive_search_coverage", "parser_support")


def test_existence_suppressed_by_enumeration_cue():
    # "list all that exist" is an enumeration, not an existence yes/no.
    assert not qr.is_existence_question("存在する案件をすべて挙げてください")


def test_reuses_passed_contract_without_reclassifying():
    contract = qc.classify("旧版と最新版の変更点は何ですか")
    d = qr.classify_route("旧版と最新版の変更点は何ですか", contract=contract)
    assert d.contract == contract.contract == qc.VERSION_DIFF


def test_route_decision_to_dict_is_json_shaped():
    d = qr.classify_route("京橋のtrain.xlsxの平均年齢を教えて")
    payload = d.to_dict()
    assert payload["route"] == "NUMERIC"
    assert payload["budget"]["tool_budget_remaining"] == 1
    assert payload["has_hard_lane"] is True


def test_route_from_archetype_maps_backbone():
    assert qr.route_from_archetype("version_diff") == qr.VERSION_DIFF
    assert qr.route_from_archetype("derived_calculation") == qr.NUMERIC
    assert qr.route_from_archetype("enum_set") == qr.ENUM
    assert qr.route_from_archetype("fact_lookup") == qr.LOOKUP


def test_route_agreement_counts_and_distribution():
    rows = [
        {"question": "京橋のtrain.xlsxの平均年齢を教えて", "archetype": "derived_calculation"},
        {"question": "旧版と最新版の変更点は何ですか", "archetype": "version_diff"},
        {"question": "該当する項目をすべて挙げてください", "archetype": "enum_set"},
        {"question": "太字かつ下線の箇所を抽出してください", "archetype": "document_extract"},
    ]
    report = qr.route_agreement(rows)
    assert report.total == 4
    assert 0.0 <= report.rate <= 1.0
    assert sum(report.distribution.values()) == 4
    assert 0.0 <= report.hard_lane_rate <= 1.0
    # NUMERIC / VERSION_DIFF / ENUM agree with their archetype route; FORMAT refines document_extract.
    assert report.rate == 1.0
