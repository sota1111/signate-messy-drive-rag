"""SOT-2584 — tests for the Evidence Packet builder + pre-inject directive."""
from __future__ import annotations

from src.rag.agent import evidence_packet as ep
from src.rag.agent import query_router as qr


def test_disabled_by_default():
    assert ep.enabled() is False


def test_packet_json_shape_has_research_fields():
    packet = ep.build_packet("京橋のtrain.xlsxの平均年齢を教えて")
    payload = packet.to_dict()
    for key in ("route", "resolved_documents", "required_evidence", "evidence",
                "missing", "tool_budget_remaining"):
        assert key in payload, key
    assert payload["route"] == qr.NUMERIC


def test_missing_is_all_required_when_no_evidence_prefilled():
    packet = ep.build_packet("京橋のtrain.xlsxの平均年齢を教えて")
    # In this landing evidence starts empty → every required slot is missing.
    assert tuple(packet.missing) == tuple(packet.required_evidence)
    assert packet.evidence == {}


def test_tool_budget_matches_route_contract():
    numeric = ep.build_packet("京橋のtrain.xlsxの平均年齢を教えて")
    assert numeric.tool_budget_remaining == qr.ROUTE_BUDGET[qr.NUMERIC].tool_budget_remaining
    enum = ep.build_packet("該当する項目をすべて挙げてください")
    assert enum.tool_budget_remaining == 0  # ENUM = scan + fallback 0


def test_directive_mentions_slots_budget_and_primary_lane():
    packet, directive = ep.build_directive("京橋のtrain.xlsxの平均年齢を教えて")
    assert "Evidence Packet" in directive
    assert "compute" in directive          # NUMERIC primary lane
    assert "missing" in directive
    assert "tool_budget_remaining" in directive
    # The JSON packet is embedded verbatim for the agent.
    assert packet.to_json() in directive


def test_directive_notes_budget_exhausted_is_operational():
    _packet, directive = ep.build_directive("京橋のtrain.xlsxの平均年齢を教えて")
    assert "BUDGET_EXHAUSTED" in directive


def test_build_packet_reuses_decision():
    decision = qr.classify_route("旧版と最新版の変更点は何ですか")
    packet = ep.build_packet("旧版と最新版の変更点は何ですか", decision=decision)
    assert packet.route == qr.VERSION_DIFF
    assert packet.primary_lane == "version_diff"


def test_no_registry_artifact_yields_empty_resolved_documents(monkeypatch):
    # Fail-open: no resolver → empty resolved_documents, directive still built (agent falls back).
    from src.rag.index import document_registry as dr
    monkeypatch.setattr(dr, "get_resolver", lambda *a, **k: None)
    packet, directive = ep.build_directive("存在しないファイル_xyz.xlsx の値")
    assert packet.resolved_documents == ()
    assert "find_files" in directive or "canonical_route" in directive
