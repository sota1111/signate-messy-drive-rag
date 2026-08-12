"""SOT-2670 — two-tier answer schema (RAG_TWO_TIER_ANSWER, default OFF).

Verifies that the ``submit_answer`` schema and answer parsing are structurally two-tier when the flag is
ON (scored answer = bare_answer; full_answer preserved as evidence), and byte-identical to the legacy
single-``answer`` schema when OFF.
"""
from __future__ import annotations

from src.rag.agent import investigator as inv
from src.rag.agent.investigator import (
    ABSTAIN,
    SUBMIT_ANSWER,
    build_tools,
    scored_answer_text,
    submit_answer_tool,
)
from src.rag.tools.profile import CorpusProfile


# --------------------------------------------------------------------------- OFF (default) byte-identical
def test_off_schema_is_legacy_single_answer(monkeypatch):
    monkeypatch.delenv("RAG_TWO_TIER_ANSWER", raising=False)
    tool = submit_answer_tool()
    # OFF returns the exact legacy object (byte-identical serve path).
    assert tool is inv.SUBMIT_ANSWER_TOOL
    props = tool.parameters["properties"]
    assert set(props) == {"answer", "confidence", "evidence", "method"}
    assert tool.parameters["required"] == ["answer"]


def test_off_scored_answer_is_answer_field(monkeypatch):
    monkeypatch.delenv("RAG_TWO_TIER_ANSWER", raising=False)
    # Even if the model somehow supplied bare_answer, OFF ignores it → legacy behavior.
    args = {"answer": "13ページ", "bare_answer": "SHOULD_BE_IGNORED", "confidence": 0.9}
    assert scored_answer_text(args) == "13ページ"
    ans = inv._answer_from_args(args)
    assert ans.answer == "13ページ"
    assert "full_answer" not in ans.evidence


def test_off_build_tools_uses_legacy_submit(monkeypatch):
    monkeypatch.delenv("RAG_TWO_TIER_ANSWER", raising=False)
    tools = build_tools(CorpusProfile())
    submit = next(t for t in tools if t.name == SUBMIT_ANSWER)
    assert submit is inv.SUBMIT_ANSWER_TOOL


# --------------------------------------------------------------------------- ON: structural two-tier
def test_on_schema_requires_bare_answer(monkeypatch):
    monkeypatch.setenv("RAG_TWO_TIER_ANSWER", "1")
    tool = submit_answer_tool()
    props = tool.parameters["properties"]
    assert {"full_answer", "bare_answer"} <= set(props)
    # bare_answer is the required (structurally enforced) field; not the legacy answer.
    assert tool.parameters["required"] == ["bare_answer"]
    # The field descriptions carry the verbatim-copy / 該当なし contract.
    assert "該当なし" in props["bare_answer"]["description"]
    assert "原文どおり" in props["bare_answer"]["description"]


def test_on_scored_answer_is_bare_answer(monkeypatch):
    monkeypatch.setenv("RAG_TWO_TIER_ANSWER", "1")
    args = {
        "full_answer": "13ページ(スライド13)の「8. 費用見積」に記載があるため。",
        "bare_answer": "13ページ",
        "confidence": 0.8,
        "evidence": "proposal.pptx",
    }
    assert scored_answer_text(args) == "13ページ"
    ans = inv._answer_from_args(args)
    assert ans.answer == "13ページ"          # scored value = bare_answer
    assert "full_answer" in ans.evidence      # reasoning preserved for traceability
    assert "13ページ(スライド13)" in ans.evidence
    assert "proposal.pptx" in ans.evidence    # original evidence retained


def test_on_bare_answer_empty_falls_back_to_answer(monkeypatch):
    monkeypatch.setenv("RAG_TWO_TIER_ANSWER", "1")
    args = {"answer": "0円", "bare_answer": "", "confidence": 0.5}
    assert scored_answer_text(args) == "0円"


def test_on_abstain_via_bare_answer(monkeypatch):
    monkeypatch.setenv("RAG_TWO_TIER_ANSWER", "1")
    args = {"full_answer": "全文を確認したが記載なし。", "bare_answer": ABSTAIN}
    ans = inv._answer_from_args(args)
    assert inv.is_abstain(ans.answer)
    assert ans.confidence == 0.0


def test_on_gaitou_nashi_is_not_abstain(monkeypatch):
    # 「該当なし」is a legitimate scored answer (none-found), NOT a 棄権.
    monkeypatch.setenv("RAG_TWO_TIER_ANSWER", "1")
    args = {"bare_answer": "該当なし", "full_answer": "母集団を全数確認した結果、該当項目は存在しない。"}
    ans = inv._answer_from_args(args)
    assert ans.answer == "該当なし"
    assert not inv.is_abstain(ans.answer)
