"""SOT-2639 — offline tests for wiring the Gemini investigator's finalization through the shared
commit gate (:mod:`src.rag.agent.commit_gate`, SOT-2637).

These are network-free: the loop is driven by a scripted fake model (same harness as
``tests/test_investigator.py``) with the real deterministic tool layer. They pin the three properties
the issue's acceptance criteria require:

* **OFF byte-identical** — with ``RAG_COMMIT_GATE`` unset the finalization is unchanged and no
  ``commit_gate`` telemetry key appears (the gate is entirely unwired).
* **ON equivalence / no double application** — a committed answer is served VERBATIM (the gate is the
  terminal decision, not an added formatter — it never re-runs ``formatting.py`` on the model's
  gold-form value), so an ON commit matches the OFF commit byte-for-byte.
* **ON guard + bounded retry** — an ungrounded numeric commit is REJECTED and fed back through the
  existing in-band retry channel, and because the gate degrades to ABSTAIN after
  ``RAG_COMMIT_GATE_ABSTAIN_AFTER`` consecutive rejects the retry can never loop.
"""
from __future__ import annotations

import pytest

from src.rag.agent import investigator as inv
from src.rag.agent.investigator import (
    ABSTAIN,
    SUBMIT_ANSWER,
    AgentTool,
    Call,
    Step,
    Usage,
    investigate,
    is_abstain,
)


# --------------------------------------------------------------------------- scripted fake model
class ScriptedModel:
    """Replays a fixed list of ``Step`` s (mirrors tests/test_investigator.py)."""

    def __init__(self, steps, *, model_name="fake-model"):
        self._steps = list(steps)
        self._i = 0
        self.model_name = model_name

    def next(self, tool_responses):
        if self._i >= len(self._steps):
            return Step(function_calls=(), final_text=ABSTAIN, usage=Usage(1, 1))
        step = self._steps[self._i]
        self._i += 1
        return step


def _submit(answer, *, confidence=0.9, evidence="e", method="m") -> Step:
    return Step(function_calls=(Call(SUBMIT_ANSWER, {
        "answer": answer, "confidence": confidence, "evidence": evidence, "method": method}),),
        usage=Usage(50, 10))


def _compute_tool(value):
    """A ``compute`` tool that always returns ``value`` (a numeric-source record the gate can ground on)."""
    return AgentTool("compute", "d", {"type": "object", "properties": {}},
                     lambda **kw: {"value": value, "evidence": {}, "method": {}})


def _tools(value=20):
    return [_compute_tool(value), inv.SUBMIT_ANSWER_TOOL]


# --------------------------------------------------------------------------- OFF byte-identical
def test_commit_gate_off_is_unwired(monkeypatch):
    monkeypatch.delenv("RAG_COMMIT_GATE", raising=False)
    model = ScriptedModel([
        Step(function_calls=(Call("compute", {"expr": "df['b'].mean()"}),), usage=Usage(100, 20)),
        _submit("20", method="compute"),
    ])
    res = investigate(model, "…", _tools(20), max_turns=5, contract="numeric")
    assert res.stop_reason == "answered"
    assert res.answer.answer == "20"                      # unchanged commit
    assert "commit_gate" not in res.interventions          # gate never ran ⇒ no telemetry key


# --------------------------------------------------------------------------- ON equivalence
def test_commit_gate_on_commits_grounded_value_verbatim(monkeypatch):
    """A grounded numeric commit passes the gate and is served VERBATIM — no re-formatting (no double
    application), so the ON answer is byte-identical to the OFF answer."""
    monkeypatch.setenv("RAG_COMMIT_GATE", "1")
    model = ScriptedModel([
        Step(function_calls=(Call("compute", {"expr": "df['b'].mean()"}),), usage=Usage(100, 20)),
        _submit("20", method="compute"),
    ])
    res = investigate(model, "…", _tools(20), max_turns=5, contract="numeric")
    assert res.stop_reason == "answered"
    assert res.answer.answer == "20"                       # verbatim — the model's own value, not reformatted
    cg = res.interventions["commit_gate"]
    assert cg["verdict"] == "COMMIT"
    assert cg["enabled"] is True and cg["route"] == "numeric"


def test_commit_gate_on_matches_off_for_non_numeric_route(monkeypatch):
    """A non-numeric route has no numeric-grounding requirement; the gate COMMITs the value verbatim, so
    ON and OFF serve the identical answer."""
    steps = lambda: [_submit("東京都", method="lookup")]
    monkeypatch.delenv("RAG_COMMIT_GATE", raising=False)
    off = investigate(ScriptedModel(steps()), "本社の所在地は?", _tools(), max_turns=4,
                      contract="simple_lookup")
    monkeypatch.setenv("RAG_COMMIT_GATE", "1")
    on = investigate(ScriptedModel(steps()), "本社の所在地は?", _tools(), max_turns=4,
                     contract="simple_lookup")
    assert off.answer.answer == on.answer.answer == "東京都"
    assert on.interventions["commit_gate"]["verdict"] == "COMMIT"


def test_commit_gate_on_passes_through_abstain(monkeypatch):
    """A deliberate abstain is already terminal; the gate records it as ABSTAIN and does not alter it —
    identical to OFF."""
    monkeypatch.setenv("RAG_COMMIT_GATE", "1")
    model = ScriptedModel([_submit(ABSTAIN, confidence=0.0)])
    res = investigate(model, "…", _tools(), max_turns=4, contract="numeric")
    assert is_abstain(res.answer.answer)
    assert res.interventions["commit_gate"]["verdict"] == "ABSTAIN"
    assert res.interventions["commit_gate"]["already_abstain"] is True


# ------------------------------------------------------ ON is equivalence-preserving without enforcement
def test_commit_gate_on_observational_keeps_ungrounded_value_verbatim(monkeypatch):
    """SOT-2639 最重要制約 — with RAG_COMMIT_GATE=1 but no RAG_COMMIT_GATE_ENFORCE, the gate is
    OBSERVATIONAL: it records the (here REJECT/ABSTAIN) verdict in telemetry, but the loop's own inline
    guards stay authoritative and the model's committed value is served VERBATIM. This is the idx30
    regression guard — a legitimate derived answer the compute-record heuristic can't ground must NOT be
    degraded to abstain when only RAG_COMMIT_GATE is on (that would break champion equivalence)."""
    monkeypatch.setenv("RAG_COMMIT_GATE", "1")
    monkeypatch.delenv("RAG_COMMIT_GATE_ENFORCE", raising=False)
    model = ScriptedModel([
        Step(function_calls=(Call("compute", {"expr": "x"}),), usage=Usage(10, 5)),
        _submit("1.18%", method="derived"),   # value the compute record (20) cannot ground
    ])
    res = investigate(model, "…", _tools(20), max_turns=5, contract="numeric")
    assert res.answer.answer == "1.18%"                 # served verbatim — no degrade (equivalence)
    assert res.stop_reason == "answered"
    cg = res.interventions["commit_gate"]
    assert cg["verdict"] in {"REJECT", "ABSTAIN"}        # the gate SAW the ungrounded commit …
    assert cg["enabled"] is True                          # … recorded it, but did not enforce it


# --------------------------------------------------------------------------- ON guard + bounded retry (enforced)
def test_commit_gate_on_rejects_ungrounded_numeric_then_abstains(monkeypatch):
    """With RAG_COMMIT_GATE_ENFORCE=1 an ungrounded numeric commit (no matching compute record) is
    REJECTED and retried in-band; after the consecutive-reject threshold the gate degrades to ABSTAIN, so
    the retry is bounded (never loops). This is the enforcement path SOT-2640 uses for claude-mcp."""
    monkeypatch.setenv("RAG_COMMIT_GATE", "1")
    monkeypatch.setenv("RAG_COMMIT_GATE_ENFORCE", "1")
    monkeypatch.setenv("RAG_COMMIT_GATE_ABSTAIN_AFTER", "2")
    # compute grounds only the value 20; the model insists on the ungrounded 999 twice.
    model = ScriptedModel([
        Step(function_calls=(Call("compute", {"expr": "x"}),), usage=Usage(10, 5)),
        _submit("999", method="guess"),   # REJECT #1 → in-band retry
        _submit("999", method="guess"),   # REJECT #2 → streak hits threshold → ABSTAIN
    ])
    res = investigate(model, "…", _tools(20), max_turns=8, contract="numeric")
    assert is_abstain(res.answer.answer)
    cg = res.interventions["commit_gate"]
    assert cg["verdict"] == "ABSTAIN"
    assert any("reject_streak_abstain" in r for r in cg["reasons"])


def test_commit_gate_on_reject_recovers_when_model_grounds(monkeypatch):
    """The REJECT is a retry, not a hard fail: if the model re-derives a grounded value on the next turn
    the gate COMMITs it (proving the reject feeds the ordinary in-band retry flow)."""
    monkeypatch.setenv("RAG_COMMIT_GATE", "1")
    monkeypatch.setenv("RAG_COMMIT_GATE_ENFORCE", "1")
    model = ScriptedModel([
        _submit("999", method="guess"),                                   # REJECT #1 (no compute yet)
        Step(function_calls=(Call("compute", {"expr": "x"}),), usage=Usage(10, 5)),
        _submit("20", method="compute"),                                  # now grounded → COMMIT
    ])
    res = investigate(model, "…", _tools(20), max_turns=8, contract="numeric")
    assert res.answer.answer == "20"
    assert res.interventions["commit_gate"]["verdict"] == "COMMIT"
