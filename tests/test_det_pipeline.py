"""SOT-2603 (Stage0) — offline tests for the deterministic pipeline router (入口ゲート).

All network-free: the classifier runs deterministically, the registry is a pure dict, and the live
``answer_question`` path is exercised with a *fake* model factory (no Vertex call). The invariants under
test are the Stage0 contract: default OFF ⇒ byte-identical; empty registry ⇒ every question routes to
the LLM loop; a grounded pipeline short-circuits the loop; an ungrounded pipeline falls back
(回答数を減らさない).
"""
from __future__ import annotations

import pytest

from src.rag.agent import det_pipeline as dp
from src.rag.agent import investigator as inv
from src.rag.agent import question_contract as qc
from src.rag.agent.investigator import ABSTAIN, SUBMIT_ANSWER, Call, Step, Usage
from src.rag.tools import contract as _contract


# --------------------------------------------------------------------------- registry cleanup fixture
@pytest.fixture(autouse=True)
def _restore_registry():
    """Snapshot and restore the module-global registry so tests never leak registrations."""
    saved = dict(dp._REGISTRY)
    try:
        yield
    finally:
        dp._REGISTRY.clear()
        dp._REGISTRY.update(saved)


# --------------------------------------------------------------------------- scripted fake model
class _ScriptedModel:
    def __init__(self, steps, *, model_name="fake-model"):
        self._steps = list(steps)
        self._i = 0
        self.model_name = model_name

    def next(self, _tool_responses):
        if self._i >= len(self._steps):
            return Step(function_calls=(), final_text=ABSTAIN, usage=Usage(1, 1))
        step = self._steps[self._i]
        self._i += 1
        return step


def _submit(answer, *, confidence=0.9):
    return Step(function_calls=(Call(SUBMIT_ANSWER, {"answer": answer, "confidence": confidence}),),
                usage=Usage(10, 5))


def _numeric_question():
    # deterministically classifies as NUMERIC (see test_routing.py)
    return "京橋信用ソリューションズの train.xlsx の loan_amnt の平均は。"


# --------------------------------------------------------------------------- flag / registry unit tests
def test_flag_default_off(monkeypatch):
    monkeypatch.delenv("RAG_DET_PIPELINE_ROUTER", raising=False)
    assert dp.enabled() is False


def test_flag_reads_env(monkeypatch):
    monkeypatch.setenv("RAG_DET_PIPELINE_ROUTER", "1")
    assert dp.enabled() is True
    monkeypatch.setenv("RAG_DET_PIPELINE_ROUTER", "off")
    assert dp.enabled() is False


def test_wave_a1_wires_version_diff():
    # Stage0 shipped the registry empty; Wave A1 (SOT-2605) registers the first per-type pipeline. The
    # router lazily bootstraps the pipelines package, so ``version_diff`` is discovered on lookup.
    assert "version_diff" in dp.registered_contracts()


def test_register_and_lookup():
    fn = lambda q, *, profile=None: _contract.make("v", engine="test")
    dp.register("numeric", fn)
    assert "numeric" in dp.registered_contracts()


def test_register_rejects_empty_type():
    with pytest.raises(ValueError):
        dp.register("", lambda q, *, profile=None: None)


def test_register_rejects_non_callable():
    with pytest.raises(TypeError):
        dp.register("numeric", 123)  # type: ignore[arg-type]


def test_register_duplicate_raises_without_replace():
    dp.register("numeric", lambda q, *, profile=None: None)
    with pytest.raises(ValueError):
        dp.register("numeric", lambda q, *, profile=None: None)
    # replace=True overwrites without raising
    dp.register("numeric", lambda q, *, profile=None: None, replace=True)


def test_unregister_is_noop_when_absent():
    dp.unregister("does_not_exist")  # must not raise


# --------------------------------------------------------------------------- resolve() unit tests
def test_resolve_none_when_disabled():
    dp.register("numeric", lambda q, *, profile=None: _contract.make("v", engine="test"))
    # flag off (force not set) ⇒ never runs the pipeline
    assert dp.resolve("q", "numeric") is None


def test_resolve_none_when_contract_unset():
    assert dp.resolve("q", None, force=True) is None


def test_resolve_none_when_unregistered():
    assert dp.resolve("q", "numeric", force=True) is None


def test_resolve_returns_normalized_contract_when_grounded():
    dp.register("numeric",
                lambda q, *, profile=None: _contract.make(42, engine="pandas", evidence={"file": "t.xlsx"}))
    out = dp.resolve(_numeric_question(), "numeric", force=True)
    assert out is not None
    assert _contract.is_contract(out)
    assert out["value"] == 42
    assert out["evidence"]["file"] == "t.xlsx"
    assert out["method"]["engine"] == "pandas"


def test_resolve_falls_back_on_none_value():
    # a contract whose value is None means "resolved nothing" ⇒ fall back to the LLM loop
    dp.register("numeric", lambda q, *, profile=None: _contract.make(None, engine="test"))
    assert dp.resolve("q", "numeric", force=True) is None


def test_resolve_falls_back_on_pipeline_returning_none():
    dp.register("numeric", lambda q, *, profile=None: None)
    assert dp.resolve("q", "numeric", force=True) is None


def test_resolve_falls_back_on_malformed_result():
    dp.register("numeric", lambda q, *, profile=None: {"not": "a contract"})
    assert dp.resolve("q", "numeric", force=True) is None


def test_resolve_falls_back_when_pipeline_raises():
    def boom(q, *, profile=None):
        raise RuntimeError("pipeline blew up")

    dp.register("numeric", boom)
    assert dp.resolve("q", "numeric", force=True) is None  # never propagates into the answer path


def test_resolve_env_gated(monkeypatch):
    dp.register("numeric", lambda q, *, profile=None: _contract.make("v", engine="test"))
    monkeypatch.setenv("RAG_DET_PIPELINE_ROUTER", "1")
    assert dp.resolve("q", "numeric") is not None
    monkeypatch.setenv("RAG_DET_PIPELINE_ROUTER", "0")
    assert dp.resolve("q", "numeric") is None


# --------------------------------------------------------------------------- answer_question wiring
def test_router_off_is_byte_identical_even_with_a_registered_pipeline(monkeypatch):
    # flag OFF (default): a registered pipeline must NOT be consulted; the LLM loop answers as before.
    monkeypatch.delenv("RAG_DET_PIPELINE_ROUTER", raising=False)
    called = {"pipeline": False}

    def pipeline(q, *, profile=None):
        called["pipeline"] = True
        return _contract.make("決定論の値", engine="test")

    dp.register("numeric", pipeline)

    def fake_factory(question, tools, *, model=None, system=None):
        return _ScriptedModel([_submit("モデル主導の値", confidence=0.9)])

    monkeypatch.setattr(inv, "gemini_model_factory", fake_factory)
    res = inv.answer_question(_numeric_question(),
                              ledger=False, calc_ledger=False, research=False, enumeration=False)

    assert res.contract == qc.NUMERIC
    assert called["pipeline"] is False
    assert res.answer.answer == "モデル主導の値"
    assert res.model != "deterministic"


def test_router_empty_registry_routes_to_loop(monkeypatch):
    # flag ON but Stage0 registry empty ⇒ every question still routes to the LLM loop (byte-identical).
    monkeypatch.setenv("RAG_DET_PIPELINE_ROUTER", "1")

    def fake_factory(question, tools, *, model=None, system=None):
        return _ScriptedModel([_submit("モデル主導の値", confidence=0.9)])

    monkeypatch.setattr(inv, "gemini_model_factory", fake_factory)
    res = inv.answer_question(_numeric_question(),
                              ledger=False, calc_ledger=False, research=False, enumeration=False)

    assert res.contract == qc.NUMERIC
    assert res.answer.answer == "モデル主導の値"
    assert res.model != "deterministic"


def test_router_short_circuits_loop_when_pipeline_grounds_value(monkeypatch):
    # flag ON + a grounded pipeline for this contract ⇒ answer WITHOUT ever building the model.
    monkeypatch.setenv("RAG_DET_PIPELINE_ROUTER", "1")
    dp.register("numeric",
                lambda q, *, profile=None: _contract.make(
                    "12.3", engine="pandas", evidence={"file": "train.xlsx"}, confidence=0.95))

    def fake_factory(question, tools, *, model=None, system=None):
        raise AssertionError("LLM loop must not be entered when the router grounds an answer")

    monkeypatch.setattr(inv, "gemini_model_factory", fake_factory)
    res = inv.answer_question(_numeric_question(),
                              ledger=False, calc_ledger=False, research=False, enumeration=False)

    assert res.model == "deterministic"
    assert res.contract == qc.NUMERIC
    assert res.answer.answer == "12.3"
    assert res.answer.confidence == pytest.approx(0.95)
    assert res.tool_calls == ["det_pipeline:numeric"]
    assert "train.xlsx" in res.answer.evidence


def test_router_falls_back_to_loop_when_pipeline_ungrounded(monkeypatch):
    # flag ON + a pipeline that resolves nothing ⇒ fall back to the LLM loop (回答数を減らさない).
    monkeypatch.setenv("RAG_DET_PIPELINE_ROUTER", "1")
    dp.register("numeric", lambda q, *, profile=None: _contract.make(None, engine="test"))

    def fake_factory(question, tools, *, model=None, system=None):
        return _ScriptedModel([_submit("フォールバック値", confidence=0.8)])

    monkeypatch.setattr(inv, "gemini_model_factory", fake_factory)
    res = inv.answer_question(_numeric_question(),
                              ledger=False, calc_ledger=False, research=False, enumeration=False)

    assert res.model != "deterministic"
    assert res.answer.answer == "フォールバック値"
    assert res.tool_calls != ["det_pipeline:numeric"]
