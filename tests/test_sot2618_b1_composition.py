"""SOT-2618 (Wave A + B1 composition) — B1/B2 split into independent sub-gates under the master router.

This is the focused, network-free verification for the adopted **Wave A + B1** composition: with the
master router ON and the per-Wave defaults (B1 ON, B2 OFF),

* B1 (document_extract / format_check) keeps grounding the highlighted-pivot extraction conditions
  (idx7/15/80 型) — reproducing SOT-2613's local win (match17/wrong1 相当), and
* B2 (fact_lookup / simple_lookup) falls back to the LLM loop (returns ``None`` from the router) —
  removing SOT-2613's regression source so simple_lookup returns to Wave A level,

while B2 stays recoverable (``RAG_DET_PIPELINE_B2=1`` grounds it again) and the router-OFF serve path
is byte-identical. The pure sub-gate mechanics live in ``tests/test_det_pipeline.py``; the corpus-gated
cases here prove the composition against the real Office corpus when present.
"""
from __future__ import annotations

import os

import pytest

from src.rag.agent import det_pipeline as dp
from src.rag.agent import formatting
from src.rag.agent import pipelines as _pipelines  # noqa: F401 — force self-registration


@pytest.fixture(autouse=True)
def _restore_registry():
    dp.registered_contracts()  # ensure the real pipelines are bootstrapped before snapshotting
    saved = dict(dp._REGISTRY)
    try:
        yield
    finally:
        dp._REGISTRY.clear()
        dp._REGISTRY.update(saved)


# --------------------------------------------------------------------------- sub-gate wiring (corpus-free)
def test_b1_and_b2_are_registered_but_independently_gated(monkeypatch):
    # Both Wave B pipelines self-register; the split is at the gate, not the registry.
    contracts = dp.registered_contracts()
    assert "format_check" in contracts and "simple_lookup" in contracts
    monkeypatch.delenv("RAG_DET_PIPELINE_B1", raising=False)
    monkeypatch.delenv("RAG_DET_PIPELINE_B2", raising=False)
    assert dp.wave_enabled("format_check") is True   # B1 ON by default (adopted)
    assert dp.wave_enabled("simple_lookup") is False  # B2 OFF by default (regression source)


def test_router_off_is_byte_identical_for_both_waves(monkeypatch):
    # Master router OFF ⇒ neither Wave B pipeline runs regardless of its own flag (serve path unchanged).
    monkeypatch.delenv("RAG_DET_PIPELINE_ROUTER", raising=False)
    monkeypatch.setenv("RAG_DET_PIPELINE_B1", "1")
    monkeypatch.setenv("RAG_DET_PIPELINE_B2", "1")
    assert dp.resolve("黄色にハイライトされたセルの抽出条件と集計", "format_check") is None
    assert dp.resolve("フェーズ3のタスク名は何ですか", "simple_lookup") is None


# --------------------------------------------------------------------------- corpus-gated composition
_CORPUS_PRESENT = (
    __import__("config").settings.CORPUS_DIR.exists()
    if os.getenv("RAG_SKIP_CORPUS_TESTS") not in {"1", "true", "yes", "on"} else False
)

# idx7/15/80 (B1 document_extract) — same fixtures as tests/test_document_extract_pipeline.py.
_B1_CASES = [
    ("青潮モビリティサービスの基礎分析.pptxにおいて、黄色ハイライトされている数値に対応するデータの"
     "抽出条件と集計内容を答えてください。",
     "hr=8、weekday=2で抽出されたデータに対する最大 / temp"),
    ("東都人材プラットフォームのtrain.xlsxにおいて、Sheet1の黄色にハイライトされたセルの抽出条件と"
     "集計内容を答えてください。",
     "Gender=Male、target=2、Age=40-44、Country=Spainで抽出されたデータに対する個数"),
    ("東都人材プラットフォームのtrain.xlsxにおいて、Sheet2の黄色にハイライトされたセルの抽出条件と"
     "集計内容を答えてください。",
     "Gender=Male、target=3、Age=30-34、Profession=Software Engineerで抽出されたデータに対する個数"),
]


@pytest.mark.skipif(not _CORPUS_PRESENT, reason="corpus (data/share_drive) not present")
@pytest.mark.parametrize("question,gold", _B1_CASES)
def test_wave_a_plus_b1_grounds_document_extract(monkeypatch, question, gold):
    # Router ON + default gates (B1 ON, B2 OFF): B1 still grounds idx7/15/80 (reproduces SOT-2613 win).
    monkeypatch.setenv("RAG_DET_PIPELINE_ROUTER", "1")
    monkeypatch.delenv("RAG_DET_PIPELINE_B1", raising=False)
    monkeypatch.delenv("RAG_DET_PIPELINE_B2", raising=False)
    contract = dp.resolve(question, "format_check")
    assert contract is not None, "B1 should still ground under the Wave A + B1 composition"
    formatted = formatting.format_contract(contract, question, contract_type="format_check", force=True)
    assert formatted["value"] == gold


@pytest.mark.skipif(not _CORPUS_PRESENT, reason="corpus (data/share_drive) not present")
def test_wave_a_plus_b1_routes_fact_lookup_to_llm(monkeypatch):
    # Router ON + B2 OFF (default): the fact_lookup pipeline does NOT short-circuit — simple_lookup falls
    # back to the LLM loop (Wave A level), dropping SOT-2613's regression. idx89 型 question.
    q = "青葉バイオメディカル機器のスケジュールで、フェーズ3のタスク名は何ですか。"
    monkeypatch.setenv("RAG_DET_PIPELINE_ROUTER", "1")
    monkeypatch.delenv("RAG_DET_PIPELINE_B2", raising=False)
    assert dp.resolve(q, "simple_lookup") is None
    # …and it is recoverable: with B2 explicitly ON, the pipeline is consulted again.
    monkeypatch.setenv("RAG_DET_PIPELINE_B2", "1")
    # (grounding depends on the corpus fixture; we only assert the gate now lets the pipeline run.)
    assert dp.wave_enabled("simple_lookup") is True
