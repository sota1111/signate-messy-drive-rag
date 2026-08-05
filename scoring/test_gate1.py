"""Network-free regressions for the 関門1 valid30 gate (SOT-2475).

valid30 is scored through the production Gemini investigation agent (``gen='investigator'``, the same
backend as the real submission via ``src.rag.run``). These tests pin the wiring — default backend,
gen forwarding into ``runner.run`` — and the scoring path, with a stub runner + stub judge so no LLM /
network / real files are touched.
"""
import inspect

import pandas as pd

from scoring import crag
from scoring import gate1 as G
from src.rag import run as runner
from src.rag.run import DEFAULT_GEN, GEN_CHOICES


def _write_headerless(path, mapping):
    pd.DataFrame(list(mapping.items())).to_csv(path, header=False, index=False)


def test_default_backend_is_the_production_investigator_path():
    """gate1 defaults to the production Gemini agent, not the legacy text path (SOT-2475)."""
    assert inspect.signature(G.main).parameters["gen"].default == "investigator"
    # The CLI shares the exact backend menu with the real submission pipeline, so it can never diverge.
    assert DEFAULT_GEN == "investigator"
    assert "investigator" in GEN_CHOICES


def test_missing_preds_solves_valid_with_the_selected_agent(tmp_path, monkeypatch):
    """When predictions are missing gate1 runs the RAG on the valid split forwarding ``gen`` verbatim,
    then scores the produced predictions against the official GT."""
    gt = {1: "alpha", 2: "beta"}
    gt_csv = tmp_path / "valid_txt.csv"
    _write_headerless(gt_csv, gt)
    monkeypatch.setattr(G.settings, "VALID_GROUND_TRUTH", gt_csv)

    preds_path = tmp_path / "predictions_valid.csv"
    calls = {}

    def fake_run(split, out, limit, workers, hard, gen):
        calls.update(split=split, gen=gen, out=out)
        _write_headerless(out, {1: "alpha", 2: "wrong"})

    monkeypatch.setattr(runner, "run", fake_run)

    scored = {}

    def fake_score_pairs(pairs):
        scored["pairs"] = list(pairs)
        results = [{"judged": "Perfect" if p == t else "Incorrect",
                    "points": 1.0 if p == t else -1.0, "pred": p, "truth": t} for p, t in pairs]
        return sum(r["points"] for r in results) / len(results), results

    monkeypatch.setattr(crag, "score_pairs", fake_score_pairs)
    monkeypatch.setattr(G.settings, "ARTIFACTS_DIR", tmp_path)

    G.main(preds_path, run_first=True, gen="investigator")

    assert calls["split"] == "valid"
    assert calls["gen"] == "investigator"          # gen forwarded to the production runner
    # scored the produced predictions against the official GT, GT-index order
    assert scored["pairs"] == [("alpha", "alpha"), ("wrong", "beta")]
    out = pd.read_csv(tmp_path / "gate1_scoring.csv")
    assert list(out["judged"]) == ["Perfect", "Incorrect"]


def test_existing_preds_are_scored_without_running(tmp_path, monkeypatch):
    """With predictions already on disk gate1 scores them directly (no RAG run)."""
    gt_csv = tmp_path / "valid_txt.csv"
    _write_headerless(gt_csv, {1: "alpha"})
    monkeypatch.setattr(G.settings, "VALID_GROUND_TRUTH", gt_csv)
    monkeypatch.setattr(G.settings, "ARTIFACTS_DIR", tmp_path)

    preds_path = tmp_path / "preds.csv"
    _write_headerless(preds_path, {1: "alpha"})

    def boom(*a, **k):  # must not be reached — preds already exist
        raise AssertionError("runner.run must not be called when predictions exist")

    monkeypatch.setattr(runner, "run", boom)
    monkeypatch.setattr(crag, "score_pairs",
                        lambda pairs: (1.0, [{"judged": "Perfect", "points": 1.0,
                                              "pred": p, "truth": t} for p, t in pairs]))

    G.main(preds_path, run_first=True, gen="investigator")
    assert (tmp_path / "gate1_scoring.csv").exists()
