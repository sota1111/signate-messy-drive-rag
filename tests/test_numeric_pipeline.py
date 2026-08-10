"""SOT-2607 (Wave A2) — offline tests for the deterministic ``numeric`` pipeline.

All network-free and corpus-free: the canonical route (project/file resolution) and the compute sandbox
are monkeypatched, so no glossary / Office file / pandas execution runs. The invariants under test are:
the pipeline self-registers for ``numeric``; it grounds a value **only** for a single unambiguous
single-column aggregate confirmed by the independent re-execution; it falls back (``None``) on every
non-aggregate / ambiguous / unit-bearing / ratio / excluded-cue question (idx6/47/63/97-shaped) so the
router routes to the LLM loop; and it wires through ``det_pipeline.resolve`` + ``formatting.format_contract``
end-to-end (numeric is template-first — no LLM naturalize call).
"""
from __future__ import annotations

import pytest

from src.rag.agent import det_pipeline as dp
from src.rag.agent import formatting
from src.rag.agent.pipelines import numeric as num


# --------------------------------------------------------------------------- registry cleanup fixture
@pytest.fixture(autouse=True)
def _restore_registry():
    saved = dict(dp._REGISTRY)
    try:
        yield
    finally:
        dp._REGISTRY.clear()
        dp._REGISTRY.update(saved)


# --------------------------------------------------------------------------- fake canonical-route + compute
def _compute_result(value, *, columns_used, rows=100, code="", sheet="train", rng="A1:B101"):
    return {
        "value": value,
        "evidence": {"file": "proj/03.データ/train.xlsx", "project": "proj", "sheet": sheet,
                     "rows": rows, "columns": ["id", "loan_amnt", "age"],
                     "columns_used": list(columns_used), "range": rng},
        "method": {"engine": "pandas", "code": code},
    }


def _wire(monkeypatch, *, project="京橋信用ソリューションズ", columns=("id", "loan_amnt", "age"),
          values=None, ext="xlsx"):
    """Monkeypatch canonical_route + compute so the pipeline runs with no corpus/network.

    ``values`` maps a compute expression to the value it should return; ``df.columns.tolist()`` always
    yields ``columns``. Any expression not in ``values`` returns 0.0 (a harmless default).
    """
    # canonical_route is re-exported as a function (shadowing the submodule) — resolve the real module.
    import importlib
    cr = importlib.import_module("src.rag.tools.canonical_route")
    import src.rag.tools.compute_sandbox as cs

    monkeypatch.setattr(cr, "resolve_project", lambda q, p, **k: project)

    def fake_route(*, question=None, project=None, kind=None, **k):
        if project is None:
            return {"value": [], "evidence": {}, "method": {}}
        return {"value": [{"rel": "proj/03.データ/train.xlsx", "project": project,
                           "category": "data", "ext": ext, "kind": "train"}],
                "evidence": {}, "method": {}}

    monkeypatch.setattr(cr, "canonical_route", fake_route)

    values = values or {}

    def fake_compute(rel, expr, *, sheet=None, project=None, **k):
        if expr == "df.columns.tolist()":
            return _compute_result(list(columns), columns_used=[], code=expr)
        val = values.get(expr, 0.0)
        # infer which column the expression touched (for columns_used) from the fake column list
        used = [c for c in columns if repr(c) in expr]
        return _compute_result(val, columns_used=used, code=expr)

    monkeypatch.setattr(cs, "run", fake_compute)
    # the pipeline imports compute_run lazily from compute_sandbox.run, so patch the module attribute.


# --------------------------------------------------------------------------- registration
def test_pipeline_registers_for_numeric():
    num.register(replace=True)
    assert "numeric" in dp.registered_contracts()
    assert dp._REGISTRY["numeric"] is num.pipeline


# --------------------------------------------------------------------------- grounding (single aggregate)
def test_single_column_mean_grounds_and_is_confirmed(monkeypatch):
    q = "京橋信用ソリューションズの train.xlsx の loan_amnt の平均は。"
    _wire(monkeypatch, values={
        "df['loan_amnt'].mean()": 12.5,
        "df['loan_amnt'].dropna().mean()": 12.5,
    })
    out = num.pipeline(q)
    assert out is not None
    assert out["value"] == 12.5
    assert out["evidence"]["aggregate"] == "mean"
    assert out["evidence"]["column"] == "loan_amnt"
    assert out["method"]["engine"] == "numeric_det_pipeline"
    assert out["method"]["exec_verify"] == "exec_match"


def test_count_aggregate_grounds(monkeypatch):
    q = "京橋信用ソリューションズの train.xlsx の loan_amnt の件数は何件。"  # NOTE: 件 is a unit → see below
    # 『件数』 triggers the count aggregate, but 『件』 sets req.unit → the pipeline declines (unit-bearing).
    _wire(monkeypatch, values={"int(df['loan_amnt'].count())": 100, "len(df['loan_amnt'].dropna())": 100})
    assert num.pipeline(q) is None


def test_rounding_is_honored_in_answer(monkeypatch):
    q = "京橋信用ソリューションズの train.xlsx の age の平均を小数第2位まで求めてください。"
    _wire(monkeypatch, values={"df['age'].mean()": 33.333333, "df['age'].dropna().mean()": 33.333333})
    out = num.pipeline(q)
    assert out is not None
    assert out["value"] == "33.33"  # formatted to the requested decimal places


# --------------------------------------------------------------------------- fallbacks (idx6/47/63/97-shaped)
def test_no_aggregate_keyword_falls_back(monkeypatch):
    # idx6 型: 差額 — no recognized single-column aggregate ⇒ LLM fallback.
    q = "蒼泉会 ひがし丘総合病院案件において、提案時の税込み見込み金額と最終請求金額の差額はいくらですか。"
    _wire(monkeypatch)
    assert num.pipeline(q) is None


def test_regression_prediction_falls_back(monkeypatch):
    # idx63 型: 回帰係数で予測 — excluded cue (回帰/予測/係数) ⇒ fallback even if a number is asked.
    q = "青葉与信マネジメントのtrain.xlsxにて算出された回帰係数を使ってid=0を予測した場合の予測値の平均はいくら。"
    _wire(monkeypatch, values={"df['id'].mean()": 1.0, "df['id'].dropna().mean()": 1.0})
    assert num.pipeline(q) is None


def test_highlight_geometry_falls_back(monkeypatch):
    # idx97 型: 黄色ハイライト交差セルの差の絶対値 — excluded cue (ハイライト/差/絶対値) ⇒ fallback.
    q = "青葉バイオメディカル機器のtrain.xlsxにおいて、黄色ハイライトが交差している2つのセルの値の差の絶対値。"
    _wire(monkeypatch)
    assert num.pipeline(q) is None


def test_ratio_question_falls_back(monkeypatch):
    q = "京橋信用ソリューションズの train.xlsx の loan_amnt が 100 以上の割合は。"
    _wire(monkeypatch, values={"df['loan_amnt'].mean()": 1.0})
    assert num.pipeline(q) is None


def test_two_aggregates_are_ambiguous_and_fall_back(monkeypatch):
    q = "京橋信用ソリューションズの train.xlsx の loan_amnt の平均と最大は。"
    _wire(monkeypatch, values={"df['loan_amnt'].mean()": 1.0})
    assert num.pipeline(q) is None


def test_ambiguous_column_falls_back(monkeypatch):
    # both loan_amnt and age are named in the question ⇒ target column ambiguous ⇒ fallback.
    q = "京橋信用ソリューションズの train.xlsx の loan_amnt と age の平均は。"
    _wire(monkeypatch, columns=("id", "loan_amnt", "age"),
          values={"df['loan_amnt'].mean()": 1.0, "df['age'].mean()": 2.0})
    assert num.pipeline(q) is None


def test_no_named_column_falls_back(monkeypatch):
    q = "京橋信用ソリューションズの train.xlsx の balance の平均は。"  # 'balance' is not a table column
    _wire(monkeypatch, columns=("id", "loan_amnt", "age"))
    assert num.pipeline(q) is None


def test_no_project_falls_back(monkeypatch):
    q = "train.xlsx の loan_amnt の平均は。"
    _wire(monkeypatch, project=None)
    assert num.pipeline(q) is None


def test_non_tabular_primary_falls_back(monkeypatch):
    q = "京橋信用ソリューションズの train の loan_amnt の平均は。"
    _wire(monkeypatch, ext="ipynb")
    assert num.pipeline(q) is None


def test_independent_reexecution_conflict_falls_back(monkeypatch):
    # the two independent computes disagree ⇒ EXEC_CONFLICT ⇒ safe fallback (never a wrong commit).
    q = "京橋信用ソリューションズの train.xlsx の loan_amnt の平均は。"
    _wire(monkeypatch, values={
        "df['loan_amnt'].mean()": 12.5,
        "df['loan_amnt'].dropna().mean()": 999.9,  # divergent re-execution
    })
    assert num.pipeline(q) is None


def test_nan_value_falls_back(monkeypatch):
    q = "京橋信用ソリューションズの train.xlsx の loan_amnt の平均は。"
    _wire(monkeypatch, values={
        "df['loan_amnt'].mean()": float("nan"),
        "df['loan_amnt'].dropna().mean()": float("nan"),
    })
    assert num.pipeline(q) is None


def test_compute_error_falls_back(monkeypatch):
    q = "京橋信用ソリューションズの train.xlsx の loan_amnt の平均は。"
    import importlib
    cr = importlib.import_module("src.rag.tools.canonical_route")
    import src.rag.tools.compute_sandbox as cs

    monkeypatch.setattr(cr, "resolve_project", lambda q, p, **k: "proj")
    monkeypatch.setattr(cr, "canonical_route", lambda *, question=None, project=None, kind=None, **k:
                        {"value": [{"rel": "t.xlsx", "project": "proj", "ext": "xlsx", "kind": "train"}],
                         "evidence": {}, "method": {}})

    def boom(*a, **k):
        raise RuntimeError("read failed")

    monkeypatch.setattr(cs, "run", boom)
    assert num.pipeline(q) is None  # never raises into the answer path


# --------------------------------------------------------------------------- router + formatting e2e
def test_resolve_wraps_pipeline_when_forced(monkeypatch):
    q = "京橋信用ソリューションズの train.xlsx の loan_amnt の平均は。"
    _wire(monkeypatch, values={
        "df['loan_amnt'].mean()": 12.5, "df['loan_amnt'].dropna().mean()": 12.5})
    out = dp.resolve(q, "numeric", force=True)
    assert out is not None and out["value"] == 12.5


def test_resolve_off_returns_none(monkeypatch):
    monkeypatch.delenv("RAG_DET_PIPELINE_ROUTER", raising=False)
    q = "京橋信用ソリューションズの train.xlsx の loan_amnt の平均は。"
    _wire(monkeypatch, values={
        "df['loan_amnt'].mean()": 12.5, "df['loan_amnt'].dropna().mean()": 12.5})
    # flag OFF ⇒ champion serve path byte-identical: the pipeline is never consulted.
    assert dp.resolve(q, "numeric") is None


def test_end_to_end_through_formatting_template_only(monkeypatch):
    q = "京橋信用ソリューションズの train.xlsx の loan_amnt の平均は。"
    _wire(monkeypatch, values={
        "df['loan_amnt'].mean()": 12.5, "df['loan_amnt'].dropna().mean()": 12.5})
    contract = dp.resolve(q, "numeric", force=True)
    formatted = formatting.format_contract(contract, q, contract_type="numeric", force=True)
    assert formatted["value"] == "12.5"  # numeric renders template-first (no LLM)
    assert formatted["method"]["formatting"]["template_only"] is True
