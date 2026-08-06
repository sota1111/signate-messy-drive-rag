from __future__ import annotations

import pandas as pd
import pytest

from config import settings
from scoring import deterministic, synth
from src.rag import archetype, compute, generate
from src.rag.corpus import walk
from src.rag.tools import compute_sandbox


def _valid(idx: int) -> tuple[str, str]:
    q = pd.read_csv(settings.QUESTIONS_VALID).set_index("index")
    gt = pd.read_csv(settings.VALID_GROUND_TRUTH, header=None, index_col=0)
    return str(q.at[idx, "question"]), str(gt.at[idx, 1])


def test_valid_cross_aggregates_are_computed_exactly():
    for idx in (3, 8, 13):
        question, truth = _valid(idx)
        assert deterministic.score(compute.answer_question(question), truth, "numeric") == "Perfect"
        assert archetype.classify(question) == "cross_aggregate"


def test_generate_routes_computation_without_llm(monkeypatch):
    # SOT-2424: the compute hard module direct-commits only for a hold-out-validated archetype.
    monkeypatch.setattr(generate, "_load_trust",
                        lambda: {"cross_aggregate": {"holdout_validated": True}})
    question, truth = _valid(13)
    monkeypatch.setattr(generate.llm, "generate", lambda *a, **k: (_ for _ in ()).throw(AssertionError()))
    result = generate.answer_question(question)
    assert result["answer"] == truth
    assert result["verified"] is True


def test_unresolved_compute_question_abstains(monkeypatch):
    monkeypatch.setattr(generate, "_load_trust",
                        lambda: {"cross_aggregate": {"holdout_validated": True}})
    result = generate.answer_question("存在しない案件のfoo列の平均値を算出してください。")
    assert result["answer"] == settings.ABSTAIN
    assert result["confidence"] == "compute-unresolved"


def test_synth_contains_cross_aggregate_items():
    items = []
    synth.gen_cross_aggregate(items)
    assert len(items) == 3
    assert all(compute.answer_question(i.question) == i.truth for i in items)


def test_unconditional_csv_column_stats_are_computed_exactly():
    items = []
    synth.gen_csv(items)
    assert items
    for item in items:
        assert deterministic.score(
            compute.column_stat_answer(item.question), item.truth, "numeric"
        ) == "Perfect"


def test_csv_column_stat_rejects_conditional_questions():
    items = []
    synth.gen_csv(items)
    question = items[0].question.replace("の平均値", "が10以上のときの平均値")
    assert compute.column_stat_answer(question) is None


def test_generate_routes_validated_csv_stat_without_llm(monkeypatch):
    items = []
    synth.gen_csv(items)
    item = next(i for i in items if i.archetype == "csv_column_mean")
    monkeypatch.setattr(
        generate,
        "_load_trust",
        lambda: {"csv_column_mean": {"trust": True, "holdout_validated": True}},
    )
    monkeypatch.setattr(
        generate.llm,
        "generate",
        lambda *a, **k: (_ for _ in ()).throw(AssertionError()),
    )
    result = generate.answer_question(item.question)
    assert deterministic.score(result["answer"], item.truth, "numeric") == "Perfect"
    assert result["verified"] is True


# --------------------------------------------------------------------------- compute_sandbox (SOT-2461)
def _canonical_train_csv():
    """First project whose canonical ``03.データ/train.csv`` exists (skip the ML-repo copy)."""
    refs = [r for r in walk() if r.ext == "csv" and r.name == "train.csv"
            and r.rel.endswith("03.データ/train.csv")]
    return refs[0] if refs else None


def test_compute_sandbox_matches_gold_aggregates_exactly():
    """Representative aggregate/average/count問 が同一ファイルの pandas ゴールドと厳密一致."""
    ref = _canonical_train_csv()
    if ref is None:
        pytest.skip("no canonical train.csv in corpus")
    df = pd.read_csv(ref.path)
    numeric = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c]) and c.lower() != "id"]
    assert numeric, "expected at least one numeric column"
    col = numeric[0]
    cases = {
        f"df['{col}'].mean()": float(df[col].mean()),
        f"df['{col}'].max()": float(df[col].max()),
        "len(df)": len(df),
        f"int((df['{col}'] > df['{col}'].mean()).sum())": int((df[col] > df[col].mean()).sum()),
    }
    for expr, gold in cases.items():
        out = compute_sandbox.run(ref.rel, expr)
        assert out["value"] == gold, f"{expr}: {out['value']!r} != gold {gold!r}"


def test_compute_sandbox_matches_known_gold_value():
    """Hand-verified known aggregation: かえで総合病院 train.csv の Age 平均 = 45.325."""
    refs = [r for r in walk() if r.ext == "csv" and r.name == "train.csv"
            and "かえで総合病院" in r.project and r.rel.endswith("03.データ/train.csv")]
    if not refs:
        pytest.skip("かえで総合病院 train.csv not present")
    out = compute_sandbox.run(refs[0].rel, "round(df['Age'].mean(), 3)")
    assert out["value"] == 45.325


def test_compute_sandbox_returns_uniform_contract():
    ref = _canonical_train_csv()
    if ref is None:
        pytest.skip("no canonical train.csv in corpus")
    col = next(c for c in pd.read_csv(ref.path).columns
               if pd.api.types.is_numeric_dtype(pd.read_csv(ref.path)[c]))
    out = compute_sandbox.run(
        ref.rel, f"round(df['{col}'].mean(), 2)",
        intermediates={"n": "len(df)", "total": f"round(float(df['{col}'].sum()), 2)"},
    )
    assert set(out) == {"value", "evidence", "method"}
    ev = out["evidence"]
    assert ev["file"] == ref.rel
    assert ev["rows"] == len(pd.read_csv(ref.path))
    assert col in ev["columns"] and col in ev["columns_used"]
    assert ev["range"]  # A1-style bounding box of the touched column(s)
    m = out["method"]
    assert m["engine"] == "pandas"
    assert m["code"] == f"round(df['{col}'].mean(), 2)"
    assert m["intermediates"]["n"] == len(pd.read_csv(ref.path))
    assert "df" in m["allowed_api"] and "pd" in m["allowed_api"]


def test_compute_sandbox_reads_xlsx_data_only_and_picks_data_sheet():
    """xlsx は data_only で読み、空の active(グラフ) ではなく実データ(train)シートを選ぶ."""
    xrefs = [r for r in walk() if r.ext == "xlsx" and r.name == "train.xlsx"]
    if not xrefs:
        pytest.skip("no train.xlsx in corpus")
    ref = xrefs[0]
    out = compute_sandbox.run(ref.rel, "len(df)")
    assert out["value"] > 0, "must not read the empty chart sheet"
    assert out["evidence"]["sheet"] not in (None, "グラフ")
    # the xlsx data sheet mirrors the sibling csv row count where both exist
    csv = [r for r in walk() if r.ext == "csv" and r.name == "train.csv"
           and r.project == ref.project and r.rel.endswith("03.データ/train.csv")]
    if csv:
        assert out["value"] == len(pd.read_csv(csv[0].path))


@pytest.mark.parametrize("expr", [
    "__import__('os').system('echo hi')",
    "open('/etc/passwd').read()",
    "df.__class__.__mro__",
    "df.values.__class__",
    "[x for x in range(3)]",
    "(lambda: 1)()",
])
def test_compute_sandbox_rejects_unsafe_expressions(expr):
    ref = _canonical_train_csv()
    if ref is None:
        pytest.skip("no canonical train.csv in corpus")
    with pytest.raises(compute_sandbox.ComputeError):
        compute_sandbox.run(ref.rel, expr)


def test_compute_sandbox_resolves_nfc_reference_by_name():
    ref = _canonical_train_csv()
    if ref is None:
        pytest.skip("no canonical train.csv in corpus")
    # resolving by corpus-relative string (NFC path) yields the same file as the FileRef itself
    by_ref = compute_sandbox.run(ref, "len(df)")
    by_rel = compute_sandbox.run(ref.rel, "len(df)")
    assert by_ref["value"] == by_rel["value"]


# --------------------------------------------------------------------------- SOT-2487: project scope
def _ambiguous_bare_name() -> tuple[str, list] | None:
    """A bare filename (e.g. ``train.xlsx``) shared by ≥2 projects, with its FileRefs."""
    from collections import Counter

    refs = [r for r in walk() if r.project]
    counts = Counter(r.name for r in refs)
    for name in ("train.xlsx", "train.csv"):
        if counts.get(name, 0) >= 2:
            return name, [r for r in refs if r.name == name]
    return None


def test_compute_project_scope_disambiguates_shared_filename():
    """SOT-2487: a project= hint narrows an otherwise-ambiguous shared table to one file."""
    picked = _ambiguous_bare_name()
    if picked is None:
        pytest.skip("no shared bare table name across projects")
    name, refs = picked
    target = refs[0]
    # Bare name alone is ambiguous and must NOT silently pick a file (precision guard).
    with pytest.raises(compute_sandbox.ComputeError) as exc:
        compute_sandbox._resolve(name)
    # the error steers the caller to the project= filter and lists candidate projects
    assert "存在プロジェクト" in str(exc.value)
    # a project hint (even a partial company name) resolves to that project's own table
    resolved = compute_sandbox._resolve(name, target.project)
    assert resolved.rel == target.rel
    assert resolved.project == target.project


def test_compute_project_scope_partial_name_matches():
    """A loose project hint (substring of the full legal name) still resolves."""
    picked = _ambiguous_bare_name()
    if picked is None:
        pytest.skip("no shared bare table name across projects")
    name, refs = picked
    target = refs[0]
    # drop a common legal-form prefix to simulate how a question names the project loosely
    partial = target.project.replace("株式会社", "").replace("医療法人社団", "").strip()
    if not partial or partial == target.project:
        partial = target.project[-4:]
    resolved = compute_sandbox._resolve(name, partial)
    assert resolved.project == target.project
    # and compute runs end-to-end against the scoped frame
    out = compute_sandbox.run(name, "df.shape[0]", project=partial)
    assert isinstance(out["value"], int) and out["value"] >= 0
    assert out["evidence"]["project"] == target.project


def test_compute_project_scope_no_match_raises_not_guesses():
    """A non-matching project errors (→ abstain) rather than falling back to a wrong file."""
    picked = _ambiguous_bare_name()
    if picked is None:
        pytest.skip("no shared bare table name across projects")
    name, _ = picked
    with pytest.raises(compute_sandbox.ComputeError):
        compute_sandbox._resolve(name, "存在しない架空の案件名XYZ")


def test_compute_project_scope_is_backward_compatible():
    """Unscoped resolution of a fully-qualified rel is byte-identical with/without project=None."""
    ref = _canonical_train_csv()
    if ref is None:
        pytest.skip("no canonical train.csv in corpus")
    a = compute_sandbox.run(ref.rel, "len(df)")
    b = compute_sandbox.run(ref.rel, "len(df)", project=None)
    assert a == b
