"""Offline tests for the canonical_route tool (no LLM / network) — SOT-2494.

    .venv/bin/python -m pytest scoring/test_canonical_route.py -q

Two layers:

* **Synthetic mini-corpus** (always runs): builds a tiny share-drive-shaped drive of two projects, each
  with a train.xlsx / train.csv, an analysis src/modeling.py + notebooks/01_eda.ipynb + leaderboard.csv,
  plus a hand-built glossary — then pins project resolution, kind inference, canonical ranking, the
  ``.xlsx`` extension hint, and the direct-compute (``expr``) route. It uses **only** the injected
  mini-drive + glossary, proving the tool hard-codes no corpus-specific value (受け入れ条件②・移植性).
* **Real-corpus gold pins** (skipped when the corpus is absent): the SOT-2486 retrieval_miss frontier —
  idx61 (analysis code → modeling.py) and idx80 / idx97 (train.xlsx data questions) — resolve their
  canonical target 100% deterministically, and the seating/aggregate cases the SOT-2486 needle heuristic
  false-attributed to train.csv (idx13/46/58) are correctly *declined* (they are answered by the existing
  seating_lookup / corpus_aggregate tools, not this data-file route).
"""
from __future__ import annotations

import importlib
from pathlib import Path

import pytest

from src.rag import corpus
from src.rag.extract.glossary import Glossary
from src.rag.tools import contract as C

# The package __init__ re-exports the ``canonical_route`` *function*, shadowing the submodule attribute;
# import the module explicitly so ``cr`` is the module (not the function) — same guard as corpus_aggregate.
cr = importlib.import_module("src.rag.tools.canonical_route")

_CORPUS_PRESENT = bool(corpus.walk())
_needs_corpus = pytest.mark.skipif(not _CORPUS_PRESENT, reason="corpus not present")


# --------------------------------------------------------------------------- synthetic mini-corpus
def _write_train_csv(path: Path, rows: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    body = "\n".join(f"{i},{i * 10}" for i in range(1, rows + 1))
    path.write_text("id,value\n" + body + "\n", encoding="utf-8")


def _write_train_xlsx(path: Path, rows: int) -> None:
    from openpyxl import Workbook

    path.parent.mkdir(parents=True, exist_ok=True)
    wb = Workbook()
    ws = wb.active
    ws.title = "train"
    ws.append(["id", "value"])
    for i in range(1, rows + 1):
        ws.append([i, i * 10])
    wb.save(str(path))


def _write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def _build_project(root: Path, folder: str, *, csv_rows: int, xlsx_rows: int) -> None:
    """Lay out one project the way the real drive does: 03.データ + 04.分析/analysis_project/*."""
    base = root / "プロジェクト" / folder
    _write_train_csv(base / "03.データ" / "train.csv", csv_rows)
    _write_train_xlsx(base / "03.データ" / "train.xlsx", xlsx_rows)
    ap = base / "04.分析" / "analysis_project"
    _write_train_csv(ap / "data" / "train.csv", csv_rows)
    _write_text(ap / "src" / "modeling.py",
                "def build_model():\n    return dict(n_estimators=300, learning_rate=0.1)\n")
    _write_text(ap / "src" / "features.py", "def make_features(df):\n    return df\n")
    _write_text(ap / "scripts" / "run_train.py", "from src.modeling import build_model\n")
    _write_text(ap / "notebooks" / "01_eda.ipynb", '{"cells": [], "metadata": {}}\n')
    _write_text(ap / "notebooks" / "01_eda_old.ipynb", '{"cells": [], "metadata": {}}\n')
    _write_text(base / "04.分析" / "analysis_outputs" / "experiments" / "leaderboard.csv",
                "model,score\ngbdt,0.9\n")


# Two projects; the glossary maps a loose alias ("ACME" / "つばき") to each canonical folder name.
_ACME = "エイコー商事株式会社"
_TSUBAKI = "医療法人 つばき記念クリニック"


@pytest.fixture()
def mini(tmp_path: Path) -> tuple[Path, Glossary]:
    root = tmp_path / "share_drive"
    _build_project(root, _ACME, csv_rows=7, xlsx_rows=5)
    _build_project(root, _TSUBAKI, csv_rows=11, xlsx_rows=9)
    corpus.walk.cache_clear()  # the mini-drive is a fresh path; drop any real-corpus memoization

    g = Glossary()
    g.company_aliases = {_ACME: [_ACME, "ACME", "エイコー"], _TSUBAKI: [_TSUBAKI, "つばき"]}
    for company, aliases in g.company_aliases.items():
        for a in aliases:
            g.case_to_company.setdefault(a, company)
    return root, g


def _route(mini, question, **kw):
    root, g = mini
    return cr.canonical_route(question, corpus_dir=root, glossary=g, **kw)


# --------------------------------------------------------------------------- project resolution
def test_resolve_project_from_alias(mini):
    root, g = mini
    assert cr.resolve_project("ACMEのtrain.xlsxを見たい", None, corpus_dir=root, glossary=g) == _ACME
    assert cr.resolve_project("つばきの分析コード", None, corpus_dir=root, glossary=g) == _TSUBAKI


def test_resolve_project_explicit_hint_wins(mini):
    root, g = mini
    # A partial folder name resolves symmetrically, independent of the glossary.
    assert cr.resolve_project(None, "つばき記念", corpus_dir=root, glossary=g) == _TSUBAKI


def test_resolve_project_none_for_unknown(mini):
    root, g = mini
    assert cr.resolve_project("どこの案件でもない全体集計", None, corpus_dir=root, glossary=g) is None


# --------------------------------------------------------------------------- kind inference
def test_infer_kinds_code_vs_train_vs_notebook():
    assert cr.infer_kinds("分析コードの n_estimators は？")[0].name == "code"
    assert cr.infer_kinds("train.xlsx の集計")[0].name == "train"
    assert cr.infer_kinds("EDA ノートブックのセル")[0].name == "notebook"
    assert cr.infer_kinds("leaderboard のスコア")[0].name == "leaderboard"
    assert cr.infer_kinds("向かいの人の内線番号") == []  # a seating question signals no canonical file


# --------------------------------------------------------------------------- canonical resolution
def test_route_code_resolves_modeling_first(mini):
    r = _route(mini, "ACMEの分析コードで実際に渡される n_estimators と learning_rate は？")
    assert C.is_contract(r)
    assert r["evidence"]["project"] == _ACME
    assert r["evidence"]["resolved_kinds"] == ["code"]
    assert r["evidence"]["primary"].endswith("src/modeling.py")  # canonical over run_train.py/features.py


def test_route_train_xlsx_extension_hint(mini):
    # "train.xlsx" (even before Japanese text) narrows the multi-ext train kind to the .xlsx copy.
    r = _route(mini, "ACMEのtrain.xlsxにおいて黄色ハイライトのセル")
    assert r["evidence"]["primary"].endswith("03.データ/train.xlsx")
    assert all(rec["ext"] == "xlsx" for rec in r["value"])  # csv copies excluded by the hint


def test_route_train_prefers_canonical_data_folder(mini):
    # No ext hint → both csv/xlsx match, but the 03.データ canonical copy sorts ahead of analysis_project/data.
    r = _route(mini, "つばきの学習データの行数")
    assert r["evidence"]["primary"].startswith("プロジェクト/" + _TSUBAKI + "/03.データ/")


def test_route_notebook_excludes_old_variant(mini):
    r = _route(mini, "ACMEのEDAノートブック")
    assert r["evidence"]["primary"].endswith("notebooks/01_eda.ipynb")
    # the _old variant is de-prioritized, never the primary
    assert "01_eda_old" not in r["evidence"]["primary"]


def test_route_leaderboard(mini):
    r = _route(mini, "つばきの leaderboard のスコア")
    assert r["evidence"]["primary"].endswith("experiments/leaderboard.csv")


def test_route_explicit_kind_override(mini):
    # kind= overrides keyword inference even when the question would infer nothing.
    r = _route(mini, "ACMEの資料", kind="train")
    assert r["evidence"]["resolved_kinds"] == ["train"]
    assert r["value"] and all(rec["kind"] == "train" for rec in r["value"])


def test_route_unknown_kind_raises(mini):
    with pytest.raises(cr.CanonicalRouteError):
        _route(mini, "ACMEの資料", kind="bogus")


def test_route_declines_when_no_project_or_kind(mini):
    # A cross-project / non-data question resolves to nothing → value == [] (caller stays abstained).
    r = _route(mini, "全体でもっとも多く関わる人は？")
    assert r["value"] == []
    assert r["evidence"]["primary"] is None


# --------------------------------------------------------------------------- direct compute connection
def test_route_expr_direct_compute(mini):
    # expr against a tabular primary runs compute_sandbox and returns the value, tagged with the route.
    r = _route(mini, "ACMEのtrain.csvの行数", expr="len(df)")
    assert r["value"] == 7  # ACME csv has 7 rows
    assert r["method"]["route"] == "canonical_route→compute"
    assert r["evidence"]["canonical_route"]["primary"].endswith("03.データ/train.csv")


def test_notebook_statistic_routes_expr_to_backing_train_data(mini):
    """A prose-vs-output statistic can be recomputed even when the question names an .ipynb."""
    question = "ACMEの01_eda.ipynbでvalueの平均を確認してください"
    assert [k.name for k in cr.infer_kinds(question)] == ["notebook", "train"]
    r = _route(mini, question, expr="df['value'].mean()")
    assert r["value"] == 40.0
    route = r["evidence"]["canonical_route"]
    assert route["resolved_kinds"] == ["notebook", "train"]
    assert route["primary"].endswith("03.データ/train.csv")
    assert r["method"]["route"] == "canonical_route→compute"


def test_route_expr_ignored_for_non_tabular_primary(mini):
    # A code primary (.py) is not tabular → expr is ignored, the file list is returned instead.
    r = _route(mini, "ACMEの分析コード", expr="len(df)")
    assert isinstance(r["value"], list)
    assert r["value"][0]["kind"] == "code"


def test_contract_shape(mini):
    assert C.is_contract(_route(mini, "ACMEのtrain.xlsx"))
    assert C.is_contract(_route(mini, "全体集計"))


def test_no_hardcoded_corpus_values():
    """受け入れ条件②: the module embeds no corpus-specific company/alias/password literal.

    Every real project/alias string lives in the runtime glossary (社内用語集.docx) or is discovered by
    walking the drive — never in this source. Guard against a regression that pins a real company name.
    """
    src = Path(cr.__file__).read_text(encoding="utf-8")
    for corpus_value in ("京橋", "青葉", "蒼樹会", "データアステル", "東都人材", "みなみ野"):
        assert corpus_value not in src


# --------------------------------------------------------------------------- real-corpus gold pins
# SOT-2486 retrieval_miss frontier: genuine canonical data/code questions the chunk retriever missed.
_Q61 = ("京橋信用ソリューションズの分析コードにおいて、今回の学習で勾配ブースティング法のモデルに実際に"
        "渡される n_estimators、learning_rate、random_state はそれぞれいくつですか。")
_Q80 = ("株式会社東都人材プラットフォームのtrain.xlsxにおいて、Sheet2の黄色にハイライトされたセルの"
        "抽出条件と集計内容を答えてください。")
_Q97 = ("青葉バイオメディカル機器のtrain.xlsxにおいて、黄色ハイライトが交差している2つのセルの値の差の"
        "絶対値を計算してください。")


@_needs_corpus
def test_idx61_code_resolves_modeling_py():
    r = cr.canonical_route(_Q61)
    assert r["evidence"]["project"] == "京橋信用ソリューションズ株式会社"
    assert r["evidence"]["resolved_kinds"] == ["code"]
    assert r["evidence"]["primary"].endswith("src/modeling.py")
    # modeling.py is genuinely where the gradient-boosting hyperparameters live.
    assert "modeling.py" in "\n".join(rec["rel"] for rec in r["value"])


@_needs_corpus
@pytest.mark.parametrize("question, project", [
    (_Q80, "株式会社東都人材プラットフォーム"),
    (_Q97, "株式会社青葉バイオメディカル機器"),
])
def test_idx80_97_train_xlsx_resolves(question, project):
    r = cr.canonical_route(question)
    assert r["evidence"]["project"] == project
    assert r["evidence"]["resolved_kinds"] == ["train"]
    assert r["evidence"]["primary"] == f"プロジェクト/{project}/03.データ/train.xlsx"


@_needs_corpus
@pytest.mark.parametrize("question", [
    "データアステル社の中でもっとも多くの案件にかかわっている人の内線番号を教えてください。",       # idx13
    "着手金が最も高い案件について、その案件のESの内線番号を教えてください。",                       # idx46
    "社内管理フォルダにあるFMにおいて、井上さんの向かいに座っている方のEXTを教えてください。",       # idx58
])
def test_seating_aggregate_cases_are_declined(question):
    """idx13/46/58 — the SOT-2486 needle heuristic false-attributed these seating/aggregate answers
    (whose gold is an EXT number that coincidentally appears in train.csv) to train.csv. They are NOT
    canonical data-file questions, so the route correctly declines and leaves them to seating_lookup /
    corpus_aggregate — no hallucinated train.csv resolution."""
    r = cr.canonical_route(question)
    assert r["value"] == []
    assert r["evidence"]["primary"] is None
