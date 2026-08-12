"""Offline tests for deterministic pre-search query distillation (SOT-2672 / query_distill).

No corpus / no network / no LLM. Asserts: (1) condition/question scaffolding is removed while content
nouns survive, (2) a glossary company is moved to the ``project`` scope and stripped from the query
(company→scope転用), (3) rare/ID tokens are retained, (4) the tool is OFF by default (byte-identical),
(5) the corpus-vocabulary bridge deterministically substitutes out-of-vocabulary tokens, and (6)
text_search / unified_search record the 蒸留前後 diagnostic and apply the scope when the flag is ON.
"""
from __future__ import annotations

from pathlib import Path

import pytest

from src.rag.corpus import FileRef
from src.rag.extract.glossary import Glossary
from src.rag.index import text_fts
from src.rag.tools import query_distill as qd
from src.rag.tools import text_search, unified_search


# --------------------------------------------------------------------------- enabled / OFF invariant
def test_disabled_by_default(monkeypatch):
    monkeypatch.delenv("RAG_QUERY_DISTILL", raising=False)
    assert qd.enabled() is False


def test_enabled_flag(monkeypatch):
    monkeypatch.setenv("RAG_QUERY_DISTILL", "1")
    assert qd.enabled() is True


# --------------------------------------------------------------------------- (b) condition removal
@pytest.mark.parametrize("query,expected", [
    ("契約金額は何ですか", "契約金額"),
    ("2019年度売上高を知りたい", "2019年度売上高"),
    ("担当者をすべて挙げてください", "担当者"),
    ("案件はいくつありますか", "案件"),
    ("予算について教えてください", "予算"),
    ("システムとは何か", "システム"),
])
def test_condition_phrase_removal(query, expected):
    # no company in these, no corpus needed → only condition removal runs.
    res = qd.distill(query, glossary=Glossary())
    assert res.query == expected
    assert res.removed_phrases  # something formulaic was stripped
    assert res.changed


def test_content_noun_survives_internal_particle():
    # 『2019年度の売上高』の内部の「の」は残す(会社直後の「の」だけを落とす)。
    res = qd.distill("2019年度の売上高を教えてください", glossary=Glossary())
    assert res.query == "2019年度の売上高"


# --------------------------------------------------------------------------- (c) rare/ID retention
def test_id_token_retained_through_removal():
    res = qd.distill("EXT1234 の担当は誰ですか", glossary=Glossary())
    assert "EXT1234" in res.query
    assert "EXT1234" in res.kept_id_tokens


def test_empty_after_removal_keeps_original():
    res = qd.distill("教えてください", glossary=Glossary())
    assert res.query  # never distilled to empty


# --------------------------------------------------------------------------- (a) company → scope
def test_company_moved_to_scope_and_stripped(monkeypatch):
    import importlib
    canonical_route = importlib.import_module("src.rag.tools.canonical_route")
    g = Glossary()
    g.company_aliases = {"エービーシー株式会社": ["ABC社", "ABC", "A-01"]}
    # registry resolves the company to a real corpus folder name (patched — no corpus walk in this test).
    monkeypatch.setattr(canonical_route, "resolve_project",
                        lambda *a, **k: "01.案件/エービーシー株式会社")
    res = qd.distill("ABC社の2019年度売上高を知りたい", glossary=g)
    assert res.scope_project == "01.案件/エービーシー株式会社"
    assert res.removed_company == "ABC社"
    assert "ABC" not in res.query
    assert res.query == "2019年度売上高"


def test_company_not_stripped_when_scope_unresolved(monkeypatch):
    import importlib
    canonical_route = importlib.import_module("src.rag.tools.canonical_route")
    g = Glossary()
    g.company_aliases = {"エービーシー株式会社": ["ABC社"]}
    # registry cannot resolve a folder → keep the company token, never guess a scope.
    monkeypatch.setattr(canonical_route, "resolve_project", lambda *a, **k: None)
    res = qd.distill("ABC社の担当者を教えてください", glossary=g)
    assert res.scope_project is None
    assert res.removed_company is None
    assert "ABC社" in res.query


def test_explicit_project_hint_skips_company_detection():
    g = Glossary()
    g.company_aliases = {"X株式会社": ["X社"]}
    res = qd.distill("X社の売上を教えて", project="X株式会社", glossary=g)
    # explicit project given → no scope derived, company kept, only condition removal applies.
    assert res.scope_project is None
    assert "X社" in res.query


# --------------------------------------------------------------------------- corpus-vocabulary bridge
def test_bridge_tokens_substitutes_oov_to_rarest_superstring():
    vocab = {"売上高": 5.0, "売上": 1.2, "契約": 2.0}
    bridged, subs = qd._bridge_tokens(["売上", "契約"], vocab)
    # "売上" is in-vocab → passes through unchanged; "契約" in-vocab → unchanged. No subs.
    assert subs == []
    assert bridged == ["売上", "契約"]


def test_bridge_tokens_oov_token_bridged():
    vocab = {"売上高": 5.0, "総売上高": 6.0, "契約": 2.0}
    # "売上" is NOT in vocab; candidates {売上高, 総売上高} contain it → rarest (max idf) wins = 総売上高.
    bridged, subs = qd._bridge_tokens(["売上"], vocab)
    assert subs == [("売上", "総売上高")]
    assert bridged == ["総売上高"]


def test_bridge_tokens_no_vocab_is_noop():
    assert qd._bridge_tokens(["a", "b"], {}) == (["a", "b"], [])


# --------------------------------------------------------------------------- text_fts index fixture
def _write(dir_: Path, rel: str, text: str, project: str = "ACME") -> FileRef:
    p = dir_ / rel
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(text, encoding="utf-8")
    ext = rel.rsplit(".", 1)[-1].lower()
    return FileRef(path=p, project=project, category="", rel=rel, name=Path(rel).name, ext=ext)


@pytest.fixture()
def built(tmp_path: Path, monkeypatch):
    root = tmp_path / "share"
    corpus = [
        _write(root, "acme_report.txt",
               "本レポートの2019年度売上高は 1200 百万円です。\n担当は山田さん。", project="ACME"),
        _write(root, "beta_report.txt",
               "本レポートの2019年度売上高は 800 百万円です。", project="BETA"),
    ]
    db = tmp_path / "text_fts.db"
    monkeypatch.setattr(text_fts, "default_report_path", lambda: tmp_path / "report.json")
    text_fts.build(corpus, out_path=db)
    text_fts.reset_cache()
    monkeypatch.setattr(text_fts, "default_out_path", lambda: db)
    return db


def test_idf_vocab_reads_corpus_tokens(built, monkeypatch):
    monkeypatch.setenv("RAG_TEXT_FTS", "1")
    text_fts.reset_cache()
    vocab = text_fts.idf_vocab()
    # digits split the CJK run, so the corpus vocabulary holds the 2-char shingle 売上 (not 売上高).
    assert "売上" in vocab and vocab["売上"] > 0


def test_idf_vocab_empty_when_disabled(built, monkeypatch):
    monkeypatch.delenv("RAG_TEXT_FTS", raising=False)
    text_fts.reset_cache()
    assert text_fts.idf_vocab() == {}


# --------------------------------------------------------------------------- text_search integration
def test_text_search_off_is_byte_identical(built, monkeypatch):
    monkeypatch.setenv("RAG_TEXT_FTS", "1")
    monkeypatch.delenv("RAG_QUERY_DISTILL", raising=False)
    text_fts.reset_cache()
    res = text_search.text_search("2019年度売上高を教えてください")
    assert "distill" not in res["evidence"]
    assert res["method"]["query"] == "2019年度売上高を教えてください"


def test_text_search_distill_records_diagnostic(built, monkeypatch):
    monkeypatch.setenv("RAG_TEXT_FTS", "1")
    monkeypatch.setenv("RAG_QUERY_DISTILL", "1")
    text_fts.reset_cache()
    res = text_search.text_search("2019年度売上高を教えてください")
    diag = res["evidence"]["distill"]
    assert diag["before"] == "2019年度売上高を教えてください"
    assert diag["after"] == "2019年度売上高"
    assert res["value"], "distilled query still hits the corpus"


def test_text_search_distill_scope_narrows(built, monkeypatch):
    import importlib
    canonical_route = importlib.import_module("src.rag.tools.canonical_route")
    monkeypatch.setenv("RAG_TEXT_FTS", "1")
    monkeypatch.setenv("RAG_QUERY_DISTILL", "1")
    text_fts.reset_cache()
    g = Glossary()
    g.company_aliases = {"ACME": ["エーカンパニー", "ACME"]}
    import src.rag.extract.glossary as gl
    monkeypatch.setattr(gl, "load", lambda: g)
    monkeypatch.setattr(canonical_route, "resolve_project", lambda *a, **k: "ACME")
    res = text_search.text_search("エーカンパニーの2019年度売上高を教えて")
    assert res["evidence"]["distill"]["scope_project"] == "ACME"
    assert res["evidence"]["filters"]["project"] == "ACME"
    # scoped to ACME only → the BETA copy must not appear
    assert res["value"] and all(h["project"] == "ACME" for h in res["value"])


# --------------------------------------------------------------------------- unified_search integration
def test_unified_search_off_no_distill_key(monkeypatch):
    monkeypatch.setenv("RAG_UNIFIED_SEARCH", "1")
    monkeypatch.delenv("RAG_QUERY_DISTILL", raising=False)
    rs = [unified_search.Retriever("a", lambda q, h: [{"doc_id": "D", "locator": "l", "text": "t"}])]
    res = unified_search.search("会社の売上を教えて", retrievers=rs)
    assert "distill" not in res["evidence"]["coverage"]


def test_unified_search_distill_records_and_passes_distilled_query(monkeypatch):
    monkeypatch.setenv("RAG_UNIFIED_SEARCH", "1")
    monkeypatch.setenv("RAG_QUERY_DISTILL", "1")
    seen: dict[str, str] = {}

    def _capture(q, h):
        seen["query"] = q
        return [{"doc_id": "D", "locator": "l", "text": "t"}]

    rs = [unified_search.Retriever("a", _capture)]
    res = unified_search.search("2019年度売上高を教えてください", retrievers=rs)
    cov = res["evidence"]["coverage"]
    assert cov["distill"]["after"] == "2019年度売上高"
    assert seen["query"] == "2019年度売上高"  # retrievers receive the distilled query
