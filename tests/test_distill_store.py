"""SOT-2658 — build-time document distillation store, offline tests.

ネットワーク/実 Gemini/実コーパス非依存。蒸留器(distiller)を注入して build ロジック(ユニット分割・
content-hash キャッシュ・幻覚ガード)を、合成ストアで serve 側 API(search / candidate_doc_ids /
text_records)と registry distill tier 配線、RAG_DISTILL_STORE 既定 OFF の byte-identical 挙動、
serve 時 Gemini ゼロ(RAG_FORBID_GEMINI 併用)を検証する。
"""
from __future__ import annotations

import json

import pytest

from src.rag.corpus import FileRef
from src.rag.index import distill_store as dz
from src.rag.index import document_registry as dr


# --------------------------------------------------------------------------- unit slicing
def test_units_xlsx_splits_per_sheet():
    text = (
        "[シート: 売上] 範囲 A1:C3\n2025 | 100\n"
        "[シート: 原価] 範囲 A1:B2\n2025 | 40\n"
    )
    units = dz._units("xlsx", text)
    assert [u[0] for u in units] == ["sheet", "sheet"]
    assert [u[1] for u in units] == ["売上", "原価"]
    assert "売上" in units[0][2] and "原価" not in units[0][2]


def test_units_non_xlsx_is_single_doc():
    units = dz._units("pdf", "本文テキスト")
    assert units == [("doc", "", "本文テキスト")]


def test_content_hash_stable_and_nfc():
    import unicodedata
    a = "会議録"                       # NFC
    b = unicodedata.normalize("NFD", a)  # NFD variant
    assert dz.content_hash(a) == dz.content_hash(b)


# --------------------------------------------------------------------------- hallucination guard
def test_sanitize_drops_unsupported_facts_and_records_reason():
    source = "総レコード数は 7,352 件。担当は 斎藤悠斗。システムは Vertex AI を使用。"
    raw = {
        "answerable_questions": ["総レコード数は？", "総レコード数は？"],  # dup collapses
        "summary": "データ概要の説明。",
        "key_facts": ["7352", "9999"],           # 7352 supported (7,352 folds), 9999 hallucinated
        "mentioned_ids": [],
        "mentioned_people": ["斎藤悠斗", "存在しない人"],
        "mentioned_systems": ["Vertex AI"],
    }
    rec, dropped = dz.sanitize(raw, source)
    assert rec["key_facts"] == ["7352"]
    assert rec["mentioned_people"] == ["斎藤悠斗"]
    assert rec["mentioned_systems"] == ["Vertex AI"]
    assert rec["answerable_questions"] == ["総レコード数は？"]  # de-duplicated
    reasons = {(d["field"], d["value"]) for d in dropped}
    assert ("key_facts", "9999") in reasons
    assert ("mentioned_people", "存在しない人") in reasons


def test_has_signal_requires_something():
    assert not dz._has_signal({"answerable_questions": [], "summary": "", "key_facts": []})
    assert dz._has_signal({"answerable_questions": ["q"], "summary": "", "key_facts": []})


# --------------------------------------------------------------------------- build + cache
def _ref(tmp_path, rel: str, content: str, ext: str = "md") -> FileRef:
    p = tmp_path / rel
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(content, encoding="utf-8")
    return FileRef(path=p, project="Alpha", category="report", rel=rel, name=p.name, ext=ext)


def test_build_uses_content_hash_cache(tmp_path, monkeypatch):
    out = tmp_path / "distill_store.jsonl"
    ref = _ref(tmp_path, "報告書.md",
               "総レコード数は 7,352 件。担当は 斎藤悠斗。分析は Vertex AI で実施。" * 2)

    calls = {"n": 0}

    def fake_distiller(unit_text: str) -> dict:
        calls["n"] += 1
        return {
            "answerable_questions": ["総レコード数は何件か", "誰が担当か"],
            "summary": "報告書の要約。",
            "key_facts": ["7,352"],
            "mentioned_people": ["斎藤悠斗"],
            "mentioned_systems": ["Vertex AI"],
        }

    r1 = dz.build([ref], out, distiller=fake_distiller)
    assert r1["report"]["distilled"] == 1 and calls["n"] == 1
    # second build over the unchanged file reuses the cached record — no new distiller call.
    r2 = dz.build([ref], out, distiller=fake_distiller)
    assert r2["report"]["reused"] == 1 and r2["report"]["distilled"] == 0 and calls["n"] == 1

    lines = [json.loads(l) for l in out.read_text(encoding="utf-8").splitlines()]
    assert lines[0]["schema"] == dz.SCHEMA
    rec = lines[1]
    assert rec["doc_id"] == "報告書.md" and rec["key_facts"] == ["7,352"]
    assert "content_hash" in rec


def test_build_skips_short_and_records_failures(tmp_path):
    out = tmp_path / "s.jsonl"
    short = _ref(tmp_path, "stub.md", "短い")                     # below MIN_BODY_CHARS
    good = _ref(tmp_path, "本文.md", "総レコード数は 7,352 件と明記されている。" * 3)

    def boom(_t):  # noqa: ANN001
        raise RuntimeError("gemini down")

    res = dz.build([short, good], out, distiller=boom)
    # short unit never reaches the distiller; the good unit's distiller error is recorded, not raised.
    assert res["report"]["units"] == 0
    assert any("gemini down" in s.get("reason", "") for s in res["report"]["skipped"])


# --------------------------------------------------------------------------- serve API + OFF gate
@pytest.fixture()
def _built_store(tmp_path, monkeypatch):
    out = tmp_path / "distill_store.jsonl"
    monkeypatch.setattr(dz, "default_out_path", lambda: out)
    ref = _ref(tmp_path, "Alpha/報告書.md",
               "四半期の売上は目標を達成した。契約番号 C-2025-001。" * 3)

    def fake_distiller(_t):  # noqa: ANN001
        return {
            "answerable_questions": ["四半期の売上目標は達成できたか"],
            "summary": "売上実績の報告。",
            "key_facts": ["C-2025-001"],
            "mentioned_ids": ["C-2025-001"],
        }

    dz.build([ref], out, distiller=fake_distiller)
    dz.load.cache_clear()
    return out


def test_search_disabled_returns_empty(_built_store, monkeypatch):
    monkeypatch.delenv("RAG_DISTILL_STORE", raising=False)  # default OFF
    assert dz.search("四半期の売上目標は達成できたか") == []


def test_search_enabled_matches_answerable_question(_built_store, monkeypatch):
    monkeypatch.setenv("RAG_DISTILL_STORE", "1")
    hits = dz.search("四半期の売上目標は達成できたか")
    assert hits and hits[0][0] == "Alpha/報告書.md"


def test_search_never_calls_gemini(_built_store, monkeypatch):
    """Serve path must be Gemini-free: forbidding genai must NOT break search (build-only distillation)."""
    monkeypatch.setenv("RAG_DISTILL_STORE", "1")
    monkeypatch.setenv("RAG_FORBID_GEMINI", "1")
    hits = dz.search("四半期の売上目標は達成できたか")
    assert hits and hits[0][0] == "Alpha/報告書.md"


def test_text_records_shape(_built_store, monkeypatch):
    rows = dz.text_records()
    assert rows and rows[0]["source"] == "distill" and rows[0]["doc_id"] == "Alpha/報告書.md"
    assert "四半期" in rows[0]["text"]


# --------------------------------------------------------------------------- registry distill tier
def _registry_rows() -> list[dict]:
    # A document whose filename/title/entities share NO trigrams with the question, so the deterministic
    # tiers (exact/alias/version/entity/lexical) all return empty and the distill tier is what fires.
    return [{
        "doc_id": "Alpha/報告書.md", "case_id": "Alpha", "full_path": "Alpha/報告書.md",
        "normalized_filename": "report.md", "filename_aliases": [], "extension": "md",
        "title": "report", "title_aliases": [], "named_entities": [], "detected_dates": [],
        "version_tokens": [], "version_family_id": "", "predecessor_doc_id": "", "synopsis": "report",
    }]


def test_registry_distill_tier_off_by_default(_built_store, monkeypatch):
    monkeypatch.delenv("RAG_DISTILL_STORE", raising=False)
    resolver = dr.Resolver(_registry_rows())
    assert resolver.resolve("四半期の売上目標は達成できたか", project="Alpha") == []


def test_registry_distill_tier_fires_when_enabled(_built_store, monkeypatch):
    monkeypatch.setenv("RAG_DISTILL_STORE", "1")
    resolver = dr.Resolver(_registry_rows())
    res = resolver.resolve("四半期の売上目標は達成できたか", project="Alpha")
    assert res and res[0].doc_id == "Alpha/報告書.md" and res[0].tier == "distill"
