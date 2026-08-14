"""SOT-2702 — 「別契約」と明記された役割の決定論抽出レーン（cycle10, idx52）の offline テスト。

LLM/corpus 不要。合成 FTS ヒットで決定論束縛の規律を固定する: OFF ⇒ None（byte-identical）、
idx52=案件名指し × ``X（別契約）`` 一意 ⇒ 逐語ラベル直答、曖昧/未マーク/意図なし/案件未名指し ⇒ defer。
"""
from __future__ import annotations

import pytest

from src.rag.agent import sep_contract_lane as L

_MINAMINO = "医療法人社団 蒼樹会 みなみ野女性医療センター"

_Q52 = ("蒼樹会 みなみ野女性医療センターの今後の運用に関する記載の中で、データアステル側の役割として"
        "「別契約」と明記されているものを抽出してください。")


def _hit(project, text, locator="page:8", doc_id="d1"):
    return {"project": project, "text": text, "locator": locator, "doc_id": doc_id}


def _ocr_rec(project, full_text, locus="ページ9", rel="d.pdf"):
    return {"project": project, "full_text": full_text, "locus": locus, "rel": rel}


@pytest.fixture()
def wired(monkeypatch):
    """Flag ON + text_fts enabled + a synthetic FTS result carrying the marked role label.

    SOT-2717 — also stub the image-OCR store loader to empty so the FTS-driven cases stay hermetic (the
    real on-disk store DOES hold the minamino label; each test that exercises the OCR fallback stubs its
    own records)."""
    monkeypatch.setenv(L.SEP_CONTRACT_ROLE_FLAG, "1")
    monkeypatch.setattr(L._tf, "enabled", lambda: True)
    monkeypatch.setattr(L._tf, "search", lambda *a, **k: [
        _hit(_MINAMINO, "監視ダッシュボード構築(別契約)", "page:8"),
        _hit(_MINAMINO, "監視ダッシュボード構築(別契約)", "page:9"),  # dup page ⇒ same label
        _hit("京橋信用ソリューションズ株式会社", "カテゴリ別契約率確認", "line:27"),  # substring, no marker
        _hit("白峰信用リスク評価株式会社", "本番移行は別契約として扱う。", "page:4"),  # bare mention
    ])
    monkeypatch.setattr("src.rag.index.image_ocr_store.load", lambda *a, **k: [])
    return L


def test_off_is_none(monkeypatch, wired):
    monkeypatch.delenv(L.SEP_CONTRACT_ROLE_FLAG, raising=False)
    assert L.resolve(_Q52) is None  # OFF ⇒ byte-identical fallback


def test_idx52_unique_role_label(wired):
    r = L.resolve(_Q52)
    assert r is not None
    assert r["value"] == "監視ダッシュボード構築"  # verbatim, paren-stripped by the marker regex
    assert r["evidence"]["project"] == _MINAMINO
    assert r["method"]["selection"] == "separate_contract_role"


def test_disabled_text_fts_defers(monkeypatch, wired):
    # FTS disabled AND the OCR store carries no same-project marker ⇒ defer (no evidence in either source).
    monkeypatch.setattr(L._tf, "enabled", lambda: False)
    assert L.resolve(_Q52) is None


def test_ocr_store_fallback_recovers(monkeypatch, wired):
    # SOT-2717 robustness — FTS surfaces NO marked label (e.g. index built OCR-unaware), but the persisted
    # image-OCR store carries the ``X（別契約）`` role for the named project ⇒ recover via the durable store.
    monkeypatch.setattr(L._tf, "search", lambda *a, **k: [])
    monkeypatch.setattr("src.rag.index.image_ocr_store.load", lambda *a, **k: [
        _ocr_rec(_MINAMINO, "…今後の運用 監視ダッシュボード構築(別契約) …", "ページ9"),
        _ocr_rec("京橋信用ソリューションズ株式会社", "カテゴリ別契約率の確認", "スライド4"),
    ])
    r = L.resolve(_Q52)
    assert r is not None
    assert r["value"] == "監視ダッシュボード構築"
    assert r["evidence"]["candidates"][0]["store"] == "image_ocr_store"


def test_ocr_store_fallback_respects_cross_project(monkeypatch, wired):
    # Neither source has a same-project marker (only a different project) ⇒ still defer under the fallback.
    monkeypatch.setattr(L._tf, "search", lambda *a, **k: [])
    monkeypatch.setattr("src.rag.index.image_ocr_store.load", lambda *a, **k: [
        _ocr_rec("白峰信用リスク評価株式会社", "追加解析(別契約)", "ページ4"),
    ])
    assert L.resolve(_Q52) is None


def test_no_marker_defers(monkeypatch, wired):
    # Only bare/substring 別契約 mentions in the named project ⇒ no (別契約) label ⇒ defer.
    monkeypatch.setattr(L._tf, "search", lambda *a, **k: [
        _hit(_MINAMINO, "本番移行は別契約として扱う。"),
        _hit(_MINAMINO, "カテゴリ別契約率の確認"),
    ])
    assert L.resolve(_Q52) is None


def test_ambiguous_labels_defer(monkeypatch, wired):
    # Two DISTINCT marked labels in the named project ⇒ ambiguous ⇒ fail-open (never guess).
    monkeypatch.setattr(L._tf, "search", lambda *a, **k: [
        _hit(_MINAMINO, "監視ダッシュボード構築(別契約)"),
        _hit(_MINAMINO, "保守運用(別契約)"),
    ])
    assert L.resolve(_Q52) is None


def test_project_not_named_defers(wired):
    # 別契約 + intent, but the question names no project the marked label lives in ⇒ defer.
    q = "今後の運用で別契約として明記された役割を抽出してください。"
    assert L.resolve(q) is None


def test_no_intent_defers(wired):
    # Names the project and 別契約 appears, but no extraction intent ⇒ not this shape ⇒ defer.
    q = "蒼樹会 みなみ野女性医療センターの別契約について教えて。"
    assert L.resolve(q) is None


def test_cross_project_marker_not_bound(monkeypatch, wired):
    # The only marked label belongs to a DIFFERENT project than the question names ⇒ defer (no cross-bind).
    monkeypatch.setattr(L._tf, "search", lambda *a, **k: [
        _hit("白峰信用リスク評価株式会社", "追加解析(別契約)"),
    ])
    assert L.resolve(_Q52) is None
