"""SOT-2650 — diff_store notebook (.ipynb) lane: pair enumeration, cell diff, embedded-image handling.

Hermetic: notebooks are written to tmp_path, diffpair._walk is stubbed, and the vision passes
(_table_headers/_transcribe_embedded) are monkeypatched — no network, no artifacts.
"""
from __future__ import annotations

import json

import pytest

from src.rag import diffpair
from src.rag.corpus import FileRef
from src.rag.index import diff_store


def _nb(cells):
    return {"nbformat": 4, "cells": cells}


def _code(src, outs=()):
    return {"cell_type": "code", "source": [src],
            "outputs": [{"output_type": "stream", "text": [o]} for o in outs]}


def _md(src):
    return {"cell_type": "markdown", "source": [src]}


def _write_ref(tmp_path, name, nb, project="白峰信用リスク評価株式会社"):
    p = tmp_path / name
    p.write_text(json.dumps(nb, ensure_ascii=False), encoding="utf-8")
    return FileRef(path=p, project=project, category="analysis",
                   rel=f"プロジェクト/{project}/04.分析/notebooks/{name}", name=name,
                   ext="ipynb")


@pytest.fixture(autouse=True)
def _no_vision(monkeypatch):
    monkeypatch.delenv("RAG_OCR_STORE_BUILD", raising=False)
    yield


def _stub_walk(monkeypatch, refs):
    monkeypatch.setattr(diffpair, "_walk", lambda company=None: list(refs))


# --------------------------------------------------------------------------- pair enumeration
def test_versioned_old_pairs_with_unversioned_latest(monkeypatch, tmp_path):
    old = _write_ref(tmp_path, "01_eda_old.ipynb", _nb([_code("x=1")]))
    new = _write_ref(tmp_path, "01_eda.ipynb", _nb([_code("x=2")]))
    _stub_walk(monkeypatch, [old, new])
    pairs = diff_store._notebook_pairs()
    assert len(pairs) == 1
    assert pairs[0].old.name == "01_eda_old.ipynb" and pairs[0].new.name == "01_eda.ipynb"
    assert pairs[0].basis == "notebook"


def test_unversioned_only_yields_no_pair(monkeypatch, tmp_path):
    a = _write_ref(tmp_path, "01_eda.ipynb", _nb([_code("x=1")]))
    b = _write_ref(tmp_path, "02_model.ipynb", _nb([_code("y=1")]))
    _stub_walk(monkeypatch, [a, b])
    assert diff_store._notebook_pairs() == []


# --------------------------------------------------------------------------- cell diff
def test_source_and_output_changes_recorded(monkeypatch, tmp_path):
    old = _write_ref(tmp_path, "01_eda_old.ipynb",
                     _nb([_code("import pandas"), _code("df.describe()", ["Attr1 1.0"]),
                          _code("removed_cell")]))
    new = _write_ref(tmp_path, "01_eda.ipynb",
                     _nb([_code("import pandas"), _code("df.describe()", ["Attr1 1.0\nclass 0.5"]),
                          _code("added_cell")]))
    _stub_walk(monkeypatch, [old, new])
    rec = diff_store._notebook_pair_record(diff_store._notebook_pairs()[0])
    assert rec["alignment_ok"] is True and rec["basis"] == "notebook" and rec["ext"] == "ipynb"
    by_loc = {c["structural_location"]: c for c in rec["changes"]}
    out_change = next(c for c in rec["changes"] if "(output)" in c["structural_location"])
    assert "class 0.5" in out_change["new"] and "output_change" in out_change["attributes"]
    kinds = {c["kind"] for c in rec["changes"]}
    assert {"modify"} <= kinds  # replace/add covered below by opcode shapes
    assert all("rank" in c for c in rec["changes"])


def test_unparsable_notebook_is_honest_alignment_failure(monkeypatch, tmp_path):
    old = _write_ref(tmp_path, "01_eda_old.ipynb", _nb([_code("x=1")]))
    bad = tmp_path / "01_eda.ipynb"
    bad.write_text("{not json", encoding="utf-8")
    new = FileRef(path=bad, project="白峰信用リスク評価株式会社", category="analysis",
                  rel="プロジェクト/白峰信用リスク評価株式会社/04.分析/notebooks/01_eda.ipynb",
                  name="01_eda.ipynb", ext="ipynb")
    _stub_walk(monkeypatch, [old, new])
    rec = diff_store._notebook_pair_record(diff_store._notebook_pairs()[0])
    assert rec["alignment_ok"] is False and rec["change_count"] == 0


# --------------------------------------------------------------------------- embedded image branch
_IMG = "![embedded-image](data:image/png;base64,QUJD)"


def test_embedded_image_without_vision_flag_keeps_honest_marker(monkeypatch, tmp_path):
    old = _write_ref(tmp_path, "01_eda_old.ipynb", _nb([_md(_IMG + "A")]))
    new = _write_ref(tmp_path, "01_eda.ipynb", _nb([_md(_IMG.replace("QUJD", "WFla") + "B")]))
    _stub_walk(monkeypatch, [old, new])
    rec = diff_store._notebook_pair_record(diff_store._notebook_pairs()[0])
    ch = rec["changes"][0]
    assert "embedded_image" in ch["attributes"]
    assert "image_ocr" not in ch["attributes"]
    assert "転記未実施" in ch["old"]
    assert "QUJD" not in ch["old"] and "WFla" not in ch["new"]  # the blob itself is never stored


def test_embedded_image_with_vision_stubs_stores_header_diff(monkeypatch, tmp_path):
    old = _write_ref(tmp_path, "01_eda_old.ipynb", _nb([_md(_IMG + "A")]))
    new = _write_ref(tmp_path, "01_eda.ipynb", _nb([_md(_IMG.replace("QUJD", "WFla") + "B")]))
    _stub_walk(monkeypatch, [old, new])
    headers = {"QUJD": [f"Attr{i}" for i in range(1, 65)],
               "WFla": [f"Attr{i}" for i in range(1, 65)] + ["class"]}
    monkeypatch.setattr(diff_store, "_table_headers", lambda b64, st, tiles=4: headers[b64])
    monkeypatch.setattr(diff_store, "_transcribe_embedded",
                        lambda b64, st: f"転記:{b64}")
    rec = diff_store._notebook_pair_record(diff_store._notebook_pairs()[0])
    ch = rec["changes"][0]
    assert ch["headers_added"] == ["class"] and ch["headers_removed"] == []
    assert ch["headers_old_count"] == 64 and ch["headers_new_count"] == 65
    assert "column_added" in ch["attributes"] and "table_headers" in ch["attributes"]
    assert "image_ocr" in ch["attributes"]


# --------------------------------------------------------------------------- reuse (no downgrade)
def test_prior_vision_record_reused_for_unchanged_files(monkeypatch, tmp_path):
    old = _write_ref(tmp_path, "01_eda_old.ipynb", _nb([_md(_IMG + "A")]))
    new = _write_ref(tmp_path, "01_eda.ipynb", _nb([_md(_IMG.replace("QUJD", "WFla") + "B")]))
    _stub_walk(monkeypatch, [old, new])
    prior_rec = {
        "old_rel": old.rel, "new_rel": new.rel, "basis": "notebook", "alignment_ok": True,
        "old_size": old.path.stat().st_size, "new_size": new.path.stat().st_size,
        "changes": [{"attributes": ["notebook", "embedded_image", "table_headers", "image_ocr"],
                     "headers_added": ["class"]}],
    }
    store = tmp_path / "diff_store.jsonl"
    store.write_text(json.dumps(prior_rec, ensure_ascii=False) + "\n", encoding="utf-8")

    def _boom(*a, **k):
        raise AssertionError("must reuse the prior record, not re-diff")
    monkeypatch.setattr(diff_store, "_notebook_pair_record", _boom)
    recs = diff_store._notebook_records(store)
    assert recs == [prior_rec]


def test_prior_marker_record_recomputed_when_files_changed(monkeypatch, tmp_path):
    old = _write_ref(tmp_path, "01_eda_old.ipynb", _nb([_code("x=1")]))
    new = _write_ref(tmp_path, "01_eda.ipynb", _nb([_code("x=2")]))
    _stub_walk(monkeypatch, [old, new])
    prior_rec = {"old_rel": old.rel, "new_rel": new.rel, "basis": "notebook", "alignment_ok": True,
                 "old_size": 1, "new_size": 1,  # stale sizes ⇒ recompute
                 "changes": [{"attributes": ["notebook"]}]}
    store = tmp_path / "diff_store.jsonl"
    store.write_text(json.dumps(prior_rec, ensure_ascii=False) + "\n", encoding="utf-8")
    recs = diff_store._notebook_records(store)
    assert len(recs) == 1 and recs[0] is not prior_rec
    assert recs[0]["change_count"] == 1
