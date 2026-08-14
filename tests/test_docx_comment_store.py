"""SOT-2711 — docx コメントアンカーストア + 決定論 lookup レーン（cycle11, idx49）の offline テスト.

ネットワーク/LLM/実コーパス非依存。in-memory zip で OOXML コメント抽出器（comments.xml 本文 +
document.xml の commentRangeStart/End アンカー逐語）を、合成ストア（``docx_comment_store.load`` を
monkeypatch）で決定論束縛（idx49 会議録 3 本中コメント 1 本のみ = 一意）と精度優先の deferral、
``RAG_DOCX_COMMENT_ANCHOR`` 既定 OFF の byte-identical 挙動を検証する。
"""
from __future__ import annotations

import io
import zipfile

import pytest

from src.rag.agent import docx_comment_lane as L
from src.rag.agent import fact_layer as fl
from src.rag.index import docx_comment_store as S
from src.rag.corpus import FileRef

_W = "http://schemas.openxmlformats.org/wordprocessingml/2006/main"


# =========================================================================== OOXML 抽出（in-memory docx zip）
def _docx_zip(*, document_xml: str, comments_xml: str | None) -> bytes:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as z:
        z.writestr("word/document.xml", document_xml)
        if comments_xml is not None:
            z.writestr("word/comments.xml", comments_xml)
    return buf.getvalue()


def _document_with_comment(cid: str, anchor: str) -> str:
    return (f'<w:document xmlns:w="{_W}"><w:body><w:p>'
            f'<w:commentRangeStart w:id="{cid}"/>'
            f'<w:r><w:t>{anchor}</w:t></w:r>'
            f'<w:commentRangeEnd w:id="{cid}"/>'
            f'<w:r><w:t>（コメント外の本文）</w:t></w:r>'
            f'</w:p></w:body></w:document>')


def _comments_xml(cid: str, author: str, text: str) -> str:
    return (f'<w:comments xmlns:w="{_W}"><w:comment w:id="{cid}" w:author="{author}">'
            f'<w:p><w:r><w:t>{text}</w:t></w:r></w:p></w:comment></w:comments>')


def _ref(rel: str, name: str, project: str) -> FileRef:
    return FileRef(path="/tmp/x.docx", rel=rel, name=name, project=project, ext="docx",
                   category="meeting")


def test_compute_doc_extracts_anchor_and_comment(monkeypatch):
    data = _docx_zip(
        document_xml=_document_with_comment("7", "WBS・進捗管理台帳確定（タスク割振・ガント更新）"),
        comments_xml=_comments_xml("7", "h.ikeshita", "要確認"))
    ref = _ref("プロジェクト/株式会社東都人材プラットフォーム/05.会議/会議録/会議録_2025-08-18.docx",
               "会議録_2025-08-18.docx", "株式会社東都人材プラットフォーム")
    monkeypatch.setattr(S, "_office_bytes", lambda r: data)
    rec = S.compute_doc(ref)
    assert rec is not None and rec["n_comments"] == 1
    c = rec["comments"][0]
    assert c["anchor_text"] == "WBS・進捗管理台帳確定（タスク割振・ガント更新）"
    assert c["comment_text"] == "要確認" and c["author"] == "h.ikeshita"
    assert rec["doc_kind"] == ["会議録"]


def test_compute_doc_none_without_comments(monkeypatch):
    # コメント parts があっても実コメントが無ければ記録しない（欠測を偽装しない）。
    data = _docx_zip(document_xml=f'<w:document xmlns:w="{_W}"><w:body><w:p>'
                                  f'<w:r><w:t>本文のみ</w:t></w:r></w:p></w:body></w:document>',
                     comments_xml=f'<w:comments xmlns:w="{_W}"></w:comments>')
    ref = _ref("プロジェクト/X/05.会議/会議録/会議録_2025-09-08.docx", "会議録_2025-09-08.docx", "X")
    monkeypatch.setattr(S, "_office_bytes", lambda r: data)
    assert S.compute_doc(ref) is None


# =========================================================================== 合成ストア + レーン
def _records():
    todo = "株式会社東都人材プラットフォーム"
    base = f"プロジェクト/{todo}/05.会議/会議録"
    return [
        {"project": todo, "rel": f"{base}/会議録_2025-08-18.docx", "name": "会議録_2025-08-18.docx",
         "doc_kind": ["会議録"], "n_comments": 1,
         "comments": [{"id": "7", "author": "h.ikeshita", "comment_text": "要確認",
                       "anchor_text": "WBS・進捗管理台帳確定（タスク割振・ガント更新）", "loc": "comment:7"}]},
        # 他の会議録はコメント無し ⇒ ストアに載らない（= 一意化に寄与）。
    ]


@pytest.fixture
def store(monkeypatch):
    monkeypatch.setattr(S, "load", lambda path=None: _records())
    monkeypatch.setenv("RAG_DOCX_COMMENT_ANCHOR", "1")
    monkeypatch.setenv("RAG_FACT_LAYER", "1")
    return None


def test_idx49_comment_anchor_binding(store):
    r = L.resolve("東都人材プラットフォームの会議録において、コメントがついている部分をそのまま抽出してください。")
    assert r is not None and r["value"] == "WBS・進捗管理台帳確定（タスク割振・ガント更新）"
    assert r["evidence"]["comment_text"] == "要確認"
    assert r["method"]["selection"] == "comment_anchor"


def test_idx49_via_fact_layer(store):
    r = fl.resolve("東都人材プラットフォームの会議録において、コメントがついている部分をそのまま抽出してください。",
                   "simple_lookup")
    assert r is not None and r["value"] == "WBS・進捗管理台帳確定（タスク割振・ガント更新）"


def test_defers_without_comment_cue(store):
    # 「コメント」語が無い質問はこのレーンの対象外（従来経路へ）。
    assert L.resolve("東都人材プラットフォームの会議録の要点を教えてください。") is None


def test_defers_unknown_project(store):
    assert L.resolve("存在しない会社の会議録でコメントがついている部分を抽出してください。") is None


# --------------------------------------------------------------------------- OFF byte-identical
def test_off_is_inert(monkeypatch):
    monkeypatch.setattr(S, "load", lambda path=None: _records())
    monkeypatch.delenv("RAG_DOCX_COMMENT_ANCHOR", raising=False)
    assert L.enabled() is False
    assert L.resolve("東都人材プラットフォームの会議録のコメントがついている部分を抽出してください。") is None


def test_off_lane_not_in_fact_layer(monkeypatch):
    monkeypatch.setattr(S, "load", lambda path=None: _records())
    monkeypatch.setenv("RAG_FACT_LAYER", "1")
    monkeypatch.delenv("RAG_DOCX_COMMENT_ANCHOR", raising=False)
    assert fl.resolve("東都人材プラットフォームの会議録のコメントがついている部分を抽出してください。",
                      "simple_lookup") is None


# --------------------------------------------------------------------------- store schema roundtrip
def test_store_load_schema_roundtrip(tmp_path):
    p = tmp_path / "docx_comment_store.jsonl"
    S.write_store(_records(), path=p)
    S.reset_cache()
    rows = S.load(path=p)
    assert len(rows) == 1 and rows[0]["comments"][0]["anchor_text"].startswith("WBS")
