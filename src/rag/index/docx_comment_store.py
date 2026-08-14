"""docx コメントアンカーストア（SOT-2711 / cycle11, idx49）.

Sonnet gold100 cycle11 で LLM(claude-mcp) 経路に乗っていた idx49（東都人材 会議録の「コメントがついて
いる部分をそのまま抽出」→ gold ``WBS・進捗管理台帳確定（タスク割振・ガント更新）``）を決定論昇格する
ためのストア。docx のコメントは OOXML の **構造化データ**（``word/comments.xml`` にコメント本文、
``document.xml`` の ``commentRangeStart``/``commentRangeEnd`` にアンカー範囲）であり、質問非依存で全数抽出
できる。

本モジュールは build 時に一度だけ、探索に依存せず全案件の全 docx を走査し、各コメントを
``{project, rel, doc_kind, comments:[{id, author, comment_text, anchor_text, loc}]}`` で焼く
（質問も gold も見ない網羅計算のみ）。読み出し/配線は :mod:`src.rag.agent.docx_comment_lane`。

Design invariants（format_facts_store と同一）
---------------------------------------------
* **Opt-in at serve time.** :func:`enabled` (``RAG_DOCX_COMMENT_ANCHOR``) が runtime のみ gate する。
  default OFF ⇒ champion serve path は byte-identical。
* **Build は LLM フリー・追加的.** OOXML の決定論読取のみ。暗号化 docx は passwords 経由で復号、読めない
  1 本はスキップして継続。genai 呼び出しはしない。
* **Question-independent / No hardcoding.** universe は全案件の全 docx。gold 値・idx 番号は持たない。
  anchor 逐語は ``w:t`` 連結の byte-exact（SOT-2666「完全ラベル写経」教訓）。
* **Fail-open.** artifact 欠落・解析不能はすべて空へフォールバック（回帰ゼロ）。
"""
from __future__ import annotations

import io
import json
import os
import unicodedata
import xml.etree.ElementTree as ET
import zipfile
from pathlib import Path
from typing import Any, Mapping, Sequence

from config import settings
from src.rag.corpus import FileRef, nfc, walk

SCHEMA = "docx-comment-anchor-store"
SCHEMA_VERSION = 1

_ON = {"1", "true", "yes", "on"}

_DOC_EXTS = {"docx"}

_W = "{http://schemas.openxmlformats.org/wordprocessingml/2006/main}"

# doc-kind をパスのフォルダ名/ファイル名から決めるための語彙（format_facts_store と同一）。
_KIND_FOLDERS = (
    "会議録", "議事録", "報告資料", "報告書", "提案書", "契約書", "見積書", "仕様書",
    "定例", "月次",
)


def enabled() -> bool:
    """True when the serve path may consult the docx comment-anchor store (default OFF — opt-in)."""
    return os.getenv("RAG_DOCX_COMMENT_ANCHOR", "0").strip().lower() in _ON


def default_out_path() -> Path:
    return settings.ARTIFACTS_DIR / "docx_comment_store.jsonl"


def default_report_path() -> Path:
    return settings.ARTIFACTS_DIR / "docx_comment_store_build_report.json"


# --------------------------------------------------------------------------- helpers
def _office_bytes(ref: FileRef) -> bytes | None:
    """暗号化 docx を passwords 経由で復号したバイト列（非暗号化なら None ⇒ path 直読）。"""
    try:
        from src.rag.tools.highlight_extract import _office_bytes as _ob
        return _ob(ref, None)
    except Exception:  # noqa: BLE001 — 復号不能は呼び出し側で path 直読/スキップ
        return None


def _zip_bytes(ref: FileRef) -> bytes | None:
    data = _office_bytes(ref)
    if data is not None:
        return data
    try:
        return Path(ref.path).read_bytes()
    except Exception:  # noqa: BLE001
        return None


def _comment_texts(zf: zipfile.ZipFile) -> dict[str, dict[str, str]]:
    """``word/comments.xml`` から {id: {author, text}}（無ければ空）。"""
    out: dict[str, dict[str, str]] = {}
    if "word/comments.xml" not in zf.namelist():
        return out
    try:
        root = ET.fromstring(zf.read("word/comments.xml"))
    except Exception:  # noqa: BLE001
        return out
    for c in root.findall(_W + "comment"):
        cid = c.get(_W + "id")
        if cid is None:
            continue
        author = c.get(_W + "author") or ""
        text = "".join(t.text or "" for t in c.iter(_W + "t"))
        out[cid] = {"author": nfc(author), "text": nfc(text)}
    return out


def _anchor_texts(zf: zipfile.ZipFile) -> dict[str, str]:
    """``document.xml`` の commentRangeStart/End に囲まれた本文逐語を {id: anchor_text} で全数抽出.

    文書順（ElementTree の深さ優先 = 読み順）に走査し、開いているコメント範囲すべてに ``w:t`` を配る。
    複数コメントが同じ本文にかかる/入れ子でも各 id の逐語を byte-exact に組む。
    """
    try:
        root = ET.fromstring(zf.read("word/document.xml"))
    except Exception:  # noqa: BLE001
        return {}
    open_ids: set[str] = set()
    parts: dict[str, list[str]] = {}
    for el in root.iter():
        tag = el.tag
        if tag == _W + "commentRangeStart":
            cid = el.get(_W + "id")
            if cid is not None:
                open_ids.add(cid)
                parts.setdefault(cid, [])
        elif tag == _W + "commentRangeEnd":
            open_ids.discard(el.get(_W + "id"))
        elif tag == _W + "t" and open_ids and el.text:
            for cid in open_ids:
                parts[cid].append(el.text)
    return {cid: nfc("".join(chunks)) for cid, chunks in parts.items()}


# --------------------------------------------------------------------------- doc-kind
def _folder_kinds(rel: str, name: str) -> list[str]:
    hay = nfc(rel) + " " + nfc(name)
    kinds: list[str] = []
    for k in _KIND_FOLDERS:
        if k in hay and k not in kinds:
            kinds.append(k)
    return kinds


# --------------------------------------------------------------------------- universe / build
def _universe(refs: Sequence[FileRef]) -> list[FileRef]:
    out: list[FileRef] = []
    for r in refs:
        if r.ext not in _DOC_EXTS:
            continue
        name = nfc(r.name).lower()
        rel = nfc(r.rel).lower()
        if name.startswith("~$") or "/old/" in rel or name.startswith("old"):
            continue
        out.append(r)
    out.sort(key=lambda r: nfc(r.rel))
    return out


def compute_doc(ref: FileRef) -> dict[str, Any] | None:
    """1 docx のコメント（本文＋アンカー逐語）を全数抽出（コメント 0 なら None）。"""
    data = _zip_bytes(ref)
    if data is None:
        return None
    try:
        zf = zipfile.ZipFile(io.BytesIO(data))
    except Exception:  # noqa: BLE001
        return None
    texts = _comment_texts(zf)
    anchors = _anchor_texts(zf)
    ids = sorted(set(texts) | set(anchors), key=lambda x: (len(x), x))
    comments: list[dict[str, Any]] = []
    for cid in ids:
        anchor = anchors.get(cid, "")
        meta = texts.get(cid, {})
        # コメント本文もアンカー逐語も空なら記録しない（欠測を偽装しない）。
        if not anchor and not meta.get("text"):
            continue
        comments.append({
            "id": cid, "author": meta.get("author", ""),
            "comment_text": meta.get("text", ""), "anchor_text": anchor,
            "loc": f"comment:{cid}",
        })
    if not comments:
        return None
    return {
        "project": nfc(ref.project), "rel": nfc(ref.rel), "name": nfc(ref.name),
        "doc_kind": _folder_kinds(ref.rel, ref.name),
        "n_comments": len(comments), "comments": comments,
    }


def build(refs: Sequence[FileRef] | None = None, *, out: Path | None = None,
          write_report: bool = True) -> dict[str, Any]:
    """全案件の全 docx を走査し、コメントアンカーストアを構築（LLM 非使用・べき等・fail-open）。"""
    refs = list(refs) if refs is not None else list(walk())
    universe = _universe(refs)
    records: list[dict[str, Any]] = []
    skipped = 0
    for ref in universe:
        try:
            rec = compute_doc(ref)
        except Exception:  # noqa: BLE001 — 一文書の失敗で build を落とさない
            rec = None
        if rec is not None:
            records.append(rec)
        else:
            skipped += 1
    stats = write_store(records, out)
    report = {
        "schema": SCHEMA, "version": SCHEMA_VERSION,
        "universe": len(universe), "records": len(records), "skipped": skipped,
        "total_comments": sum(r["n_comments"] for r in records),
        "docs_with_comments": [r["rel"] for r in records],
    }
    if write_report:
        rp = default_report_path()
        rp.parent.mkdir(parents=True, exist_ok=True)
        rp.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return {"records": len(records), "docs": stats["docs"], "report": report}


# --------------------------------------------------------------------------- io
def write_store(records: Sequence[Mapping[str, Any]], path: Path | None = None) -> dict[str, int]:
    """Atomically write a reproducible JSONL store (schema header + rel-sorted rows)."""
    out = Path(path) if path is not None else default_out_path()
    out.parent.mkdir(parents=True, exist_ok=True)
    ordered = sorted(records, key=lambda r: (r.get("project", ""), r.get("rel", "")))
    tmp = out.with_suffix(out.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8", newline="\n") as handle:
        handle.write(json.dumps({"schema": SCHEMA, "version": SCHEMA_VERSION},
                                ensure_ascii=False, sort_keys=True) + "\n")
        for rec in ordered:
            handle.write(json.dumps(rec, ensure_ascii=False, sort_keys=True) + "\n")
    tmp.replace(out)
    reset_cache()
    return {"docs": len(ordered)}


_LOAD_CACHE: dict[str, list[dict[str, Any]]] = {}


def reset_cache() -> None:
    _LOAD_CACHE.clear()


def load(path: Path | None = None) -> list[dict[str, Any]]:
    """Load the comment-anchor rows (memoized). ``[]`` when absent/unreadable/schema-mismatch (回帰ゼロ)."""
    out = Path(path) if path is not None else default_out_path()
    key = str(out)
    if key in _LOAD_CACHE:
        return _LOAD_CACHE[key]
    rows: list[dict[str, Any]] = []
    try:
        with open(out, encoding="utf-8") as handle:
            header = handle.readline()
            meta = json.loads(header) if header.strip() else {}
            if isinstance(meta, dict) and meta.get("schema") == SCHEMA \
                    and meta.get("version") == SCHEMA_VERSION:
                for line in handle:
                    line = line.strip()
                    if line:
                        rows.append(json.loads(line))
    except Exception:  # noqa: BLE001
        rows = []
    _LOAD_CACHE[key] = rows
    return rows


def norm(text: Any) -> str:
    return unicodedata.normalize("NFKC", str(text or "")).replace(" ", "").replace("　", "").lower()


if __name__ == "__main__":
    summary = build()
    print(f"[build] docx_comment_store records={summary['records']} docs={summary['docs']} "
          f"total_comments={summary['report']['total_comments']} -> {default_out_path()}")
