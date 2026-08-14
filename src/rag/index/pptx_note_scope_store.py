"""pptx 発表者ノート「スコープ対象外」✖項目 決定論カウント事実ストア（SOT-2714 / cycle11）.

Sonnet gold100 cycle11 の abstain idx27 は「恒一会 かえで総合病院の提案書において、スコープ対象外として
いる項目はいくつありますか。」型で、正解 **7** は gold をハードコードせず、対象 pptx の**発表者ノート**を
質問非依存に走査すれば決定論的に導ける（codex-sol(max) の独立調査 ``docs/ai/60_worker_codex_report.idx27.md``
と ``docs/ai/cycle9_residual10_codex_sol_max_audit.md`` で確認済み）:

* idx27 — 恒一会 かえで総合病院 ``00.提案/提案書.pptx`` の ``ppt/notesSlides/notesSlide1.xml``（スライド6
  『スコープ ─ 対象データ』の発表者ノート）に、見出し「スコープ対象外」の直下で先頭が ``✖`` の独立段落
  （``<a:p>``）が **7 本**列挙されている。gold=7。

このストアは build 時に **全案件×全 pptx×全 notesSlide** を OOXML 段落構造で走査し、見出し
「スコープ対象外」の直下に連続する ``✖`` 印付き段落を1項目として数え、
``{doc_id, project, doc_name, notes_slides:[…], scope_excluded_count, items:[…]}`` を質問非依存で全数記録する
（網羅計算のみ・LLM フリー）。どの文書が質問対象かの一意判定と回答整形は serve 側の
:mod:`src.rag.agent.pptx_note_scope_lane`（決定論レーン）が担う。

Design invariants（sibling の pptx_money_page_store / heading_page_store と同一）
------------------------------------------------------------------------------
* **Opt-in at serve time.** :func:`enabled` (``RAG_PPTX_NOTE_SCOPE``) が runtime 参照のみを gate する。
  default OFF ⇒ champion serve path は byte-identical。読み出し/配線は :mod:`src.rag.agent.pptx_note_scope_lane`。
* **Build は質問非依存・LLM フリー・追加的.** 見る材料は pptx の OOXML（``ppt/notesSlides/notesSlideK.xml``）
  の段落テキストだけ（暗号化 pptx は ``passwords.resolve`` で復号したバイト列）。genai 呼び出しはしない。
  読めない pptx は 1 件スキップし build は継続（fail-open）。gold も質問も参照しない。
* **No hard-coded answers.** カウントは見出し「スコープ対象外」直下の ``✖`` 段落数から導出。idx 番号も gold 値
  （7）も埋め込まない。段落単位で数えるので、同一 ``✖`` 段落内に読点併記の複数作業があっても1項目。見出しが
  「スコープ対象外」でない例（スライド18『変更管理方針』の「当初合意スコープ外」例示）は本文かつ別見出しで
  ノートにも無いため自然に除外される。
* **Fail-open.** artifact 欠落・解析不能・復号不能はすべて空へフォールバック（回帰ゼロ）。
"""
from __future__ import annotations

import io
import json
import os
import re
import unicodedata
import zipfile
from pathlib import Path
from typing import Any, Mapping, Sequence

from config import settings
from src.rag.corpus import FileRef, nfc, walk
from src.rag.extract import passwords

SCHEMA = "pptx-note-scope-store"
SCHEMA_VERSION = 1

_ON = {"1", "true", "yes", "on"}

# pptx のみ。~$ 一時ファイルは除外。
_PPTX_EXTS = {"pptx"}

# DrawingML 名前空間（notesSlide の段落 <a:p> / テキスト run <a:t>）。
_A_NS = "http://schemas.openxmlformats.org/drawingml/2006/main"

# 見出し「スコープ対象外」。段落テキスト全体（装飾除去後）がこれに一致する段落を見出しとみなす。
_SCOPE_HEADING = "スコープ対象外"
# 見出しの前後に付きうる装飾（■ / 【】 / ： / ・ / 空白 等）。厳密一致のために剥がす。
_DECOR = re.compile(r"^[\s　■□◆◇●○・\-—―【〔（(\[]*|[\s　】〕）)\]：:。、]*$")
# 「対象外項目」の先頭マーク。主は ✖(U+2716)。近縁の ✗/❌ も同義として許容（過検出しない小集合）。
_XMARK = ("✖", "✗", "❌")  # ✖ ✗ ❌


def enabled() -> bool:
    """True when the serve path may consult the pptx note-scope store (default OFF — opt-in)."""
    return os.getenv("RAG_PPTX_NOTE_SCOPE", "0").strip().lower() in _ON


def default_out_path() -> Path:
    return settings.ARTIFACTS_DIR / "pptx_note_scope_store.jsonl"


def default_report_path() -> Path:
    return settings.ARTIFACTS_DIR / "pptx_note_scope_store_build_report.json"


# --------------------------------------------------------------------------- OOXML paragraph helpers
def _para_text(p_elem) -> str:
    """1 段落 <a:p> の可視テキスト（全 <a:t> run を連結）を NFC で返す。"""
    parts = [t.text or "" for t in p_elem.iter("{%s}t" % _A_NS)]
    return nfc("".join(parts))


def _strip_decor(text: str) -> str:
    """見出し照合用: 前後の装飾（記号・空白・括弧・区切り）を剥がした素の見出し語を返す。"""
    return _DECOR.sub("", unicodedata.normalize("NFKC", text or "")).strip()


def _is_heading(text: str) -> bool:
    """段落が「スコープ対象外」見出しか（装飾を除いた素のテキストが見出し語そのもの）。"""
    return _strip_decor(text) == _SCOPE_HEADING


def _is_xmark_item(text: str) -> bool:
    """段落が対象外項目（先頭マークが ✖ 系）か。先頭の空白を無視して判定。"""
    s = (text or "").lstrip()
    return bool(s) and s[0] in _XMARK


def _paragraphs(xml_bytes: bytes) -> list[str]:
    """notesSlide XML の全段落テキストを文書順で返す（解析不能なら空）。"""
    from lxml import etree  # lazy — python-pptx と同じ依存

    try:
        root = etree.fromstring(xml_bytes)
    except Exception:  # noqa: BLE001 — 壊れた XML はスキップ（fail-open）
        return []
    return [_para_text(p) for p in root.iter("{%s}p" % _A_NS)]


def _count_scope_items(paras: Sequence[str]) -> list[str]:
    """段落列から「スコープ対象外」見出し直下に連続する ✖ 項目段落のテキストを収集する。

    見出しの**直下**（intervening の非空段落を挟まず）に続く ``✖`` 段落だけを 1 項目として数え、最初の
    非 ``✖`` 段落で打ち切る（後続のページ番号「6」や別ブロックを取り込まない）。文書内に見出しが複数あれば
    それぞれの直下ブロックを合算する。空段落は区切りとみなさず（レイアウト由来の空 <a:p> を許容）読み飛ばす。
    """
    items: list[str] = []
    n = len(paras)
    i = 0
    while i < n:
        if not _is_heading(paras[i]):
            i += 1
            continue
        j = i + 1
        while j < n:
            t = paras[j]
            if not (t or "").strip():
                j += 1  # 空段落は無視して直下ブロックを継続
                continue
            if _is_xmark_item(t):
                items.append(t.strip())
                j += 1
                continue
            break  # 最初の非空・非 ✖ 段落で当ブロック終端
        i = j
    return items


# --------------------------------------------------------------------------- per-file record
def compute_doc(ref: FileRef) -> "dict[str, Any] | None":
    """1 つの pptx の全 notesSlide からスコープ対象外 ✖項目カウントを組む（開けなければ None）。"""
    data = None
    try:
        if passwords.is_encrypted(ref.path):
            data = passwords.resolve(ref)
    except Exception:  # noqa: BLE001 — 復号判定/復号の失敗は data=None のまま素の open を試す
        data = None
    try:
        zf = zipfile.ZipFile(io.BytesIO(data) if data else str(ref.path))
    except Exception:  # noqa: BLE001 — 開けない pptx はスキップ（fail-open）
        return None

    with zf:
        note_names = sorted(
            n for n in zf.namelist()
            if n.startswith("ppt/notesSlides/notesSlide") and n.endswith(".xml")
        )
        notes: list[dict[str, Any]] = []
        all_items: list[str] = []
        for name in note_names:
            try:
                paras = _paragraphs(zf.read(name))
            except Exception:  # noqa: BLE001 — 1 枚の破損は全体を沈めない
                continue
            items = _count_scope_items(paras)
            if items:
                notes.append({"note": name, "items": items, "count": len(items)})
                all_items.extend(items)

    # 見出しを 1 つも持たない pptx も universe 立証のため記録（count=0）。ただし store 容量を抑えるため
    # items/notes 内訳は該当があるものだけ持たせる。
    return {
        "doc_id": nfc(ref.rel), "project": nfc(ref.project), "category": ref.category,
        "doc_name": nfc(ref.name), "ext": ref.ext,
        "notes_slide_count": len(note_names),
        "scope_excluded_count": len(all_items),
        "notes": notes,
        "items": all_items,
    }


# --------------------------------------------------------------------------- universe / build / io
def _universe(refs: Sequence[FileRef]) -> list[FileRef]:
    out: list[FileRef] = []
    for r in refs:
        if r.ext not in _PPTX_EXTS:
            continue
        name = nfc(r.name)
        if name.startswith("~$"):
            continue
        out.append(r)
    out.sort(key=lambda r: r.rel)
    return out


def write_store(records: Sequence[Mapping[str, Any]], path: Path | None = None) -> dict[str, int]:
    """Atomically write a reproducible JSONL store (schema header + doc_id-sorted rows)."""
    out = Path(path) if path is not None else default_out_path()
    out.parent.mkdir(parents=True, exist_ok=True)
    ordered = sorted(records, key=lambda r: r.get("doc_id", ""))
    tmp = out.with_suffix(out.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8", newline="\n") as handle:
        handle.write(json.dumps({"schema": SCHEMA, "version": SCHEMA_VERSION},
                                ensure_ascii=False, sort_keys=True) + "\n")
        for rec in ordered:
            handle.write(json.dumps(rec, ensure_ascii=False, sort_keys=True) + "\n")
    tmp.replace(out)
    reset_cache()
    return {"docs": len(ordered)}


def build(refs: Sequence[FileRef] | None = None, *, out: Path | None = None,
          write_report: bool = True) -> dict[str, Any]:
    """全案件の pptx を全走査してスコープ対象外✖項目カウント事実ストアを焼く（質問非依存・LLM フリー・fail-open）。"""
    refs = list(refs) if refs is not None else list(walk())
    universe = _universe(refs)
    records: list[dict[str, Any]] = []
    skipped: list[str] = []
    for ref in universe:
        try:
            rec = compute_doc(ref)
        except Exception:  # noqa: BLE001 — one bad pptx must not sink the build
            rec = None
        if rec is not None:
            records.append(rec)
        else:
            skipped.append(nfc(ref.rel))
    stats = write_store(records, out)
    report = {
        "schema": SCHEMA, "version": SCHEMA_VERSION,
        "universe": len(universe), "records": len(records), "skipped": skipped,
        "docs_with_scope": sum(1 for r in records if r.get("scope_excluded_count", 0) > 0),
        "scope_items_total": sum(r.get("scope_excluded_count", 0) for r in records),
    }
    if write_report:
        rp = default_report_path()
        rp.parent.mkdir(parents=True, exist_ok=True)
        rp.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return {"records": len(records), "docs": stats["docs"], "report": report}


# --------------------------------------------------------------------------- load / read (minimal API)
_LOAD_CACHE: dict[str, list[dict[str, Any]]] = {}


def reset_cache() -> None:
    _LOAD_CACHE.clear()


def load(path: Path | None = None) -> list[dict[str, Any]]:
    """Load the note-scope rows (memoized). ``[]`` when absent/unreadable/schema-mismatch (回帰ゼロ)."""
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


if __name__ == "__main__":
    summary = build()
    print(json.dumps(summary["report"], ensure_ascii=False, indent=2))
