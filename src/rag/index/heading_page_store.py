"""見出し→印字ページ locator ストア（SOT-2668 / 事前計算事実層 追補 — クラスタC5）.

Sonnet gold100 cycle5 の abstain のうち idx12/18 は「見出しは特定できるが**印字ページ番号**へ到達
できない」型で棄権していた（`docs/ai/sonnet_cycle_analysis/cycle5.md` クラスタC5）:

* idx12 — docx「報告資料_2025-07-08.docx」の見出し「WBS観点の進捗状況」は何ページ（gold=印字p2）。
* idx18 — 会議ID:M04 の会議録PDF で「進捗サマリ」が記載されているページ番号（gold=印字p2）。

docx は XML 上にレンダリング済みページ境界を持たない（このコーパスの報告資料 docx は
``lastRenderedPageBreak`` / 明示改ページ 0、``docProps/app.xml`` の Pages も陳腐化）ので、build 時に一度だけ
**LibreOffice headless で docx→PDF へレンダ**し、pymupdf でページ毎テキストと印字フッタ番号を取り、
各見出し（python-docx の Heading スタイル段落）→印字ページ番号を確定する。PDF は text-layer があれば
pymupdf、スキャンPDF（会議録）は前処理で確定済みの永続 OCR（``[ページ N]`` マーカ）を読み、番号付き節見出しを
抽出する。印字ページ=フッタの ``N``（表紙除外規約, SOT-2612/2549）で、検出できなければ物理 1-based に退避。

Design invariants（sibling の action_row_store / visual_store と同一）
--------------------------------------------------------------------
* **Opt-in at serve time.** :func:`enabled` (``RAG_HEADING_PAGE_STORE``) が runtime 参照のみを gate する。
  default OFF ⇒ champion serve path は byte-identical。読み出し/配線は :mod:`src.rag.agent.heading_page_lane`。
* **Build は質問非依存・LLM フリー・追加的.** 見る材料は既存の永続 OCR / text-layer / docx スタイルと、
  build 時の決定論レンダ（LibreOffice）だけ。genai 呼び出しはしない。読めない文書は 1 件スキップし build は
  継続（fail-open）。gold も質問も参照しない。
* **No hard-coded answers.** 値は毎見出し出典（doc/page）付き。idx 番号も gold ページ番号も埋め込まない。
* **Fail-open.** artifact 欠落・解析不能・soffice 不在はすべて空へフォールバック（回帰ゼロ）。
"""
from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import tempfile
import unicodedata
from pathlib import Path
from typing import Any, Mapping, Sequence

from config import settings
from src.rag.corpus import FileRef, nfc, walk

SCHEMA = "heading-page-store"
SCHEMA_VERSION = 1

_ON = {"1", "true", "yes", "on"}

# docx/pdf のみ（xlsx/pptx は別ストア）。~$ 一時ファイルは除外。
_DOC_EXTS = {"docx", "pdf"}

# OCR/テキストのページ区切りマーカ（action_row_store と同型: "[ページ N]"）。
_PAGE_MARK = re.compile(r"\[ページ\s*(\d+)\]")
# 会議ID / 資料ID 等、文書が自称する識別コード（ページ1ヘッダ由来のみ採用）。
_DOC_ID_DECL = re.compile(r"(?:会議|資料|報告|文書|ドキュメント)ID\s*[:：]\s*([A-Za-z]-?\d{1,3})")
_DATE_IN_NAME = re.compile(r"(\d{4})-(\d{2})-(\d{2})")
# 番号付き節見出し: "4. 進捗サマリ" / "2.3 WBS観点の進捗状況"（全角ドット/読点も許容）。
_NUMBERED_HEADING = re.compile(r"^\s*(\d+(?:[.．]\d+)*)\s*[.．、)）]?\s+(\S.{0,58})$")
# 見出しの先頭番号（"2.3 " / "4. " / "4、"）— マッチ時に剥がす。
_LEADING_NUM = re.compile(r"^\s*\d+(?:[.．]\d+)*\s*[.．、)）]?\s*")


def enabled() -> bool:
    """True when the serve path may consult the heading→page store (default OFF — opt-in)."""
    return os.getenv("RAG_HEADING_PAGE_STORE", "0").strip().lower() in _ON


def default_out_path() -> Path:
    return settings.ARTIFACTS_DIR / "heading_page_store.jsonl"


def default_report_path() -> Path:
    return settings.ARTIFACTS_DIR / "heading_page_store_build_report.json"


def norm(text: Any) -> str:
    """NFKC + 空白除去 + 小文字化（見出し/質問の突合キー）。"""
    return re.sub(r"\s+", "", unicodedata.normalize("NFKC", str(text or ""))).lower()


def strip_heading_number(text: str) -> str:
    """先頭の節番号（"2.3 " / "4. "）を落として見出し本文だけを返す。"""
    return _LEADING_NUM.sub("", str(text or "")).strip()


# --------------------------------------------------------------------------- printed-page detection
def _printed_page(page_text: str, physical: int) -> int:
    """1 ページ分のテキストから印字ページ番号を推定（フッタ優先, 無ければ物理 1-based）。

    規約（SOT-2549/2612「競合記載は確定版=印字ページ番号を優先」）: フッタの ``N / TOTAL`` か、末尾行が
    数字のみ（フッタのページ番号）を印字番号として採る。どちらも無ければ物理インデックスに退避（表紙が
    別ページなら footer 検出がそれを吸収する）。
    """
    text = (page_text or "").strip()
    # "N / TOTAL" フッタ（最後の出現）。
    slash = list(re.finditer(r"(\d{1,3})\s*[/／]\s*\d{1,3}", text))
    if slash:
        try:
            return int(slash[-1].group(1))
        except ValueError:
            pass
    # 末尾行が数字のみ（フッタのページ番号）。
    m = re.search(r"(?:^|\n)\s*(\d{1,3})\s*$", text)
    if m:
        n = int(m.group(1))
        if 1 <= n <= 999:
            return n
    return physical


# --------------------------------------------------------------------------- per-page text sources
def _ocr_pages(ref: FileRef) -> "list[tuple[int, str]] | None":
    """永続 OCR テキストを ``[ページ N]`` で分割 → [(印字ページ, chunk)]。無ければ None。"""
    try:
        from src.rag.index import ocr_store
        text = ocr_store.lookup(ref)
    except Exception:  # noqa: BLE001
        text = None
    if not text or not text.strip():
        return None
    text = nfc(text)
    pages: list[tuple[int, str]] = []
    page = 1
    buf: list[str] = []
    for line in text.splitlines():
        pm = _PAGE_MARK.search(line)
        if pm:
            if buf:
                pages.append((page, "\n".join(buf)))
                buf = []
            page = int(pm.group(1))
            continue
        buf.append(line)
    if buf:
        pages.append((page, "\n".join(buf)))
    return pages or None


def _pdf_textlayer_pages(ref: FileRef) -> "list[tuple[int, str]] | None":
    """text-layer を持つ PDF のページ毎テキスト → [(印字ページ, text)]。空(=スキャン)なら None。"""
    try:
        import fitz  # PyMuPDF
    except Exception:  # noqa: BLE001
        return None
    try:
        doc = fitz.open(str(ref.path))
    except Exception:  # noqa: BLE001
        return None
    out: list[tuple[int, str]] = []
    total_len = 0
    try:
        for i in range(doc.page_count):
            t = doc[i].get_text() or ""
            total_len += len(t.strip())
            out.append((_printed_page(t, i + 1), t))
    finally:
        doc.close()
    if total_len == 0:  # scanned image PDF → let the OCR path handle it
        return None
    return out


def _render_docx_pages(ref: FileRef) -> "list[tuple[int, str]] | None":
    """docx を LibreOffice headless で PDF へレンダ → ページ毎テキスト [(印字ページ, text)]。失敗時 None。"""
    soffice = shutil.which("soffice") or shutil.which("libreoffice")
    if not soffice:
        return None
    try:
        import fitz  # PyMuPDF
    except Exception:  # noqa: BLE001
        return None
    tmp = tempfile.mkdtemp(prefix="hpstore_")
    try:
        proc = subprocess.run(
            [soffice, "--headless", "--convert-to", "pdf", "--outdir", tmp, str(ref.path)],
            capture_output=True, text=True, timeout=180, env={**os.environ, "HOME": tmp})
        if proc.returncode != 0:
            return None
        pdfs = [f for f in os.listdir(tmp) if f.lower().endswith(".pdf")]
        if not pdfs:
            return None
        doc = fitz.open(os.path.join(tmp, pdfs[0]))
        out: list[tuple[int, str]] = []
        try:
            for i in range(doc.page_count):
                t = doc[i].get_text() or ""
                out.append((_printed_page(t, i + 1), t))
        finally:
            doc.close()
        return out or None
    except Exception:  # noqa: BLE001 — soffice hiccup must not sink the build
        return None
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


# --------------------------------------------------------------------------- heading extraction
def _docx_style_headings(ref: FileRef) -> list[str]:
    """python-docx で Heading/見出し スタイル段落のテキストを doc 順に返す。"""
    try:
        import docx
        d = docx.Document(str(ref.path))
    except Exception:  # noqa: BLE001
        return []
    heads: list[str] = []
    for para in d.paragraphs:
        style = (para.style.name if para.style else "") or ""
        text = (para.text or "").strip()
        if not text:
            continue
        if style.lower().startswith("heading") or "見出し" in style or style.lower().startswith("title"):
            heads.append(nfc(text))
    return heads


def _numbered_headings(pages: Sequence[tuple[int, str]]) -> list[dict[str, Any]]:
    """ページ毎テキストから番号付き節見出し行を抽出（[{text, page}]）。"""
    out: list[dict[str, Any]] = []
    for page, chunk in pages:
        for raw in nfc(chunk).splitlines():
            line = raw.strip()
            m = _NUMBERED_HEADING.match(line)
            if not m:
                continue
            title = m.group(2).strip()
            # 表・箇条の値行を弾く: パイプ/長すぎ/末尾が読点でない短句のみ採る。
            if "|" in title or "｜" in title or len(title) < 2:
                continue
            out.append({"text": title, "page": page})
    return out


def _assign_docx_heading_pages(headings: Sequence[str],
                               pages: Sequence[tuple[int, str]]) -> list[dict[str, Any]]:
    """docx の見出しテキストをレンダ済みページに単調割当（[{text, page}]）。"""
    norm_pages = [(p, norm(t)) for (p, t) in pages]
    out: list[dict[str, Any]] = []
    last_phys = 0
    for h in headings:
        key = norm(strip_heading_number(h)) or norm(h)
        if not key:
            continue
        page = None
        for phys, (printed, ptext) in enumerate(norm_pages):
            if phys < last_phys:
                continue
            if key in ptext:
                page = printed
                last_phys = phys
                break
        if page is None:  # 割当先が見つからない見出しは記録しない（偽ページを作らない）
            continue
        out.append({"text": h, "page": page})
    return out


def _doc_ids(pages: Sequence[tuple[int, str]]) -> list[str]:
    """文書が自称する識別コード（会議ID/資料ID 等）をページ1ヘッダから抽出（区切り正規化）。"""
    if not pages:
        return []
    p1 = nfc(pages[0][1])
    ids: list[str] = []
    for m in _DOC_ID_DECL.finditer(p1):
        code = re.sub(r"[\s\-_]", "", m.group(1)).upper()
        if code and code not in ids:
            ids.append(code)
    return ids


def _doc_date(ref: FileRef) -> str | None:
    m = _DATE_IN_NAME.search(nfc(ref.name))
    return f"{m.group(1)}-{m.group(2)}-{m.group(3)}" if m else None


def compute_doc(ref: FileRef) -> dict[str, Any] | None:
    """1 文書から見出し→印字ページ レコードを組む（見出しが取れなければ None）。"""
    if ref.ext == "docx":
        pages = _render_docx_pages(ref)
        headings = _assign_docx_heading_pages(_docx_style_headings(ref), pages) if pages else []
        # docx でも番号付き節見出しを補完（スタイル欠落文書のため）。
        if pages:
            seen = {(norm(h["text"]), h["page"]) for h in headings}
            for h in _numbered_headings(pages):
                if (norm(h["text"]), h["page"]) not in seen:
                    headings.append(h)
    else:  # pdf
        pages = _pdf_textlayer_pages(ref) or _ocr_pages(ref)
        headings = _numbered_headings(pages) if pages else []
    if not pages or not headings:
        return None
    page_count = max(p for p, _ in pages)
    rec: dict[str, Any] = {
        "doc_id": nfc(ref.rel), "project": nfc(ref.project), "category": ref.category,
        "doc_name": nfc(ref.name), "ext": ref.ext, "date": _doc_date(ref),
        "doc_ids": _doc_ids(pages), "page_count": page_count,
        "headings": headings,
    }
    return rec


# --------------------------------------------------------------------------- universe / build / io
def _universe(refs: Sequence[FileRef]) -> list[FileRef]:
    out: list[FileRef] = []
    for r in refs:
        if r.ext not in _DOC_EXTS:
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
    """docx/pdf を全走査して見出し→印字ページ ストアを焼く（質問非依存・LLM フリー・fail-open）。"""
    refs = list(refs) if refs is not None else list(walk())
    universe = _universe(refs)
    records: list[dict[str, Any]] = []
    skipped: list[str] = []
    for ref in universe:
        try:
            rec = compute_doc(ref)
        except Exception:  # noqa: BLE001 — one bad doc must not sink the build
            rec = None
        if rec is not None:
            records.append(rec)
        else:
            skipped.append(nfc(ref.rel))
    stats = write_store(records, out)
    report = {
        "schema": SCHEMA, "version": SCHEMA_VERSION,
        "universe": len(universe), "records": len(records), "skipped": skipped,
        "headings": sum(len(r.get("headings", [])) for r in records),
        "docx": sum(1 for r in records if r.get("ext") == "docx"),
        "pdf": sum(1 for r in records if r.get("ext") == "pdf"),
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
    """Load the heading→page rows (memoized). ``[]`` when absent/unreadable/schema-mismatch (回帰ゼロ)."""
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
