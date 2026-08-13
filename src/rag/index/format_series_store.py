"""書式付き数値の文書系列ストア（SOT-2693 / cycle8 C4, idx17）.

Sonnet gold100 cycle7/8 の abstain idx17 は、証拠が **文書横断の書式付き数値系列** にあり、
質問非依存に「同じ報告系列の各文書から 黄色ハイライト×赤字 の数値を日付順に並べる」手段が無いのが
主因だった（`docs/ai/sonnet_cycle_analysis/cycle8.md` C4）。

* **idx17**（青葉与信マネジメント＝AYM の MM＝会議録系列）: 各 会議録 の本文で **黄色ハイライトかつ赤字**
  になっている数値（対象）を、最初の MM から最後の MM まで日付順に並べ、上昇率
  ``(最後 − 最初) / 最初 × 100`` を求める。ハイライト値の存在は format_events で確認済みだったが、
  「MM＝会議録という系列の定義・全件列挙（日付順）」＋「文書横断の 黄×赤 数値系列」＋「上昇率派生」の
  手段が無かった。

本モジュールは build 時に一度だけ、全案件の **報告系列文書**（05.会議 配下の `会議録`/`報告資料` 系列）を
**質問を見ずに全数** 走査し、各文書の 黄×赤 数値を日付順の系列として焼き込む。docx は既存の
:mod:`src.rag.tools.format_events`（fill×font 色, SOT-2585/2564）を、PDF は pdfplumber の
文字色 × 黄色矩形（ハイライト）重なりを、いずれも決定論・genai 非使用で読む
（`derived_metrics` / `visual_store` / `xlsx_formula_trace` と同一の LLM-free・追加的・fail-open 規律）。
読み出し/配線は :mod:`src.rag.agent.format_series_lane`。

Design invariants
-----------------
* **Opt-in at serve time.** :func:`enabled` (``RAG_FORMAT_SERIES``) が runtime 参照のみ gate する。
  default OFF ⇒ champion serve path は byte-identical。
* **Build は LLM フリー・追加的.** format_events（決定論）＋ pdfplumber（文字色/矩形）のみ。読めない文書は
  1 件スキップして継続。
* **Question-independent.** universe は全案件の 会議録/報告資料 系列。質問も gold も参照せず、全 黄×赤 数値を
  網羅抽出する。系列種別は **パスのフォルダ名**（会議録/報告資料）から、日付は **ファイル名の YYYY-MM-DD**
  から決定論的に付与する。
* **No hardcoding.** gold 値・idx 番号・上昇率を一切持たない。全値は原ファイルから読み、出典（doc/日付）付き。
* **Fail-open.** artifact 欠落・解析不能はすべて空へフォールバック（回帰ゼロ）。
"""
from __future__ import annotations

import json
import os
import re
import unicodedata
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from config import settings
from src.rag.corpus import FileRef, nfc, walk

SCHEMA = "format-series-store"
SCHEMA_VERSION = 1

_ON = {"1", "true", "yes", "on"}

# 報告系列文書の対象拡張子（会議録=pdf, 報告資料=docx 等）。
_DOC_EXTS = {"pdf", "docx"}
# パスのフォルダ名から系列種別を決める（会議録=MM / 報告資料=RP）。他の系列名もここに載れば拾える。
_SERIES_FOLDERS = ("会議録", "報告資料", "議事録", "定例", "月次")
# 日付をファイル名/パスから拾う（YYYY-MM-DD / YYYY_MM_DD / YYYYMMDD）。
_DATE_RES = (
    re.compile(r"(\d{4})[-_./](\d{1,2})[-_./](\d{1,2})"),
    re.compile(r"(\d{4})(\d{2})(\d{2})"),
)
# 黄×赤の対象値として拾う数値トークン（カンマ区切り整数・小数）。英数字直後の桁は拾わない
# （"F1" の "1" のような識別子内数字を対象値と誤認しないため — precision-first）。
_NUM_RE = re.compile(r"(?<![A-Za-z0-9])-?\d[\d,]*(?:\.\d+)?")


def enabled() -> bool:
    """True when the serve path may consult the formatted-value series store (default OFF — opt-in)."""
    return os.getenv("RAG_FORMAT_SERIES", "0").strip().lower() in _ON


def default_out_path() -> Path:
    return settings.ARTIFACTS_DIR / "format_series_store.jsonl"


def default_report_path() -> Path:
    return settings.ARTIFACTS_DIR / "format_series_store_build_report.json"


# --------------------------------------------------------------------------- helpers
def norm(text: Any) -> str:
    return unicodedata.normalize("NFKC", str(text or "")).replace(" ", "").replace("　", "").lower()


def _series_type(rel: str) -> str | None:
    """系列種別（フォルダ名）を rel から決定論抽出（会議録/報告資料/…）。無ければ None。"""
    parts = [nfc(p) for p in str(rel or "").split("/")]
    for p in parts:
        for folder in _SERIES_FOLDERS:
            if folder in p:
                return folder
    return None


def _doc_date(ref: FileRef) -> str | None:
    """文書日付（YYYY-MM-DD）をファイル名/パスから決定論抽出。無ければ None。"""
    hay = nfc(ref.name) + " " + nfc(ref.rel)
    for rx in _DATE_RES:
        m = rx.search(hay)
        if m:
            y, mo, d = int(m.group(1)), int(m.group(2)), int(m.group(3))
            try:
                if 1 <= mo <= 12 and 1 <= d <= 31:
                    return f"{y:04d}-{mo:02d}-{d:02d}"
            except Exception:  # noqa: BLE001
                return None
    return None


def _numbers(text: str) -> list[str]:
    """テキストから数値トークンを抽出（順序保存）。"""
    return _NUM_RE.findall(str(text or ""))


def _to_float(token: str) -> float | None:
    try:
        return float(str(token).replace(",", ""))
    except (TypeError, ValueError):
        return None


# --------------------------------------------------------------------------- docx (format_events)
def _docx_yellow_red(ref: FileRef) -> list[dict[str, Any]]:
    """docx 本文の 黄色ハイライト×赤字 数値イベント（決定論 format_events を流用）。"""
    from src.rag.tools import format_events as fe

    out: list[dict[str, Any]] = []
    try:
        res = fe.format_events(ref, fill="黄", font_color="赤")
    except Exception:  # noqa: BLE001 — 読めない docx はスキップ
        return out
    for item in res.get("value", []):
        for tok in _numbers(item.get("value")):
            f = _to_float(tok)
            if f is None:
                continue
            out.append({"raw": tok, "value": f,
                        "loc": (item.get("evidence") or {}).get("paragraph")})
    return out


# --------------------------------------------------------------------------- pdf (color × highlight)
def _is_red(color: Any) -> bool:
    """非ストローク色が赤系か（RGB / CMYK / gray いずれも許容）。"""
    if isinstance(color, (tuple, list)):
        if len(color) == 3:
            r, g, b = color
            return r > 0.5 and g < 0.45 and b < 0.45
        if len(color) == 4:
            c, m, y, k = color
            return m > 0.5 and y > 0.5 and c < 0.35 and k < 0.4
        if len(color) == 1:
            return False
    return False


def _is_yellow(color: Any) -> bool:
    """塗り色が黄系か（RGB / CMYK いずれも許容）。"""
    if isinstance(color, (tuple, list)):
        if len(color) == 3:
            r, g, b = color
            return r > 0.7 and g > 0.7 and b < 0.6
        if len(color) == 4:
            c, m, y, k = color
            return y > 0.4 and c < 0.3 and m < 0.45 and k < 0.35
    return False


def _pdf_yellow_red(ref: FileRef) -> list[dict[str, Any]]:
    """PDF 本文で 赤字 かつ 黄色矩形（ハイライト）に重なる文字列 → 数値イベント。"""
    try:
        import pdfplumber  # lazy
    except Exception:  # noqa: BLE001
        return []
    out: list[dict[str, Any]] = []
    try:
        with pdfplumber.open(str(ref.path)) as pdf:
            for pi, page in enumerate(pdf.pages, 1):
                yrects = [rc for rc in page.rects if _is_yellow(rc.get("non_stroking_color"))]
                if not yrects:
                    continue
                chars = []
                for ch in page.chars:
                    if not _is_red(ch.get("non_stroking_color")):
                        continue
                    cx = (ch["x0"] + ch["x1"]) / 2.0
                    cy = (ch["top"] + ch["bottom"]) / 2.0
                    inside = any(rc["x0"] - 1 <= cx <= rc["x1"] + 1
                                 and rc["top"] - 2 <= cy <= rc["bottom"] + 2 for rc in yrects)
                    if inside:
                        chars.append(ch["text"])
                text = "".join(chars)
                for tok in _numbers(text):
                    f = _to_float(tok)
                    if f is None:
                        continue
                    out.append({"raw": tok, "value": f, "loc": f"ページ{pi}"})
    except Exception:  # noqa: BLE001 — 壊れた PDF はスキップ
        return []
    return out


# --------------------------------------------------------------------------- per-doc / build
def _doc_events(ref: FileRef) -> list[dict[str, Any]]:
    if ref.ext == "docx":
        return _docx_yellow_red(ref)
    if ref.ext == "pdf":
        return _pdf_yellow_red(ref)
    return []


def _universe(refs: Sequence[FileRef]) -> list[FileRef]:
    out: list[FileRef] = []
    for r in refs:
        if r.ext not in _DOC_EXTS:
            continue
        if _series_type(r.rel) is None:
            continue
        name = nfc(r.name).lower()
        if name.startswith("~$") or "/old/" in nfc(r.rel).lower():
            continue
        out.append(r)
    out.sort(key=lambda r: r.rel)
    return out


def build(refs: Sequence[FileRef] | None = None, *, out: Path | None = None,
          write_report: bool = True) -> dict[str, Any]:
    """全案件の報告系列を走査し、書式付き数値系列ストアを構築（LLM 非使用・べき等・fail-open）。

    レコードは ``(project, series_type)`` 単位で、その系列の全文書を **日付順** に並べ、各文書の
    黄×赤 数値イベントを保持する。日付が取れない文書は rel 順で末尾に回す。
    """
    refs = list(refs) if refs is not None else list(walk())
    universe = _universe(refs)
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for ref in universe:
        stype = _series_type(ref.rel)
        if stype is None:
            continue
        try:
            events = _doc_events(ref)
        except Exception:  # noqa: BLE001 — 一文書の失敗で build を落とさない
            events = []
        entry = {
            "doc_id": nfc(ref.rel), "doc_name": nfc(ref.name), "ext": ref.ext,
            "date": _doc_date(ref), "events": events,
        }
        grouped.setdefault((nfc(ref.project), stype), []).append(entry)

    records: list[dict[str, Any]] = []
    for (project, stype), docs in grouped.items():
        docs.sort(key=lambda d: (d.get("date") is None, d.get("date") or "", d.get("doc_id") or ""))
        records.append({
            "project": project, "series_type": stype,
            "n_docs": len(docs), "docs": docs,
        })
    stats = write_store(records, out)
    report = {
        "schema": SCHEMA, "version": SCHEMA_VERSION,
        "universe": len(universe), "records": len(records),
        "series": [{"project": r["project"], "series_type": r["series_type"],
                    "docs": r["n_docs"],
                    "docs_with_events": sum(1 for d in r["docs"] if d["events"])}
                   for r in records],
    }
    if write_report:
        rp = default_report_path()
        rp.parent.mkdir(parents=True, exist_ok=True)
        rp.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return {"records": len(records), "docs": stats["docs"], "report": report}


# --------------------------------------------------------------------------- io
def write_store(records: Sequence[Mapping[str, Any]], path: Path | None = None) -> dict[str, int]:
    """Atomically write a reproducible JSONL store (schema header + (project,series) sorted rows)."""
    out = Path(path) if path is not None else default_out_path()
    out.parent.mkdir(parents=True, exist_ok=True)
    ordered = sorted(records, key=lambda r: (r.get("project", ""), r.get("series_type", "")))
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
    """Load the series records (memoized). ``[]`` when absent/unreadable/schema-mismatch (回帰ゼロ)."""
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


def series_for(project_hint: str, series_type: str, *,
               path: Path | None = None) -> dict[str, Any] | None:
    """``(project, series_type)`` に一致する系列レコード（部分一致・空白/ケース無視）。無ければ None。"""
    hint = norm(project_hint)
    st = norm(series_type)
    if not hint or not st:
        return None
    for r in load(path):
        if norm(r.get("series_type")) != st:
            continue
        pr = norm(r.get("project"))
        if pr and (hint in pr or pr in hint):
            return r
    return None


if __name__ == "__main__":
    summary = build()
    print(f"[build] format_series_store records={summary['records']} "
          f"docs={summary['docs']} -> {default_out_path()}")
