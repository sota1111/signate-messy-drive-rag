"""PDF 数値表（帯×税率）の doc-table ストア＋帯別差分派生（SOT-2693 / cycle8 C4, idx48）.

Sonnet gold100 cycle6/7/8 の abstain idx48 は、証拠が PDF の **価格帯×税率表** にあり、file_grep で
言及行までは到達できても表全体を予算内で読み切れず棄権していた（`docs/ai/sonnet_cycle_analysis/cycle8.md`
C4）。cycle6 では MATCH していた＝テキスト到達可能なので、欠けているのは証拠ではなく **表の doc-table
収載＋帯別差分派生**（浅到達）である。

* **idx48**（青嶺不動産アセットマネジメント「ニューヨーク不動産市場の最新動向調査.pdf」）: マンション税の
  価格帯別 現行税率/新税率表から、**現行税率からの絶対値の増加が最も小さい価格帯**（argmin）を返す。
  例: 100万ドル超-500万ドル以下 は現行が範囲 1.00%-1.50%・新 1.425% ⇒ 絶対増加が全帯で最小。

本モジュールは build 時に一度だけ、全 PDF を **質問を見ずに全数** 走査し、``現行(率)`` と ``新(率)`` の
2 数値列を持つ「帯×率」表を検出して各帯の ``|新 − 現行|`` を事前計算し、argmin/argmax を焼き込む。
pdfplumber の ``extract_tables`` のみ（genai 非使用）。読み出し/配線は :mod:`src.rag.agent.rate_table_lane`。

Design invariants
-----------------
* **Opt-in at serve time.** :func:`enabled` (``RAG_RATE_TABLE``) が runtime 参照のみ gate する。
  default OFF ⇒ champion serve path は byte-identical。
* **Build は LLM フリー・追加的.** pdfplumber で表セルを読むだけ。読めない PDF は 1 件スキップして継続。
* **Question-independent.** universe は全 PDF。質問も gold も参照せず、率2列の全「帯×率」表を網羅計算する。
* **No hardcoding.** gold 値・idx 番号・帯名を一切持たない。全値は原ファイルから読み、出典（doc/ページ）付き。
* **Fail-open.** artifact 欠落・解析不能はすべて空へフォールバック（回帰ゼロ）。
"""
from __future__ import annotations

import json
import os
import re
import unicodedata
from pathlib import Path
from typing import Any, Mapping, Sequence

from config import settings
from src.rag.corpus import FileRef, nfc, walk

SCHEMA = "rate-table-store"
SCHEMA_VERSION = 1

_ON = {"1", "true", "yes", "on"}

# 「現行」率列 / 「新」率列 のヘッダ語。
_CURRENT_HDR = ("現行", "現在", "従来", "旧")
_NEW_HDR = ("新税率", "新しい", "新率", "提案", "改定後", "変更後", "新")
# 帯（価格帯）ヘッダ語 / セル内に含まれる帯らしさの語。
_BAND_HDR = ("価格帯", "帯", "区分", "レンジ", "range")
_BAND_CELL_CUES = ("ドル", "円", "以下", "以上", "超", "未満", "～", "-")
# 率トークン: "1.425%" / "1.00% - 1.50%" / "0.0%"（％は任意）。
_PCT_RE = re.compile(r"-?\d+(?:\.\d+)?\s*%?")


def enabled() -> bool:
    """True when the serve path may consult the rate-table store (default OFF — opt-in)."""
    return os.getenv("RAG_RATE_TABLE", "0").strip().lower() in _ON


def default_out_path() -> Path:
    return settings.ARTIFACTS_DIR / "rate_table_store.jsonl"


def default_report_path() -> Path:
    return settings.ARTIFACTS_DIR / "rate_table_store_build_report.json"


# --------------------------------------------------------------------------- helpers
def norm(text: Any) -> str:
    return unicodedata.normalize("NFKC", str(text or "")).replace(" ", "").replace("　", "").lower()


def _clean(cell: Any) -> str:
    return unicodedata.normalize("NFKC", str(cell or "")).replace("\n", " ").strip()


def _norm_band(text: str) -> str:
    """帯ラベルを正規化（範囲 ' - ' 区切りは保持し、各境界の内部空白＝PDF 折返し由来を除去）。

    例: ``"100 万ドル超 - 500 万ドル以 下"`` → ``"100万ドル超 - 500万ドル以下"``（gold 書式）。
    """
    s = unicodedata.normalize("NFKC", str(text or "")).replace("\n", "").strip()
    parts = re.split(r"\s*[-−–—]\s*", s)
    parts = [re.sub(r"[ \t　]+", "", p) for p in parts if p.strip()]
    return " - ".join(parts)


def _parse_rate(cell: str) -> dict[str, Any] | None:
    """率セルを解析。単一値 ⇒ {value}, 範囲 ⇒ {low, high, mid}。数値が無ければ None。"""
    nums = [float(m.group(0).replace("%", "").strip())
            for m in _PCT_RE.finditer(str(cell or "")) if m.group(0).strip() not in ("", "%")]
    nums = [n for n in nums]
    if not nums:
        return None
    if len(nums) == 1:
        return {"raw": _clean(cell), "value": nums[0]}
    low, high = min(nums[0], nums[-1]), max(nums[0], nums[-1])
    return {"raw": _clean(cell), "low": low, "high": high, "mid": (low + high) / 2.0}


def _rate_repr(rate: Mapping[str, Any]) -> float:
    """率の代表値（範囲は中点）。"""
    if "value" in rate:
        return float(rate["value"])
    return float(rate["mid"])


def _is_band_cell(text: str) -> bool:
    t = _clean(text)
    if not t:
        return False
    return any(cue in t for cue in _BAND_CELL_CUES) and bool(re.search(r"\d", t))


# --------------------------------------------------------------------------- table detection
def _header_columns(row: list[str]) -> "tuple[int, int, int] | None":
    """ヘッダ行から (band_col, current_col, new_col) を決める。決められなければ None。

    「新」は「現行」より後方の列を優先（両者が同一セル群に無いよう区別）。
    """
    cells = [_clean(c) for c in row]
    cur_col = next((i for i, c in enumerate(cells) if any(h in c for h in _CURRENT_HDR)), None)
    if cur_col is None:
        return None
    new_col = None
    for i, c in enumerate(cells):
        if i == cur_col:
            continue
        if any(h in c for h in _NEW_HDR):
            new_col = i
    if new_col is None or new_col == cur_col:
        return None
    band_col = next((i for i, c in enumerate(cells) if any(h in c for h in _BAND_HDR)), None)
    if band_col is None:
        band_col = 0 if 0 not in (cur_col, new_col) else next(
            (i for i in range(len(cells)) if i not in (cur_col, new_col)), None)
    if band_col is None:
        return None
    return band_col, cur_col, new_col


def _extract_rate_tables(ref: FileRef) -> list[dict[str, Any]]:
    """PDF から「帯×現行率×新率」表を検出し、帯別 |新−現行| ＋ argmin/argmax を焼く。"""
    try:
        import pdfplumber  # lazy
    except Exception:  # noqa: BLE001
        return []
    tables: list[dict[str, Any]] = []
    try:
        with pdfplumber.open(str(ref.path)) as pdf:
            # ページ順に全表の行を平坦化（帯行がページ跨ぎでも拾えるよう、直近ヘッダを引き継ぐ）。
            cols: "tuple[int, int, int] | None" = None
            caption = ""
            bands: list[dict[str, Any]] = []
            page_of: list[int] = []

            def flush() -> None:
                nonlocal bands, cols, caption
                if cols and len(bands) >= 2:
                    tables.append(_finalize(bands, caption, ref))
                bands = []

            for pi, page in enumerate(pdf.pages, 1):
                for tbl in page.extract_tables() or []:
                    for row in tbl:
                        cells = [_clean(c) for c in row]
                        hc = _header_columns(cells)
                        if hc is not None:
                            flush()
                            cols = hc
                            caption = " ".join(c for c in cells if c)
                            continue
                        if cols is None:
                            continue
                        bcol, ccol, ncol = cols
                        band = _norm_band(cells[bcol]) if bcol < len(cells) else ""
                        cur = _parse_rate(cells[ccol]) if ccol < len(cells) else None
                        new = _parse_rate(cells[ncol]) if ncol < len(cells) else None
                        if band and _is_band_cell(band) and cur and new:
                            bands.append({"band": band, "current": cur, "new": new, "page": pi})
                            page_of.append(pi)
            flush()
    except Exception:  # noqa: BLE001
        return []
    return tables


def _finalize(bands: list[dict[str, Any]], caption: str, ref: FileRef) -> dict[str, Any]:
    """帯別 |新−現行| を計算し argmin/argmax を付す。範囲の現行は中点で代表。"""
    enriched: list[dict[str, Any]] = []
    for b in bands:
        cur_v = _rate_repr(b["current"])
        new_v = _rate_repr(b["new"])
        enriched.append({
            "band": b["band"], "current": b["current"], "new": b["new"],
            "current_repr": cur_v, "new_repr": new_v,
            "abs_increase": abs(new_v - cur_v), "increase": new_v - cur_v, "page": b["page"],
        })
    diffs = [e["abs_increase"] for e in enriched]
    argmin = min(range(len(enriched)), key=lambda i: diffs[i])
    argmax = max(range(len(enriched)), key=lambda i: diffs[i])
    # 一意性（同点でないこと）を記録しておく（serve 側の precision-first 判定に使う）。
    min_unique = sum(1 for d in diffs if abs(d - diffs[argmin]) < 1e-9) == 1
    max_unique = sum(1 for d in diffs if abs(d - diffs[argmax]) < 1e-9) == 1
    return {
        "caption": caption, "n_bands": len(enriched), "bands": enriched,
        "argmin_band": enriched[argmin]["band"], "argmin_unique": min_unique,
        "argmax_band": enriched[argmax]["band"], "argmax_unique": max_unique,
    }


# --------------------------------------------------------------------------- build / io
def _universe(refs: Sequence[FileRef]) -> list[FileRef]:
    out = [r for r in refs if r.ext == "pdf" and not nfc(r.name).startswith("~$")]
    out.sort(key=lambda r: r.rel)
    return out


def compute_doc(ref: FileRef) -> dict[str, Any] | None:
    tables = _extract_rate_tables(ref)
    if not tables:
        return None
    return {"doc_id": nfc(ref.rel), "project": nfc(ref.project), "doc_name": nfc(ref.name),
            "rate_tables": tables}


def build(refs: Sequence[FileRef] | None = None, *, out: Path | None = None,
          write_report: bool = True) -> dict[str, Any]:
    """全 PDF を走査し、税率表 doc-table ＋帯別差分派生を焼く（LLM 非使用・べき等・fail-open）。"""
    refs = list(refs) if refs is not None else list(walk())
    universe = _universe(refs)
    records: list[dict[str, Any]] = []
    for ref in universe:
        try:
            rec = compute_doc(ref)
        except Exception:  # noqa: BLE001
            rec = None
        if rec is not None:
            records.append(rec)
    stats = write_store(records, out)
    report = {
        "schema": SCHEMA, "version": SCHEMA_VERSION,
        "universe": len(universe), "records": len(records),
        "tables": sum(len(r.get("rate_tables", [])) for r in records),
    }
    if write_report:
        rp = default_report_path()
        rp.parent.mkdir(parents=True, exist_ok=True)
        rp.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return {"records": len(records), "docs": stats["docs"], "report": report}


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


_LOAD_CACHE: dict[str, list[dict[str, Any]]] = {}


def reset_cache() -> None:
    _LOAD_CACHE.clear()


def load(path: Path | None = None) -> list[dict[str, Any]]:
    """Load the rate-table records (memoized). ``[]`` when absent/unreadable/schema-mismatch (回帰ゼロ)."""
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


def docs_for(hint: str, *, path: Path | None = None) -> list[dict[str, Any]]:
    """project / ファイル名の部分一致でレコードを引く（空白/ケース無視）。"""
    h = norm(hint)
    rows = load(path)
    if not h:
        return list(rows)
    return [r for r in rows if h in norm(r.get("doc_id")) or h in norm(r.get("project"))
            or h in norm(r.get("doc_name"))]


if __name__ == "__main__":
    summary = build()
    print(f"[build] rate_table_store records={summary['records']} "
          f"tables={summary['report']['tables']} -> {default_out_path()}")
