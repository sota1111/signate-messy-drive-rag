"""Notebook chart-image drawing-attribute store (SOT-2685 / cycle7 K2 — ipynb 描画チャートの build 時読み取り).

Sonnet gold100 の abstain のうち idx56/idx66 は、証拠が **ノートブック(01_eda.ipynb)が描いたチャート画像の
描画属性** にあり、テキスト/データからは確定できないのが主因だった（`docs/ai/sonnet_cycle_analysis/cycle7.md`
§1 K2）:

* **idx56**（ひがし丘 目的変数分析）— y軸に *実際に印字されている* 目盛りの最大値（gold 1200）。これは
  matplotlib の描画属性で、データ度数（charges の最頻値件数 1256）からは確定できない（軸目盛りは 1200 で
  丸められて印字される）。read_chart_values 相当の ipynb 画像読みが存在しなかった。
* **idx66**（京橋 日付分析）— チャートで件数が最も高い日（gold 20日）。これはデータ側（day 列の日別件数
  argmax）から独立検算できるが、チャート自体を読む手段がなかった。

本モジュールは build 時に一度だけ、全案件の 01_eda 系 notebook の **チャート画像を質問非依存で網羅** し、
vision（build 限定で Gemini 使用可）で「軸ラベル・y軸目盛り値（と最大目盛り）・系列カテゴリ→値・最頻
カテゴリ」を構造化して ``artifacts/notebook_chart_store.jsonl`` に焼く。**可能な限りデータ側から独立検算**
（日付分析＝day 列の日別件数 argmax / 目的変数分析＝目的変数の value_counts argmax）し、vision の最頻カテゴリと
一致すれば ``verified: true`` を付す（不一致は ``verified: false``）。serve path は事前計算済みを読むだけで
**genai 呼び出しゼロ**（``RAG_FORBID_GEMINI=1`` でも動く）。読み出し/配線は :mod:`src.rag.agent.nb_chart_lane`。

チャート画像の所在は notebook から決定論的に導く（gold 参照なし）:
* 各 code セルの ``savefig(FIG_DIR / '<name>.png')`` と直前の ``## N. <見出し>`` を対応付け、その ``<name>.png``
  を同じ ``analysis_project/reports/figures/`` の実ファイル(FileRef, NFD 耐性)へ解決する。
* on-disk figure が無いセクションは、セルの埋め込み画像出力(base64 PNG)を代替ソースとして使う。

Design invariants（sibling の ocr_store / visual_store と同一）:
* **Opt-in at serve time.** :func:`enabled` (``RAG_NB_CHART_STORE``) が runtime 参照のみ gate。default OFF ⇒
  champion serve path は byte-identical。
* **Opt-in at build time too.** vision コールを spend するので build も ``RAG_NB_CHART_STORE_BUILD`` (default
  OFF) で gate し、prior 記録(pixel sha)を再利用して再課金しない。
* **Question-independent / No hardcoding.** universe は notebook 構造から導く。gold 値・idx は一切持たない。
* **Fail-open.** artifact 欠落・解析不能・vision 失敗はすべて空/スキップへフォールバック（回帰ゼロ）。
"""
from __future__ import annotations

import base64
import hashlib
import json
import os
import re
import statistics
import unicodedata
from collections import Counter
from pathlib import Path
from typing import Any, Sequence

from config import settings
from src.rag.corpus import FileRef, nfc, walk

SCHEMA = "nb-chart-store"
SCHEMA_VERSION = 1

_ON = {"1", "true", "yes", "on"}

# `## 5. 目的変数分析` のような番号付き見出しからセクション(番号, タイトル)を取る（区切りは半角/全角ピリオド）。
_HEAD = re.compile(r"^#{1,4}\s*(\d+)\s*[.．]\s*(.+?)\s*$", re.M)
# savefig(FIG_DIR / 'target_distribution.png', ...) の保存ファイル名。
_SAVEFIG = re.compile(r"savefig\(\s*(?:FIG_DIR\s*/\s*)?[\"']([^\"']+\.png)[\"']", re.I)


def enabled() -> bool:
    """True when the serve path may consult the store (default OFF — opt-in)."""
    return os.getenv("RAG_NB_CHART_STORE", "0").strip().lower() in _ON


def build_enabled() -> bool:
    """True when the index build may spend Gemini vision calls to (re)build the store (default OFF)."""
    return os.getenv("RAG_NB_CHART_STORE_BUILD", "0").strip().lower() in _ON


def _samples() -> int:
    """How many vision reads to take per image; the committed discrete fields use their mode."""
    try:
        return max(1, int(os.getenv("NB_CHART_STORE_SAMPLES", "3")))
    except ValueError:
        return 3


def default_out_path() -> Path:
    return settings.ARTIFACTS_DIR / "notebook_chart_store.jsonl"


def default_report_path() -> Path:
    return settings.ARTIFACTS_DIR / "notebook_chart_store_build_report.json"


# --------------------------------------------------------------------------- 案件識別キー（visual_store と同一規約）
_CORP = r"(?:株式会社|医療法人社団|一般社団法人|一般財団法人|有限会社|合同会社|合資会社)"
_CORP_PREFIX = re.compile(rf"^{_CORP}\s*")
_CORP_SUFFIX = re.compile(rf"\s*{_CORP}$")


def owner_key(value: Any) -> str:
    s = unicodedata.normalize("NFKC", str(value or ""))
    s = _CORP_PREFIX.sub("", s)
    s = _CORP_SUFFIX.sub("", s)
    return re.sub(r"[\s　]", "", s).lower()


# --------------------------------------------------------------------------- notebook parsing
def _notebook_sections(nb: dict[str, Any]) -> list[dict[str, Any]]:
    """Each ``## N. 見出し`` section that produces a chart, with its savefig names + embedded PNG bytes."""
    sections: list[dict[str, Any]] = []
    cur: tuple[str, str] | None = None
    for cell in nb.get("cells", []):
        src = "".join(cell.get("source", []) or [])
        if cell.get("cell_type") == "markdown":
            m = _HEAD.search(src)
            if m:
                cur = (m.group(1), nfc(m.group(2).strip()))
            continue
        if cell.get("cell_type") != "code":
            continue
        savefigs = [nfc(x) for x in _SAVEFIG.findall(src)]
        embedded = _embedded_pngs(cell)
        if not savefigs and not embedded:
            continue
        number, title = cur if cur else ("", "")
        sections.append({"number": number, "title": title,
                         "savefigs": savefigs, "embedded": embedded})
    return sections


def _embedded_pngs(cell: dict[str, Any]) -> list[bytes]:
    """Decoded PNG bytes of a code cell's image outputs (base64), in output order."""
    out: list[bytes] = []
    for o in cell.get("outputs", []) or []:
        data = o.get("data", {}) or {}
        for mt, val in data.items():
            if not mt.startswith("image/png"):
                continue
            raw = "".join(val) if isinstance(val, list) else val
            try:
                out.append(base64.b64decode(raw))
            except Exception:  # noqa: BLE001 — a corrupt output must not sink the notebook
                continue
    return out


def _figure_index(refs: Sequence[FileRef]) -> dict[tuple[str, str], FileRef]:
    """``(owner_key(project), filename) → FileRef`` for every reports/figures PNG (NFD-safe on-disk names)."""
    idx: dict[tuple[str, str], FileRef] = {}
    for r in refs:
        if r.ext != "png" or "reports/figures" not in nfc(r.rel):
            continue
        idx[(owner_key(r.project), nfc(r.name))] = r
    return idx


# --------------------------------------------------------------------------- vision read
def _vision_prompt() -> str:
    return (
        "あなたはmatplotlibで描画されたグラフ画像を精密に読むアシスタントです。次のJSONだけを出力してください:\n"
        '{"chart_type":"", "title":"", "x_label":"", "y_label":"", '
        '"y_axis_tick_labels":[数値を小さい順に全て], "y_axis_max_tick":数値, '
        '"x_categories":[], "peak_category":"", "peak_value":数値}\n'
        "y_axis_tick_labels は左(主)y軸に実際に印字されている目盛りラベルの数値を全て。y_axis_max_tick はその最大値。"
        "peak_category は棒/線で最も高い x のカテゴリ(ラベル文字列)、peak_value はその値。"
        "左右2軸ある場合は主軸(左)を使う。数値は単位を除いた数のみ。読めない項目は null。"
    )


def _parse_vision_json(text: str) -> dict[str, Any] | None:
    """Extract the JSON object from a vision reply (```json fences / surrounding prose tolerated)."""
    if not text:
        return None
    s = text.strip()
    if s.startswith("```"):
        s = re.sub(r"^```[a-zA-Z]*\s*", "", s)
        s = re.sub(r"\s*```$", "", s).strip()
    start, end = s.find("{"), s.rfind("}")
    if start < 0 or end <= start:
        return None
    try:
        obj = json.loads(s[start:end + 1])
    except Exception:  # noqa: BLE001
        return None
    return obj if isinstance(obj, dict) else None


def _num(v: Any) -> float | None:
    try:
        return float(str(v).replace(",", "").strip())
    except (TypeError, ValueError):
        return None


def _read_image(image: bytes, samples: int) -> dict[str, Any] | None:
    """Vision-read one chart image ``samples`` times → consensus of the discrete committed fields.

    The committed fields (``y_axis_max_tick`` — a printed drawing attribute — and ``peak_category``)
    take the mode across samples; ``unanimous_*`` marks agreement of all reads (the serve lane commits
    a vision-only value only when unanimous). Approximate fields (peak_value / tick list) are metadata.
    """
    from src.rag import llm

    reads: list[dict[str, Any]] = []
    for _ in range(max(1, samples)):
        try:
            txt = llm.generate(_vision_prompt(), images=[llm.Image(data=image, mime_type="image/png")],
                               model=llm.settings.VISION_MODEL, max_output_tokens=1024, temperature=0.0)
        except Exception:  # noqa: BLE001 — one failed vision call must not sink the record
            continue
        obj = _parse_vision_json(txt)
        if obj is not None:
            reads.append(obj)
    if not reads:
        return None

    max_ticks = [int(v) for v in (_num(r.get("y_axis_max_tick")) for r in reads) if v is not None]
    peaks = [nfc(str(r.get("peak_category"))) for r in reads
             if r.get("peak_category") not in (None, "", "null")]
    y_max = _mode(max_ticks)
    peak = _mode(peaks)
    peak_vals = [_num(r.get("peak_value")) for r in reads]
    peak_vals = [v for v in peak_vals if v is not None]
    tick_list = next((r.get("y_axis_tick_labels") for r in reads
                      if _num(r.get("y_axis_max_tick")) == y_max), reads[0].get("y_axis_tick_labels"))
    first = reads[0]
    return {
        "chart_type": nfc(str(first.get("chart_type") or "")) or None,
        "title": nfc(str(first.get("title") or "")) or None,
        "x_label": nfc(str(first.get("x_label") or "")) or None,
        "y_label": nfc(str(first.get("y_label") or "")) or None,
        "y_axis_tick_labels": tick_list,
        "y_axis_max_tick": y_max,
        "peak_category": peak,
        "peak_value": (round(statistics.median(peak_vals), 4) if peak_vals else None),
        "samples": len(reads),
        "unanimous_y_max": (y_max is not None and all(t == y_max for t in max_ticks) and len(max_ticks) == len(reads)),
        "unanimous_peak": (peak is not None and all(p == peak for p in peaks) and len(peaks) == len(reads)),
    }


def _mode(items: list[Any]) -> Any:
    return Counter(items).most_common(1)[0][0] if items else None


# --------------------------------------------------------------------------- data-side independent check
def _notebook_train(nb_ref: FileRef, refs: Sequence[FileRef]) -> FileRef | None:
    """The ``analysis_project/data/train.csv`` beside a notebook (same project), else None."""
    for r in refs:
        if r.ext == "csv" and r.name == "train.csv" \
                and owner_key(r.project) == owner_key(nb_ref.project) \
                and "analysis_project/data" in nfc(r.rel):
            return r
    return None


def _data_check(nb_ref: FileRef, title: str, refs: Sequence[FileRef], peak_category: Any) -> dict[str, Any] | None:
    """Independently recompute the section's peak category from the training table (fail-closed → None).

    * 日付分析 — the day-of-month column's per-day record count argmax (the chart plots groupby(day).size).
    * 目的変数分析 — the target column's value_counts argmax (the chart plots the target's category counts).
    """
    tref = _notebook_train(nb_ref, refs)
    if tref is None:
        return None
    try:
        import pandas as pd
        df = pd.read_csv(tref.path)
    except Exception:  # noqa: BLE001
        return None
    try:
        if "日付" in title or "date" in title.lower():
            col = _pure_day_column(df)
            if col is None:
                return None
            counts = df.groupby(col).size().sort_values(ascending=False)
            top = counts.index[0]
            cat = str(int(top)) if float(top).is_integer() else str(top)
            return {"kind": "day_count_argmax", "column": str(col), "category": cat,
                    "count": int(counts.iloc[0]),
                    "agrees": _cat_eq(cat, peak_category)}
        if "目的変数" in title or "target" in title.lower():
            col = df.columns[-1]
            vc = df[col].value_counts()
            top = vc.index[0]
            cat = str(int(top)) if isinstance(top, float) and float(top).is_integer() else str(top)
            return {"kind": "target_count_argmax", "column": str(col), "category": cat,
                    "count": int(vc.iloc[0]), "agrees": _cat_eq(cat, peak_category)}
    except Exception:  # noqa: BLE001
        return None
    return None


def _pure_day_column(df: Any) -> Any | None:
    """A column whose numeric values all lie in 1..31 (a day-of-month column), preferring one named 'day'."""
    import pandas as pd

    def is_day(series: Any) -> bool:
        s = pd.to_numeric(series, errors="coerce").dropna()
        return len(s) > 0 and bool(s.between(1, 31).all())

    if "day" in df.columns and is_day(df["day"]):
        return "day"
    for col in df.columns:
        if pd.api.types.is_numeric_dtype(df[col]) and is_day(df[col]):
            return col
    return None


def _cat_eq(a: Any, b: Any) -> bool:
    """Category equality tolerant of int/float/str formatting (``'20'`` == ``'20.0'`` == ``20``)."""
    if a is None or b is None:
        return False
    na, nb = _num(a), _num(b)
    if na is not None and nb is not None:
        return na == nb
    return nfc(str(a)).strip() == nfc(str(b)).strip()


# --------------------------------------------------------------------------- per-notebook build
def _chart_source(section: dict[str, Any], project: str,
                  fig_index: dict[tuple[str, str], FileRef]) -> tuple[bytes, str, str] | None:
    """The chart image bytes for a section: on-disk figure preferred, else the first embedded PNG."""
    key = owner_key(project)
    for name in section.get("savefigs", []):
        ref = fig_index.get((key, nfc(name)))
        if ref is not None:
            try:
                return ref.path.read_bytes(), "figure_file", nfc(name)
            except OSError:
                continue
    embedded = section.get("embedded") or []
    if embedded:
        return embedded[0], "embedded", ""
    return None


def compute_notebook(nb_ref: FileRef, refs: Sequence[FileRef],
                     fig_index: dict[tuple[str, str], FileRef], *,
                     samples: int, prior: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    """One notebook's chart records (one per section that produced a readable chart image)."""
    try:
        nb = json.loads(nb_ref.path.read_text(encoding="utf-8"))
    except Exception:  # noqa: BLE001
        return []
    out: list[dict[str, Any]] = []
    doc_id = nfc(nb_ref.rel)
    for sec in _notebook_sections(nb):
        src = _chart_source(sec, nb_ref.project, fig_index)
        if src is None:
            continue
        image, source_kind, figure_name = src
        sha = hashlib.sha256(image).hexdigest()
        rec_key = f"{doc_id}#{sec['number']}#{sha}"
        old = prior.get(rec_key)
        if old is not None and old.get("samples"):
            out.append(old)
            continue
        vision = _read_image(image, samples)
        if vision is None:
            continue
        data_check = _data_check(nb_ref, sec["title"], refs, vision.get("peak_category"))
        from src.rag import llm
        rec = {
            "record_key": rec_key,
            "doc_id": doc_id,
            "project": nfc(nb_ref.project),
            "notebook": nfc(nb_ref.name),
            "section_number": sec["number"],
            "section_title": sec["title"],
            "figure": figure_name or None,
            "source": source_kind,
            "pixel_sha256": sha,
            "vision_model": llm.settings.VISION_MODEL,
            "data_check": data_check,
            "verified": bool(data_check and data_check.get("agrees")),
            **{k: v for k, v in vision.items()},
        }
        out.append(rec)
    return out


# --------------------------------------------------------------------------- build / io
def _universe(refs: Sequence[FileRef]) -> list[FileRef]:
    out = [r for r in refs if r.ext == "ipynb"]
    out.sort(key=lambda r: r.rel)
    return out


def write_store(records: Sequence[dict[str, Any]], path: Path | None = None) -> dict[str, int]:
    """Atomically write a reproducible JSONL store (schema header + (doc_id, section) sorted rows)."""
    out = Path(path) if path is not None else default_out_path()
    out.parent.mkdir(parents=True, exist_ok=True)
    ordered = sorted(records, key=lambda r: (r.get("doc_id", ""), str(r.get("section_number", ""))))
    tmp = out.with_suffix(out.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8", newline="\n") as handle:
        handle.write(json.dumps({"schema": SCHEMA, "version": SCHEMA_VERSION},
                                ensure_ascii=False, sort_keys=True) + "\n")
        for rec in ordered:
            handle.write(json.dumps(rec, ensure_ascii=False, sort_keys=True) + "\n")
    tmp.replace(out)
    reset_cache()
    return {"records": len(ordered)}


def build(refs: Sequence[FileRef] | None = None, *, out: Path | None = None,
          samples: int | None = None, write_report: bool = True) -> dict[str, Any]:
    """Read every notebook's chart images with vision once and bake the store (question-independent).

    Incremental: an existing record whose ``record_key`` (doc_id#section#pixel-sha) still matches is
    reused verbatim — a rebuild never re-spends vision on an unchanged image.
    """
    refs = list(refs) if refs is not None else list(walk())
    n_samples = samples if samples is not None else _samples()
    fig_index = _figure_index(refs)
    prior = _load_prior(out)

    records: list[dict[str, Any]] = []
    for nb_ref in _universe(refs):
        try:
            records.extend(compute_notebook(nb_ref, refs, fig_index, samples=n_samples, prior=prior))
        except Exception:  # noqa: BLE001 — one bad notebook must not sink the build
            continue
    stats = write_store(records, out)
    report = {
        "schema": SCHEMA, "version": SCHEMA_VERSION,
        "notebooks": len(_universe(refs)), "records": len(records),
        "verified": sum(1 for r in records if r.get("verified")),
        "with_data_check": sum(1 for r in records if r.get("data_check")),
        "figure_sourced": sum(1 for r in records if r.get("source") == "figure_file"),
        "embedded_sourced": sum(1 for r in records if r.get("source") == "embedded"),
    }
    if write_report:
        rp = default_report_path()
        rp.parent.mkdir(parents=True, exist_ok=True)
        rp.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return {"records": stats["records"], "report": report}


def _load_prior(out: Path | None) -> dict[str, dict[str, Any]]:
    prior: dict[str, dict[str, Any]] = {}
    for rec in load(out):
        key = rec.get("record_key")
        if key:
            prior[key] = rec
    return prior


# --------------------------------------------------------------------------- load / read
_LOAD_CACHE: dict[str, list[dict[str, Any]]] = {}


def reset_cache() -> None:
    _LOAD_CACHE.clear()


def load(path: Path | None = None) -> list[dict[str, Any]]:
    """Load the chart records (memoized). ``[]`` when absent/unreadable/schema-mismatch (回帰ゼロ)."""
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


def docs_for_project(project_hint: str, *, path: Path | None = None) -> list[dict[str, Any]]:
    """Chart records whose project matches ``project_hint`` (corporate-form / spacing insensitive)."""
    hint = owner_key(project_hint)
    rows = load(path)
    if not hint:
        return list(rows)
    return [r for r in rows if hint and (hint in owner_key(r.get("project", ""))
                                         or owner_key(r.get("project", "")) in hint)]


if __name__ == "__main__":
    summary = build()
    print(json.dumps(summary["report"], ensure_ascii=False, indent=2))
