"""SOT-2699 — 統計表 rank/ratio 事前計算ストア（cycle9）.

cycle8 abstain の idx99（派生計算: 「死亡率が最も高い都道府県の死亡率は、4番目に低い都道府県の死亡率の
何倍か」）を **質問非依存** の事前計算で回収する。根本原因は「表の生値は :mod:`doc_reach_store` に抽出済み
だが、rank-k・ペア比（序数 → 値 → 何倍/差）へ写す派生計算とレーンが無い」ことなので、本ストアは新しい
抽出を足さず、既存 :mod:`src.rag.index.doc_reach_store` の全 docx/pptx/pdf テーブルを走査して、

* **数値列を header 名でグルーピングした系列**（同一 header の複数列＝高い側/低い側の分割ランキング表を
  1 系列へ統合。例: みなみ野 糖尿病統計の「死亡率（%）」列 = ワースト側 col + ベスト側 col の 10 値）
* **昇順/降順ソート済みの (label, value) 系列**（rank1..N の値を O(1) で引ける）

を **決定論・LLM 非使用（cost $0）・二重検算（sorted を entries から独立再導出して一致検証）・fail-closed**
で焼く。serve レーン（:mod:`src.rag.agent.derived_ranking_lane`）が序数（最も高い/低い・N番目に高い/低い）
をパースして store lookup し、何倍（比）/差 を丸め指定つきで直答する。

``RAG_DERIVED_RANKING`` 既定 OFF ⇒ serve レーン／ツールは None（tool 集合/スキーマ/serve path は
byte-identical）。build 自体は常に実行可能（べき等）。ストアの load は flag 非依存（読むだけ）。
"""
from __future__ import annotations

import json
import os
import re
import unicodedata
from pathlib import Path
from typing import Any

from config import settings

from src.rag.index import doc_reach_store

SCHEMA = "derived_ranking_store"
SCHEMA_VERSION = 1
_ON = {"1", "true", "yes", "on"}

_LOAD_CACHE: dict[str, dict[str, Any]] = {}

# 系列として採用する最小の数値件数（rank/比を意味づけられる最小）。
MIN_SERIES_VALUES = 2
# ある列を「数値列」とみなすのに必要な数値セルの下限割合（散文中の 1 個のノイズ数値を系列化しない）。
MIN_NUMERIC_FRACTION = 0.5
# 病的な表が系列を氾濫させないための上限。
MAX_SERIES_PER_PROJECT = 400
MAX_VALUES_PER_SERIES = 200

# 数値セル: 先頭に符号可・カンマ桁区切り可・末尾に % 可、それ以外の文字を含むものは不採用（億/万/円/人 等）。
_NUMERIC_RE = re.compile(r"^[+\-]?\d{1,3}(?:,\d{3})*(?:\.\d+)?%?$|^[+\-]?\d+(?:\.\d+)?%?$")
# header からの単位・括弧注記の除去（metric core 抽出用）。全角/半角括弧。
_PAREN_RE = re.compile(r"[（(【\[].*?[）)】\]]")


def enabled() -> bool:
    """serve レーン／ツールを有効にするか。既定 OFF（``RAG_DERIVED_RANKING``）⇒ byte-identical。"""
    return os.getenv("RAG_DERIVED_RANKING", "0").strip().lower() in _ON


def default_out_path() -> Path:
    return settings.ARTIFACTS_DIR / "derived_ranking_store.jsonl"


def default_report_path() -> Path:
    return settings.ARTIFACTS_DIR / "derived_ranking_store_build_report.json"


# --------------------------------------------------------------------------- helpers
def _norm(text: Any) -> str:
    """NFKC 正規化 + 空白除去 + lower（照合キー）。"""
    return unicodedata.normalize("NFKC", str(text or "")).replace(" ", "").replace("　", "").lower()


def metric_core(header: Any) -> str:
    """header から括弧注記/単位を落とした metric コア（質問の metric キーワード照合用）。

    例: ``死亡率（%）`` → ``死亡率`` / ``平均年収 (万円)`` → ``平均年収``。
    """
    h = unicodedata.normalize("NFKC", str(header or ""))
    h = _PAREN_RE.sub("", h)
    return h.replace(" ", "").replace("　", "").strip().lower()


def parse_numeric(cell: Any) -> float | None:
    """セル文字列を数値へ（カンマ除去・末尾 % 許容）。純粋な数値でなければ None（fail-closed）。

    ``18.2`` → 18.2 / ``7.3`` → 7.3 / ``10.5%`` → 10.5 / ``5億3,700万人`` → None（単位語を含む）。
    """
    s = unicodedata.normalize("NFKC", str(cell or "")).strip()
    if not s:
        return None
    if not _NUMERIC_RE.match(s):
        return None
    s2 = s.rstrip("%").replace(",", "")
    try:
        return float(s2)
    except ValueError:
        return None


def _has_percent(cell: Any) -> bool:
    return unicodedata.normalize("NFKC", str(cell or "")).strip().endswith("%")


def _nearest_label(row: list[str], j: int) -> str:
    """行 ``row`` の列 ``j`` の値に紐づく label = j より左で最も近い非数値・非空セル（無ければ ""）。"""
    for k in range(j - 1, -1, -1):
        cell = row[k]
        if cell and cell.strip() and parse_numeric(cell) is None:
            return cell.strip()
    return ""


def _table_series(table: dict[str, Any]) -> list[dict[str, Any]]:
    """1 テーブルから header 名でグルーピングした数値系列を作る（決定論・二重検算・fail-closed）。"""
    rows = table.get("rows") or []
    if len(rows) < 2:
        return []
    header = rows[0]
    body = rows[1:]
    ncols = max((len(r) for r in rows), default=0)

    # 列ごとに (数値件数, 非空件数, 値+label) を集計し、数値列だけ header-core でまとめる。
    groups: dict[str, dict[str, Any]] = {}
    for j in range(ncols):
        col_header = header[j] if j < len(header) else ""
        core = metric_core(col_header)
        values: list[dict[str, Any]] = []
        nonempty = 0
        unit_percent = False
        for r in body:
            if j >= len(r):
                continue
            cell = r[j]
            if cell and cell.strip():
                nonempty += 1
            v = parse_numeric(cell)
            if v is None:
                continue
            if _has_percent(cell):
                unit_percent = True
            values.append({"label": _nearest_label(r, j), "value": v})
        # 数値列判定: 非空セルの過半が数値、かつ最低件数を満たす。
        if nonempty == 0 or len(values) < MIN_SERIES_VALUES:
            continue
        if (len(values) / nonempty) < MIN_NUMERIC_FRACTION:
            continue
        # header-core が空（見出し無し表）の列は他列と混ぜず列固有キーで独立系列にする。
        key = core if core else f"__col{j}__"
        g = groups.setdefault(key, {"metric_key": core, "headers": [], "entries": [],
                                    "unit_percent": False, "columns": []})
        g["headers"].append(unicodedata.normalize("NFKC", str(col_header or "")).strip())
        g["columns"].append(j)
        g["entries"].extend(values)
        g["unit_percent"] = g["unit_percent"] or unit_percent

    series: list[dict[str, Any]] = []
    for key, g in groups.items():
        entries = g["entries"][:MAX_VALUES_PER_SERIES]
        if len(entries) < MIN_SERIES_VALUES:
            continue
        # 二重検算: sorted を entries から独立に再導出し、値列が一致することを確認（fail-closed）。
        asc = sorted(entries, key=lambda e: e["value"])
        desc = sorted(entries, key=lambda e: e["value"], reverse=True)
        asc_vals = [e["value"] for e in asc]
        if asc_vals != sorted(e["value"] for e in entries):
            continue  # ソート不整合 ⇒ この系列は焼かない
        if [e["value"] for e in desc] != asc_vals[::-1]:
            continue
        series.append({
            "metric_key": g["metric_key"],
            "header": g["headers"][0] if g["headers"] else "",
            "headers": g["headers"],
            "columns": g["columns"],
            "unit": "%" if g["unit_percent"] else None,
            "n": len(entries),
            "entries": entries,
            "sorted_asc": asc,
            "sorted_desc": desc,
        })
    return series


# --------------------------------------------------------------------------- build
def build(*, out: Path | None = None, write_report: bool = True) -> dict[str, Any]:
    """doc_reach_store の全テーブルから rank/ratio 系列を構築し JSONL へ書き出す（べき等・LLM 非使用）。"""
    data = doc_reach_store.load()
    docs = data.get("docs") or []
    by_project: dict[str, list[dict[str, Any]]] = {}
    for rec in docs:
        project = rec.get("project") or ""
        for table in rec.get("tables") or []:
            for s in _table_series(table):
                if len(by_project.get(project, [])) >= MAX_SERIES_PER_PROJECT:
                    break
                s = {"rel": rec.get("rel"), "name": rec.get("name"), "table_id": table.get("table_id"),
                     "locus": table.get("locus"), "caption": table.get("caption"), **s}
                by_project.setdefault(project, []).append(s)

    records = [{"project": p, "series": s} for p, s in sorted(by_project.items()) if s]

    out_path = out or default_out_path()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as fh:
        fh.write(json.dumps({"schema": SCHEMA, "version": SCHEMA_VERSION,
                             "n_projects": len(records)}, ensure_ascii=False) + "\n")
        for rec in records:
            fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
    _LOAD_CACHE.pop(str(out_path), None)

    report = {
        "projects": len(records),
        "total_series": sum(len(r["series"]) for r in records),
        "metric_keys": sorted({s["metric_key"] for r in records for s in r["series"] if s["metric_key"]}),
    }
    if write_report:
        with open(default_report_path(), "w", encoding="utf-8") as fh:
            json.dump(report, fh, ensure_ascii=False, indent=2)
    return {"projects": len(records), "report": report}


def load(path: Path | None = None) -> dict[str, Any]:
    """ストアを読み込む（memoized）。欠損/スキーマ不一致 ⇒ ``{"by_project": {}}``（回帰ゼロ）。"""
    out = path or default_out_path()
    key = str(out)
    if key in _LOAD_CACHE:
        return _LOAD_CACHE[key]
    by_project: dict[str, list[dict[str, Any]]] = {}
    try:
        with open(out, encoding="utf-8") as fh:
            header = fh.readline()
            meta = json.loads(header) if header.strip() else {}
            if isinstance(meta, dict) and meta.get("schema") == SCHEMA \
                    and meta.get("version") == SCHEMA_VERSION:
                for line in fh:
                    line = line.strip()
                    if not line:
                        continue
                    rec = json.loads(line)
                    by_project[rec.get("project") or ""] = rec.get("series") or []
    except Exception:  # noqa: BLE001
        by_project = {}
    data = {"by_project": by_project}
    _LOAD_CACHE[key] = data
    return data


if __name__ == "__main__":
    summary = build()
    print(json.dumps(summary["report"], ensure_ascii=False, indent=2))
