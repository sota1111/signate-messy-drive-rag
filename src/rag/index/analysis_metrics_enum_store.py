"""SOT-2698 — 分析出力 metrics.json の enum フィールド網羅ストア（cycle9, idx32）.

cycle9 abstain の idx32（青嶺不動産 metrics.json ``feature_selection.selected_columns`` の数値交互作用
特徴量列名）は、証拠が ``04.分析/analysis_outputs/metrics.json`` に **そのまま在る** のに到達レーンが無く
（既存 :mod:`src.rag.index.analysis_xref_store` は最終報告テキストのみ抽出で metrics.json 非対応）、cycle7
MATCH → cycle8 BUDGET_EXHAUSTED のチャーンで棄権していた。

本ストアは全案件の metrics.json を **質問非依存** に走査し、per-case で enum（リスト値）フィールドを網羅
抽出する（純粋な決定論・JSON パースのみ・LLM/genai 非使用 ⇒ cost $0）:

* **selected_columns** — ``feature_selection.selected_columns`` の全列（順序保存）。
* **interaction_columns** — そのうち **分析コードが生成した数値交互作用特徴量**（列名に ``__x__`` 区切りを
  含む部分集合。青嶺 = ``BOROUGH__x__BLOCK`` … ``LOT__x__ZIP CODE`` の 6 列）。
* **excluded_columns** — ``feature_selection.excluded_columns`` の列名（除外理由つきの dict は名前へ）。
* **ordered_feature_columns** / **enum_fields** — metrics.json 直下のリスト値フィールド（順序保存）を汎用に
  焼く（質問を見ない網羅抽出。将来の enum 問いにも決定論到達できる）。

規律: ``feature_selection.selected_columns`` を持つ metrics.json を案件ごとに一意選択（``analysis_outputs``
配下優先、無ければ最短 rel）。``RAG_ANALYSIS_METRICS_ENUM`` 既定 OFF ⇒ serve レーンは None（tool 集合/
スキーマ/serve path は byte-identical）。build 自体は常に実行可能（べき等）。
"""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Sequence

from config import settings

from src.rag import corpus
from src.rag.corpus import FileRef, nfc

SCHEMA = "analysis_metrics_enum_store"
SCHEMA_VERSION = 1
_ON = {"1", "true", "yes", "on"}

# 分析コードが生成する数値交互作用特徴量の列名区切り（``BOROUGH__x__BLOCK``）。
INTERACTION_MARKER = "__x__"

_LOAD_CACHE: dict[str, list[dict[str, Any]]] = {}


def enabled() -> bool:
    """serve レーンを有効にするか。既定 OFF（``RAG_ANALYSIS_METRICS_ENUM``）⇒ byte-identical。"""
    return os.getenv("RAG_ANALYSIS_METRICS_ENUM", "0").strip().lower() in _ON


def default_out_path() -> Path:
    return settings.ARTIFACTS_DIR / "analysis_metrics_enum_store.jsonl"


def default_report_path() -> Path:
    return settings.ARTIFACTS_DIR / "analysis_metrics_enum_store_build_report.json"


# --------------------------------------------------------------------------- helpers
def _load_json(ref: FileRef) -> Any | None:
    try:
        return json.loads(ref.path.read_text(errors="replace"))
    except Exception:  # noqa: BLE001 — 壊れた JSON は無視（回帰ゼロ）
        return None


def _metrics_refs(project: str, refs: Sequence[FileRef]) -> list[FileRef]:
    """案件の metrics.json 候補（``analysis_outputs`` 配下優先、最短 rel 安定選択）。"""
    cands = [r for r in refs
             if r.project == project and r.ext == "json" and r.name.lower() == "metrics.json"
             and not nfc(r.name).startswith("~$")]
    cands.sort(key=lambda r: (0 if "analysis_outputs" in nfc(r.rel) else 1, len(r.rel)))
    return cands


def _column_names(value: Any) -> list[str]:
    """リスト値フィールドを列名の list へ正規化（str はそのまま、dict は ``column``/``name`` キーを拾う）。"""
    out: list[str] = []
    if not isinstance(value, list):
        return out
    for item in value:
        if isinstance(item, str):
            out.append(item)
        elif isinstance(item, dict):
            name = item.get("column") or item.get("name") or item.get("col")
            if isinstance(name, str):
                out.append(name)
    return out


def _enum_fields(obj: dict[str, Any]) -> dict[str, list[str]]:
    """metrics.json 直下の「文字列 list」フィールドを汎用に焼く（順序保存・質問非依存の網羅抽出）。"""
    out: dict[str, list[str]] = {}
    for key, value in obj.items():
        if isinstance(value, list) and value and all(isinstance(x, str) for x in value):
            out[key] = list(value)
    return out


# --------------------------------------------------------------------------- per-case
def _build_case(project: str, refs: Sequence[FileRef]) -> dict[str, Any] | None:
    """metrics.json から enum フィールドを網羅抽出（``feature_selection.selected_columns`` を持つものを選択）。"""
    chosen: FileRef | None = None
    metrics: dict[str, Any] | None = None
    fallback: tuple[FileRef, dict[str, Any]] | None = None
    for ref in _metrics_refs(project, refs):
        data = _load_json(ref)
        if not isinstance(data, dict):
            continue
        if fallback is None:
            fallback = (ref, data)
        fs = data.get("feature_selection")
        if isinstance(fs, dict) and isinstance(fs.get("selected_columns"), list):
            chosen, metrics = ref, data
            break
    if metrics is None and fallback is not None:
        chosen, metrics = fallback
    if metrics is None or chosen is None:
        return None

    rec: dict[str, Any] = {"project": project, "metrics_rel": chosen.rel}
    fs = metrics.get("feature_selection")
    if isinstance(fs, dict):
        selected = _column_names(fs.get("selected_columns"))
        if selected:
            rec["selected_columns"] = selected
            rec["interaction_columns"] = [c for c in selected if INTERACTION_MARKER in c]
        excluded = _column_names(fs.get("excluded_columns"))
        if excluded:
            rec["excluded_columns"] = excluded
    ordered = _column_names(metrics.get("ordered_feature_columns"))
    if ordered:
        rec["ordered_feature_columns"] = ordered
    enum_fields = _enum_fields(metrics)
    if enum_fields:
        rec["enum_fields"] = enum_fields
    # project + metrics_rel 以外に何か焼けた時だけ残す。
    return rec if len(rec) > 2 else None


# --------------------------------------------------------------------------- build
def build(refs: Sequence[FileRef] | None = None, *, out: Path | None = None,
          write_report: bool = True) -> dict[str, Any]:
    """全案件の metrics.json enum レコードを構築し JSONL へ書き出す（べき等・LLM 非使用）。"""
    all_refs = list(refs) if refs is not None else corpus.walk()
    projects = sorted({r.project for r in all_refs if r.project and r.category != "internal"})
    records: list[dict[str, Any]] = []
    for project in projects:
        try:
            rec = _build_case(project, all_refs)
        except Exception:  # noqa: BLE001 — 1 案件の失敗が全体を壊さない
            continue
        if rec is not None:
            records.append(rec)

    out_path = out or default_out_path()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as fh:
        fh.write(json.dumps({"schema": SCHEMA, "version": SCHEMA_VERSION,
                             "n_cases": len(records)}, ensure_ascii=False) + "\n")
        for rec in records:
            fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
    _LOAD_CACHE.pop(str(out_path), None)

    report = {
        "cases": len(records),
        "with_selected_columns": sorted(r["project"] for r in records if "selected_columns" in r),
        "interaction_columns": {r["project"]: r.get("interaction_columns")
                                for r in records if r.get("interaction_columns")},
    }
    if write_report:
        with open(default_report_path(), "w", encoding="utf-8") as fh:
            json.dump(report, fh, ensure_ascii=False, indent=2)
    return {"cases": len(records), "report": report}


def load(path: Path | None = None) -> list[dict[str, Any]]:
    """ストアを読み込む（memoized）。欠損/スキーマ不一致 ⇒ 空（回帰ゼロ）。"""
    out = path or default_out_path()
    key = str(out)
    if key in _LOAD_CACHE:
        return _LOAD_CACHE[key]
    records: list[dict[str, Any]] = []
    try:
        with open(out, encoding="utf-8") as fh:
            header = fh.readline()
            meta = json.loads(header) if header.strip() else {}
            if isinstance(meta, dict) and meta.get("schema") == SCHEMA \
                    and meta.get("version") == SCHEMA_VERSION:
                for line in fh:
                    line = line.strip()
                    if line:
                        records.append(json.loads(line))
    except Exception:  # noqa: BLE001
        records = []
    _LOAD_CACHE[key] = records
    return records


if __name__ == "__main__":
    summary = build()
    print(json.dumps(summary["report"], ensure_ascii=False, indent=2))
