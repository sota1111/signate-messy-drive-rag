"""SOT-2678 — 分析成果物 raw ファイルストア（cycle6 K2）.

cycle6 abstain の「分析成果物 raw ファイル非対応」クラスタ（idx32/61/62、二次 35/36/73）を **質問非依存**
の事前計算ストアで解消する。根本原因: 案件の分析成果物である ``.py`` / ``.json`` / ``.csv`` / 設定ファイルは、

* ``find_files`` の 0件（``analysis_outputs/`` 下の生成物・``analysis_project/src/`` のコードは通常の文書
  検索索引に載らない）、
* ``read_office`` の非対応（office/pdf 抽出器で .py/.json/.csv を開けない）

の二重で investigator から到達不能になり、予算切れ棄権していた（idx32: 青嶺 metrics.json/features.py、
idx61: 京橋 modeling.py の n_estimators/learning_rate/random_state コード適用値、idx62: 青葉与信
leaderboard.csv の上位2件設定差分）。

本ストアは全案件×全成果物を質問非依存に走査し、次を焼き込む:

* **per-file レコード** — project/rel/name/ext + 切り詰めなし本文（capped）。json は parsed 構造、
  csv/tsv は header+rows を同梱（serve レーンが offset/keyword 窓・行参照で 8000 字境界の向こう側へ到達）。
* **per-case rollup** — ``config``(project_config.json) / ``metrics``(metrics.json) / ``leaderboard``
  (実験 CSV を primary_value 降順で行化) / ``applied_hyperparams``（config の ``model_params`` ＋
  modeling.py のコード上デフォルト ``to_int/to_float(model_params.get(k), D)`` をマージした案件別適用値、
  provenance 付き・best-effort）。

規律: 純粋な決定論抽出（標準ライブラリ + 既存 passwords ヘルパのみ、LLM/genai 非使用 ⇒ cost $0）。
serve は本 JSONL を読むだけ（DB経路 — RAW_FILE_TOOLS ではない, SOT-2660）。``RAG_RAW_ARTIFACT_STORE``
既定 OFF ⇒ serve ツールは None（tool 集合/スキーマ/serve path は byte-identical）。build 自体は常に
実行可能（べき等）。
"""
from __future__ import annotations

import csv
import io
import json
import os
import re
import unicodedata
from pathlib import Path
from typing import Any, Sequence

from config import settings

from src.rag import corpus
from src.rag.corpus import FileRef, nfc

SCHEMA = "raw_artifact_store"
SCHEMA_VERSION = 1
_ON = {"1", "true", "yes", "on"}

_LOAD_CACHE: dict[str, dict[str, Any]] = {}

# 対象拡張子（分析成果物 = コード / 構造化データ / 設定）。xlsx/docx/pptx/pdf は doc_reach / visual /
# fact ストアが担うので対象外。lock/png 等のバイナリ・依存ロックは除外。
_ARTIFACT_EXTS = ("py", "json", "csv", "tsv", "toml", "yaml", "yml", "cfg", "ini", "ipynb")

# 病的ファイルがストア/tool 出力を氾濫させないための上限（決定論・質問非依存）。
MAX_TEXT_CHARS = 200_000
MAX_CSV_ROWS = 400
MAX_CSV_COLS = 60
MAX_CELL_CHARS = 300
MAX_JSON_CHARS = 40_000  # parsed JSON を再直列化した文字列の上限


def enabled() -> bool:
    """serve ツールを有効にするか。既定 OFF（``RAG_RAW_ARTIFACT_STORE``）⇒ byte-identical。"""
    return os.getenv("RAG_RAW_ARTIFACT_STORE", "0").strip().lower() in _ON


def default_out_path() -> Path:
    return settings.ARTIFACTS_DIR / "raw_artifact_store.jsonl"


def default_report_path() -> Path:
    return settings.ARTIFACTS_DIR / "raw_artifact_store_build_report.json"


# --------------------------------------------------------------------------- helpers
def norm(text: Any) -> str:
    """NFKC 正規化 + 空白除去 + lower（keyword 照合キー）。"""
    return unicodedata.normalize("NFKC", str(text or "")).replace(" ", "").replace("　", "").lower()


def _read_text(ref: FileRef) -> str:
    """成果物テキストを読む（BOM 許容・decode 失敗は best-effort 置換）。"""
    try:
        raw = ref.path.read_bytes()
    except Exception:  # noqa: BLE001
        return ""
    for enc in ("utf-8-sig", "utf-8", "cp932"):
        try:
            return raw.decode(enc)
        except Exception:  # noqa: BLE001
            continue
    return raw.decode("utf-8", errors="replace")


def _clip_cell(value: Any) -> str:
    s = "" if value is None else str(value)
    return s.replace("\r", " ").replace("\n", " ").strip()[:MAX_CELL_CHARS]


def _parse_csv(text: str) -> dict[str, Any] | None:
    """csv/tsv を header + rows(dict) に正規化（上限で束ねる）。空なら None。"""
    if not text.strip():
        return None
    delim = "\t" if text.count("\t") > text.count(",") else ","
    try:
        reader = csv.reader(io.StringIO(text), delimiter=delim)
        raw_rows = [row for row in reader if any((c or "").strip() for c in row)]
    except Exception:  # noqa: BLE001
        return None
    if not raw_rows:
        return None
    header = [_clip_cell(c) for c in raw_rows[0][:MAX_CSV_COLS]]
    rows: list[dict[str, str]] = []
    for r in raw_rows[1 : 1 + MAX_CSV_ROWS]:
        cells = [_clip_cell(c) for c in r[:MAX_CSV_COLS]]
        rows.append({header[i] if i < len(header) else f"col{i}": cells[i]
                     for i in range(len(cells))})
    return {"header": header, "rows": rows, "n_rows": len(raw_rows) - 1}


def _parse_json(text: str) -> Any | None:
    try:
        obj = json.loads(text)
    except Exception:  # noqa: BLE001
        return None
    dumped = json.dumps(obj, ensure_ascii=False)
    if len(dumped) > MAX_JSON_CHARS:
        # 巨大 JSON は本文(text)側で窓参照できるので parsed は落とす（サイズ暴発回避）。
        return None
    return obj


# --------------------------------------------------------------------------- per-file
def build_file(ref: FileRef) -> dict[str, Any] | None:
    """1 成果物を per-file レコードへ正規化。読めない/空なら None。"""
    text = _read_text(ref)
    if not text.strip():
        return None
    rec: dict[str, Any] = {
        "project": ref.project,
        "rel": ref.rel,
        "name": ref.name,
        "ext": ref.ext,
        "category": ref.category,
        "n_chars": len(text),
        "n_lines": text.count("\n") + 1,
        "text": text[:MAX_TEXT_CHARS],
    }
    if ref.ext == "json":
        parsed = _parse_json(text)
        if parsed is not None:
            rec["json"] = parsed
    elif ref.ext in ("csv", "tsv"):
        parsed = _parse_csv(text)
        if parsed is not None:
            rec["csv"] = parsed
    return rec


# --------------------------------------------------------------------------- hyperparam resolution
# modeling.py のコード上デフォルト抽出: `to_int(model_params.get("n_estimators"), 300)` /
# `to_float(model_params.get("learning_rate"), 0.1)` 形を拾う（分岐非依存・決定論）。
_TO_DEFAULT = re.compile(
    r"to_(?:int|float)\(\s*model_params\.get\(\s*[\"']([A-Za-z0-9_]+)[\"']\s*\)\s*,\s*([0-9]+(?:\.[0-9]+)?)\s*\)"
)
# 選択モデルのコンストラクタに直書きされたリテラル（`n_estimators=500,` `learning_rate=0.3,`）を透明化。
_LITERAL_KW = re.compile(r"\b([A-Za-z0-9_]+)\s*=\s*([0-9]+(?:\.[0-9]+)?)\b")


def _code_defaults(modeling_src: str) -> dict[str, float | int]:
    out: dict[str, float | int] = {}
    for key, default in _TO_DEFAULT.findall(modeling_src or ""):
        out[key] = float(default) if "." in default else int(default)
    return out


def _code_literals(modeling_src: str) -> dict[str, list[Any]]:
    """モデルコンストラクタに直書きされたハイパラリテラルの一覧（透明性用・分岐差の手掛かり）。"""
    lit: dict[str, set[str]] = {}
    for key, val in _LITERAL_KW.findall(modeling_src or ""):
        if key in ("n_estimators", "learning_rate", "max_depth", "min_samples_leaf",
                   "random_state", "alpha", "max_iter", "c", "test_size"):
            lit.setdefault(key, set()).add(val)
    return {k: sorted(v, key=lambda s: (len(s), s)) for k, v in sorted(lit.items())}


def resolve_hyperparams(config: dict[str, Any] | None, metrics: dict[str, Any] | None,
                        modeling_src: str) -> dict[str, Any] | None:
    """config の ``model_params`` ＋ modeling.py のコード上デフォルトをマージした「適用ハイパラ」を返す。

    best-effort・provenance 付き（分岐/リテラル上書きの可能性を明示）。resolve 不能なら None。
    """
    if not config and not modeling_src:
        return None
    cfg = config or {}
    model_params = cfg.get("model_params")
    if not isinstance(model_params, dict):
        model_params = {}
    code_defaults = _code_defaults(modeling_src)
    applied: dict[str, Any] = {}
    for key, default in code_defaults.items():
        applied[key] = model_params.get(key, default)
    # model_params で追加指定されたが code_defaults に無いキーも取り込む。
    for key, val in model_params.items():
        applied.setdefault(key, val)
    random_state = cfg.get("random_state")
    if random_state is not None:
        applied["random_state"] = random_state
    if not applied:
        return None
    task_type = (metrics or {}).get("task_type") if isinstance(metrics, dict) else None
    return {
        "model_type": cfg.get("model_type"),
        "task_type": task_type,
        "applied": applied,
        "config_model_params": model_params,
        "code_defaults": code_defaults,
        "code_literals": _code_literals(modeling_src),
        "provenance": (
            "適用値 = config.model_params があればそれ、無ければ modeling.py の "
            "to_int/to_float(model_params.get(k), D) コード上デフォルト。random_state は config 由来。"
            "task_type/model_type の分岐でリテラル直書き（code_literals）が優先される場合があるので確認。"
        ),
    }


# --------------------------------------------------------------------------- leaderboard rollup
# error 系メトリクスは lower-is-better（回帰の rmse/mae 等）。それ以外（f1/accuracy/auc 等）は higher-is-better。
_LOWER_IS_BETTER = ("rmse", "mae", "mse", "rmsle", "error", "loss", "logloss")


def _leaderboard_rows(rec: dict[str, Any]) -> dict[str, Any] | None:
    """実験 leaderboard.csv レコードを primary_value のメトリクス方向に沿って並べた rollup にする。

    primary_metric 列が誤差系（rmse/mae 等）なら昇順（=最良が先頭）、それ以外は降順。方向が読めない時は降順。
    """
    parsed = rec.get("csv") or {}
    rows = parsed.get("rows") or []
    if not rows:
        return None
    header = parsed.get("header") or []
    metric_col = None
    for cand in ("primary_value", "value", "score", "f1_macro"):
        if cand in header:
            metric_col = cand
            break
    # メトリクス名（primary_metric 列 or メトリクス値列名そのもの）から最良方向を決める。
    metric_name = ""
    if rows and "primary_metric" in header:
        metric_name = str(rows[0].get("primary_metric") or "").lower()
    elif metric_col:
        metric_name = metric_col.lower()
    lower_is_better = any(tok in metric_name for tok in _LOWER_IS_BETTER)

    def _key(row: dict[str, Any]) -> float:
        try:
            return float(row.get(metric_col, ""))
        except (TypeError, ValueError):
            return float("inf") if lower_is_better else float("-inf")

    ordered = sorted(rows, key=_key, reverse=not lower_is_better) if metric_col else rows
    return {"rel": rec.get("rel"), "metric_col": metric_col, "metric_name": metric_name or None,
            "lower_is_better": lower_is_better, "header": header, "rows": ordered}


# --------------------------------------------------------------------------- per-case rollup
def _pick(records: list[dict[str, Any]], *, name_contains: str, ext: str) -> dict[str, Any] | None:
    key = name_contains.lower()
    hits = [r for r in records if r.get("ext") == ext and key in (r.get("name") or "").lower()]
    if not hits:
        return None
    # analysis_project/configs を analysis_outputs 生成物より優先しない: 最短 rel を安定選択。
    return sorted(hits, key=lambda r: len(r.get("rel") or ""))[0]


def _build_case_rollup(project: str, records: list[dict[str, Any]]) -> dict[str, Any]:
    config_rec = _pick(records, name_contains="project_config", ext="json")
    metrics_rec = _pick(records, name_contains="metrics", ext="json")
    modeling_rec = _pick(records, name_contains="modeling", ext="py")
    leaderboard_rec = _pick(records, name_contains="leaderboard", ext="csv")
    config = config_rec.get("json") if config_rec else None
    metrics = metrics_rec.get("json") if metrics_rec else None
    modeling_src = modeling_rec.get("text") if modeling_rec else ""
    rollup: dict[str, Any] = {
        "project": project,
        "artifacts": [{"rel": r.get("rel"), "name": r.get("name"), "ext": r.get("ext")}
                      for r in records],
    }
    if config is not None:
        rollup["config"] = {"rel": config_rec.get("rel"), "value": config}
    if metrics is not None:
        rollup["metrics"] = {"rel": metrics_rec.get("rel"), "value": metrics}
    if leaderboard_rec is not None:
        lb = _leaderboard_rows(leaderboard_rec)
        if lb is not None:
            rollup["leaderboard"] = lb
    applied = resolve_hyperparams(config, metrics, modeling_src)
    if applied is not None:
        applied["config_rel"] = config_rec.get("rel") if config_rec else None
        applied["modeling_rel"] = modeling_rec.get("rel") if modeling_rec else None
        rollup["applied_hyperparams"] = applied
    return rollup


# --------------------------------------------------------------------------- build
def build(refs: Sequence[FileRef] | None = None, *, out: Path | None = None,
          write_report: bool = True) -> dict[str, Any]:
    """全成果物の per-file レコード + per-case rollup を構築し JSONL へ書き出す（べき等・LLM 非使用）。"""
    all_refs = list(refs) if refs is not None else corpus.walk()
    arts = [r for r in all_refs if r.ext in _ARTIFACT_EXTS and not nfc(r.name).startswith("~$")]
    records: list[dict[str, Any]] = []
    skipped: list[str] = []
    for ref in arts:
        try:
            rec = build_file(ref)
        except Exception as exc:  # noqa: BLE001 — 1 ファイルの失敗が全体を壊さない
            skipped.append(f"{ref.rel}: {type(exc).__name__}: {exc}")
            continue
        if rec is None:
            skipped.append(f"{ref.rel}: empty/unreadable")
            continue
        records.append(rec)

    by_project: dict[str, list[dict[str, Any]]] = {}
    for rec in records:
        by_project.setdefault(rec.get("project") or "", []).append(rec)
    rollups = [_build_case_rollup(proj, recs) for proj, recs in sorted(by_project.items()) if proj]

    out_path = out or default_out_path()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as fh:
        fh.write(json.dumps({"schema": SCHEMA, "version": SCHEMA_VERSION,
                             "n_cases": len(rollups)}, ensure_ascii=False) + "\n")
        for roll in rollups:
            fh.write(json.dumps({"_kind": "case", **roll}, ensure_ascii=False) + "\n")
        for rec in records:
            fh.write(json.dumps({"_kind": "file", **rec}, ensure_ascii=False) + "\n")
    _LOAD_CACHE.pop(str(out_path), None)

    report = {
        "files": len(records),
        "cases": len(rollups),
        "skipped": skipped,
        "by_ext": {e: sum(1 for r in records if r.get("ext") == e) for e in _ARTIFACT_EXTS},
        "cases_with_hyperparams": sorted(r["project"] for r in rollups if "applied_hyperparams" in r),
        "cases_with_leaderboard": sorted(r["project"] for r in rollups if "leaderboard" in r),
    }
    if write_report:
        with open(default_report_path(), "w", encoding="utf-8") as fh:
            json.dump(report, fh, ensure_ascii=False, indent=2)
    return {"files": len(records), "cases": len(rollups), "report": report}


def load(path: Path | None = None) -> dict[str, Any]:
    """ストアを読み込む（memoized）。欠損/スキーマ不一致 ⇒ 空（回帰ゼロ）。

    返り値: ``{"files": [...], "cases": [...], "files_by_project": {...}, "cases_by_project": {...}}``。
    """
    out = path or default_out_path()
    key = str(out)
    if key in _LOAD_CACHE:
        return _LOAD_CACHE[key]
    files: list[dict[str, Any]] = []
    cases: list[dict[str, Any]] = []
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
                    obj = json.loads(line)
                    if obj.get("_kind") == "case":
                        cases.append(obj)
                    else:
                        files.append(obj)
    except Exception:  # noqa: BLE001
        files, cases = [], []
    files_by_project: dict[str, list[dict[str, Any]]] = {}
    for rec in files:
        files_by_project.setdefault(rec.get("project") or "", []).append(rec)
    cases_by_project = {c.get("project"): c for c in cases}
    data = {"files": files, "cases": cases,
            "files_by_project": files_by_project, "cases_by_project": cases_by_project}
    _LOAD_CACHE[key] = data
    return data


if __name__ == "__main__":
    summary = build()
    print(json.dumps(summary["report"], ensure_ascii=False, indent=2))
