"""SOT-2678 — 分析成果物 raw ファイルの serve 配線（cycle6 K2 の lookup ツール）.

:mod:`src.rag.index.raw_artifact_store` が build 時に質問非依存で焼いた per-file レコード（.py/.json/.csv/
設定ファイルの切り詰めなし本文 + parsed 構造）と per-case rollup（config/metrics/leaderboard/適用ハイパラ）を、
cycle6 で「find_files 0件・read_office 非対応で分析成果物へ到達できず予算切れ棄権」だった raw ファイル欠落
クラスタ（idx32/61/62、二次 35/36/73）を **投資者が呼べる lookup ツール** として接続する。

決定論の直答はしない（idx32『selected_columns×交互作用特徴量』/idx61『コード適用ハイパラ』/idx62『上位2件
設定差分』はいずれも表/コードを読んで質問固有の値を組む型で、単一スカラの厳密束縛には落ちない ⇒ wrong リスク
回避のため precision-first で reach のみ提供、LLM ループへ委ねる）:

* **artifact_grep(case/keyword/file/ext)** — 分析成果物(.py/.json/.csv/.toml/.ipynb)を keyword で grep し、
  一致行を行番号+近傍行+出自(rel)付きで返す。keyword 省略時は file の先頭窓を返す。read_office 非対応・
  find_files 0件でも本ツールでコード/設定/生成物の中身へ直接到達する。
* **analysis_artifact_lookup(case/kind)** — 案件の事前計算 rollup を構造化して返す。kind=hyperparams は
  config.model_params＋modeling.py コード上デフォルトをマージした適用ハイパラ（provenance 付き）、
  kind=leaderboard は実験 CSV をメトリクス最良順に並べた行、kind=config/metrics は該当 JSON を返す。

規律: OFF（``RAG_RAW_ARTIFACT_STORE`` 既定）⇒ :func:`tool` は []（ツール集合/関数スキーマ/MCP surface は
byte-identical）。serve 中の追加 LLM 呼び出しは 0（事前計算済み JSONL を読むだけ = DB経路, RAW_FILE_TOOLS
ではない, SOT-2660）。曖昧な case は None ではなく keyword 横断検索へフォールバックしリーチを最大化する。
"""
from __future__ import annotations

from typing import Any, Callable

from src.rag.index import raw_artifact_store as _ras
from src.rag.tools import contract as _contract

ARTIFACT_GREP = "artifact_grep"
ANALYSIS_ARTIFACT_LOOKUP = "analysis_artifact_lookup"
_STR = {"type": "string"}

# tool 出力が investigator の _jsonable(max_str=8000/max_items=60) を超えないよう控えめに束ねる。
_MAX_FILES_OUT = 8
_MAX_HITS_PER_FILE = 12
_NEIGHBORS = 2                 # 一致行の前後に返す近傍行数
_HEAD_LINES = 60              # keyword 無し時に返す先頭窓の行数
_MAX_LB_ROWS_OUT = 12


def _top2_setting_diff(lb: dict[str, Any]) -> dict[str, Any] | None:
    """上位2行の設定差分を質問非依存に要約する（スコア/識別列は比較対象外）。"""
    rows = lb.get("rows") or []
    if len(rows) < 2:
        return None
    excluded = {"trial_index", "status", "primary_metric", "primary_value",
                "secondary_metric", "secondary_value"}
    first, second = rows[0], rows[1]
    differences = {
        key: {"rank1": first.get(key), "rank2": second.get(key)}
        for key in lb.get("header") or []
        if key not in excluded and first.get(key) != second.get(key)
    }
    return {
        "rank1": {"trial_index": first.get("trial_index"),
                  "score": first.get(lb.get("metric_col"))},
        "rank2": {"trial_index": second.get("trial_index"),
                  "score": second.get(lb.get("metric_col"))},
        "setting_differences": differences,
    }


def enabled() -> bool:
    return _ras.enabled()


# --------------------------------------------------------------------------- case / file selection
def _bind_project(text: str) -> str | None:
    try:
        from src.rag.extract import glossary
        return glossary.load().company_of(text)
    except Exception:  # noqa: BLE001
        return None


def _select_files(store: dict[str, Any], *, case: str, file: str, ext: str) -> list[dict[str, Any]]:
    """case(会社名)/ file(rel・名前部分一致)/ ext で対象成果物を選ぶ。両方空なら全成果物。"""
    files: list[dict[str, Any]] = store.get("files") or []
    if ext:
        e = ext.strip().lower().lstrip(".")
        files = [f for f in files if (f.get("ext") or "") == e]
    if file:
        key = _ras.norm(file)
        hit = [f for f in files if key in _ras.norm(f.get("rel")) or key in _ras.norm(f.get("name"))]
        if hit:
            return hit
    project = _bind_project(case) if case else None
    if project:
        scoped = [f for f in files if f.get("project") == project]
        if scoped:
            return scoped
    if case:
        key = _ras.norm(case)
        scoped = [f for f in files if key in _ras.norm(f.get("project")) or key in _ras.norm(f.get("rel"))]
        if scoped:
            return scoped
    return files


def _resolve_case_rollup(store: dict[str, Any], case: str) -> dict[str, Any] | None:
    cases_by_project: dict[str, Any] = store.get("cases_by_project") or {}
    if not case:
        return None
    project = _bind_project(case)
    if project and project in cases_by_project:
        return cases_by_project[project]
    key = _ras.norm(case)
    for proj, roll in cases_by_project.items():
        if key and key in _ras.norm(proj):
            return roll
    return None


# --------------------------------------------------------------------------- artifact_grep
def _grep_file(rec: dict[str, Any], key: str) -> list[dict[str, Any]]:
    text = rec.get("text") or ""
    lines = text.splitlines()
    if not key:
        return [{"line_no": 1, "context": lines[:_HEAD_LINES]}]
    hits: list[dict[str, Any]] = []
    for i, line in enumerate(lines):
        if key in _ras.norm(line):
            a = max(0, i - _NEIGHBORS)
            b = min(len(lines), i + _NEIGHBORS + 1)
            hits.append({"line_no": i + 1, "context": lines[a:b]})
            if len(hits) >= _MAX_HITS_PER_FILE:
                break
    return hits


def _artifact_grep(case: str = "", keyword: str = "", file: str = "", ext: str = "") -> dict[str, Any]:
    store = _ras.load()
    files = _select_files(store, case=case, file=file, ext=ext)
    key = _ras.norm(keyword)
    out: list[dict[str, Any]] = []
    for f in files:
        hits = _grep_file(f, key)
        if not hits:
            continue
        out.append({"file": f.get("rel"), "project": f.get("project"), "ext": f.get("ext"),
                    "n_lines": f.get("n_lines"), "hits": hits})
        if len(out) >= _MAX_FILES_OUT:
            break
    found = bool(out)
    return _contract.make(
        {"files": out, "count": len(out)}, engine="raw_artifact_store",
        evidence={"applicable": True, "case": case or None, "file": file or None,
                  "keyword": keyword or None, "ext": ext or None, "found": found,
                  "files_scanned": len(files)},
        scheme="artifact-grep", note=None if found else "一致する成果物なし")


# --------------------------------------------------------------------------- analysis_artifact_lookup
def _lookup(case: str = "", kind: str = "") -> dict[str, Any]:
    store = _ras.load()
    roll = _resolve_case_rollup(store, case)
    if roll is None:
        return _contract.make(None, engine="raw_artifact_store",
                              evidence={"applicable": True, "case": case or None, "found": False},
                              note="該当案件の成果物 rollup なし")
    k = _ras.norm(kind)
    value: Any
    scheme: str
    if "hyper" in k or "param" in k or "モデル" in k or "model" in k:
        value = roll.get("applied_hyperparams")
        scheme = "artifact-hyperparams"
    elif "leader" in k or "board" in k or "実験" in k or "trial" in k or "experiment" in k:
        lb = roll.get("leaderboard") or {}
        value = ({**lb, "top2_comparison": _top2_setting_diff(lb),
                  "rows": (lb.get("rows") or [])[:_MAX_LB_ROWS_OUT]} if lb else None)
        scheme = "artifact-leaderboard"
    elif "metric" in k or "スコア" in k or "評価" in k:
        value = roll.get("metrics")
        scheme = "artifact-metrics"
    elif "config" in k or "設定" in k:
        value = roll.get("config")
        scheme = "artifact-config"
    else:
        # kind 未指定/不明 ⇒ rollup 全体（artifacts 一覧 + 各構造化ブロックの要約）を返す。
        lb = roll.get("leaderboard") or {}
        value = {
            "project": roll.get("project"),
            "artifacts": roll.get("artifacts"),
            "has": {"config": "config" in roll, "metrics": "metrics" in roll,
                    "leaderboard": "leaderboard" in roll,
                    "applied_hyperparams": "applied_hyperparams" in roll},
            "applied_hyperparams": roll.get("applied_hyperparams"),
            "leaderboard_top": (lb.get("rows") or [])[:3] if lb else None,
        }
        scheme = "artifact-rollup"
    found = value is not None
    return _contract.make(
        value, engine="raw_artifact_store",
        evidence={"applicable": True, "case": case or None, "project": roll.get("project"),
                  "kind": kind or None, "found": found},
        scheme=scheme, note=None if found else "該当 rollup ブロックなし")


# --------------------------------------------------------------------------- tool wiring
def tool() -> "list[tuple[str, str, dict[str, Any], Callable[..., Any]]]":
    """RAG_RAW_ARTIFACT_STORE OFF ⇒ [] ⇒ ツール集合/関数スキーマ/MCP surface は byte-identical。"""
    if not enabled():
        return []
    return [
        (
            ARTIFACT_GREP,
            "分析成果物 grep: 案件の .py/.json/.csv/.toml/.ipynb（コード・設定・生成物）を keyword で検索し、"
            "一致行を行番号+近傍行+出自(ファイル rel)付きで返す。case(会社名の一部)/file(ファイル名の一部)/"
            "ext で対象を絞れる。keyword を省くと file の先頭窓を返す。read_office は .py/.json/.csv を開けず "
            "find_files も analysis_outputs/src を拾えないが、本ツールならコード/設定/CSV の中身へ直接到達できる。"
            "『features.py の交互作用特徴量』『modeling.py の n_estimators』『metrics.json の値』はこれで引く。",
            {"type": "object", "properties": {"case": _STR, "keyword": _STR, "file": _STR, "ext": _STR},
             "required": []},
            _artifact_grep,
        ),
        (
            ANALYSIS_ARTIFACT_LOOKUP,
            "分析成果物 rollup 参照: 案件の事前計算まとめを構造化して返す。kind='hyperparams' は "
            "project_config.json の model_params と modeling.py のコード上デフォルトをマージした『実行時適用"
            "ハイパラ』(applied)を provenance 付きで返す(model_params 空でもコード適用値=n_estimators 等を解決)。"
            "kind='leaderboard' は実験 CSV をメトリクス最良順に並べた行(上位比較=設定差分に使う)。"
            "kind='config'/'metrics' は該当 JSON。case のみで kind 省略時は rollup 概要。",
            {"type": "object", "properties": {"case": _STR, "kind": _STR}, "required": ["case"]},
            _lookup,
        ),
    ]
