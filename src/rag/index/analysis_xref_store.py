"""SOT-2691 — 分析成果物クロス参照ストア（cycle8 C2）.

cycle8 abstain の「分析成果物への決定論到達」クラスタ（idx36/60/73、二次 53）を **質問非依存** の事前計算
ストアへ焼き込む。根本原因は「成果物（最終報告 / metrics.json / 実装 .py / train.csv）に事実は在るが、
質問→案件/段階/管理ID/カテゴリ列への束縛経路が浅く、投資者が予算 5 回内で到達できない」こと。既存
:mod:`src.rag.index.raw_artifact_store`（.py/.json/.csv 生読解）と用語集の案件束縛を土台に、本ストアは
**全案件を横断走査** し per-case で次を焼く（純粋な決定論抽出・LLM/genai 非使用 ⇒ cost $0）:

* **incomplete_ids**（idx60）— 最終報告資料の「未完事項 / 要アクション（未完事項）」節に列挙された管理ID
  （``AI-05`` 等）を節スコープで抽出。白峰 = ``AI-05 / AI-08 / AI-09``。全文 FTS には出るが「未完」語との
  近接束縛が無く投資者が発見できなかった軸。
* **categorical_columns** + **onehot_threshold**（idx73）— 案件 train.csv のカテゴリ列（object/category
  dtype、id/target 除外）ごとの ``nunique`` と、実装 ``features.py`` の One-Hot 閾値
  （``MAX_CATEGORICAL_UNIQUE``、fallback = run_summary/metrics の ``categorical_unique_limit``）。閾値未満の
  カテゴリ列が One-Hot 対象。かえで = Gender（nunique 2 < 100）。
* **selected_features** + **eng_ft**（idx53 stretch）— 最終報告の「選択特徴量」節の特徴量名から、train.csv の
  原列に無いもの（``_ord``/``×``/``-`` 等の派生）を **エンジニアリング特徴量 = ENG-FT** として分類。
  東都 = 6（Age_ord/Exp_ord/Edu_ord/Age×Exp/Age-Exp/Edu×Exp）。
* **staged_metrics**（idx36 infra）— metrics.json の最終 F1（``f1_macro``、フル精度）と leaderboard の段階別
  ベスト（8桁丸め）を honest に併記。**中間段階のフル精度値は成果物に丸めしか無い**ため、serve レーンは
  idx36 を発火しない（precision-first, SOT-2687 の教訓 = 正解を焼けない問いは honest abstain）。

規律: 純粋な決定論抽出（python-pptx/docx + pdfplumber + pandas + 標準ライブラリ、既存 office/plain
ヘルパのみ）。``RAG_ANALYSIS_XREF`` 既定 OFF ⇒ serve レーン／ツールは None（tool 集合/スキーマ/serve path は
byte-identical）。build 自体は常に実行可能（べき等）。
"""
from __future__ import annotations

import json
import os
import re
import unicodedata
from pathlib import Path
from typing import Any, Sequence

from config import settings

from src.rag import corpus
from src.rag.corpus import FileRef, nfc
from src.rag.extract import office, passwords, plain

SCHEMA = "analysis_xref_store"
SCHEMA_VERSION = 1
_ON = {"1", "true", "yes", "on"}

_LOAD_CACHE: dict[str, list[dict[str, Any]]] = {}

# 未完事項の節に列挙される管理ID（AI-05 / WBS-3 / M-01 等）。行頭 ID + コロン/全角コロンで束縛する。
_ID_LINE_RE = re.compile(r"^\s*([A-Za-z]{1,5})[-‐ー－]?(\d{1,3})\s*[:：]")
# 未完事項 節ヘッダ（「要アクション（未完事項）」「未完事項」「未対応」等）。
_INCOMPLETE_HEADER_RE = re.compile(r"未完事項|要アクション|未対応事項|未完了事項")
# 選択特徴量 節ヘッダ。
_SELECTED_FEATURE_HEADER_RE = re.compile(r"選択特徴量")
# 図形塗り色マーカー（pptx 抽出が付ける ``【図形塗り:赤】<name>``）— 特徴量名の前置ノイズとして除去する。
_SHAPE_FILL_RE = re.compile(r"【図形塗り:[^】]*】")
_SLIDE_MARK_RE = re.compile(r"^\[スライド")
# 実装ソースの One-Hot 閾値定数。
_MAX_CAT_UNIQUE_RE = re.compile(r"MAX_CATEGORICAL_UNIQUE\s*=\s*(\d+)")


def enabled() -> bool:
    """serve レーン／ツールを有効にするか。既定 OFF（``RAG_ANALYSIS_XREF``）⇒ byte-identical。"""
    return os.getenv("RAG_ANALYSIS_XREF", "0").strip().lower() in _ON


def default_out_path() -> Path:
    return settings.ARTIFACTS_DIR / "analysis_xref_store.jsonl"


def default_report_path() -> Path:
    return settings.ARTIFACTS_DIR / "analysis_xref_store_build_report.json"


# --------------------------------------------------------------------------- helpers
def _norm(text: Any) -> str:
    return unicodedata.normalize("NFKC", str(text or "")).replace(" ", "").replace("　", "").lower()


def _id_sort_key(idv: str) -> tuple[str, int]:
    m = re.match(r"([A-Za-z]+)[-‐ー－]?(\d+)", idv)
    if not m:
        return (idv, 0)
    return (m.group(1).upper(), int(m.group(2)))


def _report_text(ref: FileRef) -> str:
    """最終報告資料のフルテキスト（切り詰め前）。pptx/docx/pdf 対応・LLM 非使用。"""
    data = None
    try:
        if passwords.is_encrypted(ref.path):
            data = passwords.resolve(ref)
            if not data:
                return ""
    except Exception:  # noqa: BLE001
        data = None
    try:
        if ref.ext == "pptx":
            return office.extract_pptx(ref, data)
        if ref.ext == "docx":
            return office.extract_docx(ref, data)
        if ref.ext == "pdf":
            return plain.extract_pdf(ref)
    except Exception:  # noqa: BLE001 — 1 文書の失敗が全体を壊さない
        return ""
    return ""


def _final_report_ref(project: str, refs: Sequence[FileRef]) -> FileRef | None:
    """案件の最終報告資料（``_old`` は除外）。report カテゴリ優先、無ければ '最終報告' を rel に含むもの。"""
    cands = [r for r in refs
             if r.project == project and r.ext in ("pptx", "docx", "pdf")
             and ("最終報告" in r.rel or (r.category == "report"))
             and "old" not in nfc(r.name).lower()
             and not nfc(r.name).startswith("~$")]
    if not cands:
        return None
    # '最終報告' を含むものを最優先、次に report カテゴリ、最後に rel 最短で安定選択。
    cands.sort(key=lambda r: (0 if "最終報告" in r.rel else 1,
                              0 if r.category == "report" else 1, len(r.rel)))
    return cands[0]


def _train_ref(project: str, refs: Sequence[FileRef]) -> FileRef | None:
    """案件の train.csv（03.データ 配下を優先、無ければ project 内の train.csv）。"""
    cands = [r for r in refs if r.project == project and r.name.lower() == "train.csv"]
    if not cands:
        return None
    cands.sort(key=lambda r: (0 if "03." in r.rel or "データ" in r.rel else 1, len(r.rel)))
    return cands[0]


def _features_src(project: str, refs: Sequence[FileRef]) -> str:
    for r in refs:
        if r.project == project and r.name == "features.py":
            try:
                return r.path.read_text(errors="replace")
            except Exception:  # noqa: BLE001
                return ""
    return ""


def _load_json_ref(project: str, refs: Sequence[FileRef], name: str) -> dict[str, Any] | None:
    for r in refs:
        if r.project == project and r.name == name and r.ext == "json":
            try:
                return json.loads(r.path.read_text(errors="replace"))
            except Exception:  # noqa: BLE001
                return None
    return None


# --------------------------------------------------------------------------- idx60 未完事項 ID
def extract_incomplete_ids(text: str) -> dict[str, Any] | None:
    """最終報告テキストの「未完事項」節に列挙された管理IDを節スコープで抽出（節が無ければ None）。

    節ヘッダ行（未完事項/要アクション…）の直後から、行頭が ``ID:`` 形の連続行を集める。連続ブロックが
    切れた（節境界・※注記・別見出し）ら停止する。IDは prefix+数字（``AI-05`` 等）で昇順に一意化する。
    """
    lines = text.splitlines()
    header_i = None
    for i, line in enumerate(lines):
        if _INCOMPLETE_HEADER_RE.search(line):
            header_i = i
            break
    if header_i is None:
        return None
    ids: list[str] = []
    section_lines: list[str] = []
    for line in lines[header_i + 1:]:
        stripped = line.strip()
        if not stripped:
            if ids:  # 空行はブロック内では許容、ただし ID を拾い始めた後の空行では継続を許す
                continue
            continue
        m = _ID_LINE_RE.match(line)
        if m:
            idv = f"{m.group(1).upper()}-{int(m.group(2)):02d}"
            if idv not in ids:
                ids.append(idv)
            section_lines.append(stripped[:120])
            continue
        # ID 行を1つでも拾った後に非ID行が来たら節ブロック終了。
        if ids:
            break
        # まだ ID を拾っていない段階で別スライド/別節に入ったら諦める（節が空）。
        if _SLIDE_MARK_RE.match(line) or _INCOMPLETE_HEADER_RE.search(line):
            continue
    if not ids:
        return None
    ids_sorted = sorted(ids, key=_id_sort_key)
    return {"ids": ids_sorted, "section_header": lines[header_i].strip()[:80],
            "section_lines": section_lines}


# --------------------------------------------------------------------------- idx73 categorical nunique
def _read_train_columns(ref: FileRef) -> tuple[list[str], dict[str, dict[str, Any]]] | None:
    """train.csv の全列名と、object/category dtype 列の ``nunique`` を返す。読めなければ None。"""
    try:
        import pandas as pd
    except Exception:  # noqa: BLE001
        return None
    for enc in ("utf-8-sig", "utf-8", "cp932"):
        try:
            df = pd.read_csv(ref.path, encoding=enc)
            break
        except Exception:  # noqa: BLE001
            df = None
    if df is None:
        return None
    columns = [str(c) for c in df.columns]
    cat: dict[str, dict[str, Any]] = {}
    for c in df.columns:
        s = df[c]
        is_cat = (s.dtype == object) or str(s.dtype) == "category"
        if is_cat:
            cat[str(c)] = {"nunique": int(s.dropna().nunique()), "dtype": str(s.dtype)}
    return columns, cat


def _onehot_threshold(features_src: str, run_summary: dict[str, Any] | None,
                      metrics: dict[str, Any] | None) -> int | None:
    """One-Hot（高カーディナリティ除外）閾値: features.py の ``MAX_CATEGORICAL_UNIQUE`` を最優先、
    fallback = run_summary/metrics の ``categorical_unique_limit``。"""
    m = _MAX_CAT_UNIQUE_RE.search(features_src or "")
    if m:
        return int(m.group(1))
    for src in (run_summary, metrics):
        if isinstance(src, dict) and isinstance(src.get("categorical_unique_limit"), int):
            return int(src["categorical_unique_limit"])
    return None


def _build_onehot(project: str, refs: Sequence[FileRef]) -> dict[str, Any] | None:
    train = _train_ref(project, refs)
    if train is None:
        return None
    cols = _read_train_columns(train)
    if cols is None:
        return None
    columns, cat = cols
    features_src = _features_src(project, refs)
    run_summary = _load_json_ref(project, refs, "run_summary.json")
    metrics = _load_json_ref(project, refs, "metrics.json")
    threshold = _onehot_threshold(features_src, run_summary, metrics)
    target = None
    if isinstance(metrics, dict):
        target = metrics.get("target_column")
    if not target and isinstance(run_summary, dict):
        target = run_summary.get("target_column")
    # id/target を除いたカテゴリ列。
    excluded = {str(target)} if target else set()
    excluded |= {c for c in cat if c.lower() in ("id", "index")}
    categorical = {c: v for c, v in cat.items() if c not in excluded}
    onehot_targets = None
    if threshold is not None:
        onehot_targets = sorted(c for c, v in categorical.items()
                                if isinstance(v.get("nunique"), int) and v["nunique"] < threshold)
    return {
        "train_rel": train.rel,
        "threshold": threshold,
        "threshold_source": "features.py:MAX_CATEGORICAL_UNIQUE" if _MAX_CAT_UNIQUE_RE.search(features_src or "")
        else "categorical_unique_limit",
        "target_column": target,
        "categorical": categorical,
        "onehot_targets": onehot_targets,
        "all_columns": columns,
    }


# --------------------------------------------------------------------------- idx53 selected/eng-ft
def extract_selected_features(text: str, original_columns: Sequence[str]) -> dict[str, Any] | None:
    """最終報告の「選択特徴量」節から特徴量名を抽出し、原列に無いものを ENG-FT として分類する。

    抽出方針（決定論）: 「選択特徴量」節ヘッダ以降、``■ 原特徴量 … エンジニアリング特徴量`` の凡例行まで
    のトークンを候補にし、``【図形塗り:赤】`` マーカーや除外注記を落として特徴量名列を作る。原列名集合に
    含まれないものを ENG-FT とする。候補が作れなければ None。
    """
    lines = text.splitlines()
    header_i = None
    for i, line in enumerate(lines):
        if _SELECTED_FEATURE_HEADER_RE.search(line):
            header_i = i
    if header_i is None:
        return None
    orig_norm = {_norm(c) for c in original_columns}
    selected: list[str] = []
    for line in lines[header_i + 1:]:
        stripped = _SHAPE_FILL_RE.sub("", line).strip()
        if not stripped:
            continue
        # 凡例行に到達したら特徴量列挙は終了。
        if "原特徴量" in stripped and "エンジニアリング特徴量" in stripped:
            break
        if _SLIDE_MARK_RE.match(stripped) or stripped.startswith("["):
            break
        # 注記・欠損統計・除外列などのメタ行は特徴量名ではない。
        if any(tok in stripped for tok in ("除外列", "欠損", "件（", "→", "：", ":", "特徴量数", "行数", "クラス")):
            continue
        # 1トークン（列名相当）だけを特徴量として扱う。区切りを含む説明文は除外。
        token = stripped
        if len(token) > 24 or " " in token or "、" in token or "。" in token:
            continue
        if token and token not in selected:
            selected.append(token)
    if not selected:
        return None
    eng_ft = [f for f in selected if _norm(f) not in orig_norm]
    return {"selected": selected, "selected_count": len(selected),
            "eng_ft": eng_ft, "eng_ft_count": len(eng_ft),
            "original_columns": list(original_columns)}


# --------------------------------------------------------------------------- idx36 staged metrics (infra)
def _build_staged_metrics(project: str, refs: Sequence[FileRef]) -> dict[str, Any] | None:
    """metrics.json の最終 F1（フル精度）と leaderboard の段階別ベスト（8桁丸め）を honest に併記。

    中間段階のフル精度値は成果物に丸めしか無い（leaderboard = 8桁）。serve レーンは idx36 を発火しない
    （precision-first, SOT-2687）。ここでは透明性のため段階値を焼くのみ。
    """
    metrics = _load_json_ref(project, refs, "metrics.json")
    final_f1 = None
    if isinstance(metrics, dict):
        v = metrics.get("f1_macro")
        if isinstance(v, (int, float)):
            final_f1 = float(v)
    if final_f1 is None:
        return None
    return {"final_f1_macro": final_f1, "final_f1_source": "metrics.json:f1_macro",
            "full_precision": True,
            "intermediate_full_precision_available": False,
            "note": "中間段階(線形系 T04 等)のフル精度 F1 は成果物に丸め(leaderboard 8桁)しか無いため "
                    "idx36 の全精度差は serve で発火しない(honest abstain, SOT-2687)。"}


# --------------------------------------------------------------------------- per-case
def _build_case(project: str, refs: Sequence[FileRef]) -> dict[str, Any]:
    rec: dict[str, Any] = {"project": project}
    report_ref = _final_report_ref(project, refs)
    report_text = _report_text(report_ref) if report_ref is not None else ""
    if report_ref is not None:
        rec["report_rel"] = report_ref.rel
    # idx60
    if report_text:
        inc = extract_incomplete_ids(report_text)
        if inc is not None:
            rec["incomplete_ids"] = inc
    # idx73
    onehot = _build_onehot(project, refs)
    if onehot is not None:
        rec["onehot"] = onehot
    # idx53
    if report_text:
        train = _train_ref(project, refs)
        cols = _read_train_columns(train) if train is not None else None
        original = cols[0] if cols else []
        feat = extract_selected_features(report_text, original)
        if feat is not None:
            rec["selected_features"] = feat
    # idx36 infra
    staged = _build_staged_metrics(project, refs)
    if staged is not None:
        rec["staged_metrics"] = staged
    return rec


# --------------------------------------------------------------------------- build
def build(refs: Sequence[FileRef] | None = None, *, out: Path | None = None,
          write_report: bool = True) -> dict[str, Any]:
    """全案件の per-case 分析クロス参照レコードを構築し JSONL へ書き出す（べき等・LLM 非使用）。"""
    all_refs = list(refs) if refs is not None else corpus.walk()
    projects = sorted({r.project for r in all_refs if r.project and r.category != "internal"})
    records: list[dict[str, Any]] = []
    for project in projects:
        try:
            rec = _build_case(project, all_refs)
        except Exception:  # noqa: BLE001 — 1 案件の失敗が全体を壊さない
            continue
        # project キー以外に何か焼けた場合のみ残す。
        if len(rec) > 1:
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
        "with_incomplete_ids": sorted(r["project"] for r in records if "incomplete_ids" in r),
        "with_onehot": sorted(r["project"] for r in records if "onehot" in r),
        "with_selected_features": sorted(r["project"] for r in records if "selected_features" in r),
        "incomplete_ids": {r["project"]: r["incomplete_ids"]["ids"]
                           for r in records if "incomplete_ids" in r},
        "onehot_targets": {r["project"]: r["onehot"].get("onehot_targets")
                           for r in records if "onehot" in r},
        "eng_ft_count": {r["project"]: r["selected_features"]["eng_ft_count"]
                         for r in records if "selected_features" in r},
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
