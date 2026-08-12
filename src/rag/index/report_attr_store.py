"""最終報告 / 中間報告 / 調査系 文書の数値属性ストア（SOT-2655 / 事前計算事実層 追補 — クラスタD）.

Sonnet gold100 cycle4 の abstain のうち idx5/28/64（numeric・BUDGET_EXHAUSTED）は、最終報告・調査PDF が
**スキャン画像** で file_grep が本文に届かないため、証拠に lookup 1発で到達できないことが主因だった
(`docs/ai/sonnet_cycle_analysis/cycle4.md` クラスタD)。本モジュールは build 時に一度だけ、案件ごとの

* **最良/最終モデルのハイパーパラメータ表** — 分析成果物 ``04.分析/analysis_outputs/metrics.json`` の
  ``model_params``（report が「最良モデルの最終評価 (metrics.json)」と明示しているもの）。max_depth 等。
* **将来フェーズ計画の想定工数** — 報告書本文（スキャンPDF は :mod:`ocr_store` の永続 OCR、text-layer は
  :func:`src.rag.extract.extract`）から ``フェーズX … 工数：a-b h`` を行単位で抽出。範囲は範囲のまま保存。
* **予測に影響が高いと報告された特徴量** — 報告書本文で「影響/重要/寄与」の近傍に現れる train 列名。
* **その特徴量群 × ターゲット相関** — train 表（数値目的変数）から Pearson 相関を決定論計算（gold 非依存）。

を **質問を見ずに** 属性化し ``artifacts/report_attr_store.jsonl`` に焼く。serve path は事前計算済みテキストを
読むだけで **genai 呼び出しゼロ**。

Design invariants（sibling の case_master / derived_metrics / ocr_store と同一）
------------------------------------------------------------------------------
* **Opt-in at serve time.** :func:`enabled` (``RAG_REPORT_ATTR_STORE``) が runtime 参照のみを gate する。
  default OFF ⇒ champion serve path は byte-identical。
* **Build は LLM フリー・追加的.** 参照は既存の永続 OCR（前処理で確定済み）と text-layer と分析 JSON と
  train 表のみ。build で新たな genai 呼び出しはしない。読めない案件は 1 件スキップし build は継続。
* **Question-independent.** universe は構造的（報告書 category ＋ 調査/報告/予測 系 doc ＋ 分析 metrics.json）。
  質問も gold も参照しない。値は毎セル出典（doc/page）付き。
* **Fail-open.** artifact 欠落・size drift・解析不能はすべて空へフォールバック（回帰ゼロ）。
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

SCHEMA = "report-attr-store"
SCHEMA_VERSION = 1

_ON = {"1", "true", "yes", "on"}

# Model hyperparameters we surface from an analysis metrics.json (a fixed, question-independent whitelist
# of scikit / boosting knobs — never a gold value). Unknown keys inside ``model_params`` are also kept.
_KNOWN_PARAM_KEYS = (
    "max_depth", "learning_rate", "max_iter", "n_estimators", "num_leaves", "min_samples_leaf",
    "min_child_samples", "subsample", "colsample_bytree", "reg_lambda", "reg_alpha", "l2_regularization",
    "max_bins", "max_leaf_nodes", "min_child_weight", "gamma", "random_state", "max_features",
)

# Documents whose numeric attributes we attribute: 報告書 (category) or a survey/report/forecast doc.
_DOC_EXTS = {"pdf", "pptx"}
_DOC_NAME_CUE = re.compile(r"報告|調査|レポート|予測|動向|市場|final|report")

# フェーズ工数: a line like "期間/工数：2-4週間/20-30h想定" — only the hour-suffixed token is parsed.
_PHASE_HEAD = re.compile(r"フェーズ\s*([A-Za-zＡ-Ｚａ-ｚ0-9０-９]+)")
_KOSU_LINE = re.compile(r"工数")
_HOURS_RANGE = re.compile(r"(\d+)\s*[-〜~－ー]\s*(\d+)\s*[hHｈＨ]")
_HOURS_SINGLE = re.compile(r"(?<![\d-])(\d+)\s*[hHｈＨ]")
_PAGE_MARK = re.compile(r"\[ページ\s*(\d+)\]")

# 影響/重要 の近傍に現れた train 列名を「高影響特徴量」とみなす（報告本文の宣言）。
_IMPACT_CUE = re.compile(r"影響|重要|寄与|効いて|効く|効果|重要度")


def _z2h(s: str) -> str:
    """全角英数を半角へ（フェーズＡ → フェーズA 等の正規化）。"""
    return unicodedata.normalize("NFKC", s)


def enabled() -> bool:
    """True when the serve path may consult the persisted report attributes (default OFF — opt-in)."""
    return os.getenv("RAG_REPORT_ATTR_STORE", "0").strip().lower() in _ON


def default_out_path() -> Path:
    return settings.ARTIFACTS_DIR / "report_attr_store.jsonl"


def default_report_path() -> Path:
    return settings.ARTIFACTS_DIR / "report_attr_store_build_report.json"


# --------------------------------------------------------------------------- doc / analysis locators
def _report_refs(project: str, refs: Sequence[FileRef]) -> list[FileRef]:
    """報告/調査系 doc（pdf/pptx, 旧版・ロック除外）— 主要文書を先頭に、コーパス順で。"""
    out: list[FileRef] = []
    for r in refs:
        if r.project != project or r.ext not in _DOC_EXTS:
            continue
        name = nfc(r.name)
        if "old" in name.lower() or name.startswith("~$"):
            continue
        if r.category == "report" or _DOC_NAME_CUE.search(name):
            out.append(r)
    # 最終報告 を優先（idx5/28/64 の権威的文書）
    out.sort(key=lambda r: (0 if "最終報告" in nfc(r.name) else 1, r.rel))
    return out


def _metrics_ref(project: str, refs: Sequence[FileRef]) -> FileRef | None:
    """案件の最終/最良モデル成果物 ``analysis_outputs/metrics.json`` を返す（analysis_project は劣後）。"""
    cands = [r for r in refs
             if r.project == project and r.ext == "json" and nfc(r.name) == "metrics.json"]
    for r in cands:
        if "analysis_outputs" in r.rel and "analysis_project" not in r.rel:
            return r
    return cands[0] if cands else None


def _report_text(ref: FileRef) -> str:
    """報告書本文: スキャンPDF は永続 OCR、その他は text-layer/extract。genai 呼び出しはしない。"""
    try:
        from src.rag.index import ocr_store
        ocr = ocr_store.lookup(ref)
        if ocr and ocr.strip():
            return nfc(ocr)
    except Exception:  # noqa: BLE001
        pass
    try:
        from src.rag.extract import extract
        doc = extract(ref, caption_images=False)
        return nfc(getattr(doc, "text", "") or "")
    except Exception:  # noqa: BLE001
        return ""


# --------------------------------------------------------------------------- attribute extractors
def _model_attrs(project: str, refs: Sequence[FileRef]) -> dict[str, Any] | None:
    """analysis metrics.json の best/final model 情報を属性化（毎セル出典付き）。"""
    ref = _metrics_ref(project, refs)
    if ref is None:
        return None
    try:
        data = json.loads(Path(ref.path).read_text(encoding="utf-8-sig"))
    except Exception:  # noqa: BLE001
        return None
    if not isinstance(data, dict):
        return None
    params_raw = data.get("model_params")
    if not isinstance(params_raw, dict):
        return None
    src = {"doc_id": nfc(ref.rel)}
    # 既知キー順 + 追加キー（決定論順）: 全パラメータを属性化（質問非依存）。
    ordered: list[str] = [k for k in _KNOWN_PARAM_KEYS if k in params_raw]
    ordered += [k for k in params_raw if k not in ordered]
    params = {k: {"value": params_raw[k], "source": src} for k in ordered
              if isinstance(params_raw[k], (int, float, str, bool))}
    metrics = {k: data[k] for k in ("rmse", "mae", "r2") if isinstance(data.get(k), (int, float))}
    return {
        "model_type": data.get("model_type"),
        "target_column": data.get("target_column"),
        "params": params,
        "metrics": metrics,
        "source": src,
    }


def _phase_attrs(text: str, report_rel: str) -> list[dict[str, Any]]:
    """報告本文から ``フェーズX … 工数：a-b h`` を行単位で抽出（範囲は範囲のまま）。"""
    phases: list[dict[str, Any]] = []
    page = 0
    cur: dict[str, Any] | None = None
    for raw in text.splitlines():
        line = _z2h(raw)
        pm = _PAGE_MARK.search(line)
        if pm:
            page = int(pm.group(1))
        head = _PHASE_HEAD.search(line)
        if head:
            cur = {"name": f"フェーズ{head.group(1)}", "key": head.group(1).upper(),
                   "hours": None, "source": {"doc_id": report_rel, "page": page}}
            phases.append(cur)
        if cur is not None and _KOSU_LINE.search(line):
            mr = _HOURS_RANGE.search(line)
            ms = _HOURS_SINGLE.search(line)
            if mr:
                lo, hi = int(mr.group(1)), int(mr.group(2))
                cur["hours"] = {"low": lo, "high": hi, "raw": mr.group(0)}
            elif ms:
                v = int(ms.group(1))
                cur["hours"] = {"low": v, "high": v, "raw": ms.group(0)}
    # 重複 key（同一フェーズが複数ページに現れる等）は最初の工数付きを残す。
    dedup: dict[str, dict[str, Any]] = {}
    for p in phases:
        k = p["key"]
        if k not in dedup or (dedup[k]["hours"] is None and p["hours"] is not None):
            dedup[k] = p
    return [dedup[k] for k in sorted(dedup)]


def _train_ref(project: str, refs: Sequence[FileRef]) -> FileRef | None:
    for ext in ("csv", "xlsx", "tsv"):
        for r in refs:
            if (r.project == project and r.ext == ext
                    and nfc(r.name).lower().startswith("train")
                    and "analysis_project" not in r.rel):
                return r
    return None


def _feature_attrs(project: str, refs: Sequence[FileRef], text: str, report_rel: str,
                   target_column: str | None) -> dict[str, Any] | None:
    """報告本文が「影響が高い」と宣言した train 列 × ターゲット Pearson 相関（決定論）。"""
    tref = _train_ref(project, refs)
    if tref is None:
        return None
    try:
        from src.rag.tools.compute_sandbox import _read_frame
        df, _title, _tr = _read_frame(tref, None)
    except Exception:  # noqa: BLE001
        return None
    import pandas as pd

    cols = [nfc(str(c)) for c in df.columns]
    target = nfc(target_column) if target_column else None
    if not target or target not in cols:
        return None
    # 本文（NFKC 正規化）に現れた列名で、影響キュー近傍のもの＝高影響特徴量。
    lines = [_z2h(l) for l in text.splitlines()]
    feature_cols = [c for c in cols if c != target]
    impact_lines = [i for i, l in enumerate(lines) if _IMPACT_CUE.search(l)]
    high: list[str] = []
    for c in feature_cols:
        for i in impact_lines:
            window = " ".join(lines[max(0, i - 2): i + 2])
            if c and c.lower() in window.lower():
                if c not in high:
                    high.append(c)
                break
    if not high:
        return None
    # 相関（|Pearson|）を高影響特徴量について決定論計算。
    corrs: dict[str, dict[str, Any]] = {}
    tser = pd.to_numeric(df[target], errors="coerce")
    for c in high:
        if c not in df.columns:
            continue
        try:
            r = pd.to_numeric(df[c], errors="coerce").corr(tser)
        except Exception:  # noqa: BLE001
            continue
        if r is None or pd.isna(r):
            continue
        corrs[c] = {"value": round(float(r), 6), "abs": round(abs(float(r)), 6),
                    "source": {"doc_id": nfc(tref.rel), "target": target}}
    if not corrs:
        return None
    return {
        "high_impact_features": {"value": high, "source": {"doc_id": report_rel}},
        "target_correlations": corrs,
        "target_column": target,
    }


def compute_case(project: str, refs: Sequence[FileRef]) -> dict[str, Any] | None:
    """1案件の報告/分析から数値属性レコードを組む（無ければ None — 欠測を偽装しない）。"""
    reports = _report_refs(project, refs)
    model = _model_attrs(project, refs)
    phases: list[dict[str, Any]] = []
    features: dict[str, Any] | None = None
    report_rel = nfc(reports[0].rel) if reports else ""
    # 主要報告書のテキスト（フェーズ工数 / 高影響特徴量）。
    for rref in reports:
        text = _report_text(rref)
        if not text.strip():
            continue
        if not phases:
            ph = _phase_attrs(text, nfc(rref.rel))
            if ph:
                phases = ph
                report_rel = nfc(rref.rel)
        if features is None:
            tgt = (model or {}).get("target_column")
            feat_rec = _feature_attrs(project, refs, text, nfc(rref.rel), tgt)
            if feat_rec is not None:
                features = feat_rec
    if not (model or phases or features):
        return None
    rec: dict[str, Any] = {"case_id": nfc(project), "report_doc": report_rel}
    if model is not None:
        rec["model"] = model
    if phases:
        rec["phases"] = phases
    if features is not None:
        rec["features"] = features
    return rec


# --------------------------------------------------------------------------- universe / build / io
def _case_universe(refs: Sequence[FileRef]) -> list[str]:
    """報告/調査 doc または分析 metrics.json を持つ全案件（コーパス順・重複排除）。"""
    seen: list[str] = []
    for r in refs:
        if not r.project or r.project in seen:
            continue
        name = nfc(r.name)
        is_doc = r.ext in _DOC_EXTS and (r.category == "report" or _DOC_NAME_CUE.search(name))
        is_metrics = r.ext == "json" and name == "metrics.json"
        if is_doc or is_metrics:
            seen.append(r.project)
    return seen


def write_store(records: Sequence[Mapping[str, Any]], path: Path | None = None) -> dict[str, int]:
    """Atomically write a reproducible JSONL store (schema header + case_id-sorted rows)."""
    out = Path(path) if path is not None else default_out_path()
    out.parent.mkdir(parents=True, exist_ok=True)
    ordered = sorted(records, key=lambda r: r.get("case_id", ""))
    tmp = out.with_suffix(out.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8", newline="\n") as handle:
        handle.write(json.dumps({"schema": SCHEMA, "version": SCHEMA_VERSION},
                                ensure_ascii=False, sort_keys=True) + "\n")
        for rec in ordered:
            handle.write(json.dumps(rec, ensure_ascii=False, sort_keys=True) + "\n")
    tmp.replace(out)
    reset_cache()
    return {"cases": len(ordered)}


def build(refs: Sequence[FileRef] | None = None, *, out: Path | None = None,
          write_report: bool = True) -> dict[str, Any]:
    """全案件の報告/分析を走査して数値属性ストアを焼く（質問非依存・LLM フリー・fail-open）。"""
    refs = list(refs) if refs is not None else list(walk())
    universe = _case_universe(refs)
    records: list[dict[str, Any]] = []
    skipped: list[str] = []
    for project in universe:
        try:
            rec = compute_case(project, refs)
        except Exception:  # noqa: BLE001 — one bad case must not sink the build
            rec = None
        if rec is not None:
            records.append(rec)
        else:
            skipped.append(project)
    stats = write_store(records, out)
    report = {
        "schema": SCHEMA, "version": SCHEMA_VERSION,
        "universe": len(universe), "records": len(records), "skipped": skipped,
        "with_model": sum(1 for r in records if "model" in r),
        "with_phases": sum(1 for r in records if r.get("phases")),
        "with_features": sum(1 for r in records if "features" in r),
    }
    if write_report:
        rp = default_report_path()
        rp.parent.mkdir(parents=True, exist_ok=True)
        rp.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return {"records": len(records), "cases": stats["cases"], "report": report}


# --------------------------------------------------------------------------- load / read (minimal API)
_LOAD_CACHE: dict[str, list[dict[str, Any]]] = {}


def reset_cache() -> None:
    _LOAD_CACHE.clear()


def load(path: Path | None = None) -> list[dict[str, Any]]:
    """Load the report-attr rows (memoized). ``[]`` when absent/unreadable/schema-mismatch (回帰ゼロ)."""
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


def case_record(case_hint: str, *, path: Path | None = None) -> dict[str, Any] | None:
    """The report-attr record for a 案件 (exact case_id preferred, else substring). ``None`` when unknown."""
    hint = nfc(case_hint).strip()
    rows = load(path)
    for row in rows:
        if nfc(row.get("case_id", "")) == hint:
            return row
    for row in rows:
        if hint and hint in nfc(row.get("case_id", "")):
            return row
    return None


if __name__ == "__main__":
    summary = build()
    print(json.dumps(summary["report"], ensure_ascii=False, indent=2))
