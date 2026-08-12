"""SOT-2655 — 報告数値属性ストアの serve-path 配線（決定論レーン + investigator ツール）.

:mod:`src.rag.index.report_attr_store` が build 時に質問非依存で焼いた案件×文書の数値属性
（最良モデルのハイパーパラメータ / 将来フェーズ工数 / 高影響特徴量×相関）を、NUMERIC contract の
**決定論直答レーン**として answer path に接続する。fact_layer と同じ精度優先の規律:

* 一意に束縛できる時だけ ``{value, evidence, method}`` を返し、少しでも曖昧なら ``None`` を返して
  LLM ループへフォールバック（回答数を減らさない・wrong を増やさない, SOT-2601 の教訓）。
* Sonnet は fact-layer ツールを確実には選ばないため（SOT-2649 の教訓）、回収の主経路は **決定論レーン**。
  ツール :func:`tool` は補助（二次 idx / 別モデル）として同時に公開する。
* ``RAG_REPORT_ATTR_STORE`` 既定 OFF ⇒ :func:`resolve` は None・:func:`tool` は None を返し、
  ツール集合・関数スキーマ・serve path は byte-identical。serve 中 genai 呼び出しは 0（事前計算済みを読むだけ）。
"""
from __future__ import annotations

import re
import unicodedata
from typing import Any, Callable, Mapping

from src.rag.index import report_attr_store as _store
from src.rag.tools import contract as _contract

# --- question anchors (case-folded, NFKC-normalized question) -----------------------------------
_MODEL_CUE = re.compile(r"モデル|パラメータ|ハイパーパラメータ|param")
_PHASE_CUE = re.compile(r"フェーズ|phase")
_KOSU_CUE = re.compile(r"工数|時間|hours?")
_SUM_CUE = re.compile(r"合計|合わせ|合算|足し|トータル|総計|総工数|全体で")
_PHASE_LETTER = re.compile(r"フェーズ\s*([A-Za-z0-9])")
_FEATURE_CUE = re.compile(r"特徴量|変数|feature")
_CORR_CUE = re.compile(r"相関")
_MAX_CUE = re.compile(r"最も|最大|一番|トップ|highest|most")
_HIGH_CUE = re.compile(r"高い|大きい|強い")
_IMPACT_CUE = re.compile(r"影響|重要|寄与")

_CORP_PREFIX = re.compile(r"^(株式会社|医療法人社団|一般社団法人|有限会社|合同会社|合資会社)\s*")


def enabled() -> bool:
    return _store.enabled()


def _norm(text: Any) -> str:
    """NFKC + whitespace-strip + case-fold（全角英数/空白/ケースを吸収して安定に突合）。"""
    return unicodedata.normalize("NFKC", str(text or "")).replace(" ", "").replace("　", "").lower()


def _case_tokens(case_id: str) -> list[str]:
    """案件名の識別トークン（法人格除去 stem と 4 文字接頭 — 質問中の言及に一致させる）。"""
    cid = _norm(case_id)
    stem = _norm(_CORP_PREFIX.sub("", str(case_id or "")))
    toks: list[str] = []
    for t in (stem, cid, stem[:4] if len(stem) >= 4 else ""):
        if t and t not in toks:
            toks.append(t)
    return toks


def _bind_case(qn: str, rows: list[dict[str, Any]],
               require: Callable[[dict[str, Any]], bool]) -> "dict[str, Any] | None":
    """質問がただ一つの案件を名指す時だけその record を返す（複数一致/不一致は None = 精度優先）。"""
    hits: list[tuple[dict[str, Any], int]] = []
    for r in rows:
        if not require(r):
            continue
        best = 0
        for t in _case_tokens(r.get("case_id", "")):
            if t and t in qn:
                best = max(best, len(t))
        if best:
            hits.append((r, best))
    if not hits:
        return None
    if len(hits) == 1:
        return hits[0][0]
    # 複数一致: 最長トークン一致が一意ならそれ、同点なら曖昧 ⇒ None。
    top = max(h[1] for h in hits)
    top_rows = [r for r, b in hits if b == top]
    return top_rows[0] if len(top_rows) == 1 else None


def _result(value: Any, *, selection: str, evidence: dict[str, Any]) -> dict[str, Any]:
    ev = {"store": "report_attr", "provenance": "precomputed (question-independent)", **evidence}
    method = {"engine": "report_attr", "contract": "numeric", "selection": selection,
              "naturalize": False, "verified_operand": True, "confidence": 1.0}
    return {"value": value, "evidence": ev, "method": method}


# --------------------------------------------------------------------------- deterministic lanes
def _model_param_lane(qn: str, rows: list[dict[str, Any]]) -> "dict[str, Any] | None":
    """最良/最終モデルの単一ハイパーパラメータ（例: max_depth）を出典付きで返す。"""
    if not _MODEL_CUE.search(qn):
        return None
    rec = _bind_case(qn, rows, lambda r: bool(r.get("model", {}).get("params")))
    if rec is None:
        return None
    params = rec["model"]["params"]
    # 質問に literal 出現するパラメータ名（アンダースコア込み）で、ちょうど 1 つ一致する時だけ発火。
    matched = [k for k in params if _norm(k) and _norm(k) in qn]
    if len(matched) != 1:
        return None
    key = matched[0]
    cell = params[key]
    val = cell.get("value")
    if val is None:
        return None
    return _result(
        val, selection="model_hyperparameter",
        evidence={"case": rec.get("case_id"), "model_type": rec["model"].get("model_type"),
                  "param": key, "source": cell.get("source"),
                  "metrics": rec["model"].get("metrics")})


def _phase_sum_lane(qn: str, rows: list[dict[str, Any]]) -> "dict[str, Any] | None":
    """質問が名指すフェーズ群の想定工数合計（範囲は範囲のまま合算）を「N〜M時間」で返す。"""
    if not (_PHASE_CUE.search(qn) and _KOSU_CUE.search(qn) and _SUM_CUE.search(qn)):
        return None
    letters = {m.group(1).upper() for m in _PHASE_LETTER.finditer(qn)}
    if len(letters) < 2:
        return None  # 合計は 2 フェーズ以上を名指す時のみ（単一フェーズは lookup で足りる）
    rec = _bind_case(qn, rows, lambda r: bool(r.get("phases")))
    if rec is None:
        return None
    by_key = {p.get("key"): p for p in rec["phases"]}
    # 完全被覆: 名指した全フェーズが工数付きで存在する時のみ決定論合算（1 つでも欠ければ defer）。
    picked = []
    lo = hi = 0
    for k in sorted(letters):
        p = by_key.get(k)
        if p is None or not isinstance(p.get("hours"), Mapping):
            return None
        lo += int(p["hours"]["low"])
        hi += int(p["hours"]["high"])
        picked.append(p)
    value = f"{lo}時間" if lo == hi else f"{lo}〜{hi}時間"
    return _result(
        value, selection="phase_effort_sum",
        evidence={"case": rec.get("case_id"), "phases": sorted(letters),
                  "components": [{"key": p["key"], "hours": p["hours"], "source": p.get("source")}
                                for p in picked],
                  "sum_low": lo, "sum_high": hi})


def _feature_corr_lane(qn: str, rows: list[dict[str, Any]]) -> "dict[str, Any] | None":
    """報告が「影響が高い」とした特徴量のうち、ターゲット相関が最大の特徴量名を返す。"""
    if not (_FEATURE_CUE.search(qn) and _CORR_CUE.search(qn) and _MAX_CUE.search(qn)
            and _HIGH_CUE.search(qn) and _IMPACT_CUE.search(qn)):
        return None
    rec = _bind_case(qn, rows, lambda r: bool(r.get("features", {}).get("high_impact_features")))
    if rec is None:
        return None
    feats = rec["features"]
    high = feats.get("high_impact_features", {}).get("value") or []
    corrs = feats.get("target_correlations") or {}
    scored = [(f, corrs[f]["abs"]) for f in high if f in corrs and corrs[f].get("abs") is not None]
    if not scored:
        return None
    scored.sort(key=lambda x: x[1], reverse=True)
    # 曖昧回避: 上位 2 件が同点なら決定論に選べない ⇒ defer。
    if len(scored) >= 2 and scored[0][1] == scored[1][1]:
        return None
    best = scored[0][0]
    return _result(
        best, selection="feature_max_target_correlation",
        evidence={"case": rec.get("case_id"), "high_impact_features": high,
                  "correlations": {f: corrs[f]["value"] for f, _ in scored},
                  "target": feats.get("target_column"),
                  "source": feats.get("high_impact_features", {}).get("source")})


_LANES = (_model_param_lane, _phase_sum_lane, _feature_corr_lane)


def resolve(question: str) -> "dict[str, Any] | None":
    """NUMERIC 質問を報告数値属性ストアで一意に束縛できる時だけ contract を返す（fail-open）。"""
    if not enabled():
        return None
    try:
        rows = _store.load()
        if not rows:
            return None
        qn = _norm(question)
        for lane in _LANES:
            res = lane(qn, rows)
            if res is not None and _contract.is_contract(res) and res.get("value") is not None:
                return _contract.ensure_contract(res)
    except Exception:  # noqa: BLE001 — a broken lane must fall back, never break the answer path
        return None
    return None


# --------------------------------------------------------------------------- investigator tool (補助)
REPORT_ATTR_LOOKUP = "report_attr_lookup"
_STR = {"type": "string"}


def _report_attr_lookup(case: str, attribute: str = "") -> dict[str, Any]:
    """報告数値属性ストアを引く: case で案件を特定し、attribute（例 'max_depth' / 'phases' /
    'high_impact_features' / 'correlations'）を出典付きで返す。空 attribute なら利用可能属性を列挙。"""
    rec = _store.case_record(case)
    if rec is None:
        return _contract.make(None, engine="report_attr",
                              evidence={"applicable": True, "case": case, "found": False},
                              note="該当案件なし")
    model = rec.get("model") or {}
    params = model.get("params") or {}
    feats = rec.get("features") or {}
    attr = unicodedata.normalize("NFKC", str(attribute or "")).strip()
    if not attr:
        available = (["model_type", "metrics", "phases", "high_impact_features", "correlations"]
                     + [f"param:{k}" for k in params])
        return _contract.make(
            {"case_id": rec.get("case_id"), "available": available}, engine="report_attr",
            evidence={"applicable": True, "case": case, "found": True}, scheme="report-attr-index")
    low = attr.lower()
    value: Any = None
    ev: dict[str, Any] = {"applicable": True, "case": rec.get("case_id"), "attribute": attr}
    if low in ("phases", "phase", "工数", "フェーズ"):
        value = rec.get("phases")
    elif low in ("high_impact_features", "features", "特徴量"):
        value = feats.get("high_impact_features", {}).get("value")
    elif low in ("correlations", "correlation", "相関"):
        value = {k: v.get("value") for k, v in (feats.get("target_correlations") or {}).items()}
    elif low in ("model_type", "model"):
        value = model.get("model_type")
    elif low in ("metrics",):
        value = model.get("metrics")
    else:
        key = low[6:] if low.startswith("param:") else attr
        cell = params.get(key)
        if cell is not None:
            value = cell.get("value")
            ev["source"] = cell.get("source")
    return _contract.make(value, engine="report_attr",
                          evidence={**ev, "resolved": value is not None}, scheme="report-attr-lookup")


def tool() -> "tuple[str, str, dict[str, Any], Callable[..., Any]] | None":
    """報告数値属性 lookup ツール（OFF の時 None ⇒ ツール集合/MCP surface は byte-identical）。"""
    if not enabled():
        return None
    return (
        REPORT_ATTR_LOOKUP,
        "報告数値属性ストア: 案件の最終報告/分析から事前計算した最良モデルのハイパーパラメータ表・"
        "将来フェーズの想定工数・高影響特徴量とターゲット相関を出典付きで引く。case に案件名(一部可)、"
        "attribute に 'max_depth'/'phases'/'high_impact_features'/'correlations' 等を渡す(空なら属性一覧)。"
        "スキャンPDFの本文は事前OCR済みなので file_grep 反復は不要。",
        {"type": "object", "properties": {"case": _STR, "attribute": _STR}, "required": ["case"]},
        _report_attr_lookup,
    )
