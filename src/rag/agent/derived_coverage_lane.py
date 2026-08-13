"""SOT-2679 — 派生メトリクス網羅拡張の serve 配線（決定論直答レーン + investigator ツール）.

:mod:`src.rag.index.derived_metrics` が build 時に質問非依存で焼いた派生量のうち、cycle6 K3 で compute 依存の
まま棄権していた 2 型を **決定論直答レーン** として接続する:

* **idx4 — 目的変数と相関が最も高い数値特徴量**（例: ひがし丘 train の charges と相関最大の数値特徴量）。
  ``correlations.with_target`` から id 的な列を除いた |corr| 最大の特徴量を返す。厳密な単独最大の時だけ確定し、
  拮抗（top と 2位の |r| が同値）なら None を返して LLM ループへ委ねる（precision-first）。
* **idx24 — 1つでも欠損値がある行数が最も多い案件（主略称）**。全案件の ``missing.missing_row_count``
  （canonical train 表・全列走査・独立検算）を読み、full-coverage（全案件に非 None の欠損行数）かつ厳密単独最大の
  時だけ、その案件の主略称（case_master の abbrev）を返す。SOT-2663 の「余剰予算 ad-hoc 列挙が誤案件『白峰』を
  確定させる」逆流は、ストア到達（= canonical 全案件の決定論比較）のみが安全経路であることの実証。full-coverage で
  ない／単独最大でない時は None（棄権）で、Incorrect を作らない（idx24 ガード: Missing→MATCH のみ許容）。

規律（fact_layer / visual_lane と同じ）:
* 一意・厳密に束縛できる時だけ ``{value, evidence, method}`` を返し、少しでも曖昧なら ``None`` を返して LLM
  ループへフォールバック（回答数を減らさない・wrong を増やさない, SOT-2601 の教訓）。
* Sonnet は fact-layer ツールを確実には選ばない（SOT-2649）ので回収の主経路は **決定論レーン**。ツール
  :func:`tool` は補助（別 idx / 別モデル）として同時に公開する。
* ``RAG_DERIVED_COVERAGE`` 既定 OFF ⇒ :func:`resolve` は None・:func:`tool` は None を返し、ツール集合・関数
  スキーマ・serve path は byte-identical。serve 中の追加 LLM 呼び出しは 0（事前計算済みを読むだけ）。
"""
from __future__ import annotations

import os
import re
import unicodedata
from typing import Any, Callable

from src.rag.index import derived_metrics as _dm
from src.rag.tools import contract as _contract

DERIVED_COVERAGE_LOOKUP = "derived_coverage_lookup"
_STR = {"type": "string"}
_ON = {"1", "true", "yes", "on"}

# 企業接頭辞（案件束縛時に stem を取り出す）。
_CORP_PREFIX = re.compile(r"^(株式会社|医療法人社団|医療法人|有限会社|合同会社|一般社団法人)\s*")

# --- idx4「目的変数と相関が最も高い数値特徴量」cues (NFKC + 空白除去 + 小文字化) --------------------
_CORR_CUE = re.compile(r"相関|corr(?:elation)?")
_HIGHEST_CUE = re.compile(r"最も高い|一番高い|最大|最も強い|最も相関|highest|最も大きい")
_FEATURE_CUE = re.compile(r"特徴量|説明変数|変数|カラム|feature")
# SOT-2688 (cycle7 K5, idx91) — 相関の符号修飾。負→ r<0 の厳密単独最小、正→ r>0 の厳密単独最大。
# 修飾なしは従来どおり |r| 最大（idx4 を保存）。RAG_CORR_SIGN OFF ⇒ 常に修飾なし＝byte-identical。
_NEG_CORR_CUE = re.compile(r"負の相関|負相関|マイナスの相関|マイナス相関|逆相関|negative")
_POS_CORR_CUE = re.compile(r"正の相関|正相関|プラスの相関|プラス相関|positive")

# --- idx24「1つでも欠損値がある行数が最も多い案件」cues -------------------------------------------
_MISSING_CUE = re.compile(r"欠損|欠測|missing|na値|null")
_ROWS_CUE = re.compile(r"行数|行が|行を|行の|レコード|rows?")
_MOST_CUE = re.compile(r"最も多い|一番多い|最大|多い(?:案件|企業|会社|プロジェクト)")
_CASE_CUE = re.compile(r"案件|企業|会社|プロジェクト")


def enabled() -> bool:
    """serve レーン／ツールを有効にするか。既定 OFF（``RAG_DERIVED_COVERAGE``）⇒ byte-identical。"""
    return os.getenv("RAG_DERIVED_COVERAGE", "0").strip().lower() in _ON


def corr_sign_enabled() -> bool:
    """SOT-2688 (cycle7 K5, idx91) — 相関レーンの符号対応を有効にするか。既定 OFF（``RAG_CORR_SIGN``）。

    OFF ⇒ :func:`_top_corr_lane` は符号修飾を一切見ず従来どおり |r| 最大のみ ⇒ idx4 も idx91 も
    現行と byte-identical（idx91 は wrong のまま）。ON かつ質問に負/正修飾がある時だけ符号対応が発火する。
    """
    return os.getenv("RAG_CORR_SIGN", "0").strip().lower() in _ON


def _norm(text: Any) -> str:
    return unicodedata.normalize("NFKC", str(text or "")).replace(" ", "").replace("　", "").lower()


def _result(value: Any, *, selection: str, evidence: dict[str, Any]) -> dict[str, Any]:
    ev = {"store": "derived_metrics", "provenance": "precomputed (independent dual verification)", **evidence}
    method = {"engine": "derived_metrics", "contract": "numeric", "selection": selection,
              "naturalize": False, "verified_operand": True, "confidence": 1.0}
    return _contract.ensure_contract({"value": value, "evidence": ev, "method": method})


# --------------------------------------------------------------------------- case binding
def _bind_unique_case(rows: list[dict[str, Any]], q: str) -> dict[str, Any] | None:
    """質問文に案件名（正式名 or 接頭辞除去 stem）が一意に現れる時だけ、その 1 レコードを返す。"""
    hits: list[dict[str, Any]] = []
    for r in rows:
        cid = _norm(r.get("case_id"))
        stem = _norm(_CORP_PREFIX.sub("", str(r.get("case_id") or "")))
        matched = False
        for t in {cid, stem}:
            if not t or len(t) < 4:
                continue
            # 正式名／stem がそのまま、または先頭 6/4 文字が質問に現れれば候補。
            if t in q or (len(t) >= 6 and t[:6] in q) or t[:4] in q:
                matched = True
                break
        if matched:
            hits.append(r)
    return hits[0] if len(hits) == 1 else None


# --------------------------------------------------------------------------- idx4 lane
def _top_corr_feature(rec: dict[str, Any], *, sign: "str | None" = None) -> "tuple[str, float, float] | None":
    """レコードの with_target を再走査し、id 的な列を除いた相関特徴量を **厳密単独** で返す。

    ``sign`` (SOT-2688):
    * ``None`` — 従来どおり |r| の **厳密単独最大**（top と 2 位の |r| が同値なら None）。
    * ``"neg"`` — r<0 に限定した **厳密単独最小 r**（最も強い負の相関）。負の相関が無ければ None。
    * ``"pos"`` — r>0 に限定した **厳密単独最大 r**（最も強い正の相関）。正の相関が無ければ None。
    いずれも拮抗（1・2 位が同値）なら None（precision-first — 単独でなければ確定しない）。
    """
    corr = rec.get("correlations") or {}
    with_target = corr.get("with_target") or {}
    target = corr.get("target")
    signed: list[tuple[float, str]] = []  # (r, col) — id 的な列/target を除いた符号付き相関
    for col, r in with_target.items():
        if r is None:
            continue
        try:
            rv = float(r)
        except (TypeError, ValueError):
            continue
        if rv != rv:  # NaN
            continue
        cl = str(col).strip().lower()
        if cl in _dm._ID_LIKE or col == target:
            continue
        signed.append((rv, col))
    if not signed:
        return None
    if sign == "neg":
        cand = [(rv, col) for rv, col in signed if rv < 0]
        if not cand:
            return None
        cand.sort(key=lambda x: x[0])                       # 最も負（昇順）が先頭
        if len(cand) >= 2 and cand[0][0] == cand[1][0]:      # 拮抗 ⇒ 単独最小でない
            return None
        rv, col = cand[0]
        return (abs(rv), col, rv)
    if sign == "pos":
        cand = [(rv, col) for rv, col in signed if rv > 0]
        if not cand:
            return None
        cand.sort(key=lambda x: x[0], reverse=True)          # 最も正（降順）が先頭
        if len(cand) >= 2 and cand[0][0] == cand[1][0]:      # 拮抗 ⇒ 単独最大でない
            return None
        rv, col = cand[0]
        return (abs(rv), col, rv)
    ranked = sorted(((abs(rv), col, rv) for rv, col in signed), key=lambda x: x[0], reverse=True)
    if len(ranked) >= 2 and ranked[0][0] == ranked[1][0]:    # |r| 拮抗 ⇒ 単独最大でない
        return None
    return ranked[0]


def _corr_sign(q: str) -> "str | None":
    """質問文（正規化済み）から相関の符号修飾を判定。RAG_CORR_SIGN OFF なら常に None（byte-identical）。"""
    if not corr_sign_enabled():
        return None
    if _NEG_CORR_CUE.search(q):
        return "neg"
    if _POS_CORR_CUE.search(q):
        return "pos"
    return None


def _top_corr_lane(question_raw: str, rows: list[dict[str, Any]]):
    q = _norm(question_raw)
    if not (_CORR_CUE.search(q) and _HIGHEST_CUE.search(q) and _FEATURE_CUE.search(q)):
        return None
    rec = _bind_unique_case(rows, q)
    if rec is None:
        return None
    sign = _corr_sign(q)
    top = _top_corr_feature(rec, sign=sign)
    if top is None:
        return None
    _abs_r, feature, r = top
    corr = rec.get("correlations") or {}
    basis = {
        None: "correlations.with_target |r| 単独最大（id 的な列は特徴量から除外）",
        "neg": "correlations.with_target r<0 の厳密単独最小（最も強い負の相関・id 的な列は除外）",
        "pos": "correlations.with_target r>0 の厳密単独最大（最も強い正の相関・id 的な列は除外）",
    }[sign]
    evidence = {
        "case": rec.get("case_id"), "train_file": (rec.get("sources") or {}).get("train_file"),
        "target": corr.get("target"), "feature": feature, "pearson_r": r, "sign": sign or "abs",
        "basis": basis,
    }
    return _result(feature, selection="top_correlated_feature", evidence=evidence)


# --------------------------------------------------------------------------- idx24 lane
def _abbrev_for(case_id: str) -> str | None:
    """case_master の abbrev（社内用語集由来の 主略称）を case_id 一致で引く。無ければ None。"""
    try:
        from src.rag.index import case_master
        for r in case_master.load():
            if _norm(r.get("case_id")) == _norm(case_id):
                abbrev = r.get("abbrev")
                if not abbrev:
                    abbrev = ((r.get("attributes") or {}).get("abbrev") or {}).get("value")
                return abbrev or None
    except Exception:  # noqa: BLE001 — 主略称解決不能でも欠測を偽装しない（呼び出し側で None を扱う）
        return None
    return None


def _missing_argmax_lane(question_raw: str, rows: list[dict[str, Any]]):
    q = _norm(question_raw)
    if not (_MISSING_CUE.search(q) and _MOST_CUE.search(q)
            and _ROWS_CUE.search(q) and _CASE_CUE.search(q)):
        return None
    # full-coverage certificate: 全案件に非 None の欠損行数がある時だけ argmax を確定する（部分母集団で
    # 答えない — 1 件でも欠測があれば真の最大を取り違えうる, precision-first）。
    counts: list[tuple[int, dict[str, Any]]] = []
    for r in rows:
        mrc = ((r.get("missing") or {}).get("missing_row_count"))
        if not isinstance(mrc, int):
            return None
        counts.append((mrc, r))
    if len(counts) < 2:
        return None
    counts.sort(key=lambda x: x[0], reverse=True)
    if counts[0][0] == counts[1][0]:  # 拮抗（単独最大でない）⇒ defer
        return None
    top_count, rec = counts[0]
    case_id = rec.get("case_id") or ""
    abbrev = _abbrev_for(case_id)
    value = abbrev or case_id  # 主略称を優先（無ければ正式名 — 誤案件は返さない）
    if not value:
        return None
    evidence = {
        "case": case_id, "abbrev": abbrev, "missing_row_count": top_count,
        "runner_up": {"case": counts[1][1].get("case_id"), "missing_row_count": counts[1][0]},
        "universe_cases": len(counts),
        "train_file": (rec.get("sources") or {}).get("train_file"),
        "basis": ("canonical train 表（03.データ／analysis_project 除外）× 全列走査の欠損行数を全案件で"
                  "事前計算し、その厳密単独最大（母集団=全案件で確定）。"),
        "completeness": "全案件に missing_row_count 充足（母集団確定）",
    }
    return _result(value, selection="cross_case_missing_row_argmax", evidence=evidence)


# --------------------------------------------------------------------------- serve entry
def resolve(question: str) -> "dict[str, Any] | None":
    """idx4/idx24 の決定論直答（束縛できれば contract、曖昧なら None）。OFF なら常に None（byte-identical）。"""
    if not enabled():
        return None
    try:
        rows = _dm.load()
    except Exception:  # noqa: BLE001 — 壊れたストアは fall back、答えパスを壊さない
        return None
    if not rows:
        return None
    try:
        result = _missing_argmax_lane(question, rows)
        if result is None:
            result = _top_corr_lane(question, rows)
    except Exception:  # noqa: BLE001
        return None
    if result is None or not _contract.is_contract(result):
        return None
    normalized = _contract.ensure_contract(result)
    return normalized if normalized.get("value") is not None else None


# --------------------------------------------------------------------------- investigator tool (補助)
def _tool_handler(case: str = "", metric: str = "") -> dict[str, Any]:
    """派生メトリクス網羅拡張の lookup（補助ツール）。

    * ``metric`` に 'missing'/'欠損'/'ranking' を含む ⇒ 全案件の欠損行数ランキング（主略称付き, 降順）。
    * ``case`` 指定 ⇒ その案件の top_correlated_feature と欠損行数ブロックを返す。
    """
    rows = _dm.load()
    m = _norm(metric)
    want_ranking = (not case) or ("missing" in m) or ("欠損" in m) or ("ranking" in m) or ("行数" in m)
    if want_ranking and (not case):
        ranked = []
        for r in rows:
            mrc = ((r.get("missing") or {}).get("missing_row_count"))
            ranked.append({"case_id": r.get("case_id"), "abbrev": _abbrev_for(r.get("case_id") or ""),
                           "missing_row_count": mrc,
                           "row_count": (r.get("missing") or {}).get("row_count")})
        ranked.sort(key=lambda x: (x["missing_row_count"] is None, -(x["missing_row_count"] or 0)))
        return _contract.make(ranked, engine="derived_metrics",
                              evidence={"applicable": True, "kind": "missing_row_ranking",
                                        "universe_cases": len(ranked)},
                              scheme="derived-metrics-missing-ranking")
    rec = _dm.case_metrics(case)
    if rec is None:
        return _contract.make(None, engine="derived_metrics",
                              evidence={"applicable": True, "case": case, "found": False},
                              note="該当案件なし")
    corr = rec.get("correlations") or {}
    value = {"case_id": rec.get("case_id"), "top_correlated_feature": corr.get("top_feature"),
             "target": corr.get("target"), "missing": rec.get("missing")}
    return _contract.make(value, engine="derived_metrics",
                          evidence={"applicable": True, "case": case, "found": True,
                                    "train_file": (rec.get("sources") or {}).get("train_file")},
                          scheme="derived-metrics-coverage")


def tool() -> "tuple[str, str, dict[str, Any], Callable[..., Any]] | None":
    """RAG_DERIVED_COVERAGE OFF ⇒ None ⇒ ツール集合/関数スキーマ/MCP surface は byte-identical。"""
    if not enabled():
        return None
    return (
        DERIVED_COVERAGE_LOOKUP,
        "派生メトリクス網羅: (1)『目的変数と相関が最も高い数値特徴量』は case を渡すとその案件の "
        "top_correlated_feature を出典付きで返す。(2)『欠損値を1つでも含む行数が最も多い案件』は metric に "
        "'missing_ranking' 等を渡すと全案件の欠損行数ランキング（主略称付き, 降順）を返す（canonical train 表・"
        "全列走査・独立検算の事前計算値）。実行時 compute の当て推量反復の代替。",
        {"type": "object", "properties": {"case": _STR, "metric": _STR}, "required": []},
        _tool_handler,
    )
