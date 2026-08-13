"""SOT-2699 — 統計表 rank/ratio 直答レーン（cycle9, idx99）.

:mod:`src.rag.index.derived_ranking_store` が build 時に質問非依存で焼いた「統計表の header 別数値系列
＋昇順/降順ソート」を読み、序数（最も高い/低い・N番目に高い/低い）× 比（何倍）/差 の派生計算質問を
**決定論直答** する。

回収する型（idx99）:
    「<案件> の <調査> において、<metric> が最も高い <entity> の <metric> は、N番目に低い <entity> の
     <metric> の何倍ですか。小数第2位まで求めてください。」
    例: みなみ野 糖尿病統計 死亡率 最高(青森 18.2) ÷ 4番目に低い(滋賀 7.3) = 2.49。

規律（fact_layer / derived_coverage_lane と同じ precision-first）:
* 案件が一意束縛でき、質問の metric キーワードに一致する系列がその案件で **ちょうど1つ**、序数が
  厳密にパースでき、丸め指定があり、二重検算が一致した時だけ ``{value, evidence, method}`` を返す。
  少しでも曖昧なら ``None`` を返し LLM ループへフォールバック（回答数を減らさない・wrong を増やさない）。
* ``RAG_DERIVED_RANKING`` 既定 OFF ⇒ :func:`resolve`/:func:`tool` は None、serve path は byte-identical。
  serve 中の追加 LLM 呼び出しは 0（事前計算済みを読むだけ）。
"""
from __future__ import annotations

import os
import re
import unicodedata
from typing import Any, Callable

from src.rag.index import derived_ranking_store as _store
from src.rag.tools import contract as _contract

DERIVED_RANKING_LOOKUP = "derived_ranking_lookup"
_STR = {"type": "string"}
_ON = {"1", "true", "yes", "on"}

_CORP_PREFIX = re.compile(r"^(株式会社|医療法人社団|医療法人|一般社団法人|有限会社|合同会社|社会福祉法人)\s*")

# 序数（漢数字 → int）。
_KANJI_NUM = {"一": 1, "二": 2, "三": 3, "四": 4, "五": 5, "六": 6, "七": 7, "八": 8, "九": 9, "十": 10}
# rank1（最上位/最下位）と N番目。降順(高い/大きい)・昇順(低い/小さい)。
_TOP_DESC = re.compile(r"最も高い|一番高い|最も大きい|一番大きい|最大")
_TOP_ASC = re.compile(r"最も低い|一番低い|最も小さい|一番小さい|最小")
_NTH_DESC = re.compile(r"(\d+|[一二三四五六七八九十]+)番目に(?:高い|大きい)")
_NTH_ASC = re.compile(r"(\d+|[一二三四五六七八九十]+)番目に(?:低い|小さい)")
# 演算子。
_RATIO_CUE = re.compile(r"何倍|倍(?:です|でしょう|ですか|になり)?")
# 丸め指定「小数第N位」。
_ROUND_RE = re.compile(r"小数第(\d+|[一二三四五六七八九十]+)位")


def enabled() -> bool:
    """serve レーン／ツールを有効にするか。既定 OFF（``RAG_DERIVED_RANKING``）⇒ byte-identical。"""
    return os.getenv("RAG_DERIVED_RANKING", "0").strip().lower() in _ON


def _norm(text: Any) -> str:
    return unicodedata.normalize("NFKC", str(text or "")).replace(" ", "").replace("　", "").lower()


def _int_of(token: str) -> int | None:
    token = token.strip()
    if token.isdigit():
        return int(token)
    if token in _KANJI_NUM:
        return _KANJI_NUM[token]
    return None


def _result(value: Any, *, selection: str, evidence: dict[str, Any]) -> dict[str, Any]:
    ev = {"store": "derived_ranking", "provenance": "precomputed (question-independent, dual-verified)",
          **evidence}
    method = {"engine": "derived_ranking", "contract": "numeric", "selection": selection,
              "naturalize": False, "verified_operand": True, "confidence": 1.0}
    return _contract.ensure_contract({"value": value, "evidence": ev, "method": method})


# --------------------------------------------------------------------------- case binding
def _bind_case(by_project: dict[str, list[dict[str, Any]]], q: str) -> str | None:
    """質問文に案件名（正式名 / 接頭辞除去 stem / 末尾セグメント）が一意に現れる時だけ project を返す。"""
    hits: list[str] = []
    for project in by_project:
        tokens = set()
        full = _norm(project)
        if len(full) >= 4:
            tokens.add(full)
        stem = _norm(_CORP_PREFIX.sub("", project))
        if len(stem) >= 4:
            tokens.add(stem)
        # 空白区切りの各セグメント（「医療法人社団 蒼樹会 みなみ野女性医療センター」→ 末尾センター名）。
        for seg in str(project).split():
            ns = _norm(seg)
            if len(ns) >= 4:
                tokens.add(ns)
        if any(t in q for t in tokens):
            hits.append(project)
    return hits[0] if len(hits) == 1 else None


# --------------------------------------------------------------------------- metric-series binding
def _bind_series(series: list[dict[str, Any]], q: str) -> dict[str, Any] | None:
    """質問の metric キーワードに header-core が一致する系列を **一意** に束縛（複数一致は defer）。

    照合は「metric が助詞（が/は/の/を）に直接続く」出現に限る。単なる部分一致だと案件名の一部
    （例: 「みなみ野女性医療センター」の『女性』が『女性』列見出しに一致）を metric と誤認しうるため、
    ランキング文脈で metric として実際に使われている語（『死亡率が最も高い』『死亡率の何倍』）だけを拾う
    （precision-first）。
    """
    matched: list[dict[str, Any]] = []
    for s in series:
        mk = s.get("metric_key")
        if not mk or s.get("n", 0) < _store.MIN_SERIES_VALUES:
            continue
        if re.search(re.escape(mk) + r"[がはのを]", q):
            matched.append(s)
    # 同一 metric_key の系列が複数（別表）ある時も defer。厳密に 1 系列のみ確定。
    if len({s["metric_key"] for s in matched}) != 1 or len(matched) != 1:
        return None
    return matched[0]


# --------------------------------------------------------------------------- ordinal parsing
def _parse_operands(q: str) -> "list[tuple[int, str, int]] | None":
    """質問から (rank, direction, position) の序数列を出現順に抽出する。

    direction ∈ {"desc","asc"}。position は元質問中の出現オフセット（演算子解釈の順序決定に使う）。
    """
    ops: list[tuple[int, str, int]] = []
    for m in _TOP_DESC.finditer(q):
        ops.append((1, "desc", m.start()))
    for m in _TOP_ASC.finditer(q):
        ops.append((1, "asc", m.start()))
    for m in _NTH_DESC.finditer(q):
        n = _int_of(m.group(1))
        if n:
            ops.append((n, "desc", m.start()))
    for m in _NTH_ASC.finditer(q):
        n = _int_of(m.group(1))
        if n:
            ops.append((n, "asc", m.start()))
    if not ops:
        return None
    ops.sort(key=lambda x: x[2])
    return ops


def _value_at(series: dict[str, Any], rank: int, direction: str) -> "dict[str, Any] | None":
    arr = series["sorted_desc"] if direction == "desc" else series["sorted_asc"]
    if rank < 1 or rank > len(arr):
        return None
    return arr[rank - 1]


# --------------------------------------------------------------------------- ratio lane
def _ratio_lane(question_raw: str, by_project: dict[str, list[dict[str, Any]]]):
    """序数2つの比（A/B 何倍）を丸め指定つきで直答（idx99 型）。"""
    q = _norm(question_raw)
    if not _RATIO_CUE.search(q):
        return None
    rm = _ROUND_RE.search(q)
    if not rm:
        return None  # 丸め指定なしは format 曖昧 ⇒ defer（precision-first）
    decimals = _int_of(rm.group(1))
    if decimals is None:
        return None
    project = _bind_case(by_project, q)
    if project is None:
        return None
    series = _bind_series(by_project.get(project) or [], q)
    if series is None:
        return None
    ops = _parse_operands(q)
    if ops is None or len(ops) < 2:
        return None
    # 出現順で最初の2序数を A(分子)/B(分母) とする（「A は B の何倍」）。
    a_rank, a_dir, _ = ops[0]
    b_rank, b_dir, _ = ops[1]
    a = _value_at(series, a_rank, a_dir)
    b = _value_at(series, b_rank, b_dir)
    if a is None or b is None:
        return None
    denom = b["value"]
    if denom == 0:
        return None
    ratio = a["value"] / denom
    # 二重検算: rank1 は entries からの max/min とも一致することを確認（fail-closed）。
    vals = [e["value"] for e in series["entries"]]
    if a_rank == 1:
        if a["value"] != (max(vals) if a_dir == "desc" else min(vals)):
            return None
    if b_rank == 1:
        if b["value"] != (max(vals) if b_dir == "desc" else min(vals)):
            return None
    value = f"{round(ratio, decimals):.{decimals}f}"
    evidence = {
        "case": project, "metric": series.get("metric_key"), "table": series.get("caption"),
        "rel": series.get("rel"), "locus": series.get("locus"),
        "numerator": {"rank": a_rank, "direction": a_dir, "label": a.get("label"), "value": a["value"]},
        "denominator": {"rank": b_rank, "direction": b_dir, "label": b.get("label"), "value": b["value"]},
        "ratio_full": ratio, "decimals": decimals, "unit": series.get("unit"),
        "n_values": series.get("n"),
        "basis": ("doc_reach_store の統計表を header 名で系列化（分割ランキング表は同一 header 列を統合）し、"
                  "昇順/降順ソート済みの rank-k 値から比を決定論計算（二重検算・事前計算）。"),
    }
    return _result(value, selection="ranked_pair_ratio", evidence=evidence)


# --------------------------------------------------------------------------- serve entry
def resolve(question: str) -> "dict[str, Any] | None":
    """idx99 の決定論直答（束縛できれば contract、曖昧なら None）。OFF なら常に None（byte-identical）。"""
    if not enabled():
        return None
    try:
        data = _store.load()
    except Exception:  # noqa: BLE001 — 壊れたストアは fall back、答えパスを壊さない
        return None
    by_project = data.get("by_project") or {}
    if not by_project:
        return None
    try:
        result = _ratio_lane(question, by_project)
    except Exception:  # noqa: BLE001
        return None
    if result is None or not _contract.is_contract(result):
        return None
    normalized = _contract.ensure_contract(result)
    return normalized if normalized.get("value") is not None else None


# --------------------------------------------------------------------------- investigator tool (補助)
def _tool_handler(question: str = "", case: str = "", metric: str = "") -> dict[str, Any]:
    """統計表 rank 系列の lookup（補助ツール）。case+metric でその系列の昇順/降順ランキングを返す。"""
    data = _store.load()
    by_project = data.get("by_project") or {}
    q = _norm(question) if question else ""
    project = _bind_case(by_project, q) if q else None
    if project is None and case:
        cn = _norm(case)
        cands = [p for p in by_project if cn in _norm(p)]
        project = cands[0] if len(cands) == 1 else None
    if project is None:
        return _contract.make(None, engine="derived_ranking",
                              evidence={"applicable": True, "bound": False},
                              note="案件を一意束縛できず")
    series = by_project.get(project) or []
    mk = _norm(metric)
    if mk:
        series = [s for s in series if s.get("metric_key") and mk in s["metric_key"]]
    value = [{"metric": s.get("metric_key"), "caption": s.get("caption"), "rel": s.get("rel"),
              "n": s.get("n"), "sorted_desc": s.get("sorted_desc"), "sorted_asc": s.get("sorted_asc")}
             for s in series[:12]]
    return _contract.make(value, engine="derived_ranking",
                          evidence={"applicable": True, "case": project, "series": len(value)},
                          scheme="derived-ranking-lookup")


def tool() -> "tuple[str, str, dict[str, Any], Callable[..., Any]] | None":
    """RAG_DERIVED_RANKING OFF ⇒ None ⇒ ツール集合/関数スキーマ/MCP surface は byte-identical。"""
    if not enabled():
        return None
    return (
        DERIVED_RANKING_LOOKUP,
        "統計表 rank/ratio: 文書中の統計表を header 名で数値系列化し昇順/降順で事前ソートした結果を返す。"
        "『<metric> が最も高い/N番目に低い〜の何倍/差』の派生計算に、序数(最上位/最下位・N番目)の値を"
        "出典付きで引ける。case（案件名）と metric（列見出しキーワード, 例: 死亡率）で系列を絞る。",
        {"type": "object", "properties": {"question": _STR, "case": _STR, "metric": _STR}, "required": []},
        _tool_handler,
    )
