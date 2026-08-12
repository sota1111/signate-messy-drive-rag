"""SOT-2647 (事前計算事実層 5/5) — wire the 4 precomputed stores into the serve path.

Parent PLAN SOT-2602 (決定論先行パイプラインへの反転). The four build-time-only stores —
案件マスタ (SOT-2643 :mod:`src.rag.index.case_master`), IDマスタ (SOT-2644
:mod:`src.rag.index.id_master`), 派生メトリクス (SOT-2645 :mod:`src.rag.index.derived_metrics`),
版ペア差分 (SOT-2646 :mod:`src.rag.index.diff_store`) — were shipped Done as build-time artifacts
with **no serve-path consumer**. This module is the single place that connects them to the answer
path in **both** forms the inverted architecture prescribes:

* **(a) deterministic direct-answer lanes** — :func:`resolve` maps a classified contract type to the
  matching store and returns a ``{value, evidence, method}`` contract **only when the question binds
  unambiguously to a unique store value** (値＋出典で回答, formatting 層経由). Any ambiguity ⇒ ``None``
  ⇒ the caller falls back to the LLM loop (回答数を減らさない, wrong を増やさない). This is precision-first
  per the SOT-2601 lesson (発火条件を緩めた統合が fail した): the lane never fires on a loose bind.
* **(b) first-class investigator tools** — :func:`tools` returns 4 tools (``case_filter`` /
  ``id_lookup`` / ``metric_lookup`` / ``diff_lookup``) that let the LLM loop pull an exhaustive /
  O(1) store fact in one call instead of the ``file_grep`` repetition that exhausts the turn budget
  (BUDGET32 診断: file_grep = 全ツール呼びの 70%). Because the investigator's
  :func:`build_tools` is the MCP server's single source of truth
  (:mod:`src.rag.mcp.server`), registering here publishes to the Gemini surface and the MCP surface
  at once — 単一情報源.

Design invariants
-----------------
* **RAG_FACT_LAYER default OFF ⇒ byte-identical.** The one master flag gates the lane, the tools, and
  the prompt discipline line. Off (the default) ⇒ :func:`resolve` short-circuits to ``None``,
  :func:`tools` returns ``[]`` (tool set / function schema unchanged), and the prompt is unchanged.
* **No corpus fact / no hardcoding.** This layer holds no answers and no idx numbers; every value is
  read from a store the build produced question-independently, and every value carries its store
  provenance (doc/sheet/cell) so the commit gate can treat it as a **verified operand**
  (:data:`src.rag.agent.commit_gate` grounding set).
* **Fail-open, never raise into the answer path.** Any read/parse error ⇒ ``None`` / empty result.
* **Coverage honesty.** A lane fires only where the store report certified the required attributes as
  present; the hard-core idx a store reported as カバレッジ外 are never force-fired (they fall back).
"""
from __future__ import annotations

import os
import re
from typing import Any, Callable, Mapping

from src.rag.tools import contract as _contract

# The 4 store tool names — also the commit_gate grounding set (verified operands, SOT-2637/2647).
CASE_FILTER = "case_filter"
ID_LOOKUP = "id_lookup"
METRIC_LOOKUP = "metric_lookup"
DIFF_LOOKUP = "diff_lookup"
TOOL_NAMES = frozenset({CASE_FILTER, ID_LOOKUP, METRIC_LOOKUP, DIFF_LOOKUP})


def _env_flag(key: str, default: bool) -> bool:
    raw = os.getenv(key)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def enabled() -> bool:
    """Whether the precomputed fact layer (lanes + tools + prompt line) is active.

    **Default OFF** (``RAG_FACT_LAYER``) so the champion serve path stays byte-identical: off ⇒
    :func:`resolve` returns ``None`` for every question and :func:`tools` returns ``[]`` (the tool
    set, function-call schema, and MCP surface are unchanged).
    """
    return _env_flag("RAG_FACT_LAYER", False)


# --------------------------------------------------------------------------- lazy store access
def _stores():
    """Import the 4 stores lazily (fail-open: any import error ⇒ ``None`` placeholders)."""
    try:
        from src.rag.index import case_master, derived_metrics, diff_store, id_master
        return case_master, id_master, derived_metrics, diff_store
    except Exception:  # noqa: BLE001 — a broken store import must never break the answer path
        return None, None, None, None


def _norm(text: Any) -> str:
    return re.sub(r"\s+", "", str(text or ""))


# --------------------------------------------------------------------------- provenance helpers
def _attr(row: Mapping[str, Any], attr: str) -> Mapping[str, Any]:
    """The ``{value, source}`` cell for ``attr`` on a case-master row (empty dict when absent)."""
    cell = (row.get("attributes") or {}).get(attr)
    return cell if isinstance(cell, Mapping) else {}


def _attr_value(row: Mapping[str, Any], attr: str) -> Any:
    return _attr(row, attr).get("value")


def _case_label(row: Mapping[str, Any]) -> str:
    """The 主略称 for a case, falling back to the case name (never invented)."""
    return str(row.get("abbrev") or _attr_value(row, "abbrev")
               or _attr_value(row, "case_name") or row.get("case_id") or "")


# =========================================================================== (b) investigator tools
def _case_filter(filters: str = "", select: str = "", project: str = "") -> dict[str, Any]:
    """Filter the case master by attribute equality and project the requested columns (with provenance).

    ``filters`` is ``attr=value`` pairs joined by ``;`` / ``,`` (e.g. ``"status=完了;apr_code=APR-M1"``);
    ``select`` is a comma list of attribute names to include (empty ⇒ a standard column set). The result
    lists every matching case with each selected cell's ``value`` + ``source`` and the total universe size
    (row count == 案件数 ⇒ a self-evident completeness certificate for 完全列挙). No aggregation is done
    here — the LLM composes 列挙/合計/差 from the exhaustive rows.
    """
    case_master, _idm, _dm, _ds = _stores()
    if case_master is None:
        return _contract.make(None, engine="case_master", evidence={"applicable": False},
                              note="store 読込不可")
    rows = case_master.load()
    equals = _parse_kv(filters)
    matched = [r for r in rows if all(_attr_value(r, a) == v for a, v in equals.items())]
    cols = [c.strip() for c in select.split(",") if c.strip()] or _STD_CASE_COLS
    proj = _norm(project)
    out_rows: list[dict[str, Any]] = []
    for r in matched:
        if proj and proj not in _norm(r.get("case_id")) and proj not in _norm(_case_label(r)):
            continue
        cells = {c: {"value": _attr_value(r, c), "source": _attr(r, c).get("source")} for c in cols}
        out_rows.append({"case_id": r.get("case_id"), "abbrev": _case_label(r), "cells": cells})
    value = [{"abbrev": x["abbrev"], **{c: x["cells"][c]["value"] for c in cols}} for x in out_rows]
    return _contract.make(
        value, engine="case_master",
        evidence={"applicable": True, "filters": equals, "select": cols, "rows": out_rows,
                  "universe_size": len(rows), "matched": len(out_rows),
                  "completeness": "row_count==案件数 (母集団確定)"},
        scheme="case-master-filter", provenance="per-cell source (doc/sheet/cell)")


def _id_lookup(id: str, id_kind: str = "", project: str = "") -> dict[str, Any]:  # noqa: A002 - schema name
    """Reverse-lookup every occurrence of an ID code (原文行 + 隣接列 + 所在), separator-insensitive.

    Wraps :func:`id_master.lookup`. Returns each occurrence's full ``line``, table ``columns``, and locator
    (doc/sheet/slide/page/row) so an "ID→内容そのまま抜き出す" question is answered from the corpus text
    verbatim without ``file_grep`` iteration. Unknown ID ⇒ ``value`` is an empty list.
    """
    _cm, id_master, _dm, _ds = _stores()
    if id_master is None:
        return _contract.make(None, engine="id_master", evidence={"applicable": False},
                              note="store 読込不可")
    occ = id_master.lookup(id, id_kind=(id_kind or None), project=(project or None))
    value = [{"id": o.get("id"), "id_kind": o.get("id_kind"), "doc_id": o.get("doc_id"),
              "sheet": o.get("sheet"), "slide": o.get("slide"), "page": o.get("page"),
              "row": o.get("row"), "line": o.get("line"), "columns": o.get("columns"),
              "column_header": o.get("column_header"), "label": o.get("label")}
             for o in occ]
    return _contract.make(
        value, engine="id_master",
        evidence={"applicable": True, "id": id, "id_kind": id_kind or None,
                  "project": project or None, "occurrences": len(occ)},
        scheme="id-reverse-lookup")


def _metric_lookup(case: str, metric: str = "") -> dict[str, Any]:
    """Look up a case's precomputed derived metrics (column stats / percentiles / ratios / OLS / F1 sweep).

    Wraps :func:`derived_metrics.case_metrics`. With ``metric`` a dotted path (e.g.
    ``model.threshold_sweep.best_f1``, ``model.prediction_id0.prediction``, ``ratios.<col>.p90_minus_p50``)
    the single scalar is returned with its source; empty ``metric`` ⇒ the available metric keys are listed
    so the LLM can pick the exact path. Values are the store's independently-verified computations (回帰係数
    は OLS フィット — ``sources.coef_source`` を honest に同梱).
    """
    _cm, _idm, derived_metrics, _ds = _stores()
    if derived_metrics is None:
        return _contract.make(None, engine="derived_metrics", evidence={"applicable": False},
                              note="store 読込不可")
    rec = derived_metrics.case_metrics(case)
    if rec is None:
        return _contract.make(None, engine="derived_metrics",
                              evidence={"applicable": True, "case": case, "found": False},
                              note="該当案件なし")
    sources = rec.get("sources") or {}
    if not metric:
        return _contract.make(
            {"case_id": rec.get("case_id"), "available": _metric_keys(rec)},
            engine="derived_metrics",
            evidence={"applicable": True, "case": case, "found": True,
                      "train_file": sources.get("train_file"), "coef_source": sources.get("coef_source")},
            scheme="derived-metrics-index")
    val = _dig(rec, metric)
    return _contract.make(
        val, engine="derived_metrics",
        evidence={"applicable": True, "case": case, "metric": metric, "resolved": val is not None,
                  "train_file": sources.get("train_file"), "coef_source": sources.get("coef_source")},
        scheme="derived-metrics-lookup")


def _diff_lookup(question: str = "", old: str = "", new: str = "", project: str = "",
                 exclude_attributes: str = "", require_attributes: str = "") -> dict[str, Any]:
    """Read a precomputed version-pair diff's SUBSTANTIVE-ranked changes, with question-side filters.

    Wraps :func:`diff_store.lookup_pair` + :func:`diff_store.filter_changes`. Resolves the pair from
    explicit ``old``/``new`` substrings (or, failing that, the version pair the ``question`` refers to via
    the differ), then returns the ranked changes minus ``exclude_attributes`` (e.g. ``status_transition``
    for idx95 「未着手→完了 を除く」). The diff itself is precomputed (LLM-free); this only reads + filters.
    """
    _cm, _idm, _dm, diff_store = _stores()
    if diff_store is None:
        return _contract.make(None, engine="diff_store", evidence={"applicable": False},
                              note="store 読込不可")
    recs = diff_store.lookup_pair(old_rel=(old or None), new_rel=(new or None),
                                  project=(project or None))
    if not recs and project:
        # ``lookup_pair`` matches project exactly; retry with a substring match so a 主略称/短縮名 binds.
        pn = _norm(project)
        recs = [r for r in diff_store.load()
                if pn in _norm(r.get("project"))
                and (not old or _norm(old) in _norm(r.get("old_rel")))
                and (not new or _norm(new) in _norm(r.get("new_rel")))]
    if not recs and question:
        recs = _resolve_pair_from_question(diff_store, question)
    if not recs:
        return _contract.make(None, engine="diff_store",
                              evidence={"applicable": True, "resolved": False},
                              note="版ペア不確定(明示old/new か project を指定)")
    if len(recs) > 1:
        # Ambiguous: the project has multiple version pairs. Return the candidates for the LLM to pick with
        # a narrower old/new — never silently answer the wrong pair (precision-first).
        cands = [{"old": r.get("old_rel"), "new": r.get("new_rel"), "basis": r.get("basis"),
                  "change_count": len(r.get("changes") or [])} for r in recs]
        return _contract.make(None, engine="diff_store",
                              evidence={"applicable": True, "resolved": False, "ambiguous": True,
                                        "candidates": cands},
                              note="複数の版ペアが該当(old/new でペアを一意に指定してください)")
    rec = recs[0]
    changes = diff_store.filter_changes(
        rec, exclude_attributes=_split(exclude_attributes), require_attributes=_split(require_attributes))
    # NB: store records keep the texts under "old"/"new" (diffpair.RankedChange.to_dict) — the prior
    # "before"/"after" reads always returned None (SOT-2650 fix). Notebook records may also carry a
    # deterministic table-header diff; surface it when present.
    value = []
    for c in changes[:12]:
        item = {"rank": c.get("rank"), "kind": c.get("kind"), "intent": c.get("intent"),
                "before": c.get("old", c.get("before")), "after": c.get("new", c.get("after")),
                "structural_location": c.get("structural_location"), "attributes": c.get("attributes")}
        for k in ("headers_added", "headers_removed", "headers_old_count", "headers_new_count"):
            if c.get(k) not in (None, []):
                item[k] = c[k]
        value.append(item)
    return _contract.make(
        value, engine="diff_store",
        evidence={"applicable": True, "resolved": True, "old_file": rec.get("old_rel"),
                  "new_file": rec.get("new_rel"), "version_basis": rec.get("basis"),
                  "total_changes": len(rec.get("changes") or []), "after_filter": len(changes)},
        scheme="version-pair-diff")


def _resolve_pair_from_question(diff_store, question: str) -> list[dict[str, Any]]:
    """Best-effort: pick the stored pair whose project the question names (single hit only, else none)."""
    try:
        rows = diff_store.load()
    except Exception:  # noqa: BLE001
        return []
    q = _norm(question)
    hits = [r for r in rows if _norm(r.get("project")) and _norm(r.get("project")) in q]
    return hits if len(hits) == 1 else []


# --------------------------------------------------------------------------- tool plumbing
_STD_CASE_COLS = ["case_name", "company", "contract_type", "contract_amount_incl_tax",
                  "apr_code", "approval_level", "status", "train_rows"]


def _parse_kv(spec: str) -> dict[str, str]:
    """Parse ``"a=b;c=d"`` / ``"a=b,c=d"`` into a dict (whitespace-tolerant; empty ⇒ {})."""
    out: dict[str, str] = {}
    for part in re.split(r"[;,]", spec or ""):
        if "=" in part:
            k, _, v = part.partition("=")
            k, v = k.strip(), v.strip()
            if k:
                out[k] = v
    return out


def _split(spec: str) -> list[str]:
    return [s.strip() for s in re.split(r"[;,]", spec or "") if s.strip()]


def _dig(obj: Any, path: str) -> Any:
    """Walk a dotted ``path`` into nested mappings; ``None`` on any miss."""
    cur: Any = obj
    for key in path.split("."):
        if isinstance(cur, Mapping) and key in cur:
            cur = cur[key]
        else:
            return None
    return cur


def _metric_keys(rec: Mapping[str, Any]) -> list[str]:
    """A compact catalog of the dotted metric paths available for a case record."""
    keys: list[str] = []
    for top in ("column_stats", "percentiles", "ratios", "correlations", "histograms"):
        sub = rec.get(top)
        if isinstance(sub, Mapping) and sub:
            keys.extend(f"{top}.{k}" for k in list(sub)[:24])
    model = rec.get("model")
    if isinstance(model, Mapping):
        if isinstance(model.get("threshold_sweep"), Mapping):
            keys.append("model.threshold_sweep.best_f1")
            keys.append("model.threshold_sweep.best_threshold")
        if model.get("prediction_id0") is not None:
            keys.append("model.prediction_id0.prediction")
        if isinstance(model.get("coefficients"), Mapping):
            keys.append("model.coefficients")
    return keys


# (name, description, JSON-Schema parameters, handler) — investigator maps these to AgentTool so both the
# Gemini function-call surface and the MCP tools/list surface are populated from this ONE list (単一情報源).
_STR = {"type": "string"}


def _report_attr_tool() -> "tuple[str, str, dict[str, Any], Callable[..., Any]] | None":
    """SOT-2655 報告数値属性 lookup ツール（RAG_REPORT_ATTR_STORE OFF ⇒ None ⇒ surface byte-identical）。"""
    try:
        from src.rag.agent import report_attr_lane
        return report_attr_lane.tool()
    except Exception:  # noqa: BLE001 — a broken optional tool must never break the tool set
        return None


def _case_finance_tool() -> "tuple[str, str, dict[str, Any], Callable[..., Any]] | None":
    """SOT-2654 財務・工数 lookup tool (default OFF)."""
    try:
        from src.rag.agent import case_finance_lane
        return case_finance_lane.tool()
    except Exception:  # noqa: BLE001
        return None


def _action_row_tool() -> "tuple[str, str, dict[str, Any], Callable[..., Any]] | None":
    """SOT-2652 action-row lookup tool (default OFF ⇒ None ⇒ surface byte-identical)."""
    try:
        from src.rag.agent import action_row_lane
        return action_row_lane.tool()
    except Exception:  # noqa: BLE001 — a broken optional tool must never break the tool set
        return None


def _visual_tool() -> "tuple[str, str, dict[str, Any], Callable[..., Any]] | None":
    """SOT-2653 xlsx visual-facts lookup tool (default OFF ⇒ None ⇒ surface byte-identical)."""
    try:
        from src.rag.agent import visual_lane
        return visual_lane.tool()
    except Exception:  # noqa: BLE001 — a broken optional tool must never break the tool set
        return None


def _heading_page_tool() -> "tuple[str, str, dict[str, Any], Callable[..., Any]] | None":
    """SOT-2668 見出し→印字ページ lookup tool (default OFF ⇒ None ⇒ surface byte-identical)."""
    try:
        from src.rag.agent import heading_page_lane
        return heading_page_lane.tool()
    except Exception:  # noqa: BLE001 — a broken optional tool must never break the tool set
        return None


def _derived_coverage_tool() -> "tuple[str, str, dict[str, Any], Callable[..., Any]] | None":
    """SOT-2679 派生メトリクス網羅（相関最大特徴量 / 案件横断欠損行数）lookup tool (default OFF ⇒ None)."""
    try:
        from src.rag.agent import derived_coverage_lane
        return derived_coverage_lane.tool()
    except Exception:  # noqa: BLE001 — a broken optional tool must never break the tool set
        return None


def _schedule_tool() -> "tuple[str, str, dict[str, Any], Callable[..., Any]] | None":
    """SOT-2680 スケジュール/ID/体制クロス参照 lookup tool (default OFF ⇒ None ⇒ surface byte-identical)."""
    try:
        from src.rag.agent import schedule_lane
        return schedule_lane.tool()
    except Exception:  # noqa: BLE001 — a broken optional tool must never break the tool set
        return None


def _doc_reach_tools() -> "list[tuple[str, str, dict[str, Any], Callable[..., Any]]]":
    """SOT-2677 文書テーブル/全文チャンク lookup ツール群 (default OFF ⇒ [] ⇒ surface byte-identical)。"""
    try:
        from src.rag.agent import doc_lookup_lane
        return doc_lookup_lane.tool()
    except Exception:  # noqa: BLE001 — a broken optional tool must never break the tool set
        return []


def tools() -> list[tuple[str, str, dict[str, Any], Callable[..., Any]]]:
    """The 4 fact-layer tools, or ``[]`` when the layer is OFF (⇒ tool set / MCP surface byte-identical)."""
    if not enabled():
        return []
    optional = [x for x in (_report_attr_tool(), _case_finance_tool(), _action_row_tool(),
                            _visual_tool(), _heading_page_tool(), _derived_coverage_tool(),
                            _schedule_tool())
                if x is not None]
    optional += _doc_reach_tools()
    return optional + [
        (CASE_FILTER,
         "案件マスタ(全案件×標準属性, 各セル出典付き)を属性一致でフィルタし、該当案件を出典付きで返す。"
         "『APR-M3の案件を略称で列挙し契約金額(税込)を合計』『完了かつAPR-M1で顧客データ≥10000行の案件』の"
         "ような横断列挙/横断比較は file_grep を反復せず本ツールを使う。filters は 'status=完了;apr_code=APR-M1' の"
         "ように属性=値を;区切りで、select は返す属性名をカンマ区切りで渡す(空なら標準列)。行数=案件数なので"
         "母集団は自明に確定(完全列挙の根拠)。集計(合計/差/件数)は返った全行から自分で組む。",
         _obj({"filters": _STR, "select": _STR, "project": _STR}),
         _case_filter),
        (ID_LOOKUP,
         "IDマスタ逆引き: タスク/アクション/マイルストーン/チェックポイント/WBS等のID出現を全件、原文行+隣接列+"
         "所在(doc/sheet/slide/page/row)付きで返す。『特定IDの内容をそのまま抜き出す』質問は file_grep を"
         "反復せず本ツールに ID を渡す(区切り非依存: 'M-01'='m01')。id_kind/project で絞れる。",
         _obj({"id": _STR, "id_kind": _STR, "project": _STR}, ["id"]),
         _id_lookup),
        (METRIC_LOOKUP,
         "派生メトリクスストア: 案件の train データから事前計算した列統計/分位点/相関/ヒストグラム/比/OLS予測/"
         "F1最大化閾値スイープを引く。metric に 'model.threshold_sweep.best_f1'・"
         "'model.prediction_id0.prediction'・'ratios.<列>.p90_minus_p50' 等のドット経路を渡すと単一スカラを"
         "出典付きで返す(空なら利用可能な指標キー一覧)。回帰係数は OLS フィット(coef_source を同梱)。"
         "実行時 compute の当て推量反復の代替。",
         _obj({"case": _STR, "metric": _STR}, ["case"]),
         _metric_lookup),
        (DIFF_LOOKUP,
         "版ペア差分ストア: 旧版→新版の構造diffを実質的変更(SUBSTANTIVE)順で事前確定済みで返す。old/new に"
         "ファイル名の一部、または project を渡してペアを特定する。exclude_attributes に 'status_transition' 等を"
         "渡すと質問側の除外条件(idx95『未着手→完了を除く』)を決定論適用できる。diff は事前計算(LLM非依存)。",
         _obj({"question": _STR, "old": _STR, "new": _STR, "project": _STR,
               "exclude_attributes": _STR, "require_attributes": _STR}),
         _diff_lookup),
    ]


def _obj(props: dict[str, dict[str, Any]], required: "list[str] | None" = None) -> dict[str, Any]:
    return {"type": "object", "properties": props, "required": list(required or [])}


# =========================================================================== (a) deterministic lanes
# Contract types classified upstream (:mod:`src.rag.agent.question_contract`).
_ENUM = frozenset({"full_enumeration", "cross_aggregate"})
_DERIVED = frozenset({"numeric"})

# Canonical single-scalar derived metrics with an unambiguous NL anchor (idx57/idx63 型). A lane fires only
# on one of these — a scalar with a store-verified value + provenance — never on a free-form derivation.
# Each anchor's predicate runs on the case-folded, whitespace-stripped question; it must be SPECIFIC enough
# that a loose mention (a bare "F1" / "id") does NOT fire (precision-first, SOT-2601 の教訓).
_ID0_RE = re.compile(r"id[=＝]?0")
_METRIC_ANCHORS: tuple[tuple[str, str, Callable[[str], bool]], ...] = (
    # (dotted path, human label, predicate over the case-folded normalized question)
    ("model.threshold_sweep.best_f1", "F1最大化閾値でのF1",
     lambda q: "f1" in q and ("閾値" in q or "最大化" in q)),
    ("model.prediction_id0.prediction", "id=0の予測値",
     lambda q: "予測" in q and bool(_ID0_RE.search(q))),
)
# Aggregation cues that make an enum answer a composite (列挙＋集計) — the lane defers those to the tool.
_AGG_CUE = re.compile(r"合計|総額|平均|中央値|最大|最小|最も|何件|件数|差|割合|比率|比べ|上位")
# SOT-2649 empty-composite lane cues. A 列挙+契約金額合計 composite over an EMPTY apr_code match-set is
# deterministic (該当なし): an empty superset stays empty under any conjunctive refinement and every
# aggregation over it is vacuous. Complement/disjunction semantics break that proof ⇒ hard defer.
_SUM_CUE = re.compile(r"合計|総額")
_NONSUM_AGG_CUE = re.compile(r"平均|中央値|最大|最小|最も|何件|件数|割合|比率|比べ|上位|差")
_UNSAFE_SET_CUE = re.compile(r"以外|除|不要|必要ない|必要のない|でない|ではない|含まな|または|もしくは|あるいは")
# _EXTRA_PREDICATE minus the tokens attributable to the amount-sum request itself (契約金額(税込)の合計):
# any OTHER attribute predicate still defers the composite (precision-first — same boundary as idx87).
_COMPOSITE_EXTRA = re.compile(
    r"完了|未着手|進行|中止|ステータス|状態|かつ|うち|以上|以下|超|未満|行|サンプル|"
    r"医療|固定|精算|着手金|期間|担当|工数|会社|APR-M(?![123]).")
_ENUM_CUE = re.compile(r"すべて|全て|全部|列挙|挙げ|漏れなく|該当する.*を|主略称|略称で")
_APR_CODE = re.compile(r"APR[-\s]?M([123])", re.IGNORECASE)
# ANY extra case-attribute predicate beyond the single APR code makes the enumeration multi-condition
# (idx87 の「完了案件のうち…顧客データ…10000行以上」型) — the APR-only lane would then return a SUPERSET,
# so it MUST defer to ``case_filter`` + the LLM (precision-first, SOT-2601). This is a deliberately broad
# block: a bare "APR-Mx の案件を略称で列挙" fires; anything naming another attribute defers.
_EXTRA_PREDICATE = re.compile(
    r"完了|未着手|進行|中止|ステータス|状態|かつ|うち|以上|以下|超|未満|行|サンプル|"
    r"金額|円|医療|固定|精算|着手金|期間|担当|契約|工数|会社|APR-M(?![123]).")
# SOT-2650 amount-difference enumeration (idx67 型): 「完了案件のうち APR-Mx に該当する案件の中で、
# 提案時金額と FR 時の金額が異なる案件を略称ですべて」. Both amounts are standard case-master attributes,
# so under a full-coverage certificate over the filtered universe the enumeration is deterministic.
_AMOUNT_DIFF_CUE = re.compile(
    r"提案時?の?金額.{0,14}(?:FR|最終報告|最終請求)時?の?金額|"
    r"(?:FR|最終報告|最終請求)時?の?金額.{0,14}提案時?の?金額")
_DIFF_NE_CUE = re.compile(r"異な|一致しない|同じでない|違う")
# Any attribute predicate OUTSIDE this shape's vocabulary (完了 status + APR code + the two amounts)
# still defers — same precision-first boundary as the other composites.
_DIFF_EXTRA_PREDICATE = re.compile(
    r"未着手|進行|中止|以上|以下|超|未満|行|サンプル|医療|固定|精算|着手金|期間|担当|工数|会社|"
    r"合計|総額|平均|中央値|最大|最小|上位|契約金額|着手|円")


def _sole_apr_predicate(question: str) -> bool:
    """True iff the question's ONLY case filter is a single APR-Mx code (no other attribute predicate)."""
    return not _EXTRA_PREDICATE.search(question)


def _derived_lane(question: str):
    """A single store-verified scalar for a NUMERIC question that binds a unique case + canonical metric.

    Fires only when: exactly one store case matches a case hint in the question AND exactly one metric
    anchor's tokens are present AND that metric resolves to a non-null store value. Otherwise ``None`` —
    ambiguous case, ambiguous/absent metric, or a free-form derivation all defer to the LLM loop.

    SOT-2649: a shared-stem mention (「青葉のTX」— 青葉与信/青葉バイオ) binds via *metric presence*:
    when every candidate matched through the SAME single token, the store itself disambiguates — only
    the case that actually carries the anchored metric (non-null) can be meant. Two DIFFERENT explicit
    names in one question (comparison shape) still defer.
    """
    _cm, _idm, derived_metrics, _ds = _stores()
    if derived_metrics is None:
        return None
    q = _norm(question)
    rows = derived_metrics.load()
    ql = q.lower()
    anchored = [(path, label) for (path, label, pred) in _METRIC_ANCHORS if pred(ql)]
    if len(anchored) != 1:
        return None
    path, label = anchored[0]
    hits = [r for r in rows if _norm(r.get("case_id")) and _norm(r.get("case_id"))[:6] in q]
    # Prefer a company-name substring hit; require exactly one case bound.
    if len(hits) != 1:
        # fall back to substring of company tokens (青葉与信 / prefix-stripped stem 青葉 等)
        matched = {id(r): sorted((tok for tok in _case_tokens(r) if tok and tok in q),
                                 key=len, reverse=True) for r in rows}
        hits = [r for r in rows if matched[id(r)]]
        if len(hits) > 1:
            # an explicit (longer) name beats a bare stem; a tie must be the SAME token (single
            # ambiguous mention), never two distinct names — that is a cross-case comparison ⇒ defer
            best = max(len(matched[id(r)][0]) for r in hits)
            top = [r for r in hits if len(matched[id(r)][0]) == best]
            if len({matched[id(r)][0] for r in top}) != 1:
                return None
            if len(top) > 1:
                # store-side disambiguation: exactly one candidate may carry the anchored metric
                top = [r for r in top if _dig(r, path) is not None]
            hits = top
        if len(hits) != 1:
            return None
    rec = hits[0]
    val = _dig(rec, path)
    if val is None:
        return None
    sources = rec.get("sources") or {}
    evidence = {
        "case": rec.get("case_id"), "metric": path, "metric_label": label,
        "train_file": sources.get("train_file"), "coef_source": sources.get("coef_source"),
        "store": "derived_metrics", "provenance": "precomputed (independent dual verification)",
    }
    method = {"engine": "derived_metrics", "contract": "numeric", "selection": "single_scalar_anchor",
              "naturalize": False, "verified_operand": True, "confidence": 1.0}
    return {"value": val, "evidence": evidence, "method": method}


_CORP_PREFIX = re.compile(r"^(株式会社|医療法人社団|有限会社|合同会社)\s*")


def _case_tokens(rec: Mapping[str, Any]) -> list[str]:
    """Normalized case-name tokens usable as question hints (company name, leading 4+ chars)."""
    cid = _norm(rec.get("case_id"))
    toks = [cid[:6]] if len(cid) >= 6 else ([cid] if cid else [])
    # a distinctive leading company token (e.g. 青葉与信) — 4 chars is enough to disambiguate 10 cases
    if len(cid) >= 4:
        toks.append(cid[:4])
    # prefix-stripped stems (株式会社青葉バイオ… → 青葉バイオ / 青葉): questions name companies without
    # the corporate form; the 2-char stem is a *candidate generator* only — a stem-tied bind must still
    # pass the same-token + metric-presence guards in the derived lane (SOT-2649)
    stem = _norm(_CORP_PREFIX.sub("", str(rec.get("case_id") or "")))
    for n in (4, 2):
        if len(stem) >= n:
            toks.append(stem[:n])
    seen: list[str] = []
    for t in toks:
        if t and t not in seen:
            seen.append(t)
    return seen


def _amount_diff_enum(question: str, apr: str):
    """SOT-2650 — deterministic 提案時金額≠FR時金額 enumeration over the (完了, APR-Mx) universe.

    Fires only when: the amount-difference shape is the question's ONLY non-status/APR predicate
    (:data:`_DIFF_EXTRA_PREDICATE` defers anything else), AND the store certifies full coverage —
    apr_code/status for EVERY case, and BOTH amounts for every case inside the filtered universe.
    A single missing cell defers (an incomplete universe could silently drop a qualifying case).
    """
    case_master, _idm, _dm, _ds = _stores()
    if case_master is None:
        return None
    if _DIFF_EXTRA_PREDICATE.search(question):
        return None
    rows = case_master.load()
    if not rows:
        return None
    if any(_attr_value(r, "apr_code") is None for r in rows):
        return None
    use_status = "完了" in question
    if use_status and any(_attr_value(r, "status") is None for r in rows):
        return None
    universe = [r for r in rows
                if _attr_value(r, "apr_code") == apr
                and (not use_status or _attr_value(r, "status") == "完了")]
    # Full-coverage certificate on the compared attributes: every universe member must carry BOTH
    # amounts (else defer — never enumerate over partial evidence).
    amounts: list[tuple[Any, int, int]] = []
    for r in universe:
        prop = _attr_value(r, "proposal_amount_incl_tax")
        fr = _attr_value(r, "fr_amount_incl_tax")
        if not isinstance(prop, int) or not isinstance(fr, int):
            return None
        amounts.append((r, prop, fr))
    matched = [(r, prop, fr) for (r, prop, fr) in amounts if prop != fr]
    labels = [_case_label(r) for (r, _p, _f) in matched]
    if matched and any(not l for l in labels):
        return None
    evidence = {
        "filter": {"apr_code": apr, **({"status": "完了"} if use_status else {})},
        "predicate": "proposal_amount_incl_tax != fr_amount_incl_tax",
        "universe_size": len(rows), "filtered": len(universe), "matched": len(matched),
        "cases": [{"abbrev": _case_label(r), "proposal_amount_incl_tax": p, "fr_amount_incl_tax": f,
                   "proposal_source": _attr(r, "proposal_amount_incl_tax").get("source"),
                   "fr_source": _attr(r, "fr_amount_incl_tax").get("source")}
                  for (r, p, f) in matched],
        "store": "case_master",
        "completeness": "row_count==案件数 かつ 対象属性 filtered universe 全充足 (母集団確定)",
    }
    method = {"engine": "case_master", "contract": "full_enumeration",
              "selection": "amount_diff_enumeration", "naturalize": False,
              "verified_operand": True, "confidence": 1.0}
    value = "、".join(labels) if labels else "該当なし"
    return {"value": value, "evidence": evidence, "method": method}


def _enum_lane(question: str):
    """An exhaustive 略称 enumeration for an APR-code FULL_ENUMERATION question — no aggregation.

    Fires only when: exactly one APR-Mx code is named, an enumeration cue is present, NO aggregation cue is
    present (合計/件数/差 … ⇒ composite ⇒ defer), and the store fully covers the filter attributes for the
    whole universe (apr_code 10/10). Returns the 主略称 list with per-case provenance + a completeness
    certificate (row count == 案件数). Composite enum/aggregate questions defer to ``case_filter`` + the LLM.
    """
    case_master, _idm, _dm, _ds = _stores()
    if case_master is None:
        return None
    codes = {m.group(1) for m in _APR_CODE.finditer(question)}
    if len(codes) != 1:
        return None
    if not _ENUM_CUE.search(question):
        return None
    # Complement/disjunction phrasing (以外/または…) voids both the enumeration and the empty-set proof.
    if _UNSAFE_SET_CUE.search(question):
        return None
    if _AMOUNT_DIFF_CUE.search(question) and _DIFF_NE_CUE.search(question):
        return _amount_diff_enum(question, f"APR-M{next(iter(codes))}")
    composite = bool(_AGG_CUE.search(question))
    if composite:
        # SOT-2649: only the 「APR-Mx を列挙し契約金額(税込)を合計」 shape may proceed (and it answers
        # only on an empty match-set below); any other aggregation or extra predicate defers.
        if _NONSUM_AGG_CUE.search(question) or not _SUM_CUE.search(question):
            return None
        if "金額" not in question or _COMPOSITE_EXTRA.search(question):
            return None
    # Defer any multi-condition enumeration (extra attribute predicate ⇒ APR-only would over-return).
    elif not _sole_apr_predicate(question):
        return None
    apr = f"APR-M{next(iter(codes))}"
    rows = case_master.load()
    if not rows:
        return None
    # Completeness precondition: apr_code present for EVERY case (universe fully attributed) — else the
    # enumeration could silently drop a case, so we defer rather than answer incompletely.
    if any(_attr_value(r, "apr_code") is None for r in rows):
        return None
    matched = [r for r in rows if _attr_value(r, "apr_code") == apr]
    if composite:
        # Deterministic ONLY for the empty set (該当なし): non-empty composites have a multi-part
        # answer whose format is genuinely ambiguous ⇒ defer to case_filter + the LLM as before.
        if matched:
            return None
        evidence = {
            "filter": {"apr_code": apr}, "universe_size": len(rows), "matched": 0,
            "store": "case_master", "completeness": "row_count==案件数 (母集団確定)",
            "aggregation": "空集合ゆえ vacuous (該当案件なし ⇒ 合計対象なし)",
        }
        method = {"engine": "case_master", "contract": "full_enumeration",
                  "selection": "apr_code_empty_composite", "naturalize": False,
                  "verified_operand": True, "confidence": 1.0}
        return {"value": "該当なし", "evidence": evidence, "method": method}
    labels = [_case_label(r) for r in matched]
    if not labels or any(not l for l in labels):
        return None
    value = "、".join(labels)
    evidence = {
        "filter": {"apr_code": apr}, "universe_size": len(rows), "matched": len(matched),
        "cases": [{"abbrev": _case_label(r), "apr_code_source": _attr(r, "apr_code").get("source")}
                  for r in matched],
        "store": "case_master", "completeness": "row_count==案件数 (母集団確定)",
    }
    method = {"engine": "case_master", "contract": "full_enumeration", "selection": "apr_code_enumeration",
              "naturalize": False, "verified_operand": True, "confidence": 1.0}
    return {"value": value, "evidence": evidence, "method": method}


def resolve(question: str, contract: "str | None", *, profile: Any = None,
            force: bool = False) -> "dict[str, Any] | None":
    """Deterministic direct-answer lane for a store-bindable question — the fact-layer entry gate.

    Returns a ``{value, evidence, method}`` contract **only when** the layer is enabled (or ``force`` for
    unit tests), the ``contract`` type maps to a store lane, and that lane binds the question to a unique
    store value. In every other case — layer OFF (default), unmapped contract, or an ambiguous/absent bind
    — returns ``None`` so the caller falls back to the LLM loop (回答数を減らさない, wrong を増やさない). Never
    raises into the answer path.
    """
    if not (force or enabled()):
        return None
    if not contract:
        return None
    try:
        if contract in _DERIVED:
            result = _derived_lane(question)
            if result is None:
                # SOT-2655 — 報告数値属性ストア(最良モデルparam/フェーズ工数/高影響特徴量相関)の直答レーン。
                # 派生メトリクスレーンが束縛できない NUMERIC のみに後置(既存挙動は不変)。RAG_REPORT_ATTR_STORE
                # OFF なら report_attr_lane.resolve() は None ⇒ byte-identical。
                try:
                    from src.rag.agent import report_attr_lane
                    result = report_attr_lane.resolve(question)
                except Exception:  # noqa: BLE001 — 壊れた任意レーンは fall back、答えパスを壊さない
                    result = None
        elif contract in _ENUM:
            result = _enum_lane(question)
        else:
            result = None
        # SOT-2654 spans numeric and simple-lookup contracts. The optional lane performs its own strict
        # semantic binding and is default OFF, so unsupported questions remain byte-identical fallbacks.
        if result is None:
            try:
                from src.rag.agent import case_finance_lane
                result = case_finance_lane.resolve(question)
            except Exception:  # noqa: BLE001
                result = None
        # SOT-2652 — action-row（AI/タスク行）ストアの simple_lookup 直答レーン（idx20/70 型）。上位レーンが
        # 束縛できない時のみ後置。RAG_ACTION_ROW_STORE OFF ⇒ action_row_lane.resolve() は None ⇒ byte-identical。
        if result is None:
            try:
                from src.rag.agent import action_row_lane
                result = action_row_lane.resolve(question)
            except Exception:  # noqa: BLE001 — 壊れた任意レーンは fall back、答えパスを壊さない
                result = None
        # SOT-2653 — xlsx 可視事実（chart系列→カラム / ハイライト全数 / 相関CF / 行・列ハイライト交差）の
        # 決定論直答レーン（idx39/65/82/97 型）。上位レーンが束縛できない時のみ後置。RAG_VISUAL_STORE OFF ⇒
        # visual_lane.resolve() は None ⇒ byte-identical。
        if result is None:
            try:
                from src.rag.agent import visual_lane
                result = visual_lane.resolve(question)
            except Exception:  # noqa: BLE001 — 壊れた任意レーンは fall back、答えパスを壊さない
                result = None
        # SOT-2668 — 見出し→印字ページ locator ストアの fact_lookup 直答レーン（idx12/18 型）。上位レーンが
        # 束縛できない時のみ後置。RAG_HEADING_PAGE_STORE OFF ⇒ heading_page_lane.resolve() は None ⇒ byte-identical。
        if result is None:
            try:
                from src.rag.agent import heading_page_lane
                result = heading_page_lane.resolve(question)
            except Exception:  # noqa: BLE001 — 壊れた任意レーンは fall back、答えパスを壊さない
                result = None
        # SOT-2679 — 派生メトリクス網羅拡張（相関最大特徴量 idx4 / 案件横断欠損行数 argmax idx24）の決定論
        # 直答レーン。上位レーンが束縛できない時のみ後置。RAG_DERIVED_COVERAGE OFF ⇒ resolve() は None ⇒
        # byte-identical。idx24 は full-coverage かつ厳密単独最大の時だけ確定（SOT-2663 逆流ガード）。
        if result is None:
            try:
                from src.rag.agent import derived_coverage_lane
                result = derived_coverage_lane.resolve(question)
            except Exception:  # noqa: BLE001 — 壊れた任意レーンは fall back、答えパスを壊さない
                result = None
        # SOT-2680 — スケジュール/ID/体制クロス参照の決定論直答レーン（ID種別集計 idx92 / チェックポイント→
        # タスク idx96 / マイルストーン×役職→タスク idx94 / 役職→タスク数 idx72）。上位レーンが束縛できない時
        # のみ後置。RAG_SCHEDULE_STORE OFF ⇒ schedule_lane.resolve() は None ⇒ byte-identical。
        if result is None:
            try:
                from src.rag.agent import schedule_lane
                result = schedule_lane.resolve(question)
            except Exception:  # noqa: BLE001 — 壊れた任意レーンは fall back、答えパスを壊さない
                result = None
    except Exception:  # noqa: BLE001 — a broken lane must fall back, not break the answer path
        return None
    if result is None or not _contract.is_contract(result):
        return None
    normalized = _contract.ensure_contract(result)
    if normalized.get("value") is None:
        return None
    return normalized
