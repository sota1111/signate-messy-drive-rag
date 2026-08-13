"""SOT-2686 — xlsx 数式依存トレース＋記載回帰係数の行適用レーン（cycle7 K3）.

cycle7 abstain の「xlsx 計算構造カバレッジ欠落」クラスタ（idx47 / idx83）を、build 時に焼いた
:mod:`src.rag.index.xlsx_formula_trace` ストアを serve 時に **質問非依存に決定論束縛** して回収する。
新しい build は足さず、事前計算済みの成果物を読むだけ（Gemini 呼び出しゼロ・$0）。

回収する2つの束縛（precision-first: 一意・厳密に束縛できる時だけ確定、少しでも曖昧なら None→LLM）:

* **idx47 数式依存トレース** — 案件 train.xlsx の黄色ハイライトセル（誤差 ``(予測−実測)^2``）の数式が参照する
  *データ行* を辿り、質問が問う属性（例「建設年」→ ``YEAR BUILT``）をその行から返す。
  例: 青嶺 ``B22`` → ``Sheet1`` 行 26118 → ``YEAR BUILT``=1899 → 「1899年」。
* **idx83 記載回帰係数の行適用** — 案件 train.xlsx の係数表（切片＋列名付き係数）を ``index=N`` 行へ当てはめた
  予測値（事前計算済み）を、質問指定の小数桁で返す。例: みなみ野 index=1770 → 0.38317。

両ターゲットは fact_layer.resolve の **決定論直答（route=deterministic）** で回収する。投資者（LLM）ツール
サーフェスには何も足さない — ``RAG_XLSX_FORMULA_TRACE`` を ON にしても投資者へ提示するツール集合は champion と
byte-identical なので、LLM ルートの他問（番兵含む）は本フラグの影響を受けない。``RAG_XLSX_FORMULA_TRACE``
既定 OFF ⇒ :func:`resolve` は None を返し serve path は byte-identical。RAG_FACT_LAYER の下位レーンとして
fact_layer.resolve の末尾に後置される。
"""
from __future__ import annotations

import importlib
import os
import re
import unicodedata
from typing import Any

from src.rag.index import xlsx_formula_trace as _store
from src.rag.tools import contract as _contract

_ON = {"1", "true", "yes", "on"}

# Question keyword (normalized) → the referenced-row header (normalized) it asks for. Precision-first:
# a keyword only binds when exactly one referenced-row header matches its target.
_ATTRIBUTE_TARGETS: dict[str, str] = {
    "建設年": "yearbuilt",
    "建築年": "yearbuilt",
    "築年": "yearbuilt",
    "建造年": "yearbuilt",
    "yearbuilt": "yearbuilt",
}

# Cues that mark a highlighted-error-formula tracing question.
_HIGHLIGHT_CUES = ("ハイライト", "黄色", "誤差", "予測値", "予測", "対象")
# Cues that mark a documented-regression apply question.
_REGRESSION_CUES = ("回帰", "係数")
# At least one of these must appear too (apply / predict verbs).
_APPLY_CUES = ("当てはめ", "適用", "予測")

_INDEX_RE = re.compile(r"index\s*[=＝:：]?\s*(\d+)", re.IGNORECASE)
_INDEX_JP_RE = re.compile(r"インデックス\s*[=＝:：]?\s*(\d+)")
_DECIMALS_RE = re.compile(r"小数第\s*([0-9０-９]+)\s*位")


def enabled() -> bool:
    """True when the serve path should consult the xlsx formula-trace lane (default OFF)."""
    return os.getenv("RAG_XLSX_FORMULA_TRACE", "0").strip().lower() in _ON


def _norm(value: Any) -> str:
    return unicodedata.normalize("NFKC", str(value or "")).replace(" ", "").replace("　", "").lower()


def _norm_header(value: Any) -> str:
    return unicodedata.normalize("NFKC", str(value if value is not None else "")).replace(" ", "").strip().lower()


def _result(value: Any, *, selection: str, evidence: dict[str, Any]) -> dict[str, Any]:
    ev = {"store": "xlsx_formula_trace", "provenance": "precomputed (question-independent)", **evidence}
    method = {"engine": "xlsx_formula_trace", "contract": "deterministic", "selection": selection,
              "naturalize": False, "verified_operand": True, "confidence": 1.0}
    return _contract.ensure_contract({"value": value, "evidence": ev, "method": method})


# --------------------------------------------------------------------------- case binding (glossary)
def _resolve_case_docs(question: str) -> list[dict[str, Any]]:
    """Store records for the case the question names (glossary 主略称/別名 → 案件フォルダ). ``[]`` if ambiguous."""
    try:
        glossary = importlib.import_module("src.rag.extract.glossary").load()
        company = glossary.company_of(question)
    except Exception:  # noqa: BLE001
        company = None
    if not company:
        return []
    return _store.docs_for_project(company)


def _to_int_digits(text: str) -> str:
    return unicodedata.normalize("NFKC", text)


# --------------------------------------------------------------------------- lane 1: highlight-formula trace
def _highlight_formula_attribute(qn: str, question: str) -> dict[str, Any] | None:
    if not any(_norm(c) in qn for c in _HIGHLIGHT_CUES):
        return None
    # which attribute does the question ask for?
    target_header = None
    matched_kw = None
    for kw, tgt in _ATTRIBUTE_TARGETS.items():
        if kw in qn:
            if target_header is not None and tgt != target_header:
                return None  # two different attributes asked → ambiguous
            target_header = tgt
            matched_kw = kw
    if target_header is None:
        return None
    docs = _resolve_case_docs(question)
    hfs = [(d, hf) for d in docs for hf in d.get("highlight_formulas", [])]
    if len(hfs) != 1:  # need exactly one highlighted formula in the bound case
        return None
    doc, hf = hfs[0]
    rows = hf.get("referenced_rows", [])
    if len(rows) != 1:  # need a single unambiguous referenced data row
        return None
    attrs = rows[0].get("attributes", {})
    matches = [(h, v) for h, v in attrs.items() if _norm_header(h) == target_header]
    if len(matches) != 1:
        return None
    header, value = matches[0]
    if value is None:
        return None
    text = _format_year(value) if target_header == "yearbuilt" else str(value)
    return _result(text, selection="highlight_formula_trace", evidence={
        "case": doc.get("project"), "doc": doc.get("doc_name"),
        "highlight_cell": f"{hf.get('sheet')}!{hf.get('cell')}", "formula": hf.get("formula"),
        "referenced_row": {"sheet": rows[0].get("sheet"), "row": rows[0].get("row"), "id": rows[0].get("id")},
        "attribute": header, "raw_value": value, "keyword": matched_kw,
    })


def _format_year(value: Any) -> str:
    if isinstance(value, (int, float)) and not isinstance(value, bool) and float(value).is_integer():
        return f"{int(value)}年"
    return f"{value}年"


# --------------------------------------------------------------------------- lane 2: documented-regression apply
def _regression_apply(qn: str, question: str) -> dict[str, Any] | None:
    if not all(_norm(c) in qn for c in _REGRESSION_CUES):
        return None
    if not any(_norm(c) in qn for c in _APPLY_CUES):
        return None
    m = _INDEX_RE.search(question) or _INDEX_JP_RE.search(question) or _INDEX_RE.search(_to_int_digits(question))
    if not m:
        return None
    index = str(int(m.group(1)))
    docs = _resolve_case_docs(question)
    regs = [(d, rg) for d in docs for rg in d.get("regressions", [])]
    if len(regs) != 1:  # need exactly one coefficient table in the bound case
        return None
    doc, rg = regs[0]
    preds = rg.get("predictions", {})
    if index not in preds:
        return None
    value = preds[index]
    dm = _DECIMALS_RE.search(question)
    decimals = int(_to_int_digits(dm.group(1))) if dm else 5
    text = f"{round(float(value), decimals):.{decimals}f}"
    return _result(text, selection="documented_regression_apply", evidence={
        "case": doc.get("project"), "doc": doc.get("doc_name"),
        "coef_table": f"{rg.get('coef_sheet')}!{rg.get('coef_cell')}", "data_sheet": rg.get("data_sheet"),
        "index_column": rg.get("index_column"), "index": index, "intercept": rg.get("intercept"),
        "coefficients": rg.get("coefficients"), "raw_prediction": value, "decimals": decimals,
    })


_LANES = (_highlight_formula_attribute, _regression_apply)


def resolve(question: str) -> dict[str, Any] | None:
    """xlsx 計算構造質問を焼き込みストアで一意束縛できる時だけ contract を返す（fail-open・OFF なら None）。"""
    if not enabled():
        return None
    try:
        qn = _norm(question)
        for lane in _LANES:
            res = lane(qn, question)
            if res is not None and _contract.is_contract(res) and res.get("value") is not None:
                return _contract.ensure_contract(res)
    except Exception:  # noqa: BLE001 — a broken lane must fall back, never break the answer path
        return None
    return None
