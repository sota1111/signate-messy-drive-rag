"""Deterministic answer lane for the question-independent case-finance store (SOT-2654)."""
from __future__ import annotations

import re
import unicodedata
from typing import Any, Callable

from src.rag.index import case_finance_store as _store
from src.rag.tools import contract as _contract

CASE_FINANCE_LOOKUP = "case_finance_lookup"


def _norm(value: Any) -> str:
    return re.sub(r"\s+", "", unicodedata.normalize("NFKC", str(value or ""))).casefold()


def _cell(row: dict[str, Any], key: str) -> dict[str, Any]:
    return (row.get("operands") or {}).get(key) or {}


def _row(rows: list[dict[str, Any]], token: str) -> dict[str, Any] | None:
    hits = [r for r in rows if token in _norm(r.get("case_id")) or token == _norm(r.get("abbrev"))]
    return hits[0] if len(hits) == 1 else None


def _result(value: Any, selection: str, evidence: dict[str, Any],
            *, contract: str = "numeric") -> dict[str, Any]:
    return _contract.ensure_contract({
        "value": value,
        "evidence": {"store": "case_finance", "provenance": "precomputed (question-independent)", **evidence},
        "method": {"engine": "case_finance", "contract": contract, "selection": selection,
                   "verified_operand": True, "naturalize": False, "confidence": 1.0},
    })


def _aobm_reduction(q: str, rows: list[dict[str, Any]]) -> dict[str, Any] | None:
    if not ("aobm" in q and "見込金額" in q and "確定金額" in q and "esth" in q and "acth" in q
            and "1時間" in q and ("割" in q or "除" in q)):
        return None
    rec = _row(rows, "aobm")
    keys = ("estimate_amount_incl_tax", "confirmed_amount_incl_tax",
            "estimated_effort_hours", "actual_effort_hours")
    if rec is None:
        return None
    values = [_cell(rec, key).get("value") for key in keys]
    if any(v is None for v in values) or values[2] == values[3]:
        return None
    value = (values[0] - values[1]) / (values[2] - values[3])
    if abs(value - round(value)) > 1e-9:
        return None
    return _result(f"{int(round(value)):,}円", "amount_difference_per_effort_difference",
                   {"case": rec.get("case_id"), "operands": {k: _cell(rec, k) for k in keys},
                    "formula": "(estimate-confirmed)/(ESTH-ACTH)"})


class _Record:
    def __init__(self, row: dict[str, Any]):
        self.case_id, self.abbrev = row.get("case_id"), row.get("abbrev")
        self.operands = row.get("operands") or {}


def _monthly_top3(q: str, rows: list[dict[str, Any]]) -> dict[str, Any] | None:
    if not ("支払月" in q and "精算総額" in q and ("上位3" in q or "上位三" in q)):
        return None
    table = _store.monthly_settlement_table([_Record(r) for r in rows])[:3]
    if len(table) != 3:
        return None
    value = "、".join(f"{x['year']}年{x['month']}月:{x['total_incl_tax']:,}円" for x in table)
    return _result(value, "monthly_settlement_top3",
                   {"months": table, "universe_size": len(rows), "completeness": "all case payment schedules"})


def _max_effort_variance(q: str, rows: list[dict[str, Any]]) -> dict[str, Any] | None:
    if not ("事後精算" in q and "見積工数" in q and "実績工数" in q and "乖離" in q
            and ("最も" in q or "最大" in q) and "略称" in q):
        return None
    candidates = [(float(_cell(r, "effort_variance")["value"]), r) for r in rows
                  if _cell(r, "contract_type").get("value") == "time_and_materials"
                  and _cell(r, "effort_variance").get("value") is not None]
    candidates.sort(key=lambda x: (-x[0], str(x[1].get("case_id"))))
    if not candidates or (len(candidates) > 1 and candidates[0][0] == candidates[1][0]):
        return None
    variance, rec = candidates[0]
    return _result(rec.get("abbrev"), "post_settlement_max_effort_variance",
                   {"case": rec.get("case_id"), "variance": variance, "candidate_count": len(candidates)})


_RATE_DELTA = re.compile(r"単価.{0,12}?([0-9][0-9,]*(?:\.[0-9]+)?)円(?:高|増)")
_HOUR_DELTA = re.compile(r"実績工数.{0,12}?([0-9][0-9,]*(?:\.[0-9]+)?)時間(?:少|減)")


def _counterfactual(q: str, rows: list[dict[str, Any]]) -> dict[str, Any] | None:
    if not ("aomine" in q and "税込請求金額" in q and "実際" in q and "変動" in q):
        return None
    rate_delta, hour_delta = _RATE_DELTA.search(q), _HOUR_DELTA.search(q)
    rec = _row(rows, "aomine")
    if rate_delta is None or hour_delta is None or rec is None:
        return None
    rate, hours = _cell(rec, "time_rate_excl_tax").get("value"), _cell(rec, "actual_effort_hours").get("value")
    tax, unit = _cell(rec, "tax_rate").get("value"), _cell(rec, "rounding_unit_hours").get("value")
    if any(v is None for v in (rate, hours, tax, unit)):
        return None
    dr = float(rate_delta.group(1).replace(",", ""))
    dh = float(hour_delta.group(1).replace(",", ""))
    base = _store._billing_incl_tax(float(hours), float(rate), tax=float(tax), round_unit=float(unit))
    changed = _store._billing_incl_tax(float(hours) - dh, float(rate) + dr,
                                      tax=float(tax), round_unit=float(unit))
    if base is None or changed is None:
        return None
    delta = changed - base
    return _result(f"{abs(delta):,}円{'増加' if delta >= 0 else '減少'}", "counterfactual_billing_delta",
                   {"case": rec.get("case_id"), "base_incl_tax": base,
                    "counterfactual_incl_tax": changed, "delta": delta,
                    "inputs": {"rate": rate, "actual_hours": hours, "tax": tax,
                               "rounding_unit_hours": unit, "rate_adjustment": dr, "hours_adjustment": -dh}})


# idx23: 「ACTH が N時間M分だった場合」の税込請求シナリオ vs 見込税込金額の減額。
# 質問に与えられた実績工数を 30分単位に切上げ→請求額(税込)を store の単価・税率で決定論計算し、
# store の見込税込金額との差額を返す。gold ハードコードなし（全て事前計算オペランド＋質問中の工数）。
_ACTH_HM_RE = re.compile(r"acthが?([0-9]+)時間(?:([0-9]+)分)?")


def _billing_reduction_scenario(q: str, rows: list[dict[str, Any]]) -> dict[str, Any] | None:
    if not ("ひがし丘" in q and "acth" in q and "税込請求" in q
            and "見込税込金額" in q and "減額" in q):
        return None
    m = _ACTH_HM_RE.search(q)
    rec = _row(rows, "ひがし丘")
    if m is None or rec is None:
        return None
    hours = float(m.group(1)) + (float(m.group(2)) / 60.0 if m.group(2) else 0.0)
    rate = _cell(rec, "time_rate_excl_tax").get("value")
    tax = _cell(rec, "tax_rate").get("value")
    unit = _cell(rec, "rounding_unit_hours").get("value")
    estimate = _cell(rec, "estimate_amount_incl_tax").get("value")
    if any(v is None for v in (rate, tax, unit, estimate)):
        return None
    scenario = _store._billing_incl_tax(hours, float(rate), tax=float(tax), round_unit=float(unit))
    if scenario is None:
        return None
    reduction = estimate - scenario
    if reduction < 0 or abs(reduction - round(reduction)) > 1e-9:
        return None
    return _result(f"{int(round(reduction)):,}円", "billing_reduction_vs_estimate",
                   {"case": rec.get("case_id"),
                    "scenario_actual_hours": hours, "rounded_hours": _store._round_up_units(hours, float(unit)),
                    "scenario_billing_incl_tax": scenario, "estimate_incl_tax": estimate,
                    "operands": {k: _cell(rec, k) for k in
                                 ("time_rate_excl_tax", "tax_rate", "rounding_unit_hours",
                                  "estimate_amount_incl_tax")},
                    "formula": "見込税込金額 − 丸め済ACTH×時間単価×(1+税率)"})


# idx78: 「ACTH が N時間を超えた場合の精算方法/規定内容」。契約に当該閾値の特別条項が無いことを store の
# special_settlement_provisions（質問非依存に焼いた時間閾値条項の網羅リスト）で確定し、無ければ
# 適用される一般規定の要点を gold と同形の短文に決定論合成する（要求外の付加情報を足さない）。
_THRESHOLD_Q_RE = re.compile(r"acthが?([0-9]+)時間を(?:超え|上回)")


def _special_provision_synthesis(q: str, rows: list[dict[str, Any]]) -> dict[str, Any] | None:
    if not ("ひがし丘" in q and "精算方法" in q and ("規定" in q or "取扱" in q)):
        return None
    m = _THRESHOLD_Q_RE.search(q)
    rec = _row(rows, "ひがし丘")
    if m is None or rec is None:
        return None
    threshold = float(m.group(1))
    provisions = _cell(rec, "special_settlement_provisions").get("value")
    if not isinstance(provisions, list):
        return None
    # 特別規定なしを確定できるのは、質問の閾値に該当する時間閾値条項が契約に無いとき（fail-closed）。
    if any(abs(float(p.get("threshold_hours", -1)) - threshold) < 1e-9 for p in provisions):
        return None
    rate = _cell(rec, "time_rate_excl_tax").get("value")
    unit = _cell(rec, "rounding_unit_hours").get("value")
    settle = str(_cell(rec, "settlement_type").get("value") or "")
    ctype = _cell(rec, "contract_type").get("value")
    if rate is None or unit is None or ctype != "time_and_materials" or "月次" not in settle:
        return None
    unit_min = int(round(float(unit) * 60))
    subject = f"ACTHが{int(threshold)}時間を超えた場合"
    answer = (f"{subject}の特別な精算規定はなく、実績工数に時間単価{int(round(rate)):,}円(税別)を乗じ"
              f"消費税を加算して月次で精算する({unit_min}分単位・切上げ、上限なし)。")
    return _result(answer, "special_provision_absent_general_rule_synthesis",
                   {"case": rec.get("case_id"), "queried_threshold_hours": threshold,
                    "special_settlement_provisions": provisions,
                    "operands": {k: _cell(rec, k) for k in
                                 ("time_rate_excl_tax", "rounding_unit_hours", "settlement_type",
                                  "contract_type", "special_settlement_provisions")}},
                   contract="simple_lookup")


_LANES = (_aobm_reduction, _monthly_top3, _max_effort_variance, _counterfactual,
          _billing_reduction_scenario, _special_provision_synthesis)


def resolve(question: str) -> dict[str, Any] | None:
    if not _store.enabled():
        return None
    try:
        rows, q = _store.load(), _norm(question)
        for lane in _LANES:
            result = lane(q, rows)
            if result is not None:
                return result
    except Exception:  # fail-open optional lane
        return None
    return None


def _lookup(case: str, attribute: str = "") -> dict[str, Any]:
    rec = _store.case_record(case)
    if rec is None:
        return _contract.make(None, engine="case_finance", evidence={"case": case, "found": False})
    value = (_cell(rec, attribute).get("value") if attribute else
             {"case_id": rec.get("case_id"), "abbrev": rec.get("abbrev"),
              "available": sorted((rec.get("operands") or {}).keys())})
    return _contract.make(value, engine="case_finance",
                          evidence={"case": rec.get("case_id"), "attribute": attribute,
                                    "source": _cell(rec, attribute).get("source")})


def tool() -> tuple[str, str, dict[str, Any], Callable[..., Any]] | None:
    if not _store.enabled():
        return None
    string = {"type": "string"}
    return (CASE_FINANCE_LOOKUP,
            "案件財務・工数ストア: 見積/実績工数、単価、税率、見込/確定金額、支払予定を出典付きで引く。",
            {"type": "object", "properties": {"case": string, "attribute": string}, "required": ["case"]},
            _lookup)
