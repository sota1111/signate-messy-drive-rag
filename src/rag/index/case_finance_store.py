"""案件財務・工数派生ストア — 全案件 × 契約/提案/最終報告の財務・工数オペランドを質問非依存で事前計算する
事実層 (SOT-2654 / 事前計算事実層 拡張).

親 SOT-2651 cycle4 クラスタC の Sonnet gold100 abstain — idx37/40/55/76/98（numeric/simple_lookup・
BUDGET_EXHAUSTED）— は、実行時に「金額・工数・単価・税率・支払スケジュールのオペランドを収集 → 式構築 →
実行 → 検算」を 12〜18 ターンで完了できずに枯渇していた。cycle3 idx67（提案/FR金額の決定論列挙）と同じ勝ち筋
を **金額・工数・支払・単価へ横展開** する。

方針: **実行時に証拠を探すのではなく、事実をオフラインで網羅抽出しておく**。全案件 × 標準の財務/工数属性
（見込工数 ESTH / 実績工数 ACTH / 時間単価 RATE / 税率 / 見込金額（税込）/ 確定金額（税込）/ 支払予定月と
金額 / 精算方式）を index 時に一度だけ決定論抽出し、派生量（案件別 見積 vs 実績 工数乖離、月次精算総額、
train データ欠損行数）まで焼く。横断比較・反実仮想計算は「テーブルの決定論 lookup + 制限された算術」に帰着する。

**正当性の歯止め（必読）**: 事前計算は **質問を見ない網羅計算** に限定する — 全案件 × 標準属性を機械的に
抽出するのであり、特定設問を狙った選別計算・gold 値のハードコードは一切しない。設問→必要属性の対応表
(:func:`finance_hard_core_coverage`) は *ビルドレポート用の診断* にすぎず、ストア本体
(``case_finance.jsonl``) は完全に質問非依存。数値は decimal-aware 抽出＋独立再計算で確定し、確定できない
セルは ``value=None`` + ``reason``（欠測を偽装しない）。

Design invariants（既存の兄弟事前ストア — case_master / derived_metrics / report_attr_store — に倣う）:
  * **全案件カバレッジ + universe 証明.** :func:`build` は契約フォルダを持つ全案件（= case_master と同じ
    universe）を列挙し、行数 = universe を保証する。
  * **決定論・LLM フリー.** 全値はコーパスからの決定論抽出（正規表現 / decimal 算術）で確定。Gemini 呼び出し
    なし・ネットワークなし・再現バイト一致。抽出器は case_master と同じセルに一致するよう既存正答を尊重する。
  * **各値に出典必須.** 値には ``source: {doc_id, snippet, basis}`` を付ける。確定金額（税込）のように導出値は
    ``basis`` に導出式（実績工数×時間単価×(1+税率)）を明記する（gold 由来ではないことを honest に記録）。
  * **Opt-in at serve time.** :func:`enabled` (``RAG_CASE_FINANCE_STORE``) は *実行時参照のみ* を gate する。
    アーティファクトは index 時に常に additive・guarded に（再）ビルドされる。既定 OFF ⇒ champion serve path は
    バイト一致。読み出し/配線は :mod:`src.rag.agent.case_finance_lane`。
  * **再配布安全.** ストアは財務/工数の集計値 + 出典のみ。パスワード等の秘密値は絶対に入れない。
"""
from __future__ import annotations

import importlib
import json
import math
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

from config import settings
from src.rag.corpus import FileRef, nfc, walk

# case_master の抽出器・universe 規則・金額パターンを流用（値の一貫性を担保）。
_cm = importlib.import_module("src.rag.index.case_master")
_dm = importlib.import_module("src.rag.index.derived_metrics")
ca = importlib.import_module("src.rag.tools.corpus_aggregate")

SCHEMA = "case-finance"
SCHEMA_VERSION = 1

_ON = {"1", "true", "yes", "on"}

# コーパス全体の消費税率（決裁基準・全契約で一貫: 税抜×1.10=税込）。質問非依存の定数。
_TAX_RATE = 0.10
# 工数丸め単位（契約書「工数計上は30分単位」「30分未満は30分に切り上げ」）。反実仮想 billing の決定論丸め。
_ROUND_UNIT_HOURS = 0.5

_REL_TOL = 1e-6
_ABS_TOL = 1e-6


def enabled() -> bool:
    """True when the serve path should consult the case-finance store (default OFF — opt-in)."""
    return os.getenv("RAG_CASE_FINANCE_STORE", "0").strip().lower() in _ON


def default_out_path() -> Path:
    return settings.ARTIFACTS_DIR / "case_finance.jsonl"


def report_out_path() -> Path:
    return settings.ARTIFACTS_DIR / "case_finance_build_report.json"


# --------------------------------------------------------------------------- source / cell helpers
def _source(doc_id: str, *, snippet: str | None = None, basis: str | None = None) -> dict[str, Any]:
    src: dict[str, Any] = {"doc_id": nfc(doc_id)}
    if snippet is not None:
        src["snippet"] = snippet[:160]
    if basis is not None:
        src["basis"] = basis
    return src


def _cell(value: Any, source: dict[str, Any]) -> dict[str, Any]:
    return {"value": value, "source": source}


def _missing(reason: str) -> dict[str, Any]:
    return {"value": None, "source": None, "reason": reason}


def _extract_text(ref: FileRef) -> str:
    try:
        doc = ca._extract(ref, caption_images=False)  # no vision/network for a text field scan
        return getattr(doc, "text", "") or ""
    except Exception:  # noqa: BLE001 — one unreadable doc must not sink the row
        return ""


def _texts_for(project: str, refs: Sequence[FileRef], categories: set[str]) -> list[tuple[FileRef, str]]:
    """(ref, text) for a project's documents in the given categories, drafts/old excluded."""
    out: list[tuple[FileRef, str]] = []
    for r in refs:
        if r.project != project or r.category not in categories:
            continue
        low = r.name.lower()
        if "draft" in low or "/old/" in ("/" + r.rel.lower()):
            continue
        out.append((r, _extract_text(r)))
    return out


# --------------------------------------------------------------------------- decimal-aware extractors
_DEC = r"([0-9][0-9,]*(?:\.[0-9]+)?)"  # decimal-aware yen/hours figure (工数 156.5 のような小数を許容)
_HOUR_UNIT = r"\s*(?:時間|h)"


def _to_num(s: str) -> float:
    return float(s.replace(",", ""))


def _labeled_number(texts: Sequence[tuple[FileRef, str]], label: str, *,
                    unit: str) -> tuple[float, dict[str, Any]] | None:
    """A label whose number (with the required ``unit``) sits on the SAME line or the next 2 lines.

    Precision-first (case_master._labeled_amount と同じ規律): all matches for one label must collapse to a
    single distinct value, else the label is ambiguous ⇒ ``None``. The unit is REQUIRED on the value line
    so a bare adjacent number never leaks in (工数抽出のノイズ源を断つ)."""
    label_re = re.compile(label)
    same_re = re.compile(label + r"\s*[：:|＝=]?\s*[¥￥]?\s*" + _DEC + unit)
    val_re = re.compile(r"^\s*[¥￥]?\s*" + _DEC + unit + r"\s*$")
    values: dict[float, dict[str, Any]] = {}
    for ref, text in texts:
        lines = text.splitlines()
        for i, line in enumerate(lines):
            if not label_re.search(line):
                continue
            m = same_re.search(line)
            found: float | None = None
            snippet = line.strip()
            if m:
                found = _to_num(m.group(1))
            else:
                for nxt in lines[i + 1:i + 3]:
                    vm = val_re.match(nxt)
                    if vm:
                        found = _to_num(vm.group(1))
                        snippet = (line.strip() + " / " + nxt.strip())[:160]
                        break
            if found is not None and found not in values:
                values[found] = _source(ref.rel, snippet=snippet)
    if len(values) == 1:
        v, src = next(iter(values.items()))
        return v, src
    return None


_ESTH_LABELS = [r"想定総工数", r"見込工数", r"見積工数", r"想定工数"]
_ACTH_LABELS = [r"実績工数"]
_RATE_LABELS = [r"時間単価", r"単価"]


def _extract_effort(texts: Sequence[tuple[FileRef, str]], labels: Sequence[str]) -> tuple[float, dict[str, Any]] | None:
    for lab in labels:
        hit = _labeled_number(texts, lab, unit=_HOUR_UNIT)
        if hit is not None:
            return hit
    return None


def _extract_rate(texts: Sequence[tuple[FileRef, str]]) -> tuple[float, dict[str, Any]] | None:
    """時間単価（税抜, 円/時間）. 「時間単価：20,000円（税抜）/時」「¥25,000/時間」等。"""
    yen = r"\s*(?:円|/\s*時間?|/時)"
    for lab in _RATE_LABELS:
        hit = _labeled_number(texts, lab, unit=yen)
        if hit is not None:
            return hit
    # 「¥25,000/時間」形式（ラベルなし・スライド分割）
    rate_re = re.compile(r"[¥￥]\s*" + _DEC + r"\s*/\s*時間")
    vals: dict[float, dict[str, Any]] = {}
    for ref, text in texts:
        for line in text.splitlines():
            m = rate_re.search(line)
            if m:
                v = _to_num(m.group(1))
                vals.setdefault(v, _source(ref.rel, snippet=line.strip()))
    if len(vals) == 1:
        v, src = next(iter(vals.items()))
        return v, src
    # 「時間単価は25,000円（税別）とする」等、ラベルと数値の間に格助詞(は/が/を/に/も)が入り、
    # 数値の後ろに税別注記が続く本文形式（ひがし丘 6.2条など）。同一契約内で単価が一意なときのみ確定。
    particle_re = re.compile(r"時間単価[はがをにも、\s]*[¥￥]?\s*" + _DEC + r"\s*円")
    pvals: dict[float, dict[str, Any]] = {}
    for ref, text in texts:
        for line in text.splitlines():
            m = particle_re.search(line)
            if m:
                pvals.setdefault(_to_num(m.group(1)), _source(ref.rel, snippet=line.strip()))
    if len(pvals) == 1:
        v, src = next(iter(pvals.items()))
        return v, src
    return None


# 工数（時間）閾値付きの条件条項（「N時間を超える/上回る場合」）。質問非依存に契約から網羅抽出する。
# rounding 規則の「30分を超え…」等は 分単位なので拾わない（時間閾値のみ）。idx78 型の
# 「特別な精算規定の有無」判定に使う: 質問の閾値が一つも該当しなければ特別規定なしと確定できる。
_SPECIAL_PROVISION_RE = re.compile(r"(\d[\d,]*)\s*時間を(?:超え|上回)(?:た|る)?")


def _extract_special_provisions(texts: Sequence[tuple[FileRef, str]]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    seen: set[float] = set()
    for ref, text in texts:
        for line in text.splitlines():
            for m in _SPECIAL_PROVISION_RE.finditer(line):
                hours = _to_num(m.group(1))
                if hours in seen:
                    continue
                seen.add(hours)
                out.append({"threshold_hours": hours, "clause": line.strip()[:200],
                            "source": _source(ref.rel, snippet=line.strip()[:200])})
    return out


# --------------------------------------------------------------------------- payment schedule
# 支払スケジュール行: 「N | マイルストーン | 比率 | 税抜 | 税額 | 税込 | 期限 [| date]」または本文中の 期限 date。
_DATE_RE = re.compile(r"(20\d{2})[-/年](\d{1,2})[-/月](\d{1,2})")
_INCL_AMOUNT_RE = re.compile(r"([0-9][0-9,]{4,})\s*円")


def _extract_payment_schedule(texts: Sequence[tuple[FileRef, str]]) -> list[dict[str, Any]]:
    """支払（精算）予定: (税込金額, 支払年月, 出典) の列（質問非依存の網羅抽出, best-effort）。

    行に「税込金額」と「20YY-MM-DD の支払/検収期限」が同居する行だけを拾う（精算総額の月次集計の素材）。
    複数案件横断の月次集計は :func:`build_report` の cross-case テーブルで組む。"""
    out: list[dict[str, Any]] = []
    seen: set[tuple[int, int, int]] = set()
    for ref, text in texts:
        for line in text.splitlines():
            # Schedule rows start with their sequence number. Contract summary/header rows also contain
            # pipes, a date and the gross amount; accepting those double-counts a contract as a payment.
            if re.match(r"^\s*(?:第\s*)?\d+(?:\s*回)?\s*\|", line) is None:
                continue
            if not ("支払" in line or "精算" in line or "検収" in line or "着手金" in line):
                continue
            dm = _DATE_RE.search(line)
            if dm is None:
                continue
            amounts = [int(a.replace(",", "")) for a in _INCL_AMOUNT_RE.findall(line)]
            if not amounts:
                continue
            year, month, day = int(dm.group(1)), int(dm.group(2)), int(dm.group(3))
            incl = max(amounts)  # 税込 is the largest figure in a 税抜|税額|税込 row
            key = (year, month, incl)
            if key in seen:
                continue
            seen.add(key)
            out.append({"amount_incl_tax": incl, "year": year, "month": month, "day": day,
                        "source": _source(ref.rel, snippet=line.strip())})
    return out


# --------------------------------------------------------------------------- derived arithmetic
def _round_up_units(hours: float, unit: float = _ROUND_UNIT_HOURS) -> float:
    """契約の工数丸め: unit(=0.5h=30分) 単位に切り上げ（「30分未満は30分に切り上げ」）。"""
    if unit <= 0:
        return hours
    return math.ceil(round(hours / unit, 9)) * unit


def _billing_incl_tax(actual_hours: float, rate_excl: float, *, tax: float = _TAX_RATE,
                      round_unit: float = _ROUND_UNIT_HOURS) -> int | None:
    """T&M 税込請求額 = 丸め済み実績工数 × 時間単価(税抜) × (1+税率)。整数円のときのみ確定（fail-closed）。"""
    rounded = _round_up_units(actual_hours, round_unit)
    excl = rounded * rate_excl
    incl = excl * (1.0 + tax)
    # 独立検算: 税抜を先に整数化してから税込（税抜が整数円になる契約前提）。
    if abs(excl - round(excl)) > 1e-6:
        return None
    incl_int = round(excl) * (100 + int(round(tax * 100))) / 100
    if abs(incl - incl_int) > 1e-3 or abs(incl_int - round(incl_int)) > 1e-6:
        return None
    return int(round(incl_int))


# --------------------------------------------------------------------------- record construction
STANDARD_OPERANDS = [
    "contract_type", "settlement_type", "time_rate_excl_tax", "tax_rate", "rounding_unit_hours",
    "estimated_effort_hours", "actual_effort_hours", "effort_variance",
    "estimate_amount_incl_tax", "confirmed_amount_incl_tax",
    "missing_row_count", "payment_schedule", "special_settlement_provisions",
]


@dataclass
class CaseFinance:
    case_id: str
    abbrev: str | None
    operands: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return {"case_id": self.case_id, "abbrev": self.abbrev, "operands": self.operands}


def _train_missing_rows(project: str, refs: Sequence[FileRef]) -> tuple[int, dict[str, Any]] | None:
    """train 表で 1 つ以上欠損のある行数（質問非依存の標準派生量, idx24 型）。"""
    ref = _dm._train_ref(project, refs)
    if ref is None:
        return None
    try:
        from src.rag.tools.compute_sandbox import _read_frame
        df, _t, _r = _read_frame(ref, None)
    except Exception:  # noqa: BLE001
        return None
    try:
        miss = int(df.isna().any(axis=1).sum())
    except Exception:  # noqa: BLE001
        return None
    return miss, _source(ref.rel, basis="pandas: (df.isna().any(axis=1)).sum() 欠損≥1行数")


def _make_record(project: str, refs: Sequence[FileRef], glossary: Any) -> CaseFinance:
    ops: dict[str, Any] = {}
    abbrev = ca._abbrev_for(project, glossary)

    contract_texts = [(r, _extract_text(r)) for r in ca._contract_refs(project, refs)]
    proposal_texts = _texts_for(project, refs, {"proposal"})
    report_texts = _texts_for(project, refs, {"report"})
    joined_contract = "\n".join(t for _, t in contract_texts)

    fixed = ca._is_fixed(joined_contract)
    ctype = "固定金額" if fixed else "time_and_materials"
    ops["contract_type"] = _cell(ctype, _source(_first_doc(contract_texts),
                                                basis="固定金額の記載有無→契約形態"))

    # 精算方式（case_master と同じ抽出）。
    settlement = _cm._extract_settlement(proposal_texts + contract_texts)
    ops["settlement_type"] = settlement if settlement else _missing("精算方式を抽出できない")

    # 時間単価（税抜）。
    rate = _extract_rate(contract_texts + proposal_texts)
    ops["time_rate_excl_tax"] = (_cell(rate[0], rate[1]) if rate
                                 else _missing("時間単価（税抜）を契約書/提案書から一意抽出できない"))

    ops["tax_rate"] = _cell(_TAX_RATE, _source(_cm._DECREE_DOC_ID,
                                               basis="コーパス全体の消費税率10%（税抜×1.10=税込）"))
    ops["rounding_unit_hours"] = (
        _cell(_ROUND_UNIT_HOURS, _source(_first_doc(contract_texts), basis="工数計上30分単位・端数切り上げ"))
        if re.search(r"30\s*分単位|30分未満", joined_contract)
        else _missing("工数丸め単位（30分等）を契約書から確認できない"))

    esth = _extract_effort(proposal_texts + contract_texts, _ESTH_LABELS)
    ops["estimated_effort_hours"] = (_cell(esth[0], esth[1]) if esth
                                     else _missing("見込/想定/見積工数（時間）を一意抽出できない"))
    acth = _extract_effort(report_texts + contract_texts, _ACTH_LABELS)
    ops["actual_effort_hours"] = (_cell(acth[0], acth[1]) if acth
                                  else _missing("実績工数（時間）を最終報告から一意抽出できない"))

    # 案件別 見積vs実績 工数乖離（質問非依存の標準派生量, idx55 型）。
    if esth and acth:
        var = abs(acth[0] - esth[0])
        ops["effort_variance"] = _cell(round(var, 6), _source(
            _first_doc(report_texts) or _first_doc(contract_texts),
            basis=f"|実績工数{acth[0]} − 見込工数{esth[0]}| = {round(var, 6)}"))
    else:
        ops["effort_variance"] = _missing("見込工数と実績工数の双方が揃わないため乖離を算出できない")

    # 見込金額（税込）= 提案/契約の税込見込額（case_master の提案金額抽出器を流用）。
    prop = (_cm._first_unique_int(proposal_texts, _cm._PROPOSAL_AMOUNT_PATTERNS)
            or _cm._labeled_amount(proposal_texts, _cm._PROPOSAL_AMOUNT_LABELS)
            or _cm._first_unique_int(contract_texts, _cm._PROPOSAL_AMOUNT_PATTERNS))
    ops["estimate_amount_incl_tax"] = (_cell(prop[0], prop[1]) if prop
                                       else _missing("見込金額（税込）を一意抽出できない"))

    # 確定金額（税込）: T&M は 実績工数×時間単価×(1+税率)（決定論導出・出典は導出式）。固定金額は FR/契約額。
    confirmed = None
    if not fixed and acth and rate:
        billed = _billing_incl_tax(acth[0], rate[0])
        if billed is not None:
            confirmed = _cell(billed, _source(
                acth[1]["doc_id"],
                basis=f"確定金額（税込）=丸め済実績工数{_round_up_units(acth[0])}×時間単価{rate[0]}×(1+{_TAX_RATE})"))
    if confirmed is None:
        fr = (_cm._first_unique_int(report_texts, _cm._FR_AMOUNT_PATTERNS)
              or _cm._labeled_amount(report_texts, _cm._FR_AMOUNT_LABELS)
              or _cm._fr_amount_from_excl(report_texts))
        confirmed = (_cell(fr[0], fr[1]) if fr
                     else _missing("確定金額（税込）を導出/抽出できない"))
    ops["confirmed_amount_incl_tax"] = confirmed

    # train 欠損≥1行数（idx24 型）。
    miss = _train_missing_rows(project, refs)
    ops["missing_row_count"] = (_cell(miss[0], miss[1]) if miss
                                else _missing("train 表が読めない/存在しないため欠損行数を算出できない"))

    # 支払スケジュール（月次精算集計の素材, idx40 型）。
    sched = _extract_payment_schedule(contract_texts + proposal_texts)
    ops["payment_schedule"] = (_cell(sched, _source(_first_doc(contract_texts),
                                                    basis="支払/精算予定行（税込金額×支払年月）の網羅抽出"))
                               if sched else _missing("支払スケジュール行を抽出できない"))

    # 工数閾値付き条件条項（「N時間を超える場合」）の網羅列挙（idx78 型の特別規定有無判定の素材）。
    # 空リストは「そのような閾値条項が契約に無い」= 特別な精算規定なし、を honest に表す（欠測ではない）。
    provisions = _extract_special_provisions(contract_texts)
    ops["special_settlement_provisions"] = _cell(
        provisions, _source(_first_doc(contract_texts),
                            basis="契約書中の『N時間を超える/上回る場合』条項の網羅抽出（時間閾値のみ）"))

    return CaseFinance(case_id=nfc(project), abbrev=abbrev, operands=ops)


def _first_doc(texts: Sequence[tuple[FileRef, str]]) -> str:
    return texts[0][0].rel if texts else ""


# --------------------------------------------------------------------------- build
def build(refs: list[FileRef] | None = None, *, out: Path | None = None,
          glossary: Any = None, write_report: bool = True) -> dict[str, Any]:
    """Precompute + persist the case-finance store (one row per 案件) with per-cell provenance.

    Deterministic and LLM-free. Additive & guarded — a failure here never corrupts the retrieval index
    (the caller in :mod:`src.rag.index` swallows exceptions)."""
    refs = refs if refs is not None else walk()
    g = glossary if glossary is not None else _cm._load_glossary()
    universe = _cm._case_universe(refs)
    records = [_make_record(p, refs, g) for p in universe]

    counts = write_store(records, out)
    report = build_report(records, universe, out or default_out_path())
    if write_report:
        rp = report_out_path()
        rp.parent.mkdir(parents=True, exist_ok=True)
        rp.write_text(json.dumps(report, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
                      encoding="utf-8")
    return {**counts, "report": report}


def write_store(records: Sequence[CaseFinance], path: Path | None = None) -> dict[str, int]:
    """Atomically write a reproducible JSONL store (schema header + case_id-sorted rows)."""
    out = path or default_out_path()
    out.parent.mkdir(parents=True, exist_ok=True)
    ordered = sorted(records, key=lambda r: r.case_id)
    tmp = out.with_suffix(out.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8", newline="\n") as handle:
        handle.write(json.dumps({"schema": SCHEMA, "version": SCHEMA_VERSION},
                                ensure_ascii=False, sort_keys=True) + "\n")
        for rec in ordered:
            handle.write(json.dumps(rec.to_dict(), ensure_ascii=False, sort_keys=True) + "\n")
    tmp.replace(out)
    reset_cache()
    return {"cases": len(ordered)}


def _op_value(rec: CaseFinance, name: str) -> Any:
    return (rec.operands.get(name) or {}).get("value")


def monthly_settlement_table(records: Sequence[CaseFinance]) -> list[dict[str, Any]]:
    """全案件の支払スケジュールを (年,月) で集計した精算総額テーブル（降順, idx40 型の素材）。"""
    agg: dict[tuple[int, int], dict[str, Any]] = {}
    for rec in records:
        sched = _op_value(rec, "payment_schedule") or []
        for item in sched:
            key = (item["year"], item["month"])
            slot = agg.setdefault(key, {"year": key[0], "month": key[1], "total_incl_tax": 0,
                                        "cases": []})
            slot["total_incl_tax"] += int(item["amount_incl_tax"])
            slot["cases"].append({"case_id": rec.case_id, "abbrev": rec.abbrev,
                                  "amount_incl_tax": int(item["amount_incl_tax"])})
    return sorted(agg.values(), key=lambda s: (-s["total_incl_tax"], s["year"], s["month"]))


def effort_variance_table(records: Sequence[CaseFinance], *, settlement_type: str | None = None
                          ) -> list[dict[str, Any]]:
    """案件別 見積vs実績 工数乖離テーブル（降順, idx55 型）。settlement_type 指定で精算方式フィルタ。"""
    out: list[dict[str, Any]] = []
    for rec in records:
        var = _op_value(rec, "effort_variance")
        if var is None:
            continue
        st = _op_value(rec, "settlement_type")
        if settlement_type is not None and st != settlement_type:
            continue
        out.append({"case_id": rec.case_id, "abbrev": rec.abbrev, "variance": var,
                    "estimated": _op_value(rec, "estimated_effort_hours"),
                    "actual": _op_value(rec, "actual_effort_hours"), "settlement_type": st})
    return sorted(out, key=lambda x: (-x["variance"], x["case_id"]))


def missing_row_table(records: Sequence[CaseFinance]) -> list[dict[str, Any]]:
    """案件別 train 欠損≥1行数テーブル（降順, idx24 型）。"""
    out = [{"case_id": r.case_id, "abbrev": r.abbrev, "missing_row_count": _op_value(r, "missing_row_count")}
           for r in records if _op_value(r, "missing_row_count") is not None]
    return sorted(out, key=lambda x: (-x["missing_row_count"], x["case_id"]))


def build_report(records: Sequence[CaseFinance], universe: Sequence[str], out_path: Path) -> dict[str, Any]:
    n = len(records) or 1
    fill: dict[str, int] = {}
    for op in STANDARD_OPERANDS:
        fill[op] = sum(1 for r in records if _op_value(r, op) not in (None, [], {}))
    return {
        "certificate": {
            "schema": SCHEMA, "schema_version": SCHEMA_VERSION, "question_independent": True,
            "universe_basis": "case_master と同じ 01.契約 フォルダを持つ全案件（社内管理除く）",
            "case_count": len(records), "row_count_equals_universe": len(records) == len(universe),
            "computation_basis": ("全案件 × 標準財務/工数属性（工数/単価/税率/見込・確定金額/支払スケジュール）"
                                  "の decimal-aware 決定論抽出＋派生量（乖離/月次精算/欠損行数）。gold 非依存・LLM 非関与。"),
        },
        "operand_fill": {op: {"filled": fill[op], "missing": len(records) - fill[op],
                              "missing_rate": round((len(records) - fill[op]) / n, 4)}
                         for op in STANDARD_OPERANDS},
        "derived_tables": {
            "effort_variance_top": effort_variance_table(records)[:5],
            "missing_row_top": missing_row_table(records)[:5],
            "monthly_settlement_top": [{"year": s["year"], "month": s["month"],
                                        "total_incl_tax": s["total_incl_tax"], "n_cases": len(s["cases"])}
                                       for s in monthly_settlement_table(records)[:5]],
        },
        "finance_hard_core_coverage": finance_hard_core_coverage(records),
        "out_path": str(out_path),
    }


# --------------------------------------------------------------------------- hard-core coverage (diagnostic)
_FINANCE_HARD_CORE: dict[str, dict[str, Any]] = {
    "idx37": {"gist": "AOBM (見込金額−確定金額)税込 ÷ (ESTH−ACTH) の1時間あたり減少金額",
              "need": ["estimate_amount_incl_tax", "confirmed_amount_incl_tax",
                       "estimated_effort_hours", "actual_effort_hours"], "hint": "青葉バイオ"},
    "idx76": {"gist": "AOMINE 単価+2000/実績-11.2h の反実仮想 税込請求額の変動",
              "need": ["time_rate_excl_tax", "actual_effort_hours", "tax_rate", "rounding_unit_hours"],
              "hint": "青嶺"},
    "idx55": {"gist": "事後精算案件で 見積vs実績 工数乖離が最大の案件（主略称）",
              "need": ["effort_variance"], "hint": "青嶺", "cross_case": True},
    "idx24": {"gist": "train 欠損≥1行数が最多の案件（主略称）",
              "need": ["missing_row_count"], "hint": "青嶺", "cross_case": True},
    "idx40": {"gist": "支払月ごとの精算総額 上位3ヶ月と総額",
              "need": ["payment_schedule"], "hint": "", "cross_case": True},
    "idx98": {"gist": "TM案件の RATE 変更の想定年月日",
              "need": [], "hint": "",
              "note": "RATE 変更履歴（想定年月日）は契約/提案の明示記載が無く本ストアの網羅抽出対象外（棄権）。"},
}


def _find(records: Sequence[CaseFinance], hint: str) -> CaseFinance | None:
    for r in records:
        if hint and hint in r.case_id:
            return r
    return None


def finance_hard_core_coverage(records: Sequence[CaseFinance]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for idx, spec in _FINANCE_HARD_CORE.items():
        need = spec.get("need") or []
        rec = _find(records, spec.get("hint") or "")
        per: dict[str, bool] = {}
        if rec is not None:
            for op in need:
                per[op] = _op_value(rec, op) is not None
        out[idx] = {
            "gist": spec["gist"], "required_operands": need,
            "matched_case": rec.case_id if rec is not None else None,
            "cross_case": bool(spec.get("cross_case")),
            "operands_present": per,
            "solvable_from_store": bool(need) and bool(per) and all(per.values()),
            "note": spec.get("note", ""),
        }
    return out


# --------------------------------------------------------------------------- load / read (minimal API)
_LOAD_CACHE: dict[str, list[dict[str, Any]]] = {}


def reset_cache() -> None:
    _LOAD_CACHE.clear()


def load(path: Path | None = None) -> list[dict[str, Any]]:
    """Load the case-finance rows (memoized). ``[]`` when absent/unreadable/schema-mismatch (回帰ゼロ)."""
    out = path or default_out_path()
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
    except Exception:
        rows = []
    _LOAD_CACHE[key] = rows
    return rows


def operand(row: dict[str, Any], name: str) -> Any:
    return (row.get("operands", {}).get(name) or {}).get("value")


def case_record(case_hint: str, *, path: Path | None = None) -> dict[str, Any] | None:
    """The finance row for a 案件 (exact case_id preferred, else substring). ``None`` when absent."""
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
    print(json.dumps({"cases": summary["cases"],
                      "certificate": summary["report"]["certificate"],
                      "derived_tables": summary["report"]["derived_tables"]},
                     ensure_ascii=False, indent=2))
