"""人物×案件×役割×連絡先 全数結合マスタストア — 契約書の乙(受託)担当者・着手金・座席表(内線)・
契約書署名欄(甲=クライアント主担当者) を質問非依存でオフライン全数結合する事実層 (SOT-2713 / cycle11).

親 SOT-2708 cycle11 §3 で機械分類した「マスタ結合」クラスタ idx13/43/46 は cycle10 で全て MATCH だが
**Sonnet(claude-mcp) LLM 経路**（churn リスク）で通っていた。同経路は複数ツール（``corpus_aggregate`` の
staff/deposit 集計＋``seating_lookup`` の内線引き＋``contact_master`` の署名欄）を毎回組み合わせて解くため
非決定で揺れる。cycle8 の申し送り「次は LLM 経路 lookup の決定論昇格」に従い、その結合を **ビルド時に一度
全数計算**しておき、serve では純粋な決定論 lookup に帰着させる（SOT-2698 の再適用: 証拠は既存の決定論抽出器で
到達可 = 事前結合テーブル＋自動レーンだけで安価に churn を除去）。

対象設問（質問非依存の一般形）:
  * idx13 — 受託(乙=データアステル)側で最も多くの案件に関与する人物 → その内線番号。
  * idx43 — 案件の甲(クライアント)側 主担当者のフルネーム（契約書署名欄が権威ソース, SOT-2707 継承）。
  * idx46 — 着手金が最も高い案件の、指定役割(例 ES)担当者 → その内線番号。

方針: **実行時に証拠を探すのではなく、結合をオフラインで網羅計算しておく**。全案件 × 全人物 × 全役割 ×
連絡先属性を index 時に一度だけ決定論結合する（質問を見ない）:
  * 案件テーブル: 全案件（case_master と同じ 01.契約 universe）× {着手金(税込)・役割→乙担当者(``ES``/``PM``/
    ``DE``…)・甲主担当者(署名欄)・略称/別名}。抽出は既存の決定論抽出器を流用（``corpus_aggregate`` /
    ``contact_master_store``）。
  * 人物テーブル: 乙担当者を横断集計し、{関与案件集合・案件数・座席(内線/役割/POD)} を焼く。座席は
    ``seating_chart`` の reviewed ディレクトリ（pixel-hash pin, LLM フリー）を姓照合で結合する。
  * 派生集計: 人物あたり案件数（一票/案件, ``corpus_aggregate._staff_counts`` と同規律）。

正当性の歯止め（必読）:
  * **質問非依存の網羅計算のみ.** 全案件 × 標準属性を機械結合するのみ。特定設問を狙った選別・gold 値の
    ハードコードは一切しない。値は原文の構造化フィールド／既存決定論抽出器の出力を写経する。
  * **決定論・LLM フリー・再現バイト一致.** Gemini 呼び出しなし・ネットワークなし（座席は reviewed anchor,
    未知画像はフォールバックせず seat=None）。
  * **一意性（fail-closed）.** argmax（最多案件人物・最高着手金案件）が同率タイのとき、または座席が姓照合で
    一意に定まらないときは確定しない（読み出し側レーンが棄権）。
  * **Opt-in at serve time.** :func:`enabled` (``RAG_MASTER_JOIN_LOOKUP``) は *実行時参照のみ* を gate する。
    アーティファクトは index 時に常に additive・guarded に（再）ビルドされる。既定 OFF ⇒ champion serve path は
    バイト一致。読み出し/配線は :mod:`src.rag.agent.master_join_lane`。
"""
from __future__ import annotations

import importlib
import json
import os
import unicodedata
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Sequence

from config import settings
from src.rag.corpus import FileRef, nfc, walk

# case_master の universe 規則、corpus_aggregate の契約抽出器（着手金・乙担当者役割）、contact_master の
# 署名欄抽出、seating_chart の内線ディレクトリを流用（値の一貫性を担保）。
_cm = importlib.import_module("src.rag.index.case_master")
ca = importlib.import_module("src.rag.tools.corpus_aggregate")
_contact = importlib.import_module("src.rag.index.contact_master_store")

SCHEMA = "master-join"
SCHEMA_VERSION = 1

_ON = {"1", "true", "yes", "on"}


def enabled() -> bool:
    """True when the serve path should consult the master-join store (default OFF — opt-in)."""
    return os.getenv("RAG_MASTER_JOIN_LOOKUP", "0").strip().lower() in _ON


def default_out_path() -> Path:
    return settings.ARTIFACTS_DIR / "master_join.jsonl"


def report_out_path() -> Path:
    return settings.ARTIFACTS_DIR / "master_join_build_report.json"


def _surname(full_name: str) -> str:
    """姓トークン（座席表の姓ラベルと結合するため）。氏名は『姓 名』形式なので先頭トークンを姓とする。"""
    s = unicodedata.normalize("NFKC", str(full_name or "")).strip()
    if not s:
        return ""
    return s.replace("　", " ").split(" ")[0]


# --------------------------------------------------------------------------- seating directory (内線)
def _seating_index() -> dict[str, list[dict[str, Any]]]:
    """姓 → [座席レコード] の索引（reviewed anchor のみ; 未解決なら空 = seat 焼き込みなし）。

    ``seating_chart.build_directory`` は pixel-hash pin された reviewed ディレクトリ（LLM フリー・決定論）を
    返す。ビルド環境に座席表 pptx が無い等で解決できなければ空 index（seat=None が焼かれるだけ, 回帰ゼロ）。"""
    try:
        seating = importlib.import_module("src.rag.tools.seating_chart")
        directory = seating.build_directory()
    except Exception:  # noqa: BLE001 — 座席が引けなくても人物テーブルは案件集合だけで焼く
        return {}
    idx: dict[str, list[dict[str, Any]]] = {}
    for seat in getattr(directory, "seats", ()):  # reviewed Seat は姓のみを name に持つ
        key = _surname(seat.name)
        if key:
            idx.setdefault(key, []).append(
                {"ext": seat.ext, "role": seat.role, "pod": seat.pod, "name": seat.name})
    return idx


def _seat_for(full_name: str, seat_index: dict[str, list[dict[str, Any]]]) -> dict[str, Any] | None:
    """人物（フルネーム）の座席を姓照合で引く。姓が座席表に一意に無ければ None（fail-closed）。"""
    seats = seat_index.get(_surname(full_name)) or []
    return dict(seats[0]) if len(seats) == 1 else None


# --------------------------------------------------------------------------- case / person tables
@dataclass
class CaseRow:
    case_id: str
    abbrev: str | None
    aliases: list[str]
    deposit_incl_tax: int | None
    fixed: bool
    staff: dict[str, str]           # 役割コード(ES/PM/DE/BA/QA/LeadDS) → 乙担当者フルネーム
    client_contact: str | None      # 甲(クライアント)主担当者フルネーム（署名欄, 権威ソース）

    def to_dict(self) -> dict[str, Any]:
        return {"kind": "case", "case_id": self.case_id, "abbrev": self.abbrev,
                "aliases": self.aliases, "deposit_incl_tax": self.deposit_incl_tax,
                "fixed": self.fixed, "staff": dict(self.staff), "client_contact": self.client_contact}


@dataclass
class PersonRow:
    name: str
    case_count: int
    cases: list[dict[str, Any]]     # [{case_id, abbrev, role_code}]
    seat: dict[str, Any] | None     # {ext, role, pod, name} | None

    def to_dict(self) -> dict[str, Any]:
        return {"kind": "person", "name": self.name, "case_count": self.case_count,
                "cases": self.cases, "seat": self.seat}


def _aliases_for(company: str, glossary: Any) -> list[str]:
    """glossary company_aliases から案件の別名（略称・短縮社名）を集める（甲側 case bind に使う）。"""
    out: list[str] = []
    aliases = getattr(glossary, "company_aliases", {}) or {}
    # company_to_code は folder 名≠正式社名のことがあるので、両方向（含有）で正式社名を探す。
    for formal, names in aliases.items():
        if formal and (formal in company or company in formal):
            for n in names:
                if n and n not in out:
                    out.append(str(n))
    return out


def _build_case_rows(refs: Sequence[FileRef], glossary: Any) -> list[CaseRow]:
    contracts = ca.collect_contracts(refs=list(refs), glossary=glossary)
    rows: list[CaseRow] = []
    for pc in contracts:
        project = pc.project
        # 甲主担当者は契約書署名欄（権威ソース, SOT-2707 の抽出器を流用）。
        try:
            cmrec = _contact._make_record(project, refs, glossary)
            client_contact = (cmrec.operands.get("client_contact") or {}).get("value")
        except Exception:  # noqa: BLE001 — 署名欄が読めなくても他属性は焼く
            client_contact = None
        rows.append(CaseRow(
            case_id=nfc(project),
            abbrev=pc.abbrev,
            aliases=_aliases_for(project, glossary),
            deposit_incl_tax=pc.deposit,
            fixed=bool(pc.fixed),
            staff=dict(pc.staff),
            client_contact=client_contact,
        ))
    return rows


def _build_person_rows(case_rows: Sequence[CaseRow],
                       seat_index: dict[str, list[dict[str, Any]]]) -> list[PersonRow]:
    """乙担当者を横断集計（一票/案件, corpus_aggregate._staff_counts と同規律）。"""
    cases_of: dict[str, list[dict[str, Any]]] = {}
    order: dict[str, int] = {}
    for i, cr in enumerate(case_rows):
        # 一票/案件: 同一人物が一案件で複数役割でも案件数は 1 加算（役割ごとには cases に記録）。
        seen_in_case: set[str] = set()
        for role_code, name in sorted(cr.staff.items()):
            if not name:
                continue
            cases_of.setdefault(name, []).append(
                {"case_id": cr.case_id, "abbrev": cr.abbrev, "role_code": role_code})
            order.setdefault(name, i)
            seen_in_case.add(name)
    persons: list[PersonRow] = []
    for name in sorted(cases_of, key=lambda n: order[n]):
        entries = cases_of[name]
        distinct_cases = len({e["case_id"] for e in entries})
        persons.append(PersonRow(
            name=name, case_count=distinct_cases, cases=entries,
            seat=_seat_for(name, seat_index)))
    return persons


# --------------------------------------------------------------------------- build
def build(refs: list[FileRef] | None = None, *, out: Path | None = None,
          glossary: Any = None, write_report: bool = True) -> dict[str, Any]:
    """Precompute + persist the master-join store (case rows + person rows) — deterministic, LLM-free.

    Additive & guarded — a failure here never corrupts the retrieval index (the caller in
    :mod:`src.rag.index` swallows exceptions)."""
    refs = refs if refs is not None else walk()
    g = glossary if glossary is not None else _cm._load_glossary()
    seat_index = _seating_index()
    case_rows = _build_case_rows(refs, g)
    person_rows = _build_person_rows(case_rows, seat_index)

    counts = write_store(case_rows, person_rows, out)
    universe = _cm._case_universe(refs)
    report = build_report(case_rows, person_rows, universe, seat_index, out or default_out_path())
    if write_report:
        rp = report_out_path()
        rp.parent.mkdir(parents=True, exist_ok=True)
        rp.write_text(json.dumps(report, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
                      encoding="utf-8")
    return {**counts, "report": report}


def write_store(case_rows: Sequence[CaseRow], person_rows: Sequence[PersonRow],
                path: Path | None = None) -> dict[str, int]:
    """Atomically write a reproducible JSONL store (schema header + case rows + person rows)."""
    out = path or default_out_path()
    out.parent.mkdir(parents=True, exist_ok=True)
    ordered_cases = sorted(case_rows, key=lambda r: r.case_id)
    ordered_persons = sorted(person_rows, key=lambda r: r.name)
    tmp = out.with_suffix(out.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8", newline="\n") as handle:
        handle.write(json.dumps({"schema": SCHEMA, "version": SCHEMA_VERSION},
                                ensure_ascii=False, sort_keys=True) + "\n")
        for cr in ordered_cases:
            handle.write(json.dumps(cr.to_dict(), ensure_ascii=False, sort_keys=True) + "\n")
        for pr in ordered_persons:
            handle.write(json.dumps(pr.to_dict(), ensure_ascii=False, sort_keys=True) + "\n")
    tmp.replace(out)
    reset_cache()
    return {"cases": len(ordered_cases), "persons": len(ordered_persons)}


def build_report(case_rows: Sequence[CaseRow], person_rows: Sequence[PersonRow],
                 universe: Sequence[str], seat_index: dict[str, list[dict[str, Any]]],
                 out_path: Path) -> dict[str, Any]:
    seated = sum(1 for p in person_rows if p.seat is not None)
    with_deposit = sum(1 for c in case_rows if c.deposit_incl_tax is not None)
    with_client = sum(1 for c in case_rows if c.client_contact)
    top = max(person_rows, key=lambda p: p.case_count, default=None)
    return {
        "certificate": {
            "schema": SCHEMA, "schema_version": SCHEMA_VERSION, "question_independent": True,
            "universe_basis": "case_master と同じ 01.契約 フォルダを持つ全案件（社内管理除く）",
            "case_count": len(case_rows), "row_count_equals_universe": len(case_rows) == len(universe),
            "person_count": len(person_rows), "seat_resolved": seated,
            "seating_origin": "reviewed(pixel-hash pinned)" if seat_index else "unavailable",
            "computation_basis": (
                "全案件 × 乙担当者役割(ES/PM/DE/BA/QA/LeadDS)・着手金(税込)・甲主担当者(署名欄) を決定論結合し、"
                "乙担当者を横断集計して {関与案件集合・案件数・座席(内線/POD)} を焼く。座席は reviewed anchor を"
                "姓照合。gold 非依存・LLM 非関与・argmax 同率タイ/座席非一意は fail-closed。"),
        },
        "coverage": {
            "cases_with_deposit": with_deposit, "cases_with_client_contact": with_client,
            "persons_seated": seated, "persons_total": len(person_rows),
        },
        "top_involved_person": ({"name": top.name, "case_count": top.case_count,
                                 "ext": (top.seat or {}).get("ext")} if top else None),
        "max_deposit_case": _max_deposit_summary(case_rows),
        "out_path": str(out_path),
    }


def _max_deposit_summary(case_rows: Sequence[CaseRow]) -> dict[str, Any] | None:
    priced = [c for c in case_rows if c.deposit_incl_tax is not None]
    if not priced:
        return None
    top = max(priced, key=lambda c: c.deposit_incl_tax)  # type: ignore[arg-type]
    return {"case_id": top.case_id, "abbrev": top.abbrev,
            "deposit_incl_tax": top.deposit_incl_tax, "staff": dict(top.staff)}


# --------------------------------------------------------------------------- load / read (minimal API)
_LOAD_CACHE: dict[str, dict[str, list[dict[str, Any]]]] = {}


def reset_cache() -> None:
    _LOAD_CACHE.clear()


def load(path: Path | None = None) -> dict[str, list[dict[str, Any]]]:
    """Load ``{"cases": [...], "persons": [...]}`` (memoized). Empty tables on any failure (回帰ゼロ)."""
    out = path or default_out_path()
    key = str(out)
    if key in _LOAD_CACHE:
        return _LOAD_CACHE[key]
    tables: dict[str, list[dict[str, Any]]] = {"cases": [], "persons": []}
    try:
        with open(out, encoding="utf-8") as handle:
            header = handle.readline()
            meta = json.loads(header) if header.strip() else {}
            if isinstance(meta, dict) and meta.get("schema") == SCHEMA \
                    and meta.get("version") == SCHEMA_VERSION:
                for line in handle:
                    line = line.strip()
                    if not line:
                        continue
                    row = json.loads(line)
                    if row.get("kind") == "case":
                        tables["cases"].append(row)
                    elif row.get("kind") == "person":
                        tables["persons"].append(row)
    except Exception:
        tables = {"cases": [], "persons": []}
    _LOAD_CACHE[key] = tables
    return tables


if __name__ == "__main__":
    summary = build()
    print(json.dumps({"cases": summary["cases"], "persons": summary["persons"],
                      "certificate": summary["report"]["certificate"],
                      "top_involved_person": summary["report"]["top_involved_person"],
                      "max_deposit_case": summary["report"]["max_deposit_case"]},
                     ensure_ascii=False, indent=2))
