"""corpus_aggregate — 全プロジェクト横断の決定論集約ツール (SOT-2489).

現行 :mod:`~src.rag.tools.compute_sandbox` は **単一ファイル/単一案件** の pandas 式しか実行できない。
しかし gold-100 の律速(SOT-2486 診断 retrieval_miss)には「**全プロジェクトを横断**して集約し、最大/最頻の
案件・人を特定してからその属性を引く」多段質問が含まれる(``train.xlsx``/``契約書.docx`` は案件ごとに同名で
複数存在するため単一 compute では到達できず決定論的に棄権していた)。

このツールは全案件の契約書 (``01.契約``) と学習データ (``train.*``) を **列挙 → 案件ごとにフィールド抽出 →
横断で count / max / min / period-filter** する。数値・日付は必ず決定論経路(正規表現 + pandas 行数)で確定し
(暗算・創作の禁止)、会社名は :mod:`src.rag.extract.glossary` の **主略称** に正規化して返す。

対象実測ケース(Aug5 resolve trace で棄権していた gold-100):

* idx13「データアステル社の中でもっとも多くの案件にかかわっている人」→ 契約書の乙(受託側)担当者を
  横断カウントし最頻を特定(→ 内線は :mod:`~src.rag.tools.seating_chart` で解決)。
* idx26「2025-08-15〜09-07 に契約期間が重なり **40日超** の案件を主略称で全列挙」→ 全案件の契約期間を
  横断フィルタ。
* idx31「**固定金額契約の中で** 分析データ1行あたりの契約金額(税込)が最も高い案件」→ 契約金額(税込) ÷
  train 行数を横断比較。
* idx46「**着手金が最も高い案件**」→ 全案件契約書の着手金を横断比較(→ ES の内線は seating_chart)。

移植性(受け入れ条件⑤): コーパス固有の秘密値(金額・氏名・略称)は一切ソースに埋め込まない。すべて実行時に
契約書/データ/用語集から抽出する。抽出できないフィールドは ``None`` のまま扱い、決してでっち上げない
(precision-first — 該当なし/曖昧は棄権)。
"""
from __future__ import annotations

import math
import re
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path
from typing import Any, Sequence

import pandas as pd

from src.rag.corpus import FileRef, walk
from src.rag.extract import extract as _extract
from src.rag.extract.glossary import Glossary
from src.rag.extract.glossary import load as _load_glossary
from src.rag.tools import contract as _contract
from src.rag.tools.compute_sandbox import _read_frame

# --------------------------------------------------------------------------- extraction patterns
_INT = r"([0-9][0-9,]*)"
_DATE = r"(20\d{2}-\d{2}-\d{2})"

# 契約金額(税込) total — prioritized; ``_one_amount`` takes the first pattern that yields a single value.
_TOTAL_PATTERNS = [
    rf"契約金額（税込）\s*[：:|]?\s*{_INT}円",
    rf"想定金額（税込）\s*[：:|]?\s*{_INT}円",
    rf"税込見込金額\s*[：:|]?\s*{_INT}円",
    rf"報酬総額は[^\n]*?税込\s*{_INT}\s*円",
    rf"見込金額は[^\n]*?税込\s*{_INT}\s*円",
]
# 契約期間 / 有効期間 の「YYYY-MM-DD から YYYY-MM-DD」
_PERIOD_RE = re.compile(rf"(?:契約期間|契約有効期間|有効期間)は[、,]?\s*{_DATE}\s*から\s*{_DATE}")
# 固定金額契約の判定(着手金・固定価格・固定金額の記載)
_FIXED_RE = re.compile(r"固定価格契約|着手金|契約時に金額を固定|固定金額")
# 乙(受託側)担当者の役割ラベル。名前は行内で「 - 」区切りの複数役割にも対応するため区切り手前まで採る。
_ROLES: list[tuple[str, str]] = [
    ("ES", "エグゼクティブスポンサー"),
    ("PM", "プロジェクトマネージャー"),
    ("LeadDS", "リードデータサイエンティスト"),
    ("DE", "データエンジニア"),
    ("BA", "ビジネスアナリスト"),
    ("QA", "QAレビューアー?"),
]
_NAME = r"([^\n（(\-‐–/／、,|]+)"

METRICS = {"contract_amount", "deposit", "train_rows", "amount_per_row", "period_days", "staff"}
OPS = {"max", "min", "count", "filter", "list"}

_CONTRACT_EXT = {"docx", "pdf", "pptx"}
_TRAIN_NAMES = ("train.csv", "train.xlsx", "train.tsv")


# --------------------------------------------------------------------------- field extraction
def _one_amount(text: str, patterns: Sequence[str]) -> int | None:
    """First pattern whose matches collapse to a single distinct yen value (else None)."""
    for pattern in patterns:
        values = {int(v.replace(",", "")) for v in re.findall(pattern, text)}
        if len(values) == 1:
            return values.pop()
    return None


def _extract_total(text: str) -> int | None:
    """契約金額(税込) — the primary tax-inclusive contract/estimated total."""
    return _one_amount(text, _TOTAL_PATTERNS)


def _extract_deposit(text: str) -> int | None:
    """着手金(税込) — the largest yen figure on the 着手金 row (税込 > 税額 > 税抜).

    Only fixed-price contracts carry a 着手金; T&M contracts have none → ``None`` (never guessed).
    """
    best: int | None = None
    for line in text.split("\n"):
        if "着手金" not in line:
            continue
        amounts = [int(v.replace(",", "")) for v in re.findall(r"[0-9][0-9,]{3,}", line)]
        amounts = [a for a in amounts if a >= 10_000]  # ignore stray small integers (割合% など)
        if amounts:
            best = max(amounts) if best is None else max(best, max(amounts))
    return best


def _extract_period(text: str) -> tuple[str, str] | None:
    """契約期間/有効期間 の (開始, 終了) ISO 日付。記載が無ければ None(棄権)。"""
    m = _PERIOD_RE.search(text)
    return (m.group(1), m.group(2)) if m else None


def _extract_staff(text: str) -> dict[str, str]:
    """契約書に列挙された乙(受託側=データアステル)担当者を {役割コード: 氏名} で返す。"""
    out: dict[str, str] = {}
    for code, label in _ROLES:
        m = re.search(label + r"\s*[：:]\s*" + _NAME, text)
        if m:
            name = m.group(1).strip()
            if name:
                out[code] = name
    return out


def _is_fixed(text: str) -> bool:
    return bool(_FIXED_RE.search(text))


def _to_date(s: str) -> date:
    y, m, d = (int(p) for p in s.split("-"))
    return date(y, m, d)


# --------------------------------------------------------------------------- file selection / rows
def _contract_refs(project: str, refs: Sequence[FileRef]) -> list[FileRef]:
    return [r for r in refs
            if r.project == project and r.category == "contract" and r.ext in _CONTRACT_EXT
            and "draft" not in r.name.lower() and "old" not in r.name.lower()]


def _contract_text(project: str, refs: Sequence[FileRef]) -> str:
    parts: list[str] = []
    for r in _contract_refs(project, refs):
        try:
            doc = _extract(r, caption_images=False)  # no vision/network for a text field scan
        except Exception:  # noqa: BLE001 — one unreadable contract must not sink the aggregate
            continue
        if getattr(doc, "text", ""):
            parts.append(doc.text)
    return "\n".join(parts)


def _train_ref(project: str, refs: Sequence[FileRef]) -> FileRef | None:
    """The project's canonical learning table (03.データ/train.*), preferring csv > xlsx > tsv."""
    cand = [r for r in refs
            if r.project == project and r.name in _TRAIN_NAMES and "analysis_project" not in r.rel]
    for ext in ("csv", "xlsx", "tsv"):
        for r in cand:
            if r.ext == ext:
                return r
    return None


def _count_rows(ref: FileRef | None) -> int | None:
    if ref is None:
        return None
    try:
        if ref.ext == "csv":
            return int(len(pd.read_csv(ref.path)))
        if ref.ext == "tsv":
            return int(len(pd.read_csv(ref.path, sep="\t")))
        df, _ = _read_frame(ref, None)
        return int(len(df))
    except Exception:  # noqa: BLE001 — an unreadable table → unknown row count, not a crash
        return None


# --------------------------------------------------------------------------- project record
@dataclass
class ProjectContract:
    """One project's cross-aggregatable contract facts (all fields best-effort; ``None`` = unknown)."""

    project: str
    abbrev: str | None                      # 主略称 (glossary company_to_code)
    contract_amount: int | None             # 契約金額(税込)
    deposit: int | None                     # 着手金(税込)
    train_rows: int | None                  # 学習データ行数
    period: tuple[str, str] | None          # (契約期間 開始, 終了)
    fixed: bool                             # 固定金額契約か
    staff: dict[str, str] = field(default_factory=dict)

    @property
    def name(self) -> str:
        return self.abbrev or self.project

    def amount_per_row(self) -> float | None:
        if self.contract_amount and self.train_rows:
            return self.contract_amount / self.train_rows
        return None

    def period_days(self) -> int | None:
        if self.period:
            return (_to_date(self.period[1]) - _to_date(self.period[0])).days
        return None

    def to_dict(self) -> dict[str, Any]:
        return {
            "project": self.project,
            "abbrev": self.abbrev,
            "contract_amount": self.contract_amount,
            "deposit": self.deposit,
            "train_rows": self.train_rows,
            "period": list(self.period) if self.period else None,
            "period_days": self.period_days(),
            "amount_per_row": self.amount_per_row(),
            "fixed": self.fixed,
            "staff": dict(self.staff),
        }


def _abbrev_for(project: str, glossary: Glossary) -> str | None:
    """主略称 for a project folder name, via glossary company_to_code (exact then substring)."""
    code = glossary.company_to_code.get(project)
    if code:
        return code
    for company, c in glossary.company_to_code.items():
        if company and (company in project or project in company):
            return c
    return None


def collect_contracts(*, corpus_dir: str | Path | None = None,
                      refs: Sequence[FileRef] | None = None,
                      glossary: Glossary | None = None) -> list[ProjectContract]:
    """Enumerate every project's contract facts across the whole corpus (deterministic, no LLM).

    ``corpus_dir`` overrides the corpus root (tests); ``glossary`` overrides the 主略称 source (tests).
    Returns one :class:`ProjectContract` per project that owns a ``01.契約`` folder, in corpus order.
    """
    if refs is None:
        refs = walk(Path(corpus_dir)) if corpus_dir is not None else walk()
    g = glossary if glossary is not None else _load_glossary()
    projects = [p for p in dict.fromkeys(r.project for r in refs
                                         if r.category == "contract" and r.project)]
    out: list[ProjectContract] = []
    for pj in projects:
        text = _contract_text(pj, refs)
        out.append(ProjectContract(
            project=pj,
            abbrev=_abbrev_for(pj, g),
            contract_amount=_extract_total(text),
            deposit=_extract_deposit(text),
            train_rows=_count_rows(_train_ref(pj, refs)),
            period=_extract_period(text),
            fixed=_is_fixed(text),
            staff=_extract_staff(text),
        ))
    return out


# --------------------------------------------------------------------------- aggregation
def _metric_value(rec: ProjectContract, metric: str) -> Any:
    if metric == "contract_amount":
        return rec.contract_amount
    if metric == "deposit":
        return rec.deposit
    if metric == "train_rows":
        return rec.train_rows
    if metric == "amount_per_row":
        return rec.amount_per_row()
    if metric == "period_days":
        return rec.period_days()
    return None


def _err(message: str, **extra: Any) -> dict[str, Any]:
    return _contract.make(None, engine="corpus_aggregate", evidence={"error": message, **extra})


def _staff_counts(pool: Sequence[ProjectContract]) -> list[tuple[str, int]]:
    """Cross-project count of each 乙 person (one vote per project), by count desc then first-seen."""
    counts: dict[str, int] = {}
    order: dict[str, int] = {}
    for i, r in enumerate(pool):
        for name in sorted(set(r.staff.values())):
            counts[name] = counts.get(name, 0) + 1
            order.setdefault(name, i)
    return sorted(counts.items(), key=lambda kv: (-kv[1], order[kv[0]]))


def corpus_aggregate(metric: str, op: str = "max", *,
                     fixed_only: bool = False, round_up: bool = False,
                     overlap_start: str | None = None, overlap_end: str | None = None,
                     min_days: int | None = None,
                     corpus_dir: str | Path | None = None,
                     glossary: Glossary | None = None) -> dict[str, Any]:
    """Cross-project aggregation → the unified ``{value, evidence, method}`` contract.

    Parameters
    ----------
    metric : one of ``contract_amount`` / ``deposit`` / ``train_rows`` / ``amount_per_row`` /
        ``period_days`` / ``staff``.
    op : ``max`` / ``min`` (extreme project for a numeric metric), ``count`` (metric=staff → most
        frequent 乙 person; other metrics → number of projects with a known value), ``filter``
        (period overlap window + ``min_days``), ``list`` (all project records).
    fixed_only : restrict to 固定金額契約 (idx31).
    round_up : ceil the reported max/min value (円単位で切り上げ, idx31).
    overlap_start, overlap_end, min_days : period ``filter`` window (ISO dates) and the strict
        ``> min_days`` length threshold (idx26 「40日を超えている」→ ``min_days=40``).

    ``value`` is ``None`` whenever nothing resolves (precision-first abstain); numbers/dates come only
    from the deterministic extraction, never from a model.
    """
    metric, op = str(metric), str(op)
    if metric not in METRICS:
        return _err(f"unknown metric {metric!r}", metrics=sorted(METRICS))
    if op not in OPS:
        return _err(f"unknown op {op!r}", ops=sorted(OPS))

    recs = collect_contracts(corpus_dir=corpus_dir, glossary=glossary)
    pool = [r for r in recs if r.fixed or not fixed_only]
    base_evidence = {
        "projects": len(recs),
        "considered": len(pool),
        "fixed_only": fixed_only,
    }

    # ---- staff cross-count (idx13) --------------------------------------------------------------
    if metric == "staff":
        if op not in ("count", "list"):
            return _err(f"metric=staff supports op count/list, not {op!r}")
        ranked = _staff_counts(pool)
        if op == "list":
            return _contract.make(
                [{"name": n, "count": c} for n, c in ranked],
                engine="corpus_aggregate", evidence=base_evidence, metric=metric, op=op)
        if not ranked:
            return _contract.make(None, engine="corpus_aggregate",
                                  evidence={**base_evidence, "reason": "担当者を抽出できなかった"},
                                  metric=metric, op=op)
        top, top_count = ranked[0]
        tie = [n for n, c in ranked if c == top_count]
        return _contract.make(
            {"top": top, "count": top_count, "counts": {n: c for n, c in ranked}},
            engine="corpus_aggregate",
            evidence={**base_evidence, "tie": tie if len(tie) > 1 else None,
                      "per_project": {r.name: r.staff for r in pool if r.staff}},
            metric=metric, op=op, note="乙(受託側)担当者の案件横断出現回数。最頻=top。")

    # ---- period overlap filter (idx26) ----------------------------------------------------------
    if op == "filter":
        ws = _to_date(overlap_start) if overlap_start else None
        we = _to_date(overlap_end) if overlap_end else None
        hits: list[dict[str, Any]] = []
        for r in pool:
            if not r.period:
                continue
            s, e = _to_date(r.period[0]), _to_date(r.period[1])
            days = (e - s).days
            if ws and we and not (s <= we and e >= ws):
                continue
            if min_days is not None and not (days > min_days):
                continue
            hits.append({"abbrev": r.name, "project": r.project,
                         "start": r.period[0], "end": r.period[1], "days": days})
        return _contract.make(
            [h["abbrev"] for h in hits],
            engine="corpus_aggregate",
            evidence={**base_evidence, "window": [overlap_start, overlap_end],
                      "min_days": min_days, "matches": hits},
            metric=metric, op=op,
            note="契約期間がwindowと重なりmin_days超の案件を主略称で(コーパス順)。")

    if op not in ("max", "min"):
        return _err(f"metric={metric!r} supports op max/min/filter/list, not {op!r}")

    # ---- max / min over a numeric metric (idx31 amount_per_row, idx46 deposit) -------------------
    scored = [(r, _metric_value(r, metric)) for r in pool]
    scored = [(r, v) for r, v in scored if v is not None]
    if op == "list":  # (kept for symmetry though handled above for staff)
        return _contract.make([r.to_dict() for r, _ in scored], engine="corpus_aggregate",
                              evidence=base_evidence, metric=metric, op=op)
    if not scored:
        return _contract.make(None, engine="corpus_aggregate",
                              evidence={**base_evidence, "reason": f"{metric} を持つ案件が無い"},
                              metric=metric, op=op)
    chooser = max if op == "max" else min
    rec, raw = chooser(scored, key=lambda rv: rv[1])
    value_num: Any = math.ceil(raw) if round_up else (round(raw, 2) if isinstance(raw, float) else raw)
    value: dict[str, Any] = {"abbrev": rec.name, "project": rec.project,
                             "metric": metric, "value": value_num, "staff": dict(rec.staff)}
    if metric == "amount_per_row":
        value["amount"] = rec.contract_amount
        value["rows"] = rec.train_rows
    return _contract.make(
        value, engine="corpus_aggregate",
        evidence={**base_evidence, "ranking": [
            {"abbrev": r.name, "value": (round(v, 2) if isinstance(v, float) else v)}
            for r, v in sorted(scored, key=lambda rv: rv[1], reverse=(op == "max"))[:8]]},
        metric=metric, op=op, round_up=round_up,
        note="横断集約の極値案件。主略称=abbrev、担当者=staff(ES等の後続内線引きに使用)。")
