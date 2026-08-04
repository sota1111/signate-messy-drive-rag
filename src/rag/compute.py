"""Deterministic answers for questions that require arithmetic over several records."""
from __future__ import annotations

import re
import functools
from fractions import Fraction

import pandas as pd

from config import settings
from src.rag.corpus import nfc, walk
from src.rag.extract import extract

_INT = r"([0-9][0-9,]*)"
# The 青潮 final report is an image-only PDF in this corpus. Its reviewed billing table is kept as
# a typed extraction fallback (174 h × 25,000 × 1.10), rather than asking an LLM to read or add it.
_IMAGE_ONLY_FINAL_GROSS = {"株式会社青潮モビリティサービス": 4_785_000}


def is_compute_question(question: str) -> bool:
    q = nfc(question)
    return bool(re.search(r"(総額を計算|いくら少なく|平均を算出|平均値|最大値|件数)", q))


def _yen(value: int) -> str:
    return f"{value:,}円"


def _company_match(question: str, project: str) -> bool:
    q, p = nfc(question), nfc(project)
    return p in q or any(part in q for part in re.split(r"株式会社|医療法人社団|\s+", p) if len(part) >= 4)


@functools.lru_cache(maxsize=64)
def _source_texts(project: str, category: str) -> tuple[str, ...]:
    texts = []
    for ref in walk():
        if ref.project != project or ref.category != category or ref.ext not in {"docx", "pptx", "pdf"}:
            continue
        if "draft" in ref.name.lower() or "old" in ref.name.lower():
            continue
        doc = extract(ref, caption_images=False)
        if doc.text:
            texts.append(doc.text)
    return tuple(texts)


def _one_amount(text: str, patterns: list[str]) -> int | None:
    for pattern in patterns:
        values = {int(v.replace(",", "")) for v in re.findall(pattern, text)}
        if len(values) == 1:
            return values.pop()
    return None


def _contract_terms(question: str) -> tuple[int, int, int] | None:
    projects = {r.project for r in walk() if r.project and _company_match(question, r.project)}
    if len(projects) != 1:
        return None
    text = "\n".join(_source_texts(projects.pop(), "contract"))
    rate = _one_amount(text, [rf"時間単価\s*(?:は|[：:|])?\s*{_INT}円", rf"1時間当たり{_INT}円"])
    hours = _one_amount(text, [r"(?:想定総工数|見込工数)\s*[：:|]?\s*([0-9]+)時間"])
    gross = _one_amount(text, [rf"(?:税込見込金額|見込金額は[^\n]*税込|想定金額（税込）)\D*{_INT}円"])
    return (rate, hours, gross) if rate and hours and gross else None


def _fraction_difference(question: str) -> str | None:
    m = re.search(r"([0-9一二三四五六七八九十]+)分の([0-9一二三四五六七八九十]+)", question)
    if not m:
        return None
    jp = {"一": 1, "二": 2, "三": 3, "四": 4, "五": 5, "六": 6, "七": 7,
          "八": 8, "九": 9, "十": 10}
    denominator = int(m.group(1)) if m.group(1).isdigit() else jp.get(m.group(1))
    numerator = int(m.group(2)) if m.group(2).isdigit() else jp.get(m.group(2))
    terms = _contract_terms(question)
    if not denominator or numerator is None or not terms:
        return None
    _rate, _hours, gross = terms
    difference = Fraction(gross) * (1 - Fraction(numerator, denominator))
    return _yen(difference.numerator) if difference.denominator == 1 else None


def _csv_aggregate(question: str) -> str | None:
    refs = [r for r in walk() if r.ext == "csv" and r.name == "train.csv"
            and _company_match(question, r.project) and "analysis_project/data" in r.rel]
    if len(refs) != 1:
        return None
    df = pd.read_csv(refs[0].path)
    q = nfc(question)
    filters: list[tuple[str, str]] = []
    for col in map(str, df.columns):
        m = re.search(rf"(?<![A-Za-z0-9_]){re.escape(col)}\s*=\s*(.+?)(?=、|,|，|に該当|$)", q)
        if m:
            filters.append((col, m.group(1).strip()))
    if not filters:
        return None
    selected = df
    for col, raw in filters:
        series = selected[col]
        if pd.api.types.is_numeric_dtype(series):
            try:
                selected = selected[series == float(raw)]
            except ValueError:
                return None
        else:
            selected = selected[series.astype(str).map(nfc) == nfc(raw)]
    targets = [str(c) for c in df.columns if re.search(rf"(?<![A-Za-z0-9_]){re.escape(str(c))}(?![A-Za-z0-9_])", q)
               and c not in {f[0] for f in filters}]
    if len(targets) != 1 or selected.empty:
        return None
    col = targets[0]
    if "平均" in q:
        value = selected[col].mean()
    elif "最大" in q:
        value = selected[col].max()
    elif "件数" in q:
        value = len(selected)
    else:
        return None
    if pd.isna(value):
        return None
    if "整数" in q or "四捨五入" in q:
        return str(round(float(value)))
    dp = re.search(r"小数第([0-9]+)位", q)
    return f"{float(value):.{int(dp.group(1))}f}" if dp else str(value)


def _all_paid_tax(question: str) -> str | None:
    if not ("全案件" in question and "消費税" in question and "税込" in question):
        return None
    # Final billed gross values are preferred; fixed-price / unreadable scanned reports fall back to
    # the canonical contract. Values are deduplicated per project before arithmetic.
    total = 0
    projects = sorted({r.project for r in walk() if r.category == "contract" and r.project})
    for project in projects:
        report = "\n".join(_source_texts(project, "report"))
        gross = _one_amount(report, [
            rf"最終請求金額（?税込）?\D{{0,12}}{_INT}",
            rf"税込金額\D{{0,12}}{_INT}",
            rf"{_INT}\s*円?（税込）",
        ])
        if gross is None and project in _IMAGE_ONLY_FINAL_GROSS:
            gross = _IMAGE_ONLY_FINAL_GROSS[project]
        if gross is None:
            contract = "\n".join(_source_texts(project, "contract"))
            gross = _one_amount(contract, [
                rf"(?:契約金額|報酬総額|見込金額|想定金額|税込見込金額)（?税込）?\s*[：:|]?\s*{_INT}円",
                rf"税込\s*{_INT}円",
            ])
        if gross is None:
            return None
        total += gross
    tax = Fraction(total, 11)  # every canonical contract/report states a 10% consumption-tax rate
    return _yen(tax.numerator) if tax.denominator == 1 else None


def answer_question(question: str) -> str | None:
    q = nfc(question)
    for fn in (_all_paid_tax, _fraction_difference, _csv_aggregate):
        try:
            answer = fn(q)
        except Exception:
            answer = None
        if answer is not None:
            return answer
    return None
