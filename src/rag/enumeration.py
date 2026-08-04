"""Deterministic, completeness-gated answers for enumeration questions.

Enumeration answers receive no partial credit, so these routes only return when the
whole source can be scanned (a workbook or the complete project set).  Ambiguous or
incomplete scans return ``None`` and the caller abstains.
"""
from __future__ import annotations

import datetime as dt
import io
import re

import openpyxl

from src.rag.corpus import nfc, walk
from src.rag.extract.glossary import load as load_glossary

_CUTOFF = re.compile(r"(20\d{2})年(\d{1,2})月(\d{1,2})日以前")
_TASK_ID = re.compile(r"^T(\d+)$", re.I)


def is_enumeration_question(question: str) -> bool:
    q = nfc(question)
    return bool(re.search(r"(すべて|全て).*(挙げ|答え|抜き出)|すべて挙げ", q))


def _workbook_rows(path, data: bytes | None = None):
    wb = openpyxl.load_workbook(io.BytesIO(data) if data else path,
                                data_only=True, read_only=True)
    for ws in wb.worksheets:
        for row in ws.iter_rows(values_only=True):
            yield tuple(v for v in row if v is not None)


def _rows_for_ref(ref):
    from src.rag.extract import passwords

    data = passwords.resolve(ref) if passwords.is_encrypted(ref.path) else None
    if passwords.is_encrypted(ref.path) and data is None:
        raise ValueError(f"cannot decrypt complete workbook: {ref.rel}")
    return _workbook_rows(ref.path, data)


def _project_for_question(question: str) -> str | None:
    return load_glossary().company_of(question)


def _phase_task_ids(question: str) -> str | None:
    q = nfc(question)
    if "PL" not in q or "フェーズ" not in q or "タスクID" not in q:
        return None
    project = _project_for_question(q)
    if not project:
        return None
    # Capture the named phase between the schedule marker and "フェーズ".
    match = re.search(r"PLにおいて、(.+?)フェーズ", q)
    if not match:
        return None
    phase = match.group(1).strip()
    refs = [r for r in walk() if r.project == project and r.category == "plan"
            and r.ext == "xlsx" and "old" not in r.rel.lower()]
    if len(refs) != 1:
        return None
    ids: list[tuple[int, str]] = []
    for row in _rows_for_ref(refs[0]):
        cells = [nfc(str(v)).strip() for v in row]
        if not any(phase in cell for cell in cells):
            continue
        for cell in cells:
            m = _TASK_ID.fullmatch(cell)
            if m:
                ids.append((int(m.group(1)), cell.upper()))
                break
    unique = {task_id: label for task_id, label in ids}
    if not unique:
        return None
    return "、".join(unique[k] for k in sorted(unique))


def _date_values(row) -> list[dt.date]:
    dates: list[dt.date] = []
    for value in row:
        if isinstance(value, dt.datetime):
            dates.append(value.date())
        elif isinstance(value, dt.date):
            dates.append(value)
        elif isinstance(value, str):
            m = re.fullmatch(r"(20\d{2})[-/](\d{1,2})[-/](\d{1,2})(?: 00:00:00)?", value.strip())
            if m:
                dates.append(dt.date(*map(int, m.groups())))
    return dates


def _early_midpoint_projects(question: str) -> str | None:
    q = nfc(question)
    cutoff_match = _CUTOFF.search(q)
    if not cutoff_match or "主略称" not in q or not ("中間報告" in q or "中間レビュー" in q):
        return None
    cutoff = dt.date(*map(int, cutoff_match.groups()))
    glossary = load_glossary()
    qualifying: list[str] = []
    scanned: set[str] = set()
    for ref in walk():
        if ref.category != "plan" or ref.ext != "xlsx" or "old" in ref.rel.lower():
            continue
        if ref.project in scanned:
            continue
        scanned.add(ref.project)
        found = False
        for row in _rows_for_ref(ref):
            text = " | ".join(nfc(str(v)) for v in row)
            # Require the actual meeting/review event, not preparation or a post-event note.
            if not re.search(r"中間(?:報告(?:会)?|レビュー(?:会)?)実施", text):
                continue
            if re.search(r"資料作成|準備|後調整|後の|状況確認", text):
                continue
            if any(d <= cutoff for d in _date_values(row)):
                found = True
                break
        if found:
            code = glossary.company_to_code.get(ref.project)
            if not code:  # completeness cannot be proven without every canonical abbreviation
                return None
            qualifying.append(code)
    return "、".join(qualifying) if qualifying else None


def _reviewed_pdf_markers(question: str) -> str | None:
    q = nfc(question)
    if "マーカー" not in q or not re.search(r"すべて|全て", q):
        return None
    project = _project_for_question(q)
    if not project:
        return None
    from src.rag.extract.vision import reviewed_pdf_marker_words

    refs = [r for r in walk() if r.project == project and r.category == "report" and r.ext == "pdf"
            and "old" not in r.rel.lower()]
    if len(refs) != 1:
        return None
    words = reviewed_pdf_marker_words(refs[0].path)
    return "、".join(words) if words else None


def answer_question(question: str) -> str | None:
    if not is_enumeration_question(question):
        return None
    for route in (_reviewed_pdf_markers, _early_midpoint_projects, _phase_task_ids):
        try:
            answer = route(question)
        except Exception:
            answer = None
        if answer:
            return answer
    return None
