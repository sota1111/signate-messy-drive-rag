"""Parse 社内用語集.docx into lookup maps and provide query/term expansion.

The glossary is 9 docx tables of (正式名称, 社内用語, 補足). One table (案件名, 主略称, 別名候補, 補足)
holds the project case codes used both for query expansion and password derivation.
"""
from __future__ import annotations

import functools
import unicodedata

import docx

from src.rag.corpus import nfc
from config import settings


def _norm(s: str) -> str:
    return nfc(s).strip()


class Glossary:
    def __init__(self) -> None:
        # abbreviation (社内用語) -> canonical (正式名称)
        self.abbrev_to_formal: dict[str, str] = {}
        # canonical -> abbreviation
        self.formal_to_abbrev: dict[str, str] = {}
        # project case code / alias -> canonical company name
        self.case_to_company: dict[str, str] = {}
        # canonical company -> primary case code (主略称)
        self.company_to_code: dict[str, str] = {}
        # canonical company -> list of aliases (別名候補 + code + company)
        self.company_aliases: dict[str, list[str]] = {}

    # ---- expansion API ----
    def expand_terms(self, text: str) -> list[str]:
        """Return extra strings (formal names / companies) implied by abbreviations in text."""
        extra: list[str] = []
        t = nfc(text)
        for ab, formal in self.abbrev_to_formal.items():
            if ab and ab in t:
                extra.append(formal)
        for code, company in self.case_to_company.items():
            if code and code in t:
                extra.append(company)
        # dedupe, drop those already present
        seen, out = set(), []
        for e in extra:
            if e and e not in seen and e not in t:
                seen.add(e)
                out.append(e)
        return out

    def company_of(self, text: str) -> str | None:
        """Best-effort: which project a question refers to (by code/alias/name substring)."""
        t = nfc(text)
        # prefer longer alias matches first to avoid 青葉/青葉与信 collisions
        best = None
        best_len = 0
        for company, aliases in self.company_aliases.items():
            for a in aliases:
                if a and a in t and len(a) > best_len:
                    best, best_len = company, len(a)
        return best


def _looks_like_case_table(rows: list[list[str]]) -> bool:
    header = " ".join(rows[0]) if rows else ""
    return "主略称" in header or ("案件名" in header and "別名" in header)


@functools.lru_cache(maxsize=1)
def load() -> Glossary:
    g = Glossary()
    path = None
    internal = settings.CORPUS_DIR / [d.name for d in settings.CORPUS_DIR.iterdir()
                                      if nfc(d.name) == "社内管理"][0]
    for f in internal.iterdir():
        if "用語集" in nfc(f.name) and f.suffix == ".docx":
            path = f
            break
    if path is None:
        return g
    d = docx.Document(str(path))
    for t in d.tables:
        rows = [[_norm(c.text) for c in r.cells] for r in t.rows]
        if not rows:
            continue
        if _looks_like_case_table(rows):
            for r in rows[1:]:
                if len(r) < 2 or not r[0]:
                    continue
                company, code = r[0], r[1]
                aliases = []
                if len(r) >= 3 and r[2]:
                    aliases = [a.strip() for a in r[2].replace("、", ",").split(",") if a.strip()]
                g.company_to_code[company] = code
                allids = [code, company] + aliases
                g.company_aliases[company] = [a for a in allids if a]
                for a in allids:
                    if a:
                        g.case_to_company.setdefault(a, company)
        else:
            # generic 正式名称 | 社内用語 | 補足
            for r in rows[1:]:
                if len(r) < 2 or not r[0] or not r[1]:
                    continue
                formal, abbrev = r[0], r[1]
                g.abbrev_to_formal.setdefault(abbrev, formal)
                g.formal_to_abbrev.setdefault(formal, abbrev)
    return g
