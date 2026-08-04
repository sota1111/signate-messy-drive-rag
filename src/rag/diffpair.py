"""Version-pair structural diff — answer "old版と最新版の変更点" questions deterministically.

A class of questions asks to compare an *older* and the *latest* version of the same logical
document and report what changed (valid idx9: 青嶺の提案書 old→最新 で QAレビューア 池田 直哉 →
小林 直樹). Plain retrieval fails these because the two versions are near-identical and the single
changed value is buried; the answer must come from an actual structural diff of the two files.

This module:
  * finds version pairs by filename/folder rules — an ``old/`` subfolder copy vs. the sibling in the
    parent folder, and revision-suffixed siblings (``_r1``/``_r2``, ``_v1``/``_final`` …);
  * diffs docx / pptx / xlsx at a *structural* granularity — table cells keyed by their row label,
    and flowing paragraphs aligned by sequence — so a changed value is reported as "変更前 → 変更後";
  * excludes purely cosmetic differences (whitespace / full-width vs half-width) so a reformat is not
    mistaken for a substantive change;
  * abstains (returns ``None``) when a diff question cannot be resolved to a single unambiguous pair,
    because Missing (0) beats Incorrect (−1) under the official rubric.

All heavy Office deps and corpus access are imported/read lazily inside functions and guarded, so
importing this module at serve time (lean container, no corpus) is free and never raises.
"""
from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass, field

from src.rag.corpus import FileRef, nfc

# Folder names that mark an archived / previous copy (lower-cased, NFC).
_OLD_DIR_NAMES = {"old", "旧", "旧版", "旧版本", "archive", "archives", "過去", "前回", "backup", "bak"}

# Latest-version filename hints (a bare copy directly under the doc folder is treated as latest).
_LATEST_TOKENS = {"final", "new", "最新", "新", "確定", "fix", "fixed"}
_OLDEST_TOKENS = {"old", "旧", "初版", "draft", "下書き"}

# Revision-suffix on a filename stem: must be separated from the base by _/-/space so ordinary
# words that merely contain "ver"/"v" (e.g. "server") are never mis-parsed as versioned files.
_VER_RE = re.compile(
    r"^(?P<base>.+?)[ _\-]+(?P<tok>rev|ver|version|final|old|new|旧版|旧|新|確定|r|v)(?P<num>\d*)$",
    re.IGNORECASE,
)

# A resolvable version-diff answer is a *small* set of edits. A large change count means the two
# versions realigned (rows inserted/removed) and the structural alignment is unreliable — abstain
# rather than emit dozens of spurious "変更前→変更後" rows.
_MAX_CHANGES = 6

# Question asks for a version diff — routed here ONLY when it explicitly references an OLDER version
# (旧版 / old版 / a revision-suffixed pair like _r1…_r2) AND a comparison/change verb. This keeps
# ordinary questions that merely contain "差分"/"変更"/"比較" (e.g. "モデル比較の設定差分", "RATEが
# 変更された") out of the differ, where a spurious diff would be an Incorrect (−1).
_OLD_MARKER_RE = re.compile(
    r"旧版|旧バージョン|旧ファイル|旧稿|旧版本|old版|oldバージョン|oldフォルダ|old\s*版|"
    r"以前の版|前の版|前バージョン|前回版|古い版")
# two revision-suffixed filenames named in the question, e.g. "…_r1.xlsx と …_r2.xlsx"
_REV_PAIR_RE = re.compile(r"[_\-]?(?:r|v|rev|ver|version)\s*(\d+)\b.{0,40}?[_\-]?(?:r|v|rev|ver|version)\s*(\d+)\b",
                          re.IGNORECASE)
_COMPARE_RE = re.compile(r"(比較|変更|差分|相違|変わった|変更前|前と後|前後|どう違)")

_MOJIBAKE_RE = re.compile(r"[�]")  # replacement char → treat as unreadable field

# Office filenames named explicitly in a question (e.g. "提案書_v1.pptx と 提案書_v3.pptx").
_FILE_TOKEN_RE = re.compile(r"[^\s、。「」『』（）()]+?\.(?:pptx|docx|xlsx)", re.IGNORECASE)


def _norm(s: str) -> str:
    """Fold to a comparison key: NFKC (full/half-width, spaces) + drop all whitespace + lower.

    Cosmetic-only differences (reflowed spaces, width) collapse to an equal key and are dropped;
    a real value change (池田直哉 vs 小林直樹) stays distinct."""
    s = unicodedata.normalize("NFKC", nfc(s))
    return re.sub(r"\s+", "", s).lower()


def is_diff_question(question: str) -> bool:
    q = nfc(question)
    if not _COMPARE_RE.search(q):
        return False
    return bool(_OLD_MARKER_RE.search(q) or _REV_PAIR_RE.search(q))


# --------------------------------------------------------------------------------------------
@dataclass
class VersionPair:
    old: FileRef
    new: FileRef
    base: str            # doc label without version token (e.g. "提案書")
    basis: str           # "old-folder" | "rev-suffix"


@dataclass
class Change:
    label: str           # row/field label, or "" for flowing text
    before: str
    after: str
    kind: str = "modify"  # modify | add | remove

    def render(self) -> str:
        if self.kind == "add":
            return f"{self.label + '：' if self.label else '追加: '}{self.after}".strip()
        if self.kind == "remove":
            return f"{self.label + '：' if self.label else '削除: '}{self.before}".strip()
        head = f"{self.label}：" if self.label else ""
        return f"{head}{self.before} → {self.after}"


def _rank(tok: str, num: str) -> tuple[int, int]:
    """Order key for a version token; larger = newer."""
    t = tok.lower()
    if t in _LATEST_TOKENS:
        return (3, 0)
    if t in _OLDEST_TOKENS:
        return (0, 0)
    return (1, int(num) if num.isdigit() else 0)


def _parse_version(stem: str) -> tuple[str, tuple[int, int]] | None:
    m = _VER_RE.match(nfc(stem))
    if not m:
        return None
    base = m.group("base").strip(" _-")
    if not base:
        return None
    return base, _rank(m.group("tok"), m.group("num"))


def _walk(company: str | None):
    from src.rag import corpus  # lazy: corpus may be absent at serve time

    refs = corpus.walk()
    if company:
        c = nfc(company)
        refs = [r for r in refs if nfc(r.project) and (nfc(r.project) in c or c in nfc(r.project))]
    return refs


def find_pairs(company: str | None = None) -> list[VersionPair]:
    """All version pairs (optionally within one company), each as old→new."""
    try:
        refs = _walk(company)
    except Exception:
        return []
    if not refs:
        return []

    by_dir: dict[str, list[FileRef]] = {}
    for r in refs:
        d = "/".join(r.rel.split("/")[:-1])
        by_dir.setdefault(d, []).append(r)

    pairs: list[VersionPair] = []
    seen: set[tuple[str, str]] = set()

    # (1) old-folder rule: a file inside an old/旧/archive dir vs the same-named sibling one level up.
    for d, files in by_dir.items():
        leaf = d.split("/")[-1].lower() if d else ""
        if nfc(leaf) not in _OLD_DIR_NAMES:
            continue
        parent = "/".join(d.split("/")[:-1])
        for old in files:
            for new in by_dir.get(parent, []):
                if new.name == old.name and new.ext == old.ext:
                    key = (old.rel, new.rel)
                    if key not in seen:
                        seen.add(key)
                        pairs.append(VersionPair(old, new, old.stem, "old-folder"))

    # (2) revision-suffix rule: within a dir, files sharing a base but differing version rank.
    for d, files in by_dir.items():
        groups: dict[tuple[str, str], list[tuple[tuple[int, int], FileRef]]] = {}
        for r in files:
            pv = _parse_version(r.stem)
            if pv is None:
                continue
            base, rank = pv
            groups.setdefault((base, r.ext), []).append((rank, r))
        for (base, _ext), lst in groups.items():
            if len(lst) < 2:
                continue
            lst.sort(key=lambda t: t[0])
            (_ro, old), (_rn, new) = lst[0], lst[-1]
            if old.rel == new.rel or old.rel.endswith(f"/old/{old.name}"):
                continue
            key = (old.rel, new.rel)
            if key not in seen:
                seen.add(key)
                pairs.append(VersionPair(old, new, base, "rev-suffix"))
    return pairs


# --------------------------------------------------------------------------------------------
@dataclass
class _Struct:
    cells: dict[str, tuple[str, str]] = field(default_factory=dict)  # field-key -> (row_label, raw)
    flow: list[str] = field(default_factory=list)                    # ordered paragraph texts


def _add_cell(st: _Struct, key: str, label: str, value: str) -> None:
    v = value.strip()
    if v and not _MOJIBAKE_RE.search(v):
        # keep the first occurrence of a duplicate key stable across versions
        st.cells.setdefault(key, (label.strip(), v))


def _pptx_struct(path) -> _Struct | None:
    try:
        from pptx import Presentation
    except Exception:
        return None
    try:
        prs = Presentation(str(path))
    except Exception:
        return None
    st = _Struct()
    for si, slide in enumerate(prs.slides, 1):
        ti = 0
        for shape in slide.shapes:
            if getattr(shape, "has_table", False) and shape.has_table:
                ti += 1
                for ri, row in enumerate(shape.table.rows):
                    cells = [c.text.strip() for c in row.cells]
                    label = cells[0] if cells and cells[0] else f"行{ri + 1}"
                    for ci, val in enumerate(cells):
                        if ci == 0:
                            continue
                        _add_cell(st, f"s{si}:t{ti}:{_norm(label)}:{ci}", label, val)
            elif getattr(shape, "has_text_frame", False) and shape.has_text_frame:
                for para in shape.text_frame.paragraphs:
                    line = ("".join(r.text for r in para.runs) or para.text).strip()
                    if line:
                        st.flow.append(line)
    return st


def _docx_struct(path) -> _Struct | None:
    try:
        import docx
    except Exception:
        return None
    try:
        d = docx.Document(str(path))
    except Exception:
        return None
    st = _Struct()
    for p in d.paragraphs:
        if p.text.strip():
            st.flow.append(p.text.strip())
    for ti, t in enumerate(d.tables, 1):
        for ri, row in enumerate(t.rows):
            cells = [c.text.strip() for c in row.cells]
            label = cells[0] if cells and cells[0] else f"行{ri + 1}"
            for ci, val in enumerate(cells):
                if ci == 0:
                    continue
                _add_cell(st, f"t{ti}:{_norm(label)}:{ci}", label, val)
    return st


def _xlsx_struct(path) -> _Struct | None:
    try:
        import openpyxl
    except Exception:
        return None
    try:
        wb = openpyxl.load_workbook(str(path), data_only=True, read_only=True)
    except Exception:
        return None
    st = _Struct()
    for ws in wb.worksheets:
        for row in ws.iter_rows(max_row=min(ws.max_row or 0, 300)):
            for c in row:
                if c.value is None:
                    continue
                coord = f"{ws.title}!{c.coordinate}"
                _add_cell(st, coord, coord, str(c.value))
    try:
        wb.close()
    except Exception:
        pass
    return st


def _struct(ref: FileRef) -> _Struct | None:
    if ref.ext == "pptx":
        return _pptx_struct(ref.path)
    if ref.ext == "docx":
        return _docx_struct(ref.path)
    if ref.ext == "xlsx":
        return _xlsx_struct(ref.path)
    return None


def _diff_flow(old: list[str], new: list[str]) -> list[Change]:
    import difflib

    changes: list[Change] = []
    on = [_norm(x) for x in old]
    nn = [_norm(x) for x in new]
    sm = difflib.SequenceMatcher(a=on, b=nn, autojunk=False)
    for op, i1, i2, j1, j2 in sm.get_opcodes():
        if op == "equal":
            continue
        if op == "replace" and (i2 - i1) == (j2 - j1):
            for k in range(i2 - i1):
                b, a = old[i1 + k], new[j1 + k]
                if _norm(b) != _norm(a):
                    changes.append(Change("", b, a, "modify"))
        # deletions / insertions in flowing prose are usually reformat noise, not the asked change;
        # omit them to avoid false positives (a diff question wants the changed *value*).
    return changes


def structural_diff(pair: VersionPair) -> list[Change] | None:
    """Substantive changes old→new, or None if either side can't be read structurally."""
    so, sn = _struct(pair.old), _struct(pair.new)
    if so is None or sn is None:
        return None
    changes: list[Change] = []
    # keyed cells: match by field key; report only value changes (cosmetic folded away by _norm)
    for key, (label, ov) in so.cells.items():
        if key in sn.cells:
            nlabel, nv = sn.cells[key]
            if _norm(ov) != _norm(nv):
                changes.append(Change(nlabel or label, ov, nv, "modify"))
    # flowing paragraphs: aligned sequence replacements
    changes.extend(_diff_flow(so.flow, sn.flow))
    # de-dup identical rendered changes, keep order
    seen, out = set(), []
    for c in changes:
        r = c.render()
        if r not in seen:
            seen.add(r)
            out.append(c)
    return out


# --------------------------------------------------------------------------------------------
def _doc_matches_question(pair: VersionPair, q: str) -> bool:
    """Does the question name this pair's document (by its base filename token)?"""
    base = _norm(pair.base)
    if base and base in _norm(q):
        return True
    # also accept a shared long token (≥3 chars) between the base and the question
    for tok in re.findall(r"[一-龥ぁ-んァ-ヶーA-Za-z0-9]{3,}", nfc(pair.base)):
        if _norm(tok) and _norm(tok) in _norm(q):
            return True
    return False


def _explicit_pair(question: str, company: str | None) -> VersionPair | None:
    """Pair built from two version files named verbatim in the question (old→new by rank).

    Honours questions that pin exact endpoints (e.g. "提案書_v1.pptx から 提案書_v2.pptx") so the diff
    uses the *named* versions rather than the corpus-wide oldest→newest pair."""
    if not _FILE_TOKEN_RE.search(nfc(question)):
        return None
    try:
        refs = _walk(company)
    except Exception:
        return None
    q = nfc(question)
    # Scan for known corpus filenames appearing verbatim in the question (robust to adjacent
    # particles like 「から」 that a token-splitter would swallow).
    mentioned: list[FileRef] = []
    for r in refs:
        if r.ext in ("pptx", "docx", "xlsx") and nfc(r.name) in q:
            if r.rel not in {x.rel for x in mentioned}:
                mentioned.append(r)
    versioned = [r for r in mentioned if _parse_version(r.stem)]
    cand = versioned if len(versioned) >= 2 else mentioned
    if len(cand) != 2:
        return None

    def _rk(r: FileRef) -> tuple[int, int]:
        pv = _parse_version(r.stem)
        return pv[1] if pv else (2, 0)

    old, new = sorted(cand, key=_rk)
    if _rk(old) == _rk(new):
        return None
    return VersionPair(old, new, old.stem, "explicit")


def resolve_pair(question: str) -> VersionPair | None:
    """The single version pair a diff question refers to, or None if 0 / ambiguous (>1)."""
    from src.rag.extract import glossary  # lazy

    try:
        company = glossary.load().company_of(question)
    except Exception:
        company = None
    explicit = _explicit_pair(question, company)
    if explicit is not None:
        return explicit
    pairs = find_pairs(company)
    if not pairs:
        return None
    q = nfc(question)
    matched = [p for p in pairs if _doc_matches_question(p, q)]
    if len(matched) == 1:
        return matched[0]
    # If the doc hint didn't disambiguate but there is exactly one pair for the company, use it.
    if not matched and len(pairs) == 1:
        return pairs[0]
    return None


def answer_question(question: str) -> str | None:
    """Deterministic answer for a version-diff question, or None to let the caller abstain.

    Returns the rendered "変更前 → 変更後" summary when a single pair resolves to a non-empty set of
    substantive changes; otherwise None (no pair / ambiguous / unreadable / no real change)."""
    if not is_diff_question(question):
        return None
    pair = resolve_pair(question)
    if pair is None:
        return None
    changes = structural_diff(pair)
    if not changes:
        return None
    modifies = [c for c in changes if c.kind == "modify"] or changes
    # Too many changes ⇒ the versions realigned and the diff is not uniquely resolvable → abstain.
    if len(modifies) > _MAX_CHANGES:
        return None
    return "、".join(c.render() for c in modifies)
