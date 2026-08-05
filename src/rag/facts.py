"""Fact-level indexing (SOT-2449 / R1).

High-signal artifacts that the extract layer already surfaces — highlighted cells, bold
runs, PivotTable / AutoFilter conditions, code parameter values — get buried inside the
1200-char retrieval chunks, so a highlight / parameter / condition question often fails to
retrieve the *one line* that answers it (and the LLM, reading a big chunk, overlooks it too).

This module distils those artifacts into **one-fact-per-line** rows, each carrying compact
metadata (``案件 / ファイル / 版 / 種別``). ``index.build()`` embeds every fact row as its own
independent index entry alongside the ordinary chunks (the raw chunks stay for BM25), and
``retrieve`` boosts a matching fact row so it surfaces as high-signal evidence.

Everything here is deterministic and reproducible (no LLM). The only heavy dependency
(openpyxl, reached lazily through ``pivotcond``) is imported inside the pivot helper, so
importing this module at serve time — or with only ``doc.text`` and no workbook — is free.
"""
from __future__ import annotations

import os
import re

from src.rag.corpus import FileRef, nfc

# Feature flag (default OFF — opt-in). Build + serve with RAG_FACT_INDEX=1 to add fact rows to
# the index and boost them at retrieval; with the flag unset the whole layer is a structural
# no-op (no fact rows emitted, no retrieval boost, no evidence relabelling), so the champion
# serving path stays byte-identical. It ships opt-in because reranking retrieval can change
# committed answers, and this project's hard lesson (valid30/holdout proxies are non-predictive;
# #4/#5 regressed the real leaderboard despite good local proxies) is that the production path
# must not change until confirmed on 関門3 (the real SIGNATE leaderboard), which is currently
# blocked on human re-auth. See docs/ai/experiment_ledger.jsonl (SOT-2449).
_ON = {"1", "true", "yes", "on"}


def enabled() -> bool:
    return os.getenv("RAG_FACT_INDEX", "0").strip().lower() in _ON


# Bound how many fact rows a single document may contribute, so a pathological sheet can't
# flood the index (mirrors index.MAX_CHUNKS_PER_DOC in spirit).
MAX_FACTS_PER_DOC = 120

# Curated code parameters worth a standalone fact (aligns with the config_hyperparam /
# config_model_type archetypes). Numeric or short-string assignments only.
_CODE_KEYS = (
    "model_type", "random_state", "n_estimators", "max_depth", "test_size", "n_splits",
    "learning_rate", "max_iter", "num_leaves", "subsample", "colsample_bytree", "min_child_samples",
    "reg_alpha", "reg_lambda", "C", "gamma", "kernel", "epochs", "batch_size", "hidden_size",
    "dropout", "n_neighbors", "alpha", "l1_ratio", "penalty", "solver",
)
_ASSIGN = re.compile(
    r"^\s*(?P<name>[A-Za-z_][A-Za-z0-9_]*)\s*=\s*"
    r"(?P<val>\"[^\"]*\"|'[^']*'|[0-9]+(?:\.[0-9]+)?|True|False)\s*(?:#.*)?$"
)

# xlsx highlight block: office.extract_xlsx emits "【ハイライトされたセル】" then indented
# "  COORD(color): value" rows.
_HL_CELL = re.compile(r"^\s+(?P<coord>[A-Z]+\d+)\((?P<color>[^)]+)\):\s*(?P<val>.+)$")

# Version / edition token from a filename (最新 / 旧版 / r2 / v3 / ver2 / 2024...). Best-effort
# metadata only; "" when nothing recognisable.
_VER = re.compile(
    r"(最新版?|新版|旧版|旧ファイル|前回版|初版|"
    r"(?:^|[_\-\s])(?:r|v|ver|version|版)\s*\d+|"
    r"(?:^|[_\-\s])20\d{2}(?:[01]\d[0-3]\d)?)",
    re.IGNORECASE,
)


def version_of(name: str) -> str:
    """Deterministic edition/version token from a filename stem, or '' when none is present."""
    n = nfc(name)
    stem = n.rsplit(".", 1)[0] if "." in n else n
    m = _VER.search(stem)
    if not m:
        return ""
    return re.sub(r"^[_\-\s]+", "", m.group(0)).strip()


def _header(ref: FileRef, label: str) -> str:
    proj = ref.project or "-"
    ver = version_of(ref.name)
    ver_part = f" | 版: {ver}" if ver else ""
    return f"[fact | 案件: {proj} | ファイル: {ref.rel}{ver_part} | 種別: {label}]"


def _dedupe(rows: list[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for r in rows:
        key = r.strip()
        if key and key not in seen:
            seen.add(key)
            out.append(r)
    return out


# ------------------------------- marker facts (from doc.text) --------------------------------
def _marker_facts(ref: FileRef, doc_text: str) -> list[str]:
    """Facts distilled from the formatting markers office.extract_* already writes into text."""
    out: list[str] = []
    lines = doc_text.splitlines()
    in_hl_block = False
    for line in lines:
        s = line.rstrip()
        # docx bold header: 【太字箇所】A / B / C
        if s.startswith("【太字箇所】"):
            body = s[len("【太字箇所】"):]
            for term in body.split(" / "):
                t = term.strip()
                if t:
                    out.append(f"{_header(ref, '太字')} {t}")
            in_hl_block = False
            continue
        # docx run highlight: 【ハイライト】text
        if s.startswith("【ハイライト】"):
            t = s[len("【ハイライト】"):].strip()
            if t:
                out.append(f"{_header(ref, 'ハイライト')} {t}")
            in_hl_block = False
            continue
        # pptx run highlight / shape fill: 【ハイライト:色】text or 【図形塗り:色】text
        m = re.match(r"^【(ハイライト|図形塗り):(?P<color>[^】]+)】(?P<txt>.+)$", s)
        if m:
            label = "ハイライト" if m.group(1) == "ハイライト" else "図形塗り"
            out.append(f"{_header(ref, label)} {m.group('color')}: {m.group('txt').strip()}")
            in_hl_block = False
            continue
        # xlsx highlight block start
        if s.strip() == "【ハイライトされたセル】":
            in_hl_block = True
            continue
        if in_hl_block:
            cm = _HL_CELL.match(line)
            if cm:
                out.append(f"{_header(ref, 'ハイライトセル')} "
                           f"{cm.group('coord')}({cm.group('color')}): {cm.group('val').strip()}")
                continue
            # a non-matching line ends the highlight block
            if s.strip():
                in_hl_block = False
    return out


# --------------------------------------- code facts -----------------------------------------
def _code_facts(ref: FileRef, doc_text: str) -> list[str]:
    """Standalone facts for curated code parameter assignments (model_type / hyper-params)."""
    out: list[str] = []
    for raw in doc_text.splitlines():
        m = _ASSIGN.match(raw)
        if not m:
            continue
        name = m.group("name")
        if name not in _CODE_KEYS:
            continue
        val = m.group("val").strip().strip("\"'")
        out.append(f"{_header(ref, 'パラメータ')} {name} = {val}")
    return out


# -------------------------------------- pivot facts -----------------------------------------
def _pivot_facts(ref: FileRef) -> list[str]:
    """PivotTable superlative-condition and AutoFilter facts, read from the workbook XML.

    Deterministic and additive-safe: any parse/compute failure is swallowed and simply yields
    no fact for that source (never raises into the build)."""
    out: list[str] = []
    try:
        from src.rag import pivotcond
    except Exception:
        return out
    path = ref.path
    # PivotTable: for each measure, the max-aggregation leaf-group condition.
    try:
        specs = pivotcond._pivot_specs(path)
    except Exception:
        specs = []
    for spec in specs:
        for dfname, fld, _sub in spec.data_fields:
            base = spec.fields[fld] if 0 <= fld < len(spec.fields) else ""
            if not base:
                continue
            q = f"{base}が最も高い層の抽出条件と集計内容"
            try:
                cond = pivotcond._compute_pivot(path, spec, q)
            except Exception:
                cond = None
            if cond:
                out.append(f"{_header(ref, 'Pivot条件')} {nfc(base)}が最も高い層: {cond}")
    # AutoFilter: the active extraction condition.
    try:
        fspecs = pivotcond._filter_specs(path)
    except Exception:
        fspecs = []
    for fs in fspecs:
        if fs.conditions:
            out.append(f"{_header(ref, 'フィルタ条件')} {fs.sheet}シート抽出条件: "
                       + "、".join(fs.conditions))
    return out


# ------------------------------------------ API ---------------------------------------------
def fact_rows(ref: FileRef, doc_text: str) -> list[str]:
    """All deterministic fact rows for one extracted document.

    A pure function of ``(ref, doc_text)`` — the ``enabled()`` flag is checked at the call
    sites (``index.build`` / ``retrieve``), not here, so the extraction stays independently
    testable. Each row is a self-describing, embeddable line: a bracketed metadata header
    (``案件 / ファイル / 版 / 種別``) followed by the single normalised fact. Every source is
    guarded so a malformed document degrades to fewer facts, never an exception."""
    rows: list[str] = []
    try:
        rows += _marker_facts(ref, doc_text)
    except Exception:
        pass
    if ref.ext in ("py", "toml", "ipynb"):
        try:
            rows += _code_facts(ref, doc_text)
        except Exception:
            pass
    if ref.ext == "xlsx" and not ref.name.startswith("~$"):
        rows += _pivot_facts(ref)
    return _dedupe(rows)[:MAX_FACTS_PER_DOC]
