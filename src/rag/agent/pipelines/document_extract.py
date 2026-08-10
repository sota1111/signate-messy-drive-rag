"""SOT-2611 (Wave B1) — deterministic ``document_extract`` pipeline (ハイライト → 構造読取 → 整形).

Parent PLAN SOT-2602 (決定論先行パイプラインへの反転). Sixth per-type pipeline of the inverted architecture
(after Wave A1 ``version_diff`` SOT-2605, A2 ``numeric`` SOT-2607, A3 ``enum``/``cross_aggregate`` SOT-2608,
A4 ``chart_read``/``spatial`` SOT-2609, B2 ``simple_lookup`` SOT-2612). A *書式型* document_extract — one
whose answer is the **extraction condition + aggregation content behind a highlighted pivot cell** — is
answered **without going through the Gemini investigator loop**. The Stage0 router
(:mod:`src.rag.agent.det_pipeline`) classifies such a question as ``format_check`` (the ``_FORMAT_RE`` cue
``黄色ハイライト…`` in :mod:`src.rag.agent.question_contract`) and dispatches here; Stage3
(:mod:`src.rag.agent.formatting`) leaves the value untouched (this pipeline already returns the gold-shaped
condition string — no LLM naturalize call).

The one tight recognizer this pipeline owns
-------------------------------------------
**黄色ハイライトされたセル/数値の抽出条件と集計内容** (:func:`_highlight_condition`, idx7/15/80 型).
「…において、Sheet2 の黄色にハイライトされたセルの抽出条件と集計内容を答えてください」→ Stage1 で対象ファイル
（プロジェクト＋ファイル名を質問から一意確定）とシート（``Sheet\\d+`` 明記時）を固定 → Stage2 で
ハイライトされた 1 セルを決定論取得し、そのセルが属するピボットの**祖先グループ（抽出条件）と集計名**を再構成:

* **実セルのフラット化アウトライン** (idx15, ``.xlsx`` Sheet1): :func:`src.rag.tools.highlight_extract.highlight_extract`
  で黄セルを 1 つ確定 → ヘッダ行のグルーピング列を、結合セル（前方補完）規則でハイライト行まで前方補完して
  ``列=値、…`` を復元 → 値列ヘッダの集計語（個数/平均/最大…）を集計名とする（SOT-2545 の粒度整形を構造で吸収）。
* **埋込 EMF のアウトライン** (idx80, ``.xlsx`` Sheet2 = セル空・drawing 実体): :func:`src.rag.tools.emf_pivot.extract_emf_pivot`
  で表グリッドとハイライトセルを再構成（SOT-2548 の「Sheet2 を空と誤断定しない」＝ drawing/EMF を読む）→
  アウトライン前方補完（数値グループ列が隣接列に桁プレフィクスとして寄る EMF クラスタリングを移送補正）→
  ``列=値、…`` ＋ 集計名。
* **埋込 EMF の 2D クロス集計** (idx7, ``.pptx``): :func:`src.rag.tools.emf_pivot.resolve_pivot_semantics`
  で ``行フィールド×列フィールド→集計(値フィールド)`` の署名をソース表と照合して一意確定 → ハイライトセルの
  行ラベル/列見出しで ``行f=行ラベル、列f=列見出しで抽出されたデータに対する<集計名> / <値列>`` を組む。

Precision-first negative guards (安全側フォールバック — 誤答/誤「該当なし」を出さない)
--------------------------------------------------------------------------------
対象ファイルが一意に解決できない、ハイライトが 0/複数、集計語が読めない、前方補完が抽出条件を埋め切れない、
2D 署名がソースと一致しない — いずれも ``None``（⇒ LLM フォールバック）。決定論で 1 つに絞れないものは最初から
既存 champion LLM 経路に委ねる（回答数を減らさない・wrong を増やさない）。

Design invariants (shared with the Stage0 router / Stage3 formatting / Wave A1〜B2)
--------------------------------------------------------------------------------
* **No hardcoding.** No idx number, no answer, no corpus fact（プロジェクト名/ファイル名/条件値/集計名）is stored
  here — every value is self-derived from the question via canonical_route + the highlight/EMF extractors.
  Only the generic Japanese colour vocabulary and aggregation vocabulary are named (portable, corpus-free).
* **Fail-open, never abstain-by-error.** Any read/parse/lookup failure returns ``None`` (⇒ LLM fallback);
  the pipeline never raises into the answer path and never *reduces* the number of answered questions.
* **Gated OFF with the router.** Only ever reached from the Stage0 router's det-path, gated by
  ``RAG_DET_PIPELINE_ROUTER`` (default OFF); with the flag off the champion serve path is byte-identical.
"""
from __future__ import annotations

import importlib
import re
from pathlib import Path
from typing import Any

from src.rag.agent import det_pipeline as _det_pipeline

# The base contract this pipeline owns (question_contract.FORMAT_CHECK — the contract the coarse
# ``document_extract`` archetype refines to when a 書式 cue 太字/下線/色/ハイライト is present).
CONTRACT_TYPE = "format_check"

# Generic aggregation vocabulary (pivot value-field function names). Used to spot the value column of a
# flattened-outline pivot and to render its aggregation content. Corpus-free — these are Excel pivot terms.
_AGG_TOKENS = ("個数", "合計", "平均", "最大", "最小", "中央値", "標準偏差", "分散",
               "count", "sum", "average", "max", "min", "median")

# Generic coarse Japanese colour words → the office extractor's coarse colour name (kept in sync with
# :data:`src.rag.extract.office._HEX_COARSE`). Ordered so a compound word (水色) is matched before its
# substring (青 never appears inside 水色, but the order documents the intent). No corpus colour is baked in.
_COLOR_WORDS: "tuple[tuple[str, str], ...]" = (
    ("オレンジ", "オレンジ"), ("水色", "水色"), ("黄色", "黄"), ("黄", "黄"),
    ("赤", "赤"), ("緑", "緑"), ("青", "青"), ("紫", "紫"), ("ピンク", "ピンク"),
)

_HIGHLIGHT_CUE_RE = re.compile(r"ハイライト|マーカー|塗りつぶし|強調")
_CONDITION_CUE_RE = re.compile(r"抽出条件|抽出され|集計")
_SHEET_RE = re.compile(r"(Sheet\s*\d+)", re.I)
_FILE_RE = re.compile(r"([^\s、。「」]+?\.(?:pptx|xlsx|xlsm|docx|pdf|csv|tsv))", re.I)


# --------------------------------------------------------------------------- small deterministic helpers
def _nfc(text: str) -> str:
    from src.rag.corpus import nfc
    return nfc(text or "")


def _cellstr(value: Any) -> str:
    """Normalize a cell value to a trimmed string (int-valued floats lose the ``.0``); ``None`` → ``''``."""
    if value is None:
        return ""
    if isinstance(value, bool):
        return str(value)
    if isinstance(value, float):
        return str(int(value)) if value.is_integer() else str(value)
    if isinstance(value, int):
        return str(value)
    return _nfc(str(value)).strip()


def _resolve_color(question: str) -> "str | None":
    """The single coarse colour name the question names (office vocabulary), or ``None``."""
    q = _nfc(question)
    for word, name in _COLOR_WORDS:
        if word in q:
            return name
    return None


def _target_sheet(question: str) -> "str | None":
    """The worksheet the question pins (``Sheet2`` → ``"Sheet2"``), or ``None`` when unspecified."""
    m = _SHEET_RE.search(_nfc(question))
    return re.sub(r"\s+", "", m.group(1)) if m else None


def _resolve_file_ref(question: str):
    """The single corpus :class:`FileRef` the question names (project + filename), or ``None``.

    Stage1 file確定: resolve the project via the SOT-2494 canonical route, then keep every project file
    whose basename appears verbatim in the question. Exactly one match ⇒ that ref; 0 / >1 ⇒ ``None``
    (ambiguous ⇒ safe fallback). ``import_module`` avoids the tools-package function-shadow (Wave A2〜B2).
    """
    from src.rag.corpus import walk
    cr = importlib.import_module("src.rag.tools.canonical_route")
    try:
        project = cr.resolve_project(question, None)
    except Exception:  # noqa: BLE001 — resolution failure ⇒ fallback, never breaks the answer path
        project = None
    if not project:
        return None
    q = _nfc(question)
    proj = _nfc(project)
    hits = [r for r in walk()
            if _nfc(r.project) == proj and _nfc(Path(r.rel).name) in q]
    # De-dup by rel (NFC) so an NFD/NFC pair of the same file is not counted as two candidates.
    unique = {}
    for r in hits:
        unique.setdefault(_nfc(r.rel), r)
    return next(iter(unique.values())) if len(unique) == 1 else None


# --------------------------------------------------------------------------- outline (flattened) pivot
def _agg_column(header: "list[str]") -> "tuple[int, str] | None":
    """``(index, aggregation_label)`` of the pivot's value-field header (個数/平均/…), or ``None``.

    The aggregation label is the leading token before any ``/ <value column>`` suffix (so ``"個数 / id"``
    and a bare ``"個数"`` both render as ``個数``). Scans left-to-right and takes the first header carrying an
    aggregation word — the grouping columns sit strictly to its left.
    """
    for idx, cell in enumerate(header):
        text = _nfc(cell)
        if not text:
            continue
        # ascii tokens (count/sum/…) match on a WORD boundary so ``Country`` is not read as ``count``;
        # Japanese tokens (個数/平均/…) match as a substring (they carry no boundary risk here).
        if _has_agg_token(text):
            label = text.split("/")[0].strip()
            return (idx, label) if label else None
    return None


def _has_agg_token(text: str) -> bool:
    low = text.lower()
    for tok in _AGG_TOKENS:
        if tok.isascii():
            if re.search(rf"\b{re.escape(tok)}\b", low):
                return True
        elif tok in text:
            return True
    return False


def _outline_condition(table: "list[list[str]]", hl_row: int) -> "dict[str, Any] | None":
    """Resolve a flattened-outline pivot's highlighted row to ``列=値、…で抽出されたデータに対する<集計>``.

    ``table`` is the reconstructed grid (row 0 = header). ``hl_row`` is the highlighted cell's row index.
    Grouping columns are every non-empty header left of the aggregation column; each is forward-filled
    (merged-cell / outline repeat semantics) down to ``hl_row``. A numeric grouping value that the office /
    EMF reader clustered as a ``"<digit> <rest>"`` prefix into the *next* grouping column is migrated back to
    its own column (idx80 の target が Age 列へ寄る EMF アーティファクトの補正). Any grouping condition that
    forward-fill cannot pin ⇒ ``None`` (safe fallback).
    """
    if not table or hl_row <= 0 or hl_row >= len(table):
        return None
    header = [_nfc(c) for c in table[0]]
    agg = _agg_column(header)
    if agg is None:
        return None
    agg_col, agg_label = agg
    group_cols = [i for i in range(agg_col) if header[i]]
    if not group_cols:
        return None

    state: "dict[int, str | None]" = {i: None for i in group_cols}
    for r in range(1, hl_row + 1):
        row = table[r]
        for i in group_cols:
            cell = _nfc(row[i]) if i < len(row) else ""
            # Migrate a leading "<digit> <rest>" prefix into the previous (numeric) grouping column when
            # the reader merged them — the digit belongs to col i-1, the remainder stays in col i.
            m = re.match(r"^(\d+)\s+(.*\S)$", cell)
            if m and (i - 1) in state:
                state[i - 1] = m.group(1)
                cell = m.group(2).strip()
            if cell:
                state[i] = cell

    conds: "list[tuple[str, str]]" = []
    for i in group_cols:
        val = state[i]
        if not val:
            return None  # an unfilled ancestor group ⇒ cannot state the full condition ⇒ fallback
        conds.append((header[i], val))
    cond_str = "、".join(f"{name}={val}" for name, val in conds)
    return {
        "answer": f"{cond_str}で抽出されたデータに対する{agg_label}",
        "conditions": [{name: val} for name, val in conds],
        "aggregation": agg_label,
        "shape": "outline_pivot",
    }


# --------------------------------------------------------------------------- 2D cross-tab pivot
def _twod_condition(table: "list[list[str]]", hl_cell: "dict[str, Any]", ref) -> "dict[str, Any] | None":
    """Resolve a 2D cross-tab pivot's highlighted cell via source-table semantics, or ``None``.

    Uses :func:`src.rag.tools.emf_pivot.resolve_pivot_semantics` to pin ``row_field × column_field →
    aggregation(value_field)`` against the project's source tables (a signature is returned only when it
    explains the grid unambiguously). The highlighted cell's ``row_label`` / ``col_header`` then give the
    two condition values — ``行f=行ラベル、列f=列見出しで抽出されたデータに対する<集計名> / <値列>``.
    """
    row_label = _nfc(str(hl_cell.get("row_label") or ""))
    col_header = _nfc(str(hl_cell.get("col_header") or ""))
    if not row_label or not col_header:
        return None
    emf = importlib.import_module("src.rag.tools.emf_pivot")
    try:
        sources = emf._pptx_data_sources(ref)
        sem = emf.resolve_pivot_semantics(table, sources)
    except Exception:  # noqa: BLE001
        return None
    if not sem:
        return None
    row_field = _nfc(str(sem.get("row_field") or ""))
    col_field = _nfc(str(sem.get("column_field") or ""))
    agg_label = _nfc(str(sem.get("aggregation_label") or ""))
    target = _nfc(str(sem.get("target_column") or ""))
    if not (row_field and col_field and agg_label and target):
        return None
    answer = (f"{row_field}={row_label}、{col_field}={col_header}"
              f"で抽出されたデータに対する{agg_label} / {target}")
    return {
        "answer": answer,
        "conditions": [{row_field: row_label}, {col_field: col_header}],
        "aggregation": f"{agg_label} / {target}",
        "shape": "crosstab_pivot",
    }


# --------------------------------------------------------------------------- EMF highlight dispatch
def _emf_yellow_cell(emf_bytes: bytes, color: str) -> "tuple[list[list[str]], dict[str, Any]] | None":
    """``(table, highlight_cell)`` for the single ``color``-highlighted cell in an EMF, or ``None``.

    Reconstructs the embedded EMF pivot image and keeps highlights whose RGB the office classifier names
    ``color`` (so an EMF ``#FFFF00`` fill is matched to a 黄色 question). Requires exactly one such cell —
    0 / >1 ⇒ ``None`` (ambiguous ⇒ safe fallback).
    """
    emf = importlib.import_module("src.rag.tools.emf_pivot")
    from src.rag.extract import office
    try:
        res = emf.extract_emf_pivot(emf_bytes)
    except Exception:  # noqa: BLE001 — a malformed / textless EMF ⇒ fallback
        return None
    if res.get("method", {}).get("fallback"):
        return None
    value = res.get("value") or {}
    table = value.get("table")
    if not table:
        return None
    cells: "list[dict[str, Any]]" = []
    for hl in value.get("highlights", []):
        hexname = str(hl.get("color") or "").lstrip("#")
        try:
            named = office._color_name(hexname)
        except Exception:  # noqa: BLE001
            named = None
        if named == color:
            cells.extend(hl.get("cells", []))
    if len(cells) != 1:
        return None
    return table, cells[0]


def _xlsx_sheet_emf(ref, sheet: "str | None") -> "list[bytes]":
    """Embedded EMF blobs drawn on ``sheet`` (or every sheet when ``sheet`` is ``None``), in member order.

    Reads the workbook zip's sheet → drawing → media relationship chain directly, so a worksheet whose
    cells are empty but whose content is a pasted PivotTable picture (SOT-2548 Sheet2) still yields its
    image bytes. Returns ``[]`` on any structural miss.
    """
    import zipfile
    out: "list[bytes]" = []
    try:
        z = zipfile.ZipFile(str(ref.path))
    except Exception:  # noqa: BLE001
        return out
    with z:
        names = set(z.namelist())
        try:
            wb = z.read("xl/workbook.xml").decode("utf-8", "ignore")
            rels = z.read("xl/_rels/workbook.xml.rels").decode("utf-8", "ignore")
        except Exception:  # noqa: BLE001
            return out
        sheet_rids = re.findall(r'<sheet[^>]*name="([^"]+)"[^>]*r:id="([^"]+)"', wb)
        rid_target = dict(re.findall(r'Id="([^"]+)"[^>]*?Target="([^"]+)"', rels))
        want = re.sub(r"\s+", "", sheet).lower() if sheet else None
        for name, rid in sheet_rids:
            if want is not None and re.sub(r"\s+", "", name).lower() != want:
                continue
            target = rid_target.get(rid, "")
            sheet_member = "xl/" + target.lstrip("/") if not target.startswith("xl/") else target
            base = sheet_member.rsplit("/", 1)[-1]
            srels = f"xl/worksheets/_rels/{base}.rels"
            if srels not in names:
                continue
            drawings = [t for t in re.findall(r'Target="([^"]+)"', z.read(srels).decode("utf-8", "ignore"))
                        if "drawing" in t]
            for d in drawings:
                dmember = "xl/" + d.replace("../", "")
                dbase = dmember.rsplit("/", 1)[-1]
                drels = f"xl/drawings/_rels/{dbase}.rels"
                if drels not in names:
                    continue
                for t in re.findall(r'Target="([^"]+)"', z.read(drels).decode("utf-8", "ignore")):
                    if t.lower().endswith(".emf"):
                        member = "xl/" + t.replace("../", "")
                        if member in names:
                            out.append(z.read(member))
    return out


# --------------------------------------------------------------------------- per-extension entry points
def _xlsx_highlight_condition(ref, color: str, sheet: "str | None", question: str) -> "dict[str, Any] | None":
    """Highlighted-cell extraction condition for an ``.xlsx``/``.xlsm`` — real cells first, then EMF drawing."""
    # (1) Real highlighted cells (a flattened pivot living in genuine cells, e.g. Sheet1 idx15).
    he = importlib.import_module("src.rag.tools.highlight_extract")
    try:
        res = he.highlight_extract(str(ref.path), color=color)
        items = res.get("value", []) if isinstance(res, dict) else []
    except Exception:  # noqa: BLE001
        items = []
    want = re.sub(r"\s+", "", sheet).lower() if sheet else None
    real = []
    for it in items:
        ev = it.get("evidence", {}) if isinstance(it, dict) else {}
        cell_sheet = re.sub(r"\s+", "", _nfc(str(ev.get("sheet") or ""))).lower()
        cell_ref = str(ev.get("cell") or "")
        if want is not None and cell_sheet != want:
            continue
        m = re.match(r"^([A-Za-z]+)(\d+)$", cell_ref)
        if m:
            real.append((_nfc(str(ev.get("sheet") or "")), m.group(1), int(m.group(2))))
    if len(real) == 1:
        return _realcell_outline(ref, *real[0])
    if real:  # >1 highlighted real cell in the target sheet ⇒ ambiguous ⇒ fallback
        return None

    # (2) No real highlight in the target sheet ⇒ the pivot is a pasted picture (EMF drawing), e.g. Sheet2.
    for emf_bytes in _xlsx_sheet_emf(ref, sheet):
        found = _emf_yellow_cell(emf_bytes, color)
        if found is None:
            continue
        table, cell = found
        return _outline_condition(table, int(cell.get("row", -1))) or _twod_condition(table, cell, ref)
    return None


def _realcell_outline(ref, sheet_name: str, col_letters: str, row_no: int) -> "dict[str, Any] | None":
    """Build the outline grid from real worksheet cells up to the highlighted row, then resolve it."""
    office = importlib.import_module("src.rag.extract.office")
    try:
        wb = office._xlsx_from(ref, None)
        if sheet_name not in wb.sheetnames:
            return None
        ws = wb[sheet_name]
        rows = list(ws.iter_rows(min_row=1, max_row=row_no, values_only=True))
    except Exception:  # noqa: BLE001
        return None
    if len(rows) < 2:
        return None
    table = [[_cellstr(v) for v in row] for row in rows]
    return _outline_condition(table, row_no - 1)  # sheet row N → table index N-1 (row 0 = header)


def _pptx_highlight_condition(ref, color: str) -> "dict[str, Any] | None":
    """Highlighted-cell extraction condition for a ``.pptx`` (PivotTable pasted as an embedded EMF picture)."""
    emf = importlib.import_module("src.rag.tools.emf_pivot")
    try:
        blobs = emf.emf_blobs_from_pptx(ref)
    except Exception:  # noqa: BLE001
        return None
    for _name, data in blobs:
        found = _emf_yellow_cell(data, color)
        if found is None:
            continue
        table, cell = found
        return _outline_condition(table, int(cell.get("row", -1))) or _twod_condition(table, cell, ref)
    return None


# --------------------------------------------------------------------------- recognizer
def _highlight_condition(question: str) -> "dict[str, Any] | None":
    """Deterministic 「<色>ハイライトされたセル/数値の抽出条件と集計内容」 read, or ``None`` (⇒ fallback)."""
    q = _nfc(question)
    if not (_HIGHLIGHT_CUE_RE.search(q) and _CONDITION_CUE_RE.search(q)):
        return None
    color = _resolve_color(q)
    if color is None:
        return None
    ref = _resolve_file_ref(question)
    if ref is None:
        return None
    sheet = _target_sheet(q)

    ext = (ref.ext or "").lower()
    if ext in {"xlsx", "xlsm"}:
        resolved = _xlsx_highlight_condition(ref, color, sheet, question)
    elif ext == "pptx":
        resolved = _pptx_highlight_condition(ref, color)
    else:
        return None
    if not resolved or not resolved.get("answer"):
        return None

    evidence = {
        "file": ref.rel,
        "project": ref.project,
        "sheet": sheet,
        "color": color,
        "conditions": resolved.get("conditions"),
        "aggregation": resolved.get("aggregation"),
        "route": f"highlight/emf→{resolved.get('shape')}",
    }
    method = {
        "engine": "document_extract_det_pipeline",
        "contract": CONTRACT_TYPE,
        "shape": resolved.get("shape"),
        "boundary_rules": "merged_cell_forward_fill;highlight_single_cell;source_verified_signature",
        "confidence": 1.0,
    }
    return {"value": resolved["answer"], "evidence": evidence, "method": method}


# --------------------------------------------------------------------------- pipeline
_RECOGNIZERS = (_highlight_condition,)


def pipeline(question: str, *, profile: Any = None) -> "dict[str, Any] | None":
    """Deterministic ``format_check`` answer as a ``{value, evidence, method}`` contract, or ``None``.

    Tries each tight recognizer in order and returns the first grounded contract; ``None`` ⇒ no recognizer
    could pin the extraction condition deterministically ⇒ the router falls back to the LLM loop. Never
    raises into the answer path.
    """
    try:
        q = _nfc(question)
        if not q:
            return None
        for recognizer in _RECOGNIZERS:
            result = recognizer(question)
            if result is not None and result.get("value") is not None:
                return result
        return None
    except Exception:  # noqa: BLE001 — any failure falls back to the LLM loop, never breaks the answer path
        return None


def register(*, replace: bool = True) -> None:
    """Register this pipeline for the ``format_check`` contract (idempotent: ``replace=True`` by default)."""
    _det_pipeline.register(CONTRACT_TYPE, pipeline, replace=replace)


# Self-register on import — importing :mod:`src.rag.agent.pipelines` (the router's lazy bootstrap) wires
# this pipeline into the Stage0 registry. Idempotent so a re-import / test reload never raises.
register(replace=True)
