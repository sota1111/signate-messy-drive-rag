"""chart_numcache — read the *exact* plotted values of an embedded xlsx/pptx chart from its ``numCache``.

A great many corpus questions point at a **chart** ("グラフの◯◯の値は？", "系列Xの3月の数値は？").
A vision read of the rendered picture guesses those numbers from pixels and misreads decimals, axis
scale and overlapping bars. But an Office chart does not *store a picture* — it stores the data. Every
``.xlsx`` / ``.pptx`` chart is an OOXML ``c:chartSpace`` part (``xl/charts/chartN.xml`` /
``ppt/charts/chartN.xml``) whose each series carries a **cache of the values Excel last computed**:

* ``<c:val><c:numRef><c:numCache>`` — the plotted numbers (``<c:pt idx="i"><c:v>…</c:v></c:pt>``),
* ``<c:cat><c:strCache|c:numCache>`` — the category axis labels,
* ``<c:tx><c:strRef><c:strCache>`` (or a literal ``<c:v>``) — the series name,
* scatter/bubble charts use ``<c:xVal>`` / ``<c:yVal>`` / ``<c:bubbleSize>`` instead of ``cat``/``val``.

Because the cache is Excel's own snapshot of the source cells, ``numCache`` value ``==`` the cell's
computed value — so reading it is the *exact* plot value, not a vision estimate. This tool walks those
records deterministically and returns them under the unified :mod:`src.rag.tools.contract` shape
``{value, evidence, method}`` so the generation layer can call it exactly like the other tools.

* ``value``    — ``{charts, n_charts, n_series}``. ``charts`` is a list of ``{member, chart_types,
  series:[…]}``; each series is ``{name, categories, values, x_values, y_values, bubble_sizes,
  value_formula, cat_formula, format_code}`` (axis lists absent for that chart kind are ``[]``).
* ``evidence`` — ``file`` (+ ``member`` per chart), ``n_charts``, ``n_series`` and the source formulas.
* ``method``   — ``engine="chart_numcache"``; pure XML read (no secret, NFC-normalized text).

Accepts raw chart XML ``bytes``, a whole ``.xlsx`` / ``.pptx`` file (path / :class:`FileRef` / NFC
corpus-relative reference), and resolves NFD-on-disk names NFC-aware like the rest of the toolset.
"""
from __future__ import annotations

import re
import zipfile
from pathlib import Path
from typing import Any
from xml.etree import ElementTree as ET

from src.rag.corpus import FileRef, nfc
from src.rag.tools import contract
from src.rag.tools.contract import ContractError

# Chart parts live at ``<pkg>/charts/chartN.xml`` (xlsx: ``xl/charts/``, pptx: ``ppt/charts/``). The
# sibling ``colorsN.xml`` / ``styleN.xml`` are theme, not data — match only the numbered chart body.
_CHART_MEMBER = re.compile(r"charts/chart\d+\.xml$", re.IGNORECASE)

# The chart-type element local-names whose children hold ``c:ser`` (used only to report the kind).
_CHART_TYPE_TAGS = frozenset({
    "barChart", "bar3DChart", "lineChart", "line3DChart", "pieChart", "pie3DChart", "ofPieChart",
    "doughnutChart", "areaChart", "area3DChart", "scatterChart", "bubbleChart", "radarChart",
    "stockChart", "surfaceChart", "surface3DChart",
})


def _local(tag: str) -> str:
    """Local name of a namespaced ElementTree tag (``{ns}val`` → ``val``)."""
    return tag.rsplit("}", 1)[-1]


def _child(el: ET.Element, name: str) -> ET.Element | None:
    for c in el:
        if _local(c.tag) == name:
            return c
    return None


def _children(el: ET.Element, name: str):
    return [c for c in el if _local(c.tag) == name]


def _to_number(text: str) -> Any:
    """Parse a ``<c:v>`` numeric string to int when integral else float; keep the string if unparsable."""
    s = text.strip()
    try:
        f = float(s)
    except (TypeError, ValueError):
        return nfc(s)
    return int(f) if f.is_integer() and "." not in s and "e" not in s.lower() else f


def _read_points(axis: ET.Element) -> dict[str, Any]:
    """Decode a ``c:cat``/``c:val``/``c:tx``/``c:xVal``/``c:yVal``/``c:bubbleSize`` element.

    Returns ``{values, formula, format_code, numeric}`` where ``values`` is ordered by ``c:pt`` ``idx``
    (missing indices filled with ``None``). Handles the reference forms (``numRef``/``strRef``/
    ``multiLvlStrRef`` wrapping a ``numCache``/``strCache``), the inline literal forms (``numLit``/
    ``strLit``), and a bare ``<c:v>`` (a directly-typed series name). ``numeric`` marks a number cache
    so values are parsed to int/float (the exact plot value) rather than kept as label strings.
    """
    # A directly-typed value (common for series names: ``<c:tx><c:v>Name</c:v></c:tx>``).
    direct = _child(axis, "v")
    if direct is not None and direct.text is not None:
        return {"values": [nfc(direct.text)], "formula": None, "format_code": None, "numeric": False}

    ref = next((c for c in axis if _local(c.tag) in
                ("numRef", "strRef", "multiLvlStrRef", "numLit", "strLit")), None)
    if ref is None:
        return {"values": [], "formula": None, "format_code": None, "numeric": False}

    kind = _local(ref.tag)
    formula_el = _child(ref, "f")
    formula = nfc(formula_el.text) if formula_el is not None and formula_el.text else None

    # Where the ``pt`` list lives: a *Ref wraps a *Cache; a *Lit holds the pts itself.
    # (Use ``is not None`` — an empty ElementTree element is falsy, so ``or`` would skip a real cache.)
    if kind.endswith("Ref"):
        cache = None
        for cname in ("numCache", "strCache", "multiLvlStrCache"):
            cache = _child(ref, cname)
            if cache is not None:
                break
    else:
        cache = ref
    if cache is None:
        return {"values": [], "formula": formula, "format_code": None, "numeric": kind.startswith("num")}

    numeric = _local(cache.tag).startswith("num") or kind.startswith("num")
    fmt_el = _child(cache, "formatCode")
    format_code = nfc(fmt_el.text) if fmt_el is not None and fmt_el.text else None

    # multiLvlStrCache nests pts under c:lvl; flatten to the first (deepest-labelled) level's order.
    pt_parents = _children(cache, "lvl") or [cache]
    pts: dict[int, Any] = {}
    max_idx = -1
    for parent in pt_parents:
        for pt in _children(parent, "pt"):
            try:
                idx = int(pt.get("idx", "0"))
            except ValueError:
                continue
            v_el = _child(pt, "v")
            raw = v_el.text if v_el is not None else None
            val = _to_number(raw) if (numeric and raw is not None) else (nfc(raw) if raw is not None else None)
            pts.setdefault(idx, val)
            max_idx = max(max_idx, idx)

    count_el = _child(cache, "ptCount")
    try:
        n = int(count_el.get("val")) if count_el is not None else max_idx + 1
    except (TypeError, ValueError):
        n = max_idx + 1
    n = max(n, max_idx + 1)
    values = [pts.get(i) for i in range(n)]
    return {"values": values, "formula": formula, "format_code": format_code, "numeric": numeric}


def _series_name(ser: ET.Element) -> str | None:
    tx = _child(ser, "tx")
    if tx is None:
        return None
    vals = _read_points(tx)["values"]
    return vals[0] if vals else None


def _parse_series(ser: ET.Element) -> dict[str, Any]:
    """Turn one ``c:ser`` into ``{name, categories, values, x_values, y_values, bubble_sizes, …}``."""
    out: dict[str, Any] = {
        "name": _series_name(ser),
        "categories": [], "values": [], "x_values": [], "y_values": [], "bubble_sizes": [],
        "value_formula": None, "cat_formula": None, "format_code": None,
    }
    for tag, key, fkey in (
        ("cat", "categories", "cat_formula"),
        ("val", "values", "value_formula"),
        ("xVal", "x_values", None),
        ("yVal", "y_values", None),
        ("bubbleSize", "bubble_sizes", None),
    ):
        el = _child(ser, tag)
        if el is None:
            continue
        pts = _read_points(el)
        out[key] = pts["values"]
        if fkey:
            out[fkey] = pts["formula"]
        if tag == "val" and pts["format_code"]:
            out["format_code"] = pts["format_code"]
    return out


def _parse_chart_xml(data: bytes) -> dict[str, Any]:
    """Parse one ``c:chartSpace`` XML blob into ``{chart_types, series:[…]}``.

    Raises :class:`ContractError` if ``data`` is not a chart part (no ``c:ser`` / unparsable XML).
    """
    try:
        root = ET.fromstring(data)
    except ET.ParseError as e:
        raise ContractError(f"not a chart XML: {e}") from None

    chart_types: list[str] = []
    series: list[dict[str, Any]] = []
    for el in root.iter():
        ln = _local(el.tag)
        if ln in _CHART_TYPE_TAGS:
            chart_types.append(ln)
        elif ln == "ser":
            series.append(_parse_series(el))
    if not series:
        raise ContractError("chart XML has no <c:ser> series")
    return {"chart_types": sorted(set(chart_types)), "series": series}


# --------------------------------------------------------------------------- source loading
def _looks_like_zip(data: bytes) -> bool:
    return data[:4] in (b"PK\x03\x04", b"PK\x05\x06", b"PK\x07\x08")


def _read_source(file: str | Path | FileRef | bytes | bytearray) -> tuple[bytes, str, bool]:
    """Load bytes + a display label; ``is_office`` is True for an Office (zip) package, False for raw XML."""
    if isinstance(file, (bytes, bytearray)):
        data = bytes(file)
        return data, "<bytes>", _looks_like_zip(data)
    if isinstance(file, FileRef):
        return file.path.read_bytes(), file.rel or str(file.path), True
    p = Path(file)
    if p.exists():
        return p.read_bytes(), str(p), True
    from src.rag.tools.extract_tools import resolve_ref
    ref = resolve_ref(file)
    return ref.path.read_bytes(), ref.rel or str(ref.path), True


def chart_xml_members(office: str | Path | FileRef | bytes | bytearray) -> list[tuple[str, bytes]]:
    """Return ``(member_name, xml_bytes)`` for every ``charts/chartN.xml`` in an xlsx/pptx, sorted.

    Accepts the office file (path / :class:`FileRef` / NFC corpus-relative ref) or its raw zip bytes.
    """
    data, _label, is_office = _read_source(office)
    if not is_office:
        raise ContractError("not an Office package (expected .xlsx/.pptx zip bytes)")
    import io
    out: list[tuple[str, bytes]] = []
    with zipfile.ZipFile(io.BytesIO(data)) as zf:
        for name in sorted(zf.namelist()):
            if _CHART_MEMBER.search(name):
                out.append((name, zf.read(name)))
    return out


# --------------------------------------------------------------------------- public API
def extract_chart_numcache(file: str | Path | FileRef | bytes | bytearray) -> dict[str, Any]:
    """Read every chart's ``numCache``/``strCache`` from an xlsx/pptx (or raw chart XML) → contract.

    Parameters
    ----------
    file
        Raw chart-XML ``bytes``, a whole ``.xlsx``/``.pptx`` (as zip ``bytes``, a path, a
        :class:`FileRef`, or an NFC corpus-relative reference).

    ``value`` is ``{charts, n_charts, n_series}`` (see the module docstring). Raises
    :class:`ContractError` when the input holds no readable chart series.
    """
    data, label, is_office = _read_source(file)

    parts: list[tuple[str, bytes]] = (
        chart_xml_members(data) if is_office else [(label, data)]
    )

    charts: list[dict[str, Any]] = []
    formulas: list[str] = []
    for member, blob in parts:
        try:
            parsed = _parse_chart_xml(blob)
        except ContractError:
            continue
        parsed["member"] = member if is_office else label
        charts.append(parsed)
        for s in parsed["series"]:
            for f in (s.get("value_formula"), s.get("cat_formula")):
                if f:
                    formulas.append(f)

    if not charts:
        raise ContractError(f"{label}: no readable chart numCache found")

    n_series = sum(len(c["series"]) for c in charts)
    value = {"charts": charts, "n_charts": len(charts), "n_series": n_series}
    return contract.make(
        value,
        engine="chart_numcache",
        evidence={
            "file": label,
            "n_charts": len(charts),
            "n_series": n_series,
            "members": [c["member"] for c in charts],
            "formulas": formulas,
        },
        nfc=True,
    )


def chart_series(file: str | Path | FileRef | bytes | bytearray) -> list[dict[str, Any]]:
    """Convenience: the flat list of every series across all charts (each tagged with its ``member``).

    Thin wrapper over :func:`extract_chart_numcache`; each series dict gains a ``member`` key so the
    caller can tell which chart it came from. Returns ``[]`` only if the file has charts but no series
    (``extract_chart_numcache`` raises when there is nothing at all).
    """
    result = extract_chart_numcache(file)
    out: list[dict[str, Any]] = []
    for chart in result["value"]["charts"]:
        for s in chart["series"]:
            out.append({**s, "member": chart["member"]})
    return out


def series_values(file: str | Path | FileRef | bytes | bytearray, *,
                  name: str | None = None) -> list[Any]:
    """Convenience: the exact plotted ``values`` of one series (by ``name``, else the first series).

    Matches ``name`` NFC-aware. Returns ``[]`` when no matching series exists. Useful for a direct
    "系列Xの数値は？" answer without the caller walking the nested contract.
    """
    want = nfc(name) if name is not None else None
    for s in chart_series(file):
        if want is None or (s.get("name") and nfc(str(s["name"])) == want):
            return s.get("values", [])
    return []
