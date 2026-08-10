"""SOT-2609 (Wave A4) — deterministic ``chart_read`` / ``spatial`` pipeline (chart_numcache + 座席directory → 整形).

Parent PLAN SOT-2602 (決定論先行パイプラインへの反転). Fourth per-type pipeline of the inverted architecture
(after Wave A1 ``version_diff`` SOT-2605, A2 ``numeric`` SOT-2607, A3 ``enum``/``cross_aggregate`` SOT-2608):
a chart-value or seating-spatial question is answered **without going through the Gemini investigator loop**
whenever the厳密経路 (chart ``numCache`` 厳密プロット値 / seating directory の pixel-hash pin) can ground the
value. The Stage0 router (:mod:`src.rag.agent.det_pipeline`) classifies the contract and dispatches here;
Stage3 (:mod:`src.rag.agent.formatting`) renders the grounded value into gold 書式 (numeric / list types are
template-first — no LLM naturalize call).

The two contract types this pipeline owns — and the seating overlap with A3
-------------------------------------------------------------------------
* ``chart_read`` (:func:`chart_read_pipeline`) — グラフ読取. Two tight, mutually-exclusive recognizers:
  1. **ヒストグラム最大カウント** (:func:`_histogram_max_count`, idx10 型). 「…のヒストグラムで最も多いカウント数」
     → :func:`src.rag.tools.chart_numcache.read_chart_values` の厳密経路 (numCache 優先・埋め込み画像は元データ
     再集計) ``operation='histogram_max_count'``. Vision の目視値読みは禁止（SOT-2507 踏襲）— 厳密経路が値を
     作れない時のみ棄権/フォールバック。
  2. **色/カテゴリ指定の系列点読み** (:func:`_chart_point_read`, idx33 型). 「グラフ2で x=3 のときの青色の折れ線の
     y の値を小数第5位で」→ chart_numcache の numCache から、指定グラフ・指定カテゴリ・指定色系列の厳密プロット値を
     読む。系列色は chart XML の ``srgbClr``/``schemeClr`` をテーマ解決して coarse な色名に分類し、質問の色語に
     **一意一致**する系列だけを採用（曖昧なら棄権）。
* ``spatial`` (:func:`spatial_pipeline`, idx58 型) — 座席表の空間関係（向かい/隣/右側/左側…）を座席ディレクトリ
  （SOT-2488 pixel-hash pin＋向きフレーム）で決定論解決。
* **座席列挙の A3 合成** — 「右側に座っている人をすべて挙げて」(idx44) は archetype=enum_set ゆえ分類器が
  ``full_enumeration`` を割り当てる（A3 SOT-2608 の担当）。A3 の既存挙動を **byte-identical に温存**したまま座席
  列挙を前段合成するため、:func:`register` は A3 の ``full_enumeration`` パイプラインを捕捉し、座席 recognizer を
  前段に足した薄いラッパを replace 登録する（座席キューの無い列挙質問は素通しで A3 に委譲）。

Precision-first negative guards (安全側フォールバック — 誤答/誤「該当なし」を出さない)
--------------------------------------------------------------------------------
決定論スライスは意図的に狭い。各 recognizer は対象（ファイル/グラフ/系列/座席）を**一意に固定**できない限り
``None``（⇒ LLM フォールバック）を返す。chart_numcache / seating_lookup が厳密値を作れない・曖昧な場合はいずれも
棄権へ落ち、既存 match を減らさない（回答数を減らさない）。ipynb にレンダリングされた matplotlib 図（idx56 型）は
numCache を持たず、y 軸目盛りは vision の目視読みでしか取れないため、この pipeline は該当を **grounded しない**
（numeric 契約へ分類され安全棄権のまま — 厳密経路なし）。

Design invariants (shared with the Stage0 router / Stage3 formatting / Wave A1〜A3)
--------------------------------------------------------------------------------
* **No hardcoding.** No idx number, no answer, no corpus fact（座席/氏名/EXT/カウント値）is stored here — every
  value is self-derived from the question via canonical_route + chart_numcache / seating_chart.
* **Fail-open, never abstain-by-error.** Any read/parse/lookup failure returns ``None`` (⇒ LLM fallback); the
  pipeline never raises into the answer path and never *reduces* the number of answered questions.
* **Gated OFF with the router.** Only ever reached from the Stage0 router's det-path, gated by
  ``RAG_DET_PIPELINE_ROUTER`` (default OFF); with the flag off the champion serve path is byte-identical.
"""
from __future__ import annotations

import importlib
import re
import zipfile
from typing import Any
from xml.etree import ElementTree as ET

from src.rag.agent import det_pipeline as _det_pipeline

# The two contract types this pipeline owns outright (question_contract.CHART_READ / SPATIAL).
CONTRACT_TYPES: "tuple[str, ...]" = ("chart_read", "spatial")


# --------------------------------------------------------------------------- small deterministic helpers
def _nfc(text: str) -> str:
    from src.rag.corpus import nfc
    return nfc(text or "")


# A named office/data source in the question (…の <name>.xlsx …).
_NAMED_FILE_RE = re.compile(r"([^\s　のをはがにで、。「」]+\.(?:xlsx|xlsm|docx|pptx))", re.I)
# ヒストグラム最大カウント cue.
_HIST_RE = re.compile(r"ヒストグラム|histogram", re.I)
_MAX_COUNT_RE = re.compile(r"(最も多い|最大|一番多い|最多|ピーク).{0,6}(カウント|度数|件数|本数|頻度|数)")
# A leading data-column token in「<col>のヒストグラム」(ASCII識別子 or 日本語列名 non-particle run).
_HIST_COL_RE = re.compile(r"([A-Za-z0-9_.一-鿿]+?)\s*の?\s*ヒストグラム")
# Chart index (グラフ2 / 図2 / chart 2) and category (x=3 / x が 3).
_CHART_IDX_RE = re.compile(r"(?:グラフ|ｸﾞﾗﾌ|図|chart)\s*[（(]?\s*(\d+)", re.I)
_XCAT_RE = re.compile(r"[xｘＸ]\s*[=＝はがの]?\s*(\d+)")
_DECIMALS_RE = re.compile(r"小数第\s*(\d+)\s*位")
# A honorific-bound person name (佐藤さん / 井上氏).
_PERSON_RE = re.compile(r"([^\s　のはがをにでへとや、。「」（）\d]{1,8}?)(?:さん|氏|様|君|くん)")
# Geometric seating cue — fire the seating recognizer ONLY on a genuine spatial question.
_SEATING_CUE_RE = re.compile(
    r"座席|着席|席次|座って|座った|向かい|向い|対面|正面|右側|左側|右手|左手|隣|となり|同じ列|同列|同じ行|同行")

# Color surface → coarse name the RGB classifier (:func:`src.rag.extract.office._color_name`) emits, and the
# finer shades a human still calls by that coarse name (SOT-2564 の色ファミリと同義)。青⊇水色/紺 なので
# 「青色の折れ線」が 水色 分類の系列にも一致する。純粋な語彙表 — corpus fact なし。
_COLOR_SURFACES = {
    "青": "青", "あお": "青", "ブルー": "青", "blue": "青", "紺": "青",
    "水色": "水色", "みずいろ": "水色", "シアン": "水色", "cyan": "水色",
    "赤": "赤", "あか": "赤", "レッド": "赤", "red": "赤",
    "オレンジ": "オレンジ", "橙": "オレンジ", "orange": "オレンジ",
    "緑": "緑", "みどり": "緑", "グリーン": "緑", "green": "緑",
    "紫": "紫", "むらさき": "紫", "purple": "紫",
    "黄": "黄", "きいろ": "黄", "黄色": "黄", "yellow": "黄",
    "ピンク": "ピンク", "桃": "ピンク", "pink": "ピンク",
    "灰": "灰", "グレー": "灰", "gray": "灰", "grey": "灰",
    "茶": "茶", "brown": "茶", "黒": "黒", "black": "黒",
}
_COLOR_FAMILY = {
    "青": {"青", "水色", "紺"},
    "水色": {"水色", "青"},
}


def _query_color(question_nfc: str) -> "str | None":
    """The single coarse color the question asks for (青/水色/…), or ``None`` if none / ambiguous.

    Longest surface first so 「水色」 is not shadowed by 「青」 wins; returns ``None`` when two *different*
    coarse colors appear (⇒ the recognizer falls back rather than guessing which line)."""
    found: "set[str]" = set()
    for surface in sorted(_COLOR_SURFACES, key=len, reverse=True):
        if surface in question_nfc:
            found.add(_COLOR_SURFACES[surface])
    return next(iter(found)) if len(found) == 1 else None


def _color_matches(series_color: "str | None", query: str) -> bool:
    """True when a series' classified coarse color equals the query (or is in the query's family)."""
    if not series_color:
        return False
    if series_color == query:
        return True
    return series_color in _COLOR_FAMILY.get(query, frozenset())


# --------------------------------------------------------------------------- file resolution (canonical route)
def _resolve_project(question: str) -> "str | None":
    """The single project the question is about, via the SOT-2494 canonical route (glossary), or ``None``.

    ``import_module`` (not ``from … import``) so ``src.rag.tools`` re-exporting ``resolve_project`` as a
    function does not shadow the submodule attribute at monkeypatch time (tools-shadow gotcha, shared with
    Wave A2/A3)."""
    cr = importlib.import_module("src.rag.tools.canonical_route")
    try:
        return cr.resolve_project(question, None) or None
    except Exception:  # noqa: BLE001 — resolution failure ⇒ fall back, never break the answer path
        return None


def _find_named_file(question_nfc: str, project: "str | None"):
    """The one corpus file the question names by basename (optionally within ``project``), or ``None``.

    Requires an exact NFC ``name`` match resolving to exactly one ref so an ambiguous / unnamed file falls
    back rather than guessing which document to read."""
    from src.rag.corpus import walk

    m = _NAMED_FILE_RE.search(question_nfc)
    if not m:
        return None
    fname = _nfc(m.group(1))
    refs = [r for r in walk() if _nfc(r.name) == fname and (project is None or r.project == project)]
    if len(refs) == 1:
        return refs[0]
    # a basename that repeats across projects is only usable when the project pins it uniquely
    if project is not None:
        return None
    refs_any = [r for r in walk() if _nfc(r.name) == fname]
    return refs_any[0] if len(refs_any) == 1 else None


def _resolve_train_ref(question: str, project: "str | None"):
    """The project's train/data table for a histogram question, via canonical_route, or ``None``."""
    from src.rag.corpus import walk

    cr = importlib.import_module("src.rag.tools.canonical_route")
    try:
        routed = cr.canonical_route(question=question, project=project, kind="train")
    except Exception:  # noqa: BLE001
        return None
    value = (routed or {}).get("value") or []
    rels = [v.get("rel") for v in value if isinstance(v, dict) and v.get("rel")]
    if len(rels) != 1:
        return None
    target = _nfc(rels[0])
    refs = [r for r in walk() if _nfc(r.rel) == target]
    return refs[0] if len(refs) == 1 else None


# --------------------------------------------------------------------------- chart recognizer 1: histogram
def _read_chart_values(ref, **kwargs) -> "dict[str, Any] | None":
    """Call the strict chart reader (import_module so tests can monkeypatch it); ``None`` on any failure."""
    cn = importlib.import_module("src.rag.tools.chart_numcache")
    try:
        return cn.read_chart_values(ref, **kwargs)
    except Exception:  # noqa: BLE001 — an unsupported/ambiguous chart raises ⇒ fall back, never break path
        return None


def _histogram_max_count(question: str) -> "dict[str, Any] | None":
    """Deterministic ヒストグラム最大カウント (idx10 型), or ``None`` (⇒ fallback)."""
    q = _nfc(question)
    if not (_HIST_RE.search(q) and _MAX_COUNT_RE.search(q)):
        return None
    col_m = _HIST_COL_RE.search(q)
    if not col_m:
        return None
    column = col_m.group(1).strip("の")
    if not column:
        return None
    project = _resolve_project(question)
    ref = _find_named_file(q, project) or _resolve_train_ref(question, project)
    if ref is None:
        return None
    result = _read_chart_values(ref, column=column, operation="histogram_max_count")
    if not isinstance(result, dict):
        return None
    value_block = result.get("value")
    count = value_block.get("result") if isinstance(value_block, dict) else None
    if not isinstance(count, int):
        return None
    ev = result.get("evidence") or {}
    evidence = {
        "file": ev.get("file"),
        "column": ev.get("source_column", column),
        "member": ev.get("member"),
        "bin_width": ev.get("bin_width"),
        "n_rows": ev.get("n_rows"),
        "route": "canonical_route→read_chart_values(histogram_max_count)",
    }
    method = {
        "engine": "chart_spatial_det_pipeline",
        "contract": "chart_read",
        "shape": "histogram_max_count",
        "numeric_authority": True,
        "vision_used": False,
        "confidence": 1.0,
    }
    return {"value": count, "evidence": evidence, "method": method}


# --------------------------------------------------------------------------- chart recognizer 2: point read
def _theme_accents(ref) -> "dict[str, str]":
    """Map the office file's theme color slots (accent1..6, dk/lt) to their sRGB hex, best-effort."""
    accents: "dict[str, str]" = {}
    try:
        with zipfile.ZipFile(ref.path) as z:
            themes = [n for n in z.namelist() if re.search(r"theme/theme\d+\.xml$", n, re.I)]
            if not themes:
                return accents
            block_txt = z.read(sorted(themes)[0]).decode("utf-8", "ignore")
    except Exception:  # noqa: BLE001
        return accents
    m = re.search(r"<a:clrScheme.*?</a:clrScheme>", block_txt, re.S)
    if not m:
        return accents
    block = m.group(0)
    slots = ["accent1", "accent2", "accent3", "accent4", "accent5", "accent6",
             "dk1", "dk2", "lt1", "lt2"]
    for slot in slots:
        mm = re.search(rf"<a:{slot}>(.*?)</a:{slot}>", block, re.S)
        if not mm:
            continue
        srgb = re.search(r'srgbClr val="([0-9A-Fa-f]{6})"', mm.group(1))
        sysc = re.search(r'lastClr="([0-9A-Fa-f]{6})"', mm.group(1))
        picked = srgb or sysc
        if picked:
            accents[slot] = picked.group(1)
    return accents


# schemeClr placeholder → theme slot (tx/bg map to dk/lt like OOXML resolution).
_SCHEME_SLOT = {"tx1": "dk1", "bg1": "lt1", "tx2": "dk2", "bg2": "lt2",
                "dk1": "dk1", "lt1": "lt1", "dk2": "dk2", "lt2": "lt2"}


def _series_color_names(ref, member: str) -> "list[str | None]":
    """Coarse color name for each ``<c:ser>`` (in order) of the ``member`` chart, or ``None`` per series."""
    from src.rag.extract import office as _office

    try:
        with zipfile.ZipFile(ref.path) as z:
            data = z.read(member)
        root = ET.fromstring(data)
    except Exception:  # noqa: BLE001
        return []
    accents = _theme_accents(ref)

    def _local(tag: str) -> str:
        return tag.rsplit("}", 1)[-1]

    names: "list[str | None]" = []
    for ser in (e for e in root.iter() if _local(e.tag) == "ser"):
        hexcol: "str | None" = None
        for el in ser.iter():
            ln = _local(el.tag)
            if ln == "srgbClr":
                hexcol = el.get("val")
                break
            if ln == "schemeClr":
                slot = el.get("val") or ""
                hexcol = accents.get(slot) or accents.get(_SCHEME_SLOT.get(slot, slot))
                break
        names.append(_office._color_name(hexcol) if hexcol else None)
    return names


def _round_places(value: float, places: "int | None") -> Any:
    """Round to ``places`` decimals when the question fixes them, else return the exact plotted value."""
    if places is None:
        return value
    return round(float(value), places)


def _chart_point_read(question: str) -> "dict[str, Any] | None":
    """Deterministic 色/カテゴリ指定の系列点読み (idx33 型), or ``None`` (⇒ fallback)."""
    q = _nfc(question)
    query_color = _query_color(q)
    xcat = _XCAT_RE.search(q)
    if query_color is None or xcat is None:
        return None
    if not re.search(r"折れ線|系列|線|グラフ|値", q):
        return None
    cat_i = int(xcat.group(1)) - 1  # x=N is 1-based
    if cat_i < 0:
        return None

    project = _resolve_project(question)
    ref = _find_named_file(q, project)
    if ref is None:
        return None
    result = _read_chart_values(ref)  # numCache 厳密経路（column なし = 埋め込み画像は自動的に棄権）
    if not isinstance(result, dict):
        return None
    charts = (result.get("value") or {}).get("charts") if isinstance(result.get("value"), dict) else None
    if not isinstance(charts, list) or not charts:
        return None

    chart_m = _CHART_IDX_RE.search(q)
    if chart_m:
        idx = int(chart_m.group(1)) - 1
        if not (0 <= idx < len(charts)):
            return None
        chart = charts[idx]
    elif len(charts) == 1:
        chart = charts[0]
    else:
        return None  # multiple charts but no グラフN index ⇒ ambiguous, fall back

    series = chart.get("series") or []
    if not series:
        return None
    colors = _series_color_names(ref, chart.get("member"))
    if len(colors) != len(series):
        return None
    hits = [s for s, c in zip(series, colors) if _color_matches(c, query_color)]
    if len(hits) != 1:  # color must pin exactly one series, else fall back
        return None
    values = hits[0].get("values") or []
    if not (0 <= cat_i < len(values)):
        return None
    raw = values[cat_i]
    if not isinstance(raw, (int, float)):
        return None
    places_m = _DECIMALS_RE.search(q)
    places = int(places_m.group(1)) if places_m else None
    value = _round_places(raw, places)

    evidence = {
        "file": ref.rel,
        "chart_member": chart.get("member"),
        "series_name": hits[0].get("name"),
        "series_color": next((c for s, c in zip(series, colors) if s is hits[0]), None),
        "query_color": query_color,
        "category_index_1based": cat_i + 1,
        "exact_value": raw,
        "decimal_places": places,
        "route": "chart_numcache(numCache)→color/category point read",
    }
    method = {
        "engine": "chart_spatial_det_pipeline",
        "contract": "chart_read",
        "shape": "series_point_read",
        "numeric_authority": True,
        "vision_used": False,
        "confidence": 1.0,
    }
    return {"value": value, "evidence": evidence, "method": method}


# --------------------------------------------------------------------------- spatial recognizer: seating
def _seating_lookup(**kwargs) -> "dict[str, Any] | None":
    """Call the seating directory lookup (import_module so tests can monkeypatch it); ``None`` on failure."""
    sc = importlib.import_module("src.rag.tools.seating_chart")
    try:
        return sc.seating_lookup(**kwargs)
    except Exception:  # noqa: BLE001
        return None


def _seating_field(question_nfc: str) -> str:
    """What the seating question asks to emit: EXT (内線/EXT) vs name (名前/氏名/誰)."""
    if re.search(r"内線|ext", question_nfc, re.I):
        return "ext"
    if re.search(r"名前|氏名|誰|どなた", question_nfc):
        return "name"
    return "ext"


def _seating_answer(question: str) -> "dict[str, Any] | None":
    """Deterministic 座席の空間関係解決 (向かい/右側… idx44/idx58 型), or ``None`` (⇒ fallback).

    Shared by :func:`spatial_pipeline` and the ``full_enumeration`` composition — a broad-side relation
    (右側/左側) returns a name/EXT list (idx44), a narrow relation (向かい/隣) a single value (idx58)."""
    from src.rag.tools import seating_chart as _sc

    q = _nfc(question)
    if not _SEATING_CUE_RE.search(q):
        return None
    if _sc.canonical_relation(q) is None:  # a seating cue with no resolvable relation ⇒ not this shape
        return None
    person_m = _PERSON_RE.search(q)
    if not person_m:
        return None
    name = person_m.group(1)
    field = _seating_field(q)
    result = _seating_lookup(name=name, relation=q, field=field)
    if not isinstance(result, dict):
        return None
    value = result.get("value")
    if value is None:  # seating_lookup abstains on ambiguity / not-found ⇒ fall back (回答数を減らさない)
        return None
    src_ev = result.get("evidence") or {}
    evidence = {
        "subject": name,
        "relation": src_ev.get("relation") or _sc.canonical_relation(q),
        "field": field,
        "neighbours": src_ev.get("neighbours"),
        "neighbour": src_ev.get("neighbour"),
        "origin": src_ev.get("origin"),
        "route": "seating_directory(pixel-hash pin)",
    }
    method = {
        "engine": "chart_spatial_det_pipeline",
        "contract": "spatial",
        "shape": "seating_relation_lookup",
        "confidence": 1.0,
    }
    return {"value": value, "evidence": evidence, "method": method}


# --------------------------------------------------------------------------- dispatch + registered pipelines
def _first_grounded(question: str, recognizers) -> "dict[str, Any] | None":
    for recognizer in recognizers:
        try:
            out = recognizer(question)
        except Exception:  # noqa: BLE001 — one broken recognizer must never break the answer path
            out = None
        if out is not None and out.get("value") is not None:
            return out
    return None


def chart_read_pipeline(question: str, *, profile: Any = None) -> "dict[str, Any] | None":
    """Deterministic ``chart_read`` answer as a ``{value, evidence, method}`` contract, or ``None``."""
    try:
        return _first_grounded(question, (_histogram_max_count, _chart_point_read))
    except Exception:  # noqa: BLE001 — any failure falls back to the LLM loop, never breaks the answer path
        return None


def spatial_pipeline(question: str, *, profile: Any = None) -> "dict[str, Any] | None":
    """Deterministic ``spatial`` answer as a ``{value, evidence, method}`` contract, or ``None``."""
    try:
        return _seating_answer(question)
    except Exception:  # noqa: BLE001
        return None


# A3 (SOT-2608) の ``full_enumeration`` パイプラインを捕捉して座席列挙を前段合成する（下の register で束縛）。
_A3_FULL_ENUMERATION: "Any" = None


def _composed_full_enumeration(question: str, *, profile: Any = None) -> "dict[str, Any] | None":
    """Seating-enum first (idx44), else delegate UNCHANGED to A3's ``full_enumeration`` pipeline.

    Non-seating enumeration questions bypass :func:`_seating_answer` (the seating cue guard returns
    ``None`` immediately) and are answered byte-identically by A3 — this composition only *adds* the
    seating列挙 short-circuit, it never changes A3's existing behavior."""
    seated = None
    try:
        seated = _seating_answer(question)
    except Exception:  # noqa: BLE001
        seated = None
    if seated is not None and seated.get("value") is not None:
        return seated
    if _A3_FULL_ENUMERATION is None:
        return None
    return _A3_FULL_ENUMERATION(question, profile=profile)


def register(*, replace: bool = True) -> None:
    """Register ``chart_read`` / ``spatial`` and compose seating onto A3's ``full_enumeration`` (idempotent)."""
    _det_pipeline.register("chart_read", chart_read_pipeline, replace=replace)
    _det_pipeline.register("spatial", spatial_pipeline, replace=replace)

    # Capture A3's full_enumeration pipeline (registered when :mod:`.enumeration` imported ahead of this
    # module) and front it with the seating recognizer — WITHOUT dropping A3's logic. Best-effort: if A3 is
    # absent/unregistered we leave full_enumeration untouched rather than shadowing it with a seating-only
    # handler.
    global _A3_FULL_ENUMERATION
    try:
        from src.rag.agent.pipelines import enumeration as _enum
        candidate = getattr(_enum, "full_enumeration_pipeline", None)
        # Never wrap our own composed function (idempotent re-register): only capture A3's real pipeline.
        if callable(candidate) and candidate is not _composed_full_enumeration:
            _A3_FULL_ENUMERATION = candidate
        if _A3_FULL_ENUMERATION is not None:
            _det_pipeline.register("full_enumeration", _composed_full_enumeration, replace=True)
    except Exception:  # noqa: BLE001 — a composition failure must never break registration / the answer path
        pass


# Self-register on import — importing :mod:`src.rag.agent.pipelines` (the router's lazy bootstrap) wires
# these pipelines into the Stage0 registry. Idempotent so a re-import / test reload never raises.
register(replace=True)
