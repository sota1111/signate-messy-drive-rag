"""IDマスタ逆引き表 — 全ID体系の全出現を「原文の行内容」付きで事前計算する事実層 (SOT-2644 / 事前計算事実層 2/5).

champion (Wave A net40) の abstain 全数分類 (`docs/ai/champion_abstain_wrong_classification.md`,
2026-08-11) が確定した **hard core 16問** のうち fact_lookup 型は「特定 ID の内容をそのまま抜き出す」問で、
実行時は ``file_grep`` を反復（idx98 は11連発）したまま ``BUDGET_EXHAUSTED``。08-10 の BUDGET32 診断でも
``file_grep`` が全ツール呼びの 70%(285回) を占める最大の浪費源だった。

方針: **実行時に ID を探すのではなく、コーパス中の全 ID 出現をオフラインで逆引き表化しておく**。ID 体系
(タスクID: T09 / アクションアイテム: A10 / マイルストーン: M01 / チェックポイント: CP1 / WBS-01 / AI番号…) は
**有限で機械抽出可能** — 事前に ``{id, id_kind, doc_id, sheet/page/slide, row, 行内容全文, 隣接列}`` の逆引き
表を作れば、「ID→内容」質問は探索ゼロの O(1) 照合になる。既存の逆引き索引 (SOT-2531 typed-evidence-index)
は汎用トークン索引の部分実装なので、本 issue は **ID 体系に特化した全数版** を additive に構築する。

**正当性の歯止め（必読）**: 事前計算は **質問を見ない網羅抽出** に限定する — 全文書の全 ID 出現を正規表現＋
列ヘッダ / ラベル文脈で機械走査するのであり、特定設問を狙った選別計算・gold 値のハードコードは一切しない。
設問→カバレッジの対応表 (:func:`fact_hard_core_coverage`) は *ビルドレポート用の診断* にすぎず、表本体
(``id_master.jsonl``) は完全に質問非依存。カバレッジ外は従来経路へフォールバック（劣化ゼロ）。

Design invariants（既存の兄弟事前ストア — evidence_index / structure_store / case_master — に倣う）:
  * **全出現カバレッジ.** :func:`scan_doc` は 1 文書の抽出済みテキストから ID 出現をすべて拾い、同一 ID の
    複数出現・文書横断の同名 ID を doc_id / 行番号で区別して全保持する（取りこぼしのない逆引き）。
  * **決定論・LLM フリー.** 抽出は正規表現＋列ヘッダ / ラベル文脈の決定論分類のみ。Gemini 呼び出しなし・
    ネットワークなし・再現バイト一致（2回ビルドで byte-identical）。
  * **原文粒度＋出典必須.** 各出現に行内容全文と隣接列（xlsx はパイプ区切りセル配列）を格納し、locator
    (sheet/page/slide) と行番号を付ける —「そのまま抜き出して」に原文で応えられる粒度。
  * **Opt-in at serve time.** :func:`enabled` (``RAG_ID_MASTER``) は *実行時参照のみ* を gate する。
    アーティファクトは index 時に常に additive・guarded に（再）ビルドされる。既定 OFF ⇒ champion serve
    path はバイト一致。本 issue の読み出しツールは最小限（:func:`lookup`）— 配線の本番は別 issue。
  * **再配布安全.** 表は構造メタデータ + 原文行のみ。パスワード等の秘密値は絶対に入れない。
"""
from __future__ import annotations

import json
import os
import re
import unicodedata
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

from config import settings
from src.rag.corpus import FileRef, nfc

SCHEMA = "id-master"
SCHEMA_VERSION = 1

_ON = {"1", "true", "yes", "on"}
_LINE_MAX = 500       # 行内容全文の保存上限（1セルの巨大テキスト暴走を防ぐ）
_COLS_MAX = 40        # パイプ区切り行の保持セル数上限
_CELL_MAX = 200       # 隣接セル1個あたりの保存上限

# ビジネス文書の拡張子のみを走査対象にする。依存 lock ファイル / notebook(.ipynb) / ソース(.py) は
# ハッシュ断片・エンコーディング名・セル座標など ID 形の *データ値ノイズ* を大量に生むため除外する
# （それらは「ID体系」ではない — 08-11 実測で全体の 85% がこの由来だった）。
_DOC_EXTS = {"xlsx", "docx", "pptx", "pdf", "md", "csv"}

# office 抽出の【ハイライトされたセル】注記行 "A1(青): 値" の先頭は *セル座標* であり業務 ID ではない。
# 座標を剥がし、値部分だけを ID 走査対象にする（A1/M1 のような座標が action/milestone に誤分類されるのを防ぐ）。
_HL_COORD = re.compile(r"^(?P<coord>\s*[A-Z]+\d+\([^)]*\):\s*)")

# --------------------------------------------------------------------------- 構造マーカ（抽出テキスト）
# office/plain の抽出器が出す構造マーカ。sheet/slide/page を追跡して locator に落とす。
_SHEET = re.compile(r"^\[シート:\s*(?P<name>[^\]]+?)\]")
_SLIDE = re.compile(r"^\[スライド(?P<num>\d+)")
_PAGE = re.compile(r"^\[ページ(?P<num>\d+)\]")
_MARKER_LINE = re.compile(r"^\s*(?:\[シート:|\[スライド|\[ページ|【)")

# --------------------------------------------------------------------------- ID 抽出パターン
# `[A-Za-z]{1,4}` の英字接頭辞 + 任意の区切り(-/_) + `\d{1,4}` の連番。両端は英数字・ハイフンで囲まれない
# 単独トークンに限定する。左端の ``(?<![-0-9A-Za-z])`` は "APR-M2" のような合成コードの後半 "M2" を誤検出
# しない（"-" の直後を弾く）と同時に "WBS-01" は先頭 "W" から丸ごと1トークンとして拾う。
_ID_CAND = re.compile(r"(?<![-0-9A-Za-z])(?P<pre>[A-Za-z]{1,4})[-_]?(?P<num>\d{1,4})(?![-0-9A-Za-z])")

# 英字接頭辞 → ID 種別（多字接頭辞ほど高確度。single-letter は corpus の実 ID 体系: T09/A10/M01）。
_KIND_BY_PREFIX = {
    "T": "task", "TASK": "task",
    "A": "action", "AI": "action_item",
    "M": "milestone", "MS": "milestone",
    "CP": "checkpoint",
    "WBS": "wbs",
    "PH": "phase", "PHASE": "phase",
    "STEP": "step",
    "KPI": "kpi",
    "REQ": "requirement",
}

# 列ヘッダ / 行内ラベルの日本語文脈 → ID 種別（正規表現マッチで決定論分類）。接頭辞より優先度が高い。
_LABEL_KINDS: list[tuple["re.Pattern[str]", str]] = [
    (re.compile(r"タスク\s*(?:ID|Ｉ?Ｄ|番号|No\.?)|タスクID"), "task"),
    (re.compile(r"アクション\s*アイテム|アクション\s*(?:ID|番号)|AI\s*番号"), "action_item"),
    (re.compile(r"マイルストーン"), "milestone"),
    (re.compile(r"チェックポイント"), "checkpoint"),
    (re.compile(r"WBS"), "wbs"),
    (re.compile(r"フェーズ|工程"), "phase"),
    (re.compile(r"ステップ"), "step"),
    (re.compile(r"KPI"), "kpi"),
    (re.compile(r"要件\s*(?:ID|番号)"), "requirement"),
    (re.compile(r"リスク\s*(?:ID|番号)"), "risk"),
]


def enabled() -> bool:
    """True when the serve path should consult the ID master (default OFF — opt-in).

    Mirrors ``RAG_CASE_MASTER`` / ``RAG_EVIDENCE_INDEX``: the artifact is always (re)built at index
    time, but *runtime consultation* is gated so the champion serve path stays byte-identical.
    """
    return os.getenv("RAG_ID_MASTER", "0").strip().lower() in _ON


def default_out_path() -> Path:
    return settings.ARTIFACTS_DIR / "id_master.jsonl"


def report_out_path() -> Path:
    return settings.ARTIFACTS_DIR / "id_master_build_report.json"


# --------------------------------------------------------------------------- normalization helpers
def id_norm(pre: str, num: str) -> str:
    """Separator/leading-zero-preserving canonical key: ``M-01`` and ``M01`` → ``M01`` (upper)."""
    return f"{pre.upper()}{num}"


def _clip(text: str, limit: int) -> str:
    text = " ".join(nfc(text).split())
    return text if len(text) <= limit else text[:limit] + "…"


def _kind_from_label(text: str) -> tuple[str, str] | None:
    """(kind, matched_label) for the first ID-label keyword present in ``text``; else None."""
    for pat, kind in _LABEL_KINDS:
        m = pat.search(text)
        if m:
            return kind, m.group(0)
    return None


# --------------------------------------------------------------------------- occurrence record
@dataclass(frozen=True)
class IdOccurrence:
    id: str            # NFC raw match text as printed (e.g. "T09", "M-01")
    id_norm: str       # canonical key PRE.upper()+num, separator-insensitive (e.g. "M01")
    id_kind: str       # task / action / milestone / checkpoint / wbs / … / "unknown"
    kind_basis: str    # column_header | line_label | prefix | pattern_only
    doc_id: str        # NFC corpus-relative path
    project: str
    sheet: str = ""    # locator (spreadsheet)
    slide: int = 0     # locator (slides), 1-based; 0 = n/a
    page: int = 0      # locator (pdf), 1-based; 0 = n/a
    row: int = 0       # 1-based line number within the extracted text
    line: str = ""     # full row text (原文粒度) — capped
    columns: tuple[str, ...] = ()   # pipe-split cells for spreadsheet rows (隣接列); () otherwise
    column_index: int = 0           # 1-based index of the cell holding the id; 0 = not a table row
    column_header: str = ""         # header of that column, when recoverable
    label: str = ""                 # nearby ID label text, when present

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["columns"] = list(self.columns)
        return d


def _resolve_kind(pre: str, line: str, header: str | None) -> tuple[str, str]:
    """(id_kind, kind_basis) — header label > line label > known prefix > pattern-only (unknown)."""
    if header:
        hit = _kind_from_label(header)
        if hit:
            return hit[0], "column_header"
    hit = _kind_from_label(line)
    if hit:
        return hit[0], "line_label"
    prefix_kind = _KIND_BY_PREFIX.get(pre.upper())
    if prefix_kind:
        return prefix_kind, "prefix"
    return "unknown", "pattern_only"


def _split_columns(line: str) -> list[str] | None:
    """Pipe-split cells for a spreadsheet data row (office joins cells with ' | '); None otherwise."""
    if " | " not in line or line.lstrip().startswith("【") or _MARKER_LINE.match(line):
        return None
    return [c.strip() for c in line.split(" | ")]


def _header_columns(lines: Sequence[str], sheet_start: int, row_index: int) -> list[str] | None:
    """The header row of the current sheet block: the first pipe row after the sheet marker.

    ``sheet_start`` is the line index of the active ``[シート:]`` marker (or -1 when not in a sheet);
    ``row_index`` is the current line index (exclusive upper bound). Returns None outside a sheet /
    when no header row precedes the current row."""
    if sheet_start < 0:
        return None
    for i in range(sheet_start + 1, row_index + 1):
        cols = _split_columns(lines[i])
        if cols is not None:
            return cols
    return None


def scan_doc(ref: FileRef, text: str) -> list[IdOccurrence]:
    """Extract every ID occurrence from one already-extracted document (deterministic, no LLM/network).

    ``text`` is the extractor output (with ``[シート:]`` / ``[スライドN]`` / ``[ページN]`` markers and
    ``' | '``-joined spreadsheet rows). Each occurrence keeps the full row text and — for spreadsheet
    rows — the adjacent cells + column header, so an "そのまま抜き出して" question is answered at
    row granularity with provenance.
    """
    lines = unicodedata.normalize("NFC", text).splitlines()
    sheet = ""
    slide = 0
    page = 0
    sheet_start = -1
    out: list[IdOccurrence] = []
    for idx, raw in enumerate(lines):
        line = raw.rstrip()
        if m := _SHEET.match(line):
            sheet, slide, page, sheet_start = _clip(m.group("name"), 120), 0, 0, idx
            continue
        if m := _SLIDE.match(line):
            slide, sheet, page, sheet_start = int(m.group("num")), "", 0, -1
            continue
        if m := _PAGE.match(line):
            page, sheet, slide, sheet_start = int(m.group("num")), "", 0, -1
            continue

        # Strip a leading highlight-cell coordinate ("A1(青): …") so the coordinate is not mis-read as
        # an ID; scan only the value part. ``scan_off`` keeps char offsets aligned to the full line.
        hl = _HL_COORD.match(line)
        scan_off = len(hl.group("coord")) if hl else 0
        candidates = list(_ID_CAND.finditer(line, scan_off))
        if not candidates:
            continue
        cols = _split_columns(line)
        header_cols = _header_columns(lines, sheet_start, idx) if cols is not None else None
        # map char offset → (1-based column index) for a table row, to attribute the id to its cell.
        col_bounds: list[int] = []
        if cols is not None:
            pos = 0
            for cell in line.split(" | "):
                col_bounds.append(pos)
                pos += len(cell) + 3  # len(" | ")
        clipped_cols = tuple(_clip(c, _CELL_MAX) for c in cols[:_COLS_MAX]) if cols is not None else ()

        for cm_ in candidates:
            pre, num = cm_.group("pre"), cm_.group("num")
            column_index = 0
            column_header = ""
            if cols is not None and col_bounds:
                column_index = sum(1 for b in col_bounds if b <= cm_.start())
                if header_cols and 0 < column_index <= len(header_cols):
                    column_header = _clip(header_cols[column_index - 1], _CELL_MAX)
            # Labels in unrelated adjacent cells must not reclassify this ID (for example an M01
            # cell on a row whose description mentions KPI). Keep the full row as evidence, but
            # use only the containing cell as label context when the source is tabular.
            label_context = cols[column_index - 1] if cols is not None and column_index else line
            kind, basis = _resolve_kind(pre, label_context, column_header or None)
            label = ""
            if basis in ("column_header", "line_label"):
                label = (column_header if basis == "column_header"
                         else (_kind_from_label(label_context) or ("", ""))[1])
            out.append(IdOccurrence(
                id=nfc(cm_.group(0)), id_norm=id_norm(pre, num), id_kind=kind, kind_basis=basis,
                doc_id=nfc(ref.rel), project=nfc(ref.project), sheet=sheet, slide=slide, page=page,
                row=idx + 1, line=_clip(line, _LINE_MAX), columns=clipped_cols,
                column_index=column_index, column_header=column_header, label=_clip(label, _CELL_MAX),
            ))
    return _dedup(out)


def _dedup(occ: Iterable[IdOccurrence]) -> list[IdOccurrence]:
    """Drop exact duplicate occurrences (frozen dataclass = hashable), stable deterministic order."""
    return sorted(set(occ), key=_sort_key)


def _sort_key(o: IdOccurrence) -> tuple:
    return (o.doc_id, o.row, o.column_index, o.id_norm, o.id, o.id_kind)


# --------------------------------------------------------------------------- write / build
def write_master(occ: Sequence[IdOccurrence], path: Path | None = None) -> dict[str, int]:
    """Atomically write a reproducible JSONL table (schema header + deterministically-sorted rows)."""
    out = path or default_out_path()
    out.parent.mkdir(parents=True, exist_ok=True)
    ordered = sorted(occ, key=_sort_key)
    tmp = out.with_suffix(out.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8", newline="\n") as handle:
        handle.write(json.dumps({"schema": SCHEMA, "version": SCHEMA_VERSION},
                                ensure_ascii=False, sort_keys=True) + "\n")
        for o in ordered:
            handle.write(json.dumps(o.to_dict(), ensure_ascii=False, sort_keys=True) + "\n")
    tmp.replace(out)
    reset_cache()
    return {"occurrences": len(ordered),
            "distinct_ids": len({o.id_norm for o in ordered}),
            "files": len({o.doc_id for o in ordered})}


def build(refs: list[FileRef] | None = None, *, out: Path | None = None,
          write_report: bool = True) -> dict[str, Any]:
    """Precompute + persist the exhaustive ID reverse-lookup master (every ID occurrence with 原文).

    Deterministic and LLM-free. Returns a small build summary. Additive & guarded — a failure here
    never corrupts the retrieval index (the caller in :mod:`src.rag.index` swallows exceptions).
    """
    from src.rag import corpus
    from src.rag.extract import extract  # heavy office deps — only needed at build time

    refs = refs if refs is not None else corpus.walk()
    occ: list[IdOccurrence] = []
    files_scanned = 0
    for ref in refs:
        # ``corpus.walk`` also exposes source trees, notebooks, lockfiles, and generated images.
        # Their hashes/cell coordinates/version strings look like IDs but are not business-document
        # identifiers, so keep the master scoped to the supported document formats.
        if ref.ext.lower().lstrip(".") not in _DOC_EXTS:
            continue
        try:
            doc = extract(ref, caption_images=False)  # no vision/network for a text ID scan
        except Exception:  # noqa: BLE001 — one unreadable doc must not sink the build
            continue
        text = getattr(doc, "text", "") or ""
        if not text.strip():
            continue
        files_scanned += 1
        try:
            occ.extend(scan_doc(ref, text))
        except Exception:  # noqa: BLE001
            continue

    counts = write_master(occ, out)
    report = build_report(occ, files_scanned, out or default_out_path())
    if write_report:
        rp = report_out_path()
        rp.parent.mkdir(parents=True, exist_ok=True)
        rp.write_text(json.dumps(report, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
                      encoding="utf-8")
    return {**counts, "report": report}


def build_report(occ: Sequence[IdOccurrence], files_scanned: int, out_path: Path) -> dict[str, Any]:
    """Diagnostic build report: 種別×件数 / 分類根拠分布 / 重複ID率 / 抽出元分布 / hard-core カバレッジ."""
    by_kind: dict[str, int] = {}
    by_basis: dict[str, int] = {}
    by_ext: dict[str, int] = {}
    docs_by_id: dict[str, set[str]] = {}
    for o in occ:
        by_kind[o.id_kind] = by_kind.get(o.id_kind, 0) + 1
        by_basis[o.kind_basis] = by_basis.get(o.kind_basis, 0) + 1
        ext = o.doc_id.rsplit(".", 1)[-1].lower() if "." in o.doc_id else ""
        by_ext[ext] = by_ext.get(ext, 0) + 1
        docs_by_id.setdefault(o.id_norm, set()).add(o.doc_id)
    distinct = len(docs_by_id) or 1
    cross_doc = sum(1 for docs in docs_by_id.values() if len(docs) > 1)
    return {
        "certificate": {
            "schema": SCHEMA, "schema_version": SCHEMA_VERSION,
            "question_independent": True,
            "extraction_basis": ("全文書の全ID出現を機械走査（正規表現 [A-Za-z]{1,4}-?\\d{1,4} ＋ "
                                 "列ヘッダ/ラベル文脈の決定論分類）。gold値ハードコードなし。"),
            "files_scanned": files_scanned,
        },
        "occurrence_count": len(occ),
        "distinct_ids": len(docs_by_id),
        "cross_doc_duplicate_ids": cross_doc,
        "duplicate_id_rate": round(cross_doc / distinct, 4),
        "by_kind": dict(sorted(by_kind.items())),
        "by_kind_basis": dict(sorted(by_basis.items())),
        "by_source_ext": dict(sorted(by_ext.items())),
        "fact_hard_core_coverage": fact_hard_core_coverage(occ),
        "out_path": str(out_path),
    }


# --------------------------------------------------------------------------- fact hard-core coverage
# 設問 → カバレッジ診断（診断専用・表本体には非関与・gold値非依存）。champion abstain の fact_lookup 3問
# (idx39/48/98) を対象に、その設問が参照する **文書に本マスタが ID を抽出できたか** を honest に報告する。
# 3問は実測すると「ID→内容」型ではない（グラフ列 / PDF税率 / RATE変更日）ため、本マスタが直接の solver に
# なるとは限らない — その事実（別レーンで解く領域）も明示する。これは配線判断の前提情報にすぎない。
_FACT_HARD_CORE: dict[str, dict[str, Any]] = {
    "idx39": {
        "gist": "青潮モビリティサービス train.xlsx Sheet1 のグラフ1が可視化したカラム",
        "doc_hint": "青潮モビリティ", "ext_hint": "xlsx",
        "note": ("グラフ→元カラムはID体系ではなく chart numCache / structure_store 領域。"
                 "本マスタは同文書内の ID 出現の有無のみを診断報告する。"),
    },
    "idx48": {
        "gist": "青嶺不動産 ニューヨーク不動産市場の最新動向調査.pdf のマンション税 新税率の価格帯",
        "doc_hint": "青嶺", "ext_hint": "pdf",
        "note": ("価格帯×税率は数表であり [A-Za-z]-\\d の ID 体系ではない。PDF 本文読解領域。"
                 "本マスタは同文書内の ID 出現の有無のみを診断報告する。"),
    },
    "idx98": {
        "gist": "TM案件で RATE が変更された想定年月（何年何月1日から）",
        "doc_hint": "", "ext_hint": "",
        "note": ("RATE は数字を持たない param 名（evidence_index param 型領域）で本マスタの ID パターン外。"
                 "TM=time_and_materials 契約は case_master(contract_type) の領域。"),
    },
}


def fact_hard_core_coverage(occ: Sequence[IdOccurrence]) -> dict[str, Any]:
    """設問別に「参照文書内で本マスタが抽出した ID 数」を honest 報告する診断表（gold 非依存）。

    ``doc_hint`` を doc_id に部分一致させ、その文書群から抽出できた ID 出現数・distinct ID を数える。
    件数が 0 でも欠測を偽装せず 0 と note を返す（本マスタが solver でない領域はその旨を明示）。
    """
    out: dict[str, Any] = {}
    for idx, spec in _FACT_HARD_CORE.items():
        hint = spec.get("doc_hint") or ""
        matched = [o for o in occ if hint and hint in o.doc_id] if hint else []
        docs = sorted({o.doc_id for o in matched})
        out[idx] = {
            "gist": spec["gist"],
            "doc_hint": hint,
            "matched_docs": docs,
            "ids_in_matched_docs": len(matched),
            "distinct_ids_in_matched_docs": len({o.id_norm for o in matched}),
            "id_lookup_shaped": False,   # 実測: 3問はいずれも ID→内容 の逆引き型ではない
            "note": spec["note"],
        }
    return out


# --------------------------------------------------------------------------- load / read (minimal API)
_LOAD_CACHE: dict[str, list[dict[str, Any]]] = {}


def reset_cache() -> None:
    _LOAD_CACHE.clear()


def load(path: Path | None = None) -> list[dict[str, Any]]:
    """Load the ID master rows (memoized). ``[]`` when absent/unreadable/schema-mismatch (回帰ゼロ)."""
    out = path or default_out_path()
    key = str(out)
    if key in _LOAD_CACHE:
        return _LOAD_CACHE[key]
    rows: list[dict[str, Any]] = []
    try:
        with open(out, encoding="utf-8") as handle:
            header = handle.readline()
            meta = json.loads(header) if header.strip() else {}
            if isinstance(meta, dict) and meta.get("schema") == SCHEMA \
                    and meta.get("version") == SCHEMA_VERSION:
                for line in handle:
                    line = line.strip()
                    if line:
                        rows.append(json.loads(line))
    except Exception:
        rows = []
    _LOAD_CACHE[key] = rows
    return rows


def lookup(id_value: str, *, id_kind: str | None = None, project: str | None = None,
           path: Path | None = None) -> list[dict[str, Any]]:
    """Minimal read API (本 issue 最小限・配線は別 issue): all occurrences of an ID (separator-insensitive).

    Matches on the canonical ``id_norm`` so ``lookup("M-01")`` and ``lookup("m01")`` are equivalent.
    Unknown ID / no artifact ⇒ ``[]`` (no degradation). Optional ``id_kind`` / ``project`` filters.
    """
    m = _ID_CAND.fullmatch(nfc(id_value).strip())
    key = id_norm(m.group("pre"), m.group("num")) if m else nfc(id_value).strip().upper().replace("-", "").replace("_", "")
    proj = nfc(project) if project else None
    rows = load(path)
    out: list[dict[str, Any]] = []
    for row in rows:
        if row.get("id_norm") != key:
            continue
        if id_kind is not None and row.get("id_kind") != id_kind:
            continue
        if proj is not None and nfc(row.get("project", "")) != proj:
            continue
        out.append(row)
    return out


if __name__ == "__main__":
    summary = build()
    print(json.dumps({**{k: v for k, v in summary.items() if k != "report"},
                      "certificate": summary["report"]["certificate"]},
                     ensure_ascii=False, indent=2))
