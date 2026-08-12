"""Full-corpus lexical index (SQLite FTS5 + IDF) — SOT-2657 / Cerebras 型検索基盤 1/5.

Why this exists (誤解防止のため必読)
-----------------------------------
BUDGET32 診断で ``file_grep`` の**都度全走査**が全ツール呼びの ~70% を占め、「ヒットは返るが
針が無い → パターンを変えて再 grep」の spin が棄権(BUDGET_EXHAUSTED)の主犯だった。Cerebras の
enterprise knowledge base 設計の中核教訓 ―― 生テキストは**着地した瞬間から全文インデックスで字句
検索可能**にし「リテラル(エラー文字列・フラグ名・ID)は字句一致が最良の証拠」とする ―― の移植第1弾。

このモジュールは *ビルド半分* と *低レベル検索* を持つ(``evidence_index`` と同じ二層構成):

* **ビルド(質問非依存・全量転記)**: docx 段落 / pptx テキスト・表 / xlsx 全セル(シート・座標付き)/
  PDF テキスト / OCR ストア転記文 / ipynb セル ―― を ``extract`` が付与する構造マーカ
  (``[ページN]`` / ``[シート: …]`` / ``[スライドN]`` / ``A1(色): 値``)を辿りながら**行単位**で
  ``{doc_id, locator, snippet, tokens}`` の行として FTS5 へ全量転記する。加えてコーパス全体の
  **トークン IDF 表**を構築する(希少トークン= ID・フラグ・固有語のランクを優先。Cerebras の「短い
  相槌が embedding 空間で勝ってしまう」問題の字句版対策)。

* **検索(``search``)**: クエリを同じ規則でトークン化 → FTS5 で候補行を絞り込み → **IDF 重み付き**で
  再ランク → 上位 N 件を ``{doc_id, locator, text, score}`` で返す。ミリ秒応答・LLM 関与ゼロ。
  ツール層 :mod:`src.rag.tools.text_search` が contract 形へ包んで投影する。

Design invariants(兄弟プリストア distill_store / evidence_index / ocr_store と同一)
  * **Opt-in at serve time** — :func:`enabled` (``RAG_TEXT_FTS``) が *実行時参照* のみを門番。既定 OFF
    ⇒ champion serve は byte-identical(既存ストア・file_grep は不変更で共存)。
  * **Opt-in at build time** — :func:`build_enabled` (``RAG_TEXT_FTS_BUILD``) が索引再構築を門番。
    OFF なら既存の成果物を温存(ビルドを壊さない加算ステップ)。
  * **決定論的** — 入力は sorted、成果物は content fingerprint を meta に持ち、2回ビルドで一致。
  * **Fail-open** — 成果物欠損 / 破損 / スキーマ不一致は ``[]`` を返し、file_grep 全走査へ落ちる。

Rebuild
-------
``.venv/bin/python -m src.rag.index`` (フル)または ``.venv/bin/python -m src.rag.index.text_fts``
(この索引のみ)。``RAG_TEXT_FTS_BUILD`` が truthy のときだけ実際に(再)構築する。
"""
from __future__ import annotations

import functools
import hashlib
import json
import math
import os
import re
import sqlite3
import unicodedata
from pathlib import Path
from typing import Any, Iterable, Sequence

from config import settings
from src.rag.corpus import FileRef, nfc, walk

SCHEMA = "text-fts"
SCHEMA_VERSION = 1

_ON = {"1", "true", "yes", "on"}
_SNIPPET_MAX = 300
# Candidate rows pulled from FTS5 before Python-side IDF re-ranking (bounds worst-case cost).
_CAND_POOL = 4000

# --- structural markers emitted by the extract layer (mirror file_grep._scan_text) ---
_PAGE_RE = re.compile(r"^\[ページ\s*(?P<n>\d+)\]")
_SHEET_RE = re.compile(r"^\[シート:\s*(?P<name>.*?)\]")
_SLIDE_RE = re.compile(r"^\[スライド\s*(?P<n>\d+)\]")
_CELL_RE = re.compile(r"^\s*(?P<cell>[A-Z]{1,3}\d+)\s*[(:]")
# A line that is *only* a structural marker carries no searchable content of its own.
_MARKER_ONLY_RE = re.compile(r"^\s*(?:\[ページ\s*\d+\]|\[シート:.*\]|\[スライド\s*\d+\])\s*$")

# --- deterministic tokenization (mirrors evidence_index.query_tokens semantics) ---
# ASCII/alnum run keeps internal _ . - so flag names / dotted ids / EXT-1234 stay one token.
_ASCII_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.\-]*")
# A contiguous CJK/kana run — indexed as its full form plus 2-char shingles for substring recall.
_JP_RE = re.compile(r"[぀-ヿ㐀-鿿豈-﫿々〆ヶー]+")


def tokenize(text: str) -> list[str]:
    """Deterministically split text into lexical tokens (NFC, casefolded).

    * ASCII/alnum words (≥2 chars, internal ``_.-`` kept) — exact anchors for literals/IDs/flags.
    * CJK/kana runs → the full run (≥2) **and** its 2-char shingles, so a query substring like
      "内線番号" hits a line containing it even though Japanese has no word spacing. Single-char
      runs are kept as-is. Casefolding is harmless for Japanese and friendly for ASCII.
    """
    s = nfc(text)
    toks: list[str] = []
    for m in _ASCII_RE.finditer(s):
        t = m.group().casefold().strip("._-")
        if len(t) >= 2:
            toks.append(t)
    for m in _JP_RE.finditer(s):
        run = m.group()
        if len(run) == 1:
            toks.append(run)
            continue
        toks.append(run)
        for i in range(len(run) - 1):
            toks.append(run[i:i + 2])
    return toks


def default_out_path() -> Path:
    return settings.ARTIFACTS_DIR / "text_fts.db"


def default_report_path() -> Path:
    return settings.ARTIFACTS_DIR / "text_fts_build_report.json"


def enabled() -> bool:
    """True when the serve path may consult the lexical index (default OFF — opt-in)."""
    return os.getenv("RAG_TEXT_FTS", "0").strip().lower() in _ON


def build_enabled() -> bool:
    """True when the index build may (re)build this store (default OFF — opt-in)."""
    return os.getenv("RAG_TEXT_FTS_BUILD", "0").strip().lower() in _ON


# --------------------------------------------------------------------------- build

def _where(cell: str | None, page: int | None, sheet: str | None,
           slide: int | None, lineno: int) -> str:
    """Most-specific available locator for a line (cell > page/sheet/slide > line).

    Byte-for-byte the same locator vocabulary file_grep returns, so a text_search hit and a
    file_grep hit name the same place identically.
    """
    if cell:
        return f"cell:{cell}" + (f"@sheet:{sheet}" if sheet else "")
    if page is not None:
        return f"page:{page}"
    if sheet is not None:
        return f"sheet:{sheet}"
    if slide is not None:
        return f"slide:{slide}"
    return f"line:{lineno}"


def _rows_for_doc(ref: FileRef, text: str) -> list[tuple[str, str, str, str, str]]:
    """Transcribe one extracted document into ``(doc_id, project, ext, locator, snippet)`` rows.

    One row per non-empty content line, carrying the nearest structural marker as its locator —
    the same line-scan file_grep uses, so the whole corpus text is projected losslessly (テキスト層は
    射影損失が最小 ⇒ 全量転記してよい) into searchable, precisely-located units.
    """
    doc_id = nfc(ref.rel) or nfc(ref.name)
    project = nfc(ref.project)
    rows: list[tuple[str, str, str, str, str]] = []
    page: int | None = None
    sheet: str | None = None
    slide: int | None = None
    for lineno, raw in enumerate(unicodedata.normalize("NFC", text).split("\n"), 1):
        line = raw.strip()
        m = _PAGE_RE.match(line)
        if m:
            page, sheet, slide = int(m["n"]), None, None
        elif (m := _SHEET_RE.match(line)):
            sheet, page, slide = m["name"].strip(), None, None
        elif (m := _SLIDE_RE.match(line)):
            slide, page, sheet = int(m["n"]), None, None
        if not line or _MARKER_ONLY_RE.match(line):
            continue
        cm = _CELL_RE.match(line)
        cell = cm["cell"] if cm else None
        snippet = line if len(line) <= _SNIPPET_MAX else line[:_SNIPPET_MAX] + "…"
        rows.append((doc_id, project, ref.ext, _where(cell, page, sheet, slide, lineno), snippet))
    return rows


def _idf_table(rows: Sequence[tuple[str, str, str, str, str]]) -> dict[str, tuple[int, float]]:
    """Corpus-wide token → (df, idf). df = #rows containing the token (a row = a located line).

    Smoothed IDF ``log((N+1)/(df+1)) + 1`` keeps every weight positive while making rare tokens
    (IDs / flag names / 固有名) outrank ubiquitous ones — the字句版 of Cerebras' "short filler wins
    in embedding space" fix. Deterministic given the row set.
    """
    n = len(rows)
    df: dict[str, int] = {}
    for _doc, _proj, _ext, _loc, snippet in rows:
        for tok in set(tokenize(snippet)):
            df[tok] = df.get(tok, 0) + 1
    return {tok: (d, math.log((n + 1) / (d + 1)) + 1.0) for tok, d in df.items()}


def _fingerprint(rows: Sequence[tuple[str, str, str, str, str]],
                 idf: dict[str, tuple[int, float]]) -> str:
    """Content hash over the canonical (sorted) rows + idf table — the determinism witness.

    SQLite file bytes are not guaranteed stable across builds, so determinism (受け入れ条件: 2回で
    一致) is asserted on this logical fingerprint rather than a raw byte diff.
    """
    h = hashlib.sha256()
    for row in sorted(rows):
        h.update(("\x1f".join(row) + "\x1e").encode("utf-8"))
    h.update(b"\x1d")
    for tok in sorted(idf):
        d, w = idf[tok]
        h.update(f"{tok}\x1f{d}\x1f{w:.6f}\x1e".encode("utf-8"))
    return h.hexdigest()


def build(refs: Sequence[FileRef] | None = None, out_path: Path | None = None,
          *, verbose: bool = False) -> dict[str, Any]:
    """Transcribe all corpus text into a SQLite FTS5 lexical index and persist it.

    Question-independent full-量 transcription: every content line of every document becomes one
    located, tokenized FTS5 row, plus a corpus-wide IDF table. Deterministic (sorted inserts +
    fingerprint). Per-file extraction errors are skipped, never fatal (加算ステップ). Returns
    ``{"records", "docs", "report", "out"}``.
    """
    from src.rag.extract import extract

    refs = list(refs) if refs is not None else list(walk())
    out = Path(out_path) if out_path is not None else default_out_path()
    out.parent.mkdir(parents=True, exist_ok=True)

    rows: list[tuple[str, str, str, str, str]] = []
    docs = 0
    skipped: list[dict[str, str]] = []
    for ref in sorted(refs, key=lambda r: (nfc(r.rel), nfc(r.name))):
        if ref.ext == "png":
            continue
        try:
            doc = extract(ref, caption_images=False)
        except Exception as e:  # one bad file must not sink the store
            skipped.append({"rel": nfc(ref.rel), "reason": f"extract {type(e).__name__}: {e}"})
            continue
        if getattr(doc, "error", None) or not (doc.text or "").strip():
            if getattr(doc, "error", None):
                skipped.append({"rel": nfc(ref.rel), "reason": str(doc.error)})
            continue
        doc_rows = _rows_for_doc(ref, doc.text)
        if doc_rows:
            rows.extend(doc_rows)
            docs += 1

    rows.sort()
    idf = _idf_table(rows)
    fingerprint = _fingerprint(rows, idf)

    tmp = out.with_suffix(out.suffix + ".tmp")
    if tmp.exists():
        tmp.unlink()
    conn = sqlite3.connect(str(tmp))
    try:
        conn.execute("PRAGMA journal_mode=MEMORY")
        conn.execute("PRAGMA synchronous=OFF")
        conn.execute(
            "CREATE VIRTUAL TABLE text_rows USING fts5("
            "doc_id UNINDEXED, project UNINDEXED, ext UNINDEXED, "
            "locator UNINDEXED, snippet UNINDEXED, tokens, "
            "tokenize=\"unicode61 tokenchars '_.-'\")")
        conn.execute("CREATE TABLE idf(token TEXT PRIMARY KEY, df INTEGER NOT NULL, idf REAL NOT NULL)")
        conn.execute("CREATE TABLE meta(key TEXT PRIMARY KEY, value TEXT)")
        conn.executemany(
            "INSERT INTO text_rows(doc_id, project, ext, locator, snippet, tokens) "
            "VALUES (?,?,?,?,?,?)",
            [(d, p, e, loc, snip, " ".join(tokenize(snip))) for d, p, e, loc, snip in rows])
        conn.executemany(
            "INSERT INTO idf(token, df, idf) VALUES (?,?,?)",
            [(tok, idf[tok][0], idf[tok][1]) for tok in sorted(idf)])
        meta = {
            "schema": SCHEMA,
            "version": str(SCHEMA_VERSION),
            "rows": str(len(rows)),
            "docs": str(docs),
            "tokens_distinct": str(len(idf)),
            "fingerprint": fingerprint,
        }
        conn.executemany("INSERT INTO meta(key, value) VALUES (?,?)", sorted(meta.items()))
        conn.commit()
    finally:
        conn.close()
    tmp.replace(out)

    report = {
        "records": len(rows),
        "docs": docs,
        "tokens_distinct": len(idf),
        "skipped": len(skipped),
        "fingerprint": fingerprint,
        "skipped_detail": skipped[:50],
    }
    default_report_path().parent.mkdir(parents=True, exist_ok=True)
    default_report_path().write_text(
        json.dumps(report, ensure_ascii=False, indent=1), encoding="utf-8")
    reset_cache()
    if verbose:
        print(f"  text fts: {len(rows)} rows over {docs} docs "
              f"(tokens={len(idf)}, skipped={len(skipped)}) -> {out}")
    return {"records": len(rows), "docs": docs, "report": report, "out": str(out)}


# --------------------------------------------------------------------------- serve read / search

_CONN_CACHE: dict[str, sqlite3.Connection | None] = {}
_IDF_CACHE: dict[str, dict[str, float]] = {}


def reset_cache() -> None:
    """Drop the in-process connection/idf cache (used by the build path and tests)."""
    for conn in _CONN_CACHE.values():
        try:
            if conn is not None:
                conn.close()
        except Exception:
            pass
    _CONN_CACHE.clear()
    _IDF_CACHE.clear()


def _connect(path: Path) -> sqlite3.Connection | None:
    """Open a read-only-ish connection, validating the schema header. ``None`` on any problem."""
    key = str(path)
    if key in _CONN_CACHE:
        return _CONN_CACHE[key]
    conn: sqlite3.Connection | None = None
    try:
        if path.exists():
            conn = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
            row = conn.execute("SELECT value FROM meta WHERE key='schema'").fetchone()
            ver = conn.execute("SELECT value FROM meta WHERE key='version'").fetchone()
            if not row or row[0] != SCHEMA or not ver or ver[0] != str(SCHEMA_VERSION):
                conn.close()
                conn = None
    except Exception:
        conn = None
    _CONN_CACHE[key] = conn
    return conn


def _idf_map(conn: sqlite3.Connection, path: Path) -> dict[str, float]:
    key = str(path)
    if key in _IDF_CACHE:
        return _IDF_CACHE[key]
    out: dict[str, float] = {}
    try:
        for tok, w in conn.execute("SELECT token, idf FROM idf"):
            out[tok] = w
    except Exception:
        out = {}
    _IDF_CACHE[key] = out
    return out


def idf_vocab(path: Path | None = None) -> dict[str, float]:
    """Corpus token → IDF weight (the whole vocabulary), for deterministic query-vocabulary bridging.

    Returns ``{}`` when disabled/absent/unreadable so callers (e.g. :mod:`src.rag.tools.query_distill`'s
    corpus-vocabulary retry) fail-open to no substitution. A copy is returned so callers cannot mutate the
    cached map.
    """
    if not enabled():
        return {}
    out = path or default_out_path()
    conn = _connect(out)
    if conn is None:
        return {}
    return dict(_idf_map(conn, out))


def _match_expr(qtokens: Sequence[str]) -> str:
    """FTS5 MATCH expression: OR of the query tokens, each double-quoted (phrase-literal)."""
    return " OR ".join('"' + t.replace('"', '""') + '"' for t in qtokens)


def search(query: str, *, limit: int = 20, project: str | None = None,
           ext: str | Iterable[str] | None = None,
           path: Path | None = None) -> list[dict[str, Any]]:
    """Return up to ``limit`` IDF-ranked hits ``{doc_id, locator, text, score, project, ext}``.

    Tokenizes ``query`` with the same rules used at build time, narrows to candidate rows via FTS5
    MATCH (millisecond, no extraction, no LLM), then re-ranks each candidate by the summed IDF of the
    distinct query tokens it carries — so a rare literal (ID / flag / 固有名) dominates a generic
    substring. Returns ``[]`` when disabled, absent, or on any miss so the caller falls back to
    file_grep (index is a fast path, never a truncation of truth).
    """
    if not enabled():
        return []
    out = path or default_out_path()
    conn = _connect(out)
    if conn is None:
        return []
    qtokens = list(dict.fromkeys(tokenize(query)))
    if not qtokens:
        return []
    idf = _idf_map(conn, out)
    proj = nfc(project) if project else None
    items = [ext] if isinstance(ext, str) else list(ext or [])
    exts = {e.lower().lstrip(".") for e in items if e} or None

    try:
        cur = conn.execute(
            "SELECT doc_id, project, ext, locator, snippet, tokens "
            "FROM text_rows WHERE text_rows MATCH ? ORDER BY bm25(text_rows) LIMIT ?",
            (_match_expr(qtokens), _CAND_POOL))
        candidates = cur.fetchall()
    except Exception:
        return []

    qset = set(qtokens)
    scored: list[tuple[float, int, str, dict[str, Any]]] = []
    for rank, (doc_id, rproj, rext, locator, snippet, tokens) in enumerate(candidates):
        if proj is not None and nfc(rproj or "") != proj:
            continue
        if exts is not None and (rext or "") not in exts:
            continue
        present = qset & set((tokens or "").split())
        if not present:
            continue
        # Summed IDF of matched query tokens = rare/specific matches dominate; coverage of the
        # distinct query tokens breaks near-ties toward the more complete match.
        score = sum(idf.get(t, 1.0) for t in present)
        coverage = len(present) / len(qset)
        scored.append((round(score, 6), rank, f"{doc_id}\x1f{locator}", {
            "doc_id": doc_id,
            "project": rproj or "",
            "ext": rext or "",
            "locator": locator,
            "text": snippet,
            "score": round(score, 4),
            "coverage": round(coverage, 3),
        }))
    # Deterministic order: score desc, then original bm25 rank, then stable (doc_id,locator) key.
    scored.sort(key=lambda s: (-s[0], s[1], s[2]))
    return [item for _s, _r, _k, item in scored[:max(1, limit)]]


if __name__ == "__main__":
    if not build_enabled():
        print("text_fts: build skipped (RAG_TEXT_FTS_BUILD off). Set RAG_TEXT_FTS_BUILD=1 to build.")
    else:
        print(json.dumps(build(verbose=True)["report"], ensure_ascii=False, indent=1))
