"""Build-time document distillation store (SOT-2658 / Cerebras 型検索基盤 2/5).

Cerebras の知識ベース（cerebras.ai/blog/how-we-built-our-knowledge-base）は Slack スレッドを**生のまま
embedding せず**、取り込み時に LLM で正規化文書（検索されそうな質問形・要約・登場システム/コード参照）へ
**蒸留**してから索引し「正規化で精度が大幅に向上した」と報告している（生テキストは別途全文検索用に保持 =
二重表現）。本モジュールはその教訓の当プロジェクトへの移植である。

当プロジェクトの制約（[[signate-model-policy-flash36-sonnet-dev]]）: 回答（serve）時は Gemini 禁止だが
**前処理（build）に限り Gemini 使用可**。そこで build 時に一度だけ Gemini で全文書（xlsx はシート単位）を
蒸留し、決定論ストア ``artifacts/distill_store.jsonl`` へ焼き込む。serve 側は焼き込まれた「この文書は何に
答えられるか」の意味情報を **LLM フリー・Gemini フリー**で字句照合するだけ（registry synopsis tier の強化 /
将来 SOT-2657 の全文検索 FTS への二重表現ソース）。

Design invariants (sibling pre-stores — ocr_store / case_master / diff_store と同型):

* **Opt-in at serve time.** :func:`enabled` (``RAG_DISTILL_STORE``) が runtime 参照を gate（既定 OFF ⇒
  champion serve path は byte-identical）。
* **Opt-in at build time too.** ocr_store 同様この build は Gemini を*消費*するので ``RAG_DISTILL_STORE_BUILD``
  (既定 OFF) でしか走らず、**content hash キャッシュ**で不変ユニットは再蒸留しない（Gemini 課金は初回のみ）。
* **Question-independent.** 蒸留プロンプトに gold100 の質問文を入れない — 文書本文だけから生成する。
* **Hallucination guard.** 蒸留 key_facts / mentioned_* の各値が**原文に存在する**ことを機械照合し、存在しない
  値は破棄して ``dropped`` に reason を残す（偽の事実は索引に載せない）。
* **Fail-open.** 不在/破損アーティファクト・未知 doc は既存挙動へフォールスルー（空を返す）。
"""
from __future__ import annotations

import functools
import hashlib
import json
import os
import re
import unicodedata
from pathlib import Path
from typing import Any, Callable, Iterable, Sequence

from config import settings
from src.rag.corpus import FileRef, nfc, walk

SCHEMA = "distill-store"
SCHEMA_VERSION = 1

_ON = {"1", "true", "yes", "on"}

# One distiller call produces at most this many items per list field (schema hint + guard cap).
MAX_QUESTIONS = 8
MAX_FACTS = 12
MAX_MENTIONS = 20

# A distilled unit whose source body is below this many chars is skipped (empty sheet / stub file).
MIN_BODY_CHARS = 40


def enabled() -> bool:
    """True when the serve path may consult the distilled store (default OFF — opt-in)."""
    return os.getenv("RAG_DISTILL_STORE", "0").strip().lower() in _ON


def build_enabled() -> bool:
    """True when the index build may spend Gemini calls to (re)build the store (default OFF)."""
    return os.getenv("RAG_DISTILL_STORE_BUILD", "0").strip().lower() in _ON


def default_out_path() -> Path:
    return settings.ARTIFACTS_DIR / "distill_store.jsonl"


def default_report_path() -> Path:
    return settings.ARTIFACTS_DIR / "distill_store_build_report.json"


# --------------------------------------------------------------------------- source unit slicing
_SHEET_MARKER = re.compile(r"^\[シート: (?P<name>.*?)\]", re.M)


def _units(kind: str, text: str) -> list[tuple[str, str, str]]:
    """Slice an extracted document into distillation units → list of (unit, unit_key, unit_text).

    * xlsx/xlsm → one ``sheet`` unit per worksheet (split on office.extract_xlsx's ``[シート: …]``
      markers). A workbook with no marker degrades to a single ``doc`` unit.
    * everything else → a single ``doc`` unit over the whole extracted text.

    Page-level (pdf) units are intentionally out of scope for v1 — the schema keeps ``page`` open for
    a later refinement; here pdf/docx/pptx/… are distilled whole (``doc``).
    """
    text = text or ""
    if kind in ("xlsx", "xlsm"):
        matches = list(_SHEET_MARKER.finditer(text))
        if matches:
            units: list[tuple[str, str, str]] = []
            for i, m in enumerate(matches):
                start = m.start()
                end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
                name = nfc(m.group("name").strip())
                units.append(("sheet", name, text[start:end].strip()))
            return units
    return [("doc", "", text.strip())]


def content_hash(text: str) -> str:
    """Stable content key for a unit's source text (NFC-normalized) — the re-build cache key."""
    return hashlib.sha256(nfc(text or "").encode("utf-8")).hexdigest()


# ----------------------------------------------------------------------------- Gemini distillation
DISTILL_SCHEMA = {
    "type": "object",
    "properties": {
        "answerable_questions": {"type": "array", "items": {"type": "string"}},
        "summary": {"type": "string"},
        "key_facts": {"type": "array", "items": {"type": "string"}},
        "mentioned_ids": {"type": "array", "items": {"type": "string"}},
        "mentioned_people": {"type": "array", "items": {"type": "string"}},
        "mentioned_systems": {"type": "array", "items": {"type": "string"}},
    },
    "required": ["answerable_questions", "summary", "key_facts"],
}

_DISTILL_SYSTEM = (
    "あなたは社内ドライブの1文書(またはExcelの1シート)を検索用に正規化するアシスタントです。"
    "与えられた本文だけを根拠に、次を日本語のJSONで返してください。外部知識や推測で補完しないこと。\n"
    "- answerable_questions: この文書が答えられる具体的な質問形(検索されそうな言い回し)を最大"
    f"{MAX_QUESTIONS}件。\n"
    "- summary: 1〜3文の要約。\n"
    f"- key_facts: 本文に明記された事実値(数値/日付/固有名/条件)を最大{MAX_FACTS}件。**本文に文字列として"
    "出現する値のみ**。言い換えず原文の表記を保つ。\n"
    "- mentioned_ids: 本文中のID/コード(タスクID/WBS/AI番号/契約番号など)。\n"
    "- mentioned_people: 本文中の人名。\n"
    "- mentioned_systems: 本文中のシステム/ツール/成果物名。\n"
    "質問文は一切与えられません。文書本文だけから生成してください。"
)


def _gemini_distiller(unit_text: str) -> dict[str, Any]:
    """Distill one unit's text via Gemini (build-only). Raises on hard failure (caller records it)."""
    from src.rag import llm

    raw = llm.generate(
        f"本文:\n{unit_text[:20000]}",
        system=_DISTILL_SYSTEM,
        model=llm.settings.GEN_MODEL,
        temperature=0.0,
        thinking_budget=0,
        max_output_tokens=2048,
        response_schema=DISTILL_SCHEMA,
    )
    obj = json.loads(raw)
    if not isinstance(obj, dict):
        raise ValueError("distiller returned non-object")
    return obj


# --------------------------------------------------------------------------- hallucination guard
_WS_PUNCT = re.compile(r"[\s,、，　]")


def _fold(s: str) -> str:
    """Fold a value for substring containment: NFKC, lowercase, drop whitespace & thousands separators."""
    return _WS_PUNCT.sub("", unicodedata.normalize("NFKC", s or "").lower())


def _supported(value: str, folded_source: str) -> bool:
    """True when ``value`` appears (folded) in the source — the anti-hallucination containment test."""
    v = _fold(value)
    return bool(v) and v in folded_source


def _clean_list(items: Any, cap: int) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()
    for it in items if isinstance(items, (list, tuple)) else []:
        s = nfc(str(it)).strip()
        if s and s not in seen:
            seen.add(s)
            out.append(s)
        if len(out) >= cap:
            break
    return out


def sanitize(obj: dict[str, Any], source_text: str) -> tuple[dict[str, Any], list[dict[str, str]]]:
    """Validate a raw distillation against the source (anti-hallucination) → (record_fields, dropped).

    * ``key_facts`` / ``mentioned_ids`` / ``mentioned_people`` / ``mentioned_systems`` — every value must
      appear (folded) in the source text; unsupported values are dropped and recorded with a reason.
    * ``answerable_questions`` / ``summary`` are paraphrases by design (they intentionally use the
      searcher's vocabulary, not the document's — the Cerebras "paraphrase bridge"), so they are NOT
      containment-checked; they are only length-capped and de-duplicated.
    """
    folded = _fold(source_text)
    dropped: list[dict[str, str]] = []

    def _checked(field: str, cap: int) -> list[str]:
        kept: list[str] = []
        for v in _clean_list(obj.get(field), cap * 3):  # over-collect, then filter to cap
            if _supported(v, folded):
                if v not in kept:
                    kept.append(v)
                if len(kept) >= cap:
                    break
            else:
                dropped.append({"field": field, "value": v, "reason": "not-in-source"})
        return kept

    summary = nfc(str(obj.get("summary") or "")).strip()[:600]
    record = {
        "answerable_questions": _clean_list(obj.get("answerable_questions"), MAX_QUESTIONS),
        "summary": summary,
        "key_facts": _checked("key_facts", MAX_FACTS),
        "mentioned_ids": _checked("mentioned_ids", MAX_MENTIONS),
        "mentioned_people": _checked("mentioned_people", MAX_MENTIONS),
        "mentioned_systems": _checked("mentioned_systems", MAX_MENTIONS),
    }
    return record, dropped


def _has_signal(record: dict[str, Any]) -> bool:
    """A record earns a place only if it carries some searchable signal after the guard."""
    return bool(record.get("answerable_questions") or record.get("summary")
                or record.get("key_facts"))


# --------------------------------------------------------------------------------- build / persist
def _sig(path: Path) -> tuple[int, int]:
    try:
        st = Path(path).stat()
        return int(st.st_size), int(st.st_mtime)
    except OSError:
        return (0, 0)


def build(refs: Sequence[FileRef] | None = None, out_path: Path | None = None, *,
          distiller: Callable[[str], dict[str, Any]] | None = None) -> dict[str, Any]:
    """Distill every corpus document (xlsx per-sheet) once and persist the normalized records.

    Idempotent: a prior record whose ``content_hash`` still matches is reused verbatim — a rebuild
    never re-spends Gemini on unchanged units. Per-unit failures are recorded, not raised.

    ``distiller`` is injectable (defaults to the Gemini path) so the build logic — unit slicing, the
    hash cache, and the anti-hallucination guard — is testable offline without credentials.
    """
    from src.rag.extract import extract  # heavy, build-only deps

    refs = list(refs) if refs is not None else list(walk())
    out = Path(out_path) if out_path is not None else default_out_path()
    distill = distiller or _gemini_distiller

    prior: dict[str, dict[str, Any]] = {}
    if out.exists():
        try:
            for line in out.read_text(encoding="utf-8").splitlines():
                rec = json.loads(line)
                if rec.get("schema") == SCHEMA:
                    continue
                h = rec.get("content_hash")
                if h:
                    prior[h] = rec
        except Exception:  # noqa: BLE001 — a corrupt prior store is rebuilt from scratch
            prior = {}

    records: list[dict[str, Any]] = []
    skipped: list[dict[str, str]] = []
    distilled = reused = dropped_values = 0

    for ref in refs:
        if ref.ext == "png":
            continue
        try:
            doc = extract(ref)
        except Exception as e:  # noqa: BLE001 — one bad file must not sink the store
            skipped.append({"rel": nfc(ref.rel), "reason": f"extract {type(e).__name__}: {e}"})
            continue
        if doc.error or not (doc.text or "").strip():
            if doc.error:
                skipped.append({"rel": nfc(ref.rel), "reason": doc.error})
            continue
        rel = nfc(ref.rel)
        size, mtime = _sig(ref.path)
        for unit, unit_key, unit_text in _units(doc.kind, doc.text):
            if len(unit_text.strip()) < MIN_BODY_CHARS:
                continue
            h = content_hash(unit_text)
            old = prior.get(h)
            if old and old.get("rel") == rel and old.get("unit_key", "") == unit_key:
                records.append(old)
                reused += 1
                continue
            try:
                raw = distill(unit_text)
            except Exception as e:  # noqa: BLE001 — one bad unit must not sink the store
                skipped.append({"rel": rel, "unit_key": unit_key,
                                "reason": f"distill {type(e).__name__}: {e}"})
                continue
            fields, dropped = sanitize(raw, unit_text)
            dropped_values += len(dropped)
            if not _has_signal(fields):
                skipped.append({"rel": rel, "unit_key": unit_key,
                                "reason": "no supported signal after guard"})
                continue
            rec = {
                "doc_id": rel,
                "rel": rel,
                "unit": unit,
                "unit_key": unit_key,
                "content_hash": h,
                "size": size,
                "mtime": mtime,
                "model": _model_name(distiller),
                **fields,
                "dropped": dropped,
            }
            records.append(rec)
            distilled += 1

    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        f.write(json.dumps({"schema": SCHEMA, "version": SCHEMA_VERSION}, ensure_ascii=False) + "\n")
        for rec in sorted(records, key=lambda r: (r["rel"], r.get("unit_key", ""))):
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    report = {
        "units": len(records),
        "distilled": distilled,
        "reused": reused,
        "docs": len({r["rel"] for r in records}),
        "dropped_values": dropped_values,
        "skipped": skipped,
    }
    default_report_path().write_text(json.dumps(report, ensure_ascii=False, indent=1), encoding="utf-8")
    load.cache_clear()
    return {"records": len(records), "report": report, "out": str(out)}


def _model_name(distiller: Callable | None) -> str:
    if distiller is None:
        try:
            from src.rag import llm
            return str(llm.settings.GEN_MODEL)
        except Exception:  # noqa: BLE001
            return ""
    return getattr(distiller, "__name__", "custom")


# --------------------------------------------------------------------------------- load / serve API
@functools.lru_cache(maxsize=4)
def _load_cached(path_str: str, sig: tuple) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    try:
        for line in Path(path_str).read_text(encoding="utf-8").splitlines():
            rec = json.loads(line)
            if rec.get("schema") == SCHEMA:
                if rec.get("version") != SCHEMA_VERSION:
                    return []
                continue
            if rec.get("doc_id"):
                records.append(rec)
    except Exception:  # noqa: BLE001 — fail-open: unreadable store ⇒ empty ⇒ live behaviour
        return []
    return records


def load(path: Path | None = None) -> list[dict[str, Any]]:
    p = Path(path) if path is not None else default_out_path()
    return _load_cached(str(p), _sig(p))


load.cache_clear = _load_cached.cache_clear  # type: ignore[attr-defined]


def records_for(doc_id: str, path: Path | None = None) -> list[dict[str, Any]]:
    """All distilled units for one document (by NFC rel / doc_id)."""
    did = nfc(doc_id)
    return [r for r in load(path) if r.get("doc_id") == did]


def _bigrams(s: str) -> set[str]:
    f = _fold(s)
    return {f[i:i + 2] for i in range(len(f) - 1)} if len(f) >= 2 else ({f} if f else set())


def _record_text(rec: dict[str, Any]) -> str:
    parts: list[str] = list(rec.get("answerable_questions") or [])
    if rec.get("summary"):
        parts.append(rec["summary"])
    parts += list(rec.get("key_facts") or [])
    parts += list(rec.get("mentioned_ids") or [])
    parts += list(rec.get("mentioned_systems") or [])
    return " ".join(parts)


def search(question: str, *, limit: int = 20, path: Path | None = None) -> list[tuple[str, float]]:
    """Serve-time LLM-free lexical match of ``question`` against distilled units → [(doc_id, score)].

    Character-bigram overlap over each unit's answerable_questions + summary + key_facts + ids/systems.
    Best score per doc wins; ties break by doc_id for determinism. Empty when the store is disabled or
    absent (fail-open). NEVER calls Gemini — this is the whole point of the build-time distillation.
    """
    if not enabled():
        return []
    q = _bigrams(question)
    if not q:
        return []
    best: dict[str, float] = {}
    for rec in load(path):
        rt = _bigrams(_record_text(rec))
        if not rt:
            continue
        overlap = len(q & rt)
        if not overlap:
            continue
        score = overlap / (len(q) ** 0.5)
        did = rec["doc_id"]
        if score > best.get(did, 0.0):
            best[did] = score
    ranked = sorted(best.items(), key=lambda kv: (-kv[1], kv[0]))
    return ranked[:limit]


def candidate_doc_ids(question: str, *, allowed: Iterable[str] | None = None,
                      limit: int = 20, path: Path | None = None) -> list[str]:
    """Ordered doc_ids for the registry distill tier, optionally scoped to ``allowed`` (fail-open)."""
    allow = {nfc(a) for a in allowed} if allowed is not None else None
    out: list[str] = []
    for did, _score in search(question, limit=limit * 3, path=path):
        if allow is None or did in allow:
            out.append(did)
        if len(out) >= limit:
            break
    return out


def text_records(path: Path | None = None) -> list[dict[str, Any]]:
    """Distilled units as a dual-representation source for full-text search (SOT-2657 FTS ingestion).

    Each row is ``{source:"distill", doc_id, unit, unit_key, text}`` where ``text`` is the normalized
    searchable blob (answerable_questions + summary + key_facts). SOT-2657's FTS builder can ingest
    these as a distinct source type alongside the raw text layer (Cerebras dual-representation). Returns
    [] when the store is absent — the FTS build simply gets no distill rows (fail-open).
    """
    rows: list[dict[str, Any]] = []
    for rec in load(path):
        blob = _record_text(rec)
        if blob.strip():
            rows.append({"source": "distill", "doc_id": rec["doc_id"], "unit": rec.get("unit", "doc"),
                         "unit_key": rec.get("unit_key", ""), "text": blob})
    return rows
