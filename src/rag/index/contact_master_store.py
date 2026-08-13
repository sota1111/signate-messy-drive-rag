"""担当者マスタ（契約書署名欄 権威ソース）ストア — 全案件 × 全契約書の署名欄／当事者ブロックの
構造化フィールド（会社名／部署名／主担当者／役職, クライアント側=甲・受託側=乙）を質問非依存で
ビルド時に全数抽出する事実層 (SOT-2707 / cycle10).

親 SOT-2701 cycle10 分析で機械確認済みの wrong idx21（青葉バイオメディカル機器のクライアント主担当者の
役職 → gold=人材戦略部長）は、cycle9 実測が **会議録の出席者展開形**『人事本部 人材戦略部 部長』を
写経して Incorrect だった。会議録（記述文脈）より **契約書署名欄／当事者ブロックの構造化フィールド**
『役職：』が権威ソースである。完全ラベル写経（SOT-2666）の *権威ソース選好版* をこのストア＋レーンで恒久化する。

方針: **実行時に証拠を探すのではなく、契約書の構造化フィールドをオフラインで網羅抽出しておく**。全案件 ×
全契約書（`契約書.docx` / `契約書_draft.docx` を含む。暗号化 Office は :mod:`src.rag.extract` が
`passwords.resolve` 資産で透過解読）から、当事者ブロックの `会社名：/部署名：/主担当者：/役職：` を
甲（クライアント）・乙（受託）別に re 抽出する。ラベルは原文どおり完全写経（LLM フリー）。

正当性の歯止め（必読）:
  * **質問非依存の網羅計算のみ.** 全案件 × 標準フィールドを機械抽出するのみ。特定設問を狙った選別・gold 値の
    ハードコードは一切しない。値は原文の構造化フィールドを完全写経する（正規化・言い換えをしない）。
  * **決定論・LLM フリー・再現バイト一致.** Gemini 呼び出しなし・ネットワークなし。
  * **一意性（fail-closed）.** 同一案件の全契約書で同じ (当事者, フィールド) が **一意** の値に収束するときのみ
    確定。draft と最終で食い違う等の複数候補は ``value=None``（欠測を偽装せず reason を残す）。
  * **Opt-in at serve time.** :func:`enabled` (``RAG_CONTACT_MASTER``) は *実行時参照のみ* を gate する。
    アーティファクトは index 時に常に additive・guarded に（再）ビルドされる。既定 OFF ⇒ champion serve path は
    バイト一致。読み出し/配線は :mod:`src.rag.agent.contact_master_lane`。
"""
from __future__ import annotations

import importlib
import json
import os
import re
import unicodedata
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

from config import settings
from src.rag.corpus import FileRef, nfc, walk

# case_master の universe 規則・主略称解決を流用（値の一貫性を担保）。
_cm = importlib.import_module("src.rag.index.case_master")
ca = importlib.import_module("src.rag.tools.corpus_aggregate")

SCHEMA = "contact-master"
SCHEMA_VERSION = 1

_ON = {"1", "true", "yes", "on"}

# 契約書の拡張子（当事者/署名欄の構造化フィールドを持つ形式）。
_CONTRACT_EXT = {"docx", "pdf", "pptx"}

# 構造化フィールドのラベル → 内部キー（原文ラベルを完全写経、値は正規化しない）。
_FIELD_LABELS: tuple[tuple[str, str], ...] = (
    ("会社名", "company"),
    ("部署名", "department"),
    ("主担当者", "contact"),
    ("役職", "role"),
)
_LABEL_RE = re.compile(
    r"^(会社名|部署名|主担当者|役職)\s*[：:]\s*(.+?)\s*$")

# 当事者マーカー行（甲＝クライアント / 乙＝受託）。NFKC 正規化後の短い行だけをマーカーと見なし、
# 本文中の「（以下「甲」という。）」のような長い行は取らない（$ 終端＋短さで担保）。
_PARTY_RE = re.compile(
    r"^[(（]?\s*[0-9]*\s*[)）]?\s*(甲|乙)\s*(?:[(（][^)）]{0,12}[)）])?\s*[：:]?\s*$")

_PARTY_KEY = {"甲": "client", "乙": "vendor"}

# 出力する標準フィールド（クライアント側＝甲 / 受託側＝乙）。
STANDARD_OPERANDS = [
    "client_company", "client_department", "client_contact", "client_role",
    "vendor_company", "vendor_department", "vendor_contact", "vendor_role",
]


def enabled() -> bool:
    """True when the serve path should consult the contact-master store (default OFF — opt-in)."""
    return os.getenv("RAG_CONTACT_MASTER", "0").strip().lower() in _ON


def default_out_path() -> Path:
    return settings.ARTIFACTS_DIR / "contact_master.jsonl"


def report_out_path() -> Path:
    return settings.ARTIFACTS_DIR / "contact_master_build_report.json"


# --------------------------------------------------------------------------- source / cell helpers
def _source(doc_id: str, *, snippet: str | None = None) -> dict[str, Any]:
    src: dict[str, Any] = {"doc_id": nfc(doc_id)}
    if snippet is not None:
        src["snippet"] = snippet[:160]
    return src


def _cell(value: Any, source: dict[str, Any]) -> dict[str, Any]:
    return {"value": value, "source": source}


def _missing(reason: str) -> dict[str, Any]:
    return {"value": None, "source": None, "reason": reason}


def _extract_text(ref: FileRef) -> str:
    try:
        doc = ca._extract(ref, caption_images=False)  # transparent decrypt, no vision/network
        return getattr(doc, "text", "") or ""
    except Exception:  # noqa: BLE001 — one unreadable doc must not sink the row
        return ""


def _contract_refs(project: str, refs: Sequence[FileRef]) -> list[FileRef]:
    """A project's contract documents INCLUDING drafts (署名欄の権威ソース網羅), ``old`` 版のみ除外。"""
    out: list[FileRef] = []
    for r in refs:
        if r.project != project or r.category != "contract" or r.ext not in _CONTRACT_EXT:
            continue
        low = ("/" + r.rel.lower())
        if "/old/" in low or re.search(r"(?:^|[_\-.])old(?:[_\-.]|$)", r.name.lower()):
            continue
        out.append(r)
    return out


# --------------------------------------------------------------------------- signature-block parsing
def _party_fields(text: str) -> dict[str, dict[str, tuple[str, str]]]:
    """Parse a contract's 当事者/署名欄 structured fields keyed by party then field.

    Returns ``{"client"|"vendor": {field: (value, snippet)}}`` — the FIRST occurrence of each
    ``会社名/部署名/主担当者/役職`` label under a 甲/乙 party marker (原文ラベルを完全写経). Labels seen
    before any party marker are ignored (attributed to no party). A party's later duplicate label is
    ignored (the party block's first structured statement is authoritative; the 署名欄 re-statement
    carries no ``役職：`` label so it never collides)."""
    out: dict[str, dict[str, tuple[str, str]]] = {"client": {}, "vendor": {}}
    party: str | None = None
    for raw in text.splitlines():
        line = unicodedata.normalize("NFKC", raw).strip()
        if not line:
            continue
        pm = _PARTY_RE.match(line)
        if pm is not None:
            party = _PARTY_KEY.get(pm.group(1))
            continue
        if party is None:
            continue
        lm = _LABEL_RE.match(line)
        if lm is None:
            continue
        label, value = lm.group(1), lm.group(2).strip()
        key = dict(_FIELD_LABELS).get(label)
        if key is None or not value:
            continue
        if key not in out[party]:
            out[party][key] = (value, line)
    return out


@dataclass
class ContactMaster:
    case_id: str
    abbrev: str | None
    operands: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return {"case_id": self.case_id, "abbrev": self.abbrev, "operands": self.operands}


def _make_record(project: str, refs: Sequence[FileRef], glossary: Any) -> ContactMaster:
    abbrev = ca._abbrev_for(project, glossary)
    contract_refs = _contract_refs(project, refs)

    # (party, field) → {value: snippet_source}. Collapse to a single distinct value else ambiguous.
    collected: dict[tuple[str, str], dict[str, dict[str, Any]]] = {}
    for ref in contract_refs:
        fields = _party_fields(_extract_text(ref))
        for party, fmap in fields.items():
            for key, (value, snippet) in fmap.items():
                slot = collected.setdefault((party, key), {})
                slot.setdefault(value, _source(ref.rel, snippet=snippet))

    ops: dict[str, Any] = {}
    for party in ("client", "vendor"):
        for _label, key in _FIELD_LABELS:
            op = f"{party}_{key}"
            slot = collected.get((party, key), {})
            if len(slot) == 1:
                value, src = next(iter(slot.items()))
                ops[op] = _cell(value, src)
            elif len(slot) > 1:
                ops[op] = _missing(f"{party}.{key} が契約書間で一意でない（候補{len(slot)}件）")
            else:
                ops[op] = _missing(f"{party}.{key} の構造化フィールドを契約書から抽出できない")

    return ContactMaster(case_id=nfc(project), abbrev=abbrev, operands=ops)


# --------------------------------------------------------------------------- build
def build(refs: list[FileRef] | None = None, *, out: Path | None = None,
          glossary: Any = None, write_report: bool = True) -> dict[str, Any]:
    """Precompute + persist the contact-master store (one row per 案件) with per-cell provenance.

    Deterministic and LLM-free. Additive & guarded — a failure here never corrupts the retrieval index
    (the caller in :mod:`src.rag.index` swallows exceptions)."""
    refs = refs if refs is not None else walk()
    g = glossary if glossary is not None else _cm._load_glossary()
    universe = _cm._case_universe(refs)
    records = [_make_record(p, refs, g) for p in universe]

    counts = write_store(records, out)
    report = build_report(records, universe, out or default_out_path())
    if write_report:
        rp = report_out_path()
        rp.parent.mkdir(parents=True, exist_ok=True)
        rp.write_text(json.dumps(report, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
                      encoding="utf-8")
    return {**counts, "report": report}


def write_store(records: Sequence[ContactMaster], path: Path | None = None) -> dict[str, int]:
    """Atomically write a reproducible JSONL store (schema header + case_id-sorted rows)."""
    out = path or default_out_path()
    out.parent.mkdir(parents=True, exist_ok=True)
    ordered = sorted(records, key=lambda r: r.case_id)
    tmp = out.with_suffix(out.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8", newline="\n") as handle:
        handle.write(json.dumps({"schema": SCHEMA, "version": SCHEMA_VERSION},
                                ensure_ascii=False, sort_keys=True) + "\n")
        for rec in ordered:
            handle.write(json.dumps(rec.to_dict(), ensure_ascii=False, sort_keys=True) + "\n")
    tmp.replace(out)
    reset_cache()
    return {"cases": len(ordered)}


def _op_value(rec: ContactMaster, name: str) -> Any:
    return (rec.operands.get(name) or {}).get("value")


def build_report(records: Sequence[ContactMaster], universe: Sequence[str],
                 out_path: Path) -> dict[str, Any]:
    n = len(records) or 1
    fill: dict[str, int] = {}
    for op in STANDARD_OPERANDS:
        fill[op] = sum(1 for r in records if _op_value(r, op) not in (None, "", [], {}))
    return {
        "certificate": {
            "schema": SCHEMA, "schema_version": SCHEMA_VERSION, "question_independent": True,
            "universe_basis": "case_master と同じ 01.契約 フォルダを持つ全案件（社内管理除く）",
            "case_count": len(records), "row_count_equals_universe": len(records) == len(universe),
            "computation_basis": ("全案件 × 全契約書（draft 含む）の当事者/署名欄 構造化フィールド"
                                  "（会社名/部署名/主担当者/役職, 甲=クライアント・乙=受託）の re 抽出。"
                                  "ラベル完全写経・gold 非依存・LLM 非関与・一意時のみ確定。"),
        },
        "operand_fill": {op: {"filled": fill[op], "missing": len(records) - fill[op],
                              "missing_rate": round((len(records) - fill[op]) / n, 4)}
                         for op in STANDARD_OPERANDS},
        "sample_client_roles": [
            {"case_id": r.case_id, "abbrev": r.abbrev, "client_role": _op_value(r, "client_role")}
            for r in records if _op_value(r, "client_role") is not None
        ][:10],
        "out_path": str(out_path),
    }


# --------------------------------------------------------------------------- load / read (minimal API)
_LOAD_CACHE: dict[str, list[dict[str, Any]]] = {}


def reset_cache() -> None:
    _LOAD_CACHE.clear()


def load(path: Path | None = None) -> list[dict[str, Any]]:
    """Load the contact-master rows (memoized). ``[]`` when absent/unreadable/schema-mismatch (回帰ゼロ)."""
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


def operand(row: dict[str, Any], name: str) -> Any:
    return (row.get("operands", {}).get(name) or {}).get("value")


def case_record(case_hint: str, *, path: Path | None = None) -> dict[str, Any] | None:
    """The contact row for a 案件 (exact case_id preferred, else substring). ``None`` when absent."""
    hint = nfc(case_hint).strip()
    rows = load(path)
    for row in rows:
        if nfc(row.get("case_id", "")) == hint:
            return row
    for row in rows:
        if hint and hint in nfc(row.get("case_id", "")):
            return row
    return None


if __name__ == "__main__":
    summary = build()
    print(json.dumps({"cases": summary["cases"],
                      "certificate": summary["report"]["certificate"],
                      "operand_fill": summary["report"]["operand_fill"],
                      "sample_client_roles": summary["report"]["sample_client_roles"]},
                     ensure_ascii=False, indent=2))
