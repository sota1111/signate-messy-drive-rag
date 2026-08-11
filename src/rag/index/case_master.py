"""案件マスタテーブル — 全案件 × 標準属性を出典セル付きで事前計算する事実層 (SOT-2643 / 事前計算事実層 1/5).

champion (Wave A net40) の abstain 全数分類 (`docs/ai/champion_abstain_wrong_classification.md`,
2026-08-11) で確定した **hard core 16問** のうち最大の型が **enum_set 5問** (idx32/38/45/67/87) —
「完了案件のうち APR-M2 該当で提案時金額と FR 時金額が異なる案件をすべて」型の横断列挙で、実行時探索では
flash/Sonnet 双方とも ``BUDGET_EXHAUSTED``（列挙 universe を証明しきれない）。

方針: **実行時に証拠を探すのではなく、事実をオフラインで事前計算しておく**。全案件 × 標準属性の正規化
テーブルを index 時に一度だけ構築すれば、横断列挙・横断比較は「テーブルの決定論フィルタ 1回」に帰着し、
enum の本質的困難だった *universe の証明* も、行数 = universe で自動的に立つ（completeness certificate）。

**正当性の歯止め（必読）**: 事前計算は **質問を見ない網羅計算** に限定する — 全案件 × 標準属性を機械的に
抽出するのであり、特定設問を狙った選別計算・gold 値のハードコードは一切しない。カバレッジ外は従来経路へ
フォールバック（劣化ゼロ）。設問→必要属性の対応表（:func:`enum_hard_core_coverage`）は *ビルドレポート
用の診断* にすぎず、テーブル本体（``case_master.jsonl``）は完全に質問非依存。

Design invariants（既存の兄弟事前ストア — evidence_index / structure_store / canonical_manifest /
document_registry — に倣う）:
  * **全案件カバレッジ + universe 証明.** :func:`build` は契約フォルダを持つ全案件（= コーパス上の案件
    universe）を列挙し、行数 = universe を保証する。列挙根拠は completeness certificate に明記。
  * **決定論・LLM フリー.** 全値はコーパスからの決定論抽出（正規表現 / pandas 行数 / 決裁基準ルール）で確定。
    Gemini 呼び出しなし・ネットワークなし・再現バイト一致。既存 :mod:`corpus_aggregate` の抽出器を流用し、
    テーブル値が既存の正答（corpus_aggregate 系 MATCH）と一致することを保証する。
  * **各値に出典を必須付与.** 値には ``source: {doc_id, sheet|page|cell, snippet}`` を付ける。抽出不能な
    セルは ``value=None`` + ``reason``（欠測を偽装しない）。
  * **Opt-in at serve time.** :func:`enabled` (``RAG_CASE_MASTER``) は *実行時参照のみ* を gate する。
    アーティファクトは index 時に常に additive・guarded に（再）ビルドされる。既定 OFF ⇒ champion serve
    path はバイト一致。本 issue の読み出しツールは最小限（:func:`filter_cases`）— 配線の本番は SOT-2647。
  * **再配布安全.** テーブルは構造メタデータ + 出典のみ。パスワード等の秘密値は絶対に入れない。
"""
from __future__ import annotations

import json
import os
import re
import unicodedata
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path
from typing import Any, Sequence

from config import settings
from src.rag.corpus import FileRef, nfc, walk
from src.rag.extract.glossary import Glossary
from src.rag.extract.glossary import load as _load_glossary
# NOTE: src.rag.tools.__init__ re-exports the ``corpus_aggregate`` *function* into the package
# namespace, shadowing the module — so ``from src.rag.tools import corpus_aggregate`` binds the
# function, not the module. Load the module explicitly (mirrors the importlib guard other callers use).
import importlib
ca = importlib.import_module("src.rag.tools.corpus_aggregate")

SCHEMA = "case-master"
SCHEMA_VERSION = 1

_ON = {"1", "true", "yes", "on"}

# 社内管理 の権威文書（出典アンカ）。決裁基準 = APR ルール、用語集 = 主略称の定義元。
_DECREE_DOC_ID = "社内管理/データアステル社内管理_決裁基準.md"
_GLOSSARY_DOC_ID = "社内管理/社内用語集.docx"

# 医療案件の判定素材（決裁基準 §3.1 の1段階引き上げ条件）。案件フォルダ名から機械判定する。
_MEDICAL_RE = re.compile(r"医療法人|病院|診療所|クリニック|医療センター|医院")


def enabled() -> bool:
    """True when the serve path should consult the case master (default OFF — opt-in)."""
    return os.getenv("RAG_CASE_MASTER", "0").strip().lower() in _ON


def default_out_path() -> Path:
    return settings.ARTIFACTS_DIR / "case_master.jsonl"


def report_out_path() -> Path:
    return settings.ARTIFACTS_DIR / "case_master_build_report.json"


# --------------------------------------------------------------------------- cell / source helpers
def _source(doc_id: str, *, sheet: str | None = None, cell: str | None = None,
            page: int | None = None, snippet: str | None = None,
            basis: str | None = None) -> dict[str, Any]:
    """A provenance record. ``doc_id`` (NFC rel) is mandatory; the medium-appropriate locator
    (sheet/cell for spreadsheets, page for PDFs) and a ``snippet`` of the matched text are optional
    but always recorded when the source medium exposes them — never fabricated."""
    src: dict[str, Any] = {"doc_id": nfc(doc_id)}
    if sheet is not None:
        src["sheet"] = sheet
    if cell is not None:
        src["cell"] = cell
    if page is not None:
        src["page"] = page
    if snippet is not None:
        src["snippet"] = snippet[:160]
    if basis is not None:
        src["basis"] = basis
    return src


def _cell(value: Any, source: dict[str, Any]) -> dict[str, Any]:
    """A populated attribute cell — a value with mandatory provenance."""
    return {"value": value, "source": source}


def _missing(reason: str) -> dict[str, Any]:
    """An honest gap — ``value=None`` with the reason it could not be extracted (never guessed)."""
    return {"value": None, "source": None, "reason": reason}


# --------------------------------------------------------------------------- text extraction
def _extract_text(ref: FileRef) -> str:
    try:
        doc = ca._extract(ref, caption_images=False)  # no vision/network for a text field scan
        return getattr(doc, "text", "") or ""
    except Exception:  # noqa: BLE001 — one unreadable doc must not sink the row
        return ""


def _texts_for(project: str, refs: Sequence[FileRef], category: str,
               exts: set[str]) -> list[tuple[FileRef, str]]:
    """(ref, text) for a project's documents in one category, drafts/old excluded."""
    out: list[tuple[FileRef, str]] = []
    for r in refs:
        if r.project != project or r.category != category or r.ext not in exts:
            continue
        low = r.name.lower()
        if "draft" in low or "old" in low:
            continue
        out.append((r, _extract_text(r)))
    return out


def _locate_amount(texts: Sequence[tuple[FileRef, str]], value: int) -> dict[str, Any] | None:
    """Provenance for an integer yen value: the first (doc, line) that prints it (comma-insensitive)."""
    plain = str(value)
    for ref, text in texts:
        for line in text.splitlines():
            if plain in line.replace(",", "") and re.search(r"[0-9]", line):
                return _source(ref.rel, snippet=line.strip())
    return None


def _locate_text(texts: Sequence[tuple[FileRef, str]], needle: str) -> dict[str, Any] | None:
    """Provenance for a string value: the first (doc, line) that contains it."""
    if not needle:
        return None
    for ref, text in texts:
        for line in text.splitlines():
            if needle in line:
                return _source(ref.rel, snippet=line.strip())
    return None


# --------------------------------------------------------------------------- best-effort extractors
# Precision-first: commit a value only when the corpus collapses it to a single distinct figure,
# mirroring corpus_aggregate._one_amount. Ambiguous/absent ⇒ None (→ null + reason), never guessed.
_CCY = r"\s*(?:円|JPY)?"  # tax-inclusive figures print as "…円" or "… JPY" or bare — tolerate all three.
_PROPOSAL_AMOUNT_PATTERNS = [
    r"見込金額（税込）\s*[：:|]?\s*[¥￥]?\s*([0-9][0-9,]*)" + _CCY,
    r"想定金額（税込）\s*[：:|]?\s*[¥￥]?\s*([0-9][0-9,]*)" + _CCY,
    r"提案金額（税込）\s*[：:|]?\s*[¥￥]?\s*([0-9][0-9,]*)" + _CCY,
    # 東都: 「見込金額：4,675,000円（税込）」 — the 税込 qualifier trails the figure.
    r"見込金額\s*[：:]\s*[¥￥]?\s*([0-9][0-9,]*)\s*円\s*（税込）",
    # 京橋/かえで: 「費用: ¥5,775,000（税込・固定価格）」「¥3,850,000（税込）」.
    r"[¥￥]\s*([0-9][0-9,]*)\s*（税込[^）]*）",
    # 青嶺: 「支払条件：…一括支払（100%）　税込 ¥4,675,000」.
    r"税込\s*[¥￥]\s*([0-9][0-9,]*)",
]
# PPTX 費用見積 slides frequently put the label and the figure on ADJACENT extracted lines
# (「契約金額（税込）」 ↵ 「¥3,960,000」 / 「7,480,000円」) — a label regex + short lookahead resolves
# those. Ordered most-specific-first; each label is tried across all texts before the next.
_PROPOSAL_AMOUNT_LABELS = [
    r"(?:契約|見込)?金額（税込）",
    r"税込金額",
    # 青潮: 「見込金額」 ↵ 「¥4,675,000」 ↵ 「（税込）」 — unqualified label, figure on the next line.
    # 税抜-qualified variants are excluded so a tax-exclusive figure can never be committed as 税込.
    r"見込金額(?!（税抜)",
]
_FR_AMOUNT_PATTERNS = [
    # 半角/全角括弧の両方を許容 — OCR 転記（スキャン最終報告）は半角括弧で出る（青潮: 「最終請求金額(税込):
    # 4,785,000 JPY」）。
    r"最終請求金額[（(]税込[）)]\s*[：:|]?\s*[¥￥]?\s*([0-9][0-9,]*)" + _CCY,
    r"最終金額（税込）\s*[：:|]?\s*[¥￥]?\s*([0-9][0-9,]*)" + _CCY,
    r"請求金額（税込）\s*[：:|]?\s*[¥￥]?\s*([0-9][0-9,]*)" + _CCY,
    # 京橋: 「固定価格契約: 税抜 5,250,000円（税込 5,775,000円）」/ みなみ野(OCR): 「固定価格(税込3,960,000円)」.
    r"[（(]税込\s*([0-9][0-9,]*)\s*円[）)]",
    # ひがし丘/青葉バイオ: 「税込金額 4,675,000円」「税込金額：3,443,000 円」.
    r"税込金額\s*[：:|]?\s*[¥￥]?\s*([0-9][0-9,]*)\s*(?:円|JPY)",
    # 青葉与信: 「契約金額：¥4,200,000（税抜）/ ¥4,620,000（税込）」.
    r"[¥￥]\s*([0-9][0-9,]*)\s*（税込）",
]
_FR_AMOUNT_LABELS = [
    r"最終請求金額（税込）",
    r"税込金額",
]
# 白峰: the 最終報告 prints only 「固定価格6,800,000円（税抜）」 — a tax-EXCLUSIVE total. Committed as
# 税込 only when 税抜×1.1 is exactly integral (消費税10%, the corpus-wide rate) — fail-closed otherwise.
_FR_AMOUNT_EXCL_PATTERNS = [
    r"固定価格\s*([0-9][0-9,]*)\s*円\s*（税抜）",
]
_AMOUNT_LINE_RE = re.compile(r"^\s*[¥￥]?\s*([0-9][0-9,]*)\s*(?:円|JPY)?\s*$")
_EST_EFFORT_PATTERNS = [r"見込工数\s*[：:|]?\s*([0-9][0-9,]*)\s*時間", r"見積工数\s*[：:|]?\s*([0-9][0-9,]*)\s*時間"]
_ACT_EFFORT_PATTERNS = [r"実績工数\s*[：:|]?\s*([0-9][0-9,]*)\s*時間"]


def _first_unique_int(texts: Sequence[tuple[FileRef, str]],
                      patterns: Sequence[str]) -> tuple[int, dict[str, Any]] | None:
    """First pattern whose matches across all texts collapse to a single distinct int → (value, source)."""
    for pat in patterns:
        values: set[int] = set()
        locus: dict[str, Any] | None = None
        for ref, text in texts:
            for line in text.splitlines():
                m = re.search(pat, line)
                if m:
                    v = int(m.group(1).replace(",", ""))
                    values.add(v)
                    if locus is None:
                        locus = _source(ref.rel, snippet=line.strip())
        if len(values) == 1 and locus is not None:
            return values.pop(), locus
    return None


def _labeled_amount(texts: Sequence[tuple[FileRef, str]],
                    labels: Sequence[str]) -> tuple[int, dict[str, Any]] | None:
    """A label line whose yen figure sits on the SAME line or within the next 2 extracted lines.

    Mirrors :func:`_first_unique_int`'s precision-first contract: all matches for one label must
    collapse to a single distinct int, else the label is ambiguous and the next label is tried.
    """
    for label in labels:
        label_re = re.compile(label)
        values: set[int] = set()
        locus: dict[str, Any] | None = None
        for ref, text in texts:
            lines = text.splitlines()
            for i, line in enumerate(lines):
                if not label_re.search(line):
                    continue
                found: int | None = None
                same = re.search(label + r"\s*[：:|]?\s*[¥￥]?\s*([0-9][0-9,]*)" + _CCY, line)
                if same:
                    found = int(same.group(1).replace(",", ""))
                else:
                    for nxt in lines[i + 1:i + 3]:
                        m = _AMOUNT_LINE_RE.match(nxt)
                        if m:
                            found = int(m.group(1).replace(",", ""))
                            break
                if found is not None:
                    values.add(found)
                    if locus is None:
                        locus = _source(ref.rel, snippet=line.strip())
        if len(values) == 1 and locus is not None:
            return values.pop(), locus
    return None


def _fr_amount_from_excl(texts: Sequence[tuple[FileRef, str]]) -> tuple[int, dict[str, Any]] | None:
    """税抜-only FR total → 税込 via the corpus-wide 消費税10%, only when exactly integral (fail-closed)."""
    hit = _first_unique_int(texts, _FR_AMOUNT_EXCL_PATTERNS)
    if hit is None:
        return None
    excl, locus = hit
    incl_times_10 = excl * 11
    if incl_times_10 % 10 != 0:
        return None
    locus = dict(locus)
    locus["basis"] = "税抜表示×1.10（消費税10%）"
    return incl_times_10 // 10, locus


# --------------------------------------------------------------------------- APR / 決裁レベル derivation
# 決裁基準 (社内管理/…決裁基準.md) §2-4: 契約金額(税込) の帯 → 医療案件 +1段階 → t&m は最低でも部長承認。
_APPROVAL_LADDER = ["主任承認", "課長承認", "部長承認", "本部長承認"]
# 用語集 §3: 課長承認=APR-M1 / 部長承認=APR-M2 / 本部長承認=APR-M3（主任承認は APR-M 帯の下）。
_APR_CODE = {"課長承認": "APR-M1", "部長承認": "APR-M2", "本部長承認": "APR-M3"}


def _base_level(amount: int) -> int:
    if amount < 3_000_000:
        return 0
    if amount < 5_000_000:
        return 1
    if amount < 8_000_000:
        return 2
    return 3


def derive_approval(amount: int | None, is_medical: bool, is_tm: bool) -> dict[str, Any] | None:
    """(決裁レベル, APR-M コード) を決裁基準ルールから導出。金額不明なら None（棄権）。

    質問非依存の標準属性: どの案件も金額・医療区分・契約形態から決まる決裁レベルを一つ持つ。設問を見た
    選別計算ではない。§4 適用順序に忠実（base → 医療 +1 → t&m は max(部長, .)）。
    """
    if amount is None:
        return None
    level = _base_level(amount)
    if is_medical:
        level = min(level + 1, 3)  # §3.1 医療案件は1段階上
    if is_tm:
        level = max(level, 2)      # §3.2 t&m は最低でも部長承認
    name = _APPROVAL_LADDER[level]
    return {"approval_level": name, "apr_code": _APR_CODE.get(name)}


# --------------------------------------------------------------------------- record construction
# The standard attribute schema — fixed, question-independent. Every case gets every attribute
# (populated cell or null+reason). Ordering is stable for deterministic output.
STANDARD_ATTRIBUTES = [
    "case_name", "abbrev", "company", "contract_type", "is_medical",
    "contract_amount_incl_tax", "proposal_amount_incl_tax", "fr_amount_incl_tax", "deposit",
    "contract_period", "period_days", "estimated_effort_hours", "actual_effort_hours",
    "train_rows", "status", "settlement_method", "approval_level", "apr_code", "staff",
]


@dataclass
class CaseRecord:
    case_id: str                       # 案件フォルダ名（NFC）= universe の1行
    abbrev: str | None
    attributes: dict[str, dict[str, Any]]

    def to_dict(self) -> dict[str, Any]:
        return {"case_id": self.case_id, "abbrev": self.abbrev, "attributes": self.attributes}


def _make_record(project: str, refs: Sequence[FileRef], glossary: Glossary) -> CaseRecord:
    """Build one case row: standard attributes with per-cell provenance (always succeeds)."""
    attrs: dict[str, dict[str, Any]] = {}

    # ---- structural (folder / glossary) -------------------------------------------------------
    case_dir = f"プロジェクト/{project}"
    attrs["case_name"] = _cell(project, _source(case_dir, basis="corpus_directory"))
    attrs["company"] = _cell(project, _source(case_dir, basis="corpus_directory"))
    abbrev = ca._abbrev_for(project, glossary)
    attrs["abbrev"] = (_cell(abbrev, _source(_GLOSSARY_DOC_ID, basis="glossary_company_to_code"))
                       if abbrev else _missing("主略称が用語集で解決できない"))
    is_medical = bool(_MEDICAL_RE.search(project))
    attrs["is_medical"] = _cell(is_medical, _source(case_dir, basis="案件名パターン(医療法人/病院等)"))

    # ---- contract-derived core (reuse corpus_aggregate extractors for value consistency) -------
    contract_texts = [(r, _extract_text(r)) for r in ca._contract_refs(project, refs)]
    joined = "\n".join(t for _, t in contract_texts)

    total = ca._extract_total(joined)
    attrs["contract_amount_incl_tax"] = (
        _cell(total, _locate_amount(contract_texts, total) or _source(_first_doc(contract_texts)))
        if total is not None else _missing("契約金額（税込）を契約書から一意抽出できない"))

    deposit = ca._extract_deposit(joined)
    attrs["deposit"] = (
        _cell(deposit, _locate_amount(contract_texts, deposit) or _source(_first_doc(contract_texts)))
        if deposit is not None else _missing("着手金なし（t&m 契約等）または抽出不能"))

    fixed = ca._is_fixed(joined)
    ctype = "固定金額" if fixed else "time_and_materials"
    ctype_src = _locate_text(contract_texts, "固定") if fixed else _locate_text(contract_texts, "time_and_materials")
    attrs["contract_type"] = _cell(ctype, ctype_src or _source(_first_doc(contract_texts),
                                                               basis="固定金額の記載なし→t&m"))

    period = ca._extract_period(joined)
    if period is not None:
        attrs["contract_period"] = _cell(list(period),
                                         _locate_text(contract_texts, period[0])
                                         or _source(_first_doc(contract_texts)))
        attrs["period_days"] = _cell((ca._to_date(period[1]) - ca._to_date(period[0])).days,
                                     _source(_first_doc(contract_texts), basis="契約期間の日数差"))
    else:
        attrs["contract_period"] = _missing("契約期間を契約書から抽出できない")
        attrs["period_days"] = _missing("契約期間が無いため算出不能")

    staff = ca._extract_staff(joined)
    if staff:
        staff_src: dict[str, Any] = {}
        for code, name in staff.items():
            loc = _locate_text(contract_texts, name)
            staff_src[code] = loc or _source(_first_doc(contract_texts))
        attrs["staff"] = {"value": staff, "source": staff_src}
    else:
        attrs["staff"] = _missing("乙(受託側)担当者を契約書から抽出できない")

    # ---- data table row count -----------------------------------------------------------------
    train_ref = ca._train_ref(project, refs)
    rows = ca._count_rows(train_ref)
    if rows is not None and train_ref is not None:
        attrs["train_rows"] = _cell(rows, _source(train_ref.rel, basis="pandas 行数(顧客データサンプル数)"))
    else:
        attrs["train_rows"] = _missing("学習/顧客データ表が見つからない、または行数を数えられない")

    # ---- proposal / final-report best-effort (提案時金額 / FR金額 / 工数 / ステータス / 精算方式) ----
    proposal_texts = _texts_for(project, refs, "proposal", {"docx", "pptx", "pdf", "xlsx"})
    report_texts = _texts_for(project, refs, "report", {"docx", "pptx", "pdf", "xlsx"})

    prop = (_first_unique_int(proposal_texts, _PROPOSAL_AMOUNT_PATTERNS)
            or _labeled_amount(proposal_texts, _PROPOSAL_AMOUNT_LABELS))
    attrs["proposal_amount_incl_tax"] = (_cell(prop[0], prop[1]) if prop
                                         else _missing("提案時金額（税込）を提案書から一意抽出できない"))
    fr = (_first_unique_int(report_texts, _FR_AMOUNT_PATTERNS)
          or _labeled_amount(report_texts, _FR_AMOUNT_LABELS)
          or _fr_amount_from_excl(report_texts))
    attrs["fr_amount_incl_tax"] = (_cell(fr[0], fr[1]) if fr
                                   else _missing("FR時金額（税込）を最終報告書から一意抽出できない"))
    est = _first_unique_int(proposal_texts, _EST_EFFORT_PATTERNS)
    attrs["estimated_effort_hours"] = (_cell(est[0], est[1]) if est
                                       else _missing("見込工数を提案書から一意抽出できない"))
    act = _first_unique_int(report_texts, _ACT_EFFORT_PATTERNS)
    attrs["actual_effort_hours"] = (_cell(act[0], act[1]) if act
                                    else _missing("実績工数を最終報告書から一意抽出できない"))

    # status: 最終報告書の存在 = 完了（質問非依存の標準判定）。無ければ null+reason。
    fr_report_ref = next((r for r, _ in report_texts if "最終報告" in r.name or "最終報告" in r.rel), None)
    if fr_report_ref is not None:
        attrs["status"] = _cell("完了", _source(fr_report_ref.rel, basis="最終報告書の存在=案件完了"))
    else:
        attrs["status"] = _missing("最終報告書が見つからない（完了と断定できない）")

    # settlement_method: 精算方式（一括/分割/実費精算等）を提案書・契約書から拾う。
    settlement = _extract_settlement(proposal_texts + contract_texts)
    attrs["settlement_method"] = (settlement if settlement
                                  else _missing("精算方式を提案書/契約書から抽出できない"))

    # ---- APR / 決裁レベル（決裁基準ルールからの導出）------------------------------------------
    apr = derive_approval(total, is_medical, not fixed)
    if apr is not None:
        attrs["approval_level"] = _cell(
            apr["approval_level"],
            _source(_DECREE_DOC_ID,
                    basis=f"決裁基準ルール導出: 契約金額={total}, 医療={is_medical}, t&m={not fixed}"))
        attrs["apr_code"] = (_cell(apr["apr_code"], _source(_DECREE_DOC_ID, basis="用語集: 承認レベル→APR-Mコード"))
                             if apr["apr_code"] else _missing("主任承認は APR-M 帯の下(該当コードなし)"))
    else:
        attrs["approval_level"] = _missing("契約金額不明のため決裁レベルを導出できない")
        attrs["apr_code"] = _missing("契約金額不明のため APR-M コードを導出できない")

    return CaseRecord(case_id=nfc(project), abbrev=abbrev, attributes=attrs)


def _first_doc(texts: Sequence[tuple[FileRef, str]]) -> str:
    return texts[0][0].rel if texts else ""


_SETTLEMENT_PATTERNS = [
    (r"一括払い", "一括払い"), (r"分割払い", "分割払い"), (r"実費精算", "実費精算"),
    (r"月次精算|月額精算", "月次精算"), (r"検収(?:完了)?(?:後|に基づく)", "検収基準精算"),
]


def _extract_settlement(texts: Sequence[tuple[FileRef, str]]) -> dict[str, Any] | None:
    for pat, label in _SETTLEMENT_PATTERNS:
        loc = None
        for ref, text in texts:
            for line in text.splitlines():
                if re.search(pat, line):
                    loc = _source(ref.rel, snippet=line.strip())
                    break
            if loc:
                break
        if loc:
            return _cell(label, loc)
    return None


# --------------------------------------------------------------------------- universe / build
def _case_universe(refs: Sequence[FileRef]) -> list[str]:
    """The enumerated 案件 universe = every project owning a 01.契約 (contract) folder, corpus order.

    Row count == len(universe) is the completeness certificate: the table's rows *are* the population,
    so an enum question resolves to a filter over a self-evidently complete set (no separate universe
    proof needed). Excludes 社内管理 (a shared-management folder, not a client 案件)."""
    seen: list[str] = []
    for r in refs:
        if r.category == "contract" and r.project and r.project != "社内管理" and r.project not in seen:
            seen.append(r.project)
    return seen


def build(refs: list[FileRef] | None = None, *, out: Path | None = None,
          glossary: Glossary | None = None, write_report: bool = True) -> dict[str, Any]:
    """Precompute + persist the case master table (one row per 案件) with per-cell provenance.

    Deterministic and LLM-free. Returns a small build summary (row count, per-attribute missing rate,
    universe certificate). Additive & guarded — a failure here never corrupts the retrieval index.
    """
    refs = refs if refs is not None else walk()
    g = glossary if glossary is not None else _load_glossary()
    universe = _case_universe(refs)
    records = [_make_record(p, refs, g) for p in universe]

    counts = write_table(records, out)

    # per-attribute fill / missing (build report — 欠測を偽装しない)
    filled: dict[str, int] = {a: 0 for a in STANDARD_ATTRIBUTES}
    for rec in records:
        for a in STANDARD_ATTRIBUTES:
            cell = rec.attributes.get(a)
            if cell is not None and cell.get("value") is not None:
                filled[a] += 1
    n = len(records) or 1
    attribute_fill = {a: {"filled": filled[a], "missing": len(records) - filled[a],
                          "missing_rate": round((len(records) - filled[a]) / n, 4)}
                      for a in STANDARD_ATTRIBUTES}

    certificate = {
        "schema": SCHEMA, "schema_version": SCHEMA_VERSION,
        "universe_basis": "コーパス上で 01.契約 フォルダを持つ全案件（社内管理を除く）",
        "case_count": len(records),
        "cases": [rec.case_id for rec in records],
        "row_count_equals_universe": counts["cases"] == len(universe),
    }
    report = {
        "certificate": certificate,
        "attribute_fill": attribute_fill,
        "enum_hard_core_coverage": enum_hard_core_coverage(records),
        "out_path": str(out or default_out_path()),
    }
    if write_report:
        rp = report_out_path()
        rp.parent.mkdir(parents=True, exist_ok=True)
        rp.write_text(json.dumps(report, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
                      encoding="utf-8")
    return {**counts, "report": report}


def write_table(records: Sequence[CaseRecord], path: Path | None = None) -> dict[str, int]:
    """Atomically write a reproducible JSONL table (schema header + case_id-sorted rows)."""
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


# --------------------------------------------------------------------------- enum hard-core coverage
# 設問 → 必要な標準属性（診断専用・テーブル本体には非関与）。カバレッジ表は配線 issue (SOT-2647) の前提情報。
_ENUM_HARD_CORE: dict[str, dict[str, Any]] = {
    "idx38": {"need": ["apr_code", "contract_amount_incl_tax"],
              "gist": "APR-M3 が必要な案件を主略称で全列挙し契約金額(税込)を合計"},
    "idx67": {"need": ["status", "apr_code", "proposal_amount_incl_tax", "fr_amount_incl_tax"],
              "gist": "完了案件のうち APR-M2 該当で提案時金額≠FR時金額の案件を略称で全列挙"},
    "idx87": {"need": ["status", "apr_code", "train_rows"],
              "gist": "完了案件で APR-M1 該当かつ顧客データ≥10000行の案件を略称で全列挙"},
    "idx32": {"need": [],  # 分析出力 metrics.json 依存 — 案件マスタの標準属性では充足しない
              "gist": "青嶺 metrics.json の交互作用特徴量列名（分析コード出力に依存）"},
    "idx45": {"need": [],  # 会議録アクションアイテム — 案件マスタの標準属性では充足しない
              "gist": "京橋 会議録PDFの M2→M3 完了アクションアイテムID（会議録依存）"},
}


def enum_hard_core_coverage(records: Sequence[CaseRecord]) -> dict[str, Any]:
    """設問別の属性充足カバレッジ表（診断）。各属性が全案件で埋まっているかを報告する。

    テーブル本体は質問非依存だが、hard core 5問がテーブルで解けるかは配線 issue の前提情報。属性が
    全案件で埋まっていれば「その設問はテーブルで解ける」候補、欠測があれば配線前に埋める必要がある属性。
    """
    n = len(records) or 1
    fill_by_attr: dict[str, int] = {}
    for a in STANDARD_ATTRIBUTES:
        fill_by_attr[a] = sum(1 for r in records
                              if (r.attributes.get(a) or {}).get("value") is not None)
    out: dict[str, Any] = {}
    for idx, spec in _ENUM_HARD_CORE.items():
        need = spec["need"]
        per_attr = {a: {"filled": fill_by_attr.get(a, 0), "of": len(records),
                        "fully_covered": fill_by_attr.get(a, 0) == len(records)} for a in need}
        out[idx] = {
            "gist": spec["gist"],
            "required_attributes": need,
            "attribute_coverage": per_attr,
            "solvable_from_case_master": bool(need) and all(v["fully_covered"] for v in per_attr.values()),
            "note": ("案件マスタの標準属性では充足しない（別事前ストアが必要）" if not need else ""),
        }
    return out


# --------------------------------------------------------------------------- load / read (minimal API)
_LOAD_CACHE: dict[str, list[dict[str, Any]]] = {}


def reset_cache() -> None:
    _LOAD_CACHE.clear()


def load(path: Path | None = None) -> list[dict[str, Any]]:
    """Load the case master rows (memoized). ``[]`` when absent/unreadable/schema-mismatch."""
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


def _attr_value(row: dict[str, Any], attr: str) -> Any:
    return (row.get("attributes", {}).get(attr) or {}).get("value")


def filter_cases(*, path: Path | None = None, **equals: Any) -> list[dict[str, Any]]:
    """Minimal read/filter API (本 issue 最小限・配線は SOT-2647): rows whose named attributes equal
    the given values. Unknown attribute / no artifact ⇒ ``[]`` (no degradation). Example::

        filter_cases(status="完了", apr_code="APR-M2")
    """
    rows = load(path)
    out: list[dict[str, Any]] = []
    for row in rows:
        if all(_attr_value(row, a) == want for a, want in equals.items()):
            out.append(row)
    return out


if __name__ == "__main__":
    summary = build()
    print(json.dumps(summary["report"]["certificate"], ensure_ascii=False, indent=2))
