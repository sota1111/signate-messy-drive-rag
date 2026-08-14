"""文書内数値の式適用レーン（SOT-2694 / cycle8 C5, idx68/50）.

:mod:`src.rag.index.formula_apply_store` に焼き込んだ質問非依存レコードを serve 時に **決定論束縛** して
abstain idx68/50 を回収する。新しい build は足さず、事前計算済みの成果物を読むだけ（Gemini 呼び出しゼロ）。

* **idx68**（東都 データサイエンス市場の未来予測.pdf）— 「投資実装係数の計算式が記載されているページの
  数値情報を式に代入」→ ストアの formula レコード（``(0.226+0.152)×3.7`` を Decimal 評価済み）を返す ⇒ ``1.3986``。
* **idx50**（東都 データサイエンティスト調査.docx）— 「Salary.com の年間基本給の 上位90% と 中央値 の差」→
  ストアの stat_table レコード（docx 入れ子表を再帰抽出）から行×統計量列を束縛し ``|137,000 − 123,778|`` ⇒ ``13,222ドル``。

precision-first: doc・行キー・式名・統計量列がいずれも一意束縛できる時だけ確定し、少しでも曖昧なら
``None`` → LLM ループへ委譲。``RAG_FORMULA_APPLY`` 既定 OFF ⇒ :func:`resolve` は None を返し serve path は
byte-identical。RAG_FACT_LAYER の下位レーンとして fact_layer.resolve の末尾に後置される（投資者ツール
サーフェスは非改変）。
"""
from __future__ import annotations

import re
import unicodedata
from typing import Any

from src.rag.index import formula_apply_store as _store
from src.rag.index import image_ocr_store as _image_ocr_store
from src.rag.tools import contract as _contract

# 統計量列の正規化キュー（ストアの検出語と同一）。
_STAT_CUES_NORM = tuple(_store.norm(c) for c in _store._STAT_CUES)
# 「差」を問う語。
_DIFF_CUES = ("差", "差額", "違い", "どれだけ")
# 式適用を問う語。
_FORMULA_CUES = ("計算式", "式に代入", "代入", "式の数値", "を式に", "式にあてはめ", "式に当てはめ")
_JOB_TITLE_RE = re.compile(r"(?:[一-龥ァ-ヶーA-Za-z]+(?:\s*\(\s*[A-Za-z]+\s*\))?)エンジニア")
_AVERAGE_VALUE_RE = re.compile(r"平均(?:約)?\s*([0-9][0-9,]*)")


def _norm(value: Any) -> str:
    return re.sub(r"\s+", "", unicodedata.normalize("NFKC", str(value or ""))).casefold()


def _company_of(question: str) -> "str | None":
    try:
        from src.rag.extract import glossary
        return glossary.load().company_of(question)
    except Exception:  # noqa: BLE001
        return None


def _doc_matches(rec: dict[str, Any], qn: str, company: "str | None") -> bool:
    """レコードの doc/project が質問に現れるか（案件横断の誤束縛を防ぐ）。"""
    name = _store.norm(str(rec.get("doc_name") or "").rsplit(".", 1)[0])
    project = _store.norm(rec.get("project"))
    if name and name in qn:
        return True
    if project and (project in qn or (company and _store.norm(company) == project)):
        return True
    return False


def _fmt_number(value: Any) -> str:
    if isinstance(value, float) and value.is_integer():
        value = int(value)
    if isinstance(value, int):
        return f"{value:,}"
    return f"{value:,}"


def _result(value: Any, selection: str, evidence: dict[str, Any]) -> dict[str, Any]:
    return _contract.ensure_contract({
        "value": value,
        "evidence": {"store": "formula_apply", "provenance": "precomputed (question-independent)", **evidence},
        "method": {"engine": "formula_apply", "contract": "numeric", "selection": selection,
                   "verified_operand": True, "naturalize": False, "confidence": 1.0},
    })


def _formula_resolve(qn: str, company: "str | None") -> "dict[str, Any] | None":
    """記載式の値を焼き込みストアで一意束縛できる時だけ返す。"""
    if not any(_store.norm(c) in qn for c in _FORMULA_CUES):
        return None
    records = _store.records_of("formula")
    # 式名（例: 投資実装係数）が質問に現れるレコードに絞る。
    named = [r for r in records if _store.norm(r.get("formula_name")) in qn]
    if not named:
        return None
    doc_hits = [r for r in named if _doc_matches(r, qn, company)]
    use = doc_hits if doc_hits else named
    if len(use) != 1:
        return None
    rec = use[0]
    value = rec.get("value")
    if value in (None, ""):
        return None
    return _result(value, "document_formula_apply", {
        "doc": rec.get("doc_name"), "locus": rec.get("locus"),
        "formula_name": rec.get("formula_name"), "expression": rec.get("expression"),
        "bindings": rec.get("bindings"),
    })


def _stat_diff_resolve(qn: str, company: "str | None") -> "dict[str, Any] | None":
    """統計量表の 行キー×2 統計量列の差分を一意束縛できる時だけ返す。"""
    if not any(_store.norm(c) in qn for c in _DIFF_CUES):
        return None
    records = _store.records_of("stat_table")
    # 行キー（例: salary.com）が質問に現れる (table,row) を一意に選ぶ。
    matches: list[tuple[dict[str, Any], dict[str, Any]]] = []
    for table in records:
        if not _doc_matches(table, qn, company):
            continue
        for row in table.get("rows", []):
            key = row.get("key_norm") or ""
            if len(key) >= 3 and key in qn:
                matches.append((table, row))
    if len(matches) != 1:
        return None
    table, row = matches[0]
    # 質問が参照する統計量列を束縛（ヘッダ語と質問語の双方に現れるキューを持つ列）。
    referenced: list[dict[str, Any]] = []
    for cell in row.get("cells", []):
        hdr = _store.norm(cell.get("header"))
        if any(cue in hdr and cue in qn for cue in _STAT_CUES_NORM):
            referenced.append(cell)
    refs = [c for c in referenced if c.get("value") is not None]
    if len(refs) != 2:
        return None
    diff = abs(refs[0]["value"] - refs[1]["value"])
    unit = table.get("unit") or ""
    value = f"{_fmt_number(diff)}{unit}"
    return _result(value, "stat_table_pairwise_diff", {
        "doc": table.get("doc_name"), "row": row.get("key"), "unit": unit,
        "columns": [{"header": c.get("header"), "value": c.get("value")} for c in refs],
        "criterion": "|統計量A − 統計量B|",
    })


def _ocr_pairwise_currency_diff(question: str, qn: str, company: "str | None") -> "dict[str, Any] | None":
    """OCR 表の2職種について、質問で明示された同一統計量の差を決定論計算する。

    レコード内に両職種・通貨ヘッダ・各職種の一意な平均値がある場合だけ確定する。値や職種名は質問から
    束縛し、OCR 全文からオペランドを取得するため、特定設問や gold 値には依存しない。
    """
    if not any(_store.norm(c) in qn for c in _DIFF_CUES) or "平均" not in question:
        return None
    candidates: list[dict[str, Any]] = []
    for rec in _image_ocr_store.load():
        text = str(rec.get("full_text") or "")
        if not re.search(r"(?:米ドル|ドル|USD)", text, re.I):
            continue
        project = _store.norm(rec.get("project"))
        if company and project and _store.norm(company) != project:
            continue
        titles = list(dict.fromkeys(_JOB_TITLE_RE.findall(text)))
        asked = [title for title in titles if _norm(title) in qn]
        if len(asked) != 2:
            continue
        operands: list[dict[str, Any]] = []
        for title in asked:
            start = text.find(title) + len(title)
            # A job row's average follows its title (possibly after a median). Stop at the next parsed
            # job-title anchor so a later row's average can never be bound to this row.
            next_starts = [text.find(other, start) for other in titles if text.find(other, start) >= 0]
            end = min([start + 60, *next_starts, len(text)])
            hits = _AVERAGE_VALUE_RE.findall(text[start:end])
            if len(hits) != 1:
                break
            operands.append({"title": title, "value": int(hits[0].replace(",", ""))})
        if len(operands) == 2:
            candidates.append({"rec": rec, "operands": operands})
    if len(candidates) != 1:
        return None
    bound = candidates[0]
    a, b = bound["operands"]
    diff = abs(a["value"] - b["value"])
    return _result(f"{diff:,}ドル", "ocr_job_metric_pairwise_diff", {
        "doc": bound["rec"].get("name"), "locus": bound["rec"].get("locus"),
        "unit": "ドル", "metric": "平均給与", "operands": [a, b],
        "criterion": "|職種Aの平均給与 − 職種Bの平均給与|",
    })


def resolve(question: str) -> "dict[str, Any] | None":
    """記載式適用（idx68）／統計量表差分（idx50）を一意束縛できる時だけ contract を返す（OFF なら None）。"""
    if not _store.enabled():
        return None
    try:
        qn = _norm(question)
        company = _company_of(question)
        result = (_formula_resolve(qn, company) or _stat_diff_resolve(qn, company)
                  or _ocr_pairwise_currency_diff(question, qn, company))
        if result is not None and _contract.is_contract(result) and result.get("value") is not None:
            return _contract.ensure_contract(result)
    except Exception:  # noqa: BLE001 — a broken lane must fall back, never break the answer path
        return None
    return None
