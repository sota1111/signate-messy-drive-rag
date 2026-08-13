"""Deterministic answer lane for the contact-master store — 契約書署名欄を権威ソースとする
主担当者/役職/部署 の完全ラベル直答 (SOT-2707 / cycle10).

wrong idx21（青葉バイオメディカル機器のクライアント主担当者の役職 → gold=人材戦略部長）を回収する。
cycle9 実測は会議録の出席者展開形『人事本部 人材戦略部 部長』を写経して Incorrect だった。会議録
（記述文脈）より契約書署名欄／当事者ブロックの構造化フィールド『役職：』が権威ソースである（SOT-2666
完全ラベル写経の権威ソース選好版）。

発火条件（precision-first・fail-closed）:
  * ``RAG_CONTACT_MASTER`` が ON（既定 OFF ⇒ ``resolve`` は None ⇒ byte-identical）。
  * 質問が store 全レコードの案件を **一意** に束縛できる（短縮名の誤爆を避けるため 4 文字以上一致のみ）。
  * 質問が甲（クライアント）／乙（受託）どちらの当事者かを **明示** している。
  * 該当フィールド（役職／主担当者／部署名）が store に **一意** に存在する。
上記を満たさなければ ``None`` を返し従来経路へ委譲（無理な回答化をしない）。値は原文ラベルを完全写経
（正規化・言い換えなし）で裸形式に返す。gold ハードコードなし（store の質問非依存抽出値のみ）。
"""
from __future__ import annotations

import re
import unicodedata
from typing import Any, Callable

from src.rag.index import contact_master_store as _store
from src.rag.tools import contract as _contract

CONTACT_MASTER_LOOKUP = "contact_master_lookup"


def _norm(value: Any) -> str:
    return re.sub(r"\s+", "", unicodedata.normalize("NFKC", str(value or ""))).casefold()


def _cell(row: dict[str, Any], key: str) -> dict[str, Any]:
    return (row.get("operands") or {}).get(key) or {}


# 会社名の法人格接頭辞（質問は「青葉バイオメディカル機器」のように接頭辞を落として案件名を呼ぶ）。
_CORP_PREFIX = re.compile(r"^(株式会社|医療法人社団|医療法人|一般社団法人|公益社団法人|有限会社|合同会社)\s*")


def _case_tokens(case_id: str) -> list[str]:
    """案件バインド用の候補トークン: 正規化 case_id 全体・空白分割トークン・法人格接頭辞を除いた社名。"""
    toks: list[str] = []
    cid = _norm(case_id)
    if cid:
        toks.append(cid)
    for tok in re.split(r"\s+", str(case_id or "")):
        tn = _norm(tok)
        if tn:
            toks.append(tn)
    stem = _norm(_CORP_PREFIX.sub("", str(case_id or "")))
    if stem:
        toks.append(stem)
    return toks


def _bind_case(q: str, rows: list[dict[str, Any]]) -> dict[str, Any] | None:
    """質問非依存の案件バインド: store 全レコードの case_id/略称が質問に現れる行を一意に選ぶ。

    最長一致（正規化 case_id 全体・空白分割トークン・法人格接頭辞除去社名・略称）でスコアし、最長が複数行で
    並ぶ（曖昧）ときは None（fail-closed）。短すぎるトークン（<4）は誤爆源なので採用しない。"""
    best: dict[str, Any] | None = None
    best_len = 0
    tie = False
    for r in rows:
        score = 0
        for tok in _case_tokens(r.get("case_id")):
            if len(tok) >= 4 and tok in q:
                score = max(score, len(tok))
        ab = _norm(r.get("abbrev"))
        if len(ab) >= 4 and ab in q:
            score = max(score, len(ab))
        if score > best_len:
            best, best_len, tie = r, score, False
        elif score == best_len and score > 0:
            tie = True
    return None if best is None or tie else best


# 甲＝クライアント（発注者）/ 乙＝受託（自社・データアステル）。両方/どちらも無し ⇒ 曖昧 ⇒ 委譲。
_CLIENT_CUE = ("クライアント", "顧客", "発注者", "委託者", "甲")
_VENDOR_CUE = ("受託者", "受注者", "自社", "乙", "データアステル")


def _party(q: str) -> str | None:
    client = any(_norm(c) in q for c in _CLIENT_CUE)
    vendor = any(_norm(v) in q for v in _VENDOR_CUE)
    if client == vendor:  # both or neither ⇒ ambiguous
        return None
    return "client" if client else "vendor"


def _result(value: Any, selection: str, evidence: dict[str, Any]) -> dict[str, Any]:
    return _contract.ensure_contract({
        "value": value,
        "evidence": {"store": "contact_master",
                     "provenance": "precomputed (question-independent, 契約書署名欄 完全写経)", **evidence},
        "method": {"engine": "contact_master", "contract": "simple_lookup", "selection": selection,
                   "verified_operand": True, "naturalize": False, "confidence": 1.0},
    })


# フィールド判定は precision-first に順序付ける: 役職 > 主担当者(氏名) > 部署名。
def _field_for(q: str) -> tuple[str, str] | None:
    """(operand-suffix, selection) for the queried structured field, or None when not a target shape."""
    if "役職" in q:
        return "role", "party_contact_role"
    if "主担当者" in q or "担当者" in q:
        return "contact", "party_contact_name"
    if "部署" in q:
        return "department", "party_contact_department"
    return None


def resolve(question: str) -> dict[str, Any] | None:
    if not _store.enabled():
        return None
    try:
        rows, q = _store.load(), _norm(question)
        field = _field_for(q)
        if field is None:
            return None
        party = _party(q)
        if party is None:
            return None
        rec = _bind_case(q, rows)
        if rec is None:
            return None
        suffix, selection = field
        op = f"{party}_{suffix}"
        cell = _cell(rec, op)
        value = cell.get("value")
        if not value:
            return None
        return _result(value, selection,
                       {"case": rec.get("case_id"), "party": party, "operand": op,
                        "source": cell.get("source")})
    except Exception:  # noqa: BLE001 — fail-open optional lane
        return None


def _lookup(case: str, party: str = "client", attribute: str = "") -> dict[str, Any]:
    rec = _store.case_record(case)
    if rec is None:
        return _contract.make(None, engine="contact_master", evidence={"case": case, "found": False})
    if attribute:
        op = attribute if attribute.startswith(("client_", "vendor_")) else f"{party}_{attribute}"
        cell = _cell(rec, op)
        value = cell.get("value")
        return _contract.make(value, engine="contact_master",
                              evidence={"case": rec.get("case_id"), "operand": op,
                                        "source": cell.get("source")})
    value = {"case_id": rec.get("case_id"), "abbrev": rec.get("abbrev"),
             "available": sorted((rec.get("operands") or {}).keys())}
    return _contract.make(value, engine="contact_master",
                          evidence={"case": rec.get("case_id")})


def tool() -> tuple[str, str, dict[str, Any], Callable[..., Any]] | None:
    if not _store.enabled():
        return None
    string = {"type": "string"}
    return (CONTACT_MASTER_LOOKUP,
            "担当者マスタ（契約書署名欄 権威ソース）: 案件の当事者ブロックの会社名/部署名/主担当者/役職を"
            "甲(クライアント)・乙(受託)別に出典付きで引く。attribute に 'client_role'/'role' 等、party に "
            "'client'|'vendor' を渡す（役職・主担当者は原文ラベルを完全写経）。",
            {"type": "object",
             "properties": {"case": string, "party": string, "attribute": string},
             "required": ["case"]},
            _lookup)
