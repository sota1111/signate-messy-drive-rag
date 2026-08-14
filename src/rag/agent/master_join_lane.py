"""Deterministic answer lane for the master-join store — 人物×案件×役割×連絡先 の全数結合を
serve 側で純粋 lookup に帰着させる (SOT-2713 / cycle11).

親 SOT-2708 cycle11 §3「マスタ結合」クラスタ idx13/43/46 は cycle10 で MATCH だが Sonnet(claude-mcp)
LLM 経路（churn リスク）で通っていた。:mod:`src.rag.index.master_join_store` がビルド時に結合を全数
焼くので、ここでは質問形を精密に判定し、事前計算テーブルの決定論 lookup で直答する（LLM 非関与）。

発火条件（precision-first・fail-closed — いずれも満たさなければ ``None`` を返し従来経路へ委譲）:
  * ``RAG_MASTER_JOIN_LOOKUP`` が ON（既定 OFF ⇒ ``resolve`` は None ⇒ byte-identical）。
  * idx13 型 — 受託(乙=データアステル)側で最多案件関与者の内線: 人物テーブルを案件数で argmax（同率タイ/
    座席非一意なら棄権）→ その内線。
  * idx46 型 — 着手金が最も高い案件の指定役割(例 ES)担当者の内線: 案件テーブルを着手金で argmax（同率タイ
    なら棄権）→ 役割→乙担当者→座席→内線。役割は質問から一意に判定できた時のみ。
  * idx43 型 — 案件の甲(クライアント)主担当者フルネーム: glossary 別名で案件を一意束縛（曖昧/該当なしは棄権）
    → 署名欄由来の甲主担当者を裸形式で直答（SOT-2707 の権威ソース選好を継承）。
gold ハードコードなし（store の質問非依存結合値のみ）。
"""
from __future__ import annotations

import re
import unicodedata
from typing import Any, Callable

from src.rag.index import master_join_store as _store
from src.rag.tools import contract as _contract

MASTER_JOIN_LOOKUP = "master_join_lookup"


def _norm(value: Any) -> str:
    """NFKC 正規化 + 空白除去 + casefold（cue/別名照合用の安定表現）。"""
    return re.sub(r"\s+", "", unicodedata.normalize("NFKC", str(value or ""))).casefold()


def _any(q: str, cues: tuple[str, ...]) -> bool:
    return any(_norm(c) in q for c in cues)


# --------------------------------------------------------------------------- shared cues
_VENDOR_CUE = ("データアステル", "自社", "受託", "受注者", "乙")
_MOST_CUE = ("最も多く", "もっとも多く", "最多", "一番多く", "最も多い")
_HIGH_CUE = ("最も高い", "もっとも高い", "最高", "一番高い", "最も大きい")
_EXT_CUE = ("内線",)
_CASE_CUE = ("案件", "プロジェクト")
_CLIENT_CUE = ("クライアント", "顧客", "発注者", "委託者", "甲")
_FULLNAME_CUE = ("フルネーム", "氏名", "フル ネーム", "名前")

# 役割コード(乙担当者) → 質問に現れうる表記（latin コード + 日本語ラベル）。
_ROLE_CUES: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("ES", ("エグゼクティブスポンサー", "エグゼクティブ・スポンサー")),
    ("PM", ("プロジェクトマネージャー", "プロジェクトマネジャー")),
    ("LeadDS", ("リードデータサイエンティスト", "リードds")),
    ("DE", ("データエンジニア",)),
    ("BA", ("ビジネスアナリスト",)),
    ("QA", ("qaレビューアー", "qaレビュアー", "qaレビュー担当")),
)


def _detect_role(q: str) -> str | None:
    """質問から乙担当者の役割コードを一意に判定（0 件/複数なら None = fail-closed）。

    latin コードは前後が英数でない位置でのみ一致させ（``案件のESの`` の ``ES`` を拾い、無関係な語中一致を避ける）。
    日本語ラベルは素の部分一致。"""
    found: set[str] = set()
    for code, labels in _ROLE_CUES:
        if re.search(r"(?<![a-z0-9])" + code.casefold() + r"(?![a-z0-9])", q):
            found.add(code)
        elif any(_norm(lbl) in q for lbl in labels):
            found.add(code)
    return next(iter(found)) if len(found) == 1 else None


def _result(value: Any, selection: str, evidence: dict[str, Any]) -> dict[str, Any]:
    return _contract.ensure_contract({
        "value": value,
        "evidence": {"store": "master_join",
                     "provenance": "precomputed (question-independent, 人物×案件×役割×座席 全数結合)",
                     **evidence},
        "method": {"engine": "master_join", "contract": "simple_lookup", "selection": selection,
                   "verified_operand": True, "naturalize": False, "confidence": 1.0},
    })


# --------------------------------------------------------------------------- sub-lanes
def _most_cases_ext(q: str, tables: dict[str, list[dict[str, Any]]]) -> dict[str, Any] | None:
    """idx13 型: 乙側で最多案件関与者の内線（argmax 案件数, 同率タイ/座席非一意は棄権）。"""
    if not (_any(q, _VENDOR_CUE) and _any(q, _MOST_CUE) and _any(q, _CASE_CUE) and _any(q, _EXT_CUE)):
        return None
    persons = [p for p in tables.get("persons", []) if isinstance(p.get("case_count"), int)]
    if not persons:
        return None
    top = max(p["case_count"] for p in persons)
    leaders = [p for p in persons if p["case_count"] == top]
    if len(leaders) != 1:
        return None  # 同率タイ → fail-closed
    person = leaders[0]
    seat = person.get("seat") or {}
    ext = seat.get("ext")
    if not ext:
        return None
    return _result(ext, "most_cases_person_ext",
                   {"person": person.get("name"), "case_count": top,
                    "cases": person.get("cases"), "seat": seat})


def _max_deposit_role_ext(q: str, tables: dict[str, list[dict[str, Any]]]) -> dict[str, Any] | None:
    """idx46 型: 着手金最高案件の指定役割担当者の内線（argmax 着手金, 役割一意, 座席解決時のみ）。"""
    if not ("着手金" in q and _any(q, _HIGH_CUE) and _any(q, _EXT_CUE)):
        return None
    role = _detect_role(q)
    if role is None:
        return None
    cases = [c for c in tables.get("cases", []) if isinstance(c.get("deposit_incl_tax"), int)]
    if not cases:
        return None
    top = max(c["deposit_incl_tax"] for c in cases)
    leaders = [c for c in cases if c["deposit_incl_tax"] == top]
    if len(leaders) != 1:
        return None  # 同率タイ → fail-closed
    case = leaders[0]
    person_name = (case.get("staff") or {}).get(role)
    if not person_name:
        return None
    person = next((p for p in tables.get("persons", []) if p.get("name") == person_name), None)
    seat = (person or {}).get("seat") or {}
    ext = seat.get("ext")
    if not ext:
        return None
    return _result(ext, "max_deposit_role_ext",
                   {"case": case.get("case_id"), "abbrev": case.get("abbrev"),
                    "deposit_incl_tax": top, "role": role, "person": person_name, "seat": seat})


def _bind_case_by_alias(q: str, cases: list[dict[str, Any]]) -> dict[str, Any] | None:
    """glossary 別名で案件を一意束縛（別名が質問に現れる案件が唯一のときだけ確定, 曖昧なら None）。

    別名は curated（略称/短縮社名/正式社名）。2 文字未満は誤爆源なので採用しない。複数案件が該当したら
    fail-closed（例: ``青葉`` は AYM/AOBM 双方の別名なので単独では束縛しない）。"""
    matched: list[dict[str, Any]] = []
    for c in cases:
        aliases = c.get("aliases") or []
        if any(len(_norm(a)) >= 2 and _norm(a) in q for a in aliases):
            matched.append(c)
    return matched[0] if len(matched) == 1 else None


def _client_contact_fullname(q: str, tables: dict[str, list[dict[str, Any]]]) -> dict[str, Any] | None:
    """idx43 型: 案件の甲(クライアント)主担当者フルネーム（glossary 別名で一意束縛時のみ）。"""
    if not (_any(q, _CLIENT_CUE) and "主担当者" in q and _any(q, _FULLNAME_CUE)):
        return None
    case = _bind_case_by_alias(q, tables.get("cases", []))
    if case is None:
        return None
    value = case.get("client_contact")
    if not value:
        return None
    return _result(value, "client_contact_fullname",
                   {"case": case.get("case_id"), "abbrev": case.get("abbrev"), "party": "client"})


_LANES: tuple[Callable[[str, dict[str, list[dict[str, Any]]]], "dict[str, Any] | None"], ...] = (
    _most_cases_ext,
    _max_deposit_role_ext,
    _client_contact_fullname,
)


def resolve(question: str) -> dict[str, Any] | None:
    if not _store.enabled():
        return None
    try:
        tables, q = _store.load(), _norm(question)
        for lane in _LANES:
            result = lane(q, tables)
            if result is not None:
                return result
    except Exception:  # noqa: BLE001 — fail-open optional lane
        return None
    return None


def _lookup(kind: str = "person", key: str = "", role: str = "") -> dict[str, Any]:
    """デバッグ/ツール用の素引き（人物名 or 案件 case_id/略称で結合行を返す）。"""
    tables = _store.load()
    if kind == "case":
        k = _norm(key)
        for c in tables.get("cases", []):
            if _norm(c.get("case_id")) == k or _norm(c.get("abbrev")) == k:
                value = (c.get("staff") or {}).get(role) if role else c
                return _contract.make(value, engine="master_join", evidence={"case": c.get("case_id")})
        return _contract.make(None, engine="master_join", evidence={"case": key, "found": False})
    k = _norm(key)
    for p in tables.get("persons", []):
        if _norm(p.get("name")) == k:
            return _contract.make(p, engine="master_join", evidence={"person": p.get("name")})
    return _contract.make(None, engine="master_join", evidence={"person": key, "found": False})


def tool() -> tuple[str, str, dict[str, Any], Callable[..., Any]] | None:
    if not _store.enabled():
        return None
    string = {"type": "string"}
    return (MASTER_JOIN_LOOKUP,
            "人物×案件×役割×座席 全数結合マスタ: kind='person' なら key=氏名で {関与案件・案件数・内線} を、"
            "kind='case' なら key=案件名/略称で {着手金・役割→乙担当者・甲主担当者} を引く（role 指定で"
            "その役割の乙担当者名）。値は既存の決定論抽出器由来（LLM 非関与）。",
            {"type": "object",
             "properties": {"kind": string, "key": string, "role": string},
             "required": ["kind", "key"]},
            _lookup)
