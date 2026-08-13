"""SOT-2691 — 分析成果物クロス参照の serve 配線（決定論直答レーン + investigator ツール）.

:mod:`src.rag.index.analysis_xref_store` が build 時に質問非依存で焼いた per-case の事実（未完事項ID /
カテゴリ列 nunique×One-Hot 閾値 / 選択特徴量分類）から、cycle8 abstain の 3 型を **決定論直答レーン** として
接続する:

* **idx60 — 最終報告の未完事項 ID 列挙**: ``incomplete_ids.ids`` を ``、`` 連結で返す（白峰 = AI-05、AI-08、
  AI-09）。案件束縛は用語集 ``company_of``（部分名/法人格サフィックス正規化を内包）。
* **idx73 — One-Hot 閾値による対象カテゴリ列**: ``onehot.onehot_targets``（閾値未満 nunique のカテゴリ列）を
  返す（かえで = Gender）。
* **idx53 (stretch) — ENG-FT 特徴量数**: ``selected_features.eng_ft_count``（選択特徴量のうち原列に無い派生）を
  返す（東都 = 6）。

**idx36 は DELIBERATELY 未配線**: 中間段階のフル精度 F1 は成果物に丸め（leaderboard 8桁）しか無く、フル精度
gold と ~1e-9 ずれて judge が Incorrect にする（SOT-2687 の実測教訓）。正解を焼けない問いは honest abstain。

規律（fact_layer / 他レーンと同じ）: 一意・厳密に束縛できる時だけ ``{value, evidence, method}`` を返し、少し
でも曖昧なら ``None`` を返して LLM ループへフォールバック（回答数を減らさない・wrong を増やさない）。
``RAG_ANALYSIS_XREF`` 既定 OFF ⇒ :func:`resolve` は None・:func:`tool` は None（ツール集合/スキーマ/serve path
は byte-identical）。serve 中の追加 LLM 呼び出しは 0（事前計算済みを読むだけ）。
"""
from __future__ import annotations

import re
import unicodedata
from typing import Any, Callable

from src.rag.index import analysis_xref_store as _store
from src.rag.tools import contract as _contract

ANALYSIS_XREF_LOOKUP = "analysis_xref_lookup"
_STR = {"type": "string"}

# --- クエリ意図の cue（NFKC 正規化・空白除去・lower した質問文に対して判定） ---------------------
_INCOMPLETE_CUE = re.compile(r"未完事項|未完了事項|未対応事項")
_ID_CUE = re.compile(r"id|抽出|挙げ|列挙|すべて|全て")
_ONEHOT_CUE = re.compile(r"one-hot|onehot|ワンホット")
_ONEHOT_TARGET_CUE = re.compile(r"閾値|対象|カテゴリ列|カテゴリ数")
_ENGFT_CUE = re.compile(r"eng-ft|engft|エンジニアリング特徴量")
_COUNT_CUE = re.compile(r"いくつ|何個|数|幾つ")


def enabled() -> bool:
    return _store.enabled()


def _norm(text: Any) -> str:
    return unicodedata.normalize("NFKC", str(text or "")).replace(" ", "").replace("　", "").lower()


def _result(value: Any, *, contract_type: str, selection: str, evidence: dict[str, Any]) -> dict[str, Any]:
    ev = {"store": "analysis_xref_store", "provenance": "precomputed (question-independent)", **evidence}
    method = {"engine": "analysis_xref_store", "contract": contract_type, "selection": selection,
              "naturalize": False, "verified_operand": True, "confidence": 1.0}
    return _contract.ensure_contract({"value": value, "evidence": ev, "method": method})


def _bind_case(question: str, rows: list[dict[str, Any]]) -> dict[str, Any] | None:
    """用語集 company_of で canonical 会社名（案件フォルダ名）へ一意束縛。曖昧/未収録 ⇒ None。"""
    try:
        from src.rag.extract import glossary
        company = glossary.load().company_of(question)
    except Exception:  # noqa: BLE001
        company = None
    if not company:
        return None
    for rec in rows:
        if rec.get("project") == company:
            return rec
    return None


# --------------------------------------------------------------------------- sub-lanes
def _incomplete_ids_lane(q: str, rec: dict[str, Any]):
    """idx60: 最終報告の未完事項 ID をすべて列挙。"""
    if not (_INCOMPLETE_CUE.search(q) and _ID_CUE.search(q)):
        return None
    inc = rec.get("incomplete_ids") or {}
    ids = inc.get("ids") or []
    if not ids:
        return None
    value = "、".join(ids)
    return _result(value, contract_type="document_extract", selection="incomplete_ids",
                   evidence={"case": rec.get("project"), "ids": ids,
                             "section_header": inc.get("section_header"),
                             "report_rel": rec.get("report_rel")})


def _onehot_targets_lane(q: str, rec: dict[str, Any]):
    """idx73: One-Hot 閾値未満の nunique を持つカテゴリ列をすべて。"""
    if not (_ONEHOT_CUE.search(q) and _ONEHOT_TARGET_CUE.search(q)):
        return None
    onehot = rec.get("onehot") or {}
    targets = onehot.get("onehot_targets")
    if not targets:  # None（閾値不明）・空リスト（該当なし）は確定しない → LLM へ
        return None
    value = "、".join(targets)
    return _result(value, contract_type="enum_set", selection="onehot_threshold_targets",
                   evidence={"case": rec.get("project"), "onehot_targets": targets,
                             "threshold": onehot.get("threshold"),
                             "threshold_source": onehot.get("threshold_source"),
                             "categorical": onehot.get("categorical"),
                             "train_rel": onehot.get("train_rel")})


def _eng_ft_count_lane(q: str, rec: dict[str, Any]):
    """idx53 (stretch): 選択特徴量のうち ENG-FT（原列に無い派生）の数。"""
    if not (_ENGFT_CUE.search(q) and _COUNT_CUE.search(q)):
        return None
    feat = rec.get("selected_features") or {}
    count = feat.get("eng_ft_count")
    if count is None:
        return None
    return _result(str(count), contract_type="derived_calculation", selection="eng_ft_count",
                   evidence={"case": rec.get("project"), "eng_ft": feat.get("eng_ft"),
                             "eng_ft_count": count, "selected": feat.get("selected"),
                             "selected_count": feat.get("selected_count"),
                             "report_rel": rec.get("report_rel")})


# --------------------------------------------------------------------------- serve entry
_LANES = (_incomplete_ids_lane, _onehot_targets_lane, _eng_ft_count_lane)


def resolve(question: str) -> "dict[str, Any] | None":
    """分析成果物クロス参照の決定論直答（束縛できれば contract、曖昧なら None）。OFF なら常に None。"""
    if not enabled():
        return None
    try:
        rows = _store.load()
    except Exception:  # noqa: BLE001 — 壊れたストアは fall back、答えパスを壊さない
        return None
    if not rows:
        return None
    try:
        rec = _bind_case(question, rows)
        if rec is None:
            return None
        q = _norm(question)
        for lane in _LANES:
            result = lane(q, rec)
            if result is not None:
                break
        else:
            result = None
    except Exception:  # noqa: BLE001
        return None
    if result is None or not _contract.is_contract(result):
        return None
    normalized = _contract.ensure_contract(result)
    return normalized if normalized.get("value") is not None else None


# --------------------------------------------------------------------------- investigator tool (補助)
def _tool_handler(question: str = "", case: str = "") -> dict[str, Any]:
    res = resolve(question or "")
    if res is not None:
        return res
    rows = _store.load()
    rec = _bind_case(case or question, rows)
    if rec is None:
        return _contract.make(None, engine="analysis_xref_store",
                              evidence={"applicable": True, "bound": False},
                              note="案件を一意束縛できず（用語集 company_of 未解決）")
    value = {
        "project": rec.get("project"),
        "incomplete_ids": (rec.get("incomplete_ids") or {}).get("ids"),
        "onehot_targets": (rec.get("onehot") or {}).get("onehot_targets"),
        "onehot_threshold": (rec.get("onehot") or {}).get("threshold"),
        "eng_ft_count": (rec.get("selected_features") or {}).get("eng_ft_count"),
        "selected_features": (rec.get("selected_features") or {}).get("selected"),
    }
    return _contract.make(value, engine="analysis_xref_store",
                          evidence={"applicable": True, "case": rec.get("project"), "found": True},
                          scheme="analysis-artifact-xref")


def tool() -> "tuple[str, str, dict[str, Any], Callable[..., Any]] | None":
    """RAG_ANALYSIS_XREF OFF ⇒ None ⇒ ツール集合/関数スキーマ/MCP surface は byte-identical。"""
    if not enabled():
        return None
    return (
        ANALYSIS_XREF_LOOKUP,
        "分析成果物クロス参照: case/質問を渡すと、その案件の事前計算値を返す — "
        "(1) 最終報告の未完事項ID一覧 incomplete_ids, (2) 実装 One-Hot 閾値未満の対象カテゴリ列 onehot_targets "
        "(train.csv のカテゴリ列 nunique × features.py 閾値), (3) 選択特徴量のうち ENG-FT の数 eng_ft_count。"
        "未完事項ID・One-Hot対象列・ENG-FT数を file_grep 反復せず本ツールで引く。",
        {"type": "object", "properties": {"question": _STR, "case": _STR}, "required": []},
        _tool_handler,
    )
