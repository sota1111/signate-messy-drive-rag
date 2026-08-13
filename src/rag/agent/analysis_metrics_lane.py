"""SOT-2698 — 分析出力メタデータの決定論直答レーン（cycle9, idx32/61）.

cycle9 abstain の 2 件（両方 cycle7 MATCH → cycle8 BUDGET_EXHAUSTED のチャーン = LLM 経路分散）を、
質問非依存に焼いた事前計算ストアから **決定論直答レーン** として回収する:

* **idx32（enum_set）— metrics.json の数値交互作用特徴量列名**: :mod:`src.rag.index.analysis_metrics_enum_store`
  が焼いた per-case の ``interaction_columns``（``feature_selection.selected_columns`` のうち列名に
  ``__x__`` を含む交互作用特徴量）を ``、`` 連結で返す。青嶺不動産 = ``BOROUGH__x__BLOCK`` … の 6 列。
  cue = 「交互作用」＋（metrics.json / feature_selection / selected_columns / 特徴量）。
  フラグ ``RAG_ANALYSIS_METRICS_ENUM``（既定 OFF）。
* **idx61（config_hyperparam）— 実際に適用される勾配ブースティングのハイパラ**:
  :mod:`src.rag.index.raw_artifact_store` の per-case rollup ``applied_hyperparams.applied``（config の
  ``model_params`` ＋ ``modeling.py`` のコード上デフォルトをマージした適用値）から、質問が名指しした
  パラメタ名だけを **質問中の出現順** で ``名前=値`` 形式で返す。京橋 = ``n_estimators=300、
  learning_rate=0.1、random_state=42``。cue = パラメタ名 ＋（実際に渡される / 適用される / コード上 /
  実行時）。フラグ ``RAG_ANALYSIS_CONFIG_HYPERPARAM``（既定 OFF）。

規律（fact_layer / 他レーンと同じ）: 用語集 ``company_of`` で案件を一意束縛できた時だけ
``{value, evidence, method}`` を返し、束縛不能・フィールド欠落・空集合なら ``None`` で LLM 経路へ
フォールバック（回答数を減らさない・wrong を増やさない）。両フラグ既定 OFF ⇒ 両 sub-lane は None・
:func:`tool` は None（ツール集合/スキーマ/serve path は byte-identical）。serve 中の追加 LLM 呼び出しは 0。
"""
from __future__ import annotations

import os
import re
import unicodedata
from typing import Any, Callable

from src.rag.tools import contract as _contract

ANALYSIS_METRICS_LOOKUP = "analysis_metrics_lookup"
_STR = {"type": "string"}
_ON = {"1", "true", "yes", "on"}

# --- クエリ意図の cue（NFKC 正規化・空白除去・lower した質問文に対して判定） ---------------------
_INTERACTION_CUE = re.compile(r"交互作用|interaction")
_METRICS_CUE = re.compile(r"metrics\.json|feature_selection|selected_columns|特徴量|列名")
_HYPERPARAM_APPLIED_CUE = re.compile(
    r"実際に渡され|実際に適用|適用される|適用され|コード上|実行時|渡される値|渡される")


def metrics_enum_enabled() -> bool:
    return os.getenv("RAG_ANALYSIS_METRICS_ENUM", "0").strip().lower() in _ON


def config_hyperparam_enabled() -> bool:
    return os.getenv("RAG_ANALYSIS_CONFIG_HYPERPARAM", "0").strip().lower() in _ON


def enabled() -> bool:
    """いずれかの sub-lane が有効なら True（tool 公開判定に使う）。両 OFF ⇒ byte-identical。"""
    return metrics_enum_enabled() or config_hyperparam_enabled()


def _norm(text: Any) -> str:
    return unicodedata.normalize("NFKC", str(text or "")).replace(" ", "").replace("　", "").lower()


def _company_of(question: str) -> str | None:
    try:
        from src.rag.extract import glossary
        return glossary.load().company_of(question)
    except Exception:  # noqa: BLE001
        return None


def _result(value: Any, *, contract_type: str, selection: str, engine: str,
            evidence: dict[str, Any]) -> dict[str, Any]:
    ev = {"provenance": "precomputed (question-independent)", **evidence}
    method = {"engine": engine, "contract": contract_type, "selection": selection,
              "naturalize": False, "verified_operand": True, "confidence": 1.0}
    return _contract.ensure_contract({"value": value, "evidence": ev, "method": method})


# --------------------------------------------------------------------------- idx32 metrics enum
def _metrics_enum_lane(question: str):
    """idx32: metrics.json feature_selection の数値交互作用特徴量（``__x__`` 列）をすべて列挙。"""
    if not metrics_enum_enabled():
        return None
    q = _norm(question)
    if not (_INTERACTION_CUE.search(q) and _METRICS_CUE.search(q)):
        return None
    company = _company_of(question)
    if not company:
        return None
    try:
        from src.rag.index import analysis_metrics_enum_store as store
        rows = store.load()
    except Exception:  # noqa: BLE001 — 壊れたストアは fall back
        return None
    rec = next((r for r in rows if r.get("project") == company), None)
    if rec is None:
        return None
    interaction = rec.get("interaction_columns")
    if not interaction:  # None（フィールド欠落）・空リスト（該当なし）は確定しない → LLM へ
        return None
    value = "、".join(interaction)
    return _result(value, contract_type="enum_set", selection="metrics_interaction_columns",
                   engine="analysis_metrics_enum_store",
                   evidence={"store": "analysis_metrics_enum_store", "case": company,
                             "interaction_columns": interaction,
                             "selected_columns": rec.get("selected_columns"),
                             "metrics_rel": rec.get("metrics_rel"),
                             "marker": store.INTERACTION_MARKER})


# --------------------------------------------------------------------------- idx61 config hyperparam
# 質問中のパラメタ名を単語境界で拾う（``c`` 等の短キー誤爆を防ぐ; identifier 隣接は非マッチ）。
def _match_pos(key: str, qlower: str) -> int | None:
    m = re.search(r"(?<![0-9a-z_])" + re.escape(key.lower()) + r"(?![0-9a-z_])", qlower)
    return m.start() if m else None


def _fmt_value(value: Any) -> str:
    if isinstance(value, bool):
        return str(value)
    if isinstance(value, float):
        # 0.1 -> "0.1"（整数値の float は "1.0" のまま）。指数表記を避け冗長桁を出さない。
        return repr(value)
    return str(value)


def _config_hyperparam_lane(question: str):
    """idx61: 勾配ブースティング等に実際に適用されるハイパラを、質問が名指しした順に ``名前=値`` で返す。"""
    if not config_hyperparam_enabled():
        return None
    q = _norm(question)
    if not _HYPERPARAM_APPLIED_CUE.search(q):
        return None
    company = _company_of(question)
    if not company:
        return None
    try:
        from src.rag.index import raw_artifact_store as store
        data = store.load()
    except Exception:  # noqa: BLE001
        return None
    case = (data.get("cases_by_project") or {}).get(company)
    if not isinstance(case, dict):
        return None
    ah = case.get("applied_hyperparams") or {}
    applied = ah.get("applied")
    if not isinstance(applied, dict) or not applied:
        return None
    qlower = q  # 既に NFKC+lower+空白除去済み
    asked: list[tuple[int, str]] = []
    for key in applied:
        pos = _match_pos(str(key), qlower)
        if pos is not None:
            asked.append((pos, str(key)))
    if not asked:
        return None
    asked.sort(key=lambda kv: kv[0])
    parts = [f"{key}={_fmt_value(applied[key])}" for _pos, key in asked]
    value = "、".join(parts)
    return _result(value, contract_type="config_hyperparam", selection="applied_hyperparams",
                   engine="raw_artifact_store",
                   evidence={"store": "raw_artifact_store", "case": company,
                             "asked_params": [k for _p, k in asked],
                             "applied": {k: applied[k] for _p, k in asked},
                             "model_type": ah.get("model_type"),
                             "config_rel": ah.get("config_rel"),
                             "modeling_rel": ah.get("modeling_rel"),
                             "hyperparam_provenance": ah.get("provenance")})


# --------------------------------------------------------------------------- serve entry
_LANES = (_metrics_enum_lane, _config_hyperparam_lane)


def resolve(question: str) -> "dict[str, Any] | None":
    """分析出力メタデータの決定論直答（束縛できれば contract、曖昧なら None）。両 OFF なら常に None。"""
    if not enabled():
        return None
    try:
        for lane in _LANES:
            result = lane(question or "")
            if result is not None:
                break
        else:
            result = None
    except Exception:  # noqa: BLE001 — 壊れたレーンは fall back、答えパスを壊さない
        return None
    if result is None or not _contract.is_contract(result):
        return None
    normalized = _contract.ensure_contract(result)
    return normalized if normalized.get("value") is not None else None


# --------------------------------------------------------------------------- investigator tool (補助)
def _tool_handler(question: str = "") -> dict[str, Any]:
    res = resolve(question or "")
    if res is not None:
        return res
    return _contract.make(None, engine="analysis_metrics_lane",
                          evidence={"applicable": True, "bound": False},
                          note="案件束縛不能 or 対象フィールド欠落（従来経路へフォールバック）")


def tool() -> "tuple[str, str, dict[str, Any], Callable[..., Any]] | None":
    """両フラグ OFF ⇒ None ⇒ ツール集合/関数スキーマ/MCP surface は byte-identical。"""
    if not enabled():
        return None
    return (
        ANALYSIS_METRICS_LOOKUP,
        "分析出力メタデータ lookup: 質問（案件名を含む）を渡すと事前計算値を返す — "
        "(1) metrics.json feature_selection の数値交互作用特徴量列名（__x__ 列）、"
        "(2) 勾配ブースティング等に実際に適用されるハイパラ（config.model_params ＋ modeling.py の"
        "コード上デフォルトのマージ、n_estimators/learning_rate/random_state 等）。"
        "metrics.json の交互作用特徴量・適用ハイパラを file_grep 反復せず本ツールで引く。",
        {"type": "object", "properties": {"question": _STR}, "required": []},
        _tool_handler,
    )
