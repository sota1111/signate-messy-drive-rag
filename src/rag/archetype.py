"""Question archetype classification.

The self-improvement harness (`scoring/selfimprove.py`) measures how reliably the RAG answers
each *archetype* of question (numeric config value, dataset metric, glossary lookup, …) using a
deterministic scorer, and records which archetypes are trustworthy in `config/archetype_trust.json`.

`generate.py` classifies each incoming question with `classify()` and, when its archetype was
measured as low-precision, abstains instead of guessing (a strictly *additive* guard: it only ever
turns a would-be answer into an abstention, never the reverse, so it cannot raise the Incorrect rate).

Classification is deliberately conservative — an ambiguous question returns ``"unknown"`` so the
existing self-consistency / verify gates keep full control. Each archetype declares the deterministic
comparator (`kind`) the scorer should use for it.
"""
from __future__ import annotations

import re

from src.rag import pivotcond
from src.rag.corpus import nfc

# archetype -> deterministic comparator kind (numeric / set / string), see scoring.deterministic
ARCHETYPE_KIND: dict[str, str] = {
    "enum_set": "set",
    "highlight_set": "set",
    "cross_aggregate": "numeric",
    "pivot_condition": "string",
    "config_model_type": "string",
    "config_hyperparam": "numeric",
    "metric_score": "numeric",
    "data_shape": "numeric",
    "csv_column_mean": "numeric",
    "csv_column_max": "numeric",
    "glossary_formal": "string",
    "glossary_abbrev": "string",
    "version_diff": "string",
}

# Ordered, most-specific-first. First matching pattern wins.
_RULES: list[tuple[str, re.Pattern[str]]] = [
    ("highlight_set", re.compile(r"(マーカー|ハイライト|強調).*(すべて|全て).*(抜き出|挙げ|答え)")),
    ("enum_set", re.compile(r"(すべて|全て).*(挙げ|答え|抜き出)|すべて挙げ")),
    ("cross_aggregate", re.compile(r"(総額を計算|いくら少なく|平均を算出|条件.{0,30}(平均|最大|件数))")),
    # A version-diff question: an explicit OLDER-version marker + a comparison/change verb. Kept in
    # sync with diffpair.is_diff_question so classification and routing agree.
    ("version_diff", re.compile(
        r"(旧版|旧バージョン|旧ファイル|old版|oldバージョン|oldフォルダ|old\s*版|以前の版|前の版|"
        r"前バージョン|前回版|古い版)[\s\S]*(比較|変更|差分|相違|変わった|変更前|前と後|前後|どう違)")),
    ("glossary_abbrev", re.compile(r"用語集.*略称|略称.*何|の略称は")),
    ("glossary_formal", re.compile(r"用語集.*正式名称|正式名称は|は何の略|の正式(な)?名称")),
    ("config_model_type", re.compile(r"(model_type|モデル(の)?種類|どのモデル|モデルタイプ)")),
    ("config_hyperparam", re.compile(r"(random_state|test_size|乱数(シード|種)|テストサイズ)")),
    ("metric_score", re.compile(
        r"(accuracy|f1|f1_macro|auc|auc_roc|roc|brier|precision_at|"
        r"正解率|精度スコア|評価指標.*値|スコアの値|のスコアは)")),
    ("data_shape", re.compile(
        r"(row_count|feature_count|train_rows|test_rows|行数|レコード数|列数|"
        r"特徴量(数|の数)|サンプル数|何行)")),
    ("csv_column_mean", re.compile(r"(列|カラム|column).{0,12}(平均|mean)|(平均|mean).{0,12}(列|カラム)")),
    ("csv_column_max", re.compile(r"(列|カラム|column).{0,12}(最大|max)|(最大値|最大の値).{0,12}(列|カラム)")),
]


def classify(question: str) -> str:
    """Return the archetype name for a question, or ``"unknown"`` when none applies."""
    q = nfc(question)
    # PivotTable / AutoFilter condition questions route to the deterministic pivotcond module; keep
    # detection identical to it (like diffpair/version_diff) so classification and routing agree. A
    # version-diff question is never a pivot-condition question, so this precedes the rule table.
    if pivotcond.is_pivot_condition_question(q):
        return "pivot_condition"
    for name, pat in _RULES:
        if pat.search(q):
            return name
    return "unknown"


def kind_of(archetype: str) -> str:
    """Deterministic comparator kind for an archetype (defaults to string)."""
    return ARCHETYPE_KIND.get(archetype, "string")
