"""SOT-2493 — question-contract classifier (Adaptive-RAG routing の前段).

Parent SOT-2460. Before a question is turned into a retrieval query, this module first classifies it
into a **question contract** — the *kind of promise* the answer must keep — and hands back the
contract's **completion-condition template** and a suggested processing route. Downstream Issues wire
the contract into the investigator loop (SOT-2498) and use the completion conditions as evidence
obligations (SOT-2499) / enumeration closure gates (SOT-2500); **this Issue only adds the classifier —
it does not touch the production answer path.**

The nine contracts
------------------
``simple_lookup`` (単純検索) · ``multi_hop`` (多段関係) · ``cross_aggregate`` (横断集計) ·
``full_enumeration`` (完全列挙) · ``format_check`` (書式判定) · ``chart_read`` (グラフ読取) ·
``spatial`` (空間推論) · ``version_diff`` (版差分) · ``numeric`` (数値推論).

Hybrid, deterministic-first
---------------------------
:func:`classify` runs a **deterministic** layer first (keyword / regex rules + glossary-abbrev aware
:func:`src.rag.archetype.classify` as the coarse backbone) and only falls back to ``gemini-2.5-flash``
when the question is genuinely ambiguous *and* a ``flash`` callable is supplied. The flash call is
injected (never imported at module load, never invoked on a confident deterministic hit), so the whole
classifier — including the accuracy measurement — runs network-free and reproducibly.

Relation to the archetype taxonomy
----------------------------------
``src.rag.archetype.classify`` already labels each question with a fine-grained *archetype*
(``fact_lookup`` / ``document_extract`` / ``derived_calculation`` / ``enum_set`` / ``version_diff`` …),
and the gold-100 ``archetype`` column (``artifacts/gold_100_review.csv``) is produced by exactly that
function (``scoring/gold_offline.py``). The contract taxonomy is a **routing refinement** on top of it:
several contracts (``spatial`` / ``chart_read`` / ``format_check``) are strictly more specific than the
archetype they refine (a seating question is a specialised ``fact_lookup``; a chart-value question is a
``fact_lookup`` / ``derived_calculation``; a bold/underline extraction is a ``document_extract``). So
each contract declares the set of archetypes it is *consistent* with (:data:`CONTRACT_ARCHETYPES`), and
:func:`agreement_rate` measures how often the predicted contract is consistent with the gold archetype
— the acceptance metric (≥90%), with the specificity gains recorded as principled refinements rather
than counted as errors.
"""
from __future__ import annotations

import ast
import re
from dataclasses import dataclass, field
from typing import Any, Callable, Mapping, Sequence

from config import settings
from src.rag import archetype as _archetype
from src.rag.corpus import nfc

# --------------------------------------------------------------------------- the nine contracts
SIMPLE_LOOKUP = "simple_lookup"        # 単純検索: 単一の権威的記載を引く
MULTI_HOP = "multi_hop"                # 多段関係: 中間エンティティを段階的に辿る
CROSS_AGGREGATE = "cross_aggregate"    # 横断集計: 複数文書/行を横断して集計する
FULL_ENUMERATION = "full_enumeration"  # 完全列挙: 母集団を確定し漏れなく挙げる
FORMAT_CHECK = "format_check"          # 書式判定: 太字/下線/色などの書式属性で抽出する
CHART_READ = "chart_read"              # グラフ読取: 図/グラフから数値を読み取る
SPATIAL = "spatial"                    # 空間推論: 座席表など空間関係を解く
VERSION_DIFF = "version_diff"          # 版差分: 旧版と新版の実質的変更を出す
NUMERIC = "numeric"                    # 数値推論: 型付き入力から式で導出する

CONTRACTS: tuple[str, ...] = (
    SIMPLE_LOOKUP, MULTI_HOP, CROSS_AGGREGATE, FULL_ENUMERATION, FORMAT_CHECK,
    CHART_READ, SPATIAL, VERSION_DIFF, NUMERIC,
)

# Human-readable JP label per contract (for reports / prompts).
CONTRACT_LABELS: dict[str, str] = {
    SIMPLE_LOOKUP: "単純検索", MULTI_HOP: "多段関係", CROSS_AGGREGATE: "横断集計",
    FULL_ENUMERATION: "完全列挙", FORMAT_CHECK: "書式判定", CHART_READ: "グラフ読取",
    SPATIAL: "空間推論", VERSION_DIFF: "版差分", NUMERIC: "数値推論",
}

# --------------------------------------------------------------------------- completion-condition templates
# Per-contract "what must be true before an answer may be committed" — the routing payload downstream
# Issues (SOT-2498/2499/2500) consume. Kept as ordered tuples so they read as a checklist.
CONTRACT_COMPLETION: dict[str, tuple[str, ...]] = {
    SIMPLE_LOOKUP: (
        "回答値が単一の権威的文書に明示されている",
        "値の出典(ファイル/スライド/セル)を一意に特定できる",
    ),
    MULTI_HOP: (
        "中間エンティティ(担当者/案件/期間など)を一段ずつ根拠付きで確定する",
        "各ホップの根拠を保持し、最終値までの連鎖が途切れない",
    ),
    CROSS_AGGREGATE: (
        "集計対象の母集団(全プロジェクト/全ファイル)を権威的に確定する",
        "集計関数(合計/平均/最大/件数)と対象列が型整合している",
        "決定論的に再計算できる(corpus_aggregate 等で再現可能)",
    ),
    FULL_ENUMERATION: (
        "列挙対象の母集団(全体集合)を権威的に確定する",
        "閉包条件を満たす(見落とし 0・重複 0 を保証できる)",
    ),
    FORMAT_CHECK: (
        "判定対象の書式属性(太字/下線/色/フォント)を原本から機械抽出する",
        "該当箇所を網羅し、非該当(例:日付)を除外できる",
        "表/ピボットの場合は生の行・列ラベルを元データのフィールド名へ意味解決し、抽出条件・対象列・集計方法を確定する",
    ),
    CHART_READ: (
        "対象グラフ(図番号/シート/系列)を一意に特定する",
        "軸・系列・座標を読み取り、値を再現可能な形で得る",
        "読取値の桁数/単位が質問の要求と一致する",
    ),
    SPATIAL: (
        "座席表/レイアウトの空間関係(隣/向かい/座席番号)を確定する",
        "人物⇔座席⇔内線などの対応を一意に解決できる",
    ),
    VERSION_DIFF: (
        "旧版と新版の対応ペアを一意に確定する",
        "隣接版間の実質的な変更のみを差分として抽出する",
    ),
    NUMERIC: (
        "型付き入力(数値/単位/対象列)を根拠から確定する",
        "質問が要求する量を分子・分母・母集団まで定義し、『〜のうち』の直前にある条件を分母から落とさない",
        "再実行可能な式で決定論的に導出できる",
        "計算結果の単位と丸め(小数第N位など)が質問の要求と一致する",
    ),
}

# Suggested processing route (tool / path hint) per contract — advisory only until SOT-2498 wires it.
CONTRACT_ROUTE: dict[str, str] = {
    SIMPLE_LOOKUP: "retrieve→extract",
    MULTI_HOP: "obligation-driven iterative retrieve",
    CROSS_AGGREGATE: "corpus_aggregate",
    FULL_ENUMERATION: "enumeration-closure",
    FORMAT_CHECK: "pdf_emphasis/highlight_extract",
    CHART_READ: "read_chart_values",
    SPATIAL: "seating_lookup",
    VERSION_DIFF: "version_diff",
    NUMERIC: "compute",
}

# Archetypes each contract is *consistent* with (a contract may refine, i.e. be strictly more specific
# than, the coarse archetype). Used by :func:`agreement_rate` to score the classifier against the gold
# archetype column without penalising a legitimate specificity gain. ``unknown`` is always compatible.
CONTRACT_ARCHETYPES: dict[str, frozenset[str]] = {
    SIMPLE_LOOKUP: frozenset({
        "fact_lookup", "document_extract", "glossary_formal", "glossary_abbrev",
        "config_model_type", "config_hyperparam", "metric_score",
    }),
    MULTI_HOP: frozenset({"fact_lookup", "derived_calculation"}),
    CROSS_AGGREGATE: frozenset({"cross_aggregate", "pivot_condition", "derived_calculation"}),
    FULL_ENUMERATION: frozenset({"enum_set", "highlight_set"}),
    FORMAT_CHECK: frozenset({"document_extract", "highlight_set", "fact_lookup"}),
    CHART_READ: frozenset({"fact_lookup", "derived_calculation", "data_shape"}),
    SPATIAL: frozenset({"fact_lookup"}),
    VERSION_DIFF: frozenset({"version_diff"}),
    NUMERIC: frozenset({
        "derived_calculation", "data_shape", "csv_column_mean", "csv_column_max",
        "contract_amount", "cross_aggregate",
    }),
}

# --------------------------------------------------------------------------- deterministic rules
# New-dimension detectors — the contracts that are strictly more specific than any archetype the coarse
# backbone can express. Kept tight so they fire ONLY on genuine chart/spatial/format questions.
# Geometric seating cues (隣/向かい/座席): unambiguously spatial, checked before multi-hop.
_SPATIAL_GEOM_RE = re.compile(
    r"座席表|座席|着席|席次|座って|隣に座|の隣の|左隣|右隣|向かい(に座|の席|の方|の人)|"
    r"レイアウト図|配置図")
# A bare extension/EXT directory lookup (氏名⇔EXT) also routes through the seating directory, but only
# when the question is NOT a multi-hop ("最も…な人の内線") — those are resolved as multi_hop first.
_SPATIAL_DIR_RE = re.compile(r"内線番号を教え|内線を教え|ext を教え|ext を答え", re.I)
_CHART_RE = re.compile(
    r"グラフ\s*\d|棒グラフ|円グラフ|折れ線|折線|散布図|ヒストグラム|チャート|プロット|"
    r"のグラフ(で|の|に)|グラフ(で|の|に).{0,20}(値|カウント|ビン|系列|軸)|"
    r"可視化したもの|ビンの範囲")
_FORMAT_RE = re.compile(
    r"太字|ボールド|下線|アンダーライン|イタリック|斜体|取り消し線|"
    r"フォント|文字色|背景色|セルの色|塗りつぶし|罫線|ハイライト(?:色|され|した|して)|"
    r"マーカー(の色|で)|ピボット|pivot",
    re.I)
# A multi-hop question threads one entity into the lookup of another ("最も…な人/案件 の …").
_MULTIHOP_RE = re.compile(
    r"(もっとも|最も|一番|最大|最高|最多).{0,24}(人|担当者|案件|案件名|プロジェクト|会社|社|ファイル)"
    r".{0,20}(の|が担当).{0,16}(内線|ext|担当|案件|金額|着手金|報酬|期間|日付|番号|値)")

# Archetypes the backbone maps to the NUMERIC contract (derivation / computed values).
_NUMERIC_ARCH = frozenset({
    "derived_calculation", "data_shape", "csv_column_mean", "csv_column_max", "contract_amount",
})


def _base_contract(arch: str) -> str:
    """Map the coarse archetype to its base contract (before new-dimension refinement)."""
    if arch == "version_diff":
        return VERSION_DIFF
    if arch in ("enum_set", "highlight_set"):
        return FULL_ENUMERATION
    if arch in ("cross_aggregate", "pivot_condition"):
        return CROSS_AGGREGATE
    if arch in _NUMERIC_ARCH:
        return NUMERIC
    return SIMPLE_LOOKUP  # fact_lookup / document_extract / glossary_* / config_* / metric_score / unknown


# Deterministic confidence: specific structural/new-dimension hits are trusted; a plain simple_lookup
# fallback from an ``unknown`` archetype is the least certain (candidate for flash arbitration).
_CONF_SPECIFIC = 0.9
_CONF_BASE = 0.75
_CONF_WEAK = 0.5


@dataclass(frozen=True)
class QuestionContract:
    """The classification result: the routing contract plus its completion-condition payload."""

    contract: str
    label: str
    archetype: str                       # coarse backbone archetype (src.rag.archetype.classify)
    completion_conditions: tuple[str, ...]
    route: str
    method: str                          # "deterministic" | "flash"
    confidence: float
    evidence: str = ""                   # short why-note
    consistent_archetypes: frozenset[str] = field(default_factory=frozenset)

    def is_consistent_with(self, gold_archetype: str) -> bool:
        """True when ``gold_archetype`` is one this contract may legitimately refine (or unknown)."""
        return gold_archetype == "unknown" or gold_archetype in self.consistent_archetypes

    def to_dict(self) -> dict[str, object]:
        return {
            "contract": self.contract,
            "label": self.label,
            "archetype": self.archetype,
            "completion_conditions": list(self.completion_conditions),
            "route": self.route,
            "method": self.method,
            "confidence": self.confidence,
            "evidence": self.evidence,
        }


# --------------------------------------------------------------------------- numeric quantity contract
# Numeric questions frequently fail even after a correct file lookup because the model silently changes
# *what quantity is being measured*.  In particular, Japanese ``X のうち Y の割合`` fixes X as the
# denominator population.  The helpers below turn that wording, the requested unit, and the rounding
# instruction into a small machine-checkable contract shared by the investigator and exec verifier.
_ASCII_IDENTIFIER_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")
_DECIMAL_PLACES_RE = re.compile(r"小数第\s*(\d+)\s*位")
_NUMERIC_LITERAL_RE = re.compile(r"[+-]?\d+(?:,\d{3})*(?:\.(\d+))?")
_IGNORED_SCOPE_IDENTIFIERS = frozenset({"csv", "tsv", "xlsx", "xlsm", "df", "data"})


@dataclass(frozen=True)
class NumericRequirements:
    """The quantity/unit/rounding promises explicitly requested by one numeric question."""

    ratio: bool = False
    denominator_scope: str = ""
    denominator_fields: tuple[str, ...] = ()
    denominator_operators: tuple[str, ...] = ()  # lt/lte/gt/gte/eq
    unit: str | None = None
    decimal_places: int | None = None


@dataclass(frozen=True)
class NumericValidation:
    """Result of checking a proposed numeric answer and its executed formulas against the question."""

    passed: bool
    issues: tuple[str, ...] = ()
    denominator_formula: str = ""

    def to_dict(self) -> dict[str, object]:
        return {
            "passed": self.passed,
            "issues": list(self.issues),
            "denominator_formula": self.denominator_formula,
        }


def _denominator_scope(question: str) -> str:
    """Return the condition immediately governing ``のうち`` (the denominator population)."""
    if "のうち" not in question:
        return ""
    prefix = question.split("のうち", 1)[0]
    # Drop the document/project preamble while preserving the complete local condition.
    for sep in ("。", "？", "?", "！", "!", "、", ","):
        if sep in prefix:
            prefix = prefix.rsplit(sep, 1)[-1]
    return prefix.strip()


def numeric_requirements(question: str) -> NumericRequirements:
    """Infer the explicit numeric answer contract without consulting a model or corpus facts."""
    q = nfc(question or "")
    ratio = bool(re.search(r"割合|比率|何\s*[%％]|パーセント", q, re.I))
    scope = _denominator_scope(q) if ratio else ""
    fields = tuple(dict.fromkeys(
        token for token in _ASCII_IDENTIFIER_RE.findall(scope)
        if token.lower() not in _IGNORED_SCOPE_IDENTIFIERS
    ))
    ops: list[str] = []
    for pattern, code in (
        (r"未満", "lt"), (r"以下", "lte"), (r"(?:より大き|超え|超の)", "gt"),
        (r"以上", "gte"), (r"(?:=|＝|等しい)", "eq"),
    ):
        if re.search(pattern, scope):
            ops.append(code)

    unit: str | None = None
    for pattern, canonical in (
        (r"[%％]|パーセント", "%"), (r"億円|万円|千円|円", "円"),
        (r"ドル", "ドル"), (r"件", "件"), (r"人", "人"), (r"時間", "時間"),
        (r"日(?:間|数)?", "日"), (r"倍", "倍"),
    ):
        if re.search(pattern, q):
            unit = canonical
            break
    rounding = _DECIMAL_PLACES_RE.search(q)
    return NumericRequirements(
        ratio=ratio,
        denominator_scope=scope,
        denominator_fields=fields,
        denominator_operators=tuple(ops),
        unit=unit,
        decimal_places=(int(rounding.group(1)) if rounding else None),
    )


def _step_parts(step: Any) -> tuple[str, Any]:
    if isinstance(step, Mapping):
        return str(step.get("code", "") or ""), step.get("output")
    return str(getattr(step, "code", "") or ""), getattr(step, "output", None)


def _division_denominator(code: str) -> ast.AST | None:
    try:
        tree = ast.parse(code, mode="eval")
    except (SyntaxError, ValueError):
        return None
    return next((n.right for n in ast.walk(tree) if isinstance(n, ast.BinOp)
                 and isinstance(n.op, ast.Div)), None)


def _same_number(left: Any, right: Any) -> bool:
    try:
        return abs(float(left) - float(right)) <= max(1e-12, abs(float(right)) * 1e-12)
    except (TypeError, ValueError):
        return False


def _resolved_denominator_formula(formulas: Sequence[Any]) -> str:
    """Find the final ratio denominator and follow a numeric literal back to its compute provenance."""
    parts = [_step_parts(s) for s in formulas]
    for index in range(len(parts) - 1, -1, -1):
        code, _output = parts[index]
        denominator = _division_denominator(code)
        if denominator is None:
            continue
        if isinstance(denominator, ast.Constant) and isinstance(denominator.value, (int, float)):
            for prior_code, prior_output in reversed(parts[:index]):
                if prior_code and _same_number(prior_output, denominator.value):
                    return prior_code
        try:
            return ast.unparse(denominator)
        except Exception:  # pragma: no cover - ast.unparse is available on supported Python
            return code
    return ""


def _formula_operators(code: str) -> frozenset[str]:
    try:
        tree = ast.parse(code, mode="eval")
    except (SyntaxError, ValueError):
        return frozenset()
    mapping = {ast.Lt: "lt", ast.LtE: "lte", ast.Gt: "gt", ast.GtE: "gte", ast.Eq: "eq"}
    return frozenset(mapping[type(op)] for n in ast.walk(tree) if isinstance(n, ast.Compare)
                     for op in n.ops if type(op) in mapping)


def validate_numeric_answer(question: str, answer: str,
                            formulas: Sequence[Any] = ()) -> NumericValidation:
    """Check quantity definition, unit and rounding before a numeric answer may be committed.

    ``formulas`` accepts calc-ledger compute-step dicts or ``ComputeStep``-like objects.  For a literal
    denominator (``129 / 10938``), its producing step is followed by output value so the population
    filter remains auditable instead of disappearing behind the count.
    """
    req = numeric_requirements(question)
    issues: list[str] = []
    denominator = _resolved_denominator_formula(formulas) if req.ratio else ""
    if req.ratio and req.denominator_scope:
        if not denominator:
            issues.append(
                f"量の定義が未確認: 『{req.denominator_scope}のうち』の分母を計算証跡から特定できない")
        else:
            missing_fields = [f for f in req.denominator_fields
                              if not re.search(rf"\b{re.escape(f)}\b", denominator)]
            if missing_fields:
                issues.append(
                    "量の定義が不一致: 分母に『〜のうち』条件の対象列がない(" +
                    ", ".join(missing_fields) + f"; 分母={denominator})")
            present_ops = _formula_operators(denominator)
            missing_ops = [op for op in req.denominator_operators if op not in present_ops]
            if missing_ops:
                issues.append(
                    "量の定義が不一致: 分母に『〜のうち』条件の比較がない(" +
                    ", ".join(missing_ops) + f"; 分母={denominator})")

    normalized_answer = nfc(answer or "").replace("％", "%")
    if req.unit == "%" and "%" not in normalized_answer:
        issues.append("単位が不一致: 質問は%を要求している")
    elif req.unit and req.unit not in ("%", "円") and req.unit not in normalized_answer:
        issues.append(f"単位が不一致: 質問は{req.unit}を要求している")
    elif req.unit == "円" and "円" not in normalized_answer:
        issues.append("単位が不一致: 質問は円単位を要求している")

    if req.decimal_places is not None:
        match = _NUMERIC_LITERAL_RE.search(normalized_answer)
        actual = len(match.group(1) or "") if match else None
        if actual != req.decimal_places:
            issues.append(
                f"丸めが不一致: 小数第{req.decimal_places}位を要求、回答の小数桁={actual}")
    return NumericValidation(not issues, tuple(issues), denominator)


def _build(contract: str, arch: str, method: str, confidence: float, evidence: str) -> QuestionContract:
    return QuestionContract(
        contract=contract,
        label=CONTRACT_LABELS[contract],
        archetype=arch,
        completion_conditions=CONTRACT_COMPLETION[contract],
        route=CONTRACT_ROUTE[contract],
        method=method,
        confidence=confidence,
        evidence=evidence,
        consistent_archetypes=CONTRACT_ARCHETYPES[contract],
    )


def _deterministic(q: str, arch: str) -> tuple[str, float, str] | None:
    """Return ``(contract, confidence, evidence)`` from deterministic rules, or ``None`` if ambiguous.

    ``None`` means "no confident deterministic decision" — the caller may then consult ``flash``.
    """
    base = _base_contract(arch)

    # Authoritative structural contracts (version diff / enumeration / cross-aggregate) win outright.
    if base in (VERSION_DIFF, FULL_ENUMERATION, CROSS_AGGREGATE):
        return base, _CONF_SPECIFIC, f"archetype={arch}"

    # New-dimension refinements apply only to lookup/derivation bases.
    if _SPATIAL_GEOM_RE.search(q):
        return SPATIAL, _CONF_SPECIFIC, "spatial cue (座席/隣/向かい)"
    if _CHART_RE.search(q):
        return CHART_READ, _CONF_SPECIFIC, "chart cue (グラフ/ヒストグラム/系列)"
    if base == SIMPLE_LOOKUP and _FORMAT_RE.search(q):
        return FORMAT_CHECK, _CONF_SPECIFIC, "format cue (太字/下線/色/フォント)"
    if base == SIMPLE_LOOKUP and _MULTIHOP_RE.search(q):
        return MULTI_HOP, _CONF_BASE, "multi-hop cue (最も…な人/案件 の …)"
    if _SPATIAL_DIR_RE.search(q):
        return SPATIAL, _CONF_BASE, "directory cue (内線/EXT lookup)"

    if base == NUMERIC:
        return NUMERIC, _CONF_BASE, f"archetype={arch}"
    # Archetypes with a concrete structural signal (extraction / glossary / config / metric) commit to
    # simple_lookup deterministically.
    if arch in ("document_extract", "glossary_formal", "glossary_abbrev",
                "config_model_type", "config_hyperparam", "metric_score"):
        return SIMPLE_LOOKUP, _CONF_BASE, f"archetype={arch}"

    # A bare ``fact_lookup`` (the backbone's no-signal default) or an ``unknown`` archetype with no
    # structural cue is genuinely ambiguous → defer to flash if one is supplied.
    return None


# Prompt for the flash arbiter. It only decides among the nine contracts for a genuinely ambiguous
# question; the deterministic layer has already handled every confident case.
_FLASH_SYSTEM = (
    "あなたは社内ドライブ文書QAの質問を、回答に必要な『契約型』へ分類する分類器です。"
    "次の9種のいずれか1つをコード(英字)で返してください:\n"
    "simple_lookup=単一記載を引く単純検索 / multi_hop=中間エンティティを辿る多段関係 / "
    "cross_aggregate=複数文書を横断する集計 / full_enumeration=母集団を漏れなく挙げる完全列挙 / "
    "format_check=太字/下線/色などの書式で抽出 / chart_read=図/グラフから数値を読取 / "
    "spatial=座席表など空間関係の推論 / version_diff=旧版と新版の差分 / numeric=式で数値を導出。\n"
    "推測で本文を作らず、質問が要求する処理の種類だけで1つ選び、コードのみ返すこと。"
)

_FLASH_SCHEMA = {
    "type": "object",
    "properties": {"contract": {"type": "string", "enum": list(CONTRACTS)}},
    "required": ["contract"],
}


def flash_classify(question: str, *, model: str | None = None,
                   generate: Callable[..., str] | None = None) -> str | None:
    """Production flash arbiter: ask ``gemini-2.5-flash`` for one contract code; ``None`` on failure.

    Thin wrapper over :func:`src.rag.llm.generate` (imported lazily so importing this module never
    reaches the network or requires the Gemini SDK). Returns a validated contract code, or ``None`` when
    the model is unreachable or returns something outside :data:`CONTRACTS` (the caller then keeps its
    deterministic default). ``generate`` may be injected in tests to avoid the live client.
    """
    import json

    if generate is None:
        from src.rag import llm  # lazy: the google-genai client is only needed in production
        generate = llm.generate

    try:
        raw = generate(
            question,
            system=_FLASH_SYSTEM,
            model=model or settings.GEN_MODEL,
            temperature=0.0,
            max_output_tokens=64,
            response_schema=_FLASH_SCHEMA,
        )
    except Exception:  # noqa: BLE001 — arbitration is best-effort; fall back to deterministic default
        return None
    code = None
    try:
        code = json.loads(raw).get("contract")
    except (ValueError, AttributeError):
        code = raw.strip()
    return code if code in CONTRACTS else None


def classify(question: str, *, flash: Callable[[str], str | None] | None = None) -> QuestionContract:
    """Classify ``question`` into its :class:`QuestionContract` (deterministic first, flash on ambiguity).

    ``flash`` is an injected arbiter ``question -> contract_code | None`` consulted **only** when the
    deterministic layer is inconclusive (an ``unknown`` archetype with no specific cue). Pass
    :func:`flash_classify` in production; pass ``None`` (or omit) for a fully deterministic, network-free
    classification — which is what the gold-100 accuracy measurement uses.
    """
    q = nfc(question)
    arch = _archetype.classify(q)

    decided = _deterministic(q, arch)
    if decided is not None:
        contract, conf, why = decided
        return _build(contract, arch, "deterministic", conf, why)

    # Ambiguous: try the flash arbiter, else default to the weakest simple_lookup.
    if flash is not None:
        code = flash(q)
        if code in CONTRACTS:
            return _build(code, arch, "flash", _CONF_WEAK, "flash arbitration")
    return _build(SIMPLE_LOOKUP, arch, "deterministic", _CONF_WEAK, f"fallback (archetype={arch})")


# --------------------------------------------------------------------------- accuracy measurement
@dataclass(frozen=True)
class AgreementReport:
    """Result of scoring the classifier against a gold archetype column."""

    total: int
    agree: int
    rate: float
    refinements: tuple[dict[str, str], ...]   # contract strictly more specific than the gold archetype
    mismatches: tuple[dict[str, str], ...]     # contract NOT consistent with the gold archetype

    def to_dict(self) -> dict[str, object]:
        return {
            "total": self.total, "agree": self.agree, "rate": self.rate,
            "refinements": list(self.refinements), "mismatches": list(self.mismatches),
        }


# A contract is a "refinement" (not a trivial 1:1 of the archetype's base contract) when it is one of
# the strictly-more-specific new dimensions.
_REFINEMENT_CONTRACTS = frozenset({SPATIAL, CHART_READ, FORMAT_CHECK, MULTI_HOP})


def agreement_rate(rows: list[dict[str, str]], *,
                   flash: Callable[[str], str | None] | None = None) -> AgreementReport:
    """Score classifier vs. the gold ``archetype`` column (``≥0.90`` is the SOT-2493 acceptance target).

    Each row must carry ``question`` and ``archetype``. A row *agrees* when the predicted contract is
    consistent with the gold archetype (:meth:`QuestionContract.is_consistent_with`) — i.e. the contract
    either matches the archetype's base contract or legitimately *refines* it. Refinements and genuine
    mismatches are both recorded so their validity can be reviewed (受け入れ条件①: 不一致は妥当性を記録).
    """
    agree = 0
    refinements: list[dict[str, str]] = []
    mismatches: list[dict[str, str]] = []
    for row in rows:
        q = row.get("question", "")
        gold = (row.get("archetype", "") or "unknown").strip() or "unknown"
        res = classify(q, flash=flash)
        if res.is_consistent_with(gold):
            agree += 1
            if res.contract in _REFINEMENT_CONTRACTS:
                refinements.append({
                    "question": q, "gold_archetype": gold, "contract": res.contract,
                    "note": res.evidence,
                })
        else:
            mismatches.append({
                "question": q, "gold_archetype": gold, "contract": res.contract,
                "note": res.evidence,
            })
    total = len(rows)
    rate = agree / total if total else 1.0
    return AgreementReport(total=total, agree=agree, rate=rate,
                           refinements=tuple(refinements), mismatches=tuple(mismatches))
