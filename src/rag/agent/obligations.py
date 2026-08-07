"""SOT-2499 — evidence-obligation decomposition.

Parent SOT-2460. A question is not answered by "retrieve some text and hope it justifies the answer";
it is answered by discharging the *specific evidence obligations* the answer must keep. This module
turns a question into a structured list of those obligations — the control variable the local
re-search loop (SOT-2502) uses to re-search **only the unmet obligations** instead of re-running a
low-confidence, whole-corpus retrieval.

What an obligation is
---------------------
Each obligation is a ``{obligation, kind, status, evidence_ref}`` record:

* ``obligation`` — a one-line statement of what must be established (e.g. "回答値の出典を一意に特定する").
* ``kind`` — one of the structural :data:`KINDS` (対象/版の確定・判定規則・列挙・値の対応・量の定義・
  単位正規化・丸め・計算実行・整合確認), so a loop can reason about *what type* of gap is open without
  parsing prose.
* ``status`` — ``"unmet"`` until collected evidence discharges it, then ``"met"``.
* ``evidence_ref`` — the reference/snippet of the evidence that discharged it (empty while unmet).

Deterministic-first, flash-optional (mirrors :mod:`src.rag.agent.question_contract`)
-----------------------------------------------------------------------------------
:func:`decompose` classifies the question into its :class:`~src.rag.agent.question_contract.QuestionContract`
(deterministic) and expands the contract's **completion-condition template** into a base obligation
list (:data:`CONTRACT_OBLIGATIONS`) — fully network-free and reproducible, which is what the golden
test uses. A ``gemini-2.5-flash`` arbiter may be *injected* to specialise the list per question (name
the concrete files/versions/columns); the flash call is never imported at module load and never made
on the deterministic default, so importing this module reaches neither the network nor the Gemini SDK.

Satisfaction API
----------------
:meth:`ObligationSet.check` / :meth:`ObligationSet.unmet` match the obligations against collected
evidence (an investigator's ``evidence`` text, or explicitly-typed :class:`EvidenceItem` s) and return
the obligations with refreshed ``met``/``unmet`` status. **This Issue only adds the module; wiring it
into the investigator loop is SOT-2502, so the production answer path is untouched.**
"""
from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass, field, replace
from typing import Any, Callable, Iterable, Sequence

from config import settings
from src.rag.corpus import nfc
from src.rag.agent import question_contract as qc

# --------------------------------------------------------------------------- the seven obligation kinds
SOURCE_LOCATION = "source_location"        # 対象ファイル群と版の確定
JUDGMENT_RULE = "judgment_rule"            # 判定規則(書式/空間/差分などの規則)
ENUMERATION = "enumeration"                # 全該当要素の列挙(母集団の閉包)
VALUE_MAPPING = "value_mapping"            # 値の対応(エンティティ⇔値の解決)
QUANTITY_DEFINITION = "quantity_definition"  # 量の定義(分子/分母/母集団・「〜のうち」のスコープ)
UNIT_NORMALIZATION = "unit_normalization"  # 単位正規化(桁/単位の整合)
ROUNDING = "rounding"                      # 丸め(小数第N位/切上げ/切捨て)
COMPUTATION = "computation"                # 計算実行(決定論的な再導出)
CONSISTENCY = "consistency"                # 整合確認(再計算/連鎖の整合)

KINDS: tuple[str, ...] = (
    SOURCE_LOCATION, JUDGMENT_RULE, ENUMERATION, VALUE_MAPPING, QUANTITY_DEFINITION,
    UNIT_NORMALIZATION, ROUNDING, COMPUTATION, CONSISTENCY,
)

KIND_LABELS: dict[str, str] = {
    SOURCE_LOCATION: "対象/版の確定", JUDGMENT_RULE: "判定規則", ENUMERATION: "全該当要素の列挙",
    VALUE_MAPPING: "値の対応", QUANTITY_DEFINITION: "量の定義", UNIT_NORMALIZATION: "単位正規化",
    ROUNDING: "丸め", COMPUTATION: "計算実行",
    CONSISTENCY: "整合確認",
}

UNMET = "unmet"
MET = "met"

# --------------------------------------------------------------------------- per-contract obligation templates
# Each contract's completion-condition template (question_contract.CONTRACT_COMPLETION) expressed as a
# typed obligation checklist. Ordered ``(kind, obligation-text)`` pairs so they read as a discharge
# order. Every obligation's kind is one of :data:`KINDS`; every completion condition is covered.
CONTRACT_OBLIGATIONS: dict[str, tuple[tuple[str, str], ...]] = {
    qc.SIMPLE_LOOKUP: (
        (SOURCE_LOCATION, "回答値の出典(ファイル/スライド/セル)を一意に特定する"),
        (VALUE_MAPPING, "出典に明示された回答値を対応づける"),
    ),
    qc.MULTI_HOP: (
        (SOURCE_LOCATION, "各ホップの根拠文書を特定する"),
        (VALUE_MAPPING, "中間エンティティ(担当者/案件/期間)を根拠付きで一段ずつ確定する"),
        (CONSISTENCY, "ホップ連鎖が途切れず最終値まで整合する"),
    ),
    qc.CROSS_AGGREGATE: (
        (SOURCE_LOCATION, "集計対象の母集団(全プロジェクト/全ファイル)を権威的に確定する"),
        (ENUMERATION, "母集団の全要素を漏れなく列挙する"),
        (COMPUTATION, "集計関数(合計/平均/最大/件数)で決定論的に再計算する"),
        (CONSISTENCY, "対象列と集計関数が型整合している"),
    ),
    qc.FULL_ENUMERATION: (
        (SOURCE_LOCATION, "列挙対象の母集団(全体集合)を権威的に確定する"),
        (ENUMERATION, "閉包条件を満たし全該当要素を列挙する(見落とし0)"),
        (CONSISTENCY, "重複0・非該当の除外を確認する"),
    ),
    qc.FORMAT_CHECK: (
        (SOURCE_LOCATION, "判定対象の原本(ファイル/箇所)を特定する"),
        (JUDGMENT_RULE, "判定する書式属性(太字/下線/色/フォント)を定義する"),
        (ENUMERATION, "該当箇所を網羅し非該当(例:日付)を除外する"),
        (VALUE_MAPPING, "表/ピボットでは行・列の生ラベルを元データのフィールド名へ意味解決する(該当時)"),
        (COMPUTATION, "表/ピボットでは抽出条件・対象列・集計方法の必須3要素を確定する(該当時)"),
    ),
    qc.CHART_READ: (
        (SOURCE_LOCATION, "対象グラフ(図番号/シート/系列)を一意に特定する"),
        (VALUE_MAPPING, "numCacheまたは対応する元データ列から値を厳密に対応づける"),
        (COMPUTATION, "numCacheが無い場合は元データからグラフ集計を決定論的に再導出する"),
        (UNIT_NORMALIZATION, "読取値の桁数/単位を質問の要求に合わせる"),
    ),
    qc.SPATIAL: (
        (SOURCE_LOCATION, "座席表/レイアウトの原本を特定する"),
        (JUDGMENT_RULE, "空間関係(隣/向かい/座席番号)を確定する"),
        (VALUE_MAPPING, "人物⇔座席⇔内線などの対応を一意に解決する"),
    ),
    qc.VERSION_DIFF: (
        (SOURCE_LOCATION, "旧版と新版の対応ペアを一意に確定する"),
        (JUDGMENT_RULE, "diffpairで全スライド/全シートを比較し、隣接版間の実質的な変更のみを抽出する"),
        (CONSISTENCY, "version_diffツールの決定論結果と回答が一致することを確認する"),
    ),
    qc.NUMERIC: (
        (SOURCE_LOCATION, "型付き入力(対象列/対象ファイル)を根拠から特定する"),
        (VALUE_MAPPING, "数値/単位入力を根拠から確定する"),
        (QUANTITY_DEFINITION, "要求量を分子・分母・母集団で定義し、『〜のうち』直前の条件を分母に保持する"),
        (COMPUTATION, "再実行可能な式で決定論的に導出する"),
        (UNIT_NORMALIZATION, "計算結果の単位を質問の要求に合わせる"),
        (ROUNDING, "小数第N位・切上げ・切捨て等の丸めを質問の要求に合わせる"),
    ),
}

# ``明記`` / ``そのまま`` asks for a *literal* extraction, not a semantic near-match.  Quoted terms
# are intentionally taken only from the question; this remains corpus-agnostic and cannot smuggle a
# gold answer into the agent.  The returned terms become both a typed obligation and a commit gate in
# :mod:`investigator`.
_LITERAL_REQUEST_RE = re.compile(
    r"明記|そのまま|原文(?:どおり|通り|表記)|文字列(?:として)?|literal", re.I)
_QUOTED_TERM_RE = re.compile(r"[「『\"“](.+?)[」』\"”]")


def literal_terms(question: str) -> tuple[str, ...]:
    """Quoted condition terms that must literally occur beside an answer candidate.

    A quoted project/file name in an ordinary lookup does not activate the rule: the question must
    explicitly request an exact/original/literal extraction (``明記`` / ``そのまま`` / ``原文``).
    """
    q = nfc(question or "")
    if not _LITERAL_REQUEST_RE.search(q):
        return ()
    return tuple(dict.fromkeys(
        term.strip() for term in _QUOTED_TERM_RE.findall(q) if term.strip()
    ))


@dataclass(frozen=True)
class LiteralValidation:
    """Result of checking a literal extraction against actual tool output fragments."""

    passed: bool
    terms: tuple[str, ...] = ()
    issues: tuple[str, ...] = ()


def _literal_norm(value: Any) -> str:
    return re.sub(r"\s+", "", unicodedata.normalize("NFKC", nfc(str(value or "")))).lower()


def _tool_fragments(value: Any) -> list[str]:
    """Flatten tool output while retaining row/object locality for literal co-occurrence checks."""
    if value is None:
        return []
    if isinstance(value, str):
        lines = [line.strip() for line in value.splitlines() if line.strip()]
        # Office extractors sometimes put a table cell or wrapped PDF text on adjacent lines.  Keep
        # individual lines and bounded three-line windows; never join the whole document, which would
        # let unrelated distant text satisfy the obligation.
        out = list(lines)
        out.extend(" | ".join(lines[i:i + 3]) for i in range(max(0, len(lines) - 2)))
        return out
    if isinstance(value, dict):
        fragments: list[str] = []
        scalar_values = [str(v) for v in value.values()
                         if isinstance(v, (str, int, float, bool))]
        if scalar_values:
            fragments.append(" | ".join(scalar_values))
        for nested in value.values():
            fragments.extend(_tool_fragments(nested))
        return fragments
    if isinstance(value, (list, tuple, set)):
        fragments: list[str] = []
        for item in value:
            fragments.extend(_tool_fragments(item))
        return fragments
    return [str(value)]


def validate_literal_evidence(question: str, answer: str,
                              tool_outputs: Sequence[Any]) -> LiteralValidation:
    """Require each answer candidate to co-occur with every quoted condition in tool evidence.

    The model-authored ``Answer.evidence`` is deliberately not accepted here.  Only dispatched tool
    outputs are supplied by the caller, preventing a confident model from satisfying the gate by
    merely repeating the condition term in its own prose.
    """
    terms = literal_terms(question)
    if not terms:
        return LiteralValidation(True)
    fragments = [_literal_norm(f) for output in tool_outputs for f in _tool_fragments(output)]
    norm_terms = tuple(_literal_norm(t) for t in terms)
    candidates = tuple(
        part.strip() for part in re.split(r"[、,，;；\n]+", answer or "") if part.strip()
    ) or ((answer or "").strip(),)
    issues: list[str] = []
    for candidate in candidates:
        nc = _literal_norm(candidate)
        if not nc:
            continue
        if not any(nc in fragment and all(term in fragment for term in norm_terms)
                   for fragment in fragments):
            issues.append(
                f"候補『{candidate}』と条件語" +
                "/".join(f"『{term}』" for term in terms) +
                "が同一セル/文のツール証拠に literal 共起しない")
    return LiteralValidation(not issues, terms, tuple(issues))

# --------------------------------------------------------------------------- evidence data types
@dataclass(frozen=True)
class EvidenceItem:
    """One collected piece of evidence.

    ``text`` is the free-form evidence (an investigator ``Answer.evidence`` line, a tool result). ``ref``
    is an optional locator (``"提案書.pptx#slide6"``) recorded on the obligation it discharges. ``kinds``
    optionally declares which obligation kinds this item can satisfy — a precise signal SOT-2502 can set
    when it knows what a tool result establishes; when empty, satisfaction falls back to text cues.
    """

    text: str = ""
    ref: str = ""
    kinds: frozenset[str] = field(default_factory=frozenset)


def as_evidence(evidence: "EvidenceInput") -> tuple[EvidenceItem, ...]:
    """Coerce assorted evidence inputs into ``EvidenceItem`` s.

    Accepts a single string, a single :class:`EvidenceItem`, a mapping (``{text, ref?, kinds?}``), or any
    iterable mixing those. ``None``/empty yields an empty tuple. Kept permissive so callers can pass an
    investigator's raw ``evidence`` string directly, or richer typed items once SOT-2502 tags them.
    """
    if evidence is None:
        return ()
    if isinstance(evidence, EvidenceItem):
        return (evidence,)
    if isinstance(evidence, str):
        return (EvidenceItem(text=evidence),) if evidence.strip() else ()
    if isinstance(evidence, dict):
        return _coerce_mapping(evidence)
    out: list[EvidenceItem] = []
    for e in evidence:  # iterable of the above
        out.extend(as_evidence(e))
    return tuple(out)


def _coerce_mapping(m: dict) -> tuple[EvidenceItem, ...]:
    text = str(m.get("text", "") or "")
    if not text.strip() and not m.get("ref"):
        return ()
    raw_kinds = m.get("kinds") or ()
    kinds = frozenset(k for k in raw_kinds if k in KINDS)
    return (EvidenceItem(text=text, ref=str(m.get("ref", "") or ""), kinds=kinds),)


# --------------------------------------------------------------------------- kind text cues (fallback matcher)
# When an evidence item does not declare its ``kinds`` explicitly, an obligation is discharged if the
# evidence text carries a cue for that obligation's kind. Deliberately conservative — a cue asserts the
# evidence *speaks to* the obligation, not that the answer is correct (that is the verifier's job).
_KIND_CUES: dict[str, re.Pattern[str]] = {
    SOURCE_LOCATION: re.compile(
        r"\.pptx|\.docx|\.xlsx|\.pdf|スライド|シート|セル|ページ|ファイル|旧版|最新版|新版|version|old|v\d",
        re.I),
    JUDGMENT_RULE: re.compile(
        r"太字|ボールド|下線|イタリック|フォント|文字色|背景色|罫線|ハイライト|"
        r"隣|向かい|左隣|右隣|座席番号|差分|変更点|実質的な変更|規則|条件|基準|定義"),
    ENUMERATION: re.compile(r"すべて|全て|全部|一覧|漏れ|重複|列挙|網羅|該当箇所|\d+\s*件|\d+\s*名|母集団"),
    VALUE_MAPPING: re.compile(r"対応|⇔|=|:|内線|ext|担当|は\s*[\wぁ-んァ-ヶ一-龠]+\s*(です|である)", re.I),
    QUANTITY_DEFINITION: re.compile(r"分子|分母|母集団|ベース集合|対象集合|.+のうち|割合の定義|比率の定義"),
    UNIT_NORMALIZATION: re.compile(r"円|万円|千円|%|パーセント|単位|桁|小数|kg|件数|人数|時間|日数"),
    ROUNDING: re.compile(r"小数第\s*\d+\s*位|四捨五入|切り上げ|切上げ|切り捨て|切捨て|丸め|round", re.I),
    COMPUTATION: re.compile(
        r"合計|総額|平均|最大|最小|最多|差額|計算|集計|式|corpus_aggregate|compute|再計算|=\s*\d"),
    CONSISTENCY: re.compile(r"一致|整合|検証|再計算|再現|突き合わ|確認|クロスチェック|矛盾(が)?ない"),
}


def _snippet(text: str, limit: int = 60) -> str:
    t = " ".join(text.split())
    return t if len(t) <= limit else t[: limit - 1] + "…"


# --------------------------------------------------------------------------- obligation records
@dataclass(frozen=True)
class Obligation:
    """One evidence obligation and its discharge state."""

    obligation: str
    kind: str
    status: str = UNMET
    evidence_ref: str = ""

    @property
    def met(self) -> bool:
        return self.status == MET

    def to_dict(self) -> dict[str, str]:
        return {"obligation": self.obligation, "kind": self.kind,
                "status": self.status, "evidence_ref": self.evidence_ref}


def _discharge(ob: Obligation, evidence: Sequence[EvidenceItem], *, use_text_cues: bool) -> Obligation:
    """Return ``ob`` marked ``met`` (with an ``evidence_ref``) if some evidence item discharges it."""
    if ob.met:
        return ob
    cue = _KIND_CUES.get(ob.kind)
    for item in evidence:
        if ob.kind in item.kinds:  # explicit, precise signal
            return replace(ob, status=MET, evidence_ref=item.ref or _snippet(item.text))
        if use_text_cues and cue is not None and item.text and cue.search(item.text):
            return replace(ob, status=MET, evidence_ref=item.ref or _snippet(item.text))
    return ob


@dataclass(frozen=True)
class ObligationSet:
    """A question's evidence obligations plus the contract they were derived from."""

    question: str
    contract: str
    obligations: tuple[Obligation, ...]
    method: str = "template"   # "template" | "flash"

    def check(self, evidence: "EvidenceInput", *, use_text_cues: bool = True) -> "ObligationSet":
        """Return a copy with each obligation's status refreshed against ``evidence``."""
        items = as_evidence(evidence)
        refreshed = tuple(_discharge(o, items, use_text_cues=use_text_cues) for o in self.obligations)
        return replace(self, obligations=refreshed)

    def unmet(self, evidence: "EvidenceInput" = None, *, use_text_cues: bool = True) -> tuple[Obligation, ...]:
        """The obligations still unmet — after discharging against ``evidence`` when supplied.

        With no ``evidence`` it reports the currently-unmet obligations as-is (freshly decomposed sets are
        all unmet). This is the control signal SOT-2502 re-searches on.
        """
        src = self.check(evidence, use_text_cues=use_text_cues) if evidence is not None else self
        return tuple(o for o in src.obligations if not o.met)

    @property
    def all_met(self) -> bool:
        return all(o.met for o in self.obligations)

    def to_dict(self) -> dict[str, object]:
        return {
            "question": self.question,
            "contract": self.contract,
            "method": self.method,
            "obligations": [o.to_dict() for o in self.obligations],
        }


# --------------------------------------------------------------------------- flash generation (opt-in)
_FLASH_SYSTEM = (
    "あなたは社内ドライブ文書QAの質問を、回答を正当化するために必要な『証拠義務』の構造化リストへ分解します。"
    "各義務は obligation(必要な確立事項の一文)と kind を持ちます。kind は次の9種のいずれか(英字コード):\n"
    "source_location=対象ファイル群と版の確定 / judgment_rule=判定規則 / enumeration=全該当要素の列挙 / "
    "value_mapping=値の対応 / quantity_definition=量の分子・分母・母集団 / "
    "unit_normalization=単位正規化 / rounding=丸め / computation=計算実行 / consistency=整合確認。\n"
    "与えられた契約型の完了条件を、この質問に固有(具体的なファイル名/版/対象列など)へ具体化しつつ、"
    "漏れなく最小限の義務へ分解してください。推測で答え本文を作らず、義務のみを返すこと。"
)

_FLASH_SCHEMA = {
    "type": "object",
    "properties": {
        "obligations": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "obligation": {"type": "string"},
                    "kind": {"type": "string", "enum": list(KINDS)},
                },
                "required": ["obligation", "kind"],
            },
        }
    },
    "required": ["obligations"],
}


def flash_decompose(question: str, contract: str, *, model: str | None = None,
                    generate: Callable[..., str] | None = None) -> tuple[dict[str, str], ...] | None:
    """Production flash arbiter: ask ``gemini-2.5-flash`` for a per-question obligation list.

    Returns a tuple of validated ``{obligation, kind}`` dicts (kind ∈ :data:`KINDS`), or ``None`` when the
    model is unreachable or returns nothing usable (the caller then keeps the deterministic template).
    ``generate`` may be injected in tests to avoid the live client; the ``src.rag.llm`` import is lazy so
    importing this module never reaches the network.
    """
    import json

    if generate is None:
        from src.rag import llm  # lazy: the google-genai client is only needed in production
        generate = llm.generate

    completion = "\n".join(f"- {c}" for c in qc.CONTRACT_COMPLETION.get(contract, ()))
    prompt = (
        f"契約型: {contract} ({qc.CONTRACT_LABELS.get(contract, contract)})\n"
        f"完了条件テンプレート:\n{completion}\n\n質問: {question}"
    )
    try:
        raw = generate(
            prompt,
            system=_FLASH_SYSTEM,
            model=model or settings.GEN_MODEL,
            temperature=0.0,
            max_output_tokens=512,
            response_schema=_FLASH_SCHEMA,
        )
    except Exception:  # noqa: BLE001 — flash refinement is best-effort; fall back to the template
        return None
    try:
        items = json.loads(raw).get("obligations", [])
    except (ValueError, AttributeError):
        return None
    out: list[dict[str, str]] = []
    for it in items or ():
        if not isinstance(it, dict):
            continue
        kind = str(it.get("kind", "")).strip()
        text = str(it.get("obligation", "") or "").strip()
        if kind in KINDS and text:
            out.append({"obligation": text, "kind": kind})
    return tuple(out) or None


# --------------------------------------------------------------------------- decomposition entry point
def _template_obligations(contract: str, question: str = "") -> tuple[Obligation, ...]:
    obligations = tuple(
        Obligation(obligation=text, kind=kind)
        for kind, text in CONTRACT_OBLIGATIONS.get(contract, CONTRACT_OBLIGATIONS[qc.SIMPLE_LOOKUP])
    )
    if contract == qc.SIMPLE_LOOKUP and qc.is_regulation_content_question(question):
        # A no-special-rule finding still leaves the governing fallback rule to establish.  Split its
        # semantic fields across distinct kinds so one generic lookup hit cannot make the whole answer
        # appear complete; the submit guard independently validates the final answer surface.
        obligations += (
            Obligation("特別規定の不存在時に適用される一般規定の単価と税処理を確定する", VALUE_MAPPING),
            Obligation("一般規定の課金単位と切上げ/切捨て等の丸め規則を確定する", ROUNDING),
            Obligation("一般規定の精算周期を確定する", JUDGMENT_RULE),
            Obligation("一般規定に上限/限度があるかを確定する", CONSISTENCY),
        )
    if contract == qc.CHART_READ and qc.is_gantt_week_question(question):
        obligations += (
            Obligation("週ヘッダ列のx座標系を較正する", JUDGMENT_RULE),
            Obligation("バーleft/widthと週セルの重なりから開始週・終了週を決定論的に対応づける", VALUE_MAPPING),
            Obligation("開始週候補が競合しないことを確認し、競合時は回答を確定しない", CONSISTENCY),
        )
    return obligations


def decompose(
    question: str,
    *,
    contract: str | None = None,
    flash: Callable[[str, str], Iterable[dict[str, str]] | None] | None = None,
) -> ObligationSet:
    """Decompose ``question`` into its evidence-obligation list.

    Deterministic by default: classify the question into its contract (unless ``contract`` is supplied)
    and expand the contract's completion-condition template into typed obligations — reproducible and
    network-free, which is what the golden test uses. ``flash`` is an injected arbiter
    ``(question, contract) -> [{obligation, kind}, …] | None`` consulted **only** to specialise the list
    per question; pass :func:`flash_decompose` in production, or omit for the deterministic template.
    Invalid/empty flash output is ignored and the template is used.
    """
    q = nfc(question)
    if contract is None:
        contract = qc.classify(q).contract
    elif contract not in qc.CONTRACTS:
        contract = qc.SIMPLE_LOOKUP

    method = "template"
    obligations = _template_obligations(contract, q)
    if flash is not None:
        produced = flash(q, contract)
        parsed = _obligations_from_items(produced)
        if parsed:
            obligations = parsed
            method = "flash"
    exact_terms = literal_terms(q)
    if exact_terms:
        quoted = "・".join(f"『{term}』" for term in exact_terms)
        literal_obligation = Obligation(
            obligation=f"引用条件語{quoted}が採用候補と同一セル/文に文字列として存在することを確認する",
            kind=JUDGMENT_RULE,
        )
        if literal_obligation.obligation not in {o.obligation for o in obligations}:
            obligations = (*obligations, literal_obligation)
    return ObligationSet(question=q, contract=contract, obligations=obligations, method=method)


def _obligations_from_items(items: Iterable[dict[str, str]] | None) -> tuple[Obligation, ...]:
    """Validate injected/flash obligation dicts into ``Obligation`` s (drop malformed ones)."""
    if not items:
        return ()
    out: list[Obligation] = []
    for it in items:
        if not isinstance(it, dict):
            continue
        kind = str(it.get("kind", "")).strip()
        text = str(it.get("obligation", "") or "").strip()
        if kind in KINDS and text:
            out.append(Obligation(obligation=text, kind=kind))
    return tuple(out)


# Type alias documenting the permissive evidence inputs :func:`as_evidence` accepts.
EvidenceInput = "str | EvidenceItem | dict | Iterable[str | EvidenceItem | dict] | None"
