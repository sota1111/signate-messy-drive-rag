"""SOT-2500 — full-enumeration closure protocol (権威的母集団の特定 + 閉包条件ゲート).

Parent SOT-2460. The ``enum_set`` route scores MATCH 2 / abstain 7, and three of those abstains
(idx45 針14/14・idx67 3/3・idx87 1/1) are **abstains despite every needed element being in the top-k**:
the model has the evidence but cannot *convince itself the set is complete*, so it declines. Completeness
of an enumeration is not established by "the retrieval top-k looks like it covers it" — it is established
by first pinning the **authoritative population** (用語集・一覧表・マスターシート・フォルダ目録) and then
proving a **closure condition** over it. This module is that protocol.

The two halves
--------------
1. **Authoritative-population resolver** (:func:`resolve_population`). The question names a *population
   kind* (案件/人物/略称/文書/タスク/座席…); the resolver maps that kind to the corpus's authoritative
   master for it. The one master the corpus **persists** is :class:`~src.rag.tools.profile.CorpusProfile`'s
   ``aliases`` map (canonical 会社/案件 名 ⇔ その別表記) — the 略称/案件 母集団. When the profile holds
   it, the resolver returns the resolved member set; otherwise it returns an *unresolved* population that
   names the source category the live loop must go and read (via ``find_files``/``file_grep``), never a
   guessed list.

2. **Closure gate** (:func:`evaluate_closure`). Four conditions must all hold before an enumeration may
   be committed:

   * ``population_defined`` — 母集団(全体集合)が権威的に確定している
   * ``candidate_reasons`` — 各候補に包含/除外の理由が付いている
   * ``alt_path_zero_new`` — 別の検索経路で新規候補がゼロ(=見落としなし)
   * ``count_match`` — 列挙件数と別経路の集計件数が一致

   Any unmet condition is emitted as an :data:`~src.rag.agent.obligations.ENUMERATION` obligation coded
   :data:`EVIDENCE_INCOMPLETE` — the enumeration is *not* answered, it is turned into a discharged-or-abstain
   obligation, exactly as the abstain ledger (SOT-2492) codes an incomplete enumeration.

Ordering (:func:`apply_ordering`)
---------------------------------
Enumerations receive no partial credit and must be emitted in a **canonical order** (ID昇順 / 出現順 /
座席表順) so the deterministic set scorer sees a stable, de-duplicated list.

Deterministic-first, flash-optional, network-free
--------------------------------------------------
Mirrors :mod:`src.rag.agent.question_contract` / :mod:`src.rag.agent.obligations`: every classification
runs from regex cues first and only consults an *injected* ``gemini-2.5-flash`` arbiter when the population
kind is genuinely ambiguous. The flash call is never imported at module load and never made on a confident
deterministic hit, so importing this module reaches neither the network nor the Gemini SDK.

Wiring (SOT-2500)
-----------------
:class:`EnumerationGate` is the one-shot interceptor the investigator loop uses: when a full-enumeration
question is about to abstain, it injects the closure *procedure* (:func:`closure_procedure`) **once** so the
model resolves the authoritative population and checks closure before it may abstain again. The gate is off
by default in :func:`~src.rag.agent.investigator.investigate` (byte-identical answer path); the live entry
point opts in. The procedure text injects **no corpus fact** (no 略称 answer, no member list) — it only
tells the model which *kind* of authoritative source to read and which conditions to prove, preserving the
investigator's no-fact-injection invariant.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Callable

from config import settings
from src.rag.agent import obligations as ob
from src.rag.agent import question_contract as qc
from src.rag.agent.abstain_ledger import EVIDENCE_INCOMPLETE
from src.rag.corpus import nfc
from src.rag.tools.profile import CorpusProfile

# Re-exported so callers can align an incomplete enumeration with the abstain ledger's state code.
__all__ = [
    "EVIDENCE_INCOMPLETE",
    "POPULATION_KINDS",
    "ORDERING_RULES",
    "CLOSURE_CONDITIONS",
    "detect_population_kind",
    "detect_ordering_rule",
    "apply_ordering",
    "resolve_population",
    "evaluate_closure",
    "closure_procedure",
    "EnumerationGate",
]

# --------------------------------------------------------------------------- population kinds (母集団種別)
PROJECT = "project"      # 案件/プロジェクト/会社
PERSON = "person"        # 人物/担当者/氏名
ABBREV = "abbrev"        # 略称/主略称/コード
DOCUMENT = "document"    # 文書/ファイル/資料
TASK = "task"            # タスクID/アクションID/工程
SEAT = "seat"            # 座席/席次/レイアウト
GENERIC = "generic"      # 種別を特定できない一般集合(fallback)

POPULATION_KINDS: tuple[str, ...] = (PROJECT, PERSON, ABBREV, DOCUMENT, TASK, SEAT, GENERIC)

POPULATION_LABELS: dict[str, str] = {
    PROJECT: "案件/プロジェクト", PERSON: "人物/担当者", ABBREV: "略称/コード",
    DOCUMENT: "文書/ファイル", TASK: "タスク/工程ID", SEAT: "座席", GENERIC: "一般集合",
}

# Ordered, most-specific-first cues per kind. First match wins.
_POP_CUES: tuple[tuple[str, re.Pattern[str]], ...] = (
    (ABBREV, re.compile(r"主略称|略称|コード名|略号")),
    (TASK, re.compile(r"タスク\s*ID|アクション\s*ID|工程\s*(ID|番号)|task\s*id", re.I)),
    (SEAT, re.compile(r"座席|席次|座って|着席|レイアウト|配置図|内線")),
    (PERSON, re.compile(r"氏名|担当者|参加者|メンバー|出席者|従業員|社員|(?:全員|各人|誰)")),
    (DOCUMENT, re.compile(r"ファイル|文書|資料|スライド|ドキュメント|シート|ページ")),
    (PROJECT, re.compile(r"案件|プロジェクト|会社|企業|取引先|顧客")),
)


def detect_population_kind(question: str, *, flash: Callable[[str], str | None] | None = None) -> str:
    """Return the enumeration's population kind (deterministic cues first, flash on ambiguity).

    ``flash`` is an injected arbiter ``question -> population_kind | None`` consulted **only** when no cue
    matches; pass :func:`flash_population_kind` in production, or omit for a fully deterministic,
    network-free classification. Unknown/absent → :data:`GENERIC`.
    """
    q = nfc(question)
    for kind, pat in _POP_CUES:
        if pat.search(q):
            return kind
    if flash is not None:
        code = flash(q)
        if code in POPULATION_KINDS:
            return code
    return GENERIC


# --------------------------------------------------------------------------- ordering rules (列挙順序規則)
ORDER_ID_ASC = "id_asc"          # ID/番号の昇順(T09→T10…, train_0077→train_0216…)
ORDER_APPEARANCE = "appearance"  # 原本の出現順(行順/スライド順)
ORDER_SEATING = "seating"        # 座席表の並び(空間順)

ORDERING_RULES: tuple[str, ...] = (ORDER_ID_ASC, ORDER_APPEARANCE, ORDER_SEATING)

ORDERING_LABELS: dict[str, str] = {
    ORDER_ID_ASC: "ID/番号 昇順", ORDER_APPEARANCE: "原本の出現順", ORDER_SEATING: "座席表順",
}

_ID_ORDER_RE = re.compile(r"ID|番号|昇順|順に|id", re.I)
_DIGITS_RE = re.compile(r"\d+")


def detect_ordering_rule(question: str, population_kind: str | None = None) -> str:
    """Pick the canonical emission order for the enumeration.

    Seat populations enumerate in seating order; task/ID populations (or an explicit ID/番号/昇順 cue)
    enumerate in numeric ID order; everything else preserves the原本の出現順 (the safe default that never
    reorders an inherently ordered list).
    """
    kind = population_kind or detect_population_kind(question)
    if kind == SEAT:
        return ORDER_SEATING
    if kind == TASK or _ID_ORDER_RE.search(nfc(question)):
        return ORDER_ID_ASC
    return ORDER_APPEARANCE


def _id_sort_key(item: str) -> tuple[int, int, str]:
    """Numeric-aware key: sort by the first integer run when present (T09<T10, train_0077<train_0216)."""
    m = _DIGITS_RE.search(item)
    return (0, int(m.group()), item) if m else (1, 0, item)


def apply_ordering(items: "list[str] | tuple[str, ...]", rule: str = ORDER_APPEARANCE) -> list[str]:
    """De-duplicate (first occurrence wins) and order ``items`` by ``rule``.

    ``ORDER_ID_ASC`` sorts by the first embedded integer; ``ORDER_APPEARANCE`` / ``ORDER_SEATING`` keep the
    caller's order (the original 出現順/座席順). Blank items are dropped.
    """
    seen: set[str] = set()
    deduped: list[str] = []
    for raw in items:
        s = nfc(str(raw)).strip()
        if s and s not in seen:
            seen.add(s)
            deduped.append(s)
    if rule == ORDER_ID_ASC:
        return sorted(deduped, key=_id_sort_key)
    return deduped  # appearance / seating: caller already supplies the canonical order


# --------------------------------------------------------------------------- authoritative population resolver
@dataclass(frozen=True)
class AuthoritativePopulation:
    """The resolved (or to-be-resolved) authoritative master for an enumeration's population.

    * ``members`` — the resolved full set (empty while unresolved).
    * ``resolved`` — True only when an authoritative master was actually located (not a retrieval guess).
    * ``source`` — where the master came from / must be read (``corpus_profile.aliases`` when resolved, else
      a source *category* hint the live loop should go read).
    """

    question: str
    kind: str
    label: str
    members: tuple[str, ...] = ()
    resolved: bool = False
    source: str = ""
    note: str = ""

    def __len__(self) -> int:
        return len(self.members)

    def to_dict(self) -> dict[str, object]:
        return {
            "question": self.question, "kind": self.kind, "label": self.label,
            "members": list(self.members), "resolved": self.resolved,
            "source": self.source, "note": self.note,
        }


# Which authoritative-source *category* each population kind is read from when the profile has no master
# for it — a hint for the live loop's re-search (find_files/file_grep), never an answer.
_SOURCE_HINT: dict[str, str] = {
    PROJECT: "全案件フォルダの目録(find_files で案件ディレクトリを列挙)",
    PERSON: "座席表/名簿/参加者一覧(氏名マスター)",
    ABBREV: "用語集(略称⇔正式名称の一覧表)",
    DOCUMENT: "対象フォルダのファイル目録(find_files)",
    TASK: "対象PL/工程表シートのタスクID列(全行走査)",
    SEAT: "座席表(レイアウト図)",
    GENERIC: "母集団を定義する一覧表/マスターシート",
}


def resolve_population(question: str, profile: CorpusProfile | None = None, *,
                       flash: Callable[[str], str | None] | None = None) -> AuthoritativePopulation:
    """Resolve the authoritative population for ``question`` from the corpus master when available.

    The one authoritative master the corpus *persists* is the profile's ``aliases`` map (canonical 会社/案件
    名 ⇔ 別表記) — the 略称/案件 母集団. For an ``abbrev``/``project`` question with a populated profile, the
    resolver returns that master's canonical names as the resolved member set (source ``corpus_profile.aliases``).
    For every other kind — or an empty profile — it returns an *unresolved* population naming the source
    *category* the live loop must go read (never a guessed list), so the closure gate keeps
    ``population_defined`` unmet until the master is actually located.
    """
    q = nfc(question)
    kind = detect_population_kind(q, flash=flash)
    label = POPULATION_LABELS.get(kind, kind)

    if profile is not None and kind in (ABBREV, PROJECT) and profile.aliases:
        members = tuple(sorted(profile.aliases.keys()))
        if members:
            return AuthoritativePopulation(
                question=q, kind=kind, label=label, members=members, resolved=True,
                source="corpus_profile.aliases",
                note=f"{len(members)}件の権威的マスター(canonical名)を profile.aliases から解決")

    return AuthoritativePopulation(
        question=q, kind=kind, label=label, members=(), resolved=False,
        source=_SOURCE_HINT.get(kind, _SOURCE_HINT[GENERIC]),
        note="権威的マスター未解決: 上記ソースを読み取って母集団を確定すること")


# --------------------------------------------------------------------------- closure gate (閉包条件)
C_POPULATION_DEFINED = "population_defined"  # 母集団(全体集合)が権威的に確定
C_CANDIDATE_REASONS = "candidate_reasons"    # 各候補に包含/除外理由
C_ALT_PATH_ZERO = "alt_path_zero_new"        # 別検索経路で新規候補ゼロ(見落としなし)
C_COUNT_MATCH = "count_match"                # 列挙件数 == 別経路の集計件数

CLOSURE_CONDITIONS: tuple[str, ...] = (
    C_POPULATION_DEFINED, C_CANDIDATE_REASONS, C_ALT_PATH_ZERO, C_COUNT_MATCH,
)

CLOSURE_LABELS: dict[str, str] = {
    C_POPULATION_DEFINED: "母集団(全体集合)を権威的に確定した",
    C_CANDIDATE_REASONS: "各候補に包含/除外の理由が付いている",
    C_ALT_PATH_ZERO: "別の検索経路で新規候補がゼロ(見落としなし)",
    C_COUNT_MATCH: "列挙件数が別経路の集計件数と一致する",
}


@dataclass(frozen=True)
class ClosureState:
    """Observed truth of each closure condition (all default False = nothing proven yet)."""

    population_defined: bool = False
    candidate_reasons: bool = False
    alt_path_zero_new: bool = False
    count_match: bool = False

    def get(self, condition: str) -> bool:
        return bool(getattr(self, condition, False))


@dataclass(frozen=True)
class ClosureResult:
    """Outcome of the closure gate for one enumeration."""

    satisfied: bool
    unmet: tuple[str, ...]                         # unmet condition codes (empty ⇔ satisfied)
    obligations: tuple[ob.Obligation, ...]         # one ENUMERATION obligation per unmet condition
    state_code: str = ""                           # EVIDENCE_INCOMPLETE when unmet, else ""

    def to_dict(self) -> dict[str, object]:
        return {
            "satisfied": self.satisfied,
            "unmet": list(self.unmet),
            "obligations": [o.to_dict() for o in self.obligations],
            "state_code": self.state_code,
        }


def evaluate_closure(state: "ClosureState | dict | None") -> ClosureResult:
    """Gate an enumeration on the four closure conditions.

    All four true → ``satisfied`` (the enumeration may be committed). Any unmet condition is emitted as an
    :data:`~src.rag.agent.obligations.ENUMERATION` obligation and the result is coded
    :data:`EVIDENCE_INCOMPLETE` — i.e. the enumeration becomes an evidence obligation to discharge (or a
    coded abstain), never a partial guess.
    """
    st = _coerce_state(state)
    unmet = tuple(c for c in CLOSURE_CONDITIONS if not st.get(c))
    if not unmet:
        return ClosureResult(satisfied=True, unmet=(), obligations=(), state_code="")
    obligations = tuple(
        ob.Obligation(
            obligation=f"閉包条件を満たす: {CLOSURE_LABELS[c]}",
            kind=ob.ENUMERATION,
            status=ob.UNMET,
        )
        for c in unmet
    )
    return ClosureResult(satisfied=False, unmet=unmet, obligations=obligations,
                         state_code=EVIDENCE_INCOMPLETE)


def _coerce_state(state: "ClosureState | dict | None") -> ClosureState:
    if isinstance(state, ClosureState):
        return state
    if isinstance(state, dict):
        return ClosureState(**{c: bool(state.get(c, False)) for c in CLOSURE_CONDITIONS})
    return ClosureState()


# --------------------------------------------------------------------------- injected procedure (手順注入)
def closure_procedure(question: str, population_kind: str | None = None) -> str:
    """The full-enumeration closure procedure injected into the investigator (手順注入).

    Fact-free by construction: it names the *kind* of authoritative source to resolve and the closure
    conditions to prove, but injects **no** corpus answer (no 略称, no member list) — the model reads the
    master itself via the tools, preserving the investigator's no-fact-injection invariant.
    """
    kind = population_kind or detect_population_kind(question)
    label = POPULATION_LABELS.get(kind, kind)
    source_hint = _SOURCE_HINT.get(kind, _SOURCE_HINT[GENERIC])
    order = detect_ordering_rule(question, kind)
    order_label = ORDERING_LABELS.get(order, order)
    conditions = "\n".join(f"  {i}. {CLOSURE_LABELS[c]}" for i, c in enumerate(CLOSURE_CONDITIONS, 1))
    return (
        "完全列挙の質問です。検索上位k件を母集団と見なして棄権してはいけません。次の手順で閉包を確認してください。\n"
        f"1) 権威的母集団の特定【{label}】: {source_hint} を読み、全体集合(母集団)を先に確定する"
        "(検索の当たりではなく、一覧表/マスター/目録を典拠にする)。\n"
        "2) 母集団の各要素を検査し、包含/除外の理由を1件ずつ付ける。\n"
        "3) 別の検索経路(別ツール/別索引/近傍走査)でもう一度当て、新規候補がゼロであることを確かめる。\n"
        "4) 列挙件数を別経路の集計件数と突き合わせ、一致を確認する。\n"
        "次の閉包条件がすべて充足したときのみ回答してよい:\n"
        f"{conditions}\n"
        f"5) 回答は列挙順序規則【{order_label}】で並べ、重複0で submit_answer する。\n"
        "いずれかの閉包条件を確認できない場合に限り、EVIDENCE_INCOMPLETE(根拠が部分的)として再度棄権すること"
        "(件数を推測で埋めない)。"
    )


# --------------------------------------------------------------------------- flash arbiter (opt-in)
_FLASH_SYSTEM = (
    "あなたは完全列挙質問の『母集団の種別』を分類します。次の7種のいずれか1つをコード(英字)で返してください:\n"
    "project=案件/会社 / person=人物/担当者 / abbrev=略称/コード / document=文書/ファイル / "
    "task=タスク/工程ID / seat=座席 / generic=特定できない一般集合。\n"
    "何を漏れなく挙げるべきかだけで1つ選び、コードのみ返すこと。"
)

_FLASH_SCHEMA = {
    "type": "object",
    "properties": {"kind": {"type": "string", "enum": list(POPULATION_KINDS)}},
    "required": ["kind"],
}


def flash_population_kind(question: str, *, model: str | None = None,
                          generate: Callable[..., str] | None = None) -> str | None:
    """Production flash arbiter: ask ``gemini-2.5-flash`` for one population-kind code; ``None`` on failure.

    Thin wrapper over :func:`src.rag.llm.generate` (imported lazily so importing this module never reaches
    the network). Returns a validated kind, or ``None`` when unreachable / off-vocabulary (the caller then
    keeps its deterministic :data:`GENERIC` default). ``generate`` may be injected in tests.
    """
    import json

    if generate is None:
        from src.rag import llm  # lazy: the google-genai client is only needed in production
        generate = llm.generate

    try:
        raw = generate(
            question, system=_FLASH_SYSTEM, model=model or settings.GEN_MODEL,
            temperature=0.0, max_output_tokens=32, response_schema=_FLASH_SCHEMA,
        )
    except Exception:  # noqa: BLE001 — arbitration is best-effort; fall back to the deterministic default
        return None
    code = None
    try:
        code = json.loads(raw).get("kind")
    except (ValueError, AttributeError):
        code = raw.strip()
    return code if code in POPULATION_KINDS else None


# --------------------------------------------------------------------------- investigator interceptor
# Terminal reasons for the gate (recorded for symmetry with the research director's trace).
INJECTED = "procedure_injected"   # the closure procedure was fed back once
NOT_ENUMERATION = "not_enumeration"


class EnumerationGate:
    """One-shot closure-procedure interceptor for a full-enumeration question.

    Lifecycle (called by :func:`~src.rag.agent.investigator.investigate` when enabled):

    * :meth:`is_enumeration` — whether the question is a ``full_enumeration`` contract (deterministic
      :func:`src.rag.agent.question_contract.classify`; an injected ``classify`` is used in tests).
    * :meth:`review` — invoked when the model is about to abstain. Returns the closure *procedure*
      directive the **first** time (for an enumeration question), then ``None`` — a single guided retry so
      the model resolves the authoritative population and proves closure before it may abstain again. For a
      non-enumeration question it always returns ``None`` (byte-identical to no gate).
    """

    def __init__(self, question: str, *, contract: str | None = None,
                 classify: Callable[[str], object] | None = None) -> None:
        self.question = question
        self._contract = contract
        self._classify = classify
        self._is_enum: bool | None = None
        self.injected = False
        self.terminal = ""

    def is_enumeration(self) -> bool:
        if self._is_enum is None:
            if self._contract is not None:
                self._is_enum = self._contract == qc.FULL_ENUMERATION
            else:
                classify = self._classify or qc.classify
                try:
                    self._is_enum = getattr(classify(self.question), "contract", None) == qc.FULL_ENUMERATION
                except Exception:  # noqa: BLE001 — classification is best-effort; treat as non-enum
                    self._is_enum = False
        return self._is_enum

    def review(self) -> str | None:
        """Inject the closure procedure once for an enumeration abstain; ``None`` otherwise."""
        if not self.is_enumeration():
            self.terminal = self.terminal or NOT_ENUMERATION
            return None
        if self.injected:
            return None
        self.injected = True
        self.terminal = INJECTED
        return closure_procedure(self.question)

    def trace(self) -> dict[str, object]:
        return {"is_enumeration": self.is_enumeration(), "injected": self.injected,
                "terminal": self.terminal}
