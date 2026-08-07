"""SOT-2502 — obligation-driven local re-search loop *before* an abstain.

Parent SOT-2460. A low-confidence turn must not be an *immediate* abstain: it is the trigger to start
an **additional evidence-acquisition phase** (ReCoVERR-style). This module is that phase's director.
When the investigator is about to abstain, the director:

1. Decomposes the question into its evidence obligations (SOT-2499, :mod:`src.rag.agent.obligations`)
   once, and discharges them against the evidence actually collected so far (successful tool results +
   the model's own evidence note) to find the **unmet** obligations.
2. Picks the next unmet obligation *kind* not yet re-searched and emits a **targeted** re-search
   directive — per-kind tactics (略称/正式名称展開・entity 展開・file_grep 字面検索・canonical 直行・
   別索引・局所拡大). This re-searches **only the unmet obligation**, never the whole corpus.
3. Bounds the phase with a budget (反復回数・ツール呼数; wall-clock is bounded by the investigator's
   outer ``timeout_s``). On budget exhaustion the abstain is coded :data:`BUDGET`; after every unmet
   obligation's tactic has been tried and nothing new surfaced it is coded :data:`UNANSWERABLE`.

Invariants
----------
* **Commit criterion unchanged.** The director never touches confidence or the commit-vs-abstain
  threshold (SOT-2483 の EV-negative 軸とは別物): it grows *search*, not *commit*. It only ever turns an
  abstain into either a grounded answer (the model finds new evidence) or a *coded, history-bearing*
  abstain — never a low-confidence guess.
* **即棄権が構造上不可能.** A deliberate abstain (``submit_answer=わかりません``) with budget remaining
  is structurally forced through ≥1 re-search round; the round trace is written to the abstain ledger
  (SOT-2492) so 探索履歴 always remains.
* **Observer-safe when off.** The investigator gates the whole director behind ``research`` being
  enabled, so with it off the answer path is byte-identical.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Mapping

from src.rag.agent import obligations as ob
from src.rag.agent.obligations import (
    COMPUTATION,
    CONSISTENCY,
    ENUMERATION,
    JUDGMENT_RULE,
    KIND_LABELS,
    KINDS,
    QUANTITY_DEFINITION,
    ROUNDING,
    SOURCE_LOCATION,
    UNIT_NORMALIZATION,
    VALUE_MAPPING,
)
from src.rag.tools import contract as _contract

# --------------------------------------------------------------------------- terminal reasons
# Why the re-search phase stopped; mapped onto the abstain ledger's state codes by the investigator.
BUDGET = "budget"              # 反復/ツール呼の上限に到達 → BUDGET_EXHAUSTED
UNANSWERABLE = "unanswerable"  # 全戦術を尽くしても根拠が現れず → UNANSWERABLE (不存在確認)
ANSWERED = "answered"          # 再探索で根拠に到達し回答へ転じた

# --------------------------------------------------------------------------- per-kind re-search tactics
# Each unmet obligation kind maps to concrete, targeted re-search tactics (never a whole-corpus retry).
TACTICS: dict[str, tuple[str, ...]] = {
    SOURCE_LOCATION: (
        "会社名/文書名を略称⇔正式名称で展開し find_files を撃ち直す(会社名の一部・別表記・canonical名)",
        "file_grep で対象語を字面(regex=false/true)検索し、別ファイル・別スライド・別シートを当てる",
    ),
    JUDGMENT_RULE: (
        "書式/空間/版差分の規則ツール(highlight_extract・seating_lookup・version_diff)へ質問文を字面で当てる",
        "判定属性(太字/下線/色/隣接/変更点)を1つに絞って該当箇所だけを再抽出する",
    ),
    ENUMERATION: (
        "file_grep / corpus_aggregate で母集団を広げ、近傍(前後行・同一シート・同一フォルダ)まで局所拡大して網羅する",
        "列挙の閉包条件(全件・重複0・非該当除外)を1件ずつ突き合わせる",
    ),
    VALUE_MAPPING: (
        "中間エンティティ(担当者/案件/期間)を別ツールで一段ずつ確定してから対応づける(entity 展開)",
        "別索引(seating_lookup・read_office・read_chart_values)で 値⇔エンティティ を引き直す",
    ),
    QUANTITY_DEFINITION: (
        "質問を分子・分母・母集団に書き下し、『XのうちYの割合』ではXだけを分母として compute し直す",
        "分子件数と分母件数を別々の compute 証跡として取得してから比率を再計算する",
    ),
    UNIT_NORMALIZATION: (
        "compute で桁/単位(円・万円・千円・件数・日数)を質問の要求どおりに正規化し直す",
    ),
    ROUNDING: (
        "未丸めの計算値を保持し、質問指定の小数第N位/切上げ/切捨てを最後の一段だけで適用する",
    ),
    COMPUTATION: (
        "compute で df.columns / df['列'].unique() を確認してから式を組み、決定論的に再計算する",
        "全案件横断なら corpus_aggregate に metric/op を指定して再計算する",
    ),
    CONSISTENCY: (
        "得られた値を別経路で再計算し、突き合わせて整合(矛盾なし)を確認する",
    ),
}

# Fallback tactic for any kind without an explicit entry (keeps the directive non-empty).
_GENERIC_TACTIC = "対象語の別表記・略称展開で file_grep の字面検索を撃ち直す"

# Which obligation kinds a *successful* tool result can plausibly discharge (a precise evidence signal
# so the unmet set reflects "kinds with no successful tool coverage yet", not prose cues alone).
_TOOL_KINDS: dict[str, frozenset[str]] = {
    "find_files": frozenset({SOURCE_LOCATION}),
    "file_grep": frozenset({SOURCE_LOCATION, ENUMERATION}),
    "read_office": frozenset({VALUE_MAPPING, SOURCE_LOCATION}),
    "decrypt": frozenset({SOURCE_LOCATION}),
    "read_chart_values": frozenset({VALUE_MAPPING, UNIT_NORMALIZATION}),
    "caption_image": frozenset({VALUE_MAPPING}),
    "pdf_emphasis": frozenset({JUDGMENT_RULE}),
    "pptx_pivot": frozenset({VALUE_MAPPING, ENUMERATION}),
    "highlight_extract": frozenset({JUDGMENT_RULE, ENUMERATION}),
    "compute": frozenset({COMPUTATION, UNIT_NORMALIZATION}),
    "corpus_aggregate": frozenset({COMPUTATION, ENUMERATION}),
    "seating_lookup": frozenset({VALUE_MAPPING, JUDGMENT_RULE}),
    "version_diff": frozenset({JUDGMENT_RULE, SOURCE_LOCATION}),
}


def _succeeded(resp: Any) -> bool:
    """Whether one dispatched tool response carries usable evidence (mirrors the ledger's outcome sense).

    An ``{"error": …}`` mapping fails; a ``{value, evidence, method}`` contract succeeds only with a
    non-empty ``value``; a bare non-empty list/mapping/scalar succeeds.
    """
    if isinstance(resp, Mapping):
        if "error" in resp:
            return False
        if _contract.is_contract(resp):
            return resp.get("value") not in (None, "", [], {})
        return bool(resp)
    if isinstance(resp, (list, tuple)):
        return len(resp) > 0
    return resp not in (None, "")


def _resp_text(tool_name: str, resp: Any) -> str:
    """A short, cue-matchable text view of a successful tool response (bounded for cost)."""
    return f"{tool_name}: {resp}"[:400]


# --------------------------------------------------------------------------- budget / round records
@dataclass(frozen=True)
class ResearchBudget:
    """Bounds on the re-search phase (wall-clock is bounded by the investigator's outer ``timeout_s``)."""

    max_rounds: int = 2       # distinct unmet-obligation kinds re-searched before giving up
    max_tool_calls: int = 8   # non-submit tool calls already spent; past this, re-search stops

    @classmethod
    def coerce(cls, spec: "ResearchBudget | Mapping[str, Any] | bool | None") -> "ResearchBudget":
        """Build a budget from a :class:`ResearchBudget`, a ``{max_rounds, max_tool_calls}`` mapping, or
        a truthy/absent flag (defaults)."""
        if isinstance(spec, ResearchBudget):
            return spec
        if isinstance(spec, Mapping):
            return cls(
                max_rounds=int(spec.get("max_rounds", cls.max_rounds)),
                max_tool_calls=int(spec.get("max_tool_calls", cls.max_tool_calls)),
            )
        return cls()


@dataclass(frozen=True)
class ResearchRound:
    """One targeted re-search round the director issued (recorded into the abstain ledger)."""

    round: int
    kind: str
    obligations: tuple[str, ...]
    tactics: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "round": self.round,
            "kind": self.kind,
            "obligations": list(self.obligations),
            "tactics": list(self.tactics),
        }


def _directive(kind: str, obligations: tuple[ob.Obligation, ...], tactics: tuple[str, ...]) -> str:
    """The targeted re-search instruction fed back to the model in place of accepting the abstain."""
    ob_text = "／".join(o.obligation for o in obligations)
    tac_text = "\n".join(f"  - {t}" for t in tactics)
    return (
        "棄権の前に、まだ充足していない証拠義務『だけ』を局所的に再探索してください(全体の再検索はしない)。\n"
        f"未充足の証拠義務【{KIND_LABELS.get(kind, kind)}】: {ob_text}\n"
        f"次の再探索戦術を targeted に試すこと:\n{tac_text}\n"
        "新たな根拠が得られたら submit_answer で回答を確定してください。あらゆる戦術を尽くしても根拠が"
        "コーパスに存在しない場合に限り、再度 submit_answer で棄権してください(確信度は操作しない)。"
    )


# --------------------------------------------------------------------------- the director
class ResearchDirector:
    """Drives the obligation-driven local re-search phase for one question.

    Lifecycle (called by the investigator loop):

    * :meth:`observe` — fold every dispatched tool result into the collected-evidence view (successful
      results only), so the unmet-obligation set reflects real tool coverage.
    * :meth:`review` — invoked when the model is about to abstain. Returns a targeted re-search
      *directive* to keep searching, or ``None`` to finalize the abstain (setting :attr:`terminal`).
    * :meth:`note_answered` — the model committed a real answer (re-search paid off).
    * :meth:`trace` — the ordered round history recorded into the abstain ledger.
    """

    def __init__(self, question: str, *,
                 budget: "ResearchBudget | Mapping[str, Any] | bool | None" = None,
                 decompose: Callable[[str], ob.ObligationSet] | None = None) -> None:
        self.question = question
        self.budget = ResearchBudget.coerce(budget)
        self._decompose = decompose or ob.decompose
        self._oblset: ob.ObligationSet | None = None
        self._evidence: list[ob.EvidenceItem] = []
        self._targeted: set[str] = set()
        self.rounds: list[ResearchRound] = []
        self.terminal: str = ""

    # -- obligations (decomposed lazily, once) --------------------------------------------------------
    def obligations(self) -> ob.ObligationSet:
        if self._oblset is None:
            self._oblset = self._decompose(self.question)
        return self._oblset

    # -- evidence collection (observer) ---------------------------------------------------------------
    def observe(self, tool_name: str, resp: Any) -> None:
        """Record a successful tool result as collected evidence (kinds tagged from :data:`_TOOL_KINDS`)."""
        if _succeeded(resp):
            self._evidence.append(ob.EvidenceItem(
                text=_resp_text(tool_name, resp), ref=tool_name,
                kinds=_TOOL_KINDS.get(tool_name, frozenset())))

    def _unmet(self, extra_evidence: str = "") -> tuple[ob.Obligation, ...]:
        items = list(self._evidence)
        if extra_evidence.strip():
            items.append(ob.EvidenceItem(text=extra_evidence))
        return self.obligations().unmet(items)

    # -- decision -------------------------------------------------------------------------------------
    def review(self, evidence_text: str = "", tool_call_count: int = 0) -> str | None:
        """Decide whether to keep re-searching before accepting an abstain.

        Returns a targeted re-search directive (continue) or ``None`` (finalize). On ``None`` it sets
        :attr:`terminal` to :data:`BUDGET` (budget exhausted) or :data:`UNANSWERABLE` (every unmet
        obligation's tactic already tried — thorough non-existence).
        """
        if len(self.rounds) >= self.budget.max_rounds or tool_call_count >= self.budget.max_tool_calls:
            self.terminal = BUDGET
            return None
        unmet = self._unmet(evidence_text)
        fresh = [o for o in unmet if o.kind not in self._targeted]
        if not fresh:
            # nothing unmet, or every open kind already re-searched → searched thoroughly, none found
            self.terminal = UNANSWERABLE
            return None
        kind = fresh[0].kind  # obligations keep discharge order → most fundamental gap first
        target = tuple(o for o in unmet if o.kind == kind)
        tactics = TACTICS.get(kind, (_GENERIC_TACTIC,))
        self._targeted.add(kind)
        self.rounds.append(ResearchRound(
            round=len(self.rounds) + 1, kind=kind,
            obligations=tuple(o.obligation for o in target), tactics=tactics))
        return _directive(kind, target, tactics)

    def note_answered(self) -> None:
        self.terminal = ANSWERED

    def trace(self) -> tuple[dict[str, Any], ...]:
        return tuple(r.to_dict() for r in self.rounds)
