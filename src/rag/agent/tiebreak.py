"""SOT-2471 — third-judge tie-break + abstain fallback (合議統括層).

Parent SOT-2460 Step2. This module orchestrates the two independent answer paths — the primary
:mod:`~src.rag.agent.investigator` (``gemini-2.5-pro``) and the heterogeneous
:mod:`~src.rag.agent.verifier` (``gemini-2.5-flash``) — into a single adjudicated answer, automating the
manual 30/30 tie-break step:

* **一致 → 採用**  — investigator と verifier が :func:`~src.rag.agent.verifier.compare_answers`
  （値/列挙/数値丸め/棄権の4軸）で一致すれば、primary(調査AG)の回答をそのまま採用する。
* **不一致 → 第3判定 (tie-break)** — 第3の Gemini 審判 (既定 :data:`~config.settings.JUDGE_MODEL`
  = ``gemini-2.5-pro``) が、**別プロンプト・別プロファイル**で質問を **独立に再導出** し、その回答を
  2候補と ``compare_answers`` で照合して投票する。審判が片方のみと一致 → その候補を採用。
* **依然不一致 → 棄権** — 審判がどちらの候補とも一致しない（独立に第3の値／棄権）か、両候補と同時に
  一致する（矛盾）場合は *決着不能* とみなし ``ABSTAIN`` へフォールバックする（Incorrect=−1 に対し
  Missing=0 で EV 正）。

Why the judge re-derives independently (and is NOT shown the two candidates)
---------------------------------------------------------------------------
Feeding the two conflicting candidate answers into the judge's prompt would correlate the judge's error
with whichever candidate it anchors on — exactly the coupling the heterogeneous verifier exists to
break. Instead the judge solves the question from scratch (its own tools + fresh
:class:`~src.rag.tools.profile.CorpusProfile`, a distinct 審判 prompt, and by default the *pro* tier),
and its answer casts a majority-style vote against the two candidates via the same external comparison
axes. A judge that itself abstains (or lands on a third value) matches neither candidate → 棄権, which is
the correct conservative outcome under the −1/0 scoring.

Design mirrors the investigator/verifier: the driver :func:`resolve_answer` is a *pure* function over the
abstract :class:`~src.rag.agent.investigator.Model` (scripted fakes in tests, no Vertex), and
:func:`resolve_question` injects the live Gemini paths. Every decision records a human-readable rationale
(``reason`` / ``judge_reason``) so the adjudication is auditable — 判定根拠をログ化.
"""
from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any, Callable, Sequence

from config import settings
from src.rag.agent.investigator import (
    DEFAULT_MAX_TURNS,
    DEFAULT_TIMEOUT_S,
    AgentTool,
    Answer,
    Investigation,
    Model,
    build_tools,
    investigate,
    is_abstain,
)
from src.rag.agent.verifier import Verdict, compare_answers, verify_answer
from src.rag.tools.profile import CorpusProfile

ABSTAIN = settings.ABSTAIN

# The tie-break judge deliberately runs the *hard* tier: when the primary(pro) and the heterogeneous
# verifier(flash) split, the arbiter should be at least as strong as the primary. Defaults to the
# dedicated JUDGE_MODEL (pro) so the judge tier is independently configurable from the answer path.
DEFAULT_JUDGE_MODEL = settings.JUDGE_MODEL  # gemini-2.5-pro

# Resolution status values (which branch of the 合議 decided the final answer).
STATUS_AGREED = "agreed"                       # 調査AG・検証AGが一致 → 採用
STATUS_TIEBREAK_INVESTIGATOR = "tiebreak_investigator"  # 第3判定が調査AG側を支持
STATUS_TIEBREAK_VERIFIER = "tiebreak_verifier"          # 第3判定が検証AG側を支持
STATUS_ABSTAINED = "abstained"                 # 決着不能 → 棄権

JUDGE_SYSTEM_PROMPT = (
    "あなたは第3の独立審判エージェントです。ある質問について2つの調査エージェントが異なる答えを出して"
    "対立しています。ただし他エージェントの回答は一切与えられません。あなたは両者と相関しない観点から、"
    "与えられた汎用ツールだけで質問の答えを **一から自力で再導出** し、正しい答えを確定します。あなたの"
    "答えが、対立する2候補のどちらが正しいかを決める決定票になります。\n"
    "厳守事項:\n"
    "1. 暗算・記憶・創作で答えない。必ずツールで根拠となる値を取得してから答える。他者の回答に引きずられ"
    "ない(そもそも与えられない)。\n"
    "2. パスワード・略称・書式規則などのコーパス固有事実は与えられていない。ツール(ファイル探索/grep/"
    "Office抽出/復号/pandas計算/チャート読取/画像説明)で自力発見する。\n"
    "3. 判定では次を特に厳密に確認する: (a) 列挙は漏れ・重複なく完全か、(b) 数値は正しい丸め・単位・原文"
    "表記か、(c) 本当に該当が存在するか(存在しないなら値を作らない)。\n"
    "4. 数値計算(平均・合計・件数など)は必ず compute ツールで行い、列名や絞り込み値が不明なときは"
    "`df.columns.tolist()` や `df['列'].unique().tolist()` で先に確認してから式を組む。\n"
    "5. 十分な根拠が得られたら、最終回答は必ず submit_answer ツールを1回だけ呼んで返す。answer=回答本文"
    "(値/一覧のみ、列挙は「、」区切り、金額は原文表記)、confidence=0.0〜1.0、evidence=根拠、method=導出手順。\n"
    f"6. あらゆる手段を尽くしても根拠が該当しない/得られない場合に限り answer=「{ABSTAIN}」・"
    "confidence=0.0 で submit_answer する(存在しない値を捏造しない)。決着できないなら無理に値を作らない。"
)


# --------------------------------------------------------------------------- resolution result type
@dataclass(frozen=True)
class Resolution:
    """The adjudicated outcome for one question after the 一致→採用 / 不一致→第3判定 / 決着不能→棄権 flow."""

    question: str
    answer: str                 # final adopted answer (or ABSTAIN when 決着不能)
    confidence: float           # confidence of the adopted side (0.0 when abstained)
    status: str                 # STATUS_* — which branch decided the outcome
    reason: str                 # human-readable 判定根拠 (auditable log line)
    agree: bool                 # did investigator & verifier agree (no tie-break needed)?
    investigator_answer: str
    verifier_answer: str
    verdict: Verdict            # the investigator-vs-verifier comparison (SOT-2470)
    judge_answer: str | None = None    # None when no tie-break ran (they agreed)
    judge_confidence: float = 0.0
    judge_reason: str = ""      # rationale of the third-judge vote (empty when no tie-break ran)
    evidence: str = ""          # 根拠 of the adopted answer
    method: str = ""            # derivation summary of the adopted answer
    calc_record: dict[str, Any] | None = None  # investigator's in-memory calculation evidence

    @property
    def abstained(self) -> bool:
        """The final adopted answer is 棄権 — either 決着不能 (STATUS_ABSTAINED) or a tie-break that
        sided with an abstaining candidate. Path-independent, so callers can count abstentions."""
        return is_abstain(self.answer)

    @property
    def tie_broken(self) -> bool:
        return self.status in (STATUS_TIEBREAK_INVESTIGATOR, STATUS_TIEBREAK_VERIFIER)

    def to_answer(self) -> Answer:
        """The adopted answer as the shared :class:`~src.rag.agent.investigator.Answer` schema."""
        return Answer(answer=self.answer, confidence=self.confidence,
                      evidence=self.evidence, method=self.method)

    def to_dict(self) -> dict[str, Any]:
        return {
            "question": self.question,
            "answer": self.answer,
            "confidence": self.confidence,
            "status": self.status,
            "reason": self.reason,
            "agree": self.agree,
            "investigator_answer": self.investigator_answer,
            "verifier_answer": self.verifier_answer,
            "judge_answer": self.judge_answer,
            "judge_confidence": self.judge_confidence,
            "judge_reason": self.judge_reason,
            "evidence": self.evidence,
            "method": self.method,
            "verdict": self.verdict.to_dict(),
        }


# --------------------------------------------------------------------------- pure tie-break core
def _tie_break(judge: Answer, investigator: Answer, verifier: Answer) -> tuple[str, str]:
    """Cast the judge's independent answer as a vote between the two conflicting candidates.

    Returns ``(status, reason)``. The judge matches a candidate under the same four comparison axes used
    for the investigator↔verifier reconciliation:

    * matches the investigator only → adopt investigator;
    * matches the verifier only → adopt verifier;
    * matches neither (independent third value / 棄権) OR matches both (contradiction) → 決着不能 → 棄権.
    """
    ji = compare_answers(judge.answer, investigator.answer)
    jv = compare_answers(judge.answer, verifier.answer)

    if ji.agree and not jv.agree:
        return STATUS_TIEBREAK_INVESTIGATOR, f"第3判定が調査AGを支持({ji.reason})"
    if jv.agree and not ji.agree:
        return STATUS_TIEBREAK_VERIFIER, f"第3判定が検証AGを支持({jv.reason})"
    if ji.agree and jv.agree:
        # The judge matched both, yet the two disagreed with each other — inconsistent, cannot decide.
        return STATUS_ABSTAINED, (
            "決着不能: 第3判定が両候補と一致したが両候補は相互に不一致(矛盾) — 棄権")
    return STATUS_ABSTAINED, (
        f"決着不能: 第3判定「{judge.answer}」がどちらの候補とも不一致 — 棄権")


# --------------------------------------------------------------------------- resolution assembly
def _agreed_resolution(question: str, inv: Answer, ver: Answer, verdict: Verdict) -> Resolution:
    """一致→採用: adopt the primary(investigator) answer with its evidence/method."""
    return Resolution(
        question=question, answer=inv.answer, confidence=inv.confidence,
        status=STATUS_AGREED, reason=f"調査AGと検証AGが一致 — 採用({verdict.reason})",
        agree=True, investigator_answer=inv.answer, verifier_answer=ver.answer,
        verdict=verdict, evidence=inv.evidence, method=inv.method,
    )


def _resolve_from_answers(question: str, inv: Answer, ver: Answer, verdict: Verdict,
                          judge: Answer | None) -> Resolution:
    """Pure reconciliation of the three answers into a :class:`Resolution`.

    ``judge`` is ``None`` when the investigator and verifier already agreed (no tie-break was run).
    """
    if verdict.agree:
        return _agreed_resolution(question, inv, ver, verdict)

    if judge is None:  # defensive: a disagreement must be tie-broken; without a judge, abstain.
        return Resolution(
            question=question, answer=ABSTAIN, confidence=0.0, status=STATUS_ABSTAINED,
            reason=f"不一致だが第3判定が得られず決着不能 — 棄権({verdict.reason})",
            agree=False, investigator_answer=inv.answer, verifier_answer=ver.answer,
            verdict=verdict, judge_answer=None,
        )

    status, judge_reason = _tie_break(judge, inv, ver)
    reason = f"調査AGと検証AGが不一致({verdict.reason}) → {judge_reason}"

    if status == STATUS_TIEBREAK_INVESTIGATOR:
        adopted = inv
    elif status == STATUS_TIEBREAK_VERIFIER:
        adopted = ver
    else:  # STATUS_ABSTAINED
        adopted = Answer(answer=ABSTAIN, confidence=0.0)

    return Resolution(
        question=question, answer=adopted.answer, confidence=adopted.confidence,
        status=status, reason=reason, agree=False,
        investigator_answer=inv.answer, verifier_answer=ver.answer, verdict=verdict,
        judge_answer=judge.answer, judge_confidence=judge.confidence, judge_reason=judge_reason,
        evidence=adopted.evidence, method=adopted.method,
    )


# --------------------------------------------------------------------------- independent judge run
def _judge_investigation(question: str, *, model: Model | None = None,
                         model_factory: Callable[[str, Sequence[AgentTool]], Model] | None = None,
                         profile_factory: Callable[[], CorpusProfile] | None = None,
                         max_turns: int = DEFAULT_MAX_TURNS,
                         timeout_s: float = DEFAULT_TIMEOUT_S) -> Investigation:
    """Run the third judge's independent re-derivation (its own tools bound to a fresh profile)."""
    prof = (profile_factory or CorpusProfile)()
    tools = build_tools(prof)
    if model is None:
        factory = model_factory or judge_model_factory
        model = factory(question, tools)
    return investigate(model, question, tools, max_turns=max_turns, timeout_s=timeout_s)


# --------------------------------------------------------------------------- resolution driver (pure)
def resolve_answer(question: str, investigator: Investigation, *,
                   verifier_model: Model | None = None,
                   verifier_factory: Callable[[str, Sequence[AgentTool]], Model] | None = None,
                   verifier_profile_factory: Callable[[], CorpusProfile] | None = None,
                   judge_model: Model | None = None,
                   judge_factory: Callable[[str, Sequence[AgentTool]], Model] | None = None,
                   judge_profile_factory: Callable[[], CorpusProfile] | None = None,
                   max_turns: int = DEFAULT_MAX_TURNS,
                   timeout_s: float = DEFAULT_TIMEOUT_S) -> Resolution:
    """Adjudicate an already-completed ``investigator`` :class:`Investigation` into a :class:`Resolution`.

    Runs the heterogeneous verifier against the investigator's answer; on 一致 adopts the primary answer,
    on 不一致 runs the independent third judge and tie-breaks (or abstains when 決着不能). Supply scripted
    ``verifier_model`` / ``judge_model`` (offline tests) or ``*_factory`` (live); by default the live
    Gemini paths are used. The judge is only invoked when the first two disagree — no wasted call on
    agreement.
    """
    verdict = verify_answer(
        question, investigator.answer.answer,
        model=verifier_model, model_factory=verifier_factory,
        profile_factory=verifier_profile_factory, max_turns=max_turns, timeout_s=timeout_s,
    )

    if verdict.agree:
        resolution = _resolve_from_answers(
            question, investigator.answer, _verdict_answer(verdict), verdict, judge=None)
        return replace(resolution, calc_record=investigator.calc_record)

    judge = _judge_investigation(
        question, model=judge_model, model_factory=judge_factory,
        profile_factory=judge_profile_factory, max_turns=max_turns, timeout_s=timeout_s,
    ).answer
    resolution = _resolve_from_answers(
        question, investigator.answer, _verdict_answer(verdict), verdict, judge=judge)
    return replace(resolution, calc_record=investigator.calc_record)


def _verdict_answer(verdict: Verdict) -> Answer:
    """Reconstruct the verifier's :class:`Answer` from its :class:`Verdict` (for uniform handling)."""
    return Answer(answer=verdict.verifier_answer, confidence=verdict.verifier_confidence,
                  evidence=verdict.verifier_evidence, method=verdict.verifier_method)


# --------------------------------------------------------------------------- live Gemini judge path
def judge_model_factory(question: str, tools: Sequence[AgentTool], *,
                        model: str | None = None) -> Model:
    """Fresh live Gemini conversation for the third judge: the *pro* tier + the 審判 prompt."""
    from src.rag.agent.investigator import GeminiModel, to_genai_tools

    return GeminiModel(question, to_genai_tools(tools),
                       model=model or DEFAULT_JUDGE_MODEL, system=JUDGE_SYSTEM_PROMPT)


def resolve_question(question: str, *,
                     investigator_model: str | None = None,
                     verifier_model: str | None = None,
                     judge_model: str | None = None,
                     max_turns: int = DEFAULT_MAX_TURNS,
                     timeout_s: float = DEFAULT_TIMEOUT_S) -> Resolution:
    """Convenience live entry point: investigate → heterogeneously verify → third-judge tie-break.

    Wires the three live Gemini paths end-to-end (investigator pro, verifier flash, judge pro) and
    returns the adjudicated :class:`Resolution`. Each role gets its own isolated
    :class:`~src.rag.tools.profile.CorpusProfile` so no self-discovered secret leaks across roles.
    """
    from src.rag.agent.investigator import answer_question

    investigation = answer_question(question, model=investigator_model,
                                    max_turns=max_turns, timeout_s=timeout_s)
    return resolve_answer(
        question, investigation,
        verifier_factory=lambda q, t: _verifier_live(q, t, model=verifier_model),
        judge_factory=lambda q, t: judge_model_factory(q, t, model=judge_model),
        max_turns=max_turns, timeout_s=timeout_s,
    )


def _verifier_live(question: str, tools: Sequence[AgentTool], *, model: str | None = None) -> Model:
    from src.rag.agent.verifier import verifier_model_factory

    return verifier_model_factory(question, tools, model=model)
