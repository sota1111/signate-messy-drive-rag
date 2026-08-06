"""SOT-2473 — confidence-gated abstention on verifier consensus (EV最適 commit/棄権ゲート).

Parent SOT-2460 Step2. This is the final *precision-first* decision layer that sits on top of the
合議 (:mod:`~src.rag.agent.tiebreak`): given an adjudicated :class:`~src.rag.agent.tiebreak.Resolution`
(調査AG + 検証AG + 必要時 tie-break) it decides whether to **commit** the answer or emit
「わかりません」.

Why a separate gate on top of the consensus
-------------------------------------------
The official metric scores **Correct = +1 / Incorrect = −1 / Missing (abstain) = 0**, so committing an
answer of estimated precision ``p`` has expected value ``2p − 1``: committing only pays off when
``p > 0.5``. The consensus layer already abstains when it cannot adjudicate (決着不能 → 棄権), but two
residual cases still carry EV-negative risk if committed verbatim:

* **低確信** — 調査AGと検証AGが一致していても自己申告 confidence が低い答えは precision が 0.5 を割りうる。
* **不一致→tie-break** — 第3判定で 2/3 の多数決に決着した答えは、そもそも一次意見が割れた分だけ一致案より
  precision が低い。Incorrect=−1 のもとでは、僅差の tie-break commit は棄権(0)に劣りうる。

So the gate commits **only 合議一致(``STATUS_AGREED``)かつ高確信**; every other outcome —
低確信の一致 / tie-break / 決着不能 — falls back to 棄権. This mirrors the text-path precision-first gate
in :mod:`src.rag.generate` (abstain-leaning, Missing 0 beats Incorrect −1), now expressed over the agent
consensus. Each decision records a human-readable ``reason`` so the commit/abstain call is auditable.

The core :func:`apply_gate` is a **pure** function over a :class:`Resolution` (no Vertex, trivially
testable); :func:`gate_question` wires the live consensus (:func:`~src.rag.agent.tiebreak.resolve_question`)
end-to-end and applies the gate. Thresholds are read from the environment (``config/`` is a protected
path), fully overridable per run.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any

from config import settings
from src.rag.agent.investigator import Answer
from src.rag.agent.tiebreak import (
    STATUS_AGREED,
    STATUS_TIEBREAK_INVESTIGATOR,
    STATUS_TIEBREAK_VERIFIER,
    Resolution,
    resolve_question,
)

ABSTAIN = settings.ABSTAIN


def _float_env(key: str, default: float) -> float:
    try:
        return float(os.getenv(key, str(default)))
    except (TypeError, ValueError):
        return default


def _bool_env(key: str, default: bool) -> bool:
    raw = os.getenv(key)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


# --- thresholds (env-overridable; config/ is protected so read directly here, cf. SOT-2446) --------
# Minimum self-reported confidence to commit a 合議一致 answer. Below this → 棄権. Chosen conservatively
# (>0.5 EV break-even) so a barely-confident agreement does not risk an Incorrect(−1).
GATE_COMMIT_CONFIDENCE = _float_env("GATE_COMMIT_CONFIDENCE", 0.7)
# Whether a tie-broken answer (一次意見が割れ、第3判定で決着) may commit at all. Default OFF: a disagreement
# abstains regardless of confidence (precision-first). When ON it may commit only above the stricter bar.
# SOT-2483 measured this knob on gold-100: committing the tie-break-adjudicated candidates scores
# EV −10 (3 Perfect vs 13 Incorrect) and every recovery source that adds a match also adds an
# Incorrect — so the abstain-leaning default is EV-correct. Keep OFF absent a real-LB / 関門2
# confirmation (see scoring/abstain_breakdown.py, artifacts/gold_abstain_breakdown.md).
GATE_COMMIT_ON_TIEBREAK = _bool_env("GATE_COMMIT_ON_TIEBREAK", False)
GATE_TIEBREAK_CONFIDENCE = _float_env("GATE_TIEBREAK_CONFIDENCE", 0.85)

_TIEBREAK_STATUSES = (STATUS_TIEBREAK_INVESTIGATOR, STATUS_TIEBREAK_VERIFIER)


# --------------------------------------------------------------------------- gate decision result
@dataclass(frozen=True)
class GateDecision:
    """The final commit/abstain call for one question after the precision-first confidence gate.

    ``answer`` is the *served* answer: the resolution's answer when ``commit`` is True, else
    :data:`ABSTAIN`. ``reason`` records the auditable 判定理由 (why it committed or abstained).
    """

    question: str
    answer: str            # served answer — the committed value, or ABSTAIN
    commit: bool           # True → commit the value; False → 棄権(わかりません)
    confidence: float      # confidence of the adjudicated answer (0.0 when the consensus abstained)
    reason: str            # human-readable 判定理由
    resolution: Resolution  # the underlying 合議 outcome (status/verdict/judge/evidence all preserved)

    @property
    def abstained(self) -> bool:
        return not self.commit

    def to_answer(self) -> Answer:
        """The served answer as the shared :class:`~src.rag.agent.investigator.Answer` schema."""
        return Answer(
            answer=self.answer,
            confidence=self.confidence if self.commit else 0.0,
            evidence=self.resolution.evidence if self.commit else "",
            method=self.resolution.method if self.commit else "",
        )

    def to_dict(self) -> dict[str, Any]:
        """Run-log record: the served answer/decision plus the full underlying resolution."""
        return {
            "question": self.question,
            "answer": self.answer,
            "commit": self.commit,
            "confidence": self.confidence,
            "reason": self.reason,
            "status": self.resolution.status,
            "gate_status": "commit" if self.commit else "abstain",
            "resolution": self.resolution.to_dict(),
            # Surface the consensus fields at top level so the run-log schema stays aligned with the
            # investigator/resolve backends (evidence/method/agree/…).
            "evidence": self.resolution.evidence if self.commit else "",
            "method": self.resolution.method if self.commit else "",
            "agree": self.resolution.agree,
        }


def _abstain(question: str, resolution: Resolution, reason: str) -> GateDecision:
    return GateDecision(
        question=question, answer=ABSTAIN, commit=False, confidence=0.0,
        reason=reason, resolution=resolution,
    )


# --------------------------------------------------------------------------- pure gate core
def apply_gate(resolution: Resolution, *,
               commit_confidence: float | None = None,
               commit_on_tiebreak: bool | None = None,
               tiebreak_confidence: float | None = None) -> GateDecision:
    """Apply the precision-first confidence gate to an adjudicated :class:`Resolution`.

    Commit rules (EV最適, Incorrect=−1 / Missing=0):

    * 合議が既に棄権(決着不能 / 棄権側へ決着) → そのまま棄権。
    * ``STATUS_AGREED`` かつ confidence ≥ ``commit_confidence`` → **commit**。
    * ``STATUS_AGREED`` だが confidence 不足 → 低確信につき棄権。
    * tie-break で決着(``STATUS_TIEBREAK_*``) → 既定は棄権。``commit_on_tiebreak`` 有効時のみ、より厳しい
      ``tiebreak_confidence`` を超えた場合に commit。

    Thresholds default to the module-level env-configured values; pass explicit overrides for tests.
    """
    commit_confidence = GATE_COMMIT_CONFIDENCE if commit_confidence is None else commit_confidence
    commit_on_tiebreak = GATE_COMMIT_ON_TIEBREAK if commit_on_tiebreak is None else commit_on_tiebreak
    tiebreak_confidence = GATE_TIEBREAK_CONFIDENCE if tiebreak_confidence is None else tiebreak_confidence

    q = resolution.question
    conf = resolution.confidence

    # 1) The consensus itself abstained (決着不能 or 棄権側へ tie-break). Never resurrect a value.
    if resolution.abstained:
        return _abstain(q, resolution, f"合議が棄権 — 棄権を維持({resolution.reason})")

    # 2) 合議一致 → precision-first confidence gate.
    if resolution.status == STATUS_AGREED:
        if conf >= commit_confidence:
            return GateDecision(
                question=q, answer=resolution.answer, commit=True, confidence=conf,
                reason=f"合議一致・高確信(conf {conf:.2f}≥{commit_confidence:.2f}) → commit",
                resolution=resolution,
            )
        return _abstain(
            q, resolution,
            f"合議一致だが低確信(conf {conf:.2f}<{commit_confidence:.2f}) → 棄権")

    # 3) tie-break で決着した答え(= 一次意見が割れた)。precision-first の既定は棄権。
    if resolution.status in _TIEBREAK_STATUSES:
        if commit_on_tiebreak and conf >= tiebreak_confidence:
            return GateDecision(
                question=q, answer=resolution.answer, commit=True, confidence=conf,
                reason=(f"合議不一致→tie-break決着・高確信"
                        f"(conf {conf:.2f}≥{tiebreak_confidence:.2f}, opt-in) → commit"),
                resolution=resolution,
            )
        detail = (f"低確信(conf {conf:.2f}<{tiebreak_confidence:.2f})"
                  if commit_on_tiebreak else "既定で不一致は棄権")
        return _abstain(q, resolution, f"合議不一致→tie-break決着だが{detail} → 棄権")

    # 4) Defensive: any unknown status is treated conservatively as 棄権 (never commit an unclassified
    #    outcome under Incorrect=−1 scoring).
    return _abstain(q, resolution, f"未分類の合議status({resolution.status}) → 保守的に棄権")


# --------------------------------------------------------------------------- live entry point
def gate_question(question: str, *,
                  investigator_model: str | None = None,
                  verifier_model: str | None = None,
                  judge_model: str | None = None,
                  commit_confidence: float | None = None,
                  commit_on_tiebreak: bool | None = None,
                  tiebreak_confidence: float | None = None,
                  **resolve_kwargs: Any) -> GateDecision:
    """Convenience live entry point: run the full 合議 then apply the confidence gate.

    Wires :func:`~src.rag.agent.tiebreak.resolve_question` (investigator→verifier→tiebreak, all live
    Gemini) and gates the resulting :class:`Resolution`.
    """
    resolution = resolve_question(
        question, investigator_model=investigator_model,
        verifier_model=verifier_model, judge_model=judge_model, **resolve_kwargs,
    )
    return apply_gate(
        resolution, commit_confidence=commit_confidence,
        commit_on_tiebreak=commit_on_tiebreak, tiebreak_confidence=tiebreak_confidence,
    )
