"""SOT-2637 — backend-agnostic commit gate (RAG_COMMIT_GATE, default OFF, UNWIRED).

cycle4 の設計目標は「LLM を交換しても正答率が変わらない」モデル不変アーキテクチャ。Sonnet dev gold100
(SOT-2628: net18, wrong28) の主因はモデル能力ではなく **commit 時ガードの不在**: 精度装置 (exec_verifier /
enum universe guard / Stage3 formatting / EU・棄権ポリシー) はすべて Gemini investigator ループの内部に
実装されているため、claude-mcp バックエンド (SOT-2627) は submit_answer の値を無検証で commit してしまう。

このモジュールは commit 時の精度判定を **単一の入口** に集約し、どのバックエンド (Gemini function-calling /
claude-mcp / 将来の任意モデル) からも同じ関数を通せる基盤を作る。核心は次の 1 点だけ:

    「submit された回答を、モデル非依存の同一基準で commit / reject / abstain に振り分ける」

**本 issue はモジュール新設＋既存部品の薄い集約のみ**。既存部品 (exec_verifier / enum_scan / formatting /
evidence_packet / abstain 既定) は **移動も改変もせず薄い呼び出しで再利用** する。serve path は不変更で、この
ゲートはフラグ ``RAG_COMMIT_GATE`` (既定 OFF) の裏にあり、まだどの経路からも呼ばれない (配線は後続 2 issue)。

判定 (入力 ``(question, contract, submitted_answer, evidence, session_tool_history)``):

1. **numeric / cross_aggregate**: ``session_tool_history`` に compute / corpus_aggregate 系の *成功* 記録で
   値の一致するものが無い数値回答 → ``REJECT`` (配線側が in-band 再試行 or 棄権降格に使う理由文字列つき)。
2. **enum**: universe 未証明の「該当なし」→ ``REJECT`` (SOT-2600 :func:`enum_scan.forbids_none_answer`)。
3. **証拠完全性**: Evidence Packet の required slot が missing のままの commit → 棄権降格 (packet 無効時は
   スキップ)。
4. **書式**: :func:`formatting.format_contract` の契約＋naturalizer を **値不変** で適用 (COMMIT 経路のみ)。
5. **棄権ポリシー**: Incorrect=−1 < Missing=0 の非対称を機械判定 — ``REJECT`` が n 回続いたら ``ABSTAIN`` へ
   降格 (``RAG_COMMIT_GATE_ABSTAIN_AFTER`` 既定 2)。
6. すべての判定を SOT-2629 テレメトリ形式 (``interventions['commit_gate']``) で記録する。

すべて純関数 (副作用なし)。live なもの (enum 全数走査 / EU の signals 収集) は呼び出し側が計算して注入する
(``enum_result`` など) — ゲート本体は decision を返すだけで、``inv.interventions['commit_gate']`` への書き込み
やモデル呼び出しは行わない。
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

from config import settings
from src.rag.agent import exec_verifier as _exec
from src.rag.agent import formatting as _formatting
from src.rag.agent import question_contract as _qc

# --------------------------------------------------------------------------- flag / config
_ON = {"1", "true", "yes", "on"}

ABSTAIN = settings.ABSTAIN


def enabled() -> bool:
    """Whether the commit gate runs — ``RAG_COMMIT_GATE`` (default OFF ⇒ byte-identical / unwired)."""
    return os.getenv("RAG_COMMIT_GATE", "0").strip().lower() in _ON


def enforce() -> bool:
    """Whether the gate's verdict is ENFORCED on the served answer — ``RAG_COMMIT_GATE_ENFORCE`` (default
    OFF). Single source of truth for the enforcement flag shared by every wiring point.

    SOT-2639's Gemini wiring is equivalence-preserving: with ``RAG_COMMIT_GATE=1`` alone the gate's
    decision + telemetry are recorded but the loop's own inline guards stay authoritative (the committed
    answer is served VERBATIM). SOT-2640 turns this ON for the guard-LESS ``claude-mcp`` backend, whose
    ``submit_answer`` would otherwise commit its raw value unchecked: enforcement makes REJECT feed an
    in-band retry and, after :func:`abstain_after` consecutive rejects, degrade to ABSTAIN.
    """
    return os.getenv("RAG_COMMIT_GATE_ENFORCE", "0").strip().lower() in _ON


def abstain_after() -> int:
    """Consecutive-REJECT count that trips the 棄権降格 (``RAG_COMMIT_GATE_ABSTAIN_AFTER``, default 2).

    Incorrect(−1) < Missing(0) — after this many rejects on one question, committing again is negative EV,
    so the gate degrades to ABSTAIN instead of letting the caller retry indefinitely.
    """
    raw = os.getenv("RAG_COMMIT_GATE_ABSTAIN_AFTER")
    if raw is None or not raw.strip():
        return 2
    try:
        return max(1, int(raw.strip()))
    except ValueError:
        return 2


# --------------------------------------------------------------------------- verdicts
COMMIT = "COMMIT"       # accept the (formatted) value
REJECT = "REJECT"       # precision guard failed — retry in-band or degrade (retryable)
ABSTAIN_V = "ABSTAIN"   # commit わかりません (hard blocker or 棄権ポリシー降格)

# Numeric-source tools whose *successful* record can ground a numeric commit (mirrors
# ``calc_ledger._NUMERIC_SOURCE_TOOLS`` — kept local so the gate never imports a private symbol).
_NUMERIC_TOOLS = frozenset({"compute", "corpus_aggregate", "read_chart_values", "pptx_pivot",
                            # SOT-2647 — precomputed fact-layer tools return provenance-carrying store
                            # values (per-cell 出典); a numeric answer grounded by one of these is a
                            # **verified operand**, so the gate accepts it exactly like a live compute.
                            "metric_lookup", "case_filter", "id_lookup", "diff_lookup"})

# Base contracts (question_contract) that require an executed numeric derivation to commit a number.
_NUMERIC_CONTRACTS = frozenset({_qc.NUMERIC, _qc.CROSS_AGGREGATE})

# A negative ("該当なし") enumeration conclusion — the phrasings enum_scan's guard must vet (SOT-2600).
_NONE_ANSWER_RE = re.compile(
    r"該当(?:する[^。]{0,12})?(?:なし|無し|ありません|存在し(?:ない|ません))|"
    r"存在し(?:ない|ません)|見つか(?:らない|りません|らなかった)|該当者(?:は)?(?:い|お)ません|"
    r"^\s*(?:なし|無し)\s*$|0\s*件")


# --------------------------------------------------------------------------- decision object
@dataclass(frozen=True)
class CommitDecision:
    """The gate's verdict for one committed answer, backend-agnostic.

    ``final_answer`` is the value the caller should serve: the (possibly reformatted) submitted answer on
    :data:`COMMIT`, or :data:`ABSTAIN` on the abstain branch. On :data:`REJECT` it is left as the original
    submission (the caller decides whether to retry in-band or degrade); ``reasons`` carries the machine
    reason strings so the wiring can act on them.
    """

    verdict: str
    final_answer: str
    reasons: tuple[str, ...] = ()
    telemetry: dict[str, Any] = field(default_factory=dict)

    @property
    def committed(self) -> bool:
        return self.verdict == COMMIT

    @property
    def abstained(self) -> bool:
        return self.verdict == ABSTAIN_V

    def to_dict(self) -> dict[str, Any]:
        return {
            "verdict": self.verdict,
            "final_answer": self.final_answer,
            "reasons": list(self.reasons),
            "telemetry": dict(self.telemetry),
        }


# --------------------------------------------------------------------------- contract / answer helpers
def _route(contract: Any) -> str:
    """Normalize a contract (``QuestionContract`` | mapping | str | None) to its base-contract string."""
    if contract is None:
        return ""
    if isinstance(contract, _qc.QuestionContract):
        return contract.contract
    if isinstance(contract, Mapping):
        return str(contract.get("contract") or contract.get("route") or "")
    return str(contract)


def _is_abstain(text: Any) -> bool:
    """NFKC-insensitive 棄権 check (mirrors :func:`investigator.is_abstain` without importing the loop)."""
    a = _exec._nfkc_lower(text)
    return not a or a == _exec._nfkc_lower(ABSTAIN) or "わかりません" in a or "不明" in a


def _is_none_answer(text: Any) -> bool:
    return bool(_NONE_ANSWER_RE.search(str(text or "")))


# --------------------------------------------------------------------------- session tool-history probing
def _th_name(entry: Mapping[str, Any]) -> str:
    return str(entry.get("name") or entry.get("tool") or "")


def _th_response(entry: Mapping[str, Any]) -> Any:
    for k in ("response", "result", "output"):
        if k in entry:
            return entry[k]
    return None


def _th_ok(entry: Mapping[str, Any]) -> bool:
    """Whether a tool-history entry recorded a *success* (tolerant of several shapes)."""
    for k in ("ok", "success"):
        if k in entry:
            return bool(entry[k])
    resp = _th_response(entry)
    if isinstance(resp, Mapping):
        if resp.get("error") or resp.get("status") == "error":
            return False
    return resp is not None or "value" in entry


def _th_value(entry: Mapping[str, Any]) -> Any:
    if "value" in entry:
        return entry["value"]
    resp = _th_response(entry)
    if isinstance(resp, Mapping):
        for k in ("value", "result", "answer", "output"):
            if k in resp:
                return resp[k]
    return resp


def _num_close(a: float, b: float) -> bool:
    return abs(a - b) <= max(1e-9, 1e-6 * max(abs(a), abs(b)))


def _numeric_grounded(submitted: str,
                      session_tool_history: Sequence[Mapping[str, Any]] | None) -> tuple[bool, Any]:
    """Is the numeric ``submitted`` value grounded by a successful compute/aggregate record?

    Returns ``(grounded, matched_value)``. When ``submitted`` is not a numeric literal (e.g. a
    cross_aggregate argmax returns a *column label*), the numeric-grounding requirement does not apply and
    this returns ``(True, None)`` — a non-number can never be an ungrounded arithmetic commit.
    """
    target = _exec._leading_number(submitted)
    if target is None:
        return True, None  # not a numeric literal ⇒ requirement N/A
    for entry in session_tool_history or ():
        if not isinstance(entry, Mapping):
            continue
        if _th_name(entry) not in _NUMERIC_TOOLS or not _th_ok(entry):
            continue
        val = _th_value(entry)
        num = _exec._leading_number(str(val)) if val is not None else None
        if num is not None and _num_close(num, target):
            return True, val
    return False, None


# --------------------------------------------------------------------------- formatting (value-invariant)
def _apply_formatting(question: str, route: str, submitted: str,
                      evidence: Mapping[str, Any] | None,
                      naturalizer: Any) -> tuple[str, dict[str, Any]]:
    """Apply the Stage3 書式契約 to the committed value (value-preserving). Best-effort: never破壊commit."""
    tel: dict[str, Any] = {"applied": False}
    try:
        contract = {"value": submitted, "evidence": dict(evidence or {}), "method": {}}
        out = _formatting.format_contract(contract, question, contract_type=(route or None),
                                          naturalizer=naturalizer, force=True)
        if out is not None and not _formatting._is_blank(out.get("value")):
            formatted = str(out["value"])
            tel = {"applied": formatted != submitted,
                   "rules": list((out.get("method") or {}).get("formatting", {}).get("rules", []))}
            return formatted, tel
    except Exception:  # noqa: BLE001 — formatting is polish; a failure must never drop the commit
        tel = {"applied": False, "error": True}
    return submitted, tel


# --------------------------------------------------------------------------- the gate
def evaluate(question: str,
             contract: Any,
             submitted_answer: str,
             evidence: Mapping[str, Any] | None = None,
             session_tool_history: Sequence[Mapping[str, Any]] | None = None,
             *,
             packet: Any = None,
             enum_result: Any = None,
             naturalizer: Any = None,
             prior_rejects: int = 0) -> CommitDecision:
    """Backend-agnostic commit judgment for one submitted answer.

    Parameters
    ----------
    question, contract, submitted_answer, evidence, session_tool_history
        The commit context. ``contract`` may be a :class:`question_contract.QuestionContract`, a mapping
        with ``contract``/``route``, or a bare base-contract string. ``session_tool_history`` is a sequence
        of tolerant tool-call records (``{name/tool, ok/success, value | response/result/output}``).
    packet
        Optional :class:`evidence_packet.EvidencePacket` (or any object exposing ``missing``). When present
        and it still has missing required slots, the commit is degraded to ABSTAIN (証拠完全性). ``None`` ⇒
        the check is skipped (packet 無効時はスキップ).
    enum_result
        Optional :class:`enum_scan.EnumScanResult` for a ``full_enumeration`` question — the precomputed
        universe certificate. When a "該当なし" answer is submitted and the certificate forbids a negative
        (:func:`enum_scan.forbids_none_answer`), the commit is REJECTED (SOT-2600).
    naturalizer
        Optional LLM naturalizer forwarded to :func:`formatting.format_contract` (value-invariant).
    prior_rejects
        How many times this question was already REJECTED by the gate in this session. Together with the
        current reject it drives the asymmetric 棄権ポリシー (Incorrect −1 < Missing 0).

    Returns a :class:`CommitDecision`. The gate never mutates its inputs and never writes telemetry — the
    caller records ``decision.telemetry`` under ``inv.interventions['commit_gate']`` (SOT-2629 形式).
    """
    route = _route(contract)
    submitted = "" if submitted_answer is None else str(submitted_answer)
    already_abstain = _is_abstain(submitted)

    checks: dict[str, Any] = {}
    reasons: list[str] = []

    # ---- 1. numeric / cross_aggregate — an ungrounded arithmetic commit is a −1 waiting to happen.
    numeric_ok = True
    if not already_abstain and route in _NUMERIC_CONTRACTS:
        grounded, matched = _numeric_grounded(submitted, session_tool_history)
        numeric_ok = grounded
        checks["numeric"] = {"route": route, "answer_is_number": _exec._leading_number(submitted) is not None,
                             "grounded": grounded, "matched_value": matched}
        if not grounded:
            reasons.append("numeric_ungrounded:compute/corpus_aggregate 成功記録なし")

    # ---- 2. enum universe guard — 未証明の「該当なし」は不存在の証明にならない (SOT-2600).
    enum_ok = True
    if not already_abstain and route == _qc.FULL_ENUMERATION and _is_none_answer(submitted):
        if enum_result is not None:
            from src.rag.agent import enum_scan as _enum
            forbidden = _enum.forbids_none_answer(enum_result)
            enum_ok = not forbidden
            rec = {"none_answer": True, "forbids_none": forbidden}
            cert = getattr(enum_result, "to_dict", None)
            if callable(cert):
                try:
                    rec["certificate"] = enum_result.to_dict()
                except Exception:  # noqa: BLE001
                    pass
            checks["enum"] = rec
            if forbidden:
                reasons.append("enum_none_uncertified:universe 未証明の該当なし")
        else:
            checks["enum"] = {"none_answer": True, "forbids_none": None, "reason": "no_enum_result"}

    # ---- 3. evidence completeness — required slot が missing のままの commit は棄権降格.
    evidence_missing: tuple[str, ...] = ()
    if packet is not None:
        miss = getattr(packet, "missing", None)
        if miss:
            evidence_missing = tuple(str(m) for m in miss)
            checks["evidence"] = {"missing": list(evidence_missing)}
            reasons.append("evidence_incomplete:" + ",".join(evidence_missing))

    # ---- decide verdict (precedence: evidence 完全性 → precision reject → commit) --------------------
    if already_abstain:
        verdict = ABSTAIN_V
        final = ABSTAIN
    elif evidence_missing:
        # A hard epistemic blocker: missing required evidence ⇒ 棄権降格 (not a retryable reject).
        verdict = ABSTAIN_V
        final = ABSTAIN
    elif not (numeric_ok and enum_ok):
        # A precision guard tripped. 棄権ポリシー: after ``abstain_after`` consecutive rejects, degrade to
        # ABSTAIN rather than let the caller commit a likely-wrong (−1) value again.
        streak = prior_rejects + 1
        threshold = abstain_after()
        checks["abstain_policy"] = {"prior_rejects": prior_rejects, "streak": streak,
                                    "threshold": threshold}
        if streak >= threshold:
            verdict = ABSTAIN_V
            final = ABSTAIN
            reasons.append(f"reject_streak_abstain:{streak}>={threshold}")
        else:
            verdict = REJECT
            final = submitted
    else:
        # ---- 4. 書式契約 (COMMIT 経路のみ, 値不変) --------------------------------------------------
        formatted, fmt_tel = _apply_formatting(question, route, submitted, evidence, naturalizer)
        checks["format"] = fmt_tel
        verdict = COMMIT
        final = formatted

    # ---- 6. telemetry (SOT-2629 interventions['commit_gate'] 形式) --------------------------------
    telemetry: dict[str, Any] = {
        "enabled": True,
        "verdict": verdict,
        "route": route,
        "already_abstain": already_abstain,
        "prior_rejects": prior_rejects,
        "reasons": list(reasons),
        "checks": checks,
    }
    return CommitDecision(verdict=verdict, final_answer=final,
                          reasons=tuple(reasons), telemetry=telemetry)
