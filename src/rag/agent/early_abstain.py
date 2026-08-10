"""SOT-2599 — early-abstain recalibration: forbid an iters≤2 immediate UNANSWERABLE.

Parent SOT-2460 / 統合再較正3 (SOT-2568 r2 decision系較正の1つ). The r2 loss analysis showed the
expected-utility gate (:mod:`src.rag.agent.eu_gate`, ``RAG_EU_GATE``) combined with the Evidence Packet
**budget contract** (:mod:`src.rag.agent.evidence_packet`, ``RAG_EVIDENCE_PACKET``) enforce a *原則追加探索
なし* posture: a question whose target document resolves fast but whose value the model does not immediately
see is dropped to **UNANSWERABLE at iterations≤2** — before any *cheap deterministic* probe
(``canonical_route`` / ``file_grep`` / ``find_files``) was even tried. UNANSWERABLE ballooned 18→32 and the
bulk of the MATCH→ABSTAIN regressions were iters≤2 (idx9/10/23/26/31/80).

This module is the recalibration knob. It is a *pure* decision + directive helper shared by the re-search
director (:mod:`src.rag.agent.research_loop`) and the packet directive
(:func:`src.rag.agent.evidence_packet.packet_directive`):

* The re-search director, when it is about to finalize UNANSWERABLE with **no unmet obligation left**,
  asks :func:`may_finalize_unanswerable`. While the question is still *early* (``tool_call_count`` ≤
  :data:`EARLY_ABSTAIN_MAX_CALLS`) and **no cheap deterministic tool has been tried yet**, the answer is
  ``False`` — it must run one deterministic probe first (:func:`probe_directive`). The probe is one-shot
  and bounded (recorded as one re-search round), so it can never loop and never touches the commit
  threshold: the probe either surfaces evidence the model may answer from, or the *next* abstain finalizes
  UNANSWERABLE exactly as before.
* The packet directive appends :func:`packet_relaxation_clause` so the "自由探索の残予算=0" contract is
  relaxed from *追加探索なし* to *安価な決定論探索は許す* — free *chunk* search is still not grown, only the
  deterministic lanes.

Design invariants (mirroring the sibling opt-ins — eu_gate / evidence_packet / pot_lane / enum_scan):

* **Opt-in, byte-identical OFF.** :func:`enabled` (``RAG_EARLY_ABSTAIN_RECAL``) gates the whole thing; with
  it off the director finalizes UNANSWERABLE unchanged and the packet directive is byte-identical.
* **Pure & network-free.** Every function is a total map over its arguments (env only for :func:`enabled`).
* **EV-safe.** A deterministic probe never fabricates a value (it only *reads* the corpus), so forcing one
  before UNANSWERABLE can turn a MATCH→ABSTAIN back into a MATCH but can never manufacture a wrong answer —
  precisely the "wrong→abstain 可 / match→abstain 不可" posture the re-calibration targets.
"""
from __future__ import annotations

import os

_ON = {"1", "true", "yes", "on"}

# The cheap **deterministic** probes that must be tried before an early UNANSWERABLE is accepted. These
# read the corpus structurally (registry-resolved routing / literal grep / file resolution) and never
# invent a value, so requiring one is EV-safe. Free chunk retrieval is deliberately *not* here — the
# recalibration relaxes deterministic probing only, it does not grow free exploration.
CHEAP_DETERMINISTIC_TOOLS: frozenset[str] = frozenset({
    "canonical_route", "file_grep", "find_files",
})

# iters≤2: how early a would-be UNANSWERABLE still owes a deterministic probe. Past this the abstain is
# accepted as before (the model had room to probe and did not, or the corpus really is silent).
EARLY_ABSTAIN_MAX_CALLS = 2

# The synthetic re-search round kind recorded when the director forces a probe (keeps the abstain ledger's
# 探索履歴 intact and bounds the push to a single round — it is not an obligation kind, so KINDS is unchanged).
PROBE_KIND = "deterministic_probe"


def enabled() -> bool:
    """True when the early-abstain recalibration is active (default OFF — opt-in, byte-identical)."""
    return os.getenv("RAG_EARLY_ABSTAIN_RECAL", "0").strip().lower() in _ON


def is_cheap_deterministic_tool(name: str) -> bool:
    """Whether ``name`` is one of the cheap deterministic probes tracked by the recalibration."""
    return name in CHEAP_DETERMINISTIC_TOOLS


def may_finalize_unanswerable(*, tool_call_count: int, deterministic_probe_tried: bool,
                              probe_forced: bool, recal_enabled: bool | None = None) -> bool:
    """Whether a would-be UNANSWERABLE (no unmet obligation left) may be finalized now.

    Returns ``True`` (finalize as before) unless the recalibration is on AND the question is still early
    (``tool_call_count`` ≤ :data:`EARLY_ABSTAIN_MAX_CALLS`) AND no cheap deterministic probe has been tried
    AND a probe has not already been forced this question — in which case it returns ``False`` (force one
    deterministic probe first). ``probe_forced`` makes the push strictly one-shot, so this can never loop.
    """
    on = enabled() if recal_enabled is None else recal_enabled
    if not on:
        return True
    if probe_forced or deterministic_probe_tried:
        return True
    return tool_call_count > EARLY_ABSTAIN_MAX_CALLS


def probe_directive() -> str:
    """The one-shot directive that forces a cheap deterministic probe before UNANSWERABLE (EV-safe)."""
    tools = " / ".join(sorted(CHEAP_DETERMINISTIC_TOOLS))
    return (
        "UNANSWERABLE と確定する前に、まだ安価な決定論ツールを1つも試していません。自由な chunk 検索は"
        f"増やさず、次の決定論ツールのいずれか1つ（{tools}）を、質問文から対象を自己解決させて1回だけ実行"
        "してください（会社名/文書名の略称⇔正式名称展開・字面 regex 検索・レジストリ解決）。\n"
        "その結果に具体的な根拠が現れたら submit_answer で回答を確定してください。決定論プローブでも根拠が"
        "コーパスに現れない場合に限り、再度 submit_answer で棄権してください（値の創作は禁止・確信度は操作しない）。"
    )


def packet_relaxation_clause() -> str:
    """The clause appended to the packet directive relaxing *追加探索なし* → *安価な決定論探索は許す*."""
    tools = " / ".join(sorted(CHEAP_DETERMINISTIC_TOOLS))
    return (
        f"ただし自由探索の残予算が0でも、決定論ツール（{tools}）は自由探索予算とは別枠です。UNANSWERABLE と"
        "結論する前に、これらの安価な決定論ツールを最低1回は試してください（自由な chunk 検索は増やさない）。"
    )
