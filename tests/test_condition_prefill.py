"""SOT-2621 — tests for pre-loop BranchCondition IR construction + Evidence Packet injection.

Network-free / corpus-free: condition detection is pure regex over the question text, so these exercise
the deterministic detection (相対増減 / 分岐率 / 除外), the conservative *non*-firing on ordinary NUMERIC
questions (wrong 非増加 の担保), the PoT ``ConditionIR.from_spec`` receiver, and the Evidence Packet
OFF-byte-identical / ON-injection contract without any live document read.
"""
from __future__ import annotations

from src.rag.agent import condition_prefill as cp
from src.rag.agent import evidence_packet as ep
from src.rag.agent import pot_lane as pl
from src.rag.agent import query_router as qr

# The focused what-if / 条件分岐 questions from ``docs/ai/budget32_trace_classification.md`` (verbatim).
IDX76 = ("AOMINEの契約条件において、契約単価が現状よりも2,000円高く、実績工数が11.2時間少なかった場合、"
         "税込請求金額は、実際の税込請求金額と比べていくら変動しますか。")
IDX47 = ("青嶺不動産アセットマネジメントのtrain.xlsxにおいて、黄色ハイライトセルは予測と実際の誤差を計算"
         "していますが、その予測値の対象となっている不動産の建設年を算出してください。")
IDX6 = "蒼泉会 ひがし丘総合病院案件において、提案時の税込み見込み金額と最終請求金額の差額はいくらですか。"
IDX27 = "恒一会 かえで総合病院の提案書において、スコープ対象外としている項目はいくつありますか。"
IDX57 = ("青葉のTXにて算出された回帰係数を用いて全データの予測値を計算し、正解データに対する F1 スコアが"
         "最大となるように閾値を設定したときの F1 スコアを答えてください。小数第5位まで求めてください。")


# --------------------------------------------------------------------------- flag / defaults
def test_disabled_by_default():
    assert cp.enabled() is False


def test_enabled_reads_flag(monkeypatch):
    monkeypatch.setenv("RAG_CONDITION_PREIR", "1")
    assert cp.enabled() is True
    monkeypatch.setenv("RAG_CONDITION_PREIR", "off")
    assert cp.enabled() is False


# --------------------------------------------------------------------------- 相対増減 (idx76 flagship)
def test_idx76_fires_two_relative_deltas():
    ir = cp.build_condition_ir(IDX76)
    assert ir is not None
    assert ir["condition_type"] == "assumption_change"
    adj = ir["adjustments"]
    assert len(adj) == 2
    # both deltas: 契約単価 +2000円 増加, 実績工数 -11.2時間 減少 — order preserved by text position.
    assert adj[0]["operand_hint"] == "契約単価"
    assert adj[0]["delta"] == "+2000" and adj[0]["unit"] == "円" and adj[0]["direction"] == "increase"
    assert adj[0]["order"] == 0
    assert adj[1]["operand_hint"] == "実績工数"
    assert adj[1]["delta"] == "-11.2" and adj[1]["unit"] == "時間" and adj[1]["direction"] == "decrease"
    assert adj[1]["order"] == 1
    # the skeleton leaves the LLM's blanks blank (no answer injected).
    assert ir["predicate_truth"] is None and ir["base_quantity"] == ""
    assert "少なかった場合" in ir["predicate"]


def test_relative_delta_without_yori_but_with_branch_marker():
    # a bare comparative clause is caught when a branch marker licenses the what-if reading.
    ir = cp.build_condition_ir("工数が3時間多かった場合、費用はいくらですか。")
    assert ir is not None
    assert ir["adjustments"][0]["operand_hint"] == "工数"
    assert ir["adjustments"][0]["delta"] == "+3" and ir["adjustments"][0]["direction"] == "increase"


def test_fullwidth_digits_and_separators_folded():
    ir = cp.build_condition_ir("単価が現状より１，５００円安かった場合の金額は。")
    assert ir is not None
    assert ir["adjustments"][0]["delta"] == "-1500"


# --------------------------------------------------------------------------- 分岐率 (rate branch)
def test_branch_rate_discount():
    ir = cp.build_condition_ir("一括前払いの場合は7.5%減額されます。税込金額はいくらですか。")
    assert ir is not None
    assert ir["condition_type"] == "branch_rate"
    a = ir["adjustments"][0]
    assert a["kind"] == "discount" and a["rate"] == "0.075" and a["direction"] == "decrease"


def test_branch_rate_surcharge():
    ir = cp.build_condition_ir("特急対応の場合は10%加算した金額はいくらですか。")
    assert ir is not None
    a = ir["adjustments"][0]
    assert a["kind"] == "surcharge" and a["rate"] == "0.1" and a["direction"] == "increase"


# --------------------------------------------------------------------------- 除外 (exclusion)
def test_exclusion_fires():
    ir = cp.build_condition_ir("未着手から完了への変更を除いて、変更点はいくつですか。")
    assert ir is not None
    assert ir["condition_type"] == "exclusion"
    assert ir["adjustments"][0]["kind"] == "exclusion"
    assert ir["adjustments"][0]["operand_hint"] == "未着手から完了への変更"


# --------------------------------------------------------------------------- conservative non-firing
def test_non_whatif_numeric_questions_do_not_fire():
    # idx47/6/27/57 are NUMERIC-route derived questions with no branch structure: must stay None so the
    # ordinary loop keeps handling them (no firing-condition relaxation, wrong 非増加).
    for q in (IDX47, IDX6, IDX27, IDX57):
        assert cp.detect(q) is None, q


def test_bare_case_marker_without_adjustment_does_not_fire():
    # "…した場合" with no quantifiable knob is deliberately not a fired condition.
    assert cp.detect("フェーズAとフェーズBを実施した場合の想定工数は合計で何時間ですか。") is None
    assert cp.detect("id=0を予測した場合の予測値はいくらになりますか。") is None


def test_stray_comparative_without_context_does_not_fire():
    # a comparative superlative ("もっとも多く") in an ordinary question must not be read as a delta.
    assert cp.detect("もっとも多くの案件にかかわっている人の内線番号を教えてください。") is None


def test_empty_question():
    assert cp.detect("") is None
    assert cp.build_condition_ir("") is None


# --------------------------------------------------------------------------- PoT receiver (from_spec)
def test_condition_ir_from_spec_receiver():
    ir = cp.build_condition_ir(IDX76)
    cond = pl.ConditionIR.from_spec(ir)
    assert cond.is_conditional is True
    assert len(cond.adjustments) == 2
    # extra hint keys (delta/unit/operand_hint) are ignored; only kind/rate/order become the IR.
    assert cond.adjustments[0].kind == "delta"


def test_from_spec_none_is_unconditional():
    assert pl.ConditionIR.from_spec(None).is_conditional is False


# --------------------------------------------------------------------------- directive rendering
def test_condition_directive_lists_adjustments():
    ir = cp.build_condition_ir(IDX76)
    block = cp.condition_directive(ir)
    assert "契約単価" in block and "+2000円" in block
    assert "実績工数" in block and "-11.2時間" in block
    assert "predicate_truth" in block and "base_quantity" in block


def test_condition_directive_empty_when_no_ir():
    assert cp.condition_directive(None) == ""
    assert cp.condition_directive({"adjustments": []}) == ""


# --------------------------------------------------------------------------- Evidence Packet wiring
def test_packet_off_has_no_condition_ir(monkeypatch):
    monkeypatch.delenv("RAG_CONDITION_PREIR", raising=False)
    monkeypatch.delenv("RAG_OPERAND_PREFILL", raising=False)
    dec = qr.classify_route(IDX76)
    assert dec.route == qr.NUMERIC
    packet = ep.build_packet(IDX76, decision=dec)
    assert "condition_ir" not in packet.evidence


def test_packet_off_directive_byte_identical(monkeypatch):
    # with the flag OFF the pre-inject directive must be byte-identical to the pre-SOT-2621 path: it never
    # mentions the condition skeleton.
    monkeypatch.delenv("RAG_CONDITION_PREIR", raising=False)
    monkeypatch.delenv("RAG_OPERAND_PREFILL", raising=False)
    _, directive = ep.build_directive(IDX76)
    assert "条件文の事前IR骨格" not in directive


def test_packet_on_injects_condition_ir(monkeypatch):
    monkeypatch.setenv("RAG_CONDITION_PREIR", "1")
    monkeypatch.delenv("RAG_OPERAND_PREFILL", raising=False)
    dec = qr.classify_route(IDX76)
    packet = ep.build_packet(IDX76, decision=dec)
    assert "condition_ir" in packet.evidence
    assert len(packet.evidence["condition_ir"]["adjustments"]) == 2
    # condition_ir is auxiliary — it is not a required slot, so the real slots stay in `missing`.
    assert "condition_ir" not in packet.missing
    directive = ep.packet_directive(packet)
    assert "条件文の事前IR骨格" in directive and "契約単価" in directive


def test_packet_on_non_whatif_injects_nothing(monkeypatch):
    monkeypatch.setenv("RAG_CONDITION_PREIR", "1")
    monkeypatch.delenv("RAG_OPERAND_PREFILL", raising=False)
    dec = qr.classify_route(IDX6)
    assert dec.route == qr.NUMERIC
    packet = ep.build_packet(IDX6, decision=dec)
    assert "condition_ir" not in packet.evidence  # no branch structure → no injection (no degradation)
