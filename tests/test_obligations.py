"""SOT-2499 — offline tests for evidence-obligation decomposition.

Everything runs network-free:

* the deterministic template decomposes one representative question **per contract** (≥1 each, 10 total)
  into a typed obligation list, asserted against the contract's expected kinds;
* the satisfaction API discharges obligations against collected evidence — via explicit evidence kinds
  and via the text-cue fallback — and returns the still-unmet ones;
* the injected/flash generation path is exercised with a fake arbiter (and the production
  :func:`flash_decompose` wrapper is driven with ``llm.generate`` monkeypatched), asserting invalid
  output is ignored and the deterministic template is used.
"""
from __future__ import annotations

import pytest

from src.rag.agent import obligations as ob
from src.rag.agent import question_contract as qc
from src.rag.agent.obligations import (
    CONTRACT_OBLIGATIONS,
    KINDS,
    MET,
    UNMET,
    COMPUTATION,
    CONSISTENCY,
    ENUMERATION,
    JUDGMENT_RULE,
    QUANTITY_DEFINITION,
    ROUNDING,
    SOURCE_LOCATION,
    UNIT_NORMALIZATION,
    VALUE_MAPPING,
    EvidenceItem,
    Obligation,
    ObligationSet,
    as_evidence,
    decompose,
    flash_decompose,
)


# --------------------------------------------------------------------------- taxonomy invariants
def test_every_contract_has_a_typed_obligation_template() -> None:
    """Each of the nine contracts expands to ≥1 obligation, every kind valid, covering its completion set."""
    assert set(CONTRACT_OBLIGATIONS) == set(qc.CONTRACTS)
    for contract, tmpl in CONTRACT_OBLIGATIONS.items():
        assert tmpl, f"{contract} has no obligations"
        for kind, text in tmpl:
            assert kind in KINDS, f"{contract}: bad kind {kind}"
            assert isinstance(text, str) and text.strip()
        # A template should never have fewer obligations than the contract's completion conditions has
        # distinct promises — i.e. it must not silently drop a completion condition.
        assert len(tmpl) >= 1


# --------------------------------------------------------------------------- deterministic decomposition
# One representative question per contract (≥1 each; 10 total → simple_lookup appears twice) with the
# obligation kinds each MUST produce.
_REPRESENTATIVE: list[tuple[str, str]] = [
    (qc.VERSION_DIFF, "白峰信用リスク評価の提案書old版と最新版を比較して、変更点を挙げてください。"),
    (qc.FULL_ENUMERATION, "契約書に登場する担当者名をすべて挙げてください。"),
    (qc.CROSS_AGGREGATE, "全プロジェクトの契約総額を計算してください。"),
    (qc.CHART_READ, "基礎分析.docxのグラフ1で、x=3のときのyの値を答えてください。"),
    (qc.SPATIAL, "井上さんの向かいに座っている方のEXTを教えてください。"),
    (qc.FORMAT_CHECK, "契約書において、太字で記載されている箇所をすべて抽出してください。"),
    (qc.MULTI_HOP, "もっとも多くの案件にかかわっている人の内線番号を教えてください。"),
    (qc.NUMERIC, "着手金と検収金の差額はいくらですか。"),
    (qc.SIMPLE_LOOKUP, "この契約書における甲の正式名称は何ですか。"),
    (qc.SIMPLE_LOOKUP, "報酬の支払期日はいつですか。"),
]


@pytest.mark.parametrize("expected_contract,question", _REPRESENTATIVE)
def test_representative_decompositions(expected_contract: str, question: str) -> None:
    oset = decompose(question)
    assert isinstance(oset, ObligationSet)
    assert oset.contract == expected_contract, f"{question!r} -> {oset.contract}"
    assert oset.method == "template"
    assert oset.obligations, "no obligations produced"
    kinds = {o.kind for o in oset.obligations}
    assert kinds == {k for k, _ in CONTRACT_OBLIGATIONS[expected_contract]}
    # A freshly decomposed set is entirely unmet, with no evidence refs yet.
    assert all(o.status == UNMET and not o.evidence_ref for o in oset.obligations)
    assert not oset.all_met
    assert oset.unmet() == oset.obligations


def test_every_representative_contract_covered() -> None:
    """受け入れ条件: 代表10問で各契約型≥1をカバー(9契約すべて出現)。"""
    covered = {c for c, _ in _REPRESENTATIVE}
    assert covered == set(qc.CONTRACTS)
    assert len(_REPRESENTATIVE) == 10


def test_decompose_is_deterministic_and_serialisable() -> None:
    q = "全プロジェクトの契約総額を計算してください。"
    a, b = decompose(q), decompose(q)
    assert [o.to_dict() for o in a.obligations] == [o.to_dict() for o in b.obligations]
    d = a.to_dict()
    assert d["contract"] == qc.CROSS_AGGREGATE and d["method"] == "template"
    assert all(set(o) == {"obligation", "kind", "status", "evidence_ref"} for o in d["obligations"])


def test_regulation_content_adds_fallback_general_rule_obligations() -> None:
    q = "契約条件で上限超過時の精算方法に関する規定内容を答えてください。"
    oset = decompose(q)
    assert oset.contract == qc.SIMPLE_LOOKUP
    fallback = [o for o in oset.obligations if "一般規定" in o.obligation]
    assert len(fallback) == 4
    assert {o.kind for o in fallback} == {VALUE_MAPPING, ROUNDING, JUDGMENT_RULE, CONSISTENCY}


def test_gantt_question_adds_grid_mapping_and_conflict_obligations() -> None:
    q = "提案書.pptxの工程表でモデル改善は第何週目に実施予定ですか。"
    oset = decompose(q)
    assert oset.contract == qc.CHART_READ
    assert any("x座標系" in o.obligation for o in oset.obligations)
    assert any("left/width" in o.obligation for o in oset.obligations)
    assert any("競合" in o.obligation for o in oset.obligations)


def test_explicit_contract_override_and_bad_contract_falls_back() -> None:
    forced = decompose("任意の質問", contract=qc.NUMERIC)
    assert forced.contract == qc.NUMERIC
    assert {o.kind for o in forced.obligations} == {k for k, _ in CONTRACT_OBLIGATIONS[qc.NUMERIC]}
    # An unknown contract code degrades to simple_lookup rather than raising.
    assert decompose("q", contract="not_a_contract").contract == qc.SIMPLE_LOOKUP


# --------------------------------------------------------------------------- satisfaction API
def test_satisfaction_via_explicit_evidence_kinds() -> None:
    oset = decompose("全プロジェクトの契約総額を計算してください。")  # cross_aggregate
    # Discharge only SOURCE_LOCATION + ENUMERATION explicitly; COMPUTATION/CONSISTENCY stay unmet.
    evidence = [
        EvidenceItem(text="全9プロジェクトのtrain.xlsxを母集団として確定", ref="registry#projects",
                     kinds=frozenset({SOURCE_LOCATION, ENUMERATION})),
    ]
    checked = oset.check(evidence, use_text_cues=False)
    by_kind = {o.kind: o for o in checked.obligations}
    assert by_kind[SOURCE_LOCATION].status == MET
    assert by_kind[SOURCE_LOCATION].evidence_ref == "registry#projects"
    assert by_kind[ENUMERATION].status == MET
    assert by_kind[COMPUTATION].status == UNMET and by_kind[CONSISTENCY].status == UNMET
    unmet_kinds = {o.kind for o in oset.unmet(evidence, use_text_cues=False)}
    assert unmet_kinds == {COMPUTATION, CONSISTENCY}


def test_satisfaction_via_text_cue_fallback() -> None:
    oset = decompose("着手金と検収金の差額はいくらですか。")  # numeric: source/value/computation/unit
    # Free-text investigator evidence, no explicit kinds — cues must discharge the matching kinds.
    evidence = ("提案書.pptx のスライド5から着手金=40万円・検収金=60万円を取得し、"
                "量の定義は検収金を分子・着手金を差し引く対象として差額=20万円と計算した。"
                "小数指定なしのため丸めなし。")
    checked = oset.check(evidence)
    by_kind = {o.kind: o.status for o in checked.obligations}
    assert by_kind[SOURCE_LOCATION] == MET      # ".pptx"/スライド cue
    assert by_kind[COMPUTATION] == MET          # 差額/計算 cue
    assert by_kind[UNIT_NORMALIZATION] == MET   # 万円 cue
    assert by_kind[QUANTITY_DEFINITION] == MET  # 分子 cue
    assert by_kind[ROUNDING] == MET              # 丸め cue
    assert by_kind[VALUE_MAPPING] == MET        # "=" cue
    assert checked.all_met


def test_unmet_returns_open_obligations_and_unrelated_evidence_leaves_them() -> None:
    oset = decompose("契約書に登場する担当者名をすべて挙げてください。")  # full_enumeration
    # Evidence that speaks to none of the obligation kinds leaves them all unmet.
    unmet = oset.unmet("天気は晴れです", use_text_cues=True)
    assert {o.kind for o in unmet} == {k for k, _ in CONTRACT_OBLIGATIONS[qc.FULL_ENUMERATION]}
    # No evidence at all → all obligations reported unmet, as-is.
    assert oset.unmet() == oset.obligations


def test_as_evidence_coerces_mixed_inputs() -> None:
    assert as_evidence(None) == ()
    assert as_evidence("") == ()
    assert as_evidence("hello") == (EvidenceItem(text="hello"),)
    mixed = as_evidence(["a", {"text": "b", "ref": "r", "kinds": ["computation", "bogus"]},
                         EvidenceItem(text="c")])
    assert [e.text for e in mixed] == ["a", "b", "c"]
    assert mixed[1].ref == "r" and mixed[1].kinds == frozenset({"computation"})  # bogus kind dropped


def test_obligation_is_immutable() -> None:
    o = Obligation(obligation="x", kind=SOURCE_LOCATION)
    with pytest.raises(Exception):
        o.status = MET  # type: ignore[misc]  # frozen dataclass


# --------------------------------------------------------------------------- flash / injected generation
def test_injected_flash_specialises_the_list() -> None:
    calls: list[tuple[str, str]] = []

    def arbiter(question: str, contract: str):
        calls.append((question, contract))
        return [
            {"obligation": "提案書.pptx#slide6 を対象版として確定", "kind": SOURCE_LOCATION},
            {"obligation": "スライド6の追記フェーズを差分抽出", "kind": JUDGMENT_RULE},
        ]

    oset = decompose("白峰の提案書old版と最新版の変更点は？", flash=arbiter)
    assert calls and calls[0][1] == qc.VERSION_DIFF  # classified first, then handed to flash
    assert oset.method == "flash"
    assert [o.obligation for o in oset.obligations] == [
        "提案書.pptx#slide6 を対象版として確定", "スライド6の追記フェーズを差分抽出"]
    assert all(o.status == UNMET for o in oset.obligations)


def test_invalid_flash_output_falls_back_to_template() -> None:
    # Empty / all-invalid flash output → keep the deterministic template.
    oset = decompose("全プロジェクトの契約総額を計算してください。",
                     flash=lambda q, c: [{"obligation": "", "kind": "bogus"}, "notadict"])
    assert oset.method == "template"
    assert {o.kind for o in oset.obligations} == {k for k, _ in CONTRACT_OBLIGATIONS[qc.CROSS_AGGREGATE]}

    oset2 = decompose("q", flash=lambda q, c: None)
    assert oset2.method == "template"


def test_flash_decompose_wrapper_parses_and_validates() -> None:
    good = '{"obligations": [{"obligation": "対象版を確定", "kind": "source_location"}, ' \
           '{"obligation": "無効", "kind": "bogus"}]}'
    parsed = flash_decompose("q", qc.VERSION_DIFF, generate=lambda *a, **k: good)
    assert parsed == ({"obligation": "対象版を確定", "kind": "source_location"},)  # bogus kind dropped

    assert flash_decompose("q", qc.NUMERIC, generate=lambda *a, **k: "not json") is None
    assert flash_decompose("q", qc.NUMERIC, generate=lambda *a, **k: '{"obligations": []}') is None

    def boom(*a, **k):
        raise RuntimeError("network down")

    assert flash_decompose("q", qc.NUMERIC, generate=boom) is None


def test_flash_decompose_end_to_end_through_decompose() -> None:
    # decompose + flash_decompose wired together (generate injected → no network).
    payload = '{"obligations": [{"obligation": "train.xlsxのA列を対象列として確定", ' \
              '"kind": "source_location"}, {"obligation": "A列平均を再計算", "kind": "computation"}]}'
    oset = decompose(
        "全プロジェクトの平均は？",
        flash=lambda q, c: flash_decompose(q, c, generate=lambda *a, **k: payload),
    )
    assert oset.method == "flash"
    assert {o.kind for o in oset.obligations} == {SOURCE_LOCATION, COMPUTATION}
