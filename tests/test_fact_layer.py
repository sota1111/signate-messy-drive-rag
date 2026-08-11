"""SOT-2647 (事前計算事実層 5/5) — fact-layer wiring offline tests (no network / LLM / real corpus).

Verifies the two wired forms against *synthetic* stores (each store's ``load`` monkeypatched), so the
lane grounding, tool contracts, precision-first deferral, commit-gate verified-operand grounding, the
MCP/investigator single-source exposure, and RAG_FACT_LAYER default-OFF byte-identical behavior are all
checked without depending on the gitignored ``artifacts/*.jsonl``.
"""
from __future__ import annotations

import pytest

from src.rag.agent import fact_layer as fl
from src.rag.tools import contract as _contract


# --------------------------------------------------------------------------- synthetic stores
def _case_rows():
    def cell(v, src="doc/契約書.docx#A1"):
        return {"value": v, "source": {"doc_id": src}}
    return [
        {"case_id": "青葉与信マネジメント株式会社", "abbrev": "AYM",
         "attributes": {"abbrev": cell("AYM"), "apr_code": cell("APR-M1"), "status": cell("完了"),
                        "contract_amount_incl_tax": cell(4620000), "case_name": cell("青葉与信")}},
        {"case_id": "京橋信用ソリューションズ株式会社", "abbrev": "KSS",
         "attributes": {"abbrev": cell("KSS"), "apr_code": cell("APR-M2"), "status": cell("完了"),
                        "contract_amount_incl_tax": cell(5775000), "case_name": cell("京橋信用")}},
        {"case_id": "白峰信用リスク評価株式会社", "abbrev": "SRK",
         "attributes": {"abbrev": cell("SRK"), "apr_code": cell("APR-M1"), "status": cell("完了"),
                        "contract_amount_incl_tax": cell(3000000), "case_name": cell("白峰")}},
    ]


def _derived_rows():
    return [
        {"case_id": "青葉与信マネジメント株式会社", "train_file": "…/train.xlsx",
         "sources": {"train_file": "…/train.xlsx", "coef_source": {"basis": "ols"}},
         "model": {"threshold_sweep": {"best_f1": 0.42395962, "best_threshold": 0.5},
                   "prediction_id0": {"id": 0, "prediction": 0.15001822},
                   "coefficients": {"x": 1.0}},
         "ratios": {"col": {"p90_minus_p50": 12.0}}},
    ]


def _id_rows():
    return [
        {"id": "M01", "id_norm": "M01", "id_kind": "milestone", "doc_id": "proj/sched.xlsx",
         "sheet": "Sheet1", "slide": 0, "page": 0, "row": 3, "line": "M01 | キックオフ完了 | 2025-10-01",
         "columns": ["M01", "キックオフ完了"], "column_header": "ID", "label": ""},
    ]


def _diff_rows():
    return [
        {"old_rel": "proj/青嶺/提案書_v1.pptx", "new_rel": "proj/青嶺/提案書_v3.pptx",
         "project": "株式会社青嶺不動産", "basis": "rev-suffix",
         "changes": [{"rank": 1, "kind": "modify", "intent": "SUBSTANTIVE", "before": "A", "after": "B",
                      "structural_location": "slide 2", "attributes": ["substantive", "numeric_change"]},
                     {"rank": 2, "kind": "modify", "intent": "SURFACE", "before": "未着手", "after": "完了",
                      "structural_location": "row 5", "attributes": ["status_transition"]}]},
    ]


@pytest.fixture
def stores(monkeypatch):
    """Point every store's ``load`` at a synthetic in-memory fixture + enable the layer."""
    from src.rag.index import case_master, derived_metrics, diff_store, id_master
    monkeypatch.setattr(case_master, "load", lambda path=None: _case_rows())
    monkeypatch.setattr(derived_metrics, "load", lambda path=None: _derived_rows())
    monkeypatch.setattr(id_master, "load", lambda path=None: _id_rows())
    monkeypatch.setattr(diff_store, "load", lambda path=None: _diff_rows())
    monkeypatch.setenv("RAG_FACT_LAYER", "1")
    return None


# --------------------------------------------------------------------------- OFF byte-identical
def test_off_is_inert(monkeypatch):
    monkeypatch.delenv("RAG_FACT_LAYER", raising=False)
    assert fl.enabled() is False
    assert fl.tools() == []
    assert fl.resolve("青葉与信 F1最大化閾値でのF1", "numeric") is None
    # even forced, an OFF layer's tools list is empty (tool surface unchanged)
    assert fl.tools() == []


def test_off_investigator_tool_set_excludes_fact_tools(monkeypatch):
    monkeypatch.delenv("RAG_FACT_LAYER", raising=False)
    from src.rag.agent.investigator import build_generic_tools
    from src.rag.tools.profile import CorpusProfile
    names = {t.name for t in build_generic_tools(CorpusProfile())}
    assert not (names & fl.TOOL_NAMES)


# --------------------------------------------------------------------------- (a) deterministic lanes
def test_derived_lane_fires_best_f1(stores):
    r = fl.resolve("青葉与信マネジメントの回帰係数で全データ予測のF1最大化閾値でのF1は?", "numeric")
    assert r is not None and r["value"] == 0.42395962
    assert r["method"]["verified_operand"] is True and r["evidence"]["store"] == "derived_metrics"


def test_derived_lane_fires_prediction_id0(stores):
    r = fl.resolve("青葉与信マネジメント train.xlsx の回帰係数で id=0 を予測した値は?", "numeric")
    assert r is not None and r["value"] == 0.15001822


def test_derived_lane_defers_on_bare_mention(stores):
    # a bare "F1" with no 閾値/最大化 anchor must NOT fire (precision-first)
    assert fl.resolve("青葉与信のF1スコアの意味を説明して", "numeric") is None


def test_derived_lane_defers_when_case_ambiguous(stores):
    # no case hint ⇒ cannot bind a unique case ⇒ defer
    assert fl.resolve("F1最大化閾値でのF1は?", "numeric") is None


# SOT-2649 (idx57型): a shared-stem mention (「青葉のTX」) binds via metric presence when exactly one
# candidate carries the anchored metric; both-carry / two-distinct-names still defer.
_IDX57_Q = ("青葉のTXにて算出された回帰係数を用いて全データの予測値を計算し、正解データに対する"
            "F1スコアが最大となるように閾値を設定したときのF1スコアを答えてください")


def _two_aoba_rows(bio_has_model: bool):
    rows = _derived_rows()
    bio = {"case_id": "株式会社青葉バイオメディカル機器", "train_file": "…/train.csv",
           "sources": {"train_file": "…/train.csv"},
           "model": ({"threshold_sweep": {"best_f1": 0.9, "best_threshold": 0.4}}
                     if bio_has_model else {"present": False, "reason": "no target"})}
    return rows + [bio]


def test_derived_lane_shared_stem_binds_via_metric_presence(stores, monkeypatch):
    from src.rag.index import derived_metrics
    monkeypatch.setattr(derived_metrics, "load", lambda path=None: _two_aoba_rows(False))
    r = fl.resolve(_IDX57_Q, "numeric")
    assert r is not None and r["value"] == 0.42395962
    assert r["evidence"]["case"] == "青葉与信マネジメント株式会社"


def test_derived_lane_shared_stem_defers_when_both_carry_metric(stores, monkeypatch):
    from src.rag.index import derived_metrics
    monkeypatch.setattr(derived_metrics, "load", lambda path=None: _two_aoba_rows(True))
    assert fl.resolve(_IDX57_Q, "numeric") is None


def test_derived_lane_two_distinct_names_defer(stores, monkeypatch):
    from src.rag.index import derived_metrics
    rows = _derived_rows() + [{"case_id": "京橋信用ソリューションズ株式会社", "train_file": "…/train.csv",
                               "sources": {}, "model": {"threshold_sweep": {"best_f1": 0.455}}}]
    monkeypatch.setattr(derived_metrics, "load", lambda path=None: rows)
    # two DIFFERENT explicit company names (comparison shape) must never store-disambiguate to one
    assert fl.resolve("青葉与信と京橋信用のF1最大化閾値でのF1を比べてください", "numeric") is None


def test_enum_lane_fires_pure_enumeration(stores):
    r = fl.resolve("APR-M1に該当する案件を主略称ですべて列挙してください", "full_enumeration")
    assert r is not None
    assert set(r["value"].split("、")) == {"AYM", "SRK"}
    assert r["evidence"]["universe_size"] == 3 and r["method"]["verified_operand"] is True


def test_enum_lane_defers_on_aggregation_cue(stores):
    # NON-EMPTY 列挙＋合計 composite ⇒ defer to the tool + LLM (multi-part answer format is ambiguous)
    assert fl.resolve("APR-M1の案件を略称で列挙し契約金額を合計してください", "full_enumeration") is None


def test_enum_lane_empty_composite_fires(stores):
    # SOT-2649 (idx38型): 列挙+契約金額合計 composite over an EMPTY apr_code match-set is deterministic —
    # the certified universe (apr_code 3/3) proves 0 matches, so both parts are vacuous ⇒ 該当なし.
    q = "社内管理のAPRに照らして、APR-M3が必要な案件を主略称ですべて挙げ、それらの契約金額（税込）の合計を答えてください。"
    r = fl.resolve(q, "full_enumeration")
    assert r is not None and r["value"] == "該当なし"
    assert r["method"]["selection"] == "apr_code_empty_composite"
    assert r["evidence"]["matched"] == 0 and r["evidence"]["universe_size"] == 3


def test_enum_lane_empty_composite_defers_on_negation(stores):
    # complement phrasing voids the empty-set proof (APR-M3以外 is NON-empty) ⇒ hard defer
    q = "APR-M3以外の案件を主略称ですべて挙げ、それらの契約金額（税込）の合計を答えてください"
    assert fl.resolve(q, "full_enumeration") is None


def test_enum_lane_empty_composite_defers_on_count_aggregation(stores):
    # a 件数 composite would gold-format as 「0件」, not 該当なし ⇒ only the 金額合計 shape may fire
    q = "APR-M3が必要な案件を主略称ですべて挙げ、何件あるか答えてください"
    assert fl.resolve(q, "full_enumeration") is None


def test_enum_lane_empty_composite_defers_on_extra_predicate(stores):
    # an extra attribute predicate keeps the idx87-shape boundary even for the empty composite
    q = "完了案件のうちAPR-M3が必要な案件を主略称ですべて挙げ、契約金額（税込）の合計を答えてください"
    assert fl.resolve(q, "full_enumeration") is None


def test_enum_lane_pure_enumeration_defers_on_negation(stores):
    # 「APR-M1以外を列挙」: the APR-only lane would return the M1 set itself (the complement's inverse) —
    # negation phrasing must defer the WHOLE lane, not just composites
    assert fl.resolve("APR-M1以外に該当する案件を主略称ですべて列挙してください", "full_enumeration") is None


def test_enum_lane_defers_on_multiple_codes(stores):
    assert fl.resolve("APR-M1とAPR-M2の案件を略称で列挙", "full_enumeration") is None


def test_enum_lane_defers_on_extra_predicate(stores):
    # idx87 shape: single APR code BUT extra predicates (完了 / train_rows≥10000) ⇒ APR-only would
    # over-return a superset ⇒ MUST defer to case_filter + the LLM (precision-first, SOT-2601).
    q = "完了案件のうちAPR-M1に該当し顧客データのサンプル数が10000行以上の案件を略称ですべて挙げて"
    assert fl.resolve(q, "full_enumeration") is None


def test_enum_lane_defers_when_universe_not_fully_attributed(stores, monkeypatch):
    from src.rag.index import case_master
    rows = _case_rows()
    rows[0]["attributes"]["apr_code"] = {"value": None, "source": None}  # one case unattributed
    monkeypatch.setattr(case_master, "load", lambda path=None: rows)
    assert fl.resolve("APR-M1の案件を主略称ですべて列挙", "full_enumeration") is None


def test_resolve_defers_on_unmapped_contract(stores):
    assert fl.resolve("何かの質問", "multi_hop") is None


# --------------------------------------------------------------------------- (b) tools
def test_tools_exposed_when_enabled(stores):
    names = {t[0] for t in fl.tools()}
    assert names == fl.TOOL_NAMES


def test_all_tools_return_valid_contracts(stores):
    handlers = {n: fn for n, _d, _p, fn in fl.tools()}
    assert _contract.is_contract(handlers["case_filter"]("apr_code=APR-M1", "abbrev"))
    assert _contract.is_contract(handlers["id_lookup"]("M01"))
    assert _contract.is_contract(handlers["metric_lookup"]("青葉与信", "model.threshold_sweep.best_f1"))
    assert _contract.is_contract(handlers["diff_lookup"](project="青嶺", old="提案書_v1", new="提案書_v3"))


def test_case_filter_projects_with_provenance(stores):
    handlers = {n: fn for n, _d, _p, fn in fl.tools()}
    out = handlers["case_filter"]("apr_code=APR-M1", "abbrev,contract_amount_incl_tax")
    labels = {row["abbrev"] for row in out["value"]}
    assert labels == {"AYM", "SRK"}
    assert out["evidence"]["universe_size"] == 3
    # each cell carries its source (verified operand provenance)
    assert out["evidence"]["rows"][0]["cells"]["contract_amount_incl_tax"]["source"] is not None


def test_id_lookup_returns_raw_line(stores):
    handlers = {n: fn for n, _d, _p, fn in fl.tools()}
    out = handlers["id_lookup"]("m-01")  # separator/case-insensitive via id_master.lookup
    assert out["value"] and "キックオフ完了" in out["value"][0]["line"]


def test_metric_lookup_dotted_path_and_catalog(stores):
    handlers = {n: fn for n, _d, _p, fn in fl.tools()}
    assert handlers["metric_lookup"]("青葉与信", "model.prediction_id0.prediction")["value"] == 0.15001822
    cat = handlers["metric_lookup"]("青葉与信", "")["value"]
    assert "model.threshold_sweep.best_f1" in cat["available"]


def test_diff_lookup_excludes_attribute(stores):
    handlers = {n: fn for n, _d, _p, fn in fl.tools()}
    out = handlers["diff_lookup"](project="青嶺", exclude_attributes="status_transition")
    kinds = [c["intent"] for c in out["value"]]
    assert "SUBSTANTIVE" in kinds and out["evidence"]["after_filter"] == 1  # status_transition dropped


# --------------------------------------------------------------------------- MCP / investigator single-source
def test_enabled_investigator_and_mcp_expose_fact_tools(monkeypatch):
    monkeypatch.setenv("RAG_FACT_LAYER", "1")
    from src.rag.agent.investigator import build_tools
    from src.rag.mcp import server as mcp_server
    from src.rag.tools.profile import CorpusProfile
    tools = build_tools(CorpusProfile())
    names = {t.name for t in tools}
    assert fl.TOOL_NAMES <= names
    specs = {s["name"] for s in mcp_server.tool_specs(tools)}
    assert fl.TOOL_NAMES <= specs  # single source of truth: MCP surface == investigator surface


# --------------------------------------------------------------------------- commit_gate verified operand
def test_commit_gate_treats_store_value_as_grounded_operand():
    from src.rag.agent import commit_gate, question_contract as qc
    hist = [{"name": "metric_lookup", "ok": True, "value": 0.42396}]
    dec = commit_gate.evaluate("F1は?", qc.NUMERIC, "0.42396", session_tool_history=hist)
    assert dec.verdict == commit_gate.COMMIT
    assert dec.telemetry["checks"]["numeric"]["grounded"] is True


def test_commit_gate_rejects_ungrounded_numeric_without_store():
    from src.rag.agent import commit_gate, question_contract as qc
    dec = commit_gate.evaluate("F1は?", qc.NUMERIC, "0.99999", session_tool_history=[])
    assert dec.verdict in (commit_gate.REJECT, commit_gate.ABSTAIN_V)
