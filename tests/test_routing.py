"""SOT-2498 — offline tests for contract-based routing wired into the investigator.

All network-free: the classifier runs deterministically, the routing hint is a pure string, and the
live ``answer_question`` path is exercised with a *fake* model factory that captures the injected system
prompt (no Vertex call).
"""
from __future__ import annotations

from src.rag.agent import investigator as inv
from src.rag.agent import question_contract as qc
from src.rag.agent import routing
from src.rag.agent.investigator import ABSTAIN, SUBMIT_ANSWER, Call, Step, Usage, investigate
from src.rag.agent.question_contract import CONTRACTS


# --------------------------------------------------------------------------- scripted fake model
class _ScriptedModel:
    def __init__(self, steps, *, model_name="fake-model"):
        self._steps = list(steps)
        self._i = 0
        self.model_name = model_name

    def next(self, _tool_responses):
        if self._i >= len(self._steps):
            return Step(function_calls=(), final_text=ABSTAIN, usage=Usage(1, 1))
        step = self._steps[self._i]
        self._i += 1
        return step


def _submit(answer, *, confidence=0.9):
    return Step(function_calls=(Call(SUBMIT_ANSWER, {"answer": answer, "confidence": confidence}),),
                usage=Usage(10, 5))


# --------------------------------------------------------------------------- contract → first-tools map
def test_first_tools_map_covers_every_contract():
    # every one of the nine contracts must have a non-empty, ordered first-move tool list
    assert set(routing.CONTRACT_FIRST_TOOLS) == set(CONTRACTS)
    for contract, tools in routing.CONTRACT_FIRST_TOOLS.items():
        assert tools, f"{contract} has no first-move tools"


def test_first_tools_reference_only_real_investigator_tools():
    # the routing hint may only name tools the investigator actually exposes (no phantom capability)
    from src.rag.tools.profile import CorpusProfile

    real = {t.name for t in inv.build_tools(CorpusProfile())}
    for tools in routing.CONTRACT_FIRST_TOOLS.values():
        assert set(tools) <= real


def test_data_contracts_lead_with_canonical_direct_route():
    # 数値/横断集計 → canonical direct route / aggregate first (the retrieval_miss fix, not more search)
    assert routing.CONTRACT_FIRST_TOOLS[qc.NUMERIC][0] == "canonical_route"
    assert routing.CONTRACT_FIRST_TOOLS[qc.CROSS_AGGREGATE][0] == "corpus_aggregate"
    # specialised-tool contracts lead with their dedicated tool
    assert routing.CONTRACT_FIRST_TOOLS[qc.SPATIAL][0] == "seating_lookup"
    assert routing.CONTRACT_FIRST_TOOLS[qc.VERSION_DIFF][0] == "version_diff"
    assert routing.CONTRACT_FIRST_TOOLS[qc.CHART_READ][0] == "read_chart_values"
    assert routing.CONTRACT_FIRST_TOOLS[qc.FORMAT_CHECK][0] == "highlight_extract"
    # 単純検索 keeps the fast retrieval path first
    assert routing.CONTRACT_FIRST_TOOLS[qc.SIMPLE_LOOKUP][0] in ("file_grep", "find_files")


# --------------------------------------------------------------------------- route hint content
def test_route_hint_lists_first_tools_and_completion_conditions():
    q = "最も着手金が高い案件はどこですか。"  # cross_aggregate
    contract = qc.classify(q)
    hint = routing.route_hint(contract, q)
    assert contract.label in hint
    # the recommended first tool must appear in the ordered hint
    assert routing.CONTRACT_FIRST_TOOLS[contract.contract][0] in hint
    # completion conditions are echoed so the agent knows when it may commit
    for cond in contract.completion_conditions:
        assert cond in hint


def test_route_hint_for_numeric_prefers_canonical_route():
    q = "京橋信用ソリューションズの train.xlsx の loan_amnt の平均は。"
    contract = qc.classify(q)
    assert contract.contract == qc.NUMERIC
    hint = routing.route_hint(contract, q)
    assert "canonical_route" in hint
    assert "生データ再計算・notebookの実行出力 > 文書中のmarkdown/散文" in hint
    assert "kind='train'" in hint
    # the Adaptive-RAG note tells it not to run a uniform high-cost search first
    assert "一律" in hint or "chunk" in hint


def test_chart_route_forbids_vision_numeric_evidence():
    q = "AG_ratioのヒストグラムで最も多いカウント数はいくつですか。"
    contract = qc.classify(q)
    assert contract.contract == qc.CHART_READ
    assert routing.first_tools_for(contract, q) == ("read_chart_values",)
    hint = routing.route_hint(contract, q)
    assert "operation='histogram_max_count'" in hint
    assert "画像から読んだ数値では回答を確定しない" in hint


def test_gantt_week_route_uses_native_geometry_before_vision():
    q = "提案書.pptxのモデル改善は実行予定スケジュール上、第何週目ですか。"
    contract = qc.classify(q)
    assert contract.contract == qc.CHART_READ
    tools = routing.first_tools_for(contract, q)
    assert tools[0] == "read_office"
    assert tools.index("read_office") < tools.index("caption_image")
    hint = routing.route_hint(contract, q)
    assert "【ガント週グリッド】" in hint and "半開区間" in hint and "裏取り" in hint


def test_regulation_route_hint_forbids_no_rule_only_answer():
    q = "契約条件で上限超過時の精算方法に関する規定内容を答えてください。"
    contract = qc.classify(q)
    assert routing.first_tools_for(contract, q) == ("find_files", "read_office", "file_grep")
    hint = routing.route_hint(contract, q)
    assert "『特別規定なし』は規定内容回答の完了ではない" in hint
    for field in ("単価", "課金単位", "精算周期", "上限"):
        assert field in hint
    assert "字面grepの空振りだけで棄権しない" in hint


def test_negative_correlation_hint_excludes_target_by_name_before_argmin():
    q = "顧客データにおいて、目的変数と最も強い負の相関を持つカラムは何ですか。"
    contract = qc.classify(q)
    assert contract.contract == qc.NUMERIC
    hint = routing.route_hint(contract, q)
    assert "目的変数自身を候補から列名で明示的に除外" in hint
    assert "昇順に並べた先頭（最小値）" in hint
    assert "無条件に2番目の最小値" in hint


def test_requested_name_surface_is_preserved_without_expansion_or_honorific():
    abbreviation_q = "条件に合う案件を、主略称ですべて挙げてください。"
    abbreviation_hint = routing.route_hint(qc.classify(abbreviation_q), abbreviation_q)
    assert "主略称をそのまま列挙" in abbreviation_hint
    assert "正式社名や案件名へ展開" in abbreviation_hint

    fullname_q = "甲側の主担当者をフルネームで教えてください。"
    fullname_hint = routing.route_hint(qc.classify(fullname_q), fullname_q)
    assert "人名のフルネームのみ" in fullname_hint
    assert "敬称（様/氏）" in fullname_hint


def test_data_asset_simple_lookup_leads_with_canonical_route():
    # idx61-style: a config-hyperparam lookup that lives inside modeling.py (a data asset) is a
    # simple_lookup contract, but its evidence is a needle the chunk index misses → canonical_route first.
    q = ("京橋信用ソリューションズの分析コードにおいて、勾配ブースティングの n_estimators は。")
    contract = qc.classify(q)
    assert contract.contract == qc.SIMPLE_LOOKUP
    assert routing.references_data_asset(q)
    first = routing.first_tools_for(contract, q)
    assert first[0] == "canonical_route"          # routed to the file, not the fast chunk search
    hint = routing.route_hint(contract, q)
    assert "canonical_route" in hint and ("一律" in hint or "chunk" in hint)


def test_plain_simple_lookup_keeps_fast_retrieval_path():
    # a non-data-asset simple lookup keeps the fast chunk-search first move (従来の高速経路)
    q = "この会社の代表者名は誰ですか。"
    contract = qc.classify(q)
    assert contract.contract == qc.SIMPLE_LOOKUP
    assert not routing.references_data_asset(q)
    first = routing.first_tools_for(contract, q)
    assert first[0] in ("file_grep", "find_files")
    assert first == routing.CONTRACT_FIRST_TOOLS[qc.SIMPLE_LOOKUP]


def test_route_hint_injects_no_corpus_fact():
    # the hint must name only tools + generic conditions — never a concrete corpus secret / 略称
    q = "契約金額はいくらですか。"
    hint = routing.route_hint(qc.classify(q), q)
    for leak in ("password", "パスワード", "P@ss", "京橋", "アステル"):
        assert leak not in hint


def test_routed_system_prompt_appends_hint_to_base():
    q = "契約金額はいくらですか。"
    contract = qc.classify(q)
    base = "BASE-PROMPT"
    out = routing.routed_system_prompt(base, contract, q)
    assert out.startswith(base)
    assert routing.route_hint(contract, q) in out


# --------------------------------------------------------------------------- investigate records contract
def test_investigate_records_contract_label_on_investigation():
    model = _ScriptedModel([_submit("42", confidence=0.8)])
    res = investigate(model, "…", inv.build_tools(_prof()), max_turns=3, contract="numeric")
    assert res.contract == "numeric"
    assert res.to_dict()["contract"] == "numeric"


def test_investigate_contract_defaults_to_none_and_is_byte_identical():
    # default (no contract) keeps the loop unchanged; the meta key is present but None
    model = _ScriptedModel([_submit("42", confidence=0.8)])
    res = investigate(model, "…", inv.build_tools(_prof()), max_turns=3)
    assert res.contract is None
    assert res.to_dict()["contract"] is None


def _prof():
    from src.rag.tools.profile import CorpusProfile
    return CorpusProfile()


# --------------------------------------------------------------------------- answer_question wiring (offline)
def test_answer_question_injects_routing_hint_and_records_contract(monkeypatch):
    captured = {}

    def fake_factory(question, tools, *, model=None, system=None):
        captured["system"] = system
        captured["question"] = question
        return _ScriptedModel([_submit("直行で得た値", confidence=0.9)])

    monkeypatch.setattr(inv, "gemini_model_factory", fake_factory)
    # ledgers off so the offline test writes nothing; routing on (default) is what we assert
    res = inv.answer_question(
        "京橋信用ソリューションズの train.xlsx の loan_amnt の平均は。",
        ledger=False, calc_ledger=False, research=False, enumeration=False)

    assert res.contract == qc.NUMERIC              # classified contract recorded on the answer meta
    assert res.answer.answer == "直行で得た値"
    # the base prompt is preserved and the routing hint was appended into the model's system prompt
    assert captured["system"].startswith(inv.SYSTEM_PROMPT)
    assert "契約型ルーティング" in captured["system"]
    assert "canonical_route" in captured["system"]


def test_answer_question_routing_off_uses_base_prompt_and_no_contract(monkeypatch):
    captured = {}

    def fake_factory(question, tools, *, model=None, system=None):
        captured["system"] = system
        return _ScriptedModel([_submit("x", confidence=0.5)])

    monkeypatch.setattr(inv, "gemini_model_factory", fake_factory)
    res = inv.answer_question(
        "契約金額はいくらですか。", routing=False,
        ledger=False, calc_ledger=False, research=False, enumeration=False)

    assert res.contract is None
    assert captured["system"] is None              # base SYSTEM_PROMPT (no hint) when routing is off


def test_answer_question_contract_flash_arbiter_is_consulted(monkeypatch):
    # an ambiguous question with no deterministic cue defers to the injected flash arbiter
    seen = {}

    def fake_factory(question, tools, *, model=None, system=None):
        seen["system"] = system
        return _ScriptedModel([_submit("y", confidence=0.5)])

    monkeypatch.setattr(inv, "gemini_model_factory", fake_factory)
    res = inv.answer_question(
        "これはどうですか。", routing=True, contract_flash=lambda q: qc.CHART_READ,
        ledger=False, calc_ledger=False, research=False, enumeration=False)
    # the flash verdict flows through as the recorded contract + its hint
    assert res.contract == qc.CHART_READ
    assert "read_chart_values" in seen["system"]


# --------------------------------------------------------------------------- abstain ledger carries contract
def test_abstain_ledger_records_contract(tmp_path):
    from src.rag.agent import abstain_ledger as al

    model = _ScriptedModel([Step(function_calls=(), final_text=ABSTAIN, usage=Usage(1, 1))])
    res = investigate(model, "根拠なし", inv.build_tools(_prof()), max_turns=2,
                      contract="simple_lookup", ledger=str(tmp_path / "l.jsonl"))
    assert res.contract == "simple_lookup"
    signals = al.AbstainSignals(stop_reason=res.stop_reason)
    rec = al.record_from_investigation(res, signals)
    assert rec.contract == "simple_lookup"
    assert rec.to_dict()["contract"] == "simple_lookup"
