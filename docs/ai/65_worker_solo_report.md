# Worker Report — SOT-2589 (solo continuation)

## Summary
Confidence 閾値による棄権判定を、採点式 `U = P(P) + 0.5·P(A) − P(I)` に対応する
opt-in 三段 gate（HARD_ACCEPT / SOFT_ACCEPT / ABSTAIN）へ置換できる経路を追加した。
hard evidence を主入力、retrieval を副入力、verbal confidence を最大 0.05 の補助信号とし、
`BUDGET_EXHAUSTED` は診断情報にのみ保持して棄権条件から除外した。既定は `RAG_EU_GATE=OFF`。

## Changed Files
- `src/rag/agent/eu_gate.py` — typed signal bundle、grade probabilities、expected utility、三段判定。
- `src/rag/agent/gate.py` — opt-in live wiring、既存 confidence/exec gate との接続。
- `tests/test_eu_gate.py`, `tests/test_gate.py` — utility、hard blockers、budget exclusion、配線、OFF同一性。
- `scripts/measure_eu_gate.py` — gold100 coverage/risk/expected-score/risk-coverage 診断。
- `docs/ai/eu_gate_results.md`, `docs/ai/experiment_ledger.jsonl` — 実測結果と判断を記録。

## Commands Run
- `.venv/bin/python -m pytest tests/test_eu_gate.py tests/test_gate.py -q` — PASS（53 tests）。
- `.venv/bin/python -m compileall -q src scripts tests` — PASS。
- `RAG_EU_GATE=1 PYTHONPATH=. .venv/bin/python scripts/measure_eu_gate.py` — PASS。
- `.venv/bin/python -m pytest tests/ scoring/ -q` — **1110 passed**, 9 warnings（既知 openpyxl WMF）、396.16s。
- `git diff --check` — PASS。

## Acceptance Criteria
- [x] `U = P(P)+0.5P(A)-P(I)>0` の SOFT_ACCEPT と、決定論 evidence complete の HARD_ACCEPT、epistemic blocker の ABSTAIN を実装。
- [x] hard evidence signals を主要重み、retrieval を副次、verbal confidence を最大 0.05 の補助重みにした。
- [x] `budget_exhausted` は decision で参照せず、単独では同一 signals の tier/utility/commit を変えないテストを追加。
- [x] gold100 baseline coverage 0.46、incorrect rate 0.1522、expected score 0.32 と risk-coverage curve を記録。
- [x] 既定 OFF。legacy/OFF/ON-without-signals の `GateDecision.to_dict()` byte-identical をテストし、全回帰 1110 pass。

## Risks
- offline proxy signals は回答済み pool の誤答を分離できず（mean U: MATCH 0.594 / WRONG 0.631）、
  counterfactual は coverage 0.44、expected score 0.30。したがって lane は既定 OFF のまま、昇格は live
  gold100 A/B と実 LB の確認が必要。これは blocker ではなく、ledger では `inconclusive` と記録した。

## Linear Report: POSTED
## Acceptance: PASS
## Next Action: READY_FOR_REVIEW
