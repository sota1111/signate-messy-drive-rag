# Final Report — SOT-2613

全 Wave を含む main `cb22c99` で、公式 `gemini-3.6-flash` 構成の gold100 を本 issue で一度だけ完走した。結果は match 47 / abstain 43 / wrong 10 / net 37。r1 net32 より +5 だが、直前 champion の Wave A net40 より −3 のため関門2 FAIL とし、Wave A を local champion のまま維持する。

決定論直答は 13/100（12 match / 1 wrong）、LLM フォールバックは 87/100（35 match / 43 abstain / 9 wrong）。B1 対象 document_extract は match 13→17 / wrong 2→1 と改善した一方、fact_lookup・derived・version_diff の回帰が全体 net を押し下げた。詳細比較、型別効果、状態コード、次軸を `docs/ai/gold100_inversion_final.md` に記録し、review artifacts、history、experiment ledger（rejected）を更新した。

検証は `.venv/bin/python -m compileall -q src scoring scripts` PASS、offline pytest は 1238 passed / 9 warnings。既知の live-execution hang である `tests/test_gate.py` と `tests/test_tiebreak.py` は repository の従来方針どおり除外した。npm lint/typecheck/e2e は Python-only repo のため N/A。gold100 自体は exit 0、全100問・全指標・全43棄権の state code を記録済み。

次軸は全量再試行ではなく、idx74 version_diff の deterministic naturalization、fact_lookup の同値フォーマット判定、derived の単位・閾値 formatter を focused trace で分離する。実 LB rank を一次 KPI とし、Wave A+B 構成は現時点で昇格しない。

## Acceptance: PASS

## Next Action: READY_FOR_REVIEW
