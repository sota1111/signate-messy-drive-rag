# Solo Worker Report — SOT-2601

## Summary

4較正（SOT-2597/2598/2599/2600）が全て Done・main マージ済みで、事前処理ストアが現行 corpus/main と整合していることを確認した。指定された較正後7フラグ再統合 gold100 を重複なく1回だけ実行し、match 26 / abstain 60 / wrong 14 / net 12 を得た。r1/r3 net 32 から大幅回帰したため関門2は FAIL、統合候補を rejected とし champion 候補 r1 + FORMAT_EVENTS を維持した。

## Changed Files

- `docs/ai/gold100_recalibrated_integration.md` — r1/r2/r3比較、一問遷移、状態コード、4較正の実測判定。
- `docs/ai/experiment_ledger.jsonl` — SOT-2601-F 軸を rejected として追記。
- `docs/gold_offline_history.jsonl` — gold100 実行履歴を自動追記。
- `artifacts/gold_100_review.md` / `.csv` — F 実測の100問レビュー表へ更新。

## Commands Run

- `.venv/bin/python -m scoring.gold_offline --run --workers 8 --out artifacts/gold100_sot2601_recal.json` — exit 0、100/100完走、$8.1767。
- `.venv/bin/python -m pytest` — 1130 passed、11 warnings、505.09s。
- JSON/JSONL構文検査、r1/r2/r3/F 一問遷移集計、`git diff --check`。
- npm lint/typecheck/test/e2e — N/A（Python repo、package.jsonなし。pytestで全検証）。

## Acceptance Criteria

- [x] 4較正 Done/merged と事前処理完了を確認後、gold100 を一度だけ実行し記録した。
- [x] net と r1/r2/r3 差分、一問遷移、較正効果、関門2 FAIL を report 化した。
- [x] history と experiment ledger に記録した。
- [x] 全1130テストが通過した。

## Risks

- 評価対象の統合候補は net 12 のため不採用。registry idx0/59 は回復したが、derived match は2、UNANSWERABLEは35、idx70誤答は残存した。
- ローカル評価は実LBのproxyに過ぎないが、本候補は関門2を大幅に割るためLB投入しない。

## Linear Report: POSTED

## Acceptance: PASS

## Next Action: READY_FOR_REVIEW
