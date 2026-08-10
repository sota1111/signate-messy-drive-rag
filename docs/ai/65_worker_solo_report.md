# Solo Worker Report — SOT-2610

## Summary

Wave A 全4子を含む main HEAD で、指定された公式 gold100 を一度だけ実行した。match 47 / abstain 46 / wrong 7 / net 40 となり、champion net 32 から +8、wrong は非増加だった。比較、一問遷移、型別効果、状態コード、関門2判定を `docs/ai/gold100_wave_a.md` に記録し、experiment ledger を promoted で更新した。

## Changed Files

- `scripts/sot2610_gold100.sh` — 公式モデル・champion フラグ・Wave A router を固定した再現用実行スクリプト
- `artifacts/gold_100_review.md` / `.csv` — Wave A gold100 の100問レビュー
- `docs/gold_offline_history.jsonl` — 実測履歴
- `docs/ai/gold100_wave_a.md` — r1/r2/SOT-2601比較、一問遷移、型別・状態コード、関門2判定
- `docs/ai/experiment_ledger.jsonl` — Wave A 統合軸を promoted として記録

## Commands Run

- `bash scripts/sot2610_gold100.sh` — PASS、100問完走（1回のみ）
- `.venv/bin/python -m pytest -q --ignore=tests/test_gate.py --ignore=tests/test_tiebreak.py` — 1193 passed, 7 warnings
- JSONL 全行 parse / gold JSON parse — PASS
- diff review — 意図した測定成果物・文書・履歴・ledger・再現スクリプトのみ

## Acceptance Criteria

- [x] Wave A PR#125–128 が main に統合済みであることを確認後、gold100 を一度だけ実行・記録
- [x] r1/r2/SOT-2601 差分、一問遷移、型別効果、状態コード、関門2判定を report 化
- [x] history / experiment ledger に記録
- [x] 関門2: match 39→47、wrong 7→7、net 32→40 で PASS

## Risks

- gold100 は確率的生成を含む単一実測で、Wave ごとの完全な因果アブレーションではない。
- 残存棄権46件中39件が BUDGET_EXHAUSTED。実 LB rank が最終一次 KPI。
- `tests/test_gate.py` / `tests/test_tiebreak.py` は既知の live-exec hang のため、従来どおり offline 全体テストから除外した。

## Linear Report: POSTED

## Acceptance: PASS

## Next Action: READY_FOR_REVIEW
