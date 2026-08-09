# SOT-2563 Human-review Reopen Final Report

## Summary

PR #101 のend-to-end未実測を埋め、旧 `min(cap, remaining)` がscan後のroute-switch時間を残さない欠陥を確認した。`RAG_FILE_GREP_RESERVE_S`（既定30秒）を追加し、investigator伝播時だけ残予算から予約時間を差し引く。直接呼出しの180秒capは不変。

## Changed Files

- `src/rag/tools/call_budget.py` — 別ルート選択用の予約時間をdeadline算出へ追加。
- `tests/test_file_grep_deadline.py` — env、残予算30/90/180秒、直接呼出し非劣化を追加（9 tests）。
- `scripts/sot2563_focused_answers.py` — 19問/NG subsetの本番回答・offline judge再現ハーネス。
- `docs/ai/sot2563_human_review_focused_verification.md` — baseline/cycle2/cycle3結果。
- `docs/ai/experiment_ledger.jsonl` — promoted/rejected軸を追記。

## Verification

- baseline partial: 完了9問中timeout 6、最大314.5s、旧PRだけでは不合格。
- cycle2（reserve + index candidates、19問）: match 3 / abstain 15 / wrong 1、timeout **0**、最大148.4s。idx23/41/43をmatchへ回収。
- cycle3（残NG16へ全決定論flags）: 0/15/1、timeout 2、最大234.5sで退行したためreject。
- `.venv/bin/python -m pytest -q`: **965 passed**, warnings 7（既知WMFのみ）、290.67s。
- `git diff --check`: PASS。
- npm gates: N/A（Python repo、package.jsonなし）。gold100はIssue指示どおり未実行。

## Acceptance Criteria

- [x] 単一file_grepがdeadline内で打切られ、route-switch用時間を予約する。
- [x] 対象19問でtimeout stopを0へ回収し、一部をmatch化した。
- [x] 180秒直接capと既match最大125秒を切らない既定を維持。
- [x] focused/offline reportとexperiment ledgerを更新し、gold100未実行を明記。

## Remaining Issues

cycle2で15 safe abstain / 1 wrong（idx28 Age→gold BMI）が残る。追加の既存deterministic機能一括ONは改善せず時間退行したため採用しない。残件は問い別retrieval/contract gapであり、本Issueのfile_grep単一呼出し暴走とは分離して次の改善軸にする。

## Linear Report: POSTED

## Acceptance: PASS

## Next Action: READY_FOR_REVIEW
