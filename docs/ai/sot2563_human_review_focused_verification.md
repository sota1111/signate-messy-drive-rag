# SOT-2563 human-review focused verification

## Scope

PR #101 merge後の `review=human` 指示に従い、SOT-2550 §3 の時間切れ19問を本番 investigatorで再回答した。gold100は実行していない。

## Finding and remediation

旧実装は `min(180s, remaining)` で全残予算をscanへ渡したため、`deadline_hit`後に別ルートを選ぶターンが残らなかった。baselineは完了9問中6 timeout（最大314.5s）で不合格。`RAG_FILE_GREP_RESERVE_S`（既定30s）を追加し、伝播残予算から予約時間を引いたscan deadlineに変更した。直接呼出しは従来どおり180s capで非破壊。

## Results

| cycle | scope | match | abstain | wrong | timeout | max elapsed | decision |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| baseline partial | 9 completed | 1 observed | 8 observed | 未採点 | 6 | 314.5s | fail / early stop |
| 2 reserve + index candidates | 19 | 3 | 15 | 1 | **0** | **148.4s** | promote |
| 3 all deterministic flags | cycle2 NG 16 | 0 | 15 | 1 | 2 | 234.5s | reject |

cycle2のmatchは idx23 (`398,750円`)、idx41 (`11`)、idx43 (`石川 直樹`)。idx28は `Age`（gold `BMI`）でwrong、残り15件は安全棄権。時間切れ回収は達成したが、全問いの正答化は未達であり、残件はretrieval/contract個別対策を要する。

Artifacts: `artifacts/sot2563_focused_answers_cycle2.json`, `.details.jsonl`, cycle3 equivalents, `sot2563_focused_answers_baseline_partial.log`。
