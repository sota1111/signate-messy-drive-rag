# Model Divergence Diagnostics (SOT-2638)

- flash verdicts: {'MATCH': 47, 'ABSTAIN': 45, 'WRONG': 8}
- sonnet verdicts: {'MATCH': 40, 'ABSTAIN': 36, 'WRONG': 24}
- **divergence total: 19**
  - flash MATCH → Sonnet 非MATCH: **13** [0, 4, 11, 20, 31, 35, 41, 42, 53, 59, 72, 88, 96]
  - Sonnet MATCH → flash 非MATCH: **6** [68, 70, 79, 83, 93, 94]

## Divergence buckets
- **commit_precision** (n=9) [0, 11, 20, 31, 41, 42, 59, 88, 96] — wrong 側乖離 — failing backend committed WRONG; Commit Gate should catch
- **reachability** (n=10) [4, 35, 53, 68, 70, 72, 79, 83, 93, 94] — 到達性乖離 — failing backend ABSTAINed; trace port applies
- **judge_noise** (n=0) [] — judge 揺らぎ疑い — excluded from port candidates

## Deterministic-direct verdict-agreement check
- deterministic-direct idx (n=12): [7, 10, 15, 19, 26, 33, 44, 54, 58, 74, 80, 86]
- all deterministic-direct idx agree across backends

## Mutual-port candidate lists (judge_noise excluded)
- → Sonnet (flash won): n=13 [0, 4, 11, 20, 31, 35, 41, 42, 53, 59, 72, 88, 96]
- → flash (Sonnet won): n=6 [68, 70, 79, 83, 93, 94]

## Per-idx divergence table
| idx | contract | flash | sonnet | det | direction | bucket | judge_noise_reasons |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 0 | version_diff | MATCH | WRONG |  | flash->sonnet | commit_precision |  |
| 4 | numeric | MATCH | ABSTAIN |  | flash->sonnet | reachability |  |
| 11 | format_check | MATCH | WRONG |  | flash->sonnet | commit_precision |  |
| 20 | simple_lookup | MATCH | WRONG |  | flash->sonnet | commit_precision |  |
| 31 | multi_hop | MATCH | WRONG |  | flash->sonnet | commit_precision |  |
| 35 | numeric | MATCH | ABSTAIN |  | flash->sonnet | reachability |  |
| 41 | numeric | MATCH | WRONG |  | flash->sonnet | commit_precision |  |
| 42 | format_check | MATCH | WRONG |  | flash->sonnet | commit_precision |  |
| 53 | numeric | MATCH | ABSTAIN |  | flash->sonnet | reachability |  |
| 59 | simple_lookup | MATCH | WRONG |  | flash->sonnet | commit_precision |  |
| 68 | numeric | ABSTAIN | MATCH |  | sonnet->flash | reachability |  |
| 70 | simple_lookup | ABSTAIN | MATCH |  | sonnet->flash | reachability |  |
| 72 | numeric | MATCH | ABSTAIN |  | flash->sonnet | reachability |  |
| 79 | simple_lookup | ABSTAIN | MATCH |  | sonnet->flash | reachability |  |
| 83 | numeric | ABSTAIN | MATCH |  | sonnet->flash | reachability |  |
| 88 | simple_lookup | MATCH | WRONG |  | flash->sonnet | commit_precision |  |
| 93 | simple_lookup | ABSTAIN | MATCH |  | sonnet->flash | reachability |  |
| 94 | simple_lookup | ABSTAIN | MATCH |  | sonnet->flash | reachability |  |
| 96 | simple_lookup | MATCH | WRONG |  | flash->sonnet | commit_precision |  |

## Sources
- flash_score: `artifacts/gold100_cycle4_flash.json`
- sonnet_score: `artifacts/gold100_cycle4_sonnet.json`
- flash_details: `/workspaces/signate-messy-drive-rag/artifacts/predictions_test_investigator.details.jsonl`
- sonnet_resume: `artifacts/gold100_cycle4_sonnet_resume.jsonl`

_Offline aggregation only; no LLM re-run; serve path unchanged._
