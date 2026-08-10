# Gold-100 7フラグ再統合（4較正マージ後）— SOT-2601

- 実行日時: 2026-08-10T09:43:54–09:55:50Z
- main HEAD: `55c36af`（SOT-2597/2598/2599/2600 の全較正を含む）
- コマンド: `PYTHONPATH=/tmp/genai_patch:. .venv/bin/python -m scoring.gold_offline --run --workers 8 --out artifacts/gold100_sot2601_recal.json`
- 構成: r1 の回答増フラグ群 + `RAG_FORMAT_EVENTS` を維持し、較正済み `RAG_DOCUMENT_REGISTRY / RAG_EVIDENCE_PACKET / RAG_POT_HARD_LANE / RAG_ENUM_SCAN / RAG_DIFF_ALIGN / RAG_EU_GATE` を再統合（早期棄権較正も有効化）。
- 事前処理: コーパス不変、retrieval/evidence/canonical/profile/document-registry/structure の全ストアが現行 main と整合済みのため再利用。
- 生レポート: `artifacts/gold100_sot2601_recal.json`（gitignore）、履歴: `docs/gold_offline_history.jsonl`。

## 結果: net 12（関門2 FAIL、候補は不採用）

| 指標 | r1 | r2（全ON・較正前） | r3（現champion候補） | F（較正後再統合） | F vs r3 |
| --- | ---: | ---: | ---: | ---: | ---: |
| match | 39 | 31 | 39 | **26** | **−13** |
| abstain | 54 | 60 | 54 | **60** | +6 |
| wrong | 7 | 9 | 7 | **14** | **+7** |
| **net (match−wrong)** | **32** | **22** | **32** | **12** | **−20** |
| cost | $10.31 | $8.99 | $12.16 | **$8.18** | −$3.98 |

狙いの `net > 32` を満たさず、match 非劣化・wrong 非増加の双方に違反した。関門2は **FAIL**。統合フラグ群は昇格せず、現 champion 候補（r1 + FORMAT_EVENTS、net 32）を維持する。ローカル proxy と実 LB の相関が弱い（既知 ρ≈−0.09）ため LB を一次 KPI とする方針は不変だが、この大幅なローカル回帰を持つ候補は LB 投入前に棄却する。

## r3 → F 一問遷移

| 遷移 | 件数 | idx |
| --- | ---: | --- |
| MATCH→MATCH | 23 | 0,2,3,10,13,18,20,23,26,31,43,44,46,51,52,58,59,60,81,84,86,89,93 |
| MATCH→ABSTAIN | **14** | 4,19,25,35,41,49,54,62,66,71,72,77,91,94 |
| MATCH→WRONG | **2** | 11,96 |
| ABSTAIN→MATCH | 3 | 7,85,88 |
| ABSTAIN→ABSTAIN | 42 | 5,8,9,14,17,22,24,27,29,30,32,33,34,36,37,38,39,40,42,45,47,48,55,56,57,61,64,67,68,69,73,75,76,79,82,83,87,90,92,95,98,99 |
| ABSTAIN→WRONG | **9** | 1,6,15,16,50,63,70,78,97 |
| WRONG→ABSTAIN | 4 | 12,28,53,65 |
| WRONG→WRONG | 3 | 21,74,80 |

状態コードは `UNANSWERABLE=35`、`BUDGET_EXHAUSTED=13`、`NOT_RETRIEVED=8`、`EVIDENCE_INCOMPLETE=3`、`PARSED_AMBIGUOUS=1`。r2 の早期棄権回帰を解消できず、さらに誤答が増えた。

## 4較正の実測判定

- registry（idx0/59/74）: idx0/59 は r2 ABSTAIN→F MATCH に回復し、fallback 較正は有効。idx74 は回答自体が意味同値に見えるものの judge は WRONG（r3 も WRONG）で、resolver 回帰ではない。
- PoT derived 回復: derived は `match=2 / abstain=26 / wrong=4`。r2 の match=3 からも回復せず、狙い不成立。
- iters≤2 UNANSWERABLE: `UNANSWERABLE=35`（r2=32 より悪化）。早期棄権較正は統合条件下で不成立。
- enum idx70: 引き続き「該当するIDはありません」で WRONG。universe guard が実 serve-path の断定を遮断できていない。

## 結論と次軸

4較正のうち registry fallback の回復だけは gold100 で確認できたが、PoT/EU-packet/enum の較正は統合時に目的を達成しなかった。再試行はせず（本 issue の gold100 は一度だけ）、統合軸を **rejected** として ledger に記録する。次回は gold100 全量ではなく、まず MATCH→ABSTAIN 14件・ABSTAIN→WRONG 9件と idx70 を focused serve-path trace で分離し、フラグ間相互作用と配線到達を確認してから新しい根拠がある場合のみ別 issue で再評価する。
