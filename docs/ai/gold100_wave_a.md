# Wave A 決定論パイプライン統合後 Gold-100 実測 — SOT-2610

- 実行日時: 2026-08-10T12:33:12–12:41:02Z（本 issue で 1 回のみ、約7分50秒）
- main HEAD: `c943590`（A1 PR#125 / A2 PR#126 / A3 PR#127 / A4 PR#128 を含む）
- 公式モデル: `gemini-3.6-flash`（`VERTEX_LOCATION=global`）、judge=`codex`
- コマンド: `bash scripts/sot2610_gold100.sh`（内部で `.venv/bin/python -m scoring.gold_offline --run --workers 8 --out artifacts/gold100_sot2610_waveA.json`）
- 構成: champion（r1 回答増フラグ群 + `RAG_FORMAT_EVENTS=1`）+ `RAG_DET_PIPELINE_ROUTER=1`
- 生レポート: `artifacts/gold100_sot2610_waveA.json`（gitignore）／履歴: `docs/gold_offline_history.jsonl`
- 事前処理: コーパス不変かつ Wave A は flag-gated serve path の変更のみのため、champion と同一の index/evidence/canonical/profile/document-registry/structure store を再利用した。

## 結果: net 40（関門2 PASS）

| 指標 | champion r1+FORMAT_EVENTS | r2 all-27 | SOT-2601 F | **Wave A** | Wave A vs champion |
| --- | ---: | ---: | ---: | ---: | ---: |
| match | 39 | 31 | 26 | **47** | **+8** |
| abstain | 54 | 60 | 60 | **46** | **−8** |
| wrong | 7 | 9 | 14 | **7** | **±0** |
| **net (match−wrong)** | **32** | **22** | **12** | **40** | **+8** |
| cost | $12.16 | $8.99 | $8.18 | **$13.64** | +$1.48 |

狙いの `net > 32` を達成し、match は増加、wrong は非増加だった。Wave A 候補は gold100 proxy 上で champion を更新する。

## champion → Wave A 一問遷移

| 遷移 | 件数 | idx |
| --- | ---: | --- |
| MATCH→MATCH | 34 | 0,2,3,4,10,11,13,18,19,20,23,25,26,31,35,41,43,44,46,49,51,54,58,59,60,62,66,71,72,81,86,89,91,94 |
| MATCH→ABSTAIN | 3 | 77,93,96 |
| MATCH→WRONG | 2 | 52,84 |
| ABSTAIN→MATCH | **12** | 7,16,29,30,33,42,68,69,70,75,85,90 |
| ABSTAIN→ABSTAIN | 39 | 1,5,6,8,9,14,15,17,22,24,32,34,36,37,38,39,40,45,47,48,50,55,56,57,61,63,64,67,73,76,79,82,83,87,92,95,97,98,99 |
| ABSTAIN→WRONG | 3 | 27,78,88 |
| WRONG→MATCH | 1 | 21 |
| WRONG→ABSTAIN | 4 | 12,28,53,80 |
| WRONG→WRONG | 2 | 65,74 |

純増は ABSTAIN→MATCH 12 と WRONG→MATCH 1 が牽引した。一方、旧正答5件の後退と新規誤答3件があり、総 wrong は旧誤答5件の解消と相殺され 7 のまま。後続 Wave では idx27/52/78/84/88 の focused trace を優先する。

## 型別効果（n / match / abstain / wrong）

| 型 | champion | **Wave A** | match差 | wrong差 | 判定 |
| --- | ---: | ---: | ---: | ---: | --- |
| derived_calculation | 32 / 8 / 21 / 3 | **32 / 11 / 19 / 2** | **+3** | **−1** | 改善 |
| version_diff | 6 / 1 / 4 / 1 | **6 / 1 / 4 / 1** | ±0 | ±0 | 非劣化 |
| enum_set | 9 / 3 / 6 / 0 | **9 / 3 / 6 / 0** | ±0 | ±0 | 非劣化 |
| document_extract | 24 / 12 / 11 / 1 | **24 / 13 / 9 / 2** | +1 | +1 | mixed |
| fact_lookup | 26 / 14 / 10 / 2 | **26 / 18 / 6 / 2** | **+4** | ±0 | 改善 |
| highlight_set | 1 / 1 / 0 / 0 | **1 / 1 / 0 / 0** | ±0 | ±0 | 非劣化 |
| config_hyperparam / data_shape | 各 0 / 1 / 0 | **各 0 / 1 / 0** | ±0 | ±0 | 非劣化 |

Wave A の主要対象型（derived/version/enum）の合算は match 12→15、wrong 4→3 で、目的の **match↑ / wrong↓** を達成した。A1 version_diff と A3 enum は gold100 全体では横ばい、A2 derived が主な型内改善を担った。A4 の既知 focused 対象 idx10/33/44/58 は 4/4 MATCH（うち idx33 が ABSTAIN→MATCH）。fact_lookup の +4 は全体 net のもう一つの牽引源である。型集計は確率的生成を含む単一実測であり、各 Wave の因果寄与を完全に分離したアブレーションではない。

## 棄権状態コード

| code | champion | **Wave A** | 差分 |
| --- | ---: | ---: | ---: |
| NOT_RETRIEVED | 3 | **0** | −3 |
| RETRIEVED_NOT_PARSED | 1 | **0** | −1 |
| PARSED_AMBIGUOUS | 1 | **0** | −1 |
| EVIDENCE_INCOMPLETE | 2 | **0** | −2 |
| UNANSWERABLE | 16 | **6** | −10 |
| BUDGET_EXHAUSTED | 31 | **39** | +8 |
| SPIN_CUTOFF | 0 | **1** | +1 |

棄権総数は54→46へ減少したが、残存棄権の中心は BUDGET_EXHAUSTED（39/46）へ集中した。次の改善軸は決定論レーン未到達型の探索予算圧縮であり、同じ Wave A 軸の再試行ではない。

## 関門2（SOT-2478）非劣化判定

- match 非劣化: 39→47（+8） **PASS**
- wrong 非増加: 7→7（±0） **PASS**
- net 改善: 32→40（+8） **PASS**
- Wave A 対象型合算: match 12→15、wrong 4→3 **PASS**

したがって **関門2 PASS / promoted**。ただし local gold100 と実 LB の相関は弱いため、最終採用判断の一次 KPI は引き続き leaderboard rank とする。

## 結論

Wave A 全4子を含む main で唯一の gold100 を完走し、**match 47 / abstain 46 / wrong 7 / net 40** を確認した。r1（net32）、r2（net22）、SOT-2601 F（net12）の全てを上回り、wrong 非増加も満たしたため Wave A 軸を ledger へ promoted として記録する。
