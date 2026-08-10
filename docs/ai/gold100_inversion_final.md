# Wave B 全契約型決定論パイプライン統合後 Gold-100 最終実測 — SOT-2613

- 実行日時: 2026-08-10T13:34:00–13:44:23Z（本 issue で 1 回のみ、約10分23秒）
- main HEAD: `cb22c99`（Wave A PR#125–129、B2 PR#130、B1 PR#131 を含む）
- 公式モデル: `gemini-3.6-flash`（`VERTEX_LOCATION=global`）、judge=`codex`
- コマンド: `bash scripts/sot2613_gold100.sh`（内部で `.venv/bin/python -m scoring.gold_offline --run --workers 8 --out artifacts/gold100_sot2613_waveB.json`）
- 構成: champion（r1 回答増フラグ群 + `RAG_FORMAT_EVENTS=1`）+ `RAG_DET_PIPELINE_ROUTER=1`（A1–A4 + B1–B2）
- 生レポート: `artifacts/gold100_sot2613_waveB.json`（gitignore）／履歴: `docs/gold_offline_history.jsonl`
- 事前処理: コーパス不変かつ B1/B2 は flag-gated serve path の追加のため、Wave A と同一の index/evidence/canonical/profile/document-registry/structure store を再利用した。

## 結果: net 37（関門2 FAIL / Wave B 統合軸 rejected）

| 指標 | r1 | SOT-2601 | Wave A | **Wave A+B** | vs r1 | vs Wave A |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| match | 39 | 26 | 47 | **47** | +8 | ±0 |
| abstain | 54 | 60 | 46 | **43** | −11 | −3 |
| wrong | 7 | 14 | 7 | **10** | +3 | +3 |
| **net (match−wrong)** | **32** | **12** | **40** | **37** | **+5** | **−3** |
| cost | $12.16 | $8.18 | $13.64 | **$13.49** | +$1.33 | −$0.14 |

全反転は r1 を明確に上回ったが、直前 champion である Wave A に対して match は増えず wrong が 3 増え、net は 40→37 に後退した。よって B1+B2 の統合昇格条件（match 非劣化・wrong 非増加・net 非劣化）を満たさない。

## 型別効果（n / match / abstain / wrong）

| 型 | Wave A | **Wave A+B** | match差 | wrong差 | 判定 |
| --- | ---: | ---: | ---: | ---: | --- |
| derived_calculation | 32 / 11 / 19 / 2 | **32 / 9 / 19 / 4** | −2 | +2 | 回帰 |
| document_extract | 24 / 13 / 9 / 2 | **24 / 17 / 6 / 1** | **+4** | **−1** | 改善 |
| fact_lookup | 26 / 18 / 6 / 2 | **26 / 16 / 7 / 3** | −2 | +1 | 回帰 |
| enum_set | 9 / 3 / 6 / 0 | **9 / 3 / 6 / 0** | ±0 | ±0 | 非劣化 |
| version_diff | 6 / 1 / 4 / 1 | **6 / 1 / 3 / 2** | ±0 | +1 | 回帰 |
| highlight_set | 1 / 1 / 0 / 0 | **1 / 1 / 0 / 0** | ±0 | ±0 | 非劣化 |
| config_hyperparam / data_shape | 各 0 / 1 / 0 | **各 0 / 1 / 0** | ±0 | ±0 | 非劣化 |

B1 対象の document_extract は match +4 / wrong −1 と明確に改善した。一方、単一確率実測のため B2 の純因果とは断定できないものの、fact_lookup は match −2 / wrong +1、非対象の derived は match −2 / wrong +2、version_diff は wrong +1 となり、全体 net を 3 点押し下げた。

## 決定論レーンと LLM フォールバック

`predictions_test_investigator.details.jsonl` の `det_pipeline:*` tool call を決定論直答として集計した。

| 経路 | 件数 | match | abstain | wrong | net |
| --- | ---: | ---: | ---: | ---: | ---: |
| 決定論直答 | **13 (13%)** | **12** | 0 | 1 | **11** |
| LLM フォールバック | **87 (87%)** | 35 | 43 | 9 | 26 |

決定論直答 13 件の内訳は chart_read 3/3、full_enumeration 3/3、format_check 2/2、simple_lookup 2/2、spatial 1/1、cross_aggregate 1/1 が正答、version_diff 1 件が誤答（idx74）。決定論直答の精度は 12/13 = 92.3% だが、到達率は 13% に留まり、残り 87% は従来の LLM 探索へフォールバックした。回答維持という要件は満たしたものの、全体の誤答非増加は満たさなかった。

## 棄権状態コード

| code | Wave A | **Wave A+B** | 差分 |
| --- | ---: | ---: | ---: |
| UNANSWERABLE | 6 | **9** | +3 |
| BUDGET_EXHAUSTED | 39 | **32** | −7 |
| SPIN_CUTOFF | 1 | **2** | +1 |
| その他 | 0 | **0** | ±0 |

棄権は 46→43 に減り、BUDGET_EXHAUSTED も 7 減ったが、そのうち 3 件分は wrong 増加に転じたため、単純な coverage 増加を昇格理由にはできない。

## 関門2判定

- r1 比: match 39→47、wrong 7→10、net 32→37。net は改善したが wrong 非増加を満たさない。
- 現 champion (Wave A) 比: match 47→47、wrong 7→10、net 40→37。**FAIL**。
- B1 document_extract: match 13→17、wrong 2→1。局所的には **PASS**。
- B2 fact_lookup: match 18→16、wrong 2→3。単一実測上は **FAIL**。

したがって **関門2 FAIL / 全 Wave A+B 統合軸 rejected**。Wave A (net40) を local champion として維持する。

## 残る誤答と次軸

誤答 10 件は version_diff 2（idx1,74）、derived_calculation 4（idx6,27,64,65）、fact_lookup 3（idx62,75,78）、document_extract 1（idx85）。決定論直答由来は idx74 の 1 件、残り 9 件は LLM フォールバック由来である。

次軸は同じ全量 gold100 の再試行ではなく、(1) idx74 の version_diff deterministic naturalization を focused trace で修正、(2) fact_lookup のフォーマット同値誤判定（idx62/75/78）を judge/normalizer と経路に分離、(3) derived の単位・閾値・冗長表現（idx6/64/65）を deterministic formatter 契約で固定、(4) 13% に留まる決定論到達率を誤答非増加の focused set で拡張する。実 LB rank を一次 KPI とし、Wave A+B をそのまま提出候補へ昇格しない。

## 結論

全 Wave Done 後の main `cb22c99` で唯一の gold100 を完走し、**match 47 / abstain 43 / wrong 10 / net 37** を記録した。r1 net32 は超えたが Wave A net40 から 3 点後退したため、反転全統合の最終ゲートは不合格。B1 の局所効果は維持候補だが、Wave A を champion のまま保持し、上記 focused 軸を次サイクルで検証する。
