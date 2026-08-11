# Cycle 3 統合 Gold-100 実測 — SOT-2636

- 実行日時: 2026-08-11T04:54:59–05:01:57Z（本 issue で1回のみ）
- HEAD: `60b5658`、公式モデル: `gemini-3.6-flash` (`global`)、judge=`codex`
- 構成: Wave A + B1 champion、`RAG_G2_LOOKUP_PORT=1`、`RAG_DERIVED_FORMAT_CONTRACTS=1`
- 除外: G1（改善非実証）、EU gate（効用非識別）、cycle2 4 flags（全DROP）、G3 chart（非実現可能）
- manifest preflight: PASS。生結果: `artifacts/gold100_cycle3.json`

## 結果: net 35（関門2 FAIL、champion維持）

| 指標 | Wave A champion | Cycle 3 | 差 |
| --- | ---: | ---: | ---: |
| match | 47 | **48** | +1 |
| abstain | 46 | **39** | -7 |
| wrong | 7 | **13** | **+6** |
| net | **40** | **35** | **-5** |
| cost | $13.64 | $12.76 | -$0.88 |

match は非劣化だが、`wrong <= 7` と `net > 40` をともに満たさない。Cycle 3 構成は昇格せず、Wave A champion (net 40) を維持する。

## Champion からの一問遷移

| 遷移 | 件数 | idx |
| --- | ---: | --- |
| MATCH→MATCH | 38 | 0,3,4,7,10,11,13,16,18,19,20,21,25,26,29,30,31,33,41,43,44,46,49,51,54,58,59,60,68,69,70,71,72,81,86,89,90,91 |
| MATCH→ABSTAIN | 5 | 2,23,35,42,66 |
| MATCH→WRONG | 4 | 62,75,85,94 |
| ABSTAIN→MATCH | 8 | 9,15,28,53,56,80,92,96 |
| ABSTAIN→ABSTAIN | 34 | 1,5,6,12,17,22,24,32,34,36,37,38,39,40,45,48,50,55,57,61,63,67,73,76,77,79,82,83,87,93,95,97,98,99 |
| ABSTAIN→WRONG | 4 | 8,14,47,64 |
| WRONG→MATCH | 2 | 52,74 |
| WRONG→WRONG | 5 | 27,65,78,84,88 |

新規回収8件に対して、新規wrong 8件と既存matchの棄権化5件が発生した。wrong増は G2 発火行ではなく、非発火の LLM fallback / 既存経路に局在するため、単発実測だけから G2 の直接因果とは断定しない。一方、統合構成として precision gate を通らないことは明確である。

## 型別（match / abstain / wrong）

| 型 | Champion | Cycle 3 | 所見 |
| --- | ---: | ---: | --- |
| derived_calculation | 11 / 19 / 2 | **14 / 13 / 5** | 回収+3だがwrong+3 |
| document_extract | 13 / 9 / 2 | **15 / 6 / 3** | 回収+2、wrong+1 |
| fact_lookup | 18 / 6 / 2 | **14 / 8 / 4** | **match-4、wrong+2** |
| version_diff | 1 / 4 / 1 | **2 / 3 / 1** | idx74回収、非劣化 |
| enum_set | 3 / 6 / 0 | 3 / 6 / 0 | 同値 |
| highlight_set | 1 / 0 / 0 | **0 / 1 / 0** | idx2棄権化 |
| config / data_shape | 各0 / 1 / 0 | 各0 / 1 / 0 | 同値 |

主回帰源は fact_lookup の precision/到達不安定性と、derived の積極回答化である。次軸は全量再試行ではなく、(1) MATCH→WRONG の 62/75/85/94 と ABSTAIN→WRONG の 8/14/47/64 を focused precision gate に固定、(2) fact_lookup の route 安定化、(3) derived format contracts の「意味保存だが冗長化」境界（idx8/64/65/75/84）を fail-closed にする。

## 移植11問の個別 verdict

| Group | idx | verdict |
| --- | --- | --- |
| G1 | 15 / 80 / 17 | MATCH / MATCH / ABSTAIN |
| G2 | 5 / 53 / 96 / 36 / 79 | ABSTAIN / MATCH / MATCH / ABSTAIN / ABSTAIN |
| G3 | 74 / 84 / 56 | MATCH / WRONG / MATCH |

11問中6問がMATCH、4問がABSTAIN、1問がWRONG。Champion比の新規回収は idx15/53/56/80/96 と、既存wrongから回収した idx74。G2 flag の直接発火は idx5/36/53/79/96 の5件で、MATCH 2（53/96）、ABSTAIN 3（5/36/79）、WRONG 0。

## 介入テレメトリ

| intervention | ON records | fired | MATCH | ABSTAIN | WRONG |
| --- | ---: | ---: | ---: | ---: | ---: |
| g2_lookup_port | 88 | 5 | 2 | 3 | 0 |

`interventions` は該当フラグが有効なレコードにのみ存在する SOT-2629 schema で集計した。EU gate は較正で `utility_discriminates=False` のため構成から除外され、decision 記録は0件（OFF）である。その他のDROP/OFFフラグも介入キーなし。

## 棄権状態と決定論到達

| code | Champion | Cycle 3 | 差 |
| --- | ---: | ---: | ---: |
| BUDGET_EXHAUSTED | 39 | **34** | -5 |
| UNANSWERABLE | 6 | **4** | -2 |
| SPIN_CUTOFF | 1 | **1** | 0 |

BUDGET_EXHAUSTED は5件減ったが、棄権39件中34件（87%）で依然支配的。決定論契約の直接回答は16/100で、Wave B の13/100から+3（G2回収を含む）だが、fallback側のwrong増を相殺できなかった。

## 関門2と結論

- match 非劣化 47→48: PASS
- wrong 非増加 7→13: **FAIL**
- net 改善 40→35: **FAIL**
- 昇格条件 `net > 40 && match >= 47 && wrong <= 7`: **FAIL**

Cycle 3統合候補は rejected、champion更新なし。本 issue ではLB提出しない。一次KPIは引き続き leaderboard rankであり、次サイクルは上記8件のprecision回帰をfocusedで固定してから、新しい全量軸を設計する。
