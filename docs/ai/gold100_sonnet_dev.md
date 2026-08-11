# Sonnet dev gold100 実測 — SOT-2628

- **非公式 (`official:false`)**。champion 昇格・公式非回帰判定には使用しない。
- 実行: 2026-08-11T00:40:58Z–02:16:48Z（**1時間35分50秒**）、`--workers 1`、judge=codex。
- 構成: champion Wave A + B1（B2 OFF）、investigator backend=`claude-mcp`、model=`sonnet`。
- 完走: 100/100。usage-limit 中断 **0回**、重複実行 **0回**。
- コスト: **$0.0000**。回答経路は `sonnet(claude-mcp)` 87件 + 決定論直答13件、vision 利用は0件で、Gemini課金フォールバックなし。
- 生結果: `artifacts/gold100_sonnet_dev.json`、resume: `artifacts/gold100_sonnet_dev_resume.jsonl`。

## 結果

| 指標 | flash 3.6 champion | Sonnet dev | 差分 |
|---|---:|---:|---:|
| match | 47 | **46** | -1 |
| abstain | 46 | **26** | -20 |
| wrong | 7 | **28** | +21 |
| net (match-wrong) | **40** | **18** | -22 |
| cost | $13.6369 | **$0.0000** | -$13.6369 |

Sonnet は棄権を20件減らした一方、wrong が21件増え、net は40→18へ低下した。到達性探索には有用だが、現状の judge 厳密性では全量の公式代替にはできない。

## 一問遷移

| 遷移 | 件数 | idx |
|---|---:|---|
| MATCH→MATCH | 35 | 2,3,7,10,11,13,16,18,19,21,23,25,26,30,33,35,43,44,46,49,51,54,58,60,66,69,70,71,75,81,86,89,90,91,94 |
| MATCH→ABSTAIN | 0 | — |
| MATCH→WRONG | 12 | 0,4,20,29,31,41,42,59,62,68,72,85 |
| ABSTAIN→MATCH | 9 | 5,15,17,36,53,56,79,80,96 |
| ABSTAIN→ABSTAIN | 24 | 12,14,22,28,32,37,38,39,40,45,47,48,50,55,57,63,67,73,82,83,87,97,98,99 |
| ABSTAIN→WRONG | 13 | 1,6,8,9,24,34,61,64,76,77,92,93,95 |
| WRONG→MATCH | 2 | 74,84 |
| WRONG→ABSTAIN | 2 | 52,65 |
| WRONG→WRONG | 3 | 27,78,88 |

## Sonnet で到達し flash 3.6 で非到達だった問（改善 candidate）

- idx5: 青潮モビリティサービスの最終報告にて最良モデルとしているモデルのパラメータであるmax_depthはいくらに設定されていますか。（flash=ABSTAIN → Sonnet=MATCH）
- idx15: 東都人材プラットフォームのtrain.xlsxにおいて、Sheet1の黄色にハイライトされたセルの抽出条件と集計内容を答えてください。（flash=ABSTAIN → Sonnet=MATCH）
- idx17: AYMのMMにおいて、黄色ハイライトかつREDになっている数値を対象に、最初のMMから最後のMMまでの上昇率を計算してください。上昇率は （最後の値 - 最初の値） / 最初の値 × 100 で求め、小数第2位まで答えてください。（flash=ABSTAIN → Sonnet=MATCH）
- idx36: 恒一会 かえで総合病院案件において、中間報告時点のF1スコア実測値と最終報告時点のF1スコア実測値の差を絶対値で答えてください。（flash=ABSTAIN → Sonnet=MATCH）
- idx53: TOTOのFR書にて記載のある選択特徴量のうち、ENG-FTはいくつありますか。（flash=ABSTAIN → Sonnet=MATCH）
- idx56: 蒼泉会 ひがし丘総合病院の01_eda.ipynbにおける目的変数分析の可視化において、y軸に実際に表示されている目盛りの最大値は何ですか。（flash=ABSTAIN → Sonnet=MATCH）
- idx79: 恒一会 かえで総合病院の計画フォルダ内において、データアステル側の担当者のうち、1タスク当たりの想定工数（想定工数 ÷ 担当タスク数）が最も大きい人のフルネームと、その1タスク当たりの想定工数を小数第2位で答えてください。ファイルに鍵がかかっている場合は社内管理を確認してください。（flash=ABSTAIN → Sonnet=MATCH）
- idx80: 東都人材プラットフォームのtrain.xlsxにおいて、Sheet2の黄色にハイライトされたセルの抽出条件と集計内容を答えてください。（flash=ABSTAIN → Sonnet=MATCH）
- idx96: 青葉与信マネジメントのチェックポイント2として設定されている内容に関連するタスクIDを教えてください。（flash=ABSTAIN → Sonnet=MATCH）
- idx74: 青葉与信マネジメントの提案書_v1.pptxから提案書_v2.pptxに修正されたもののうち、案件遂行に関連する変更を挙げてください。（flash=WRONG → Sonnet=MATCH）
- idx84: 東都人材プラットフォームの最終報告書で分析結果が記載されている中で、モデル毎のF1スコアがランキング形式で記載されているページ数を教えてください。（flash=WRONG → Sonnet=MATCH）

これらはモデル差を固定結論にせず、Sonnet の tool trace からプロンプト・契約型ルーティング・ツール誘導を flash 側へ移植できる候補として扱う。

## flash 3.6 では到達し Sonnet で非到達だった問

- idx0: 白峰信用リスク評価の提案書old.pptxから提案書.pptxへの更新内容のうち、案件遂行に関連する実質的な変更を挙げてください。（flash=MATCH → Sonnet=WRONG）
- idx4: 蒼泉会 ひがし丘総合病院の01_eda.ipynbを確認して、目的変数と相関が最も高い数値特徴量を教えてください。（flash=MATCH → Sonnet=WRONG）
- idx20: 東都人材プラットフォームの報告資料_2025-08-18.pdf で、渡辺遥と藤田彩の2人が担当となっている優先タスクを抽出してください。（flash=MATCH → Sonnet=WRONG）
- idx29: 恒一会 かえで総合病院のtrain.xlsx内のTPのヒストグラムで、3番目にカウント数が多いビンの範囲を小数第6位までで答えてください。（flash=MATCH → Sonnet=WRONG）
- idx31: 固定金額契約の中で、分析データ1行あたりの契約金額（税込）が最も高い案件を、主略称と1行あたりの金額で答えてください。1行あたりの金額は円単位で切り上げてください。（flash=MATCH → Sonnet=WRONG）
- idx41: AOBMのPLANにおいて、加藤さんが担当者に含まれるタスクIDはいくつありますか。（flash=MATCH → Sonnet=WRONG）
- idx42: 蒼泉会 ひがし丘総合病院のtrain.xlsxのSheet1において、黄色ハイライトされている数値に対応するデータの抽出条件と集計内容を答えてください。（flash=MATCH → Sonnet=WRONG）
- idx59: 京ソのPP_final.pptxにおいて、この案件にかかる金額の提示がまとまっているのは何ページですか。（flash=MATCH → Sonnet=WRONG）
- idx62: 青葉与信マネジメントの最終報告資料における、モデル比較で上位2件のスコア差を生んでいる設定差分は何ですか。（flash=MATCH → Sonnet=WRONG）
- idx68: 東都人材プラットフォームのデータサイエンス市場の未来予測.pdfにおいて、投資実装係数の計算式が記載されているページの数値情報を式に代入し、投資実装係数を小数で答えてください。（flash=MATCH → Sonnet=WRONG）
- idx72: KSSにおいて、データエンジニアが担当するタスクIDはいくつありますか。（flash=MATCH → Sonnet=WRONG）
- idx85: 青葉バイオメディカル機器の最終報告において、設定されたKPIとして未達成とされている項目を挙げてください。（flash=MATCH → Sonnet=WRONG）

## 運用提案

- **focused dev**: Sonnet は $0 で、flash が棄権した候補の trace 探索やツール誘導の仮説生成に使う。特に上記 ABSTAIN/WRONG→MATCH を優先する。
- **全量 dev**: 約96分/100問かかり、自律ワーカーと usage limit を共有するため、夜間・並列度1・resume前提で限定実施する。今回の中断は0回だった。
- **公式ゲート**: flash 3.6 champion を維持する。Sonnet dev の net18 は official 判定や champion 昇格に使わない。
- **次軸**: Sonnet MATCH / flash non-MATCH の tool trace を抽出し、決定論ルーターまたは flash プロンプトへ移植して focused A/B を行う。Sonnet の積極回答は wrong 増加を伴うため、evidence completeness と出力正規化を同時にゲートする。

## 記録分離

- `docs/gold_offline_history.jsonl` の本 run は `official:false`。
- `docs/ai/experiment_ledger.jsonl` も `official:false` として記録し、公式履歴・champion 系譜から分離する。
