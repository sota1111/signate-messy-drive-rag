# Sonnet gold100 改善サイクル第11次 — 失敗調査と方針 (SOT-2708)

- 入力: `artifacts/gold100_sonnet_cycle10.json` / `artifacts/predictions_test_investigator.details.jsonl`
  （cycle10 run の per-idx details）/ `docs/ai/sonnet_gold_history.jsonl` / SOT-2701 申し送りコメント
- cycle10 実測: **96 match / 1 abstain / 3 wrong / net 93**（official:false, Gemini $0）
- 本サイクル担当: 親 = Fable（分析・分解・統合）、子 = opus（並列実装）

## 1. 残余失敗の per-idx 全数分類（abstain 1 + wrong 3）

| idx | archetype | state | 分類 | 帰属（証拠つき） | 判定 |
| --- | --- | --- | --- | --- | --- |
| 98 | fact_lookup | BUDGET_EXHAUSTED | abstain | TM 案件 RATE 変更日。SOT-2690 全数調査で corpus 証拠不在を実証済み（cycle9 再確認でも新証拠なし）。 | **closed 維持**（honest abstain。無理な回答化はしない） |
| 1 | version_diff | Incorrect | wrong | deterministic 経路（diff_store direct-commit, confidence 1.0）で回答済みだが、judge が gold『中間実測値と最終値』の語彙まで要求。SOT-2706 で direct-commit 昇格まで実施し、gold 語彙はスライド非記載の解釈的語彙で質問非依存導出不能と実証。 | **closed 維持**（意味等価・honest 残余） |
| 27 | derived_calculation | Incorrect | wrong | スコープ対象外件数 5 vs gold 7。cycle6 K6 調査で gold ハードコード以外の経路なしと実証済み。 | **closed 維持** |
| 77 | document_extract | Incorrect | wrong | **再オープン（証拠実在を実ファイルで確認）**。下記 §2。 | **本サイクル primary** |

過去サイクル変更への回帰帰属: なし（cycle10 の 3 wrong は全て cycle9 以前から継続。新規レーンの誤発火痕跡なし
— interventions テレメトリで format_strip_paren fired=1/25, plan_fanout は従来どおり LLM 経路 26 件のみ）。

## 2. idx77 の真因（機械確認済み — 唯一の net 増加軸）

質問: 蒼泉会ひがし丘 train.xlsx Sheet2 の黄色ハイライト数値の抽出条件と集計内容。
gold: `children=3、smoker=no ... 合計 / age`。cycle10 回答: `children=0、行ラベル=no ... 合計 / age`（wrong）。

実ファイル `data/share_drive/プロジェクト/医療法人社団 蒼泉会 ひがし丘総合病院/03.データ/train.xlsx` Sheet2 を openpyxl で直接確認:

- 黄色セルは **E14**（FFFFFF00, 値 4674, 列ヘッダ E3=`合計 / age`）。
- D:F 列は**入れ子ピボット**: D3=`行ラベル`、D 列に外側グループ値（children: D4=0, D7=1, D10=2, **D13=3**, D16=4, D19=5）と
  その直下に内側値（`no`/`yes`）が交互に並ぶ。E14 の行は **D13=3 のグループ配下の D14=`no`** ⇒ 正解文脈は
  children=3 ∧ smoker=no。
- 現行 `RAG_HIGHLIGHT_PIVOT_CONTEXT` ビルダー（SOT-2704, train シートの行3ラベル形式に対応）は Sheet2 の入れ子形式で
  (a) 汎用ヘッダ `行ラベル` をそのまま次元名に採用し、(b) 外側グループ値のキャリーダウンをせず children=0（先頭グループ）
  を拾った。row_context=`[{children:0},{行ラベル:no}]` が details に記録されている。
- **質問非依存の正しい再構成**: ①同一ラベル列内の入れ子は「直近上方の外側グループ値」をキャリーダウン、
  ②内側値の次元名は元データ（train シート）の**値域照合**で解決（{no,yes} ⊆ smoker 列の値域。数値グループ {0..5} ⊆ children
  列の値域 — 隣接ピボット A3=`children` とも整合）、③`行ラベル`（総称ヘッダ）は次元名として採用禁止。
  gold 参照は一切不要 = ハードコードなし。

## 3. 到達性分析: LLM 経路 26 idx（churn リスク）と決定論昇格クラスタ

cycle10 details の `model` フィールドで機械分類: **74 idx が deterministic 経路**（うち match 72、wrong idx1/77）、
**26 idx が sonnet(claude-mcp) LLM 経路**（match 24 / wrong idx27 / abstain idx98）。

歴代サイクルの新規失敗は全て LLM 経路の分散チャーン（cycle8 実測: 新規失敗 5 件全てチャーン、SOT-2689）。
現 net93 を守り netの上振れを安定化するには **LLM 経路 match の決定論昇格**が最有効（cycle8 申し送り
「次はLLM経路lookupの決定論昇格」の実行）。24 match をストア族でクラスタ化:

| クラスタ | idx | 内容 / 既存資産 |
| --- | --- | --- |
| xlsx スケジュール/プラン | 2, 41, 75, 89, 90 | 行ハイライト列挙（visual store 既存・SOT-2653）、担当者タスク数（plan_coverage 既存・SOT-2692）、週次配置/フェーズ末尾/バッファ合計（schedule 資産・SOT-2680）。**ストアは在るが serve 自動発火が無い**（SOT-2698 教訓の再適用: 自動レーン追加だけで安価回収） |
| 複合書式・コメント | 3, 11, 49 | 太字 run 列挙（契約書・日付除外）、B∧U∧I run（青嶺報告資料 — SOT-2703 ストアの対象文書拡張）、docx コメントアンカー抽出（**新規: docx comments XML は構造化データ = 質問非依存全数抽出可**） |
| version_diff 到達 | 9, 14 | 「該当なし」ペア（実質変更ゼロの決定論判定）と v1→v3 列名変更 summary（diff_store 既存・SOT-2646/2700/2706 の direct-commit 対象拡大） |
| マスタ結合 | 13, 43, 46 | 最多案件人物→内線 / CT 甲側主担当 / 最高着手金案件→ES内線。contact_master（SOT-2707）× case_finance（既存）の全数結合で導出可 |
| （非対象・次サイクル候補） | 8, 25, 29, 30, 31, 35, 84, 85, 87, 93 | image OCR / nb chart / metrics / 派生計算系。既存ストア資産あり。今回はスコープ外として明示的に残す |

## 4. 本サイクルの方針と子分解（5 件 / primary 合計 14 idx）

第1目標 abstain→0 は idx98 closed（証拠不在実証済み）につき**実質達成**。本サイクルは
(1) 唯一の open wrong idx77 の決定論修正 = net +1、(2) LLM 経路 13 idx の決定論昇格 = チャーン抑止、で構成する。

| 子 | 対象 idx | 内容 |
| --- | --- | --- |
| C1 | 77 | Sheet2 入れ子ピボットの row_context 再構成修正（キャリーダウン＋値域次元名解決） |
| C2 | 2, 41, 75, 89, 90 | xlsx スケジュール/プラン既存ストアの自動発火レーン化 |
| C3 | 3, 11, 49 | 複合書式ストアの対象拡張＋docx コメントアンカーストア新設 |
| C4 | 9, 14 | version_diff direct-commit の対象拡大（該当なし判定含む） |
| C5 | 13, 43, 46 | contact/case master 全数結合 lookup レーン |

ガードレール（全子共通・継承）: gold 値ハードコード禁止 / 事前計算は質問を見ない網羅計算のみ /
serve path 変更はフラグゲート既定 OFF・OFF 時 byte-identical / focused は `run_focused_gate.py --dev` ＋
Sonnet 番兵 `scripts/sonnet_sentinels.json` 10/10 ＋ `RAG_CLAUDE_MCP_RESUME=0`（SOT-2664 replay 罠）/
Gemini 実行禁止（`RAG_FORBID_GEMINI=1`）/ 子は gold100 全量を回さない / 公式レーン不可侵。

理論上限（全子成功時）: match 97 / abstain 1 / wrong 2 = **net 95**（＋昇格 13 idx のチャーン面が deterministic 化）。

## 5. 次サイクルへの持ち越し候補

- 残 LLM 経路 10 idx（8, 25, 29, 30, 31, 35, 84, 85, 87, 93）の決定論昇格（image OCR / nb chart / metrics 系は
  既存ストアの自動発火のみで届く可能性が高い）。
- codex judge の 3 回多数決化（判定確率ノイズ対策 — SOT-2686 で idx21 揺れを A/B 実証済み。測定系変更なので
  単独サイクルで独立に検証すべき）。
- closed 軸は再訪しない: idx1（judge 語彙）/ idx27（gold ハードコードのみ）/ idx98（証拠不在）。
