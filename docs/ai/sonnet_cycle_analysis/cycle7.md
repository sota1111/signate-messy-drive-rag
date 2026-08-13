# cycle7 — Sonnet dev gold100 失敗全数分析と改善方針（SOT-2683）

- 入力: `artifacts/gold100_sonnet_cycle6.json`（cycle6 実測: **77 match / 14 abstain / 9 wrong / net68**、cost $0）、
  `artifacts/predictions_test_investigator.details.jsonl`（per-idx テレメトリ/evidence 実文）、
  `docs/ai/sonnet_gold_history.jsonl`、SOT-2676 申し送り、`docs/ai/sonnet_cycle_analysis/cycle6.md`。
- 分析方法: abstain 14 + wrong 9 の **23 件全数**を state code × 契約型 × 欠落証拠（details の evidence 実文・
  ツール列）でクロス分類。帰属はすべて details のテレメトリ実文を根拠にした（推測帰属なし）。
- state code 集計（abstain 14）: BUDGET_EXHAUSTED 10 / SPIN_CUTOFF 2 / UNANSWERABLE 2。

## 1. per-idx 全数分類

### abstain（14）

| idx | 型 | 欠落証拠（details evidence 実文より） | クラスタ |
|---|---|---|---|
| 8 | numeric | データサイエンティスト調査.docx に「Salary.com」はあるが **ML/DE 給与の数値本体（gold 14,744）が本文テキストに存在しない**（機械確認済: document.xml に 14,744/13,222 なし）。埋め込み media は `word/media/image1.emf` のみ → **給与表は EMF 画像内**。cycle6 K1 の「honest abstain」判定と整合 | K1 |
| 50 | numeric | 同上（Salary.com 中央値/上位90% の数値が本文になし、gold 13,222 は image1.emf 内とみられる）。doc_fulltext_search/doc_table_lookup 到達済みでも不在 → テキスト側は網羅済み | K1 |
| 52 | simple_lookup | みなみ野最終報告.pdf は images 16/fonts 9 の混在 PDF で「別契約」が file_grep/FTS 不在 → **該当ページが画像**。serve 中 caption_image は `GeminiForbidden` で正しくブロック | K1 |
| 68 | numeric | 未来予測.pdf の「投資実装係数」が text_search 0 件（FTS 未索引/画像ページ）。read_office は PDF 非対応 | K1 |
| 56 | numeric | ipynb 埋め込み画像（目的変数分析グラフ）の **y軸目盛り最大値** は描画属性で、データ度数（max 1256）からは確定不可。read_chart_values 相当の ipynb 画像読みが存在しない | K2 |
| 66 | simple_lookup | 京橋 01_eda.ipynb の日付分析チャートの最多件数日。ipynb 画像読み手段なし（canonical_route で notebook 特定まで到達済み） | K2 |
| 47 | numeric | 青嶺 train.xlsx: ハイライトセル B22 と回帰係数は取得済み。**B22 の数式がどの行(id)を参照するかのトレース手段がない**（→ その行の YEAR BUILT が gold 1899年） | K3 |
| 83 | numeric | みなみ野 train.xlsx: シート構成確認後に予算切れ。**記載回帰係数 × index=1770 行値の適用**が compute 数手では届かない | K3 |
| 17 | numeric | AYM「MM」の定義・対象ファイル群と **docx の黄色ハイライト×赤字(RED)** 条件の抽出手段がない（ハイライト数値の存在自体は確認済み） | K3(stretch) |
| 36 | simple_lookup | かえで F1 中間 vs 最終の差。gold `0.09619112771492555` は **全精度値=分析成果物由来**（スライドは丸め値 0.733/0.829）。段階（中間=T04 linear/最終=hist_gradient_boosting）×メトリクスの成果物クロス参照が不在 | K4 |
| 62 | fact_lookup | 青葉 最終報告 slide6 の上位2モデルまで到達。スライドに「詳細は leaderboard.csv」→ **leaderboard.csv 上位2行の設定差分（gold n_estimators 500 vs 300）への到達経路がない** | K4 |
| 34 | simple_lookup | 「MINAMINO」が find_files/file_grep/case_filter/action_row_lookup 全て 0 件。**ローマ字別名（みなみ野→MINAMINO）が case エイリアスに無い**。解決後は action_row_store で M01→M02 完了×担当者=伊藤（gold A08、A09）に到達可能 | K4 |
| 98 | simple_lookup | TM=実費精算契約 は用語集で解決済み。**RATE 変更日（gold 2025年7月1日）を示す月次単価時系列/覚書クロス参照が不在** | K4 |
| 53 | numeric | TOTO=東都 は解決済み。「FR書」該当ファイルと「ENG-FT」分類の裏付けが未発見（gold 6） | K4(stretch) |

### wrong（9）

| idx | 型 | 誤り方（evidence 実文より） | クラスタ |
|---|---|---|---|
| 91 | numeric | **決定論レーンの符号バグ（確定・再現可能）**: 質問は「最も強い**負**の相関」だが `derived_coverage_lane` の idx4 レーンが `_HIGHEST_CUE`(最も強い) にマッチして \|r\| 最大 duration(+0.40) を conf=1.0 で確定。gold=campaign（最小 r）。`src/rag/agent/derived_coverage_lane.py` の `_top_corr_feature` は符号修飾語を見ない | K5 |
| 29 | chart_read | **書式のみ**: 値は正しい（bin 494 位置も正しい）。`(6.088138, 6.288138]` vs gold `6.088138 ~ 6.288138` — 区間記法→チルダ形式の naturalization 欠落 | K5 |
| 78 | simple_lookup | **回答合成契約ミス**: evidence には「ACTH/200時間の特別規定なし＋一般規定 6.1〜6.3 条（25,000円/時・月次精算…）」が完全に取れているのに「該当なし」とだけ回答。gold は「特別規定なし＋適用される一般規定の内容」 | K5 |
| 73 | full_enumeration | config（one_hot, limit=100, exclude≥limit）到達済みで conf=0.15 のまま「わからない」。**カテゴリ列ごとの nunique と閾値の突合（→gold Gender）**まで届かず。派生ストアのカーディナリティ網羅＋enum 決定論で回収可能 | K5(stretch) |
| 1 | version_diff | 変更の同定は gold と同一（比較表削除→1行要約化）。**過剰詳細な言い回し**（T04 linear/数値列挙）で judge 不一致とみられる | K6(未着手・繰越) |
| 22 | version_diff | 同定は gold と実質同一（class 列の統計量追加）。**語彙差**（「セル8(埋め込み画像)の統計表」vs「記述統計（基本統計量）の表」） | K6(繰越) |
| 9 | document_extract | diff store が「業務提言スライドの2枚分割」を SUBSTANTIVE と分類したが gold は「該当なし」→ **見出し分割＝内容不変を cosmetic に分類する規則の欠落** | K6(繰越) |
| 14 | version_diff | v1→v3 で「該当なし」と回答、gold は列名アンダースコア化。**非隣接版の列名レベル差分 alignment**（cycle6 K5 でも既知の別軸） | K6(繰越) |
| 27 | numeric | **軸クローズ（証拠つき）**: `cycle6_k6_idx27_scope_investigation.md` が全19スライド走査で「スコープ対象外の canonical な7項目列挙は文書内に存在しない」ことを実証。安全な回収経路は gold ハードコードのみ＝禁止 → **現状維持** | closed |

## 2. cycle6 変更への回帰帰属

- wrong 7→9 の増分のうち、cycle6 昇格フラグへの**帰属シグネチャがあるのは idx91 のみ**
  （`fact_layer:numeric` レーン発火が evidence に明記。SOT-2679 の派生カバレッジレーンの符号非対応が真因）。
  これは「レーンの適用条件バグ」であり、ストア値自体は正しい（campaign も with_target に焼かれている）。
- idx9/14/22/1/73/78/29 は cycle5 以前から存在する軸（vdiff 精度・書式・回答契約）で、cycle6 フラグの
  発火痕跡なし。単発揺らぎとの区別: idx91 は決定論（毎回再現）、vdiff 群は cycle5 でも同型の誤りを記録済み。

## 3. cycle7 方針 — 残 abstain 14 を「前処理」で消し込む＋確定バグ修正

第1目標（abstain→0）: 14 件中 **12 件は特定ストア/リーチの不在**が evidence で確定しており、質問非依存の
事前計算で証拠を用意できる。idx8/50/52/68/56/66 は**画像ロック証拠**であり、「前処理に限り Gemini 使用可」
規定を初適用する（build で一度だけ vision 実行→決定論ストアに焼き込み。serve は $0 のまま）。

子issue クラスタ（5件、primary 15 idx）:

| 子 | クラスタ | primary idx | stretch |
|---|---|---|---|
| K1 | scan-PDF/docx 埋め込み画像の build-time OCR/vision 事実ストア | 8, 50, 52, 68 | — |
| K2 | ノートブック描画画像（チャート）の build-time 読み取りストア | 56, 66 | — |
| K3 | xlsx 数式依存トレース＋記載回帰係数の行適用レーン | 47, 83 | 17（docx ハイライト×赤字） |
| K4 | 案件ローマ字別名・段階メトリクス・成果物/財務クロス参照 | 34, 36, 62, 98 | 53（FR書/ENG-FT） |
| K5 | wrong 決定論修正: 相関符号レーン＋bin範囲書式＋「特別規定なし」回答契約 | 91, 29, 78 | 73（one-hot enum） |

繰越（本サイクル非着手・次サイクル候補 K6）: vdiff 精度クラスタ idx1/9/14/22。idx1/22 は同定正解×言い回し
不一致で **judge 3回多数決化（wrong フェーズの施策）と併せて扱うのが適切**。idx9 は分割=cosmetic 規則、
idx14 は非隣接版 alignment。abstain≤5 到達後に着手する。

閉じた軸: idx27（§1 の証拠により現状維持）。予算緩和軸は SOT-2663 の逆流実証により再訪しない。

## 4. ガードレール（全子共通）

- serve path 変更は必ずフラグゲート・既定 OFF。OFF 時 byte-identical。dev 構成でのみ ON。
- Gemini は **build スクリプト内のみ**（K1/K2）。serve 中は `RAG_FORBID_GEMINI=1` が例外化することを
  focused で確認。gold 値ハードコード禁止（事前計算は全案件×全属性の網羅計算のみ）。
- focused 検証は `run_focused_gate.py --dev` ＋ Sonnet 番兵 10 問（`RAG_CLAUDE_MCP_RESUME=0` 必須 —
  resume key は (model, question) のみで replay 罠がある）。子は gold100 全量を回さない。
- 現 77 MATCH を守る: 各子は自分の対象 idx ＋番兵で回帰ゼロを確認してから完了する。
- 公式レーン（flash champion・公式 gold100・LB 提出・SIGNATE CLI）に触れない。全結果 official:false。
