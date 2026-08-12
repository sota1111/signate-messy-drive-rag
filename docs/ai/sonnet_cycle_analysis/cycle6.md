# Sonnet gold100 改善サイクル第6次 — 失敗分析と方針（SOT-2676）

- 入力: `artifacts/gold100_sonnet_cycle5.json`（cycle5 統合実測, PR#192, official:false）
  = **match66 / abstain27 / wrong7 / net59**（基底構成 `scripts/sonnet_gold_cycle5.sh`）
- 詳細: `artifacts/predictions_test_investigator.details.jsonl`（2026-08-12 15:39 の cycle5 走行）
- 前回申し送り: SOT-2662 Completion Report（2026-08-12 15:43）
- 分析日: 2026-08-12（Fable, SOT-2676 親フェーズ1）

## 0. 結論（本サイクルの中心仮説）

**cycle5 で「一手前予算切れ」チャーン層は finisher がほぼ回収し終えた。残る abstain 27 の主因は
予算ではなく本物のカバレッジ欠落である。** state code は BUDGET_EXHAUSTED 17 / UNANSWERABLE 7 /
SPIN_CUTOFF 2 / PARSED_AMBIGUOUS 1 だが、evidence 全数読解の結果、per-idx の欠落はほぼすべて
「特定のストア/リーチが存在しない」に帰着する:

1. **文書の全文・表リーチ欠落（7件）** — read_office の長文切り詰め（idx8: 末尾+9989字未取得、
   idx99: ランキング表が出力途中で切れる）、PDF 表の数値非抽出（idx48/68）、pptx/docx 表の
   非索引（idx50/52/53）。
2. **分析成果物 raw ファイル非対応（4件+）** — .py が read_office 非対応・metrics.json が
   find_files 0件・leaderboard.csv 参照不能（idx32/61/62/73、二次 35/36）。
3. **派生メトリクス未網羅（2件+）** — 特徴量×目的変数の相関（idx4）、案件横断の欠損行数
   （idx24 — SOT-2663 で「余剰予算による ad-hoc compute 列挙は誤答を生む」ことが確定済み。
   ストアレーンが唯一の安全な回収経路）。
4. **スケジュール/ID/体制のクロス参照クエリ形欠落（5件）** — ID種別ごとの発行数集計（idx92）、
   氏名→役職ロスター（idx94）、チェックポイント定義→タスクID（idx96）、期間×状態差分×担当
   （idx34）、担当→タスク列挙（idx72）。
5. **version_diff の意味正規化・ペア網羅（wrong 3 + abstain 2）** — リスト追記を「変更」と表現
   （idx95）、置換を削除として過剰詳細化（idx1）、非隣接ペア v1→v3 が diff_store に無い（idx14）。

→ 第6次は前回申し送りどおり「一律予算増ではなく質問非依存カバレッジ拡張」を 4 本のストア子と
1 本の vdiff 正規化子、1 本の書式契約子に分解する。SOT-2663 の負結果（予算緩和 rejected）を
再訪しない。

## 1. wrong 7 の per-idx 分類

| idx | archetype | 症状 | 帰属 | クラスタ |
|---|---|---|---|---|
| 1 | version_diff | 「指標表が削除された」— gold「1行要約に置換」。delete+add の対を置換として畳めていない | diff 正規化（置換 collapse・要約粒度） | K5 |
| 9 | document_extract | gold=該当なし。見出し再編（クイックウィン付記・節追加）を実質変更として回答 | 見出し/構造のみ変更の非実質判定欠落 | K5二次 |
| 22 | version_diff | 「class列が追加(64→65列)」— gold「目的変数classの列の統計量が追加(Attr1〜64同一)」。実質同旨の near-miss | 主語・表現正規化（判定揺れ域） | K5二次 |
| 27 | derived_calculation | 5 と回答、gold=7（スコープ対象外の数え上げ）。conf=0.6 | 真の値誤り — 数え上げ根拠の証拠調査が先 | K6 |
| 78 | fact_lookup | 内容同旨だが冗長・追加主張（見込金額170時間等）を付加 | 回答範囲限定契約が未達（SOT-2666 の既知残） | K6二次 |
| 79 | fact_lookup | 「池田 直哉、7.00時間/タスク」— gold「池田 直哉、7.00」。単位サフィックス付加のみ | 小数指定問への単位 strip 契約欠落 | K6 |
| 95 | version_diff | 「渡辺 遥→渡辺 遥/小林 直樹に変更」— gold「小林 直樹を追加」。old⊂new の追記を変更と表現 | diff 正規化（リスト追記→「追加」） | K5 |

## 2. abstain 27 の per-idx 分類

state code: BUDGET_EXHAUSTED 17 / UNANSWERABLE 7 / SPIN_CUTOFF 2 / PARSED_AMBIGUOUS 1。
ただし evidence 本文ベースの真因は下表のとおりで、「予算があと数手あれば解けた」型は少数。
凡例:【表/全文】=文書リーチ欠落 /【raw】=コード・成果物ファイル非対応 /【派生】=派生メトリクス
未網羅 /【xref】=クロス参照クエリ形欠落 /【PARK】=対象外（無理回答化しない）。

| idx | archetype | 欠落証拠・状況（evidence 実文より） | クラスタ |
|---|---|---|---|
| 4 | derived_calc | train.xlsx 列構成まで特定済み。数値特徴量×charges の相関 compute 前に予算切れ（compute は finisher 除外=設計どおり）。相関は質問非依存に事前計算可能 | K3 |
| 8 | derived_calc | docx 冒頭~8000字のみ取得、ML/DE 給与比較は末尾+9989字側。grep 不ヒット | K1 |
| 14 | version_diff | v1→v3 非隣接ペアが diff_store に無い。v1→v2 は取得済み、v2→v3 で予算切れ | K5 |
| 17 | derived_calc | 「AYM の MM」ファイル自体を特定できず（MM=月次資料エイリアス解決欠落） | K1二次 |
| 24 | data_shape | 9案件×欠損行数。03.データ/04.分析の2重 train.csv で ambiguous×予算切れ。**SOT-2663 で余剰予算 ad-hoc 列挙→誤答『白峰』の逆流実証済み** — ストアのみが安全経路 | K3 |
| 32 | enum_set | metrics.json find_files 0件・features.py read_office 非対応 | K2 |
| 34 | document_extract | MINAMINO 解決済みも AI×M01/M02状態×担当伊藤の行に到達不能 | K4二次 |
| 35 | derived_calc | 京橋最終報告 pptx へ到達不能（find_files 0件・report_attr_lookup metrics 空） | K2二次 |
| 36 | fact_lookup | 「中間報告」文書不発見。中間vs最終 F1 の案件横断参照 | K2二次 |
| 47 | derived_calc | B22=黄ハイライト回帰予測誤差セルまで特定、参照先行（=建設年の根拠）の数式解決前に予算切れ | K3二次 |
| 48 | fact_lookup | PDF page5 の税率表見出しは発見済み、表内数値がどのツールでも非抽出 | K1二次 |
| 50 | derived_calc | idx8 と同一 docx。Salary.com の90%タイル/中央値表に到達不能 | K1 |
| 52 | document_extract | 「別契約」全 grep 0 ヒット（表/注記内の可能性）。gold=監視ダッシュボード構築 | K1二次 |
| 53 | derived_calc | TOTO FR書「選択特徴量(14変数)」発見済み、ENG-FT タグ内訳表に到達不能 | K1二次 |
| 56 | derived_calc | ipynb 出力チャートのy軸目盛り — SOT-2633 で非成立確定 | PARK |
| 61 | config_hyperparam | modeling.py 非対応で n_estimators 等のコード上適用値を確認不能 | K2 |
| 62 | fact_lookup | 上位2件の設定差分は leaderboard.csv 参照が必要と特定済み、csv 到達不能 | K2 |
| 68 | derived_calc | 「投資実装係数」が全文索引に不在（PDF 本文/式の非索引） | K1二次 |
| 71 | document_extract | AOMINE の会議録ファイル自体が find_files/search で不発見。太字∧下線∧イタリックは font_emphasis で抽出可能なのに locator 段で失敗 | K1二次 |
| 72 | derived_calc | DE=斎藤悠斗特定済み・担当者列ユニーク値確認済み、タスクID絞り込み compute 前に予算切れ | K4二次 |
| 73 | enum_set | One-Hot 閾値がコード側、カテゴリ列ユニーク数が train 側 — 両方 raw リーチ欠落 | K2二次 |
| 83 | derived_calc | metric_lookup の係数は「ストアが自前 OLS した係数」で出典文書の記載係数ではないと自己申告。記載係数の所在（train.xlsx 内回帰出力ブロック）＋index=1770 行に未到達 | K3二次 |
| 92 | derived_calc | スケジュール.xlsx を canonical 特定済み、ID種別×件数の集計前に予算切れ。ID マスタに集計クエリ形が無い | K4 |
| 94 | document_extract | MS3タスク=T07/T08/T09 まで特定済み。氏名→役職（ビジネスアナリスト）の案件別ロスターが無い | K4 |
| 96 | document_extract | 「チェックポイント2」全 grep 0 ヒット。CP定義→タスクIDの派生クロス参照が必要。gold=T05〜T08 | K4 |
| 98 | fact_lookup | TM案件6件特定済み。契約書版差分×RATE 変更の記述に未到達（diff_store の契約書網羅） | K5二次 |
| 99 | derived_calc | 死亡率ランキング表の存在確認済み、read_office 長文切り詰めで表本体が取得不能 | K1 |

チャーン確認: cycle5 の MATCH→ABSTAIN 新規転落は idx4/35/47/53/62/71/94/98 等を含むが、いずれも
evidence 上はカバレッジ欠落が真因で、cycle4.5→5 で導入したフラグ（finisher/vdiff subject/exact-label/
heading-page）への帰属シグネチャは無し（finisher は対象特定済み単一文書リードのみ+1手で、compute 系
idx4/47/72/92 は設計上対象外）。単発揺らぎとの区別は focused の OFF-control で子側が確認する。

## 3. 子issue クラスタ（6件・一次 focused 対象: 重複なし15 idx）

- **K1 — 文書テーブル/全文チャンクストア（表・長文リーチの質問非依存網羅）**
  一次: idx8/50/99。二次: 17/48/52/53/68/71。
  実装: 全 docx/pptx/pdf を対象に (a) 全表→行列保存の table store（ファイル/ページ/スライド出自付き）、
  (b) 切り詰めなし全文チャンク store（オフセット指定 read / FTS 索引）。lookup ツール配線。
  エイリアス最小辞書（MM=月次モニタリング等）と会議録 locator を二次で。
- **K2 — 分析成果物 raw ファイルストア（.py/.json/.csv リーチ）**
  一次: idx32/61/62。二次: 35/36/73。
  実装: metrics.json / leaderboard.csv / *.py / 設定ファイルの質問非依存 ingest＋read/grep lookup、
  config_hyperparam のコード適用値解決（コンストラクタ引数＋コード上デフォルト）を案件別に事前計算。
  注意: RAW_FILE_TOOLS は block-list（SOT-2660）— serve の生ファイル直読とストア経由を混同しない。
- **K3 — 派生メトリクス網羅拡張（相関・欠損行数・記載係数）**
  一次: idx4/24。二次: 47/83。
  実装: 全 train 表×全数値特徴量×目的変数の相関、全案件×canonical train 表の欠損行数
  （canonical マニフェストで 03.データ/04.分析 の二重を解消）、xlsx 記載回帰ブロックの係数＋
  全行予測値。**idx24 は SOT-2663 の逆流実証があるため、ストア到達以外での回答化を禁止**。
- **K4 — スケジュール/ID/体制クロス参照のクエリ形拡張**
  一次: idx92/94/96。二次: 34/72。
  実装: ID マスタに種別×件数集計、案件別 氏名→役職ロスター（提案書/計画書の体制記載から）、
  チェックポイント/CP 定義→タスクID 派生クロス参照、期間×状態差分×担当クエリ形。
- **K5 — version_diff 意味正規化＋非隣接ペア網羅**
  一次: wrong idx1/95、abstain idx14。二次: 9/22/98。
  実装: (a) old⊂new のリスト追記→「…を追加」正規化、(b) delete+add 同域→「置換」畳み込みと
  要約粒度、(c) diff_store への非隣接版ペア（v1→v3 等）事前計算、(d) 見出し/構造のみ変更の
  非実質判定（idx9 の 該当なし化）、(e) 契約書ペアの RATE/単価差分と適用日抽出（98）。
- **K6 — 書式契約（単位 strip・回答範囲）＋idx27 証拠調査**
  一次: idx79。二次: 78/27。
  実装: 「小数第N位で答えて」問への単位/サフィックス strip（値保存・書式のみ）、規定内容問の
  回答範囲限定強化（SOT-2666 残）。idx27 は かえで提案書の「スコープ対象外」数え上げ根拠を
  証拠調査してから対処（5 vs 7 — 証拠なき修正禁止）。

一次合計 15 idx（8/50/99/32/61/62/4/24/92/94/96/1/95/14/79）— 指示の 8〜15 上限内。

## 4. ガードレール（本サイクル固有）

- 基底構成は `scripts/sonnet_gold_cycle5.sh`（net59 実証済み）。全子はこの上にフラグゲート・
  既定OFF で積む。OFF 時 byte-identical。
- **一律予算増の再訪禁止**（SOT-2663 rejected、証拠つき）。回収はストア到達のみで行う。
- **現 wrong 7（idx1/9/22/27/78/79/95）を増やさない**: focused では各子の一次対象 MATCH 化と
  同時に、この 7 件が新たに Incorrect へ逆流しない（または対象として改善する）ことをゲートにする。
  特に idx24 の abstain→wrong 逆流（SOT-2663 実証）に注意。
- 番兵は `scripts/sonnet_sentinels.json` の 10 問（idx10/44/74/2/3/21/30/69/81/90）、
  `scripts/run_focused_gate.py --dev`。**focused 実行時は `RAG_CLAUDE_MCP_RESUME=0` 必須**
  （resume replay 罠 — SOT-2664 教訓: resume key は (model,question) のみで config 非依存）。
- Gemini は回答実行で禁止（`RAG_FORBID_GEMINI=1` 維持・cost $0 機械確認）。前処理で必要な場合も
  本サイクルの子は既存 OCR/抽出資産を優先し、genai を使う場合はビルド一度きり＋決定論ストアへ
  焼き込み＋コスト記録。gold 値ハードコード禁止・事前計算は質問を見ない網羅計算のみ。
- 子は focused のみ（gold100 全量はサイクル末の親 1 回のみ）。

## 5. 統合フェーズ（親・再開後）の手順

1. 人間コメント再取得（newest-wins）
2. 統合 focused: 全子一次対象（15 idx）＋wrong ガード 7（idx1/9/22/27/78/79/95）＋Sonnet 番兵 10
3. Sonnet dev gold100 ×1（claude-mcp・並列1・resume・Gemini $0 確認）。usage limit 逼迫時はスキップ明記
4. `docs/ai/sonnet_gold_history.jsonl` に cycle6 追記 → 申し送りコメント → PR/merge
