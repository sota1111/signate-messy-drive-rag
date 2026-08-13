# cycle10 — Sonnet dev gold100 失敗全数分析と改善方針（SOT-2701）

- 入力: cycle9 実測 `artifacts/gold100_sonnet_cycle9.json`（90 match / 6 abstain / 4 wrong / net86、
  fresh sidecar `artifacts/gold100_sonnet_cycle9_resume.jsonl` の per-idx ツール列・evidence 実文）、
  SOT-2696 申し送り、`docs/ai/sonnet_gold_history.jsonl` cycle9 行。
- 分類軸: state code × 契約型 × 欠落証拠の機械特定 × コーパス/ストア実在検証（read-only 並列調査
  3 系統＋親による直接反証チェック）。帰属はすべて実ファイル・実ストア・実コード行を根拠にした。
- state code 集計（abstain 6）: BUDGET_EXHAUSTED 4（document_extract 2 / fact_lookup 2）、
  UNANSWERABLE 2（document_extract）。

## 0. 最重要所見 — 残り損失 10 件のうち 8 件は「証拠実在・到達/コミット欠陥」

cycle9 の残余 10 件（abstain idx16/42/52/59/71/98、wrong idx1/21/22/27）を全数検証した結果、
**閉軸 2 件（idx27/98）を除く 8 件すべてで証拠がコーパスまたは既存ストアに実在する**ことを機械確認した。
欠けているのは (a) 到達（エイリアス・フィルタ・複合書式マーカー）、(b) 決定論コミット（vdiff 逐語・
権威ソース選好）のみ。よって cycle10 も「前処理ストア＋決定論レーン」の質問非依存拡張で回収する。

特筆すべき発見 2 件:

1. **idx52 は text_fts の project フィルタ完全一致バグ**。`src/rag/index/text_fts.py::search` は
   `nfc(rproj) != proj` の**完全一致**で行を落とす。serve は `text_search(query='別契約',
   project='みなみ野')` と短縮名を渡したため、行の project 列『医療法人社団 蒼樹会
   みなみ野女性医療センター』と不一致 → no_match。実際には FTS 索引に
   『監視ダッシュボード構築(別契約)』（最終報告.pdf page:8/9、OCR 由来）が**既在**だった。
   このフィルタ意味論は他の短縮 project 指定クエリ全部に波及しうる横断バグ。
2. **idx16 の evidence は実在**（read-only 調査 agent は「ファイル欠落」と誤判定したが、親が直接反証）。
   `プロジェクト/青葉与信マネジメント株式会社/05.会議/報告資料/報告資料_2025-04-29.docx` p9 に
   run『0.589』= **YELLOW ハイライト × 赤字(EE0000)** を python-docx で確認。本文に『中間報告』を含み
   M02（社内用語集: 中間報告）の報告資料と自己同定できる。失敗要因は
   (a) 『中間報告資料』がどのファイル名とも一致せず reach 不能、(b) `extract_docx` が複合書式
   （ハイライト×文字色×太字×下線×イタリックの組合せ）マーカーを生成しない、の 2 点。

## 1. per-idx 全数分類（機械確認済み証拠つき）

### abstain（6）

| idx | 契約型 | state | 証拠実在 | 欠落と根本原因 |
|---|---|---|---|---|
| 16 | document_extract | BUDGET | **有** 報告資料_2025-04-29.docx p9: YL×赤字 run『0.589』 | 『中間報告資料』の doc-kind 解決不能（M02→RP 対応が無い）＋複合書式マーカー未生成 |
| 42 | document_extract | BUDGET | **有** ひがし丘 train.xlsx Sheet1 F22（黄 FFFFFF00, 35.9509…）。階層ピボット行 A4-D22 の親ラベル上方スキャンで sex=female/smoker=yes/region=southeast/charges=2、列見出し F3『平均 / bmi』が**シート内容のみから一意導出可能** | ハイライトセル記録に周辺表文脈（親ラベル・列見出し・集計種別）が無い |
| 52 | document_extract | UNANSW | **有** text_fts に『監視ダッシュボード構築(別契約)』（みなみ野最終報告.pdf page:8/9）既在 | `text_fts.search` の project フィルタ完全一致（短縮名『みなみ野』で全行落ち）。doc_fulltext_search は画像 PDF で生テキスト無し（OCR 済みテキストは ocr_store.jsonl に有、4597 字） |
| 59 | fact_lookup | BUDGET | **有** 京橋 提案書_final.pptx スライド13『8. 費用見積』: 金額トークン 23、契約金額/支払条件の価格表、**フッター実頁テキスト『13』**（全 18 スライドに可視頁番号） | pptx の金額提示ページ事実が heading_page 系ストアに無い（同ストアは docx/pdf のみ）。cycle4 では 13ページ到達実績あり（括弧付加で wrong）→ 値は歴史的にも到達可能 |
| 71 | document_extract | BUDGET | **有** 青嶺 会議録_2025-08-06.docx para23: run『4,250,000円』= bold∧underline∧italic 全成立（python-docx 確認） | serve の find_files が 会議録 を発見できず（要因調査は子で）＋複合書式（B∧U∧I）マーカー未生成 → ストア化で探索自体を不要化 |
| 98 | fact_lookup | UNANSW | 無（SOT-2690 実証、本サイクル再確認でも新証拠なし） | **honest abstain 維持（閉軸）** |

### wrong（4）

| idx | 契約型 | 証拠 | 根本原因 |
|---|---|---|---|
| 1 | version_diff | diff_store record は正しい変更を特定済み。old スライド7 実物は表題『6. 最終モデル性能指標と中間段階との比較』・表ヘッダ『指標/中間 (T04 linear)/最終 (hist_gradient_boosting)/改善幅』・小見出し『中間段階 vs 最終モデル性能比較』 | store summary が端点テキストのみで『中間段階と最終モデルの』『中間実測値と最終値』の意味枠を欠く。**gold 相当の語彙は old スライド7 から質問非依存で導出可能**（gold ハードコード不要） |
| 21 | fact_lookup | 契約書.docx 署名欄『部署名：人事本部 人材戦略部／主担当者：山田 太一／**役職：人材戦略部長**』（draft にも同構造）。会議録は展開形『人事本部 人材戦略部 部長』 | 権威ソース（契約書署名欄の役職フィールド）選好が無く、会議録の展開形を写経した。完全ラベル写経（SOT-2666）の権威ソース版 |
| 22 | version_diff | diff_store summary『記述統計（基本統計量）の表に、目的変数 class の列の統計量が追加された（Attr1〜64は同一）』= **gold と完全一致** | plan_fanout LLM が括弧部を逐語脱落（SOT-2700 実証の再現）。**store 逐語 direct-commit なら即 Perfect** |
| 27 | derived_calculation | 提案書.pptx 全文再確認でも『対象外』明記なし（file_grep クロスチェック済み） | **閉軸維持**（gold ハードコード以外の経路なし） |

## 2. cycle9 変更への回帰帰属

- abstain 6 / wrong 4 は**全て cycle8 以前から継続する失敗**（cycle9 の新規失敗ゼロ、cycle9 統合
  focused でも regressions=[]）。cycle9 昇格レーン（case_finance_diff / metrics_enum / staged_metrics /
  derived_ranking / vdiff_struct）の誤発火痕跡は fresh sidecar のツール列に無い。
- 回帰帰属なし → 全軸を前処理/決定論の恒久対策として実装する。

## 3. cycle10 方針 — 6 子クラスタ（primary 8 idx）

第1目標 abstain→0（idx98 の honest abstain を除く実質 5 件）＋確実な決定論 wrong 3 件を並列回収する。

| 子 | クラスタ | primary | guard |
|---|---|---|---|
| C1 | text_search project フィルタのエイリアス許容（部分一致/用語集正規化）＋picture-PDF 到達の回帰ガード | 52 | — |
| C2 | 複合書式ファクトストア＋決定論 lookup レーン（YL×赤字・B∧U∧I 等の組合せ全数、doc-kind エイリアス M02/中間報告資料→RP 解決を含む） | 16, 71 | — |
| C3 | xlsx ピボット文脈ハイライトレーン（親ラベル上方スキャン＋列見出し＋集計種別のビルド時焼き込み） | 42 | — |
| C4 | pptx 金額提示ページストア（費用/見積タイトル×金額トークン密度×価格表×可視頁番号） | 59 | — |
| C5 | vdiff direct-commit 昇格（store summary 逐語コミット）＋idx1 record の意味枠強化（old スライド由来） | 1, 22 | 9, 14, 95 |
| C6 | 契約書署名欄 contact-master（主担当者×役職の権威ソースレーン） | 21 | — |

- 実装面の既知資産: direct-commit は `generate.py::deterministic_front` ＋ `_holdout_validated`
  （`config/archetype_trust.json`）、fact_layer の diff_lookup 直接回答（`RAG_FACT_LAYER`）、
  commit_gate（未配線）。C5 はこのいずれかの最小配線で実現する（新規経路の発明は不要）。
- 閉軸: idx27（gold ハードコードのみ）、idx98（証拠不在の honest abstain）。再訪しない。

## 4. ガードレール（全子共通・cycle9 から継承）

- serve path 変更は必ずフラグゲート・既定 OFF。OFF 時 byte-identical。dev 構成でのみ ON。
- 事前計算は質問を見ない網羅計算のみ。gold 値ハードコード禁止。
- Gemini は前処理ビルドのみ可・回答実行は claude-mcp（Sonnet）のみ・`RAG_FORBID_GEMINI=1`。
- focused 検証: `run_focused_gate.py --dev` ＋ Sonnet 番兵 `scripts/sonnet_sentinels.json` 10/10、
  `RAG_CLAUDE_MCP_RESUME=0`（resume key は (model,question) のみ — 構成差を replay が隠す）。
- 子は focused のみ（gold100 全量はサイクル末の親 1 回）。
- 公式レーン（flash champion・公式 gold100・LB 提出・SIGNATE CLI）に触れない。全結果 official:false。
