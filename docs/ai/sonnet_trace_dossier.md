# Sonnet 成功 trace 機序分解ドシエ — SOT-2630

flash 3.6 champion（`gold100_sot2610_waveA`: match47 / abstain46 / wrong7 / **net40**）が到達できず、
Sonnet dev gold100（`gold100_sonnet_dev`, official:false, SOT-2628）が MATCH した **11問** の到達手順を
1問ずつ分解し、移植方式（DET / HINT / FIXED?）と後続移植 issue（G1/G2/G3）への割当を確定する。

- **本ドシエは分析のみ。serve path は不変更**（本 issue では `.py` に一切触れない）。
- **gold 値は転記しない**。各問は「どの文書を・どのツール列で・どう抽出したか」の**手順**のみを記録し、
  最終数値（答え）はハードコード素材化を避けるため意図的に伏せる。中間セル参照（例 F22 / E14）は手順の
  一部として残すが、確定値の羅列はしない。
- 一次記録: Sonnet trace = `artifacts/gold100_sonnet_dev_resume.jsonl`（per-question record）/
  `artifacts/gold100_sonnet_dev.json`（集計）。flash 対比 = champion `artifacts/gold100_sot2610_waveA.json`
  ＋ issue 指定の退避 trace `artifacts/predictions_test_investigator.details.cycle2-sot2622.jsonl`
  （cycle2 で REJECT された det_pipeline フラグ ON の実測）。

## エグゼクティブサマリ

| idx | archetype | flash champion | Sonnet | 決定打（圧縮） | 移植方式 | 割当 |
|---:|---|---|---|---|---|---|
| 15 | document_extract | ABSTAIN | MATCH(0.75) | highlight_extract(黄)→単一セルF22→pivot行列ラベル逆引き→charges意味を横断裏取り | **DET** | G1 |
| 80 | document_extract | ABSTAIN | MATCH(0.95) | highlight_extract(黄)→単一セルE14→pivot逆引き→compute で条件付き集計を**自己検証** | **DET** | G1 |
| 17 | derived_calculation | ABSTAIN | MATCH(0.45) | highlight_extract 黄=0件→色無しで RED 数値を代替抽出→上昇率手計算 | **DET(tool-gap)** | G1 |
| 5 | derived_calculation | ABSTAIN | MATCH(0.55) | 報告書で最良モデル特定→**報告書に数値なし**→modeling 側コード既定値を採用 | **HINT** | G2 |
| 53 | derived_calculation | ABSTAIN | MATCH(0.82) | FR pptx slide の凡例色分けから ENG 特徴量タグを色で分類しカウント | **HINT/DET** | G2 |
| 96 | document_extract | ABSTAIN | MATCH(0.90) | schedule.xlsx を CP2→MS2→タスク群 の2ホップ横断参照 | **HINT** | G2 |
| 36 | fact_lookup | ABSTAIN | MATCH(0.90) | 中間報告/最終報告の2文書から F1 実測値を抽出し差の絶対値→本文改善量と一致確認 | **HINT** | G2 |
| 79 | fact_lookup | ABSTAIN | MATCH(0.82) | decrypt→read_office で暗号化 xlsx 読取→**compute が復号 xlsx を BadZip で開けず**→手集計 | **DET(tool-gap)** | G2 |
| 56 | derived_calculation | ABSTAIN | MATCH(0.45) | read_chart_values が埋込画像で失敗→caption_image で y 軸目盛を転記 | **DET/HINT** | G3 |
| 84 | fact_lookup | **WRONG** | MATCH(0.90) | 両者とも同一ページ/スライドに到達。**差は整形（"5" vs "5ページ"）のみ** | **FIXED?**(SOT-2619) | G3 |
| 74 | version_diff | **WRONG** | MATCH?(0.0) | **Sonnet も実際は棄権（"わかりません"）。version_diff が null 6連。judge 偽陽性** | **FIXED?/再測定** | G3 |

**最重要 caveat 2件**

1. **idx74 の "Sonnet=MATCH" は judge 偽陽性**。resume record の Sonnet 回答は文字通り `わかりません`（confidence
   0.0, `version_diff` が version ペア不確定/非隣接/大規模変更を理由に **null を6回連続返却**）。集計 JSON は
   これを MATCH に数えているが、**移植可能な成功手順は存在しない**。G3 は「到達済み gap」として扱わず、
   focused 実測でまず MATCH の再現性自体を検証すること。
2. **idx84 は純粋な naturalization 差**。flash champion（WRONG）の回答は `5（スライド6）`、Sonnet（MATCH）は
   `5ページ目（スライド6…5/15）`。**両者とも同じページ 5 に到達**しており、差は表記整形のみ。SOT-2619 の
   fact_lookup naturalization 修正で既に直っている蓋然性が高い（G3 は focused 実測で確認するだけでよい）。

---

## G1 — ハイライト抽出条件系（SOT-2631）: idx15 / idx80 / idx17

このクラスタの共通機序は **「単一の色付きセルを検出 → その pivot 上の行/列ラベルを逆引きして抽出条件を復元 →
集計内容を言語化」**。champion の `det_pipeline:format_check`（`shape=outline_pivot`, `template_only`）は
テンプレ整形止まりで、**「どの抽出条件のセルか」の意味復元をしていない**のが分岐点。

### idx15 — Sheet1 黄色ハイライトの抽出条件と集計内容
- **Sonnet 手順**（format_check, 13 turns, conf 0.75）:
  `find_files`→train.xlsx 特定 → `highlight_extract(color=黄)` で Sheet1 の単一セル **F22** を検出 →
  `compute(sheet=Sheet1)` で outline pivot のヘッダ（sex, smoker, region, charges, 平均/bmi …）と該当行を確認、
  merged cell の前方補完で抽出条件（sex=female・smoker=yes・region=southeast・charges=2）を復元 →
  `file_grep` で「charges の 0/1/2＝低/中/高価格帯」の意味を提案書/会議録/最終報告から横断裏取り →
  当該セル＝「female×smoker=yes×southeast×charges=2 グループの bmi 平均」と結論。
- **flash 分岐点**: champion=ABSTAIN。cycle2-sot2622 では `det_pipeline:format_check` が iters=1/conf1.0 で答えを
  出したが cycle2 統合は net28 で REJECT（precision 崩壊）。つまり **「単発でセル値だけ返す」det_pipeline では
  抽出条件の意味復元が欠落**し、精度が出ない。
- **移植方式 = DET**。format_check det_pipeline を拡張し、(a) `highlight_extract` で単一ハイライトセルを機械検出、
  (b) outline_pivot の行/列ラベルを merged-cell 前方補完で逆引き、(c) 抽出条件＋集計内容を構造化出力。
  charges のような**ドメイン語意（0/1/2＝価格帯）の横断裏取りは HINT として補助**。

### idx80 — Sheet2 黄色ハイライトの抽出条件と集計内容（最もクリーンな DET）
- **Sonnet 手順**（format_check, 8 turns, conf 0.95）:
  `highlight_extract(color=黄)`→Sheet2 単一セル **E14** → `read_office` で pivot 構造（行ラベル=children→smoker、
  列=合計/age・合計/bmi）を確認し E14＝children=3・smoker=no・合計/age 列に該当と特定 →
  **`compute` で train シートに children==3 & smoker=='no' の age 合計を再計算し、E14 の値と一致することを自己検証**。
- **flash 分岐点**: champion=ABSTAIN（document_extract）。cycle2-sot2622 は det_pipeline で iters=1/conf1.0 だが同様に
  条件意味を欠き REJECT。
- **移植方式 = DET（本クラスタ最優先）**。idx80 は **compute による条件付き再集計で答えが自己検証できる**ため、
  決定論パイプライン化に最も適する。移植の骨格＝`highlight_extract 単一セル → pivot ラベル逆引き →
  compute で条件付き集計を再現し一致検証`。G1 はこの idx80 を DET リファレンス実装として先行させ、idx15 を同型で拡張。

### idx17 — 黄色ハイライト かつ RED の数値の上昇率（tool-gap 露呈）
- **Sonnet 手順**（numeric, 19 turns, conf **0.45**）:
  `canonical_route`/用語集で AYM=青葉与信マネジメント, MM=会議録 と特定 → 会議録 PDF 3件 →
  **`highlight_extract(color=黄)` は全ファイル 0 件** → 色指定なし `highlight_extract` で色付きテキストを取得し
  RED 数値（0.589, 0.602）を確認 → 上昇率＝(最後−最初)/最初×100 を**手計算**（compute 未使用、turn 上限）。
  黄色ハイライトの直接確認ができず confidence は中。
- **flash 分岐点**: champion=ABSTAIN（derived_calculation, 19 turns で highlight_extract×7 の後棄権）。
  **flash も Sonnet も「黄色 ∧ RED」の複合条件を highlight_extract で検出できていない**のが本質。
- **移植方式 = DET（tool-gap 修正）**。単なるプロンプト誘導では埋まらない。真の blocker は
  **`highlight_extract` が「セル背景色＝黄」∧「フォント色＝RED」の複合条件を返せない**こと。G1 は
  highlight_extract をセル毎に「背景色 + フォント色」を返す形へ拡張し、黄背景∧赤字を決定論フィルタで抽出 →
  compute で上昇率を検算、まで通す。**現状の Sonnet 回答（RED-only 前提, conf0.45）は幸運な一致の可能性があり、
  成功手順としては採らない**。

---

## G2 — lookup / derived 系（SOT-2632）: idx5 / idx53 / idx96 / idx36 / idx79

champion はいずれも ABSTAIN（主因は BUDGET_EXHAUSTED 系の turn 超過）。Sonnet は横断参照・コード裏取り・
復号などの**追加ホップを最後まで走らせて回答**した点が分岐点。多くは誘導（契約型ヒント）で埋まる。

### idx5 — 最良モデルの max_depth（HINT・中リスク）
- **Sonnet 手順**（numeric, 20 turns, conf 0.55）: `file_grep`/`caption_image` で最終報告 PDF と modeling コードを特定 →
  報告本文＋会議録で最良モデル＝Histogram-based Gradient Boosting(回帰) を確認 →
  **報告書本文に max_depth の数値記載がない**ため `src/modeling.py` の `HistGradientBoostingRegressor` 構築部の
  `model_params.get("max_depth")` **既定値**を採用。設定上書きファイルの有無は turn 上限で未確認（Sonnet 自身が caveat）。
- **flash 分岐点**: champion=ABSTAIN（derived_calculation, model_error/budget）。「報告書に無い数値をコード既定値へ
  フォールバックする」発想に到達していない。
- **移植方式 = HINT**。契約型ルーティングに「numeric ハイパラが報告書に無い場合は modeling ソースの既定値へ
  フォールバック」ヒントを追加。**リスク中**（コード既定値＝実際の学習時値とは限らない。上書き config の探索を
  ヒントに含めること）。

### idx53 — 選択特徴量のうち ENG-FT の個数（HINT/DET）
- **Sonnet 手順**（numeric, 13 turns, conf 0.82）: TOTO=東都人材プラットフォーム と特定 → FR pptx を `read_office` →
  slide5「選択特徴量(14変数)」の凡例（原特徴量／エンジニアリング特徴量の色分け）に基づき、**赤タグ図形＝ENG 特徴量**
  を色で分類してカウント、残りが原特徴量で合計 14 と整合することを確認。
- **flash 分岐点**: champion=ABSTAIN（derived_calculation）。凡例色→カテゴリ写像による分類カウントに到達せず。
- **移植方式 = HINT/DET**。pptx 図形の塗り色を読み、凡例（legend）→カテゴリを写像して数える手順。read_office/
  highlight 系に「凡例色マッピングで特徴量を分類」ヒントを付す。整合検証（分類数の和＝総数）を必須ゲートに。

### idx96 — チェックポイント2 に関連するタスクID（HINT・低リスク）
- **Sonnet 手順**（simple_lookup, 4 turns, conf 0.90）: schedule.xlsx を読み、Sheet3 のチェックポイント表で CP2 の
  関連 MS＝MS2 を特定 → Sheet2 のマイルストーン表で MS2 の関連タスク群を特定（**CP→MS→Task の2ホップ**）。
- **flash 分岐点**: champion=ABSTAIN（document_extract）。素直な2ホップにも関わらず棄権（探索が単一シートで打切り）。
- **移植方式 = HINT（低リスク）**。「CP は MS を介してタスクに紐づく。schedule.xlsx の複数シートを跨いで解決せよ」
  という契約型ヒント。conf0.90 で機序も安定。

### idx36 — 中間/最終 F1 スコアの差の絶対値（HINT・低リスク）
- **Sonnet 手順**（simple_lookup, 5 turns, conf 0.90）: 中間報告書と最終報告書から F1 実測値をそれぞれ抽出し
  差の絶対値を算出 → 最終報告スライドの改善量記載（「＋◯◯改善」）と一致することを確認（**2文書抽出＋減算＋本文照合**）。
- **flash 分岐点**: champion=ABSTAIN（fact_lookup）。2文書に跨る数値抽出を最後まで走らせず。
- **移植方式 = HINT（低リスク）**。「中間/最終の2報告から同一指標を抽出し差を取り、本文の改善量記載で検算」。

### idx79 — 1タスク当たり想定工数が最大の担当者（DET tool-gap ＋ HINT）
- **Sonnet 手順**（simple_lookup, 13 turns, conf 0.82）: かえで総合病院フォルダ特定 → 暗号化契約書を `read_office` で
  復号読取（実施体制6名）→ 暗号化 schedule.xlsx を `decrypt` で復号確認し `read_office` で「リソース配分」想定工数と
  「WBS・タスク管理」担当者を取得 → **`compute` が当該復号 xlsx を BadZipFile で開けず**、read_office 由来の確定表で
  担当者別タスク数を手集計し 想定工数÷担当タスク数 を算出（担当実績0名は除外）。
- **flash 分岐点**: champion=ABSTAIN（fact_lookup, model_error）。復号後の集計まで到達せず。
- **移植方式 = DET（tool-gap 修正）＋ HINT**。真の blocker は **`compute` が復号済み（in-memory）xlsx を
  BadZipFile で開けない**こと。G2 は compute（および関連ツール）を decrypt 済みバッファ経由で開けるよう修正すれば、
  「担当者別タスク数の集計→工数按分」が決定論化できる。「鍵付きは社内管理を確認して復号」の誘導は HINT で補助。

---

## G3 — SOT-2619 効果の focused 実測 ＋ chart 目盛系（SOT-2633）: idx74 / idx84 / idx56

### idx84 — F1 ランキングが載るページ数（FIXED? 濃厚・書式差のみ）
- **Sonnet 手順**（simple_lookup, 5 turns, conf 0.90）: 最終報告 pptx を `read_office` で全文抽出 → 「順位｜モデル｜
  Macro F1｜Accuracy」形式のランキング表を含むスライドを特定し、その**文書内ページ表記（"5 / 15"＝5ページ目）**を回答。
- **flash 分岐点**: champion=**WRONG**。flash の回答は `5（スライド6）`、Sonnet は `5ページ目（スライド6…5/15）`。
  **両者とも同一ページ 5／スライド6 に到達しており、判定差は整形（"5" vs "5ページ"）のみ**。
- **移植方式 = FIXED?（SOT-2619）**。fact_lookup の naturalization 修正（"N"→"Nページ"）で既に MATCH 化している
  蓋然性が高い。G3 は idx84 を **focused 実測して SOT-2619 修正が効いていることを確認するだけ**でよい（新規手順の移植不要）。

### idx74 — 提案書 v1→v2 の実質変更（判定偽陽性・要再測定）
- **Sonnet 実態**（version_diff, 9 turns, conf **0.0**）: `version_diff(question=全文)` を必須初手で実行するも、
  質問文まま/v1-v2-v3 個別ペア/フルパス/簡潔表現など **計6通りで null（版ペア不確定・非隣接・大規模変更・読取不能）**。
  契約ルール（value=null は推測せず棄権）に従い **回答は "わかりません"**。find_files で3ファイルの存在は確認済み。
- **集計の矛盾**: `gold100_sonnet_dev.json` は idx74 を **MATCH に計上**しているが、上記の通り Sonnet は棄権している。
  **これは judge（codex）の偽陽性**であり、**移植可能な成功手順は存在しない**。
- **flash 分岐点**: champion=WRONG（版差から人事変更を1点回答）。真の blocker は **`version_diff` が v1→v2 の非隣接/
  大規模変更ペアで null を返す**こと。
- **移植方式 = FIXED?/再測定**。G3 はまず focused 実測で idx74 の MATCH 再現性を検証し（偽陽性なら "到達済み" から除外）、
  併せて version_diff の版ペア解決（v1/v2/v3 の隣接判定・null 要因）を調査対象とする。**SOT-2619 の naturalization は
  version_diff の null 自体は直さない**点に注意。

### idx56 — y 軸に表示されている目盛りの最大値（DET/HINT・低 conf）
- **Sonnet 手順**（numeric, 19 turns, conf **0.45**）: `canonical_route`→01_eda.ipynb 特定 → notebook 内「目的変数分析」
  セクション確認 → 出力画像 `reports/figures/target_distribution.png` を発見 → **`read_chart_values` は埋込画像で
  numCache/元列指定不可のためエラー** → `caption_image` で y 軸目盛を転記し最大目盛を確認。決定論再検証は turn 予算で未達、conf 中。
- **flash 分岐点**: champion=ABSTAIN（derived_calculation）。**ただし cycle2-sot2622 の flash は同問を caption_image 経由で
  answered/conf1.0（Sonnet と同一の到達）**。つまり **flash も caption_image で到達可能**で、champion の棄権は
  read_chart_values 失敗＋budget 超過による。
- **移植方式 = DET/HINT**。chart 系決定論パイプラインに「**read_chart_values が埋込画像（numCache 無し）で失敗した場合、
  caption_image による軸目盛転記へフォールバック**」を追加。ただし caption_image は vision＝非決定論のため、
  ガード（複数サンプル安定性・目盛間隔の等差チェック）付きで。low-conf クラスタとして扱う。

---

## 検証（本 issue の受け入れ）

- **11問全てにドシエエントリ**（idx 5/15/17/36/53/56/74/79/80/84/96）があり、各々に**機序・flash 分岐点・移植方式
  （DET/HINT/FIXED?）**を確定した。 → 上表＋各節。
- **G1/G2/G3 割当を明記**: G1=idx15/80/17（DET 中心, idx80 を DET リファレンス先行）、G2=idx5/53/96/36/79
  （HINT 中心＋idx79/compute-decrypt と idx17 相当の tool-gap）、G3=idx84(FIXED?確認)/idx74(偽陽性・再測定)/idx56(chart fallback)。
  後続 issue はこの粒度で実装に入れる。
- **serve path 不変更・gold 値転記なし**: 本ドシエは `docs/` 配下のみ。`.py`・serve 経路・レジストリは未変更。
  各問は手順のみを記載し、確定値（答え）の羅列はしていない。

## 移植時の横断リスク（後続 issue への申し送り）

1. **判定偽陽性の混入**（idx74）: Sonnet dev の MATCH 集計は codex judge の緩さで棄権を MATCH に数える例がある。
   「Sonnet=MATCH」を無条件に「到達済み手順あり」と読まず、**resume record の実回答（confidence/answer）を必ず確認**すること。
2. **積極回答＝wrong 増**: Sonnet dev 全体は abstain を20減らした代償に wrong を21増やし net40→18 に低下。移植は
   **evidence completeness と出力正規化を同時にゲート**しないと precision が崩れる（cycle2 det_pipeline 全 ON の REJECT 教訓）。
3. **tool-gap が本質の問題**（idx17 の 黄∧RED 複合検出, idx79 の compute×復号 xlsx, idx56 の read_chart_values×埋込画像）
   はプロンプト誘導では埋まらない。**ツール側の決定論修正**が移植の要。
4. **低 confidence 手順**（idx17=0.45, idx56=0.45, idx5=0.55）は Sonnet 自身が caveat を付けており、**幸運な一致の
   可能性**がある。G1/G2/G3 は移植後に focused 実測で MATCH 再現性を確認してから昇格候補に載せること。
