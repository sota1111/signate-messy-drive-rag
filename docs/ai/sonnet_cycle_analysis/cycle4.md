# Sonnet gold100 cycle4 分析（SOT-2651, 2026-08-11）

一次入力: cycle2 全量実測 `artifacts/gold100_sonnet_cycle2.json`（match44 / abstain30 / wrong26 / net18。
cycle3 は skipped_gold=true・繰越）、cycle3 focused 成果（SOT-2650, PR#171）、台帳
`docs/ai/sonnet_gold_history.jsonl`。

## cycle3 までの確定状態

- 決定論回収済み（モデル不変）: idx38（空composite enum）、idx57（語幹metric-presence）、idx63、
  **idx67**（cycle3: 提案/FR金額 10/10 + 金額差分列挙、focused Perfect）。idx87 は paren-strip で focused Perfect（LLM経路のまま）。
- idx22: notebook diff store で証拠到達済みだが judge Incorrect（回答の naturalization が長い）。
- idx93: OCR ストアで証拠到達済みだが要求粒度（A10 内容そのまま）と不一致。idx34: Missing のまま。
- idx56: SOT-2633 で**非成立判定**（y軸目盛1200は可視化から取得不能）→ 棄権のまま残す（無理な回答化をしない）。
- スキャンPDF 18本（129ページ）は build-time OCR ストア（`src/rag/index/ocr_store.py`）に転記済み。

## abstain 30 の per-idx 分類（state code × 契約型 × 欠落証拠）

state code: BUDGET_EXHAUSTED 25 / UNANSWERABLE 4 / SPIN_CUTOFF 1（cycle2 実測、全問 conf=0.0）。
iterations 13〜24 で予算切れが支配的 = 「証拠に lookup 1発で到達できない」ことが主因。

### クラスタA: スキャンPDF 行動行（action/task row）抽出不足 — 4件 [子1]
| idx | 契約 | 欠落証拠 |
|---|---|---|
| 20 | simple_lookup | 報告資料PDFの優先タスク表（担当2名条件）を行単位で取得できない |
| 34 | simple_lookup | M01→M02間で完了したAI×担当者。OCRページ被覆/リテラル共起未達（cycle3 Missing） |
| 70 | simple_lookup | 報告資料の Open 優先フォロー AI × 会議録の完了記録の突合 |
| 93 | simple_lookup | A10 内容の「そのまま抜き出し」粒度（OCR証拠到達済・粒度不一致） |

欠落は「ページテキスト」ではなく**構造化された行動行フィールド**（ID/内容/担当/期日/状態/出典）。
cycle3 申し送りどおり question-independent な action-row フィールド抽出が次の一手。

### クラスタB: xlsx ハイライト/グラフ可視事実の未事前計算 — 3件（+wrong 3件） [子2]
| idx | 契約 | 欠落証拠 |
|---|---|---|
| 39 | chart_read | train.xlsx Sheet1 グラフ1が可視化するカラム（gold=hum）— チャートメタデータ未収蔵 |
| 65 | numeric(timeout) | 相関係数シートの黄ハイライト条件（gold=相関係数<-0.99）— ハイライト集合→条件帰納が無い |
| 97 | numeric(timeout) | 黄ハイライト交差2セルの値差（gold=272）— ハイライトセル座標×値の全数収蔵が無い |

同因の wrong: idx42 / idx77（ハイライトセルの意味＝行列ラベル文脈）、idx82（オレンジ行を「存在しない」と誤答）。
既存資産: SOT-2564（CF dxf 色族、既定OFF）、SOT-2609（chart/spatial 決定論）、structure_store。

### クラスタC: 案件マスタの財務・工数派生メトリクス不足 — 5件 [子3]
| idx | 契約 | 欠落証拠 |
|---|---|---|
| 24 | numeric | 全案件×train データの「欠損≥1行数」全数計算（gold=AOMINE）※二次対象 |
| 37 | numeric | AOBM 見込/確定金額差 ÷ (ESTH−ACTH)（gold=22,000円）— 金額・工数オペランド束縛不能 |
| 40 | numeric | 支払月別精算総額 top3（gold=2025年10月:11,412,500円…）— 支払スケジュール全数集計が無い |
| 55 | simple_lookup | 事後精算案件の見積vs実績工数乖離最大（gold=AOMINE）— 横断乖離テーブルが無い |
| 76 | numeric | 単価+2,000円/実績−11.2h の反実仮想請求額（gold=79,200円の増加）— 単価/税率/工数オペランド不足 |
| 98 | simple_lookup | TM RATE 変更の想定年月日（gold=2025年7月1日）— RATE パラメータ履歴ストアが無い（cycle2 申し送り候補） |

### クラスタD: 最終報告/調査PDF の数値属性未収蔵 — 3件（+二次 3件） [子4]
| idx | 契約 | 欠落証拠 |
|---|---|---|
| 5 | numeric | 最終報告の最良モデル max_depth（gold=6）— モデルパラメータ表の属性化が無い |
| 28 | numeric | 影響大特徴量のうちターゲット相関最大（gold=BMI）— 特徴量×相関リスト未収蔵 |
| 64 | numeric | 将来フェーズA+B想定工数（gold=80〜130時間）— フェーズ計画工数の属性化が無い |

二次対象（同型・余力があれば）: idx48（新税率バンド）、idx50（Salary.com 差額）、idx68（投資実装係数）。

### 残余（今回は対象外）
- idx18（ページ番号）、idx52（別契約役割）、idx45（会議録2枚の enum）、idx32（metrics.json enum）、
  idx73（One-Hot閾値→Gender）、idx95（スケジュール r1→r2 diff timeout）、idx83（回帰係数 index=17 —
  derived_metrics の OLS 資産で回収可能性あり、次サイクル候補）、idx99 / idx64以外の UNANSWERABLE 群、idx56（非成立確定）。

## wrong 26 の分類（主クラスタのみ）

### クラスタE: 値は正しいが書式・冗長性で落ちる — 約10件 [子5]
- 説明文/根拠の付加: idx6（0円+説明）、idx8（約+括弧根拠）、idx27、idx36（約0.0962… vs 完全精度値）、
  idx42/77（意味説明過多）
- 括弧付加情報: idx41（11件(タスクID…)）、idx59（13ページ（スライド13…））、idx92（49件（内訳…））、
  idx88（+（担当：…））、idx4（bmi(相関係数…)）
- 単位/表記ゆれ: idx12（2ページ目+見出し詳細 vs 2ページ）、idx29（区間記法）、idx31（並記順序）
- cycle3 で idx81（鉤括弧）/ idx87（AYM）は focused 回復済み。残りは「値を保存したまま説明・付加を落とす」
  正規化の拡張で回収可能なクラス。既知失敗（cycle2 net28 の precision 崩壊）を踏まえ、fail-closed
  whitelist 方式を維持する。
- その他 wrong: 版差分の対象取り違え（idx0/1/14）、意味的誤答（idx9/78/82/84/85）は書式では直らない。
  idx82 はクラスタB（ハイライトストア）で証拠側から対処。

## cycle4 の子issue分解（5件・一次対象15 idx + wrong書式クラス）

1. 子1: スキャンPDF action-row フィールド抽出ストア（idx20/34/70/93）
2. 子2: xlsx ハイライト・チャート可視事実ストア（idx39/65/97、二次 wrong idx42/77/82）
3. 子3: 案件マスタ財務・工数派生メトリクス拡張（idx37/40/55/76/98、二次 idx24）
4. 子4: 最終報告/調査PDF 数値属性ストア（idx5/28/64、二次 idx48/50/68）
5. 子5: 値保存書式契約の締め直し（wrong idx4/6/8/12/29/31/36/41/59/88/92 クラス）

共通制約: 質問を見ない網羅事前計算のみ / serve path はフラグゲート既定OFF（OFF時 byte-identical）/
Gemini は前処理のみ / gold 値ハードコード禁止 / 子は focused（`run_focused_gate.py --dev`+Sonnet番兵）のみで
gold100 全量は親の統合時1回。
