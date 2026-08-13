# cycle8 — Sonnet dev gold100 失敗全数分析と改善方針（SOT-2689）

- 入力: `artifacts/gold100_sonnet_cycle7.json`（cycle7 実測: **80 match / 12 abstain / 8 wrong / net72**、cost $0）、
  `artifacts/gold100_sonnet_cycle7_resume.jsonl`（per-idx テレメトリ/evidence 実文）、
  `artifacts/gold_100_review.csv`（分類の正）、`artifacts/gold100_sonnet_cycle6.json`（前回比較）、
  `docs/ai/sonnet_gold_history.jsonl`、SOT-2683 申し送り、`docs/ai/sonnet_cycle_analysis/cycle7.md`。
- 分析方法: abstain 12 + wrong 8 の **20 件全数**を state code × 契約型 × 欠落証拠（resume jsonl の
  evidence 実文・ツール列）でクロス分類し、さらに **cycle6 verdict との突合**で「新規失敗 vs 継続失敗」を
  機械判定した。帰属はすべてテレメトリ実文を根拠にした（推測帰属なし）。
- 申し送り訂正: SOT-2683 完了コメントの abstain 列挙にあった **idx52 は誤記**（gold review / artifact とも
  cycle7 は **MATCH**）。正しい abstain 12 件は idx17/23/36/48/50/53/**60**/68/73/79/88/98。
- state code 集計（abstain 12）: BUDGET_EXHAUSTED 6 / SPIN_CUTOFF 2 / UNANSWERABLE 4。

## 0. 最重要所見 — 「新規失敗チャーン」が回収と並走している

cycle6→cycle7 の verdict 突合（機械確認済み）:

- **回収 10 件**: abstain→match idx8/34/47/52/56/62/66/83、wrong→match idx29/91（cycle7 子クラスタの狙いどおり）
- **新規失敗 7 件（cycle6 は全て MATCH）**: abstain 化 idx**23/48/60/79/88**（5件）、wrong 化 idx**16/21**（2件）
- 継続 abstain: idx17/36/50/53/68/98（＋idx73 は wrong→abstain 遷移＝precision 崩壊ではなく honest 化）

新規 abstain 5 件の state code は BUDGET_EXHAUSTED×3 ＋ UNANSWERABLE×2 に見えるが、evidence 実文では
**5 件全てが「ツール呼び出し予算（5回）内で対象文書/表に到達できなかった」型**（idx23 は timeout、
idx60 は search/find_files が最終報告資料を発見できず 0 件）。cycle6 で MATCH していた以上、
**証拠はコーパス到達可能**であり、欠けているのは証拠ではなく **浅い到達経路**である。
cycle7 昇格フラグへの帰属シグネチャ（レーン発火痕跡）は 7 件のいずれの evidence にも無い —
これは決定論バグではなく **Sonnet のツール経路分散 × 予算 5 回の構造的脆弱性**（確率的チャーン）。
予算緩和は SOT-2663 で逆流実証済みのため再訪しない。**対策は各対象の質問非依存な事実層/FTS
カバレッジ拡張で「1〜2 手で着地する」ようにすること**（cycle7 で回収した idx34/62 と同じ勝ち筋）。

## 1. per-idx 全数分類

### abstain（12）

| idx | 型 | state | cycle6 | 欠落証拠（evidence 実文より） | クラスタ |
|---|---|---|---|---|---|
| 23 | fact_lookup | BUDGET(timeout) | MATCH | ひがし丘 ACTH 155h10m の税込請求 vs 見込税込金額の減額。契約条件（25,000円/h・30分切上・見込170h・税10%）は idx78 の run が同一契約書から完全取得済み → **(170−155.5)×25,000×1.1=398,750 の請求シナリオ計算レーンが無く、生読解で timeout** | C1 |
| 98 | fact_lookup | BUDGET | abstain | TM=実費精算契約は用語集で解決済み。**RATE 変更日（gold 2025年7月1日）を示す単価/条件変更の時系列クロス参照が不在**（cycle7 K4 で未着手のまま残った軸） | C1 |
| 36 | fact_lookup | BUDGET | abstain | かえで F1 中間 vs 最終の差。gold `0.09619112771492555` は**全精度値＝分析成果物由来**（スライドは丸め値 0.733/0.829）。cycle7 K4 のクロス参照ストアに全精度の段階メトリクスが焼かれていない | C2 |
| 60 | document_extract | UNANSWERABLE | MATCH | 白峰 最終報告の未完事項 ID（AI-05/08/09）。**search が最終報告資料自体を発見できず、`find_files(project=白峰信用リスク評価)` が 0 件**（正式名は「白峰信用リスク評価株式会社」＝部分名正規化欠落）。file_grep(未完)も 0 件 → 未完事項 ID が FTS 不可視 | C2 |
| 73 | enum_set | BUDGET | wrong | one-hot 閾値の実装設定確認で **read_office が .py 非対応エラー**、file_grep 再探索は予算切れ。カテゴリ列 nunique×閾値の突合ストアも無い（gold Gender） | C2 |
| 79 | fact_lookup | UNANSWERABLE | MATCH | かえで計画フォルダのタスク一覧（担当者別タスク数・想定工数）に予算内で到達できず。当該 xlsx は暗号化（passwords.resolve 資産あり、SOT-2680 で解決実績）。**担当者別 工数/タスク数 派生メトリクスが未計算** | C3 |
| 88 | document_extract | UNANSWERABLE | MATCH | みなみ野 提案書 pptx のスケジュール案（第5週）。doc_table_lookup 該当なし・file_grep「週目」0 件 → **提案書スケジュール表（週→項目）がスケジュールストアに未収載** | C3 |
| 17 | derived_calculation | SPIN | abstain | AYM の MM（月次報告資料）系列の**黄ハイライト×赤字数値の時系列**（最初→最後の上昇率）。ハイライト値の存在は確認済みだが、文書横断の系列列挙＋上昇率計算の手段が無い | C4 |
| 48 | fact_lookup | BUDGET | MATCH | 青嶺 NY不動産 PDF の**マンション税 価格帯別 現行/新税率表**に予算内で未到達（line138 に言及確認まで）。cycle6 MATCH よりテキスト到達可能 → doc-table 網羅と帯別絶対差の派生計算が無い | C4 |
| 68 | derived_calculation | BUDGET | abstain | 投資実装係数。**式と全入力値（+22.6%/+15.2%/ROI 3.7倍）が `artifacts/ocr_store.jsonl` の p5 レコードに存在**するが、索引側 `image_ocr_store.jsonl` の同ページは 334 字に切詰められ数値が欠落（機械確認済: (0.226+0.152)×3.7=**1.3986**=gold）。serve の caption_image は GeminiForbidden で正しくブロック | C5 |
| 50 | derived_calculation | UNANSWERABLE | abstain | 東都 Salary.com 中央値/上位90%。同一 EMF 給与表から idx8（17,744）は cycle7 回収済み → **表は抽出済みで到達性の問題**。idx50 の run は file_grep で EMF 断片（Headline base salary）まで見えたが予算切れ。13,222 が store/FTS から引けるかの検証と再索引が必要 | C5 |
| 53 | derived_calculation | SPIN | abstain | TOTO=東都は解決済み（cycle7 K4）。**「FR書」→最終報告書の別名束縛と ENG-FT 特徴量分類の裏付けが未発見**（gold 6） | C2(stretch) |

### wrong（8）

| idx | 型 | cycle6 | 誤り方（evidence 実文より） | クラスタ |
|---|---|---|---|---|
| 9 | document_extract | wrong | diff store が見出し追記（「― クイックウィン」等）を SUBSTANTIVE 判定 → gold は「該当なし」。**見出しラベル追加＝内容不変を cosmetic に分類する規則の欠落**（cycle7 繰越 K6） | C6 |
| 14 | version_diff | wrong | v1→v3 の**列名アンダースコア化（loan_status 等）を SURFACE 側に落として不採用**、構成変更を substantive として回答。gold は列名変更のみ。非隣接版の列名レベル alignment（cycle6 K5 既知軸） | C6 |
| 16 | document_extract | MATCH | 黄×赤で 2 文書 2 値（4,620,000/FF0000、0.589/EE0000）を列挙、gold は 0.589 のみ。**「中間報告資料」のスコープ確定（どの報告資料が中間か）と最小集合契約が無い**。新規 wrong だが cycle7 レーン発火痕跡なし | C6 |
| 21 | fact_lookup | MATCH | 「人事本部 人材戦略部長」vs gold「人材戦略部長」。**既知の不安定番兵**（cycle7 統合 focused でも 9/10 の犯人、ON/OFF byte-identical 対照で環境非依存と実証済み）。過剰役職修飾の確率的揺れ | C6(番兵) |
| 1 | version_diff | wrong | 同定は gold と同一（比較表削除→1行要約化）。**過剰詳細な言い回し**で judge 不一致 | C6(stretch) |
| 22 | version_diff | wrong | 同定は gold と実質同一（class 列統計量の追加）。語彙差のみ | C6(stretch) |
| 78 | fact_lookup | wrong | 証拠は完全（特別規定なし＋一般規定 6.1〜6.3）で cycle7 K5 の合成契約も実装済みだが、**要求外の付加情報（料金モデル名・見込金額の性质等）を長く連ねて** judge 不一致。決定論合成レーンへの締め直しが必要 | C1 |
| 27 | derived_calculation | wrong | **軸クローズ（維持）**: `cycle6_k6_idx27_scope_investigation.md` が全19スライド走査で canonical 7 項目の不在を実証。回収経路は gold ハードコードのみ＝禁止 | closed |

## 2. cycle7 変更への回帰帰属

- 新規失敗 7 件（idx16/21/23/48/60/79/88）の evidence に **cycle7 昇格フラグのレーン発火痕跡は一切ない**
  （interventions 集計でも wrong 側 fired 0、abstain 側は plan_fanout のみ＝従来どおり）。
- idx21 は cycle7 中に ON/OFF ツール面 byte-identical の対照実験で環境非依存の確率揺れと実証済み。
- idx23/48/60/79/88 は全て「予算内到達失敗」型で、cycle6 の同一構成要素で MATCH していた。
  → **回帰ではなく経路分散チャーン**。恒久対策は §0 のとおり事実層の浅到達化（各クラスタに内蔵）。

## 3. cycle8 方針 — 残 abstain 12 の消し込み＋チャーン耐性＋決定論 wrong 修正

第1目標（abstain→0）: 12 件中 idx50/68 は**ストア済み証拠の配線/索引問題**（新規 vision 不要）、
idx23/36/48/60/73/79/88/98 は**質問非依存の事前計算/索引拡張**で証拠を用意できる。idx17/53 は
系列列挙・別名束縛の拡張で回収を狙う。あわせて cycle6→7 で顕在化した**チャーン**（前回 MATCH の
取りこぼし）を、対象 idx の事実層直結化で抑える。

子issue クラスタ（6件、primary 15 idx / stretch 4）:

| 子 | クラスタ | primary idx | stretch |
|---|---|---|---|
| C1 | 契約・請求ファクトレーン（請求シナリオ計算＋RATE変更時系列＋特別規定合成の決定論化） | 23, 98, 78 | — |
| C2 | 分析成果物クロス（全精度段階メトリクス・成果物レジストリ/未完事項ID・.py リーチ＋カーディナリティ enum） | 36, 60, 73 | 53（FR書/ENG-FT） |
| C3 | 計画・スケジュール表カバレッジ（暗号化計画xlsx復号→工数派生・提案書週次スケジュール） | 79, 88 | — |
| C4 | 書式系列・文書内数値表リーチ（黄×赤時系列＋上昇率レーン・税率表 doc-table＋帯別絶対差） | 17, 48 | — |
| C5 | 画像OCRストア配線修正＋式適用レーン（全文OCRの索引統合・EMF 給与表 FTS 到達・投資実装係数） | 68, 50 | — |
| C6 | vdiff 実質変更分類の決定論修正＋番兵安定化（見出し追記=cosmetic・列名変更=substantive・中間報告スコープ・idx21 番兵置換） | 9, 14, 16 | 1, 22 |

wrong フェーズ前倒しの限定適用: C6 は「abstain ≤5 到達後に wrong」の原則の例外として、**決定論の
ストア/契約修正のみ**（idx9/14 の分類規則、idx16 のスコープ契約）に絞って着手する。cycle7 で K5
（idx91/29/78）が同型の決定論修正として成功した前例に従う。judge 3回多数決化・言い回し系
（idx1/22 本体）は今回も見送り（stretch 扱い）。

番兵安定化（C6 内・infra）: 不安定 LLM 番兵 idx21 を、多サンプル安定性で選定した決定論 MATCH 問に
置換する（SOT-2623 の選定手順を再適用。dev ハーネスのみの変更で serve 影響なし）。以後の統合
focused の 10/10 ゲートを回復する。

閉じた軸: idx27（現状維持）。予算緩和軸（SOT-2663 逆流）・gold ハードコードは再訪しない。

## 4. ガードレール（全子共通）

- serve path 変更は必ずフラグゲート・既定 OFF。OFF 時 byte-identical。dev 構成でのみ ON。
- **Gemini は build スクリプト内のみ**。C5 は既存 OCR 成果の再索引が主で、新規 vision は原則不要。
  serve 中は `RAG_FORBID_GEMINI=1` の例外化を focused で確認。
- gold 値ハードコード禁止（事前計算は全案件×全属性/全ID/全版ペアの網羅計算のみ）。
- focused 検証は `run_focused_gate.py --dev` ＋ Sonnet 番兵 10 問（`RAG_CLAUDE_MCP_RESUME=0` 必須）。
  子は gold100 全量を回さない。
- 現 80 MATCH を守る: 各子は自分の対象 idx ＋番兵で回帰ゼロを確認してから完了する。
- 公式レーン（flash champion・公式 gold100・LB 提出・SIGNATE CLI）に触れない。全結果 official:false。
