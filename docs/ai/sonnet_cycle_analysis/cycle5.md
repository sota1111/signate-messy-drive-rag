# Sonnet gold100 改善サイクル第5次 — 失敗分析と方針（SOT-2662）

- 入力: `artifacts/gold100_sonnet_manual0812.json`（cycle4.5 手動中間実測, PR#183, official:false）
  = **match57 / abstain34 / wrong9 / net48**（基底構成 `scripts/sonnet_gold_manual_20260812.sh`）
- 前回申し送り: SOT-2651 コメント（2026-08-12 06:50 版が最新）
- 分析日: 2026-08-12（Fable, SOT-2662 親フェーズ1）

## 0. 結論（本サイクルの中心仮説）

**abstain 34 のうち 33 件が「ツール呼び出し予算(5回)を使い切った」という同一シグネチャで棄権している。**
`RAG_PLAN_FANOUT`（SOT-2661）は ON 時に per-question 非終端ツール予算を
`PLAN_FANOUT_DEFAULT_MAX_TURNS = 5`（`src/rag/agent/investigator.py:542`、env `RAG_PLAN_FANOUT_MAX_TURNS`）
へ段構成で置き換える。cycle4（fanout OFF）の abstain は iteration 13〜24 の予算切れだったのに対し、
本実測の abstain は全件 iteration 6〜12 で提出しており、多くは**「次に呼ぶべきツールと対象ファイルを
特定済みのまま、最後の1手前で予算切れ」**している（idx77=highlight_extract 対象特定済み、
idx49=3件目の会議録への format_events 残し、idx75=特定済み提案書への read_office 残し、
idx83=係数シート読み残し、idx16=format_events 照会直前、等）。

→ 第5次の最大 EV は「fanout の段予算の較正＋確定ターゲット持ち越し時の有界継続（finisher）」。
これは前処理ストア拡張（C4/C5）と独立に並列実装できる。

### MATCH→ABSTAIN churn 7 の帰属（証拠つき・確定）

churn 7 = idx8/16/49/66/75/83/92（cycle4 MATCH → 4.5 ABSTAIN）。
ツール列・evidence 全数確認の結果:

- **idx16/49/66/75/83/92 の 6 件は evidence 本文に「予算(5回)切れ」を明記**して棄権 — plan_fanout の
  段予算が探索を切り詰めたことが直接原因（単発揺らぎではない）。
- idx8 のみ予算シグネチャなし: 対象 docx を全文読了した上で「ML エンジニア vs データエンジニアの
  米国給与比較の記載が見つからない」— 証拠到達の問題（後述、C1 の二次診断対象）。

## 1. wrong 9 の per-idx 分類

| idx | archetype | 症状 | 帰属 | クラスタ |
|---|---|---|---|---|
| 0 | version_diff | ほぼ完全一致だが冒頭の「提案書スライド6」（文書名プレフィクス）欠落 | naturalization に主語/文書名プレフィクスなし | C2 |
| 1 | version_diff | 削除内容の記述が詳細すぎ「1行要約への置換」という gold の要点と不一致 | diff 候補選択・要約粒度 | C2 |
| 9 | document_extract | gold=「該当なし」なのに変更内容を列挙して回答 | 該当なし判定＋裸形式契約の欠落 | C2 |
| 14 | version_diff | rank1 の差分（STEP再編）を答え、gold（列名アンダースコア修正）と別差分 | diff_store ranked candidates を serve が rank1 しか使わない | C2 |
| 21 | fact_lookup | 「部長」— gold「人材戦略部長」の切り詰め | 肩書・ラベル完全写経の契約欠落 | C3 |
| 27 | derived_calculation | 3 と回答、gold=7（スコープ対象外の数え上げ対象を誤認） | 証拠調査が必要（真の値誤り） | C3 |
| 62 | fact_lookup | 「n_estimators（500 vs 300）」— gold「n_estimators（1位=500、2位=300）」 | ラベル付き複数値形式の契約欠落 | C3 |
| 78 | fact_lookup | 内容は gold とほぼ同旨だが冗長・追加主張（「上限は設けられていない」等）を付加 | 回答範囲限定（問われたことのみ）契約欠落 | C3 |
| 85 | document_extract | 「なし(全6項目達成)」— gold「該当なし」 | 該当なし裸形式契約の欠落 | C2 |

## 2. abstain 34 の per-idx 分類

state code 集計: BUDGET_EXHAUSTED 25 / UNANSWERABLE 8 / SPIN_CUTOFF 1。
ただし evidence 本文ベースでは **33/34 に予算/上限シグネチャ**（唯一の例外 idx8）。
全件 conf=0.0（idx27系を除く）、iteration 6〜12 提出 = fanout 5 ターン予算支配。

凡例: 【一手前】=次ツール・対象特定済みで予算切れ（finisher 直撃）/【カバレッジ】=ストア側の証拠欠落 /【PARK】=本サイクル対象外。

| idx | archetype | 欠落証拠・状況 | クラスタ |
|---|---|---|---|
| 8 | derived_calc | docx 全文読了も ML/DE 米国給与比較の記載未発見（予算シグネチャなし・churn）。別文書/PDF に証拠がある可能性 → 二次診断 | C1二次 |
| 11 | document_extract | 【一手前】pdf_emphasis 2件実行前に予算切れ（太字∧下線∧イタリック） | C1二次 |
| 12 | fact_lookup | 見出し「WBS観点の進捗状況」の印字ページ番号 — 見出し→ページ locator 不在 | C5 |
| 16 | document_extract | 【一手前・churn】format_events(黄×赤) 照会直前に予算切れ | C1 |
| 17 | derived_calc | 「AYM の MM」該当ファイル特定できず（エイリアス解決欠落; MM=月次モニタリング資料?） | C5二次 |
| 18 | fact_lookup | M04 会議録の進捗サマリ印字ページ — C5 と同型 | C5 |
| 22 | version_diff | ipynb 対象外（SOT-2646 で honest 報告確定）。無理な回答化をしない | PARK |
| 24 | data_shape | 欠損値行数最多案件 — compute×10 で予算切れ。全案件×欠損行数の質問非依存派生メトリクス候補 | PARK(次候補) |
| 32 | enum_set | metrics.json selected_columns × 生成コードのクロス参照 | PARK(次候補) |
| 34 | document_extract | MINAMINO M01→M02 完了 AI×担当者 — スケジュール/WBS 行に到達できず | C4 |
| 35 | derived_calc | F1 次順位モデルの Accuracy — report_attr_lookup 到達も rank-of クエリ形が無い | PARK(次候補) |
| 36 | fact_lookup | 中間 vs 最終 F1 差 — 案件横断バージョン間メトリクス差分 | PARK(次候補) |
| 45 | enum_set | 【カバレッジ】action_row_lookup が京橋の会議録 PDF を「該当案件なし」（ストア未収録）＋ M2→M3 状態差分 | C4 |
| 47 | derived_calc | 【一手前】B22 数式・対象行の追加照会前に予算切れ | C1二次 |
| 48 | fact_lookup | NY 不動産税率表の絶対差最小価格帯 — 予算切れ（wrong→abstain 転換組。ガード対象） | C1ガード |
| 49 | document_extract | 【一手前・churn】3件目会議録への format_events(comment) 残し | C1 |
| 50 | derived_calc | 90%タイル−中央値 — read_office 後の該当セクション読み残し | C1二次 |
| 52 | document_extract | 「別契約」明記役割の抽出 — file_grep×5 で予算切れ | C1二次 |
| 53 | derived_calc | FR書 ENG-FT 数え上げ — read_office 1回で予算切れ | C1二次 |
| 56 | derived_calc | y軸目盛り最大値 — SOT-2633 で非成立確定 | PARK |
| 61 | config_hyperparam | コード上のデフォルト値解決（n_estimators等）— text_search×5 でも確定前に上限 | PARK(次候補) |
| 66 | fact_lookup | 【churn】ipynb チャート値 — read_office 非対応→grep 未ヒットで上限。nb 出力ストア到達経路 | C1 |
| 68 | derived_calc | 投資実装係数 — 式ページ特定後の代入前に予算切れ | C1二次 |
| 72 | derived_calc | KSS データエンジニア担当タスク数 — スケジュール行×担当者集計 | C4二次 |
| 73 | enum_set | One-Hot 閾値×カテゴリ列クロス | PARK(次候補) |
| 75 | fact_lookup | 【一手前・churn】特定済み提案書.pptx への read_office 直前に予算切れ | C1 |
| 77 | document_extract | 【一手前】highlight_extract 実行直前に予算切れ（wrong→abstain 転換組。ガード優先） | C1ガード |
| 83 | derived_calc | 【一手前・churn】係数記載シート読み取り前に予算切れ | C1 |
| 92 | derived_calc | 【churn】スケジュール.xlsx BadZipFile→read_office 再試行で予算切れ | C1 |
| 93 | document_extract | 【カバレッジ】A10 が action_row_store 未収録（found:false）＋読み残し | C4 |
| 95 | version_diff | schedule r1→r2 xlsx 差分（未着手→完了除外の条件付き）— diff 候補の条件フィルタ | C2二次 |
| 96 | document_extract | 「チェックポイント2」全文 0 ヒット — スケジュール定義からの派生クロス参照が必要 | C4 |
| 98 | fact_lookup | RATE 変更時期の推定 — diff_lookup 到達も確定前に上限 | C2二次 |
| 99 | derived_calc | 死亡率比 — 対象ページ読み残し | C1二次 |

## 3. 子issue クラスタ（5件・一次 focused 対象: 重複なし15 idx）

- **C1 — plan_fanout 段予算較正＋finisher（確定ターゲット有界継続）**
  一次: idx16/49/75/83（churn 4、予算切れ証拠つき）。二次: 8/11/47/50/52/53/66/68/92/99。
  ガード: wrong→abstain 転換組（idx11/24/36/48/77/95/96）を wrong に戻さないこと＋番兵回帰ゼロ。
  実装: `RAG_PLAN_FANOUT_MAX_TURNS` 較正（5→8〜12 の focused 掃引）と/または新フラグ
  `RAG_FANOUT_FINISHER`（予算切れ時、次ツール＋対象が具体特定済みの場合のみ +N ターンの有界継続。既定OFF）。
- **C2 — version_diff 候補選択＋naturalization 文書名プレフィクス＋「該当なし」裸形式契約**
  一次: wrong idx0/9/14/85。二次: wrong idx1、abstain idx95/98。
  実装: serve が diff_store ranked candidates を質問条件（除外指定・対象範囲）で絞る選択ロジック、
  naturalization に主語/文書名プレフィクス付与、該当なし判定時は装飾なし「該当なし」のみ返す契約。
- **C3 — 完全ラベル写経・回答範囲限定の書式契約＋idx27 証拠調査**
  一次: wrong idx21/62/78。二次: idx27（真の値誤りの証拠調査）。
  実装: bare-answer 契約への追記（肩書・ラベルは原文どおり完全写経 / 複数値はラベル付き形式 /
  問われた範囲のみ・追加主張禁止）。idx27 は提案書の「スコープ対象外」数え上げ根拠を調査してから対処。
- **C4 — action-row/ID クロス参照ストアのカバレッジ拡張**
  一次: abstain idx45/93。二次: 34/72/96。
  欠落: 京橋会議録 PDF の action 行未収録（45）、A10 未収録（93）、チェックポイント定義→タスクID の
  派生クロス参照（96）、期間×担当者×状態のクエリ形（34）。全案件×全 ID 行の質問非依存網羅で拡張。
- **C5 — 見出し→印字ページ locator ストア＋資料エイリアス解決**
  一次: abstain idx12/18。二次: 17（「MM」エイリアス）。
  実装: 全 docx/pdf × 全見出し → 印字ページ番号（印字ページ=表紙除外の既知規約, SOT-2612）の
  質問非依存ストア＋lookup 配線。エイリアス（略称→資料種別）解決の最小辞書。

### C1 実測結果（子2件・確定）

- **SOT-2664 = finisher（`RAG_FANOUT_FINISHER`, max=1）→ promoted（PR#189, merged）**。予算切れ後に
  《対象特定済み単一文書リード》のみ1手追加。idx16/49/75→Perfect。compute 除外で abstain→wrong 逆流阻止。
- **SOT-2663 = 段予算較正（`RAG_PLAN_FANOUT_MAX_TURNS` 5→8）→ rejected**。finisher の上に予算=8 を積むと
  一次は 16/49/75/92→Perfect（4/6, idx92=xlsx BadZipFile→read_office 再試行を予算で回収）で「4+」は満たすが、
  **guard idx24 が Missing→Incorrect に逆流**（同一セッション OFF-control 予算=5 で idx24=Missing に復帰＝
  **予算帰属を確定**：「欠損値行数最多案件」の横断 compute 列挙が余剰ターンで誤案件『白峰』を確定）。idx95 の
  Incorrect は finisher 未発火の基底 stochastic（予算非帰属）。net ≈ 0（+idx92／−idx24）で wrong を1件増やし、
  「打ち切りでなく構造」設計（cycle2 precision 崩壊教訓）に反する。**段予算は 5 のまま据え置き**、C1 は
  finisher のみを採用。番兵は両走とも 10/10・regressions=[]。serve 変更なし＝ON/OFF byte-identical。
  harness: `scripts/sonnet_child_fanout_budget_focused.sh`（`RAG_PLAN_FANOUT_MAX_TURNS=5` で自己 OFF-control）。

## 4. ガードレール（本サイクル固有の注意）

- 基底構成は `scripts/sonnet_gold_manual_20260812.sh`（net48 実証済み）。全子はこの上に
  フラグゲート・既定OFF で積む。OFF 時 byte-identical。
- **C1 は諸刃**: 現 net48 の wrong9 は 5 ターン予算がもたらした wrong→abstain 転換 7 件を含む。
  予算を緩めると abstain→wrong 逆流のリスクがある。focused では一次対象の MATCH 化と同時に
  転換組 7 件（idx11/24/36/48/77/95/96）が Incorrect に戻らないことを必須ゲートにする。
- 番兵は現在 8/10（idx0/16 は既知 churn ペアかつ本サイクルの改善対象）。番兵判定は多サンプル安定性
  ベース（単発 MATCH 不可）。
- Gemini は回答実行で禁止（`RAG_FORBID_GEMINI=1` 維持・cost $0 機械確認）。本サイクルの子は
  前処理でも Gemini 不要（既存 OCR ストア資産を使う）。gold 値ハードコード禁止・事前計算は
  質問を見ない網羅計算のみ。

## 5. 統合フェーズ（親・再開後）の手順

1. 人間コメント再取得（newest-wins）
2. 統合 focused: 全子一次対象（15 idx）＋転換ガード 7＋Sonnet 番兵 10
3. Sonnet dev gold100 ×1（claude-mcp・並列1・resume・Gemini $0 確認）。usage limit 逼迫時はスキップ明記
4. `docs/ai/sonnet_gold_history.jsonl` に cycle5 追記 → 申し送りコメント → PR/merge
