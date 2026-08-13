# cycle9 — Sonnet dev gold100 失敗全数分析と改善方針（SOT-2696）

- 入力: `artifacts/gold100_sonnet_cycle8.json`（cycle8 実測: **89 match / 6 abstain / 5 wrong / net84**、cost $0）、
  `artifacts/gold100_sonnet_cycle8_resume.jsonl` / `gold100_sonnet_cycle8.log`、
  `artifacts/gold100_sonnet_cycle7.json`（前回比較）、`docs/ai/sonnet_gold_history.jsonl`、
  SOT-2689 申し送り、`docs/ai/sonnet_cycle_analysis/cycle8.md`。
- 分析方法: abstain 6 + wrong 5 の **11 件全数**を state code × 契約型 × 欠落証拠で分類し、cycle7 verdict と
  突合して「新規失敗 vs 継続失敗」を機械判定した。さらに各 idx について**コーパス実ファイル・既存ストア・
  レーンコードを read-only 並列調査**し、決定論到達可能性を証拠つきで確定した（本ドキュメントの根拠は全て
  実ファイルパス・実値・コード行の機械確認）。
- cycle7→8 突合（機械確認済み）: 回収 14 件（abstain→match idx17/23/48/50/53/60/68/73/79/88、
  wrong→match idx9/16/21/78）。新規失敗 5 件は **全て cycle7 MATCH**: abstain 化 idx6/32/61/99、
  wrong 化 idx95。継続 abstain idx36/98、継続 wrong idx1/14/22/27。
- state code 集計（abstain 6）: **BUDGET_EXHAUSTED 6 / 他 0**。SOT-2689 完了時の機械確認どおり、
  新規失敗 5 件の details/interventions に cycle8 レーン発火痕跡はゼロ（`vdiff_classify` 発火は run 全体で
  0 件、abstain/wrong 側 interventions は plan_fanout のみ）→ **cycle8 回帰ではなく LLM 経路分散チャーン**。

## 0. 最重要所見 — 残り損失は「LLM 経路分散」と「決定論 diff の構造欠陥」の2種類のみ

1. **abstain 6 件は全て BUDGET_EXHAUSTED** で、うち 4 件（idx6/32/61/99）は cycle7 MATCH のチャーン。
   今回の read-only 調査で **4 件全てについて証拠がコーパス/既存ストアに存在し、欠けているのは
   serve 側の決定論レーンだけ**であることを機械確認した（§1）。SOT-2689 申し送りの方針
   （予算緩和ではなく該当 lookup の決定論昇格）を、per-idx の実装計画に落とせる状態にある。
2. **idx36 の「honest abstain」は新証拠で覆る**: SOT-2691 は「中間 F1 のフル精度値は成果物に無い
   （leaderboard 8桁丸めのみ）」として意図的未配線としたが、本調査で
   `05.会議/報告資料/報告資料_2025-09-16.docx`（word/document.xml 内、3 箇所）に
   **フル精度 0.7329671168078127 が存在**することを機械確認した。
   |0.8291582445227382（metrics.json:f1_macro）− 0.7329671168078127| = **0.09619112771492555 = gold 完全一致**。
   台帳ルール（新証拠なき再訪禁止）の要件を満たす正当な再オープン。
3. **wrong 4 件（idx1/14/22/95）は全て version_diff で、共通の構造的真因**: `archetype_trust.json` の
   `version_diff.holdout_validated=false` により決定論 diff は advisory 止まり（`generate.py:270`）で、
   最終回答は plan_fanout の LLM が生成する。かつ決定論 diff 自体に per-idx の構造欠陥がある（§1）。
   RAG_VDIFF_CLASSIFY（cycle8 C6）は分類規則と prompt 契約のみで、これらの構造欠陥には触れていない。
4. 閉じた軸の維持: idx27（gold ハードコード以外の経路なし）、idx98（TM 案件の RATE 変更日は
   SOT-2690 の全数調査で証拠不在を実証済み・本サイクルの再グレップでも新証拠なし → honest abstain 維持）。

## 1. per-idx 全数分類（機械確認済み証拠つき）

### abstain（6）

| idx | 型 | state | cycle7 | 証拠の所在（機械確認） | 欠落しているもの | クラスタ |
|---|---|---|---|---|---|---|
| 6 | derived_calculation | BUDGET | MATCH | `artifacts/case_finance.jsonl` SOHK 行に estimate_amount_incl_tax=4,675,000 と confirmed_amount_incl_tax=4,675,000 が**両方格納済み**（差額 0円 = gold） | `case_finance_lane.py` は 6 ハンドラのみで「見込 vs 確定の単純差額」パターンが無い → 汎用 `_amount_difference` ハンドラ（全案件対象・fail-closed） | C1 |
| 32 | enum_set | BUDGET | MATCH | `青嶺…/04.分析/analysis_outputs/metrics.json` の feature_selection.selected_columns に `__x__` 交互作用列 6 件が**そのまま存在**（gold と一致） | analysis_xref_store は最終報告テキストのみ抽出で metrics.json 非対応。enum_set の決定論 pipeline 未登録 → metrics.json enum ストア＋レーン | C2 |
| 36 | fact_lookup | BUDGET | abstain | **新証拠**: `かえで…/05.会議/報告資料/報告資料_2025-09-16.docx` にフル精度中間 F1 0.7329671168078127、`metrics.json` に最終 0.8291582445227382。差 = gold 完全一致 | SOT-2691 が leaderboard（8桁）しか見ず意図的未配線にした。会議報告資料 docx からの段階メトリクス抽出＋`xref_coverage_lane._stage_metric_f1_diff`（実装済み・_LANES 未登録）の配線 | C3 |
| 61 | config_hyperparam | BUDGET | MATCH | `raw_artifact_store.jsonl` の per-case rollup に **applied_hyperparams が構築済み**（config model_params={} + `modeling.py:73-75` のコード既定 n_estimators=300/learning_rate=0.1 + config random_state=42 のマージ、京橋は cases_with_hyperparams に登録済み） | investigator ツールとしてのみ到達可能で自動発火レーンが無い → config_hyperparam 決定論レーン（raw_artifact_store 参照） | C2 |
| 98 | fact_lookup | BUDGET | abstain | RATE 変更日の証拠は SOT-2690 全数調査で不在実証・本サイクル再確認でも新証拠なし | — （**honest abstain 維持**。無理な回答化はしない） | closed |
| 99 | derived_calculation | BUDGET | MATCH | `doc_reach_store.jsonl` に みなみ野 糖尿病統計 docx の都道府県別死亡率表が**抽出済み**（最高 青森 18.2 / 低い方4位 滋賀 7.3、18.2÷7.3=2.493…→2.49 = gold） | rank-k・ペア比の派生計算が derived_metrics に無く、順位比レーンも無い → 統計表全列の rank-k/比率の網羅事前計算＋レーン | C3 |

### wrong（5）

| idx | 型 | cycle7 | 真因（コード行まで機械確認） | クラスタ |
|---|---|---|---|---|
| 1 | version_diff | wrong | 同定は gold と同一（比較表削除→1行要約化）だが、削除表の全セル値を列挙する**過剰詳細**で judge 不一致。`classify_change`（diffpair.py:747）にメトリクス比較表の要約的扱いが無い | C4 |
| 14 | version_diff | wrong | `_schema_underscore_renames`（diffpair.py:627-646）が v1→v3 非隣接ペアで loan_status を取りこぼす（4 列中 3 列のみ列挙）。列名変更抽出の網羅性欠陥 | C4 |
| 22 | version_diff | wrong | notebook diff（diff_store.py:398-500）が「class 列追加(64→65)」までは出すが、**追加行が目的変数の記述統計である**という意味づけを表出しない → 言い回し不一致 | C4 |
| 95 | version_diff | MATCH | `_xlsx_struct`（diffpair.py:304-325）が**セルを座標キーで固定比較** → r1→r2 の行挿入で全行シフトし、日付とテキストの偽 diff 12 件を出力（真の変更は T15 担当者に小林直樹追加のみ）。行ラベルアラインメント欠落 | C4 |
| 27 | derived_calculation | wrong | **軸クローズ維持**（cycle6 全数走査で実証済み・gold ハードコード以外の経路なし） | closed |

## 2. cycle8 変更への回帰帰属

- 新規失敗 5 件（idx6/32/61/95/99）の details/interventions に cycle8 昇格フラグのレーン発火痕跡なし
  （SOT-2689 完了時に機械確認済み。`vdiff_classify` は run 全体で発火 0、format_strip_paren は wrong 側 0）。
- 全て plan_fanout LLM 経路上の失敗で、cycle6→7、cycle7→8 と同型の**経路分散チャーン**。
  → 回帰ではない。恒久対策は per-idx の決定論レーン昇格（§1 クラスタ）＝チャーン耐性の内蔵。

## 3. cycle9 方針 — 残 abstain の決定論消し込み＋vdiff 構造修正

第1目標（abstain→0 を前処理で）: 6 件中 idx6/32/61/99 は**証拠がストア/コーパスに既在**で serve レーンの
追加のみ、idx36 は**新証拠（会議報告資料のフル精度値）**で再オープン。idx98 のみ honest abstain 維持
（証拠不在の実証済み）。達成すれば abstain は 1 件（idx98）まで落ちる。
第2目標（wrong 削減）: cycle8 C6 と同じ「決定論のストア/構造修正のみ」の限定適用で version_diff
クラスタ 4 件（idx1/14/22/95）の構造欠陥を修正する。judge 3回多数決・言い回しチューニングは
今回も見送り（scoring 側は `crag.py` の現行 votes 設定の検証のみ子で実施可）。

子issue クラスタ（4件、primary 9 idx / stretch 2）:

| 子 | クラスタ | primary idx | stretch |
|---|---|---|---|
| C1 | 案件金額 差額/合計の汎用ファクトレーン（case_finance 全案件×全金額ペアの網羅差分） | 6 | 他案件の同型金額差問合せ（チャーン耐性） |
| C2 | 分析出力メタデータ決定論レーン（metrics.json enum ＋ applied_hyperparams 自動発火） | 32, 61 | — |
| C3 | 段階メトリクス全精度＋統計表派生ランキング（会議報告資料フル精度抽出・rank-k/比率網羅） | 36, 99 | — |
| C4 | vdiff 構造決定論修正（xlsx 行ラベルアラインメント・列名変更網羅・メトリクス表要約・統計行意味づけ） | 95, 14, 1, 22 | 番兵 idx58 維持確認 |

- C1/C2/C3 は互いに独立ファイル（case_finance_lane / 新規 store+lane ×2）で並列安全。
- C4 は diffpair.py / diff_store.py に集中するため**1子に束ねて**ブランチ競合を回避する。
- 合計 primary 9 idx。全て回収なら **97 match / 1 abstain（idx98）/ 2 wrong（idx27＋残チャーン想定）級 = net 95 前後**が理論上限。
  実際はチャーン発生を見込み、統合 gold100 で net 88〜92 を現実目標とする。

## 4. ガードレール（全子共通・cycle8 から継承）

- serve path 変更は必ずフラグゲート・既定 OFF。OFF 時 byte-identical。dev 構成でのみ ON。
- Gemini は build スクリプト内のみ（本サイクルの新ストアは全て LLM-free で構築可能）。
  serve 中は `RAG_FORBID_GEMINI=1` の例外化を focused で確認。
- gold 値ハードコード禁止（事前計算は全案件×全属性/全ペア/全 rank の網羅計算のみ。
  C1 は全案件×全金額ペア、C2 は全案件× metrics.json/hyperparams、C3 は全案件×全報告×全メトリクス
  および全統計表×全数値列× rank1..10、C4 は構造アルゴリズムの修正のみ）。
- focused 検証は `run_focused_gate.py --dev` ＋ Sonnet 番兵 10 問（idx58 版、`RAG_CLAUDE_MCP_RESUME=0` 必須）。
  子は gold100 全量を回さない。
- 現 89 MATCH を守る: 各子は自分の対象 idx ＋番兵で回帰ゼロを確認してから完了する。
- 公式レーン（flash champion・公式 gold100・LB 提出・SIGNATE CLI）に触れない。全結果 official:false。
- NFD ファイル名注意: コーパスパスはシェル直書き NFC 文字列では到達できないことがある（本分析でも再現）。
  `find` 経由または `FileRef.path` 経由で解決すること。
