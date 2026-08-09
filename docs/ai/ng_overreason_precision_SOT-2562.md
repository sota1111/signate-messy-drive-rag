# SOT-2562（review=human follow-up）— 回答NG項目の再確認・再対策・検証

人間差し戻しコメント（`review=human`）:「問いの回答NGの項目を再確認し、合格するか検証して、NGの場合は
再対策、検証、確認してください。」への対応記録。gold100 は実行せず、該当 idx を focused/offline で検証。

## 1. 再確認（NG項目の特定）

基準は SOT-2550 統合実測（`docs/ai/gold100_after_error_fixes.md`）の wrong-6
`{4, 9, 14, 49, 83, 92}`。これらを **現行 main**（SOT-2563 file_grep deadline / SOT-2564
highlight・font-emphasis マージ後）で focused 再実行した（`scripts/sot2562_ng_recheck.py`、
A1–E 候補フラグ + SOT-2562/2563/2564 の測定フラグ ON、genai timeout=180s 注入）。

| idx | 現行mainの帰結 | 状態 |
| --- | --- | --- |
| 4 | `smoker`（gold `bmi`） | **wrong（真のNG）** |
| 9 | 版差分の変更を列挙（gold `該当なし`） | **wrong（真のNG）** |
| 14 | `わかりません` | safe abstain（NGでない） |
| 49 | `わかりません`（max_turns） | safe abstain（NGでない） |
| 83 | `わかりません` | safe abstain（NGでない） |
| 92 | `わかりません` | safe abstain（NGでない） |

→ SOT-2563/2564 のマージにより **idx14/49/83/92 は既に安全棄権へ移行済**（誤答=−1 は解消済）。
残る真の NG は **idx4 と idx9 の2件**で、いずれも「過剰推論（over-reasoning）」型。

## 2. 再対策（既定OFFフラグの精度ゲート2本）

champion serve は全フラグ既定 OFF で **byte-identical** を維持（GRANULARITY_NORMALIZATION /
CONFLICT_RESOLUTION と同じ配線）。ターミナルコミットで各1回のみ発火する one-shot 補正。

### (A) `RAG_NUMERIC_FEATURE_CORR` — 数値特徴量の字義厳守（idx4）
- **失敗**: 「相関が最も高い**数値特徴量**」に対し、カテゴリ列（sex/smoker）を `.map({...})` で 0/1
  に数値化して相関に含め、`smoker` が `bmi` を上回った。notebook の `corr(numeric_only=True)` と
  「数値特徴量」指定の双方に反する過剰推論。
- **対策**: compute トレイルに「相関 × `.map({` 再エンコード」を検出したら、`numeric_only=True`・
  `.map` なしで再計算し、目的変数自身と id/index を除いた相関絶対値最大の**native numeric**列を1つ
  答えるよう1回だけ差し戻す。content-blind（列名を注入しない）・EV-safe（棄権は却下しない）。
- 実装: `question_contract.validate_numeric_feature_correlation` + investigator 配線。

### (B) `RAG_RELEVANCE_STRICT` — 観点厳密の版差分列挙（idx9）
- **失敗**: 「案件遂行に関連する変更を挙げてください」に対し、版差分で見つけた変更（業務提言の
  再構成・ワークフロー説明の追記）を案件遂行関連と誤判定して列挙。gold は `該当なし`。
- **対策**: 「(観点)に関連する変更」＋版 cue の質問で、各変更が観点（担当割当・WBS・スケジュール
  日程・タスク定義・進捗管理・成果物の納品/確定 等の**実務遂行事項**）に直接関係するか厳密判定し、
  分析手法・モデリング手順・ワークフロー/フロー図説明・業務提言/示唆・章立て/文言整形は除外。
  根拠付けられる変更が無ければ `該当なし`、判断不能なら `わかりません` とするよう1回だけ差し戻す。
  content-blind・one-shot・EV-safe（棄権/既 `該当なし` は不発）。
- 実装: `question_contract.validate_relevance_enumeration` + investigator 配線。

## 3. 検証（focused/offline、gold100未実行）

両フラグ ON の focused 再実行（`scripts/sot2562_ng_recheck.py`）:

| idx | before | after | verdict |
| --- | --- | --- | --- |
| 4 | wrong `smoker` | **match `bmi`** | Perfect（24.7s） |
| 9 | wrong 過剰回答 | **match 該当なし**（「案件遂行に関連する変更はありませんでした」） | Acceptable（49.3s） |
| 14 | abstain | abstain（維持） | 誤答化なし（gold非空だが安全棄権のまま） |
| 63 | — | abstain（budget max_turns） | gate トリガ外（numeric）＝本変更と無関係 |

**結果: 6 NG → match 2 / safe-abstain 4 / wrong 0。** idx4・idx9 を正答化し、他の NG を誤答化していない。

### precision 非劣化
- 両ゲートの発火条件は極めて狭い regex トリガ（(A) 「数値特徴量」＋「相関」かつトレイルに `.map` corr、
  (B) 「(観点)に関連する変更」＋版 cue）。他の問いは新分岐に到達せず**構造的に不変**。
- idx14（同型・gold 非空）でも誤答化せず安全棄権を維持。
- champion: 両フラグ既定 OFF で新分岐は skip ＝ **byte-identical**。
- offline suite: `test_question_contract.py`（新規8件含む）/`test_investigator.py`/`test_gate.py`/
  `test_routing.py`/`test_tool_contract.py` = **207 passed**。py_compile OK / NUL 無し。

## 4. 索引再ビルド手順（参考）
本 issue は serve-time のみで index 影響なし（再ビルド不要）。索引を再生成する場合は
`.venv/bin/python -m src.rag.index`（全）/ `... -m src.rag.index.evidence_index`（typed のみ）。

## 5. 残課題
- 実 LB / gold100 全量での確定は親 SOT-2550 系の統合実行（SOT-2527 方式）へ委譲（本 issue は該当 idx の
  focused/offline 検証まで、gold100 未実行）。
- 両ゲートは候補フラグ構成での測定用。champion への昇格は封印 holdout の関門2ステップ（別 issue）。
