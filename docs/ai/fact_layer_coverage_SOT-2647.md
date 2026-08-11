# 事前計算事実層 配線カバレッジ (SOT-2647, 事前計算事実層 5/5)

champion (Wave A net40) の abstain 46 のうち **hard core 16問**（`champion_abstain_wrong_classification.md`）を、
実行時探索から**事前計算ストアへの lookup** に置き換えて回収するための配線 issue。本書は 4 ストアの
**カバレッジ**と、各 hard core idx が「本 issue で発火対象か / カバレッジ外か」を honest に記録する
（無理な発火をしない — 曖昧・非充足は従来経路へフォールバック）。

配線形態は 2 つ（`src/rag/agent/fact_layer.py`、フラグ `RAG_FACT_LAYER` 既定 OFF・OFF時 byte-identical）:

- **(a) 決定論直答レーン** `fact_layer.resolve(question, contract)` — 投機せず、**ストアに確定値が一意に束縛
  できるときのみ** `{value, evidence, method}` を返して LLM ループを短絡。曖昧なら `None`（フォールバック）。
- **(b) investigator ツール** `case_filter` / `id_lookup` / `metric_lookup` / `diff_lookup` — LLM ループが
  file_grep 反復の代わりに 1 呼びでストア事実を引く。`build_tools` 経由で **MCP にも自動公開（単一情報源）**。
- **commit_gate 連携**: 4 ツールは出典付き値を返すため commit_gate の numeric grounding 集合に加え、
  ストア由来値を「検証済み operand」として COMMIT を通す（`src/rag/agent/commit_gate.py::_NUMERIC_TOOLS`）。

## hard core 16 × ストア対応（発火対象 / カバレッジ外）

型分布: enum 5 (32,38,45,67,87) / derived 5 (40,50,57,63,97) / fact 3 (39,48,98) / doc_extract 2 (55,82) / version 1 (22)

| idx | 型 | ストア | 本issueでの扱い | 根拠（各ストアのビルド報告） |
|---|---|---|---|---|
| 38 | enum | case_master | **ツールで発火**（レーンは composite ゆえ defer） | apr_code+contract_amount 10/10。gold=「該当なし」（APR-M3 該当 0件）→ case_filter で母集団確定し 0件を確証 |
| 32 | enum | — | カバレッジ外 | metrics.json 交互作用特徴量列名（分析コード出力依存, 別ストア領域） |
| 45 | enum | — | カバレッジ外 | 会議録PDFの M2→M3 完了アクションID（会議録依存） |
| 67 | enum | case_master | カバレッジ外（部分） | proposal/fr 金額 1-2/10 のみ（テキスト非記載の実コーパス限界）→ 発火せず defer |
| 87 | enum | case_master | **ツールで発火**（レーンは multi-predicate ゆえ defer） | status/apr_code 10/10・train_rows 9/10。完了×APR-M1×≥10000行 の複合条件は case_filter+LLM で合成 |
| 57 | derived | derived_metrics | **ツールで発火**（レーンは案件名「青葉のTX」→会社名 非束縛ゆえ defer） | model.threshold_sweep.best_f1=0.42395962（gold 0.42396 一致）。設問は system 名参照ゆえ決定論レーンは束縛不可→metric_lookup+LLM |
| 63 | derived | derived_metrics | **決定論レーンで発火（モデル不変）** | model.prediction_id0.prediction=0.15001822（gold 0.15002 一致）。設問が会社名明示ゆえ一意束縛 |
| 40 | derived | — | カバレッジ外 | 案件横断の精算スケジュール集計（単一 train 表の列統計でない, settlement 層領域） |
| 50 | derived | derived_metrics | カバレッジ外（部分） | 対象値が report PDF 内の Salary.com 表で train 表の数値列でない→percentiles 非在→発火せず |
| 97 | derived | — | カバレッジ外 | 黄色ハイライト交差セル差（structure_store 領域） |
| 39 | fact | id_master | カバレッジ外 | グラフ→元カラムは ID 体系でなく chart numCache/structure_store 領域（id_lookup_shaped=false） |
| 48 | fact | id_master | カバレッジ外 | 価格帯×税率は数表で `[A-Za-z]-\d` の ID 体系でない（PDF本文読解領域） |
| 98 | fact | id_master / case_master | カバレッジ外 | RATE は数字を持たない param 名で ID パターン外（evidence_index param 型領域） |
| 55 | doc_extract | — | カバレッジ外 | 事前計算ストア対象外（doc_extract は既存 Wave B1 の領域） |
| 82 | doc_extract | — | カバレッジ外 | 同上 |
| 22 | version | diff_store | カバレッジ外 | `.ipynb` 版ペアは決定論 office diff (pptx/docx/xlsx) の対象外（notebook diff は別レーン） |

### B クラス version（誤答のみ, 回答化するが WRONG）— diff_store ツールで再挑戦

| idx | ストア | 扱い | 根拠 |
|---|---|---|---|
| 1 | diff_store | ツール（diff_lookup） | 恒一会 最終報告_old→最新（registry-family 補完ペア, alignment_ok） |
| 14 | diff_store | ツール（diff_lookup） | 青葉与信 提案書_v1→_v3（rev-suffix, schema_name_change 属性で列名修正を分類） |
| 95 | diff_store | ツール（diff_lookup） | 青嶺 スケジュール_r1→_r2、`exclude_attributes=status_transition` で「未着手→完了」除外を決定論適用 |

## 決定論レーンの発火境界（precision-first, SOT-2601 の教訓）

- **derived スカラレーン**: `numeric` 契約で、(1) 質問がストア案件を一意束縛（会社名 substring, 曖昧×→defer）、
  (2) 単一スカラアンカ（`f1`+`閾値/最大化` → best_f1 / `予測`+`id=0` → prediction_id0）が**厳密に 1 つ**、
  (3) その値が非 null、のときのみ発火。idx63 が該当（idx57 は「青葉のTX」= system 名参照 & 「青葉」が
  2案件（青葉与信/青葉バイオ）で曖昧ゆえ defer）。
- **enum 略称レーン**: `full_enumeration` で、単一 APR-Mx コード + 列挙キュー + **集計キュー無し** +
  **APR 以外の属性述語が皆無**（完了/行数/金額 等の追加条件があれば defer）+ apr_code 全案件充足、のときのみ発火。
  idx38（合計 = composite）/ idx87（完了×行数 = multi-predicate）はいずれも defer → ツール経路。

**設計上、決定論レーンで確実に発火するのは idx63（モデル不変）**。他の hard core は主に (b) ツール経路で
LLM が正しい述語を組んでストアを引く（file_grep 反復＝BUDGET_EXHAUSTED を回避）ことで回収する。
`RAG_FACT_LAYER` 既定 OFF ゆえ champion 提出経路は byte-identical（本層は opt-in レバー・champion リスク 0）。
