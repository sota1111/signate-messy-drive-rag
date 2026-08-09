# 書式(太字/下線/イタリック)抽出ギャップの解消 — SOT-2564

> 親: SOT-2550（誤答型別対策後の統合実測＋原因調査）§4「UNANSWERABLE(10) の多くは真の不能ではない — 書式/深掘り不足」の follow-up。
> **gold100 は実行していない**。該当 idx のみ focused/offline で検証（回答・champion 不変）。

## 1. 問題（root-cause §4 の再掲）

棄権60件中14件がハイライト/色/**太字/下線/イタリック**に言及する書式依存問。extract 層がセルの
色・**文字装飾**を安定して面に出せず、「根拠なし=UNANSWERABLE」または timeout に落ちていた。

- **色ハイライト**（黄/オレンジ/青…）は既に `highlight_extract`（SOT の既存ツール）が扱えていた
  （idx2 のオレンジ行・idx15/47/65 の黄色セル等は抽出層は到達可能で、残差は routing/budget 側）。
- **文字装飾（太字∧下線∧イタリック）は extract 層にまったく出ていなかった**。`highlight_extract` は
  色フィルタのみ、`office.py` の extract 面は docx の `【太字箇所】` を出すだけで、
  **「太字かつ下線かつイタリックに同時該当する箇所」を一意化する経路が存在しなかった**
  → idx11「太字、下線、イタリックのすべてに該当する箇所」= gold `4,675,000円` が見かけ上 UNANSWERABLE。

## 2. 実装（本 issue の中心 = 文字装飾の extract 面出力＋契約配線）

すべて **`RAG_FONT_EMPHASIS`（既定 OFF）** で gate。OFF のとき champion serve のツール集合・プロンプト・
`read_office` 面は byte-identical（`RAG_XLSX_EMBEDDED_IMAGE` / `RAG_STRUCTURE_STORE` と同じ規約）。

- **新規ツール `src/rag/tools/font_emphasis.py`**：xlsx/xlsm/docx/pptx/pdf から **bold/underline/italic**
  を per-span に構造化して返す（contract `{value, evidence, method}`、method に `bold/underline/italic`
  の3真偽値）。`require='太字,下線,イタリック'`（英語 `bold+underline+italic` も可）で**指定書式すべてに
  同時該当**する箇所だけに絞る。
  - **pdf**: bold=フォント名に "bold"、italic=イタリックフォント **または** 疑似イタリック（テキスト
    行列シアー `|c/d|≥0.1`、`pdf_faux_italic._shear_ratio` を再利用）、underline=ベースライン直下の薄い
    矩形/線。隣接する同一 (bold,underline,italic) グリフを1スパンに結合（シアーで分割された数字列を
    `4,675,000円` の1件に復元）。
  - **xlsx**: `cell.font.bold/underline/italic`。**docx**: run（run/run.font）の3属性、段落内の連続同一
    装飾 run を結合。**pptx**: `run.font.*`。暗号化 Office は `highlight_extract` と同じ復号経路。
- **investigator 配線**（`build_generic_tools`）：`RAG_FONT_EMPHASIS` ON のときだけ `font_emphasis` ツールを
  追加登録（OFF ではツール集合不変＝byte-identical）。
- **契約ルーティング（SOT-2498）**（`routing.py`）：`FORMAT_CHECK` かつ質問が太字/下線/イタリックに
  言及するとき、初手ツールを `font_emphasis` 先頭に差し替え＋「色ハイライトではなく font_emphasis で
  要求書式すべてに同時該当する箇所だけを回答する」ヒントを付与（flag ON 時のみ＝champion プロンプト不変）。
- **read_office 面**（`office.py`）：flag ON のとき xlsx/docx/pptx の抽出末尾に
  `【書式強調(太字/下線/イタリック)のある箇所】 [太字∧下線∧イタリック] 値 (位置)` を additive 付与
  （best-effort・例外は無視、OFF で byte-identical）。

## 3. focused / offline 検証（gold100 未実行）

`RAG_FONT_EMPHASIS=1`、`.venv`、`PYTHONPATH=repo root`。google-genai は不使用（決定論抽出のみ）なので
sitecustomize timeout 注入は不要。

| idx | 質問の書式条件 | gold | 本実装の抽出結果 | 判定 |
| --- | --- | --- | --- | --- |
| **11** | 太字∧下線∧イタリックの箇所（報告資料） | `4,675,000円` | `font_emphasis(報告資料_2025-08-06.pdf, require='太字,下線,イタリック')` → **`['4,675,000円']`**（唯一）。同値が page5 にも平文で存在するが装飾なしで正しく除外。 | **PASS（見かけ上UNANSWERABLE→gold一致）** |
| 2 | オレンジ行のタスク名（既存の色経路） | プロジェクトキックオフ実施 等 | `highlight_extract(スケジュール_r2.xlsx, color='オレンジ')` がオレンジ行(row2…)を surface し「プロジェクトキックオフ実施」等を含む（**既存 highlight_extract は本 issue で不変**） | 抽出層は到達（色経路は既存＝非対象） |

- (a) 書式メタが evidence 面に出る：`font_emphasis` の method に `bold/underline/italic` の3真偽値と
  page/bbox（pdf）・cell（xlsx）等が出力される。read_office 面にも flag ON で `【書式強調…】` が出る。
- (b) 見かけ上の UNANSWERABLE→回答：idx11 は装飾条件で `4,675,000円` を一意抽出＝gold 一致に転じる。
- **precision 非劣化**：`RAG_FONT_EMPHASIS` 既定 OFF で read_office 面・investigator ツール集合・
  routing プロンプトすべて byte-identical（単体テストで確認）。既存 `highlight_extract` は無変更。
- **深掘り不足系（idx29 ヒストグラム3番目ビン・idx30 多段条件付き割合）は対象外**（本 issue は書式抽出に集中）。

## 4. テスト

`tests/test_font_emphasis.py`（10件, 全 offline/network-free）: require 正規化（同義語/区切り/未知語
fail-open）・flag 既定 OFF・xlsx/docx/pptx の検出と require フィルタ・docx 連続 run 結合・
unsupported 拡張子・**read_office 面が flag OFF で byte-identical / ON で `【書式強調…】` 付与**・
**investigator ツールが flag ON のみ登録**・**routing が flag ON のとき font_emphasis 先頭**。
全 offline suite **512 passed**（回帰なし）。

## 5. champion への含意

`RAG_FONT_EMPHASIS` 既定 OFF ＝ champion serve は byte-identical。ON にすると書式（文字装飾）依存問の
見かけ上 UNANSWERABLE を回答へ転じる新経路が有効化される。色ハイライト経路（highlight_extract）とは
直交・非干渉。棄権削減（時間予算）系の SOT-2562/2563 とも直交。
