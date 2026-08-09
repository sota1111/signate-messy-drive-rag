# ハイライト色抽出ギャップの解消（条件付き書式 / 色ファミリ / ピボット祖先）— SOT-2564 追補

> 親: SOT-2550 §4「UNANSWERABLE(10) の多くは真の不能ではない — 書式抽出ギャップ」。
> 本追補は **human review=human 差し戻し**（「回答NGの項目を再確認し、合格するか検証、NGなら再対策」）への対応。
> **gold100 は実行していない**。該当 idx のみ focused/offline で決定論抽出を検証（回答・champion 不変）。

## 1. 差し戻しの再診断 — 「色ハイライトは既存経路で到達可能」は誤り

初回 PR #102 は文字装飾（太字/下線/イタリック, idx11）を `font_emphasis` で解消したが、**色ハイライト系
(idx2/15/47/65 ほか) は「既存 highlight_extract で到達可能・残差は routing/budget」と assert しただけで
実測していなかった**。実データで `highlight_extract` を直接叩くと、色系の主要 NG が抽出段で落ちていた:

| idx | 質問 | gold | 旧 highlight_extract 実測 | 真因 |
| --- | --- | --- | --- | --- |
| 25 | 白峰 train.xlsx の**青色**ハイライトの合計 | `-11851246` | `color=青` → **0件** | セル塗りは **水色(00B0F0)** に分類され、`青` フィルタが exact-match で落とす |
| 65 | 白峰 相関係数シートの**黄色**ハイライトの条件 | `相関係数が-0.99未満` | `color=黄` → **0件** | 黄色は**条件付き書式ルール**由来。openpyxl の `cell.fill` に出ず、solid-fill 走査が全部見落とす |
| 15 | 東都 train.xlsx の**黄色**ハイライトの抽出条件と集計内容 | `Gender=Male,target=2,Age=40-44,Country=Spain の個数` | セル値 `12` のみ（条件なし） | 階層ピボットの**祖先グループ**が面に出ず、条件を言語化できない |
| 25/65 | — | — | — | **routing gap**: `numeric` に分類され `canonical_route/compute` へ流れ、`highlight_extract` に到達しない |

加えて routing も gap: idx25/65 は `numeric` 契約で `highlight_extract` を first-move に含まず、抽出を直しても
モデルがツールを呼ばない限り証跡が出ない。

## 2. 実装（すべて `RAG_HIGHLIGHT_EXTRA` 既定 OFF で gate、champion byte-identical）

`src/rag/tools/highlight_extract.py`:

1. **条件付き書式ハイライト `_xlsx_cf_items`** — `ws.conditional_formatting` の各 `cellIs` ルールを
   1項目として surface。dxf fill(`bgColor`)→色名、`operator`/`formula`→人間可読な条件文（例
   `セルの値 < -0.99`）、および**ルールを満たす数値セル**（`matched_cells`/`matched_values`/`n_matched`）を
   evidence に出力。→ idx65 は `kind='conditional_format'` の `condition` がそのまま gold。
2. **色ファミリ照合** — `_COLOR_FAMILY={青:{青,水色,紺}, …}`。`青` フィルタが `水色` セルにも一致
   （`_color_matches(..., family=True)`）。→ idx25 は水色3セル `-11850477.4 + -669.37 + -99.06 = -11851246`（gold）。
3. **ピボット祖先コンテキスト `_pivot_ancestors`** — ハイライトセルの左各列について、同行以上で直近の非空値を
   行1ヘッダをキーに `evidence.group` へ付与。密表は同一行で確定するため安価、疎ピボットのみ上方走査。
   → idx15 は `group={Gender:Male, target:2, Age:40-44, Country:Spain}`（gold の抽出条件）+ 値 `12`。

`src/rag/agent/routing.py`:

4. **highlight 配線** — `RAG_HIGHLIGHT_EXTRA` ON かつ質問がハイライト色を参照するとき、契約に依らず
   `highlight_extract` を first-move 先頭へ（`font_emphasis`＝文字装飾ルートが優先）。プロンプトにも
   「`conditional_format` の condition を条件回答に、セル値/`n_matched` を合計/個数に、青は水色も対象」を追記。
   → idx25/65（numeric）も highlight_extract に到達。

OFF のとき CF 追加なし・`group` キーなし・色は exact-match のみ・routing は従来通り＝**byte-identical**。

## 3. focused / offline 検証（`RAG_HIGHLIGHT_EXTRA=1`, `.venv`, gold100 未実行）

| idx | 種別 | 本実装の抽出/配線結果 | 判定 |
| --- | --- | --- | --- |
| **2** | オレンジ行タスク名 | col-D オレンジ = `['プロジェクトキックオフ実施','中間報告会実施','最終報告会実施']` = gold | **PASS** |
| **11** | 太字∧下線∧イタリック | `font_emphasis` → `4,675,000円`（PR #102、不変） | **PASS** |
| **15** | 黄ピボット条件 | `group={Gender:Male,target:2,Age:40-44,Country:Spain}` + 値`12` = gold | **PASS** |
| **25** | 青ハイライト合計 | 水色3セル和 `-11851246` = gold（色ファミリ）＋ numeric→highlight_extract 配線 | **PASS** |
| **65** | 黄条件付き書式の条件 | Sheet3 `conditional_format` condition=`セルの値 < -0.99`（色=黄, n_matched=8）= gold ＋配線 | **PASS** |
| 97 | 交差する2つの黄セル | 黄列R(18)全体 ∩ 黄行79/138 → 交点 R79=10096, R138=10368, 差の絶対値=`272`=gold。行/列メタは面に出るが「行∩列」の特定はモデル推論に依存 | 証跡到達（推論要） |
| 47 | 黄セルが指す不動産の建設年 | 黄セル `B22`(誤差値) は面に出るが、gold `1899年` は予測→対象不動産→建設年の多段導出が必要（derived_calc, §4の深掘り不足系に隣接） | **抽出範囲外**（本 issue 非対象と明記） |

- **precision 非劣化**: `RAG_HIGHLIGHT_EXTRA` 既定 OFF で highlight_extract の contract・routing すべて
  byte-identical（単体テストで確認）。既存 solid-fill/色フィルタ経路は無変更。
- **深掘り不足系（idx29/30）・多段導出（idx47）は対象外**（本 issue は書式・色ハイライト抽出に集中）。

## 4. テスト

- `tests/test_highlight_extract_extra.py`（4件）: CF surfacing（ON のみ, 黄フィルタで <-0.99 のみ・赤除外・
  満たすセル）、色ファミリ（OFF で青→0 / ON で青→水色）、ピボット祖先（OFF なし / ON で group）、
  **flag OFF の byte-identical**。
- `tests/test_routing.py`（+2件）: numeric ハイライト質問が ON で highlight_extract 先頭・OFF で
  canonical_route 先頭・hint に conditional_format、文字装飾は font_emphasis 優先。
- 全 offline suite **971 passed**（回帰なし。既知の openpyxl WMF warning のみ）。

## 5. champion への含意

`RAG_HIGHLIGHT_EXTRA` 既定 OFF ＝ champion serve は byte-identical。ON で色ハイライト依存問（条件付き書式・
同系色・ピボット条件）の見かけ上 UNANSWERABLE を回答へ転じる。文字装飾経路（`RAG_FONT_EMPHASIS` /
`font_emphasis`, PR #102）とは直交・非干渉。
