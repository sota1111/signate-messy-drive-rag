# SOT-2619 — fact_lookup 書式同値誤判定 (idx62/75/78) と version_diff idx74 の focused trace

- 実施日: 2026-08-10
- 対象: SOT-2613 gold100 実測（gemini-3.6-flash / judge=codex / champion=Wave A net40）の残誤答のうち
  fact_lookup 3件 (idx62/75/78) と version_diff の唯一の決定論失点 (idx74)
- 判定確認: ローカル CRAG 判定器（`scoring/crag.py`）の deterministic 経路 + Sonnet(claude-cli) LLM 判定
  で各答案を実測（全量 gold100 は回していない）。判定器（judge）は一切変更していない。

## 誤答機序（値正誤 × 表現差分 × 落ちた判定箇所）

| idx | 型 | 値の正誤 | 表現差分 | 落ちた判定箇所 | 対応 |
| --- | --- | --- | --- | --- | --- |
| **74** | version_diff | **正** (`ビジネスアナリスト` / `藤田 彩`→`井上 里奈`。pipeline を focused 実行し確認) | LLM naturalizer が前置き＋箇条書き＋改行を付与（`案件遂行に関連する変更は以下の通りです。\n・…`）。gold は素の一文 | 判定器の deterministic `score_set`（`\n`/区切りで集合化）が verbose 多行を集合不一致とみなし **Incorrect** | **修正**: version_diff を決定論プロプロ整形へ（LLM 呼び出し撤去） |
| **75** | fact_lookup (LLM fallback) | **正** | `第4週目（第4週）` の冗長な自己言い換え括弧。gold は `第4週` | deterministic 判定 **Incorrect**（冗長差） | **修正**: normalizer に冗長自己 gloss 畳み込み追加 |
| **62** | fact_lookup (LLM fallback) | 主値は正（`n_estimators`, 300/500 を含む）だが **順位対応が欠落**（gold=`1位=500、2位=300`、答案=`300 と 500 の違い`） | 括弧内容が別物（順位割当 vs 一般説明）。純粋な表現差ではなく**内容の粒度不足** | deterministic 判定 **Incorrect** | **修正対象外**（意味保存 normalizer では順位対応を捏造できない＝gold ハードコードになる。生成側の課題として報告のみ） |
| **78** | fact_lookup (LLM fallback) | 事実は全て正（該当特別規定なし／T&M／25,000円税別／30分切上／月次／上限なし） | gold の簡潔要約に対し**過剰網羅**（`170時間` 等 gold にない具体値を追加） | 判定器の deterministic `strict_downgrade`（verbose かつ gold外 numeric → **Incorrect**）※判定器側の仕様、変更不可 | **修正対象外**（意味保存 normalizer は数値を落とせない＝事実を削れない。簡潔化は生成側の課題として報告のみ。なお champion=Wave A では idx78 は ABSTAIN=0点であり wrong ではない） |

fact_lookup の決定論パイプライン（`src/rag/agent/pipelines/fact_lookup.py`）は 2 つの tight recognizer
（schedule 順序読取 / 報告書ページ番号）だけを担当し、idx62/75/78 はいずれも **どちらにも合致せず LLM
fallback** に落ちる。したがって「決定論 formatting 層で落ちている」わけではなく、fallback 答案の表現の
問題である。idx75 のみが**意味保存 normalizer**（全経路共通の最終整形 `src/rag/normalize.py`）で安全に
畳み込める純粋な冗長差、idx62/78 は内容/冗長の問題で意味保存整形の範囲外。

## 修正（表現差由来の 2 件のみ契約化・gold値ハードコードなし）

1. **version_diff 決定論 naturalization**（`src/rag/agent/pipelines/version_diff.py`）
   - `value` を `「{行ラベル}：{旧} → {新}」`（arrow）から `「{行ラベル}が{旧}から{新}に変更」`（gold 書式の
     プロプロ一文）へ決定論レンダリング。`method['naturalize']` を **False** にし Stage3 の LLM 一発
     呼び出しを撤去 → 非決定な前置き/箇条書き混入を排除。トークンは全て corpus/diff 由来（ハードコード無）。
   - focused 判定: 旧答案 = **Incorrect** → 新答案 = **Acceptable/Perfect**（Sonnet）。
2. **冗長自己 gloss 畳み込み**（`src/rag/normalize.py`、全経路共通の意味保存整形）
   - `「HEAD（GLOSS）」` で GLOSS が（序数カウンタ `目` を除いて）HEAD の単なる言い換えのときだけ、単一
     canonical 形へ畳み込む（`第4週目（第4週）`→`第4週`）。序数の `目` は序数ノイズとして除去。
   - 集合ベースの意味保存 `_preserves_meaning_setwise`（重複値のみ落とせる／固有値・識別子は保持／新文字
     混入禁止）で gate。実情報を持つ括弧（`n_estimators（1位=500、2位=300）`, `田中（営業部）` 等）は不発。
   - focused 判定: `第4週目（第4週）` = **Incorrect** → `第4週` = **Perfect**。

## 非回帰・OFF byte-identical

- version_diff: `RAG_DET_PIPELINE_ROUTER` OFF で pipeline は不到達（既存 `test_resolve_off…` で担保）→
  champion serve path byte-identical。
- normalize: `RAG_ANSWER_NORMALIZE=0` で完全 bypass（`test_gloss_dedup_is_off_when_normalization_disabled`）。
  既定 ON でも畳み込みは純粋冗長 gloss のみ発火＝非該当答案は byte-identical（regression サンプルで確認）。
- テスト: `scoring/test_normalize.py` + `tests/test_version_diff_pipeline.py` = 39 passed（two-axis
  非回帰 proof `test_normalization_preserves_deterministic_correctness` を含む）。`tests/test_det_pipeline.py`
  / `test_routing.py` = 60 passed、`test_formatting.py`/`test_format_events.py`/`test_sot2618_b1_composition.py`
  も my-change 起因の失敗なし（`test_wiring_formats_det_answer_none_form` は main でも失敗する既存事象で本
  修正と無関係）。
- 全量 gold100 は回していない（issue 指示どおり）。
