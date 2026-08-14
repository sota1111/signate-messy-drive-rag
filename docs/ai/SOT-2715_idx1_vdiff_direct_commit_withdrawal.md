# SOT-2715 — 最終報告版差分 idx1 の gold構造一致 direct-commit：撤退記録（honest）

- 親: SOT-2708（cycle11）／根拠: `docs/ai/60_worker_codex_report.idx1.md`（codex-sol reasoning=max・確信度 high）
- 対象 idx1: 「恒一会 かえで総合病院の最終報告書 old版と最新版の実質的変更」（archetype=version_diff）
- 結論: **撤退（rejected）**。gold と同構造の1文を **diff_store の構造化レコードから非ハードコードで生成しても
  judge が Perfect/Acceptable にしない**ことを実 judge で確定。SOT-2706 の honest 断念を、より強い実測根拠で再確認。

## gold（`artifacts/predictions_test_v3_final.csv` idx1）

```
最終報告スライド7の、中間段階と最終モデルの性能比較表（AUC-ROC/F1-macro/Accuracyの中間実測値と最終値）を
削除し、改善幅のみを示す1行要約に置換した
```

## diff_store が保持する構造化レコード（質問非依存 build）

`src/rag/diffpair.collapsed_table_frames` がスライド7の消滅比較表について返す構造（実物由来・gold非参照）:

| フィールド | 値 |
| --- | --- |
| structural_location | `スライド7` |
| title | `最終モデル性能指標と中間段階との比較` |
| columns（ヘッダ行値・モデル註記除去） | `中間` / `最終` / `改善幅` |
| metrics（行ラベル） | `AUC-ROC` / `F1-macro` / `Accuracy` |
| header_label | `指標` |
| rank0 change（`_collapse_deleted_tables`） | before=`指標の比較表（AUC-ROC・F1-macro・Accuracy）` / after=`改善幅のみを示す1行要約に置換` |

`doc_kind`（`最終報告`）と `structural_location`（`スライド7`）から主語は導出可。`性能` も `title` から導出可。
**しかし gold の弁別トークン `中間実測値と最終値`（中間→「実測値」/ 最終→「値」の非対称接尾）は構造化レコードの
どこにも存在しない**（列名は素の `中間`/`最終`）。これを出すには gold 文字列のハードコードが必要になり、
受け入れ条件「gold値ハードコードなし（diff_store 構造レコードから生成）」に反する。

## 実 judge 較正（`scoring.crag` = 公式 evaluator.py rubric・strict addendum・codex/GPT-5.x）

各候補を idx1 gold と対で採点（独立サンプル、`votes=1` 反復）。判定は安定（ノイズではない）。

| 候補（`最終報告スライド7の、…改善幅のみを示す1行要約に置換した`） | 判定 |
| --- | --- |
| gold逐語（`…中間段階と最終モデルの性能比較表（…の中間実測値と最終値）…`） | **Perfect** ×5 |
| 主語省略（`…性能比較表（AUC-ROC/F1-macro/Accuracyの中間実測値と最終値）…`） | **Perfect** ×5 |
| `中間実測値と最終値`→`中間値と最終値`（実測を落とす） | Incorrect ×5 |
| `中間実測値と最終値`→`中間実測値と最終実測値`（対称・実測値） | Incorrect ×3 |
| `中間実測値と最終値`→`中間実測値・最終実測値`（・区切り） | Incorrect ×3 |
| `中間実測値と最終値`→`中間・最終の実測値` | Incorrect ×3 |
| `中間実測値と最終値`→`中間と最終の値`（列名の素直な naturalize） | Incorrect ×3 |
| metrics のみ（値節を省く：`性能比較表（AUC-ROC/F1-macro/Accuracy）`） | Incorrect ×3 |
| `性能比較表`→`指標比較表`（header_label由来）、値節は gold同一 | Incorrect ×3 |
| 現行 Sonnet 相当の簡潔回答（`性能比較表を削除した`） | Incorrect ×3 |

### 読み取れる境界

1. **`性能比較表` が必須**（`指標比較表` は他が gold 同一でも Incorrect）。`性能` は title から導出可。
2. **`中間実測値と最終値` を逐語で要求**。対称化（`最終実測値`）・区切り変更（`・`）・素直な列名 naturalize
   （`中間と最終の値`）・metrics のみ、いずれも Incorrect。判定は 3〜5 サンプルで安定。
3. `中間実測値`/`最終値` の非対称接尾（中間だけ「実測値」）は gold 著者の語彙であり、
   構造化レコード（列名 `中間`/`最終`）から原理的に導出できない ⇒ 通す唯一の道は gold ハードコード。

## 撤退判断

- 受け入れ条件①（idx1 Perfect/Acceptable・番兵10/10）: gold ハードコードなしでは**達成不能**。
- 受け入れ条件②（gold値ハードコードなし／OFF時 byte-identical）: 通す唯一の道が②違反。**②を守り撤退**。
- 受け入れ条件③（改善不成立時は撤退判断を根拠つき記録）: **本ドキュメント＋ledger で充足**。
- serve パスは無改変（byte-identical）。`vdiff_direct_lane.resolve` は idx1 で従来どおり `_summary_commit`
  非発火（rank0.summary=None）→ 従来 LLM ループへ委譲のまま。将来サイクルが `RAG_VDIFF_DC_TABLEREPL`
  的な削除表→置換クラスを再挑戦しないよう、`src/rag/agent/vdiff_direct_lane.py` にガードコメントを残した。

将来の再挑戦条件（新証拠）: (a) judge rubric/gold が緩和され値節の言い換えを許容する、または
(b) gold 語彙 `実測値` が構造化抽出で正当に得られる根拠が現れた場合のみ。それ以外は再試行しない。
