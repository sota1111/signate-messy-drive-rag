# SOT-2563 — in-time abstain root cause (review=human follow-up)

## Question (human, review=human)

> 時間内には処理できたものの、最終的に十分な証拠を確保できず abstain になった理由を調査してください。
> 回答前の事前処理で解決できないか確認してください。

PR#101 (per-call file_grep deadline) と PR#103 (`RAG_FILE_GREP_RESERVE_S` によるルート切替予約)
で **timeout 棄権は 0 に** なった。残ったのは「**時間内に処理し切ったのに証拠不足で abstain**」した
問い。本ドキュメントはその真因を実トレースで特定し、「回答前の事前処理で解決できるか」に証拠付きで
答える。gold100 は実行していない（Issue 指示どおり focused/offline のみ）。

## 対象（PR#103 cycle2 の in-time abstain 15件 + wrong 1件）

pattern A/B（`docs/ai/abstain_wrong_root_cause_SOT-2550.md` §3）から timeout が消えた後の残 NG：

- **max_turns 到達群（8）**: idx 38 / 39 / 40 / 55 / 63 / 67 / 70 / 73
- **answered だが「わかりません」群（7）**: idx 27 / 37 / 48 / 50 / 53 / 76 / 98
- （参考）wrong 1: idx 28（`Age`→gold `BMI`）

計測条件は cycle2 = **本番 investigator（Gemini-only）+ 事前処理フラグ ON**
（`RAG_EVIDENCE_INDEX=1`, `RAG_FILE_GREP_INDEX_CANDIDATES=1`, `RAG_FILE_GREP_MAX_SCAN_S=180`,
`RAG_FILE_GREP_RESERVE_S=30`）。artifacts: `sot2563_focused_answers_cycle2.json/.details.jsonl`。

## 真因 — リトリーバルのターン浪費（retrieval-turn starvation）

file_grep デッドライン（本子の主対策）は**ボトルネックではない**：15件とも `timeout` は消え、
最大 148s で時間内に完了している。真因は **反復（ターン）予算の枯渇** で、その予算が
**証拠の“在り処探し”に消費されて推論・計算へ届かない** ことにある。

15件の tool ターン内訳（`submit_answer` を除く、cycle2 実トレース集計）:

| 分類 | tool ターン数 |
| --- | ---: |
| リトリーバル（find_files / file_grep / canonical_route / read_office / caption_image / read_chart_values） | **182 / 202 = 90%** |
| 推論・計算（compute / corpus_aggregate / version_diff / pdf_emphasis） | 20 / 202 = 10% |

- 適応予算は既に 12→18 turn（`ADAPTIVE_MAX_TURNS=18`, SOT-2523）へ引き上げ済。spin 検出
  (SOT-2522)・境界再探索 (SOT-2524) も配線済。それでも **探索だけで 18 ターンを使い切る** ため、
  多段の集約/列挙/数値導出に入る前に予算が尽きて abstain する。
- `max_turns` 群は文字通り探索で打ち切り、`answered/わかりません` 群も探索し切って「証拠不足」と
  自己判断して棄権している（例 idx48 evidence=「PDF…を読み取れません」、idx98=「TM案件…見つからない」）。

### 現行 main での再現確認（verified, pre-processing ON）

現行 main（SOT-2562 PR#105 / SOT-2564 PR#104 マージ後、既定フラグ）で代表 3 idx を pre-processing ON
再計測（`sot2563_preproc_confirm.json`）:

| idx | bucket | stop | elapsed | retrieval/total turns | 結果 |
| ---: | --- | --- | ---: | --- | --- |
| 48 | 抽出欠損(PDF) | answered | 49.0s | 10/11 | abstain（PDF読取不可を再現） |
| 40 | 集約多段 | answered | 55.3s | 3/12（compute 主）| abstain |
| 73 | 探索spin多段 | max_turns | 64.2s | 17/18 | abstain（探索で予算枯渇を再現） |

timeout=0 を維持したまま 3/3 が abstain を再現。**現行 main + 事前処理 ON でも回収されない**。

## 「回答前の事前処理で解決できるか」への回答

**汎用の index 事前処理“だけ”では解決しない（確証済）。** SOT-2562 の
`evidence_index` / `index_candidates` を **ON にした状態で計測しても** 15件は 1 件も match 化せず、
現行 main でも同様。理由は、これらの問いが指す **固有エンティティ/文書に索引カバレッジが届いていない**
ため、探索が依然 90% のターンを溶かす。

一方で、**真因（探索でターンを溶かす）に直接効く事前処理は有効な軸** である。問い文には対象が
固有名で明示されている（例: 恒一会かえで総合病院 / TOTO の FR書 / AOMINE / 青嶺不動産アセット
マネジメント / 青葉与信マネジメント …）。**回答ループ開始前に、この固有名 → canonical 文書を事前解決し、
該当スライス（本文/表/実装設定）を席次へ注入**すれば、探索を ~10 ターンから ~1–2 ターンへ圧縮でき、
空いた予算が推論へ回る。これは SOT-2494（canonical 直行）/ SOT-2498（contract→tool ヒント配線）の
延長で、既存の canonical_route を「モデルが選ぶツール」から「回答前の事前解決ステップ」へ前倒しする
もの。

ただし限界も明確:

- **純粋な多段数値/集約が律速の問い**（idx37 差÷差、idx63 回帰予測小数第5位、idx76 what-if 計算、
  idx40/55/67 集約・列挙）は、証拠が席次にあっても **推論深さ（compute の連鎖）** が予算を要するため、
  事前処理単独では届かない。ここは reasoning-budget（compute 専用の増枠 or 決定論的集約ツール化）が
  別軸。
- **idx48（PDF 読取不可）** は事前処理でもツール層の抽出能力欠損（当該 PDF が全ツールで読めない）で
  あり、抽出器の対応（pre-decode / OCR 経路）が必要。

## 結論（axis verdict）

- 本子（file_grep runtime deadline）の範囲は **達成済**（timeout 棄権 0）。残 NG は本子スコープ外の
  **retrieval-turn starvation** が真因で、これは **回答前の“固有エンティティ事前解決＋スライス注入”
  事前処理**（SOT-2562 系のカバレッジ拡張＝索引でなく canonical 前倒し）で回収を狙うのが正しい。
- 汎用 index 事前処理は既に ON でも不十分＝**カバレッジ（どの文書を事前に解決するか）が鍵**。
- 純粋多段数値/集約は reasoning-budget 側の別軸。

→ 次の実装子（提案）: 「固有名→canonical 文書 pre-resolve + スライス pre-inject（既定 OFF flag、
champion byte-identical）」を focused/offline で idx 27/50/53/70/73 に対して A/B。純粋数値多段
（37/63/76）は reasoning-budget 子として分離。本 Issue は `review=human` のため、実装子起票の可否は
人間判断に委ねる。

## Artifacts

- `artifacts/sot2563_focused_answers_cycle2.json` / `.details.jsonl`（PR#103 実トレース、pre-processing ON）
- `artifacts/sot2563_preproc_confirm.json` / `.details.jsonl` / `.log`（現行 main 確認測定、3 idx）
- `scripts/sot2563_focused_answers.py`（focused/offline ハーネス、gold100 非実行）
