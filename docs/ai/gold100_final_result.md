# Gold-100 統合実測（Step11–13 全改善後の唯一の実測） — SOT-2527

- 実行日時: 2026-08-08T23:09Z（1回のみ）
- コマンド: `python -m scoring.gold_offline --run --workers 8`（gen=investigator, n=100）
- index: `python -m src.rag.index` で再ビルド済み（3026 chunks / 事前処理ストア反映）
- レポート: `artifacts/gold100_final_sot2527.json`、履歴: `docs/gold_offline_history.jsonl`
- コスト: **$7.14**（開始基準 $5.93 比 +$1.21 ≈ +20%）

## 統合候補フラグ構成（Step11–13 opt-in を全て ON）

Step12/13 の改善は champion serve path を byte-identical に保つため **既定 OFF の opt-in**。
「総合効果」を測るため以下を全て有効化して実行した（index も同構成で再ビルド）。

```
RAG_FIRST_MOVE_ROUTING=1   # SOT-2521 契約型ファーストムーブ決定論ルーティング
RAG_SPIN_DETECTION=1       # SOT-2522 空回り検出・早期打ち切り
RAG_ADAPTIVE_BUDGET=1      # SOT-2523 契約型別予算適応（既定ON）
RAG_EVIDENCE_CACHE=1       # SOT-2523 問内証拠キャッシュ（既定ON）
RAG_BUDGET_BOUNDARY_RESEARCH=1  # SOT-2524 予算切れ直前の義務駆動再探索（既定ON）
RAG_UNANSWERABLE_FALLBACK=1     # SOT-2525 UNANSWERABLE 前の決定論ツールfallback
RAG_PDF_OCR=1             # SOT-2526 画像のみPDFのGemini-vision OCR fallback（index/serve両方）
RAG_SHARE_CORPUS_PROFILE=1  # SOT-2528 corpus_profile をrun全体で永続化 / SOT-2529 事前復号キャッシュを消費
RAG_CANONICAL_MANIFEST=1    # SOT-2530 canonical-route O(1) マニフェスト
RAG_EVIDENCE_INDEX=1        # SOT-2531/2532 逆引き証拠索引 + file_grep 配線
RAG_STRUCTURE_STORE=1       # SOT-2533 構造事前ストア
```

- SOT-2509（参照選択の決定論化・version_diff 必須化）はフラグ無し＝常時ON。
- 事前処理ストアは再ビルドで生成: evidence_index 248,914 entries/382 files、structure_store 42 files、
  canonical_manifest 11 projects/171 files、corpus_profile 暗号化2件を事前復号。

## 結果サマリ（開始基準との差分）

| 指標 | 開始基準(08-08 03:03) | 統合(08-08 23:09) | 差分 |
| --- | --- | --- | --- |
| match | 21 | **27** | **+6** ✅ |
| abstain | 73 | **59** | **−14** ✅ |
| wrong | 6 | **14** | **+8** ❌（非劣化ゲート未達）|
| net(match−wrong) | 15 | **13** | **−2** ❌ |
| cost | $5.93 | $7.14 | +$1.21 |

- **棄権削減は成功**（abstain −14, non-match の 80.8% は安全棄権を維持）。
- **precision は後退**（wrong 6→14）。棄権→回答への変換の内訳が match より wrong に多く振れ、
  SIGNATE 実効スコア（correct +1 / abstain 0 / incorrect −1）の近似 net は **15→13 と低下**。

## 棄権の状態コード別残数（59件）

| コード | 開始基準 | 統合 | 差分 |
| --- | --- | --- | --- |
| BUDGET_EXHAUSTED | 54 | **38** | **−16** ✅（Step12 予算適応が主効果）|
| UNANSWERABLE | – | 10 | – |
| NOT_RETRIEVED | – | 6 | – |
| PARSED_AMBIGUOUS | – | 3 | – |
| EVIDENCE_INCOMPLETE | – | 1 | – |
| SPIN_CUTOFF | – | 1 | – |
| **合計** | 73 | **59** | −14 |

BUDGET_EXHAUSTED の残38件の archetype: derived_calculation 13 / fact_lookup 9 / document_extract 7 /
enum_set 6 / version_diff 3。**残る棄権の主軸は依然 BUDGET_EXHAUSTED（38/59=64%）と UNANSWERABLE(10)**。

## wrong 14件の内訳と原因帰属（非劣化未達の分析）

| idx | type | 判定所見 |
| --- | --- | --- |
| 62 | fact_lookup | **judge誤判定濃厚**: pred「n_estimators 1位=500/2位=300」= gold と実質一致なのに Incorrect |
| 85 | document_extract | 境界: pred「全項目達成＝未達なし」 vs gold「該当なし」＝意味は近いが不一致判定 |
| 88 | document_extract | 境界: pred が gold の上位集合（言い換え/過剰列挙） |
| 93 | document_extract | 境界: pred「前処理パイプライン」＝gold の頭部のみで truncated |
| 78 | fact_lookup | **本来棄権すべき回答化**: ACTH規定を発見できず一般規定で代替回答 |
| 80 | document_extract | **本来棄権すべき回答化**: Sheet2を空と誤断定して回答 |
| 47 | derived_calculation | 誤値 2012 vs 1899年 |
| 63 | derived_calculation | 誤値 −30.78416 vs 0.15002 |
| 97 | derived_calculation | 誤値 18948 vs 272 |
| 29 | fact_lookup | 範囲ずれ (6.102138,6.303138] vs 6.088138〜6.288138 |
| 69 | fact_lookup | 範囲ずれ 第5〜7週 vs 第5〜6週 |
| 70 | document_extract | ラベル誤り「03」vs「AI-05」 |
| 75 | fact_lookup | 競合記載で特定不可と回答（gold=第4週） |
| 84 | fact_lookup | ページ誤り 6 vs 5 |

**原因帰属**: wrong +8 の主因は 2種類。
1. **棄権→回答変換のprecisionコスト（設計どおりの副作用）**: Step12 の回答寄せ
   （SOT-2524 予算境界の義務駆動再探索・既定ON / SOT-2525 UNANSWERABLE前fallback / SOT-2521 first-move）
   が、従来 BUDGET/UNANSWERABLE で安全棄権していた限界問（idx78/80 等）を回答に転じ、その内訳が
   match より wrong に不利に振れた。SIGNATE は wrong=−1・abstain=0 のため、この変換は
   **P(correct) が閾値以上のときだけ行うべき**で、現状の変換ゲートは EV 較正が不足。
2. **judge ノイズ**: idx62 は明確な false-Incorrect、idx85/88/93 も境界。ローカル codex/gemini judge の
   厳格一致が wrong を 3〜4件 過大計上している可能性（実 SIGNATE 採点では緩和され得る）。

## 関門2 非劣化ゲート（SOT-2478）の状態

- 関門2 は「封印 champion に対し非劣化のときのみ採用」。本統合候補は **wrong 6→14・net 15→13 と後退**の
  ため **関門2 非劣化ゲートは FAIL（不採用）**。champion serve path（全フラグ既定OFF）は byte-identical の
  ままで、本測定は候補構成の効果確認のみ（champion 差し替えは行わない）。

## 次に残る棄権/誤答クラスと推奨アクション（follow-up）

1. **precision 回復が最優先**: 棄権→回答変換に **EV/信頼度ゲート**（P(correct)×(+1) > 0 すなわち
   P(correct) > 0.5 相当の閾値未満なら棄権維持）を挟み、SOT-2524/2525 の回答寄せを EV 安全化する。
   ＝「wrong を増やさずに match だけ拾う」ための per-contract 変換較正（SOT-2503 のスライス較正を変換側にも適用）。
2. **残 BUDGET_EXHAUSTED 38件**（derived_calculation/fact_lookup 主）: 予算適応後も残る導出系は
   探索律速でなく compute/parse 律速の可能性。exec_verifier（SOT-2501）到達率の底上げが軸。
3. **judge 忠実度**: idx62 型の false-Incorrect を減らすため judge 較正（SOT-2503/judge_fidelity）の見直し。

いずれも本測定 issue の範囲外（測定のみ）。上記1を筆頭に別 issue 化を推奨。
