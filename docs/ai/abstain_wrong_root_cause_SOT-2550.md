# 棄権(abstain=60) と 誤答(wrong=6) の原因調査 — SOT-2550

> 対象実測: 誤答型別対策 A1–E 完了後の統合 gold100（2026-08-09 03:40–04:15Z / gen=investigator / n=100）
> 結果: **match 34 / abstain 60 / wrong 6 / net 28**（`docs/ai/gold100_after_error_fixes.md`）
> データ源: `artifacts/gold_100_review.csv`・`artifacts/predictions_test_investigator.details.jsonl`・
> `artifacts/abstain_ledger.jsonl`（recorded_at=2026-08-09T04*）。本書は**測定済み実行ログの事後分析**であり、
> 新たな gold100 は回していない（回答・champion 不変）。

人間指示（Linear コメント）:「obstain や wrong の原因を調査してください」への回答。

---

## 1. 結論（要旨）

- **棄権60件の主因は「1本の遅い `file_grep` 全コーパス走査」**。棄権の **73%（44/60）が `BUDGET_EXHAUSTED`**、
  そのうち **40/44 が `file_grep` を含む**。棄権問の所要時間は **中央値 308s・最大 658s** に対し、
  正答(match)問は **中央値 22s・最大 125s**。＝**遅い探索が予算(時間)を食い尽くして棄権**しており、
  「原理的に答えられない棄権」ではない（後述）。
- **時間予算はターン間でしか判定されない**（`investigator.py:1291` `if clock()-start > timeout_s`）。
  同期呼び出しの `file_grep` は**呼び出し途中で中断できない**ため、公称 180/240s の予算に対し
  **単一呼び出しが 280–658s（1.5–3.6倍）暴走**し、答えを得る前に問答が終了する（→ `timeout` → 棄権）。
- `file_grep` は evidence_index の**ヒット時のみ**高速返却（`files_scanned=0`）。**index ミス時は全ファイルを
  `_extract`（Office復号+PDF/OCR含む）する全走査にフォールバック**する（`file_grep.py:164-218`）。
  発見系クエリ・書式(ハイライト/色/太字)依存クエリは index を外し、この全走査に落ちる。
- **誤答6件は「棄権→回答」変換の EV 較正不足**。Step12/13 の回答増フラグが従来は安全棄権していた難問を
  回答に転じ、その一部が **確信度1.00 のまま対象取り違え/半端回答**として wrong に振れた（別集合 `{4,9,14,49,83,92}`）。

---

## 2. 棄権60件の状態コード内訳

| state_code | 件数 | 実体 |
| --- | ---: | --- |
| **BUDGET_EXHAUSTED** | **44** | 反復/時間の予算切れ。**実体は時間切れ**（下表）。＝**回収可能**な棄権 |
| UNANSWERABLE | 10 | 「根拠なし」判定。但し gold は存在＝**書式/深掘り不足の見かけ上のUNANSWERABLE**が多数 |
| EVIDENCE_INCOMPLETE | 2 | 根拠が部分的で確定できず |
| NOT_RETRIEVED | 1 | retrieval miss |
| PARSED_AMBIGUOUS | 1 | 解析はしたが一意化できず |
| RETRIEVED_NOT_PARSED | 1 | 取得したが構造解析に失敗 |
| SPIN_CUTOFF | 1 | 空回り検出で打ち切り |

### BUDGET_EXHAUSTED(44) の実体 = 時間切れ（反復上限ではない）

`details.jsonl` の `stop_reason`:

| stop_reason | 件数 |
| --- | ---: |
| **timeout（時間切れ）** | **34** |
| max_turns（反復上限） | 8 |
| answered（「わかりません」で自己棄権） | 2 |

- **時間切れが支配的（34/44）**。反復上限(max_turns)は8件のみ。
  ＝Step12 診断（「反復上限12ターンで打ち切り」）から**制約が時間側にシフト**している。
- `file_grep` を含む BE: **40/44**。BE 所要 min=34s / median=308s / **max=658s**、
  **>180s が 35件・>300s が 24件**。（比較: match 問は median 22s / max 125s。）

---

## 3. 棄権の2大メカニズム（BUDGET_EXHAUSTED の分解）

### (A) 単一の遅い `file_grep` 全走査 = 予算1発で溶かす（13件, iter≤2）

反復1–2回で timeout した13件は、ほぼ全て `file_grep` を1本呼んだだけで 280–557s に達している:

```
idx23 it=1 [file_grep]                 elapsed=557s  err="timeout_s=180 exceeded"
idx38 it=1 [file_grep]                 elapsed=318s  err="timeout_s=240 exceeded"
idx48 it=1 [file_grep]                 elapsed=308s
idx55 it=1 [file_grep]                 elapsed=329s
idx67 it=2 [file_grep, file_grep]      elapsed=511s
idx76 it=1 [file_grep]                 elapsed=333s   … 他 idx37/41/43/50/53/70/98
```

**機序**: evidence_index を**ミス**した discovery クエリ → `file_grep` が
**全ファイルを `_extract`（Office復号・pdf・OCR）する全走査**にフォールバック（`file_grep.py:190-218`）→
同期呼び出しのため**途中で止められない**（timeout はターン間判定 `investigator.py:1291`）→
公称予算の1.5–3.6倍で暴走し、**答えを1つも得ないまま**問答終了 → BUDGET_EXHAUSTED 棄権。

### (B) 発見系の空回り = 全走査系ツールを繰り返して収束しない（≈6件, iter≥10）

```
idx40 it=14  find_files×7 + file_grep×5           elapsed=289s
idx73 it=15  file_grep×10 + find_files×2          elapsed=297s
idx63 it=16  file_grep×8 + canonical×3 + compute×3 elapsed=246s
idx27 it=11 / idx28 it=13 / idx39 it=10
```

対象ファイルを特定できず `file_grep`/`find_files` を繰り返し、収束前に時間切れ。
＝**発見（どのファイルか）フェーズが律速**（Step13 診断「BUDGET問の探索73%が発見系」と整合）。

### まとめ: (A)+(B) いずれも `file_grep` 全走査に起因

Step13 で導入した evidence_index / canonical_manifest（SOT-2528–2533）は全走査撤廃を狙ったが、
**index を外すクエリでは依然フォールバック全走査が走り**、その1本が予算を溶かす。フラグは ON でも
「ミス時フォールバック」の遅さが未対策のまま残っている。

---

## 4. 「UNANSWERABLE(10)」の多くは真の不能ではない — 書式/深掘り不足

UNANSWERABLE 10件のうち複数は **gold が実在**する。＝根拠が無いのではなく **抽出できていない**:

| idx | archetype | gold | 実体 |
| --- | --- | --- | --- |
| 2 | highlight_set | プロジェクトキックオフ実施… | **オレンジ行**の抽出が必要（書式依存） |
| 11 | document_extract | 4,675,000円 | **太字∧下線∧イタリック**同時該当（書式依存） |
| 15 | document_extract | Gender=Male… | **黄色ハイライトセル**の抽出条件（書式依存） |
| 47 | derived_calc | 1899年 | **黄色ハイライトセル**が指す対象（書式依存） |
| 65 | derived_calc | 相関係数<-0.99のセル | **黄色ハイライト**条件の言語化（書式依存） |
| 29 | fact_lookup | 6.088138 ~ 6.288138 | ヒストグラム3番目ビンを小数6位（深掘り不足） |
| 30 | derived_calc | 1.18 | 多段条件付き割合（cross-filter 派生） |

**棄権60件中14件がハイライト/色/太字/下線/イタリックに言及**する書式依存問。extract 層が
セルの色・文字装飾を安定して面に出せず、「根拠なし=UNANSWERABLE」または timeout に落ちている。
＝**書式(視覚)抽出ギャップ**が第2のクラスタ。

残る少数状態（EVIDENCE_INCOMPLETE idx25/79・NOT_RETRIEVED idx17・PARSED_AMBIGUOUS idx56・
RETRIEVED_NOT_PARSED idx57・SPIN_CUTOFF idx97）も **黄色/青ハイライト・回帰係数の多段導出**が中心で、
同じ「書式抽出 or 多段派生」の難所に属する。

---

## 5. 誤答6件の原因（棄権→回答変換の EV 較正不足）

統合(SOT-2527)の wrong-14 は A1–E で全解消済み。本実測の wrong-6 は**別集合** `{4,9,14,49,83,92}` で、
いずれも**回答増フラグが従来安全棄権していた難問を確信度1.00で回答化**して外したもの:

| idx | type | pred → gold | 誤り方 | 推奨対策 |
| --- | --- | --- | --- | --- |
| 4 | derived_calc | `smoker` → `bmi` | 相関の**対象列取り違え** | exec_verifier の same-target 照合を相関/カテゴリ列へ拡張 |
| 83 | derived_calc | `0.23988` → `0.38317` | 回帰予測の**式/係数ずれ** | 再計算の grounding 強化（対象行確定） |
| 9 | document_extract | 長文抜粋 → `該当なし` | **「該当なし」問の過剰回答**（空集合を非空化） | 空集合契約ゲート（A1の空集合等価を回答側に適用） |
| 14 | version_diff | `藤田 彩→井上 里奈` → 列名アンダースコア修正 | **差分の対象完全取り違え** | canonical 直行での対象特定・first-move ルーティング精緻化 |
| 49 | document_extract | アクション注記 → `WBS・進捗管理台帳確定` | **隣接セル取り違え** | 粒度/対象照合を document_extract へ拡張 |
| 92 | enum/count | 「特定できません…M01」→ `49` | **半端回答**（数え上げ未完で不確実回答） | 列挙クロージャ(SOT-2500)未充足時は棄権へ倒す EV ゲート |

＝**precision の穴は「難問を無理に回答化する EV 較正」に集中**。回答増と precision のトレードオフの
残差であり、型別(exec/空集合/列挙)の EV ゲートで**回答数を大きく落とさず**是正できる範囲。

---

## 6. 型別の棄権/誤答分布（どこに残っているか）

| archetype | n | match | abstain | wrong |
| --- | ---: | ---: | ---: | ---: |
| derived_calculation | 32 | 6 | **23** | 3 |
| document_extract | 24 | 9 | 13 | 2 |
| fact_lookup | 26 | 14 | 12 | 0 |
| enum_set | 9 | 3 | 6 | 0 |
| version_diff | 6 | 2 | 3 | 1 |
| config_hyperparam / data_shape / highlight_set | 各1 | 0 | 各1 | 0 |

- **derived_calculation が棄権(23)・誤答(3)の最大の残り**。多段の派生計算＋書式ハイライト由来の
  対象特定が重なり、発見/抽出フェーズで時間切れ or 対象取り違えを起こしやすい。
- fact_lookup は wrong=0（precision 健全）だが棄権12＝**取りこぼし**（時間切れ）が中心。

---

## 7. 推奨する次段（follow-up issue 候補 / 本 issue は測定・調査のみ）

棄権削減は **precision を落とさず「時間の使い方」を直す**のが本丸（回答増の EV は誤答対策で別途）:

1. **[最優先] `file_grep` 全走査フォールバックの排除/上限化**
   - index ミス時に全ファイル `_extract` へ落ちる経路に**ハードな件数/時間上限**を設け、
     超過時は「未発見」を早期に返して**別ツールへ切替**（現状は1本で予算全溶かし）。
   - あるいは evidence_index の**カバレッジ拡張**（書式・数式・スライド見出しを索引化）して
     ミス率そのものを下げる。40/44 の BE が file_grep 経由＝効果が最大。
2. **ツール単位のハード wall-clock（協調キャンセル）**
   - timeout がターン間判定のみ（`investigator.py:1291`）のため単一呼び出しを中断できない。
     `file_grep`/`_extract` に per-call デッドライン（例: 残予算の一定割合）を渡し、
     途中で打ち切って**部分結果 or 早期棄権→他ルート**に回す。時間切れ34件の多くを回収可能。
3. **書式(視覚)抽出ギャップの解消**（棄権14件＋UNANSWERABLE多数）
   - ハイライト色/太字/下線/イタリックを extract 面に安定出力し、highlight 系契約に直結配線。
4. **難問回答化の型別 EV ゲート**（wrong 6件）
   - 「該当なし=空集合」ゲート(idx9)／列挙未充足棄権(idx92)／exec same-target 列照合の相関列拡張(idx4)。

---

## 8. 関門2（SOT-2478）への含意

本候補（A1–E ON）は wrong=6＝開始基準と同値で **precision 非劣化・net strict 改善(15→28)** で PASS 済み。
本調査は**その先の棄権60を回収する軸**を特定したもので、上記 (1)(2) は precision を下げずに
abstain→match を狙える（回答増の EV 論点と直交）。champion serve は全フラグ既定 OFF で byte-identical のまま。
