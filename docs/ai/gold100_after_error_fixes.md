# Gold-100 統合実測（誤答型別対策 A1–E 完了後の唯一の実測） — SOT-2550

- 実行日時: 2026-08-09T03:40–04:15Z（1回のみ / 所要 ~35分）
- コマンド: `.venv/bin/python -m scoring.gold_offline --run --workers 8 --out artifacts/gold100_after_error_fixes.json`
  （gen=investigator, n=100）
- index: **再利用**（`index_store/` と事前処理ストア evidence_index/structure_store/canonical_manifest/corpus_profile は
  Aug-09 00:48–00:52 に再ビルド済で、コーパス（Aug-04）および A1–E マージより新しい）。A1–E の変更は全て serve-time の
  ため index への影響なし → 再ビルド不要と判断。
- レポート: `artifacts/gold100_after_error_fixes.json`（gitignore）、履歴: `docs/gold_offline_history.jsonl`
- コスト: **$7.22**（統合(SOT-2527) $7.14 比 +$0.08 ≈ 横ばい）
- genotcha 対策: google-genai `HttpOptions.timeout=None` 無限 block を `/tmp/genai_patch/sitecustomize.py`
  （PYTHONPATH 注入・180000ms・repoコード不変）で緩和。CLOSE-WAIT の半閉塞接続1本を timeout が回収したのを ss で確認。

## フラグ構成（SOT-2527 の回答増11本を維持 ＋ A1–E 精度フラグを追加）

回答増11本（Step11–13、SOT-2527 と同一集合。「ある程度回答する」挙動を維持）:

```
RAG_FIRST_MOVE_ROUTING=1  RAG_SPIN_DETECTION=1  RAG_ADAPTIVE_BUDGET=1  RAG_EVIDENCE_CACHE=1
RAG_BUDGET_BOUNDARY_RESEARCH=1  RAG_UNANSWERABLE_FALLBACK=1  RAG_PDF_OCR=1  RAG_SHARE_CORPUS_PROFILE=1
RAG_CANONICAL_MANIFEST=1  RAG_EVIDENCE_INDEX=1  RAG_STRUCTURE_STORE=1
```

今回 A1–E で追加した精度修正:

```
RAG_GRANULARITY_NORMALIZATION=1   # SOT-2545 誤答A2 回答粒度
RAG_XLSX_EMBEDDED_IMAGE=1         # SOT-2548 誤答D Sheet2埋め込み画像抽出
RAG_CONFLICT_RESOLUTION=1         # SOT-2549 誤答E 競合解決
GATE_EXEC_CORRECT=1               # SOT-2547 誤答C exec_verifier 大外し再計算矯正
# SOT-2544（judge同義正規化・crag.py）と SOT-2546（境界オフバイワン規則）はフラグ無し＝常時ON
```

## 結果サマリ（開始基準・統合基準との差分）

| 指標 | 開始基準(08-08 03:03) | 統合(SOT-2527, 08-08 23:09) | **本実測(A1–E後, 08-09 04:15)** | vs統合 | vs開始 |
| --- | --- | --- | --- | --- | --- |
| match | 21 | 27 | **34** | **+7** ✅ | +13 ✅ |
| abstain | 73 | 59 | **60** | +1 | −13 |
| wrong | 6 | 14 | **6** | **−8** ✅ | ±0 |
| **net(match−wrong)** | 15 | 13 | **28** | **+15** ✅ | **+13** ✅ |
| cost | $5.93 | $7.14 | $7.22 | +$0.08 | +$1.29 |

- **狙い達成**: wrong を 14→**6** に削減（開始基準の6まで戻す＝precision 完全回復）、net を 13→**28**（開始基準15も超過）。
- **回答数は維持**: 回答数(match+wrong)=40（統合41 とほぼ同）、abstain 59→60 と横ばい。「ある程度回答する」挙動を壊さず精度だけ上乗せ。
- non-match の **90.9%（60/66）を安全棄権**に保持。

### 型別内訳（本実測）

| type | n | match | abstain | wrong |
| --- | --- | --- | --- | --- |
| derived_calculation | 32 | 6 | 23 | 3 |
| document_extract | 24 | 9 | 13 | 2 |
| fact_lookup | 26 | 14 | 12 | 0 |
| enum_set | 9 | 3 | 6 | 0 |
| version_diff | 6 | 2 | 3 | 1 |
| config_hyperparam | 1 | 0 | 1 | 0 |
| data_shape | 1 | 0 | 1 | 0 |
| highlight_set | 1 | 0 | 1 | 0 |

fact_lookup wrong 6→**0**、enum_set wrong 0、document_extract wrong 5→2。統合で崩れていた precision が型横断で回復。

## 誤答14件（統合）の帰結 — A1–E は全件解消

統合 SOT-2527 の wrong-14 `{62,85,88,93,78,80,47,63,97,29,69,70,75,84}` は **全14件が本実測で wrong から外れた**
（正答化 or 安全棄権化）。A1–E の型別対策が設計どおり効いた:

- A1 judge同義(SOT-2544): idx62/85 → match 化（偽陰性解消）
- A2 粒度(SOT-2545): idx88/93 → match 化
- B 境界(SOT-2546): idx29/69/84 → match 化（オフバイワン矯正）
- C 対象定義(SOT-2547): idx63/97 → 再計算で match 化、idx47/70 → 対象取り違えのため安全棄権化（誤答−1→棄権0）
- D 抽出(SOT-2548): idx80/78 → 正答化 or 安全棄権化
- E 競合(SOT-2549): idx75 → match 化（第4週に確定）

## 残る誤答6件（別集合）と原因分類 → 次段提示

本実測の wrong-6 は統合14とは**別の集合** `{4, 9, 14, 49, 83, 92}`。回答増フラグが従来 BUDGET/UNANSWERABLE で
安全棄権していた別問を回答に転じ、その一部が wrong に振れたもの（統合と同じ「棄権→回答変換の precision コスト」だが、
A1–E で対象14を潰した結果 net は大幅プラス）。

| idx | type | pred → gold | 原因分類 | 推奨次段 |
| --- | --- | --- | --- | --- |
| 4 | derived_calc | `smoker` → `bmi` | 相関の**対象列取り違え**（SOT-2501既知の age/bmi 系） | exec_verifier の same-target 列照合を相関/カテゴリ列にも拡張 |
| 83 | derived_calc | `0.23988` → `0.38317` | 回帰予測**数値の式/係数ずれ** | exec_verifier 再計算の grounding 強化（対象行の確定） |
| 9 | document_extract | 長文スライド抜粋 → `該当なし` | **該当なし問の過剰回答**（空集合を非空回答化） | 「該当なし」契約に空集合判定ゲート（A1 空集合等価の回答側適用） |
| 14 | fact_lookup? | `藤田 彩 → 井上 里奈` → 列名アンダースコア修正 | **対象完全取り違え**（検索/理解ミス） | canonical 直行の対象特定強化・first-move ルーティング精緻化 |
| 49 | document_extract | アクション注記 → `WBS・進捗管理台帳確定` | **抽出セル取り違え**（隣接項目を採用） | 粒度/対象照合の document_extract への拡張 |
| 92 | enum/count | 「特定できません…M01」→ `49` | **半端回答**（数え上げ未完で不確実回答） | 列挙クロージャ(SOT-2500)未充足時は棄権へ倒す EV ゲート |

いずれも回答増変換の EV 較正不足が主因で、A1–E の範囲外（本 issue は測定のみ）。

## 関門2 非劣化ゲート（SOT-2478）の判定

- champion serve path は全フラグ既定 OFF で **byte-identical**（本測定は候補構成の効果確認、champion 差し替えは行わない）。
- 非劣化基準（precision を劣化させないこと）: 本候補は wrong=**6**＝開始基準と同値で **precision 非劣化**、かつ net 15→**28** と
  strict 改善。統合 SOT-2527 候補は wrong 6→14・net 13 で **FAIL** だったのに対し、本 A1–E 候補は **非劣化ゲートを PASS**。
- したがって本フラグ構成は「回答増を維持しつつ precision を非劣化に戻した」promotable 品質。正式な封印 holdout での関門2確定は
  昇格ステップ（別 issue）だが、少なくとも gold100 上は非劣化・net 改善で採用可能条件を満たす。

## 結論

誤答型別対策 A1–E は、統合(SOT-2527)で発生した precision 後退（wrong 6→14, net 13）を **完全に回復**し、
回答増挙動を維持したまま **net 13→28（開始基準15も超過）／wrong 14→6** を達成した。残 wrong6 は別集合の
棄権→回答変換 EV 較正不足で、次段は上表の per-idx 対策（exec_verifier 列照合拡張・該当なし空集合ゲート・
列挙未充足棄権）を推奨する（測定のみの本 issue の範囲外）。
