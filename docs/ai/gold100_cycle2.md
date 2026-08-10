# サイクル2 統合 Gold-100 実測 — SOT-2622（浪費除去 + operand束ね）

- 実行日時: 2026-08-10T19:17:32–19:24:54Z（本 issue で 1 回のみ、約7分22秒）
- main HEAD: `05fdaa2`（全 blockedBy 子 SOT-2614〜2621 の PR#134/135/133/137/139/140/141/142 を含む）
- 公式モデル: `gemini-3.6-flash`（`VERTEX_LOCATION=global`）、judge=`codex`。実行前に global でスモーク確認（`OK` 応答）。
- コマンド: `bash scripts/sot_cycle2_gold100.sh`（内部で `.venv/bin/python -m scoring.gold_offline --run --workers 8 --out artifacts/gold100_cycle2.json`）
- 生レポート: `artifacts/gold100_cycle2.json`（gitignore）／履歴: `docs/gold_offline_history.jsonl` に追記済み
- 事前処理: コーパス不変・cycle2 フラグは flag-gated serve path のみのため、champion と同一 index/evidence/canonical/profile/document-registry/structure store を再利用。

## 構成（フラグ）

champion（Wave A net40 = r1 回答増フラグ群 + `RAG_FORMAT_EVENTS=1` + `RAG_DET_PIPELINE_ROUTER=1`、B1=ON/B2=OFF）に、サイクル2の 4 フラグを追加 ON:

| フラグ | 由来 | 役割 |
| --- | --- | --- |
| `RAG_SPIN_PIVOT=1` | SOT-2614 | 同一ツール3連発で強制pivot（浪費ターン解放） |
| `RAG_SEARCH_CAP=1` | SOT-2620 | 型別 search 上限超過で registry 逆引きへ強制切替 |
| `RAG_OPERAND_PREFILL=1` | SOT-2616 | numeric operand を structure store から事前充填 |
| `RAG_CONDITION_PREIR=1` | SOT-2621 | what-if 条件文の BranchCondition IR を pre-loop 注入 |
| `RAG_DET_PIPELINE_B1=1 / B2=0` | SOT-2618 | Wave B1 単独ゲート（document_extract のみ、fact_lookup は非経由） |

> **⚠ 構成上の重要な発見**: issue 記載では「formatting契約（SOT-2617/2619 はテンプレのため常時有効）」とあったが、実装 (`src/rag/agent/formatting.py:103`) では **SOT-2617 の derived 書式契約は `RAG_DERIVED_FORMAT_CONTRACTS`（既定 OFF）でゲートされている**。issue の前提（常時有効）は誤りで、本 run では **SOT-2617 書式契約は非活性**だった。これが下記回帰の一部を直接説明する（後述・次軸①）。

## 結果: net 28（**関門2 FAIL** — champion net40 を更新せず）

| 指標 | champion (Wave A) | **cycle2** | 差分 | 判定 |
| --- | ---: | ---: | ---: | --- |
| match | 47 | **45** | **−2** | 劣化 |
| abstain | 46 | **38** | −8 | — |
| wrong | 7 | **17** | **+10** | 大幅増（回帰） |
| **net (match−wrong)** | **40** | **28** | **−12** | **FAIL** |
| cost | $13.64 | $10.96 | −$2.68 | — |

昇格条件は **net > 40 かつ match 非劣化・wrong 非増加**。cycle2 は net 28（< 40）、match −2（劣化）、wrong +10（増加）で **3 条件すべて不成立 → 関門2 FAIL**。**champion（Wave A net40）を据え置く。**

## champion(Wave A net40) → cycle2(net28) 一問遷移

| 遷移 | 件数 | idx |
| --- | ---: | --- |
| MATCH→MATCH | 40 | 0,2,3,7,10,11,13,16,18,19,20,21,23,25,26,29,31,33,41,42,43,44,46,49,51,54,58,59,60,66,69,70,71,72,75,81,85,86,89,91 |
| MATCH→ABSTAIN | 4 | 30,35,68,90 |
| MATCH→WRONG | 3 | 4,62,94 |
| ABSTAIN→MATCH | 4 | 15,56,80,96 |
| ABSTAIN→ABSTAIN | 32 | 1,5,6,14,17,22,28,32,34,36,38,39,40,45,48,50,53,55,57,61,63,67,76,77,79,82,87,92,93,95,97,98 |
| ABSTAIN→WRONG | **10** | 8,9,12,24,37,47,64,73,83,99 |
| WRONG→MATCH | 1 | 74 |
| WRONG→ABSTAIN | 2 | 52,84 |
| WRONG→WRONG | 4 | 27,65,78,88 |

**回帰の核**: 浪費除去フラグは BUDGET_EXHAUSTED を回収（39→27, −12）したが、回収された棄権のうち **MATCH化はわずか 4（ABSTAIN→MATCH）に対し WRONG化が 10（ABSTAIN→WRONG）**。加えて旧 MATCH の 3 件が WRONG 化（idx4/62/94）。「浪費」として打ち切った探索ターンは実際には**有用な証拠収集**であり、それを早期に切ると**根拠不十分のまま誤答を確定**してしまう。precision 崩壊（wrong 7→17）が abstain 削減（−8）を大きく上回った。

## BUDGET_EXHAUSTED 削減と状態コード遷移

| code | champion (Wave A) | **cycle2** | 差分 |
| --- | ---: | ---: | ---: |
| UNANSWERABLE | 6 | **11** | +5 |
| BUDGET_EXHAUSTED | 39 | **27** | **−12** |
| SPIN_CUTOFF | 1 | **0** | −1 |
| （NOT_RETRIEVED / RETRIEVED_NOT_PARSED / PARSED_AMBIGUOUS / EVIDENCE_CONFLICT / EVIDENCE_INCOMPLETE） | 0 | 0 | ±0 |
| **abstain 合計** | **46** | **38** | −8 |

浪費除去は機構的には成功（BUDGET_EXHAUSTED −12）だが、回収分の行き先は MATCH ではなく WRONG(+10) と UNANSWERABLE(+5) に流れた。**「予算枯渇の削減」自体は達成したが、それが正答率に転化しなかった**のが本サイクルの結論。

## 新規誤答 13 件の型別内訳（回帰源の分類）

| クラス | idx | 症状 | 疑わしいフラグ |
| --- | --- | --- | --- |
| **A. 早期確定による誤値**（探索打ち切り→根拠不十分で誤値を commit） | 4(age/bmi), 47(1988/1899), 83(0.37083/0.38317), 99(2.53/2.49), 24, 12 | derived/data_shape/fact_lookup で値そのものが誤り | `RAG_SPIN_PIVOT` / `RAG_SEARCH_CAP`（探索ターン早期打ち切り） |
| **B. 書式冗長 near-miss**（値はほぼ正しいが単位/冗長語で judge に落ちる） | 8("14,744ドル人"/"14,744ドル"), 37("22,000円/時間"/"22,000円"), 64(括弧内明細付き/"80〜130時間"), 62(verbose n_estimators) | 値正・整形失敗 | **SOT-2617 書式契約が OFF**（本来これで救済されるはず） |
| **C. 過剰産出**（棄権/絞込すべきを広く産出） | 9("該当なし"), 73("Gender"), 94("T09"のところ"T07,T08,T09") | over-inclusion / 該当なし誤答 | `RAG_CONDITION_PREIR`（idx94 列挙拡大） / spin/cap（idx9/73 早期産出） |

- クラス A（6件）が最大の回帰源で、`RAG_SPIN_PIVOT` + `RAG_SEARCH_CAP` の**探索早期打ち切り**に強く相関する。
- クラス B（4件）は **SOT-2617 書式契約が非活性だったこと**が主因。特に **idx64 は SOT-2617 が focused 検証で `80〜130時間` へ短縮できた literal target**（ledger 参照）だが、契約 OFF のため cycle2 では冗長のまま WRONG。idx8/37 も unit_currency/verbosity 契約の対象クラス。
- 本測定は確率的生成を含む単一実測であり、各フラグの因果寄与を完全分離したアブレーションではない（issue 注記どおり）。上記はフラグ役割と誤答症状からの**相関ベースの帰属仮説**。

## 決定論直答率

- 決定論パイプライン（Wave A + B1）経由の直答は cycle2 で **14 問**（idx 1,10,15,19,26,33,44,51,54,58,71,80,81,86 が det-method 記録）。router/B1/B2 構成は champion と同一のため、決定論直答率は約 13〜14% で **champion からほぼ不変**。cycle2 フラグは LLM ループ側に作用し、決定論経路は非改変。

## 発火統計（spin_pivot / search_cap）

- 本 run の abstain-ledger（38 棄権記録, `recorded_at ≥ 19:17`）中: `spin_pivot` 観測シグナル **27/38**、`search_cap` 観測シグナル **18/38** に出現。ガードは非常に広範に発火していた（棄権に至らず答えたケースは ledger 外のため実総発火数はこれ以上）。ガードの広範発火はクラス A の precision 崩壊と時間的に整合。

## 関門2 判定と champion 更新可否

- **関門2: FAIL**（net 28<40 / match −2 / wrong +10）。**champion = Wave A net40 を維持**。cycle2 の 4 フラグ同時 ON 構成は gold100 proxy 上で champion を更新しない → **本構成は rejected**。
- local proxy と実 LB の相関は弱い（ρ=−0.09）ため提出候補への昇格は別判断。**本 issue では提出しない**（一次KPIは leaderboard rank）。

## フラグ別採否の提案（次サイクルへ）

| フラグ | 提案 | 根拠 |
| --- | --- | --- |
| `RAG_SPIN_PIVOT` | **REJECT（要改修）** | クラス A の早期誤値 commit の主因。強制pivot前に **evidence-completeness ゲート**を挟み、根拠不十分なら pivot ではなく棄権を維持する設計が必要 |
| `RAG_SEARCH_CAP` | **REJECT（要改修）** | 同上。search 上限で registry 逆引きへ切替えても、逆引き根拠が薄いまま産出→誤答。cap 後に確度が閾値未満なら棄権へフォールバックすべき |
| `RAG_OPERAND_PREFILL` | **SUSPECT / 保留** | idx4（age vs bmi）で誤 operand 選択の疑い。単独 focused アブレーションで採否を確定（本 run では他フラグと交絡） |
| `RAG_CONDITION_PREIR` | **SUSPECT / 保留** | idx94 の列挙過剰産出に相関。単独 focused アブレーションで確定 |
| **SOT-2617 `RAG_DERIVED_FORMAT_CONTRACTS`** | **次軸① 最優先で ON 再測定** | 本 run で誤って OFF。クラス B の 3〜4 件（idx8/37/64）を救済する literal target。issue 前提（常時有効）と実装（既定OFF）の乖離を解消し、この契約を含めた再測定が最も net 改善見込み |

### 次軸（優先順）

1. **`RAG_DERIVED_FORMAT_CONTRACTS=1` を加えた再測定**: 本 run はこの契約を誤って OFF で回した。クラス B（値正・書式落ち）3〜4 件を回収でき、wrong を 17→13〜14 へ下げられる見込み。まず focused（idx8/37/64/6/65）で契約適用後の judge 判定を確認 → 次回統合測定に含める。
2. **spin_pivot / search_cap に precision ゲートを結合**: 「pivot/cap を撃つ前に evidence-completeness を評価し、不十分なら棄権を維持」する改修。浪費除去の**方向は正しい**（BUDGET_EXHAUSTED −12 は実現）が、回収分を MATCH に転化する precision 制御が欠けている。SOT-2614/2620 の設計に completeness ゲートを追加する子 issue を提案。
3. **operand_prefill / condition_preir の単独 focused アブレーション**: 交絡を排して idx4/idx94 への因果を確定し、単独採否を決める。

## 記録

- `docs/gold_offline_history.jsonl`: 追記済み（`2026-08-10T19:24:53Z`, match45/abstain38/wrong17, cost $10.96）。
- `docs/ai/experiment_ledger.jsonl`: cycle2 統合軸 = **rejected**（net40→28）を追記。
- champion 据え置き。次軸①（`RAG_DERIVED_FORMAT_CONTRACTS` 込み再測定）を次サイクルの起点とする。
