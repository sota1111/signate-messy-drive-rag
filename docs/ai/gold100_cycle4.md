# Cycle-4 収束実測 — dual-backend gold100 + model-divergence (SOT-2642)

同一 HEAD (`bd2d47d`) / 同一 champion パイプライン (Wave A + B1) で、flash 公式と Sonnet dev の gold100 を
各 1 回実行し、cycle4「commit 精度はモデル非依存にできる」主張の達成度を確定した。差は **backend と
commit_gate enforce のみ**:

| | backend | RAG_COMMIT_GATE | ENFORCE | RAG_NEUTRAL_PROMPT | official |
| --- | --- | --- | --- | --- | --- |
| flash | gemini-3.6-flash @ global | 1 | **0（観測）** | 1 | true |
| Sonnet | claude-mcp (sonnet) | 1 | **1（強制）** | 1 | false |

enforce を backend で分けた理由（SOT-2639 で実証済）: flash の investigator ループは exec_verifier /
enum guard / formatting を **inline に既に持つ** ので、gate を強制すると compute-record grounding が導出値
(idx30 1.18%) を過剰却下し等価性が壊れる。よって flash は観測のみ（答えは verbatim, テレメトリだけ記録）。
guard-less な claude-mcp backend だけが gate に commit を守らせる — これが cycle4 のアーキ主張そのもの
（決定論ゲートが欠落 inline ガードを代替する）。

## 結果サマリ

| run | match | abstain | wrong | **net** | baseline (子) | Δnet |
| --- | --- | --- | --- | --- | --- | --- |
| **flash 公式** | 47 | 45 | 8 | **39** | net40 (m47/a46/w7, SOT-2610) | **−1** |
| **Sonnet dev** | 40 | 36 | 24 | **16** | net18 (m46/a26/w28, SOT-2628) | **−2** |

- flash 生成コスト: $13.98（gemini live）. Sonnet 生成コスト: $0（flat-rate; residual Gemini 0）。
- 遷移: flash は non-match 53 中 45 (84.9%) を abstain へ落とし wrong 8。Sonnet は non-match 60 中 36
  (60.0%) を abstain、wrong 24。

## 関門2判定（flash 昇格ゲート: net > 40 かつ wrong ≤ 7）

**未達 → champion 更新せず**。flash net 39 (≤40)・wrong 8 (>7)。ただしこれは配線バグではない:

- enforce=0 なので commit_gate は flash の答えを **一切変えない**（テレメトリ: on=43 fired=43 だが値は
  verbatim, match=33/33・abstain=3/3・wrong=7/7 の内訳は「観測しただけ」）。よって net40→39 の −1 は
  gate 由来ではなく、**RAG_NEUTRAL_PROMPT のプロンプト差＋flash-3.6 の非決定床**に帰属する。
- −1 net / +1 wrong は flash-3.6 の既知の 1 標本ノイズ幅（[[signate-official-judge-stochastic]]）に収まる。
- 従って champion（Wave A+B1, neutral/gate なし, net40）を据置く。cycle4 flash 構成は「非劣化（原則）」を
  満たすが昇格には至らない。

## Sonnet 改善幅（目標: net ≥ 35, wrong ≤ 10）

**未達**。wrong 28→24（−4）は縮んだが、net 18→16（−2, match 46→40 / abstain 26→36）。commit_gate + 中立
プロンプトは **wrong を parity まで落とせなかった**。原因は帰属可能（cycle2 と違い）:

### commit_gate テレメトリ（Sonnet, details.jsonl interventions）

- 評価した commit: **83**（COMMIT 51 / ABSTAIN 降格 32）。
- 発火した reject 理由: `numeric_ungrounded`（compute/corpus_aggregate 成功記録なし）**7**、
  `reject_streak_abstain:2>=2`（連続 reject → 棄権降格）**7**。
- **弁別**: gate が実際に守れたのは *numeric-ungrounded* 系のみ。Sonnet の wrong 24 の大半は gate が
  **カバーしない契約**（simple_lookup / format_check / version_diff / multi_hop）に集中しており、gate は
  「発火するが該当しない」。よって wrong 減は −4 に留まり、abstain 側は +10（うち数問は正答も巻き込む
  過剰降格 → match −6）。

## model divergence（目標: 23 → <15, うち wrong 側 ≈ 0）

**部分達成**。`scripts/measure_model_divergence.py` 突合（`docs/ai/model_divergence_cycle4.md` /
`artifacts/model_divergence_cycle4.json`）:

- **divergence total: 23 → 19**（−4, 目標 <15 未達）。
  - flash MATCH → Sonnet 非MATCH: 13
  - Sonnet MATCH → flash 非MATCH: 6
- **wrong 側乖離（commit_precision）: 9**（目標 ≈0 **未達**）— idx [0,11,20,31,41,42,59,88,96]、**全て
  flash MATCH → Sonnet WRONG**。契約内訳 = simple_lookup ×4, format_check ×2, version_diff, multi_hop,
  numeric ×1。→ **numeric/enum 以外は現行 gate の守備範囲外**。
- 到達性乖離（reachability）: 10 — trace-port 対象。
- judge_noise: 0。deterministic-direct 12 問は両 backend 完全一致（決定論層はモデル不変を達成済）。

## 結論 — モデル不変化は本サイクルでは未達（but 原因は特定できた）

決定論パイプライン層（Wave A/B1, det-direct 12 問）は既にモデル不変。しかし **commit_gate の守備範囲が
numeric-grounding + enum-universe に限られる**ため、Sonnet の wrong を parity まで落とせず、wrong 側乖離が
9 残る。gate は「精度をモデル非依存にする」正しい機構だが、**カバレッジが不足**している。

## 次サイクルの相互移植 / gate 拡張課題リスト

1. **gate カバレッジ拡張（最優先, 既起票）**: wrong 側乖離 9 は simple_lookup / format_check /
   version_diff に集中。質問非依存の事前計算事実層 **SOT-2643-2647**（案件マスタ / ID マスタ / 派生メトリクス
   / 版ペア差分 / 配線）が、これらの契約に決定論 grounding を与え gate の enforce 対象を numeric/enum の
   外へ広げる。→ wrong 側乖離を ≈0 に落とす本命。
2. **reachability port（Sonnet→flash, 6 問）**: idx [68,70,79,83,93,94]（numeric ×2, simple_lookup ×4）
   は Sonnet が到達し flash が abstain。flash 側 abstain 回収の trace-port 候補。
3. **reachability port（flash→Sonnet, 4 問）**: idx [4,35,53,72]（numeric ×3, ほか）は flash 到達 Sonnet
   abstain。Sonnet 側の探索手順移植候補。

## 成果物

- `scripts/sot_cycle4_gold100.sh`（flash 公式）/ `scripts/sot_cycle4_sonnet_gold100.sh`（Sonnet dev）
- `artifacts/gold100_cycle4_flash.json`（official:true）/ `artifacts/gold100_cycle4_sonnet.json`（official:false）
- `artifacts/model_divergence_cycle4.json` / `docs/ai/model_divergence_cycle4.md`
- history/ledger 追記（official フラグ分離）。**LB 提出はしない**。
