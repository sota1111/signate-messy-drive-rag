# champion (Wave A net40) の abstain 46 / wrong 7 分類 — 全後続実測との突合

- 基準: SOT-2610 champion 実測（flash 3.6, match47/abstain46/wrong7）
- 突合先: cycle2（SOT-2622, net28）/ cycle3（SOT-2636, net35）/ Sonnet dev（SOT-2628, net18, official:false）
- 生成: 2026-08-11、`gold100_sot2610_waveA.json` × 各 run の abstain/wrong items × champion 窓の abstain_ledger（state code）

## ABSTAIN 46 の3分類

| クラス | 件数 | 定義 | 意味 |
|---|---:|---|---|
| **A. 回収実証済み** | **12** | どこかの実測で MATCH 化した | 到達手順は存在する。移植/採用すれば取れる |
| **B. 回答化するが誤答のみ** | **18** | 積極構成で回答化したが全て WRONG | **commit 精度の戦場**（cycle4 commit_gate の主戦域） |
| **C. hard core** | **16** | flash/Sonnet/全構成で一度も回答に至らず | 誘導では取れない。新しい証拠獲得能力が必要 |

### A. 回収実証済み 12問（code: BUDGET 9 / UNANSWERABLE 3）

| idx | 型 | 到達実績 |
|---|---|---|
| 15, 56, 80, 96 | doc_extract/derived | **3実測すべてでMATCH（安定回収・採用最有力）** |
| 53 | derived | sonnet+cycle3 |
| 9, 28, 92 | doc_extract/derived | cycle3 のみ |
| 5, 17, 36, 79 | derived/fact | sonnet のみ（flash 未移植: G1/G2 の残） |

### B. 回答化するが誤答のみ 18問

- 頻出: idx **8, 64**（3構成すべてで WRONG — 書式冗長 near-miss、SOT-2617 契約の literal target）、idx **47**（c2+c3 WRONG — 値誤り 1988/1899 系）、idx24, 76（what-if/データ形状）
- 全18: 1, 6, 8, 12, 14, 24, 34, 37, 47, 61, 64, 73, 76, 77, 83, 93, 95, 99
- 対策 = **回答化フラグ × commit_gate（cycle4）**: ゲートが誤答を棄権へ倒せば、このクラスは「安全に回答化を試せる」母集団になる

### C. hard core 16問（BUDGET 15 / SPIN 1）

- 型分布: **enum 5**（32, 38, 45, 67, 87）/ **derived 5**（40, 50, 57, 63, 97）/ fact 3（39, 48, 98）/ doc_extract 2（55, 82）/ version 1（22）
- 全構成・両モデルで未到達 = 誘導差では埋まらない。enum 5問の集中は「列挙 universe の解決能力」自体の限界を示唆
- 次の攻め口: Sonnet に**別誘導（型別ヒント・分解指示）で再挑戦させ trace を採る**（$0）→ 新手順が出た問だけ移植

## WRONG 7 の分類

| idx | 型 | 後続実測 | 分類 |
|---|---|---|---|
| **74** | version_diff | c2/c3/sonnet 全てMATCH | ✅ **修正済み**（SOT-2619、無料回収済み） |
| **52** | doc_extract | c3=MATCH | 🟡 **括弧付加クラス**: 「監視ダッシュボード構築**（別契約）**」vs gold「監視ダッシュボード構築」 |
| **84** | fact_lookup | sonnet=MATCH | 🟡 **括弧付加クラス**: 「5**（スライド6）**」vs gold「5」 |
| **88** | doc_extract | 全てWRONG | 🟡 **括弧付加クラス**: 「解釈・業務示唆整理**（担当者：松本・鈴木）**」vs gold「解釈・業務示唆整理」 |
| 27 | derived | 全てWRONG | 🔴 値誤り（5 vs 7） |
| 65 | derived | sonnet=ABSTAIN | 🔴 回答型取り違え（「14件」vs gold=セルの特定記述） |
| 78 | fact_lookup | 全てWRONG | 🔴 内容誤読（精算規定の解釈） |

### 発見: 「括弧内付加情報」書式クラスが wrong 7 中 **3問**（52/84/88）

値は正しいのに括弧で補足を付けたために judge に落ちる同一パターン。**「回答本体＋括弧内付加情報 → 付加情報を落とす」書式契約1本で3問を狙える**（値不変・既存 formatting 層への追加。ただし「（別契約）」等が意味を変えるケースの fail-closed 判定が必要）。

## 戦略サマリ

- **net 40 → 上限の見取り図**: 修正済み +1（74）／括弧クラス +3 候補（52/84/88）／A クラス採用 +12 候補／B クラスは commit_gate 前提で挑戦可 ＝ ゲート付きで net 50 台が理論射程。C の16問は現能力の外
- cycle4（commit_gate）は B クラス18問の前提装置。A クラスの安定4問（15/56/80/96）は実測3回連続 MATCH であり、次の統合測定で最初に確認すべき「取れて当然」ライン
