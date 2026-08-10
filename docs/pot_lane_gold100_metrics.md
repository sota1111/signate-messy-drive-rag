# SOT-2586 — NUMERIC PoT 強制ハードレーン: gold100 三層メトリクス

親 SOT-2568 ディープリサーチ実装順 4/7 (P1)。最弱型 `derived_calculation` (gold100 で n=32・match 6・
wrong 3、idx76 の 79,200→73,260 誤算に代表される) を **operand binding / formula / execution の三層分離**
で立て直すレーン。狙いは「operand が届かない(A)」と「計算を間違える(B)」を切り分けて個別に判定・計測する
こと。

## レーン構成 (`src/rag/agent/pot_lane.py`)

```
Evidence Binder      operand を value/unit/source(doc:sheet!cell) 付きで束縛。operand_sources_complete。
Condition Interpreter what-if/条件文を IR 化(predicate / predicate_truth / base_quantity / adjustments)。
Formula Builder      自由 Python を廃止し許可演算限定 AST のみ
                     (ADD/SUB/MUL/DIV/SUM/MEAN/RATIO/PERCENT_CHANGE/WEIGHTED_SUM/ROUND/MIN/MAX)。
Hard Executor        Decimal 厳密算術(50桁 local context — process-global context は不変更)。
Independent Verifier 同じ Bound AST を SymPy Rational / Fraction へ翻訳して独立再計算(eval/parse_expr 不使用)。
N-sample majority    一致条件 = answer ∧ operand source ∧ unit ∧ branch interpretation の全一致。
```

- 既定 **OFF** (`RAG_POT_HARD_LANE`)。OFF 時はツールセット・プロンプトともに byte-identical
  (`verify_formula` ツールも NUMERIC 追加ディレクティブも露出しない)。
- 三層はそれぞれ独立に判定され、失敗は `failure_taxonomy` の
  `EVIDENCE_INCOMPLETE`(operand 層) / `EXECUTION_DISAGREEMENT`(formula・execution 層) に対応づく。

## 計測 (`scripts/measure_pot_lane.py`)

内部ハーネスは *故障箇所を分離する診断器* であり LB 予測器ではない (local proxy ↔ 実 LB ρ=-0.09)。
本スクリプトは決定論・ネットワーク非依存で二層の計測を出力する。

### 1. Route coverage (常時・オフライン)

`artifacts/gold_100_review.csv` の `derived_calculation` 質問が NUMERIC route → compute ハードレーンへ
何件 dispatch されるか (= 強制 PoT レーンの適用対象率) を記録する。直近実測:

| metric | value |
| --- | --- |
| derived_calculation_n | 32 |
| numeric_route_rate | 0.9688 (31/32) |
| hard_lane_rate | 1.0 (32/32 が決定論ハードレーンへ) |
| sympy_backend | True |

`derived_calculation` の全 32 問が決定論ハードレーンへ振り分けられ、うち 31 問が NUMERIC route
(残り1問はより特化した route への正当な refinement)。つまり `derived_calculation` は強制 PoT レーンの
対象として完備している。

### 2. 三層 accuracy (`--details` 供給時)

`RAG_POT_HARD_LANE=1` の gold ラン details (`*.details.jsonl`) が各問に `pot_lane` verdict
(= `PoTResult.to_dict()` の形) を持つ場合、以下を集計する:

- `operand_binding_accuracy` — operand 層合格率 (operand_sources_complete)
- `formula_accuracy` — formula 層合格率 (制限AST妥当・分岐整合)
- `execution_accuracy` — execution 層合格率 (実行↔独立検算一致)
- `verifier_disagreement_rate` — EXECUTION_DISAGREEMENT 率

出力: `artifacts/pot_lane_diagnostics.json`。

> live gold ラン (Gemini 認証・コスト・timeout) は本 PR の範囲外。details 取込口を用意済みで、
> 有効提出時に上記三層 accuracy が同一スクリプトで記録される。idx76 型の分岐選択ミスは
> Condition IR の `branch_signature` 不一致として formula 層で捕捉され、多数決でも branch 軸の不一致
> として弾かれる (回帰テスト `test_conditional_branch_must_reference_base_quantity`)。

## idx76 の捕捉

79,200 を「3分の2に減額」する条件付き計算で、減額を **間違った base に適用**した誤り (→73,260) は:

1. Condition IR に `base_quantity` を明示させ、式が実際に参照する operand と突き合わせる
   (`_branch_consistent`) → 不一致なら formula 層 FAIL。
2. N-sample majority の一致条件に `branch interpretation` を含める → 正しい branch の候補だけが多数決に
   勝つ (right-looking な数値でも wrong branch は勝てない)。
