# gold-100 棄権理由の内訳 + 回収EV実測 (SOT-2483)

対象: `artifacts/predictions_test_resolve.details.jsonl`（gen=resolve: investigator→verifier→tie-break）を
`predictions_test_v3_final.csv`（gold）で採点。生成物 `artifacts/gold_abstain_breakdown.json`。
再現: `python -m scoring.abstain_breakdown --recovery-ev`。

## 棄権の内訳（決定論・LLM不要）

gold-100 のうち commit=19 / 棄権=81。棄権81件を「なぜ合議がcommitしなかったか」で分類:

| reason | 件数 | 回収可能 | 意味 |
| --- | ---: | ---: | --- |
| `both_abstain` | 40 | 0 | investigator/verifier 双方が独立に「わかりません」= **真の無回答**。回収不能 |
| `one_side_abstain` | 33 | 33 | 片方のみ値を提案・他方棄権。tie-break でも合議は棄権 |
| `enumeration_mismatch` | 7 | 7 | 列挙の要素集合が不一致 |
| `verifier_disagreement` | 1 | 1 | 両者が別々の値を主張し tie-break で決着せず |

**回収可能=41 / 回収不能(真の無回答)=40。**「根拠が強い（両AG一致・evidenceあり）のに棄権」は
**存在しない**（一致してcommitできる棄権は0件。棄権は全て無回答か一次意見の不一致）。

## 回収EV実測（回収可能41件の候補を gold と実採点judgeで照合）

公式ルーブリック Perfect +1 / Acceptable +0.5 / Missing 0 / Incorrect −1。各件は現状 0（棄権）なので
以下は commit した場合の**限界EV**。

- **commit-all（41件全回収）EV = −19.5**（Perfect 9 / Acceptable 1 / Missing 2 / **Incorrect 29**）。
- 較正: EV最大しきい値は commit=0（=棄権維持）。`adopt=False` / `signal_separates=False` /
  LOO EV=0.0 — **確信度シグナルは分離しない**。

### 候補ソース別

| ソース | n | 限界EV | 内訳 |
| --- | ---: | ---: | --- |
| investigator-solo（inv提案・ver棄権） | 7 | **+3.0** | Perfect 5 / Incorrect 2（精度0.71） |
| verifier | 18 | −12.5 | P1/A1/M2/**I14** |
| judge（tie-break決着値） | 16 | −10.0 | P3/**I13** |

## 結論（受け入れ条件への回答）

1. **「一致数↑かつ Incorrect を増やさない」は本レバーでは実測上不可能。** 一致を増やす回収ソースは
   例外なく Incorrect も増やす（最良の investigator-solo でも +5一致に対し +2 Incorrect）。tie-break/
   verifier 回収は EV を大きく毀損（−10〜−12.5）。**過剰棄権は EV負commitを防ぐ正しい設計**であり、
   緩和は gold-100 EV を悪化させる。→ **本番の gate/resolve は変更しない（byte同一）**。
2. **棄権理由内訳レポートを artifacts に出力**（本md + JSON）。✅
3. **関門2 非劣化**: 本番挙動を変えないため自明に非劣化。ledger に rejected として帰属記録
   （`docs/ai/experiment_ledger.jsonl`, axis=`gold-abstention-recovery-relaxation`）。

補足: investigator-solo の +3.0（7件, 精度0.71）は唯一の正部分集合だが、n=7・LOO不安定・
gold-100 過適合リスク（memory: local↔実LB無相関）のため、SOT-2478 昇格ゲート（関門2非劣化）と
実LB確認なしに本番採用しない。将来サイクルの候補として記録するに留める。
