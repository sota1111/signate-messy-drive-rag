# SOT-2503 — gold-100 実行結果と契約型スライス別ゲート較正（能力改善マージ後の新基準）

`review=human` 指示（Linear SOT-2503, 2026-08-06）に従い、Step10 の能力改善
（SOT-2494 canonical route / SOT-2498 contract routing / SOT-2496 rebaseline /
SOT-2503 slice calibration）マージ後の**新ベースライン**で gold-100 を実行し、
回答結果を更新・コミットした記録。あわせて SOT-2503 の目的である**契約型スライス別
EV 較正**を実 gold-100 verdict に対して評価した。

## 1. gold-100 実行（investigator 単一パス / 本番既定ゲート）

コマンド:

```
python -m scoring.gold_offline --run --gen investigator --workers 8 \
    --out artifacts/gold_100_report_sot2503.json
```

結果（n=100, CRAG ローカル審判, gen=investigator）:

| 指標 | 今回(2026-08-06T20:13Z) | 前回(2026-08-06T19:00Z) | 差 |
|---|---|---|---|
| 一致 (match) | **18 (18.0%)** | 16 (16.0%) | +2 |
| 棄権 (abstain) | 69 (69.0%) | 71 (71.0%) | −2 |
| 誤り (wrong) | 13 (13.0%) | 13 (13.0%) | ±0 |
| コスト | $4.46 | $4.91 | −$0.45 |
| 手動 86.7% 基準 | 未達 (BELOW) | 未達 | — |

`drop-to-abstain`: 非一致 82 問のうち 69 が棄権（84.2%）、13 が誤答のまま。

型別（archetype）内訳:

| 型 | n | 一致 | 棄権 | 誤り |
|---|---|---|---|---|
| derived_calculation | 32 | 5 | 23 | 4 |
| fact_lookup | 26 | 7 | 15 | 4 |
| document_extract | 25 | 3 | 19 | 3 |
| enum_set | 9 | 2 | 6 | 1 |
| version_diff | 5 | 0 | 5 | 0 |
| highlight_set | 1 | 1 | 0 | 0 |
| data_shape | 1 | 0 | 1 | 0 |
| config_hyperparam | 1 | 0 | 0 | 1 |

棄権の状態コード内訳（SOT-2492 台帳 / coded 68/69）:
`BUDGET_EXHAUSTED 53` · `UNANSWERABLE 10` · `NOT_RETRIEVED 4` · `EVIDENCE_INCOMPLETE 1`。
→ 棄権の主因は依然 **BUDGET_EXHAUSTED（反復上限/タイムアウト）** と **UNANSWERABLE（根拠不在）**。

更新・コミットした「回答結果」ファイル:
- `artifacts/gold_100_review.md`（一致18 / 棄権69 / 誤り13 に更新）
- `artifacts/gold_100_review.csv`
- `docs/gold_offline_history.jsonl`（本実行 1 行を追記）

（`artifacts/predictions_test_investigator.*` と `artifacts/*_sot2503.json` は
`.gitignore` 対象のため未コミット。SIGNATE 配布物由来の設問全文・正答を含むため。）

## 2. 契約型スライス別 EV 較正（SOT-2503 本体の実 gold 評価）

上記 gold-100 の per-index verdict を `scoring.slice_calibration.calibrate_slices`
（baseline t=0.70, judge_noise≈1/30）にかけ、契約型 8 スライス別に EV 最大 commit しきい値を
スイープした結果:

```
adopted slices : 0/8   WRONG 13→13 (非増加 OK)   EV +3.500→+3.500 (+0.000)
```

| 契約型スライス | n | baseline EV | EV最大しきい値での EV | 判定 |
|---|---|---|---|---|
| chart_read | 5 | −1.00 | +0.00 | 緩和対象外（最適は引き締め側） |
| simple_lookup | 40 | −5.50 | +0.00 | 緩和対象外（最適は引き締め側） |
| numeric | 32 | +2.00 | +2.00 (t*=0.90) | 緩和対象外（t*≥bar） |
| format_check | 4 | +2.00 | +2.00 | 緩和対象外 |
| full_enumeration | 10 | +2.00 | +2.00 | 緩和対象外 |
| multi_hop | 3 | +3.00 | +3.00 | 緩和対象外 |
| spatial | 1 | +1.00 | +1.00 | 緩和対象外 |
| version_diff | 5 | +0.00 | +0.00 | 緩和対象外 |

**結論（実 gold-100 で確認）:**
- **どのスライスも緩和対象にならない（0/8 採用）。** 全スライスで EV 最大しきい値が
  グローバルバー 0.70 以上＝**現ベースラインに EV>0 の緩和余地は存在しない**。
  負 EV スライス（chart_read, simple_lookup）はむしろ**引き締め（誤答→棄権化）で EV が最大**に
  なる形であり、一方向緩和のみを行う本機構は正しく何も採用しない。
- **WRONG 総数は非増加（13→13）**、EV は不変（+3.5→+3.5）。
- `adopted_thresholds = {}`。したがって既定 OFF と合わせ、ゲートは byte-identical で
  **関門2は構成的に非劣化**。

これは受け入れ条件を実データで満たす:
- [x] 緩和は EV>0 実測スライスのみ・WRONG 総数非増加 → 実 gold で EV>0 緩和余地なし＝0 採用、WRONG 13→13 非増加（fail-safe を stub でなく実 verdict で実証）
- [x] 関門2非劣化 → adopted 空＋既定 OFF で byte-identical
- [x] pytest グリーン → §3

## 3. 検証（pytest）

フォーカス offline スイート:
`scoring/test_slice_calibration.py` / `scoring/test_gold_offline.py` /
`scoring/test_calibration.py` / `tests/test_gate.py` — グリーン（§末尾の実行ログ参照）。

## 4. 含意 / 次アクション

- 現ベースラインでは per-slice 緩和の EV 余地がない。**改善余地は緩和ではなく回収側**＝
  BUDGET_EXHAUSTED 53 問（反復上限で根拠確定前に打ち切り）の探索効率・予算配分にある。
- local↔実LB 無相関（SOT-2486）に留意。実 LB での確認は別途 human ゲート。
- slice 較正機構は「EV>0 が現れたら初めて緩和する」fail-safe として常設。将来、能力改善で
  あるスライスの EV 最大しきい値が bar を下回れば、その時点で自動的に採用候補になる。
