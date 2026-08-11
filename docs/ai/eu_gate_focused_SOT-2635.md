# SOT-2635 — EU gate (RAG_EU_GATE) focused calibration on the answer path

**Parent** SOT-2602 cycle3 7/8 (H7: EUゲート不在). Pairs the commit-time expected-utility gate with the
answer-increasing config so a recovered answer commits **only when EU>τ**, else 棄権維持.

## 前提の修正（SOT-2634 の結果を反映）

本issueは「SOT-2634 で KEEP になった回答増フラグ集合＋EUゲート」を前提にしていたが、**SOT-2634 (PR#151)
は spin/cap/prefill/preir を全て DROP** と確定した（KEEP集合＝空）。よって EU ゲートは cycle3 の
**trace移植回収 (SOT-2631-33)** と対で機能する位置づけに読み替えた。

## 配線の穴（実装で判明・修正済み）

- production gold100 (`gold_offline --run --gen investigator`) も `run_focused_gate.py` も **`investigator.answer_question`
  を使う**が、EU ゲート判定は `scoring.gate`（`gate_question`/`apply_gate`）側にしか無く、**この本経路では一度も
  発火していなかった**（issue記載の「本経路で一度も較正・実測されていない」の正体）。
- 修正: `investigator.answer_question` の **commit 直前に EU 判定を配線**（`_apply_answer_eu_gate`、既定OFF
  byte-identical）。answered/abstained を問わず **`interventions.eu_gate` に決定を全ケース記録**
  （tier / U / commit / flipped / signal bundle — SOT-2629 テレメトリ）。commit かつ EU≤τ のとき 棄権へ倒す。
- `eu_gate.py` は構造不変で **commit 閾値 τ を env-tunable 化**（`RAG_EU_GATE_TAU`、既定 0.0 = 従来の U>0）。
  単発pass には 合議 verifier 信号が無いため、確率モデルの主入力は決定論lane / canonical解決 / evidence充足 /
  数値exec一致。

## 測定（OFFICIAL flash-3.6・`run_focused_gate.py` 経由・番兵付き）

`scripts/sot2635_eu_gate.sh -9`（champion Wave A ＝ SOT-2634 baseline と同一env ＋ `RAG_EU_GATE=1`
`RAG_EU_GATE_TAU=-9`）。**τ=-9 プローブ**＝全 commit（answers は baseline と同一）にして、answered 全 idx の
**U をテレメトリ収集**。flip は (U, τ) の純関数なので、この1回で **任意の τ の focused net をオフライン確定**できる。

- 実行: `artifacts/focused_gate_sot2635_eu_taum9.json`（official=true）。番兵 9/10（回帰=idx30 のみ＝
  baseline でも回帰する非決定な脆弱番兵＝flash3.6 ノイズ床、EUゲート起因ではない。probe は flip 0）。
- 解析: `scripts/sot2635_analyze.py`。

### 決定的所見: **U は本経路で wrong と correct を分離しない**

| idx | verdict | U | commit | 種別 |
|---|---|---|---|---|
| 4  | Perfect   | **+0.260** | True | 正答 |
| 9  | Perfect   | **+0.260** | True | 正答 |
| 27 | Incorrect | **+0.260** | True | 誤答 |
| 47 | Incorrect | **+0.260** | True | 誤答 |
| 番兵 0/2/11/16 | Perfect/Acceptable | **+0.260** | True | 番兵 |
| 番兵 10/19/33/44/58 | Perfect | +0.365 | True | 番兵 |
| その他 target | Missing(棄権) | −0.125 | already_abstain | — |

**answered な正答(idx4,9)・誤答(idx27,47)・番兵4件が全て U=+0.260 に潰れる。** evidence を持つ単発回答は
route/正誤に関わらず同一の signal bundle（canonical=True・evidence_slots=True・verifier=False）になり、U が
一意に定まるため。

### τ sweep（`sot2635_analyze.py`、flip は決定論）

| τ | focused net | wrong倒し | correct誤倒し | 番兵回帰 |
|---|---|---|---|---|
| **0.000** | +0.00 | 0 | 0 | **0** ← 唯一の回帰ゼロ（但し flip 0＝OFFと同一） |
| 0.260 | +0.00 | 2 | 2 | 4 |
| 0.365 | +0.00 | 2 | 2 | 9 |

**誤答(idx27/47)を倒す τ≥0.260 は、同値の正答(idx4/9)と番兵4件も必ず倒す** → net は +0.00 のまま precision
だけ崩壊（番兵回帰4）。回帰ゼロで済む τ は 0.0（＝何も倒さない＝OFF と同一）のみ。

→ **SOT-2589 の offline 診断 `utility_discriminates=False` を、本番の単発pass・official flash-3.6 で再確認**。
弁別信号は 合議 の **answer-verifier 一致**（`answer_verifier_agrees`）であり、単発 investigator 経路はこれを
産まないため、U だけでは wrong 抑制と回収維持を両立できない。

## 結論 — cycle3 (SOT-2636) 統合構成

1. **単発pass の gold100 統合に `RAG_EU_GATE=1` は入れない。** 本経路では τ=0 は no-op（OFF と byte-identical）、
   τ>0.26 は「全部棄権」型の過剰締め（net 0・番兵回帰）で、issue が禁じた過剰締めに該当する。**RAG_EU_GATE は
   既定OFF のまま据え置く**（レジストリ変更なし）。
2. **配線とテレメトリは残す（恒久価値）。** `answer_question` の commit直前 EU 判定＋`eu_gate.decision` 全ケース
   記録は、将来 EU ゲートを効かせる前提（下記）が揃ったときに即再測定できる土台。criterion #2 は達成。
3. **EU ゲートを効かせるには弁別信号が要る。** (a) 回収を 合議/verifier 経路（`gate.gate_question` = SOT-2589 で
   verifier 信号込みで配線済み）に通す、または (b) 単発pass に異種 answer-verifier を1段足して
   `answer_verifier_agrees` を実測する。cycle3 の trace移植回収は前者に載せるのが筋。
4. **単発pass の唯一の precision レバーは hard blocker**（数値 exec 不一致 / source conflict / parser不能 /
   corpus absent）で、これは U と独立に発火する。本 focused では answered な誤答(idx27/47)が数値exec不一致型で
   なかったため未捕捉。数値型の誤答には引き続き exec gate（GATE_EXEC_CORRECT）が効く。

## 生成物

- 配線: `src/rag/agent/investigator.py`（`_apply_answer_eu_gate` / `_eu_signals_from_investigation`、既定OFF）
- 較正knob: `src/rag/agent/eu_gate.py`（`commit_threshold()`=`RAG_EU_GATE_TAU` 既定0.0 / `base_prior()`=`RAG_EU_GATE_BASE`、構造不変）
- 実行driver: `scripts/sot2635_eu_gate.sh`（champion Wave A ＋ EUゲートON、`run_focused_gate.py` 経由）
- 解析: `scripts/sot2635_analyze.py`
- gate JSON: `artifacts/focused_gate_sot2635_eu_taum9.json`（official=true）
- テスト: `tests/test_eu_gate.py`（τ/base knob）・`tests/test_eu_gate_answer_path.py`（配線・flip・全ケース記録・OFF byte-identical）
- ledger: `docs/ai/experiment_ledger.jsonl`
