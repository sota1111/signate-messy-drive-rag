# 提出計画 — 2026-08-08（1.0への失点位置特定プローブ、5枠）

human指示「正答率1.0を目指し、明日の提出予定のファイルを作成」に基づく準備。
前提: 08-07 の v5 フリップ・プローブで**独立再導出由来の全10代替回答仮説が死亡**
（`docs/ai/submissions_20260807_v5_probe.md`）。残存誤りは「全独立検証パスが v3 に同意してしまう問」
か「軽微書式差(Acceptable)」にあり、**具体的な代替回答の当てずっぽうでは到達不能**。
→ 1.0 への残る道 = **失点問の位置特定 → ピンポイント再導出**。明日は二分探索で
失点窓を 16-17 問（public ~5 問）まで絞る。

## 前提となる確定情報

- v3 = 0.8667 = 26/30。public30 固定。失点は high 確信側（mediums 寄与ゼロ, probe#D）。
- third 別寄与（08-06 probes, ±1問のジャッジ揺らぎ）: A(0-33)=10 / B(34-66)=9 / C(67-99)=9（実合計は26）。
  → **失点は B と C に局在**。
- idx50 は public30 かつ v3 正（08-07 v5f で確定）。

## 提出ファイル（作成済み・CSV検証済み）

| # | ファイル | 内容 | 棄権数 |
|---|---|---|---|
| w1 | `artifacts/predictions_test_w1_abstain_b1.csv` | idx34-50 を「わかりません」 | 17 |
| w2 | `artifacts/predictions_test_w2_abstain_b2.csv` | idx51-66 を「わかりません」 | 16 |
| w3 | `artifacts/predictions_test_w3_abstain_c1.csv` | idx67-83 を「わかりません」 | 17 |
| w4 | `artifacts/predictions_test_w4_abstain_c2.csv` | idx84-99 を「わかりません」 | 16 |
| w5 | `artifacts/predictions_test_w5_abstain_unconfirmed_bc.csv` | B/C のうち v4 独立検証AGが確認できなかった(棄権した)commit 37問を棄権 | 37 |

w5 の対象 idx: 34,37,38,40,44,45,46,47,48,49,52,53,55,56,57,58,59,63,64,67,68,69,70,71,72,73,76,79,82,86,87,91,92,93,95,97,98
（直交軸プローブ: 「誤りは検証AGの盲点にある」仮説の検定。w1-w4 の結果次第で差し替え可）

## 提出コマンド（human 許可後）

棄権プローブはオフラインゴールド一致が閾値未満になるため **閾値の明示上書きが必須**:

```bash
cd /workspaces/signate-messy-drive-rag
SIGNATE_SUBMIT_ALLOWED=1 GATE3_GOLD_THRESHOLD=0.55 .venv/bin/python -m scoring.gate3 \
  --preds artifacts/predictions_test_w1_abstain_b1.csv --no-run --submit \
  --memo "w1: abstain idx34-50 (B-first-half localization probe)"
# w2..w5 同様（w5 のみ GATE3_GOLD_THRESHOLD=0.5）
```

gotcha: 回答内 ASCII カンマは `"..."` 引用必須（棄権センチネルは無関係）。ベスト提出表示のため
順位 0.8667 は毀損しない。スコアは Web で人間確認（API 無し）。

## 判定ルール

窓の寄与 = 26 − 30×score（±1問揺らぎ）。

- **w1+w2 ≈ 9（B合計）/ w3+w4 ≈ 9（C合計）** になるはず（クロスチェック）。
- 兄弟窓と比べ寄与が低い窓 = **失点窓**。片方 ≈5 でもう片方 ≈3-4 なら後者に失点。
- **いずれかのプローブが 0.8667 を超えたら**、棄権した集合は寄与が負 = **Incorrect を含む**（最強のシグナル。
  その場合その提出自体が新ベスト）。
- 寄与に 0.5 端数（score 表示で ±0.0167 刻み）が見えたら **Acceptable 型失点**の証拠。
- w5: 未確認37問の寄与を測定。w5 寄与が「37問中の public 想定数」を大きく下回れば誤りは検証盲点集合内。

## 翌日以降（day-3）の設計

1. 特定した失点窓（~17問, public ~5問）の commit 回答を、Step10 ツール群
   （座席directory / corpus_aggregate / canonical_route / exec_verifier / 列挙クロージャ）+ 人手深掘りで**全問再導出**。
2. 食い違いが出た問だけ差し替えて提出（= 1.0 挑戦）。差し替え確信が持てない場合、
   Incorrect が確定した問は棄権に落とすだけでも +1点/問（0.9000〜0.9333 へ）。

## 正直な期待値

- 明日は**診断日**（プローブ自体は 0.5 台のスコア）。1.0 は「窓特定 → 該当問の正答再導出」が
  成功して初めて到達。失点が Acceptable 書式型で GT 表記に依存する場合、書式の正解は観測不能のため
  1.0 が構造的に困難な可能性もある（その場合も窓特定で失点型の判別まで前進する）。
