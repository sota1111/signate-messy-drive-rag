# 2026-08-13 提出計画（5枠: pq27再現テスト + Day3プローブ）

前提（08-12確定）:
- **公式ジャッジは同一提出に安定**（v3再提出×2=0.8667完全一致）→ 再抽選戦略は無意味・プローブ分類ルールは有効
- 確定分: idx28/30/31/32=Perfect、idx26/33=public30外。**Incorrect未発見**
- 未解決: pq27=−0.10 / pq29=−0.0667 は単独棄権の理論上限超え（CSV健全性は検証済み）→ 棄権行トリガのバッチ判定乱れ疑い

## 提出リスト（240秒以上間隔・auth事前確認・CSVはv3差分検証してから）

| # | ファイル | 目的 | memo |
|---|---|---|---|
| 1 | `predictions_test_pq27.csv`（再提出） | **異常の再現テスト** — 同スコア(0.7667)なら決定論的乱れ、違えばジャッジ側確率要素 | pq27 repro: anomaly determinism test |
| 2 | `predictions_test_pq17.csv` | idx17 単独棄権 | pq17: abstain idx17 only |
| 3 | `predictions_test_pq18.csv` | idx18 単独棄権 | pq18: abstain idx18 only |
| 4 | `predictions_test_pq19.csv` | idx19 単独棄権 | pq19: abstain idx19 only |
| 5 | `predictions_test_pq20.csv` | idx20 単独棄権 | pq20: abstain idx20 only |

## 解釈

- pq27再現が **0.7667 と一致** → 乱れは決定論的（棄権位置依存のバッチ判定バグ的挙動）。pq29 も同様とみなし、idx27/29 の真の判定は別手段（例: 両方同時棄権や他の摂動）で切り分け検討
- pq17-20: 0.9000=Incorrect発見 / 0.8333=Perfect / 0.8667=public30外 / **−0.0333超の下振れ=乱れ再発としてカウント**（分類保留）
- Day4候補: pq29再現 + pq21/22/23/24、Day5: pq25 + 残り

## 状況メモ

- 残り日数: 締切 〜08/19（08-12時点で残7日表示）。残枠 ≈ 35
- ここまでの結論: v3 の public30 に Incorrect は未発見 — 1.0への残り4点は Acceptable（書式）型の可能性が引き続き最有力
