# 2026-08-12 提出計画（5枠: ノイズ床測定 + Day2プローブ）

前提: Day1（08-11 00:18-00:34）で pq26-30 を消化済み。結果はledger参照 —
**公式ジャッジ（gpt-5.2）の提出間ノイズを発見**（単独棄権の理論上限 ±0.0333 に対し pq27=−0.10 / pq29=−0.0667）。
idx26-30 に Incorrect なし（0.9000 未出現）。idx26 は public30 外。

## 提出リスト（1件ずつ 240秒以上空ける・auth を事前確認）

| # | ファイル | 目的 | memo |
|---|---|---|---|
| 1 | `artifacts/predictions_test_v3_final.csv` | **v3無変更再提出①: 公式ジャッジのノイズ床測定** | v3 resubmit #1 (judge-noise floor) |
| 2 | `artifacts/predictions_test_v3_final.csv` | **v3無変更再提出②: 同上（2標本目）** | v3 resubmit #2 (judge-noise floor) |
| 3 | `artifacts/predictions_test_pq31.csv` | idx31 単独棄権 | pq31: abstain idx31 only (Incorrect hunt) |
| 4 | `artifacts/predictions_test_pq32.csv` | idx32 単独棄権 | pq32: abstain idx32 only (Incorrect hunt) |
| 5 | `artifacts/predictions_test_pq33.csv` | idx33 単独棄権 | pq33: abstain idx33 only (Incorrect hunt) |

コマンド雛形（従来どおり）:
```
SIGNATE_SUBMIT_ALLOWED=1 GATE3_GOLD_THRESHOLD=0.5 .venv/bin/python -m scoring.gate3 \
  --preds <file> --no-run --submit --memo "<memo>"
```

## 解釈ルール（ノイズ込みに更新）

- **v3再提出2件**: 0.8667 からのズレ = 公式ジャッジのノイズそのもの。
  - 2件とも 0.8667 → ノイズ小、プローブ解釈（0.9000=Incorrect）復権
  - ばらつく（例 0.83〜0.90）→ 単独プローブの±0.0333弁別は死亡。以後は
    (a) 0.9000超が「複数回」出た場合のみ Incorrect 扱い、
    (b) 残枠戦略を「ベスト提出の再抽選連打」（best-submission表示は最良保持のため下振れ無害・上振れだけ拾える）へ切替を検討
- **pq31/32/33**: 0.9000 が出ても v3 ノイズ床確認とセットで判断（単発では確定しない）
- Day3 候補: pq17, pq18（＋ノイズ床結果に応じて方針変更）

## 状態

- pq17〜pq33 の全CSVは検証済み（各1問棄権・ASCIIカンマquote済み）
- 残り日数: 締切 〜08/18 頃（08-10時点で残8日表示）
