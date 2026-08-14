# Worker Report — SOT-2715（solo / claude:opus）

## Summary
最終報告版差分 idx1 の gold構造一致 direct-commit を試みたが、**実 judge 較正により非ハードコードでは
Perfect/Acceptable 到達不能と確定 → honest 撤退（rejected）**。受け入れ条件②（gold値ハードコードなし／OFF時
byte-identical）を守り、③（撤退判断の根拠記録）を充足。serve パス無改変（byte-identical 確認済み）。

## Changed Files
- `docs/ai/SOT-2715_idx1_vdiff_direct_commit_withdrawal.md` — 撤退記録（gold構造・構造化レコード・実 judge 較正表・撤退判断）
- `src/rag/agent/vdiff_direct_lane.py` — 削除比較表→改善幅要約 置換クラスを direct-commit 対象にしない旨のガードコメント（コメントのみ・serve 無改変）
- `docs/ai/experiment_ledger.jsonl` — axis=vdiff idx1 gold構造一致 direct-commit / result=rejected の1行追記

## Commands Run
- `scoring.crag`（公式 rubric+strict, codex/GPT-5.x）で idx1 gold と候補を対採点（5候補×5 + 6候補×3）
- `collapsed_table_frames` で構造化レコード確認（title/columns/metrics/header_label）
- `vdiff_direct_lane.resolve(idx1)` を全フラグ組で確認 → 全て None（byte-identical）
- `pytest tests/test_diff_store.py tests/test_version_diff_pipeline.py scoring/test_vdiff_direct_commit.py -q` → **45 passed**
- `scripts/check_flag_manifest.py` → exit 0（コメント内フラグ名は reader 非該当で無害）

## 実 judge 較正（要点）
- gold: `最終報告スライド7の、中間段階と最終モデルの性能比較表（AUC-ROC/F1-macro/Accuracyの中間実測値と最終値）を削除し、改善幅のみを示す1行要約に置換した`
- gold逐語 / 主語省略のみ → **Perfect ×5**。
- `性能比較表`→`指標比較表` → Incorrect ×3（`性能`必須。title から導出可）。
- `中間実測値と最終値` を非逐語化（`中間値と最終値`／`中間実測値と最終実測値`／`・区切り`／`中間と最終の値`／metricsのみ）→ 全て Incorrect（3〜5サンプル安定）。
- `実測値` の非対称接尾は構造化レコード（列名=素の `中間`/`最終`）に非在 → 非ハードコード導出不能。通す唯一の道が gold ハードコード＝条件②違反。

## Acceptance Criteria
- [ ] idx1 が focused で Perfect/Acceptable・番兵10/10 — **未達（原理的に非ハードコードでは不能）→ 撤退**
- [x] gold値ハードコードなし／OFF時 byte-identical — **維持**（resolve は idx1 で全フラグ None、serve 無改変）
- [x] 改善不成立時は撤退判断を根拠つき記録 — **充足**（withdrawal doc + ledger + code コメント）

## Risks
- なし（serve 無改変・byte-identical）。将来サイクルが同クラスを再挑戦しないようガードコメント＋ledger で明示。
  再挑戦は新証拠（judge 緩和 or `実測値` の正当な構造抽出）が出た時のみ。

## Linear Report: POSTED
## Acceptance: PASS
## Next Action: READY_FOR_REVIEW
