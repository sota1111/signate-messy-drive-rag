# Worker Report — SOT-2607 (solo=claude:opus)

## Summary
反転アーキ PLAN SOT-2602 の **Wave A2**: `numeric` / derived_calculation 契約を LLM ループ非経由の
決定論パイプライン化した。Stage0 ルーター(SOT-2603)が `numeric` と判定 → 新規
`src/rag/agent/pipelines/numeric.py` が **canonical_route → compute → exec_verifier 照合 → Stage3 整形**
で値を導出し、対象定義を一意に確定できない/独立再計算と一致しない場合は `None` を返して従来 LLM ループへ
フォールバックする(precision-first、回答数を減らさない)。Wave A1 `version_diff`(SOT-2605)と同じ形。

## Changed Files
- `src/rag/agent/pipelines/numeric.py` — 新規。numeric 決定論パイプライン本体（自己登録）。
- `src/rag/agent/pipelines/__init__.py` — `numeric` を import して Stage0 レジストリへ配線。
- `tests/test_numeric_pipeline.py` — 新規 19 件（network/corpus-free）。
- `tests/test_det_pipeline.py` — Stage0 テストを numeric 実登録へ更新（合成契約名で mechanics 分離、
  `test_wave_a2_wires_numeric` 追加、配線テストは `replace=True`/`unregister`）。
- `tests/test_formatting.py` — 配線 2 件の `dp.register("numeric",…)` を `replace=True` 化。
- `docs/ai/experiment_ledger.jsonl` — `axis=det-pipeline-numeric` を append。

## 設計（発火スコープ = precision-first）
決定論で対象定義が一意確定する時のみ commit する:
1. **Stage1**: `resolve_project`(glossary)で単一案件解決 → `canonical_route(kind='train')` の
   tabular primary を確定（chunk 検索を迂回）。単一案件が解決しない/非表なら `None`。
2. **Stage2**: 認識済み単一集約（平均/合計/最大/最小/中央値/標準偏差/分散/件数）が **ちょうど1つ**、
   その対象列（NFC ≥2 字が質問に literal 出現し表の列と **一意一致**）が **ちょうど1つ** の時のみ
   `compute` 実行。`_EXCLUSION_CUES`（ハイライト/回帰/予測/係数/差額/絶対値/版/相関 …）と
   `numeric_requirements` の `ratio`/`unit` は事前に除外。
3. **独立再計算 + 照合**: `dropna` 再定式化の第2 compute を `exec_verifier.compare_execution`
   （値/件数/対象列/単位 + SOT-2508 契約検証）で照合。`EXEC_MATCH` のみ commit、それ以外
   （曖昧/不一致/契約違反/確証不能）は `None`。小数第N位のみ honor（純表示変換）。
- ハードコードなし（idx/答え/列/scale 非注入・質問+corpus から自己導出）。
- `RAG_DET_PIPELINE_ROUTER` 既定 **OFF** ⇒ `resolve` が `enabled()` 前で短絡 ⇒ champion serve path
  byte-identical。

## Commands Run
- `.venv/bin/python -m pytest tests/test_numeric_pipeline.py …` → 19 passed（新規）。
- 実コーパス grounding 実測: 京橋信用ソリューションズ train.csv `age` 平均 →
  pipeline value=`40.95101002654084`（compute 生値と一致・`exec_match`）。
- `.venv/bin/python -m pytest --ignore=tests/test_gate.py --ignore=tests/test_tiebreak.py` →
  **1161 passed**（`test_gate.py`/`test_tiebreak.py` は live-exec の pre-existing hang=main でも同様）。
- `py_compile` OK。

## Acceptance Criteria
- [x] idx6/47/63/97 が match または確証不能時は安全棄権/フォールバック・wrong 増ゼロ —
  4 件とも「単一列×単純集約」でない（差額/建設年逆引き/回帰予測/ハイライト幾何）ため `None` フォール
  バック ⇒ champion の現 match（6/47/63=わかりません、97=18948）を保存。単体テストで idx6/63/97 型を明示検証。
- [x] numeric 以外は経路不変・offline green — 既定 OFF で byte-identical、offline 1161 passed。
- [x] ハードコードなし・ledger 帰属記録 — 自己導出のみ、`axis=det-pipeline-numeric` を ledger へ記録。

## Risks
- 発火スコープは意図的に狭い（単一列集約のみ）。net の実効果は **gold100 未実行**（Wave A 末の統合子
  SOT-2610 で一括実測）＝未測定、**実LB確認が最終ゲート**（local proxy ρ=-0.09）。
- 英語集約キーワード（max/min/sum …）の部分一致は理論上の誤検知余地があるが、単一列一意一致 +
  独立再計算 + 契約検証 + 除外キューの多段ガードで wrong commit を抑止（かつ既定 OFF）。

## Linear Report: POSTED
## Acceptance: PASS
## Next Action: READY_FOR_REVIEW
