# SOT-2633 (G3) — SOT-2619 の flash 実効 focused 実測 ＋ idx56 chart 目盛り拡張の成立判定

PLAN SOT-2602 cycle3 5/8（G3）。trace ドシエ `docs/ai/sonnet_trace_dossier.md`（SOT-2630）の G3 割当
（idx84/74/56）と、SOT-2619 の 5 ターゲット（idx74/84/78/62/75）を対象に、**(a) 現 main champion の
flash 実効を focused 実測**し、**(b) idx56 の決定論拡張の成立/非成立を判定**した記録。serve path は不変更
（`.py` に触れない measurement）。gold 値の転記なし。

---

## (a) SOT-2619 修正の flash 実効（focused gate, OFFICIAL flash-3.6, 一度だけ）

- runner: `scripts/sot2633_focused.sh` → `scripts/run_focused_gate.py --label sot2633_g3_2619verify`
- env: SOT-2610 Wave A net40 champion（`RAG_DET_PIPELINE_ROUTER=1`、`RAG_ANSWER_NORMALIZE` 既定 ON）＝
  **現 main そのまま**。新規フラグ追加なし。model=gemini-3.6-flash / VERTEX_LOCATION=global / official=true。
- 結果 JSON: `artifacts/focused_gate_sot2633_g3_2619verify.json`（recordedAt 2026-08-11T04:46:13Z）
- **GATE PASS**・番兵 10/10 MATCH・regressions=[]（既存 MATCH 非劣化）

| idx | archetype | champion(SOT-2610) | 現 main verdict | 判定 |
|---:|---|---|---|---|
| **74** | version_diff | **WRONG** | **Perfect**（route=deterministic） | ✅ **SOT-2619 で回収（実装不要）** |
| 84 | fact_lookup | WRONG | Incorrect（`5ページ（スライド6）`） | ✗ 未回収（naturalize は "5→5ページ" 化したが `（スライド6）` gloss が残り judge 不一致） |
| 78 | fact_lookup | ABSTAIN | Incorrect（冗長 over-answer） | ✗ 未回収（SOT-2619 が「意味保存 normalizer で不可・報告のみ」と既述の通り。champion は ABSTAIN=非wrong） |
| 62 | fact_lookup | ABSTAIN/WRONG | Incorrect（順位対応欠落） | ✗ 未回収（内容粒度不足＝生成側。SOT-2619 既述） |
| 75 | fact_lookup | WRONG | Incorrect（`第4週（4週目）`） | ✗ 未回収（gloss-dedup は `第4週目（第4週）` を畳むが flash は語順違いの `第4週（4週目）` を出力、規則外） |

### 確定事項（(b) 以降のベースライン）

- **idx74 = 実装不要で回収済み**。SOT-2619 の version_diff 決定論 naturalization（`{label}が{before}から
  {after}に変更`、`naturalize=False`）が **flash でも決定論経路（route=deterministic, wall 0.7s）で発火し
  Perfect**。champion Wave A net40 は idx74 を WRONG に計上していたため、**現 main は focused 上 idx74 分
  wrong−1 / match+1**（＝ champion 実効は net40 → 実質 net42 相当）。この分は SOT-2636 統合測定のベースライン
  に反映する（本 issue は focused のみ・gold100 全量は回さない）。
- idx84/78/62/75 は flash では未回収。SOT-2619 の naturalize/normalizer は「値は正、表現同値」問題のみを狙った
  もので、**idx84 の付随 gloss `（スライド6）`・idx75 の語順違い gloss・idx78 の過剰網羅・idx62 の内容粒度不足**は
  いずれも意味保存 normalizer の対象外（SOT-2619 の commit note と一致）。これらは新規の生成側/正規化側の別軸課題
  であり、本 issue のスコープ（SOT-2619 効果の確定＋idx56）外。**焦らず据置**（wrong を増やさない）。

---

## (b) idx56（ipynb 可視化 y 軸目盛りの最大値）の決定論拡張 — 判定: **非成立（移植しない）**

trace ドシエ（SOT-2630, idx56 節）と Wave A4 実装（`src/rag/agent/pipelines/chart_spatial.py`）を突合し、
**決定論拡張は成立しない**と判定した。根拠:

1. **機械的な数値源が存在しない**。idx56 は `reports/figures/target_distribution.png`（01_eda.ipynb に
   matplotlib でレンダリングされた埋込画像）の **y 軸目盛りの最大値**を問う。`read_chart_values`（chart_numcache）
   の厳密経路は **numCache（プロット元数値）または元列再集計**に依存するが、**ipynb レンダリング画像は numCache を
   持たず、元列（度数分布の軸目盛り）も存在しない**（Wave A4 が `chart_spatial.py` L33-35 で既に明記:
   「ipynb にレンダリングされた matplotlib 図（idx56 型）は numCache を持たず、y 軸目盛りは vision の目視読みで
   しか取れないため、この pipeline は該当を grounded しない」）。
2. **唯一の到達経路が vision（非決定論）**。Sonnet も champion(cycle2) も `caption_image` で軸目盛を目視転記して
   到達している（conf 0.45〜1.0 と不安定）。しかし chart 決定論パイプラインは **vision の目視値読みを禁止**して
   いる（SOT-2507 踏襲、`chart_spatial.py` recognizer 1 のコメント「Vision の目視値読みは禁止」）。目盛りの
   等差性チェックや複数サンプル安定性ガードを足しても、**根底の値取得が OCR/vision に依存する限り決定論化は
   できない**（本 issue の指示「機械的に確定できない場合は移植しない」に該当）。
3. したがって **Wave A4 の安全棄権を維持**（idx56 は numeric 契約へ分類され厳密経路なし＝棄権のまま）。新規挙動を
   足さないため **`RAG_G3_CHART_PORT` フラグは導入せず、serve path は byte-identical**（gold ハードコードも当然なし）。

### 非成立の帰結

- chart 系既存 MATCH（idx10/33/44/58）は本 issue で不変更（focused gate の番兵 idx10/33/44/58 が 10/10 MATCH で
  非回帰を確認済み）。
- idx56 を決定論で回収するには「ipynb 実行→figure の数値メタ（axis limits/ticks）を Agg backend から機械取得」等の
  **別軸のツール新設**が必要で、本 issue のスコープ外。SOT-2636 統合の昇格候補には**載せない**（vision 依存＝
  precision リスク、cycle2 の「積極回答＝wrong 増」教訓）。

---

## 受け入れ条件との対応

- [x] SOT-2619 修正の flash 実効（idx74/84/78/62/75 の verdict）が focused gate で確定・記録 → 上表・
  `artifacts/focused_gate_sot2633_g3_2619verify.json`・ledger。**idx74 = Perfect 回収を確定**。
- [x] idx56 の決定論拡張の成立/非成立を判定 → **非成立**（数値源なし・vision 依存＝SOT-2507 禁止）。成立しないため
  focused gate PASS 条件（新規挙動）は不発生。番兵 10/10 で既存 chart MATCH 非回帰は確認済み。
- [x] OFF 時 byte-identical・gold 値ハードコードなし → serve path（`src/**.py`）不変更。追加は measurement runner
  （`scripts/sot2633_focused.sh`）と本 doc と ledger のみ。
