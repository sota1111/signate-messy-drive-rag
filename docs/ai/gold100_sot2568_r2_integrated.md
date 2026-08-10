# Gold-100 統合実測 r2/r3（ディープリサーチ7子 SOT-2583〜2589 マージ後） — SOT-2568

- 実行日時: 2026-08-10T03:21–04:05Z（2回）
- コマンド（各回共通）:
  `PYTHONPATH=/tmp/genai_patch:. .venv/bin/python -m scoring.gold_offline --run --workers 8 --out artifacts/gold100_sot2568_<r2|r3_fmtonly>.json`
  （gen=investigator 単一パス, n=100, CRAG判定=codex）
- レポート: `artifacts/gold100_sot2568_r2.json` / `artifacts/gold100_sot2568_r3_fmtonly.json`（gitignore）
  ／履歴: `docs/gold_offline_history.jsonl`（recordedAt=03:32:59Z / 04:04Z 付近の2行）
- レビュー表: `artifacts/gold_100_review.md` / `.csv`（**r3 = r1セット+FORMAT_EVENTS 構成を反映**）
- コスト: r2 $8.99 ＋ r3 $12.16

## 事前処理（完了確認）

コーパス（Aug-04）不変。全ストアが現行 main（PR #116 マージ後）と整合していることを確認し、再生成は不要と判定:

| ストア | 生成時刻 | 整合根拠 |
| --- | --- | --- |
| retrieval index (chunks+embeddings) | 08-09 23:42 | 以降の extract 変更は flag-gated のみ（RAG_FORMAT_EVENTS OFF時 byte-identical） |
| evidence_index | 08-09 23:38 | builder 最終変更 08-09 09:20 |
| canonical_manifest / corpus_profile | 08-09 23:39/23:40 | builder 変更なし |
| document_registry（404レコード） | 08-09 23:40 | SOT-2583 以降 builder 変更なし |
| structure_store（format_events 21ファイル込） | 08-10 00:33 | SOT-2588 の diffpair 変更は flag-gated 追加のみで stored version_pairs を無効化しない |

## r2: 統合全ON（従来20フラグ＋新7フラグ）＝ **net 22（回帰 −10）**

新7フラグ: `RAG_DOCUMENT_REGISTRY / RAG_EVIDENCE_PACKET / RAG_FORMAT_EVENTS / RAG_POT_HARD_LANE / RAG_ENUM_SCAN / RAG_DIFF_ALIGN / RAG_EU_GATE`（従来セットは r1 = `docs/ai/gold100_sot2568_result.md` と同一）。

| 指標 | r1（08-09 17:35, 前回最良） | **r2 全ON** | 差分 |
| --- | --- | --- | --- |
| match | 39 | **31** | −8 ❌ |
| abstain | 54 | **60** | +6 |
| wrong | 7 | **9** | +2 ❌ |
| **net** | **32** | **22** | **−10** ❌ |
| cost | $10.31 | $8.99 | −$1.32 |

### 一問単位の遷移（r1→r2）

| 遷移 | 件数 | idx |
| --- | --- | --- |
| MATCH→ABSTAIN | **17** | 0,3,4,9,10,23,25,26,31,35,59,61,66,72,74,80,91 |
| ABSTAIN→MATCH | 9 | 19,54,60,62,79,85,88,90,94 |
| ABSTAIN→WRONG | 6 | 22,33,37,50,70,96 |
| WRONG→ABSTAIN | 4 | 12,14,16,65 |
| MATCH→WRONG | 1 | 21 |
| WRONG→MATCH | 1 | 49 |

### 効いたもの（リサーチの狙い通り）

- **旧誤答7件中5件を解消**: idx12（registry hard-abstain が指定ファイル過剰推論を遮断）/ idx14 / idx16 / idx65 → 安全棄権、idx49（docx コメント＝FORMAT_EVENTS）→ **正答化**。
- **BUDGET_EXHAUSTED 27→16**: packet/registry の事前解決で探索律速を実際に圧縮。旧 BUDGET だった idx54/62/85 が正答化。
- **enum レーンの実勝ち**: idx19（T04〜T17 全列挙）が symbolic 全数走査で正答化。document_extract 11→13。

### 壊したもの（統合の主因）

1. **PoT 強制レーン（RAG_POT_HARD_LANE）**: derived_calculation match 8→3。検算不成立時に terminal enforcement が正答をも棄権へ落とす。
2. **早期棄権（RAG_EU_GATE ＋ packet の budget contract）**: MATCH→ABSTAIN 17件の多くが iterations≤2 の即時 UNANSWERABLE（例 idx9/10/23/26/31/80）。「原則追加探索なし」の契約が、旧来なら数ターンで拾えていた証拠を拾わせない。UNANSWERABLE 18→32 に膨張。
3. **registry hard-abstain の過剰発火**: idx0/59/74 が `DOC_NOT_FOUND_AFTER_EXHAUSTIVE_MANIFEST_SCAN`（0 iterations）で棄権。旧経路では文書に到達し正答していた＝リゾルバの別名解決が不足したまま hard constraint だけ先に効いた。
4. **新規誤答6件**: enum「該当なし」断定（idx70 ↔ completeness certificate の狙いと逆）、diff-align の実質変更取り違え（idx22）、数値バインド誤り（idx33/50）、粒度過剰（idx37「22,000円/時間」vs gold「22,000円」、idx21「人事本部 人材戦略部」vs「人材戦略部長」）。

## r3: 増分アブレーション（r1セット＋RAG_FORMAT_EVENTS のみ）＝ **net 32（ベースライン維持）**

| 指標 | r1 | **r3** | 差分 |
| --- | --- | --- | --- |
| match | 39 | **39** | ±0 |
| abstain | 54 | **54** | ±0 |
| wrong | 7 | **7** | ±0 |
| **net** | **32** | **32** | **±0** ✅ |
| document_extract wrong | 2 | **1** | −1 ✅（idx49 正答化） |

- **idx49（docx コメント）WRONG→MATCH を統合外でも再現**。書式・コメントの決定論抽出（SOT-2585）は集計を落とさず狙いだけ直す唯一のクリーンな増分。
- 参考: r1→r3 でも 20 問がステータス変動（±2〜3 の集計差は run-to-run ノイズ帯。r2 の −10 はノイズを大きく超える）。

## 結論と次軸

- **事前処理は完了**（全ストア main 整合）。gold100 統合実測の結論: **7子の部品は個別に機能する**（旧誤答5/7解消・BUDGET 27→16・enum 実勝ち）が、**統合セットは decision 系3フラグの較正不足で net 22 に回帰**。champion serve path は全フラグ既定 OFF のため byte-identical（本番後退なし）。
- **測定セットの現時点最良 = r1セット＋RAG_FORMAT_EVENTS（net 32）**。tracked の `gold_100_review.*` はこの構成。
- 次軸（再統合の前提となる較正、いずれも「wrong→abstain は許すが match→abstain は許さない」方向）:
  1. registry hard-abstain を「リゾルバ未解決 ⇒ 旧探索へフォールバック」に緩和（idx0/59/74 は到達可能だった）。
  2. PoT レーンの verification 不成立時は棄権でなく free-form 回答へデグレード（derived 8→3 の回復）。
  3. EU gate / packet budget contract の早期棄権閾値を再較正（iters≤2 の即時 UNANSWERABLE 17件が対象）。
  4. enum の「該当なし」断定は universe 未カバー時に禁止（idx70、certificate の本来仕様）。
- 実 LB が一次 KPI（ローカル proxy ρ=−0.09）。本測定は故障分離の診断であり、昇格判断は実 LB 確認を経ること。
