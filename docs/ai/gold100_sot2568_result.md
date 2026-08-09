# Gold-100 実測（全事前処理 完了 ＋ SOT-2562/2564 精度フラグ統合後） — SOT-2568

- 実行日時: 2026-08-09T17:20–17:35Z（1回のみ / 所要 ~15.5 分）
- コマンド:
  `PYTHONPATH=/tmp/genai_patch:. .venv/bin/python -m scoring.gold_offline --run --workers 8 --out artifacts/gold100_sot2568.json`
  （gen=investigator 単一パス, n=100, CRAG判定=codex）
- レポート: `artifacts/gold100_sot2568.json`（gitignore）／履歴: `docs/gold_offline_history.jsonl`（recordedAt=2026-08-09T17:35:36Z）
- レビュー表: `artifacts/gold_100_review.md` / `.csv`（Private-strict, 追跡対象）
- コスト: **$10.31**

## 事前処理（全て完了・現行 main と整合）

`python -m src.rag.index` が単一コマンドで生成する 5 ストア＋索引を、現行 main（SOT-2562/2563/2564 マージ後）の
コードで**全て最新化**した。直前の試行は structure_store の生成ステップが中断し（`structure_store.json`
が 00:49 のまま＋`.tmp` 残置）、事前処理が未完了だった。本 run で残りを完了させ、全ストアを同一
コーパス（Aug-04, 不変）から整合させた:

| ストア | 生成物 | 状態 |
| --- | --- | --- |
| retrieval index | `index_store/chunks.jsonl` + `embeddings.npy` | 17:03（extract font-decoration dfe1ca0 反映済） |
| evidence_index | `artifacts/evidence_index.jsonl` | 16:58（coverage拡張 50cebd8 反映済） |
| canonical_manifest | `artifacts/canonical_manifest.json` | 16:59 |
| corpus_profile（事前復号） | `artifacts/corpus_profile.json` | 16:59 |
| **structure_store** | `artifacts/structure_store.json` | **17:15 再生成**（42 files / highlights 41 / charts 6 / pivots 5 / seating 1 / version_pairs 5、SOT-2564 highlight_extract CF 反映） |

コーパスは Aug-04 で不変、extract/index 経路のコードは 17:03 ビルド以降変更なしのため再embedは不要
（`docs/ai/gold100_after_error_fixes.md` SOT-2550 の index 再利用判断と同様）。structure_store のみ
SOT-2564（highlight_extract の条件付き書式ハイライト）が消費するため再生成した。

## フラグ構成（全改善 ON ＝ 統合効果を測定）

SOT-2527/2550 の統合セット（回答増11本）に、SOT-2562/2564 の精度フラグを追加した全 opt-in ON 構成:

```
# 回答増（SOT-2521〜2526 / SOT-2527 統合）
RAG_FIRST_MOVE_ROUTING=1  RAG_SPIN_DETECTION=1  RAG_ADAPTIVE_BUDGET=1  RAG_EVIDENCE_CACHE=1
RAG_BUDGET_BOUNDARY_RESEARCH=1  RAG_UNANSWERABLE_FALLBACK=1  RAG_PDF_OCR=1  RAG_SHARE_CORPUS_PROFILE=1
RAG_CANONICAL_MANIFEST=1  RAG_EVIDENCE_INDEX=1  RAG_STRUCTURE_STORE=1
# 誤答型別精度（SOT-2544〜2550）
RAG_GRANULARITY_NORMALIZATION=1  RAG_XLSX_EMBEDDED_IMAGE=1  RAG_CONFLICT_RESOLUTION=1  GATE_EXEC_CORRECT=1
# 残NG精度＋抽出強化（SOT-2562 / SOT-2564）— 本 run で新規に追加
RAG_NUMERIC_FEATURE_CORR=1  RAG_RELEVANCE_STRICT=1  RAG_HIGHLIGHT_EXTRA=1  RAG_FONT_EMPHASIS=1
RAG_FILE_GREP_INDEX_CANDIDATES=1
```

（SOT-2544 judge同義正規化・SOT-2546 境界オフバイワン・SOT-2509 参照選択決定論化はフラグ無し＝常時ON。
retrieval実験系 RAG_RERANK/RAG_RRF/RAG_FACT_INDEX 等は既定のまま OFF。）

## 結果サマリ（前回最良との差分）

| 指標 | SOT-2550(08-09 04:15) | **SOT-2568(08-09 17:35)** | 差分 |
| --- | --- | --- | --- |
| match | 34 | **39** | **+5** ✅ |
| abstain | 60 | **54** | **−6** ✅ |
| wrong | 6 | **7** | +1 |
| **net(match−wrong)** | 28 | **32** | **+4** ✅ |
| cost | $7.22 | $10.31 | +$3.09 |
| baseline(86.7%) | BELOW | BELOW | — |

- **統合改善は前進**: match +5 / net +4。非一致 61 のうち 54（88.5%）は安全棄権を維持し、precision の
  後退は wrong +1 に留まった（SIGNATE rubric近似 net は 28→32）。
- コスト増（+$3.09）は追加した精度ゲート／抽出強化による探索・検証ターン増が主因。

## 型別内訳（n / match / abstain / wrong）

```
config_hyperparam   n= 1  match= 1  abstain= 0  wrong=0
data_shape          n= 1  match= 0  abstain= 1  wrong=0
derived_calculation n=32  match= 8  abstain=22  wrong=2
document_extract    n=24  match=11  abstain=11  wrong=2
enum_set            n= 9  match= 2  abstain= 7  wrong=0
fact_lookup         n=26  match=14  abstain=11  wrong=1
highlight_set       n= 1  match= 1  abstain= 0  wrong=0
version_diff        n= 6  match= 2  abstain= 2  wrong=2
```

fact_lookup（14/26）・document_extract（11/24）が牽引。derived_calculation は依然 8/32 と最弱で、
棄権 22 の主軸（BUDGET_EXHAUSTED）が残る。

## 棄権の状態コード別（54件）

```
NOT_RETRIEVED          5
RETRIEVED_NOT_PARSED   2
PARSED_AMBIGUOUS       1
EVIDENCE_INCOMPLETE    1
UNANSWERABLE          18
BUDGET_EXHAUSTED      27
```

BUDGET_EXHAUSTED 27 / UNANSWERABLE 18 が棄権の主軸（45/54=83%）。これは SOT-2563 で診断済の
retrieval-turn starvation（探索ターンで予算枯渇）と整合し、次軸は固有名→canonical 文書 pre-resolve＋
スライス pre-inject（探索圧縮）と純粋多段数値の reasoning-budget 別軸に一致する。

## 誤答（wrong）7件

| idx | 型 | 概要 |
| --- | --- | --- |
| 1 | version_diff | スライド差分の抽出内容が gold と不一致 |
| 12 | fact_lookup | 指定ファイル未発見のまま別ファイルへ推論（過剰推論） |
| 14 | version_diff | 追記/削除ステップの取り違え |
| 16 | document_extract | 黄ハイライト「存在しない」と結論（抽出漏れ） |
| 49 | document_extract | docx コメント抽出不可を明示（機能未到達） |
| 65 | derived_calculation | 集計件数の誤り（8件） |
| 76 | derived_calculation | 増加額の誤り（73,260円） |

version_diff（idx1/14）と過剰推論（idx12/16/49）が残る誤答の主群。SOT-2562 の relevance-strict は
版差分の「関連変更」判定を厳格化したが、抽出内容自体の取り違え（idx1/14）は別軸（版差分の構造抽出精度）。

## 結論

- **事前処理は全て完了**（5 ストア＋索引が現行 main と整合）。gold100 を統合フラグ全 ON で実行し
  **match 39 / abstain 54 / wrong 7（net 32）** を確認。前回最良（net 28）を **+4** 更新した。
- champion serve path は全 opt-in フラグ既定 OFF のため byte-identical（本測定は測定用に全 ON）。
- 次軸（saturation ではない）: (1) 探索圧縮のための固有名 canonical pre-resolve＋スライス pre-inject、
  (2) derived_calculation の reasoning-budget 拡張、(3) version_diff の構造抽出精度。実 LB が一次 KPI。
