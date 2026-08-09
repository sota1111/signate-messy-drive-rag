# file_grep 全走査フォールバックの根絶 — 事前処理 (SOT-2562)

> 親: SOT-2550（誤答型別対策後の統合実測＋原因調査）。本子は「時間切れの事前処理」follow-up。
> 検証: focused/offline（gold100 は**回していない**＝回答・champion 不変）。2026-08-09。

## 1. 背景（根拠は `docs/ai/abstain_wrong_root_cause_SOT-2550.md`）

棄権60の主因＝「1本の遅い `file_grep` 全コーパス走査」。BUDGET_EXHAUSTED 44/60・うち 40 が
`file_grep` 経由・timeout 支配 34件。棄権問の所要は中央値 308s・最大 658s（match 問は中央値 22s）。
`file_grep` は evidence_index を**ミス**すると全ファイルを `_extract`（Office復号+PDF/OCR）する
全走査へフォールバックし、その1本が時間予算を溶かす。

**実測での裏取り（本作業）:** `corpus.walk()` 自体は 403 ファイルで **0.017s**（cheap）。遅さは
`_extract` にあり per-file で pdf 0.44s / xlsx 1–2s ⇒ 403 全走査で 300–600s = root-cause 実測と一致。

## 2. 対策（既定 OFF フラグ配線・champion serve は byte-identical）

### (1) 索引カバレッジ拡張 — `src/rag/index/evidence_index.py::scan_doc`
- `sheet` 型: 各 `[シート: 名]` のシート名自体を索引化（シート名を挙げる発見系クエリが index ヒット）。
- `heading` 型: スライドタイトル（`[スライドN]` 直後の最初の本文行）と markdown 見出し（`# …`）を索引化。
- 加算的・既存の number/person/alias/… 抽出は不変。**次回索引再ビルドで有効**化。

### (2) 索引ミス時の候補集合フォールバック — `evidence_index.candidate_files` + `file_grep`
索引ミス時に全ファイル `_extract` へ落ちる前に、**抽出せず**に候補ファイル集合（≤N）へ絞り、候補のみ走査:
- **filename NFC 一致**（`walk()` のみ・抽出なし）＝「対象ファイル名が設問に出る」発見系を捕捉。
- **index の value/key トークン重なり**（既存 76MB 索引を参照）＝本文語ベースの発見系を捕捉。
- スコア = ファイル毎の**distinct** strong トークン長和（distinctive な社名/固有名を重く）＋ filename ボーナス。
  per-entry 合算ではないので、大量索引ファイルが汎用語（`train`/`xlsx`）連呼で支配することはない。
- 候補ゼロなら `files_scanned=0` で早期返却＝**全走査を発生させない**。

フラグ: `RAG_FILE_GREP_INDEX_CANDIDATES`（既定OFF・`RAG_EVIDENCE_INDEX` の上に opt-in）/
`RAG_FILE_GREP_MAX_CANDIDATES`（既定12）。

## 3. focused/offline 検証（実コーパス403・prebuilt evidence_index 76MB/248,932エントリ）

### 3a. 候補導出（抽出なし・時間切れ idx群の代表 discovery 語）

| idx | discovery 語（設問由来） | 候補数 | target 順位 | 所要 |
| --- | --- | ---: | ---: | ---: |
| 48 | ニューヨーク不動産市場の最新動向調査 | 12 | **#0** | ~0.7s |
| 27 | 恒一会 かえで総合病院 提案書 スコープ対象外 | 12 | **#0** | ~0.13s |
| 63 | 青葉与信マネジメント train.xlsx 回帰係数 | 12 | **#1** | ~0.12s |
| 50 | 東都人材プラットフォーム データサイエンティスト調査 | 12 | **#0** | ~0.18s |
| 39 | 青潮モビリティサービス train.xlsx Sheet1 グラフ1 | 12 | **#0** | ~0.14s |

全ケースで対象ファイルが候補 top-12 に入り、導出は 0.1–0.7s。

### 3b. `file_grep`（候補フォールバック ON）

| query | source | files_scanned | elapsed |
| --- | --- | ---: | ---: |
| ニューヨーク不動産市場の最新動向調査 | `evidence_index_candidates` | 12 | **0.74s** |
| スコープ対象外 | `evidence_index_candidates` | 12 | 0.86s |
| One-Hot Encoding カテゴリ数閾値 | `evidence_index_candidates` | 12 | 0.32s |

**全走査フォールバック（全ファイル `_extract`）は発生せず**、`files_scanned` は上限12・所要は残予算内
（従来 300–658s → 0.3–0.9s）。単一 `file_grep` 呼び出しの予算暴走が消える。

### 3c. precision 非劣化
- champion serve は両フラグ既定 OFF で **byte-identical**（`file_grep` の全走査経路は `_scan_refs` への
  純リファクタで意味不変。`test_file_grep_disabled_never_consults_index` / `scoring/test_file_grep`
  実コーパス smoke 緑）。既に match の問いは champion（フラグOFF）で成立しており不変。
- 候補フォールバックは opt-in。候補が対象を外した場合でも、従来はその全走査が timeout →
  BUDGET_EXHAUSTED 棄権（回答ゼロ）だったため、候補限定 ≥ timeout-棄権（下振れなし）。

### 3d. 単体テスト（offline）
`tests/test_evidence_index.py`（sheet/heading カバレッジ・candidate_files ランキング）＋
`tests/test_file_grep_index.py`（候補フォールバックが走査を上限化 / フラグOFF時は従来全走査）を追加。
関連 offline suite（index/tools/scoring）**64 tests green**。

## 4. 索引再ビルド手順
```
.venv/bin/python -m src.rag.index                 # 全索引（chunks+embeddings+各side index）
.venv/bin/python -m src.rag.index.evidence_index  # 本 typed 索引のみ（embedding なし）
```
sheet/heading カバレッジ拡張は再ビルドで有効。候補フォールバックは現行 artifact のまま機能。

## 5. 姉妹 issue との分担
- 本 SOT-2562 = **事前処理**（索引カバレッジ拡張＋候補集合絞り込みで全走査を発生させない）。
- SOT-2563 = 全走査への **per-call デッドライン＋協調キャンセル**（単一呼び出しの予算暴走を途中で止める）。
両者は直交（発生抑止 × 発生時の途中打ち切り）。
