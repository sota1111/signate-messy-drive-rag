# SOT-2588 — 版差分 block-alignment + 編集意図分類レーン: gold100 メトリクス

親 SOT-2568 ディープリサーチ実装順 6/7 (P2)。誤答 idx1/idx14（version_diff の実質変更取り違え）の原因＝
**「差分の大きさと実質的変更の重要度の混同」** を解消するレーン。champion は「old vs new の largest diff」を
選ぶため、追記された概要セクション（idx1）やセクション再編（idx14）を拾い、実際の実質変更（性能比較表の削除／
データ列名のアンダースコア化）を取り違えていた。

出典（構造の借用のみ・taxonomy は enterprise 文書向けに再定義）: Re3 [ACL 2024.acl-long.255]、Edit Intent
Classification [EMNLP 2024.emnlp-main.839]、A Tale of Two Revisions [ACL 2024.findings-acl.190]。「alignment の
後に intent 分類を置く」という構造を借り、ラベル（SUBSTANTIVE/SURFACE/BOILERPLATE/LAYOUT_METADATA/UNCERTAIN）は
契約書・見積書・社内 PPT 向けに独自定義した。

## レーン構成 (`src/rag/diffpair.py`, opt-in `RAG_DIFF_ALIGN`, 既定 OFF)

```
Version family resolution   filename/folder 規則で解けない版ペアを SOT-2583 registry の version_family_id で
                            解決（idx1: `…_最終報告_old.pptx` × 無印 `…_最終報告.pptx` を family として結合）。
Structural normalization    既存の canonical _Struct（Document/Slide/Sheet/Paragraph/Table/Row/Cell）。
Block alignment             deterministic key（cell key / 段落 sequence）で整列。
Atomic changes              ADD / DELETE / MODIFY / MOVE。OFF は MODIFY のみ→表/行の純削除が不可視。ON は
                            old にあり new に無い keyed cell を DELETE、逆を ADD として明示（idx1 の slide-7 表削除）。
                            add/remove で正規化テキストが一致する塊は MOVE（= LAYOUT）。
Edit-intent classification  決定論 feature で分類（数値/金額/割合/日付/数量/固有名/人名/義務/条件/識別子=SUBSTANTIVE、
                            footer/版番号/pagination/自動日時/空白=BOILERPLATE、追記された概要=LAYOUT）。
Substantive ranking         intent 優先度 → score → 元順で安定ソート。**差分の大きさで加点しない**（in-place MODIFY が
                            最強・ADD/DELETE 塊は次点）ことで「サイズ⇄重要度」の混同を断つ。
複数候補 pre-inject          単一 winner でなく {intent, score, old, new, structural_location, reason_features} を
                            返し、生成 agent（investigator の _version_diff tool）へ候補を pre-inject。
```

- 既定 **OFF**（`RAG_DIFF_ALIGN`）。OFF 時は `_rendered_diff` が従来経路（`structural_diff`）を通り、
  `resolve_pair` の registry fallback も走らないため **champion answer は byte-identical**（全 diff 質問で HEAD と
  完全一致を確認）。
- 精度ガード（Missing 0 > Incorrect -1）: xlsx 版ペアで substantive 変更が多数（whole-sheet 再整列churn: idx22/idx95
  のスケジュール表）なら棄権を維持。pptx/docx は top 有界集合を返す（idx1 の 1 表削除は複数セルに跨るが 1 編集）。

## 計測 (`scripts/measure_diff_align.py`)

内部ハーネスは *故障箇所を分離する診断器* であり LB 予測器ではない（local proxy ↔ 実 LB ρ=-0.09）。
決定論・ネットワーク非依存（LLM 再生成なし）。`RAG_DIFF_ALIGN=1 PYTHONPATH=. .venv/bin/python scripts/measure_diff_align.py`。

### 三診断メトリクス（直近実測 / gold100 の version_diff n=6）

| metric | value | 意味 |
| --- | --- | --- |
| version_pair_accuracy | **1.0** (6/6) | 版ペアが解決（registry family fallback で idx1/idx22 の 2 件を追加回収） |
| alignment_accuracy | **1.0** (6/6) | 解決ペアが非空の atomic-change 列を生成 |
| substantive_change_precision | **1.0** | 回答を出す質問で top 候補が SUBSTANTIVE |
| registry_family_recovered_pairs | **2** | filename 規則単独では解けず registry family で解決した版ペア |

### idx 別（gold status = 対策前の champion 判定）

| idx | gold status | pair basis | top intent | ON 挙動 |
| --- | --- | --- | --- | --- |
| 0 | MATCH | explicit | SUBSTANTIVE | 正答維持（分析アプローチ全体像の追記, gold overlap 0.91） |
| 1 | **WRONG→改善** | registry-family | SUBSTANTIVE | 性能比較表（AUC-ROC/F1）の削除を第一候補に。champion の概要追記取り違えを解消 |
| 14 | **WRONG→改善** | explicit | SUBSTANTIVE | データ列名（loan_status 等）を含む STEP 変更を surface（gold token overlap 0.8） |
| 22 | ABSTAIN | registry-family | SUBSTANTIVE | 棄権維持（xlsx sheet churn ガード） |
| 74 | MATCH | explicit | SUBSTANTIVE | 正答維持（担当者 藤田 彩 → 井上 里奈, 人名 feature で substantive 化） |
| 95 | ABSTAIN | explicit | SUBSTANTIVE | 棄権維持（xlsx sheet churn ガード） |

- **回帰 0**: MATCH（idx0/74）と ABSTAIN（idx22/95）を維持しつつ、WRONG 2 件（idx1/14）で実質変更を第一候補化。
- **idx14 の限界**: v1→v3 は重い書き直しで、gold のデータ列名アンダースコア化は大きな STEP 追記ブロック内に
  埋め込まれる（0.8 token overlap で内容は surface されるが、単一の clean MODIFY として孤立はしない）。
- proxy 単独では昇格しない。既定 OFF → gold100 A/B → 実 LB 確認のゲートを通す。

## テスト

`scoring/test_diff_align.py`（分類/ランキングは corpus 非依存、registry family / xlsx ガードは corpus guard）。
既定 OFF の byte-identical は「filename 規則では解けない idx1 が OFF で棄権」で担保。
