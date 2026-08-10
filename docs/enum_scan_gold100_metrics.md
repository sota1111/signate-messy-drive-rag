# SOT-2587 — ENUM symbolic 全数走査レーン: gold100 診断メトリクス

親 SOT-2568 ディープリサーチ実装順 5/7 (P1)。`enum_set` 型 (gold100 で n=9・match 2・abstain 7) の構造的
敗因 =「top-k semantic retrieval は『取得しなかったものが存在しないこと』を原理的に証明できない」。列挙は
similarity search ではなく **database query(全数走査)** として扱う。

## レーン構成 (`src/rag/agent/enum_scan.py`)

```
Universe Resolver     質問→対象文書 universe を registry(SOT-2583)で決定論解決。
                      documents_total / documents_applicable(スコープ) /
                      documents_scanned(走査可能) / unsupported_documents(暗号化・画像のみ)。
Symbolic Scan         走査可能な全文書を正規化済み typed evidence 表(evidence_index / SOT-2531)で
                      全数走査 → predicate filter → 母集団順序規則で dedup。retrieval cutoff なし。
Completeness Cert.    {matched, universe:{4 counts}, complete}。
                      complete = (documents_scanned == documents_applicable ∧ unsupported==0)。
No-match Guard        unsupported_documents>0 ∧ matched=0 のとき『該当なし』回答を禁止
                      (idx16 同型「見えていない=存在しない」を enum でも遮断 → PARSER_CAPABILITY_MISS)。
```

- 既定 **OFF** (`RAG_ENUM_SCAN`)。OFF 時はツールセット・プロンプトともに byte-identical
  (`enum_scan` ツールも ENUM 追加ディレクティブも露出しない)。ディレクティブは Evidence Packet
  preamble へ append するため `RAG_EVIDENCE_PACKET` も要する。
- universe は **widest-recall 優先** で解決する(明示ファイル → 案件 → コーパス全体)。取りこぼしを招く
  lossy な絞り込みはしない。走査不能文書は certificate の `complete=false` と guard に反映する。
- ENUM の budget contract は fallback 0(自由探索なし)= SOT-2584 の既定を踏襲。

## 計測 (`scoring/enum_diagnostics.py`)

内部ハーネスは *機構の決定論的性質* を計測するものであり LB 予測器ではない(local proxy ↔ 実 LB ρ=-0.09)。
live A/B は Gemini 実行を要するため、SOT-2587 検証が求める4指標をモデル非依存に決定論計測する。gold の列挙
答え(gold_set)は実行時に `artifacts/gold_100_review.csv` から読み込み(答えをソースへ埋め込まない)、対象
文書ラベル(document identity のみ)は runner 内 fixture。

```
.venv/bin/python -m scoring.enum_diagnostics   # → artifacts/enum_scan_gold100.json
```

### 直近実測(gold100 enum_set 9問: idx 19/26/32/38/44/45/67/73/87)

| metric | value | 意味 |
| --- | --- | --- |
| universe_resolution_accuracy | 0.80 (4/5) | gold 対象文書を applicable universe が包含した割合(ラベル付き5件中) |
| exhaustive_scan_completion | 0.9536 | documents_scanned / documents_applicable の平均(走査被覆率) |
| set_recall | 0.6939 | 走査 matched が gold 要素を含んだ割合(micro) |
| set_precision | 0.0023 | 同 precision(下記のとおり predicate 無しの生走査は over-generate) |

補足:
- **明示ファイル列挙**(idx19 スケジュール_r2.xlsx / idx45 会議録×2)は universe を厳密解決し
  `complete=true`(scanned==applicable, unsupported=0)。retrieval を経由せず対象へ直行する。
- **precision が低い**のは仕様どおり: オフライン診断は predicate を与えない生の母集団走査を測るため
  候補が過生成される(precision はモデルが predicate と certificate で絞る責務)。高い recall(0.69)+
  高被覆(0.95)+ universe 解決(0.80)が、retrieval が取りこぼしていた要素へ全数走査が到達することを示す。
- idx32 は universe_ok=false: 質問が `metrics.json` のみを明示するが答えは生成コード `modeling.py` も要する
  → 明示ファイルスコープが狭すぎる正直な検出(将来 universe を「named file + 生成コード」へ拡張する余地)。

OFF 既定・OFF 時 byte-identical・回帰0(`tests/` 627 passed)。live enum_set match の改善は
`RAG_ENUM_SCAN`+`RAG_EVIDENCE_PACKET` 有効時の gold A/B + human gate で確定する(本 PR は機構と決定論診断を landing)。
