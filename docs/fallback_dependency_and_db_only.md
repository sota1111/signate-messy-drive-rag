# fallback依存率 KPI と RAG_DB_ONLY 診断モード (SOT-2660)

Cerebras型検索基盤 4/5（カバレッジ可視化）。「前処理で全書類をDB化し質問時はDBのみ参照」方針の
2026-08-12 敵対的レビュー結論の実装。

## 設計判断（レビュー確定・必読）

- **DB-only の serve 強制は採用しない。** 転記欠落=即棄権の cliff を作るため。生ファイルフォールバック
  （`read_office`/`file_grep`/`compute` 直読み等）は実測で約30問の正答を生産している安全網であり、本番の
  serve では常時温存する。Cerebras も同判断（生リポジトリへの ripgrep を質問時ツールとして残す）。
- ただし **DB経路のカバレッジ**（precomputed store だけで何問解けるか）は測れるべき。測らないと「DBの
  守備範囲を広げる」進捗が可視化できない。
- 目標は **fallback依存率 → 0 への漸近**であり、0の強制ではない。

## ツール分類表（`src/rag/agent/investigator.py` `RAW_FILE_TOOLS`）

serve 時に**生のコーパス文書（office/pdf/pptx/xlsx/csv の内容・ディレクトリ走査・その場 compute）を読む**
ツールを「生ファイル系」とする。それ以外は「DB経路」。ビルド時（索引/ストア構築）の生読みは対象外
＝ serve 時のアクセスのみが fallback 依存。

| 区分 | ツール |
| --- | --- |
| 生ファイル系（`RAW_FILE_TOOLS`） | `find_files` `file_grep` `read_office` `decrypt` `compute` `canonical_route` `read_chart_values` `caption_image` `pdf_emphasis` `pptx_pivot` `highlight_extract` `version_diff` `seating_lookup` `corpus_aggregate` `font_emphasis` `format_events` `enum_scan` |
| DB経路（既定許可） | `case_filter` `id_lookup` `metric_lookup` `diff_lookup`（事前計算事実層 = serve時 JSON lookup）, `verify_formula`（PoT: モデル供給の候補を検算するのみ・ファイル無読み）, `submit_answer`（終端）, 将来追加される検索/索引/蒸留ストア（block-list 方式なので新DBストアは自動的にDB経路） |

新しい**生ファイル読み**ツールを追加したときだけ `RAW_FILE_TOOLS` に追記する。新DBストアは追記不要。

## (a) raw_file_access テレメトリ（常時記録・SOT-2629 形式）

全ケース（回答／棄権問わず）で `details.interventions.raw_file_access` を記録する:

```json
"raw_file_access": {"used": true, "tools": {"read_office": 1, "compute": 2}, "blocked": {}, "db_only": false}
```

- `used` — その回答が生ファイル系ツールを実際に読んだか（fallback 依存の生シグナル）
- `tools` — 生ファイル系ツール別の実読み回数
- `blocked` — `RAG_DB_ONLY` で拒否された生ファイル呼び出し（DB_ONLY 実行時のみ非空）
- `db_only` — その実行が診断モードだったか

MCP 側（`src/rag/mcp/server.py`）は各 `tools/call` ログに `raw_file: bool` を付け、SOT-2627 の
details 再構成でセッションの raw_file_access を集計できるようにする。

## fallback依存率（`scoring/gold_offline.py` `fallback_dependency`）

gold_offline レポート／`render()` に追加:

- `fallback_rate` — **正答**のうち生ファイル系ツールを要した割合（→0 に漸近させる KPI。0 強制はしない）
- `db_only_coverage` — 生ファイルアクセス0で解けた正答数（= DB経路カバレッジ真値）
- `raw_dependent_match_idx` — 正答だが生ファイル依存の idx リスト（**DB化の未達バックログ／次の
  ストア拡張優先度**）
- `raw_file_tool_calls` — どの生ファイルツールが依存の主因かの横断集計

Sonnet サイクル台帳（`docs/ai/sonnet_gold_history.jsonl` / `experiment_ledger.jsonl`）のエントリにも
レポートの `fallback_dependency` ブロックを転記し、サイクル毎の推移を追う。

## (b) RAG_DB_ONLY 診断モード（**診断専用・既定OFF・本番既定化しない**）

`RAG_DB_ONLY=1` のとき `dispatch()` が `RAW_FILE_TOOLS` の呼び出しを理由付きで拒否し、モデルを
「DB経路のみ」で回答させる（DB経路で解けなければ棄権）。investigator ループと MCP サーバの双方が
同一の module-global を見るため、両経路で一様に効く。

**本番既定にはしない。** DB-only serve は 転記欠落=即棄権の cliff を作り、生ファイルフォールバックは
実測の安全網だからである。用途は **DB経路の真のカバレッジ実測**（precomputed store だけで解ける問数）と
**DB化未達 idx の可視化**（次のストア拡張の優先順位付け）に限る。診断実行の結果は必ず
`official:false`・「DB-onlyカバレッジ実測」とラベルして台帳を分離する。

既定 OFF の champion serve path は byte-identical（telemetry の追加キー以外に回答値・制御フロー・
ツール出力の変化なし。生ファイル呼び出しは一切 intercept されない）。
