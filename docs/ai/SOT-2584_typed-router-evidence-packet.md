# SOT-2584 — 質問型ルーター + Evidence Packet 事前生成（回答ループ前の決定論フェーズ）

親 Issue: SOT-2568（ディープリサーチ実装順 2/7・P0）。依存 SOT-2583（document registry）は Done（PR #109）。

## 目的

「検索→考える→また検索」の自由探索を反転し、**質問を型判定 → 文書確定（registry）→ 必要証拠を
Evidence Packet 化 → 生成エージェントを原則1回だけ**走らせる。最大の失敗要因である
`BUDGET_EXHAUSTED`（27/54）を epistemic 棄権から分離する。すべて **既定 OFF（`RAG_EVIDENCE_PACKET`）で
OFF 時 byte-identical**。

## 実装

新規モジュール（いずれも純粋・決定論・ネットワーク非依存）:

| モジュール | 役割 |
| --- | --- |
| `src/rag/agent/query_router.py` | 既存9契約 + pivot 検出 + 新規 EXISTENCE 検出を **6+1 route** へ写像。route 別 evidence slots / budget contract / 決定論 primary lane を付与。offline route_accuracy 計測（`route_agreement`） |
| `src/rag/agent/evidence_packet.py` | route 決定 + SOT-2583 registry Resolver で **Evidence Packet JSON**（`resolved_documents / required_evidence / evidence / missing / tool_budget_remaining`）を構築し、生成エージェントへ pre-inject する directive を生成 |
| `src/rag/agent/failure_taxonomy.py` | 7コードの failure taxonomy。**`BUDGET_EXHAUSTED` を唯一の operational（制御失敗）** とし epistemic 棄権（CORPUS_ABSENT 他）と分離。既存 abstain-ledger state / stop_reason を写像 |

配線:
- `investigate(...)` に `preamble: str | None = None` を追加。渡されたときのみ第1ターン前に平文 directive
  として注入（`first_move` と合成可）。**None（既定）は完全 no-op → byte-identical**。
- `answer_question(...)` は `RAG_EVIDENCE_PACKET` ON 時のみ packet を構築し `preamble` として渡す。
  失敗時は fail-open（`preamble=None`）。コーパス固有事実・回答は一切注入しない（文書 identity + slot 名 +
  予算のみ）。

### 6+1 route と evidence slots / budget contract

| Route | primary lane | evidence slots | 自由探索予算(fallback) |
| --- | --- | --- | --- |
| LOOKUP | (retrieval) | canonical_doc / answer_span | 1 |
| NUMERIC | compute | operands / unit / operation / rounding | 1 |
| ENUM | corpus_aggregate(全数走査) | universe / filter_predicate / scan_completion | 0 |
| VERSION_DIFF | version_diff | version_pair / aligned_block / substantive_change | 1 |
| FORMAT | highlight_extract | target_cell_or_run / effective_style / style_provenance | 1 |
| PIVOT | pptx_pivot | pivot_identity / field_item_value | 1 |
| EXISTENCE | file_grep(exhaustive) | exhaustive_search_coverage / parser_support | 0 |

## 検証（offline 診断・`scripts/measure_query_router.py` → `artifacts/query_router_diagnostics.json`）

gold100（`artifacts/gold_100_review.csv`）に対する決定論・LLM非依存の診断:

- **route_accuracy = 0.99**（router route vs gold archetype 由来 route、refinement 許容）
- **hard_lane_invocation_rate = 0.68**（決定論 primary lane へ dispatch される割合）
- **route_distribution** = LOOKUP 40 / NUMERIC 32 / ENUM 10 / FORMAT 10 / VERSION_DIFF 6 / EXISTENCE 2 / PIVOT 0
- **failure_taxonomy**（棄権を再コード化）: **operational `BUDGET_EXHAUSTED` = 27** ／ epistemic = 27
  （CORPUS_ABSENT 18 / NOT_RETRIEVED 5 / PARSER_CAPABILITY_MISS 2 / DOC_RESOLUTION_FAILED 1 /
  EVIDENCE_INCOMPLETE 1）。→ ディープリサーチの `BUDGET_EXHAUSTED 27 / UNANSWERABLE 18 / NOT_RETRIEVED 5`
  診断を再現し、**BUDGET_EXHAUSTED を epistemic 棄権から分離** して記録できることを確認。

テスト: 新規 29（`tests/test_query_router.py` / `test_evidence_packet.py` / `test_failure_taxonomy.py`）
＋ リポジトリ全 **561 passed**（回帰0）。OFF 既定で serve path は byte-identical。

## 残作業・方針

- serve-path での **live gold100 A/B（実 LB 一次 KPI、proxy 単独昇格禁止）** は human-gated 実測待ち
  （SOT-2503/2562/2564 と同じ既定 OFF・mechanism landed / relaxation deferred の前例）。
- 各 slot の hard-lane 事前充足（numeric PoT / enum symbolic / format 完全化）は本 router が primary lane
  へ昇格させた下流 SOT-2585/2586/2587 の scope。本 Issue は「型判定→文書確定→packet 化→予算契約→taxonomy」
  の土台を提供する。

## 受け入れ条件

- [x] 6+1 型の router と route 別 evidence slot / budget contract が実装されている
- [x] Evidence Packet が生成エージェントへ pre-inject され、自由探索が missing slot のみに制限される
      （directive で primary lane→不足 slot のみ最大 `tool_budget_remaining` 回に制限）
- [x] BUDGET_EXHAUSTED が epistemic 棄権と分離した failure taxonomy で記録される
- [x] gold100 で診断メトリクス（route_accuracy 等）が記録されている（offline 診断）
- [x] 既定 OFF・OFF 時 byte-identical・回帰 0
