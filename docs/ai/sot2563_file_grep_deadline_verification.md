# SOT-2563 — file_grep per-call deadline + cooperative cancel (focused/offline verification)

**gold100 は実行していない**（本子は runtime ガードの単独対策。統合実測は SOT-2527/親 SOT-2550 の管轄）。

## 目的（再掲）
`timeout_s` はターン間でしか判定されない（`investigator.py` の `if clock()-start > timeout_s`）ため、同期の
`file_grep` 全走査は**呼び出し途中で中断できず**、公称 180/240s に対し単一呼び出しが 280–658s（1.5–3.6倍）
暴走して答えを得る前に問答が終了する（`docs/ai/abstain_wrong_root_cause_SOT-2550.md` §3(A)）。実行時ガードで
単一呼び出しの予算暴走を止め、超過時は早期「未発見」で別ルートへ回す。

## 実装
- **`src/rag/tools/call_budget.py`（新規）** — per-call wall-clock 予算の伝播。ContextVar `remaining_budget()`
  で investigator が残予算を公開し、`scan_deadline()` が env 上限 `RAG_FILE_GREP_MAX_SCAN_S`（既定 **180s**）と
  残予算の**小さい方**から絶対デッドラインを算出。上限 `<=0` かつ残予算未伝播なら None（無制限＝byte-identical）。
- **`src/rag/tools/file_grep.py`** — `_scan_refs` の全走査ループを**協調キャンセル**化: 各ファイルの
  `_extract`（Office復号/PDF/OCR）**開始前**にデッドラインを判定し、到達時は途中打ち切り→それまでのヒットを
  部分結果として返し `evidence.deadline_hit=True` / `partial=True` / `note`（別ルート誘導）を付す。`file_grep()`
  に `deadline_s` / `clock` を追加（既定は call_budget から解決）。
- **`src/rag/agent/investigator.py`** — `cached_dispatch` が各ツール呼び出し直前に
  `call_budget.remaining_budget(timeout_s-(clock()-start))` で残予算を公開（file_grep 以外のツールは無視）。

## 検証（focused/offline, 実コーパス 403 files, LLM 非使用）
`scripts/verify_sot2563_deadline.py`（稀語クエリ＝全ファイル `_extract` を強制、query 非依存で全走査コストを測定）。

| 実行 | cap | elapsed | files_scanned | deadline_hit | 結果 |
| --- | --- | --- | --- | --- | --- |
| baseline（上限無効） | 0（無制限） | 43.65s | 403（全件） | false | 全走査＝従来挙動 |
| **既定 cap** | **180s** | **43.55s** | **403（全件）** | **false** | **baseline と byte-identical（正答経路を切らない）** |
| 積極 cap | 30s | 33.47s | 247 | true | デッドライン到達→途中打ち切り＋note |
| 積極 cap | 10s | 24.23s | 170 | true | 同上（より早期に別ルートへ） |

### 受け入れ条件の充足
- **(a) 単一 file_grep が per-call デッドライン内に収束**: cap で files_scanned/elapsed が単調に縮小
  （403→247→170）。超過分は「デッドライン到達時に走査中だった1ファイルの `_extract` が完走する」ぶんの
  オーバーシュート（協調キャンセルの粒度＝ファイル境界。cap10→24s は最終1ファイルの OCR ≈14s）。
  本番の全走査 300–658s に対し既定 180s なら ~180s+1ファイルへ確実に上限化。
- **(b) 超過時に早期「未発見」→別ツール切替**: `deadline_hit/partial/note` を戻り値に付与。note は
  「canonical_route・find_files・evidence_index/read_office で対象特定」を明示し、investigator に切替を促す。
- **(c) precision 非劣化**: 既定 cap=180s は実コーパス全走査（~44s）でも既 match 最大（125s, SOT-2550台帳）でも
  発火せず、baseline と files_scanned/挙動が完全一致＝champion serve byte-identical。
- **(c') timeout 棄権回収**: 本番で 280–658s の暴走全走査（＝BUDGET_EXHAUSTED/timeout 棄権の主因、gold100の
  44/60）を 180s へ上限化し、残余 60–460s を canonical/find_files 等の別ルートへ回せる。end-to-end の
  abstain→match 変換は LLM のツール選択に依存し非決定的なため（かつ gold100 は本子で実行しない方針）、
  本子では機序（暴走の発生源＝file_grep 層）を決定論的に封じたことを一次証拠とする。

### 単体テスト（`tests/test_file_grep_deadline.py`, 7件 green）
call_budget の上限/残予算→デッドライン算出（env 既定/上書き/無効/不正値、min(cap, remaining)、context復元）、
file_grep の即時打ち切り/途中打ち切り（fake clock で files_scanned=1）/未発火時 byte-identical/伝播予算尊重。

## env（保守的既定・可変）
- `RAG_FILE_GREP_MAX_SCAN_S`（既定 180.0s、`<=0` で無効化＝無制限）。

## 回帰
`tests/test_file_grep_deadline.py`(7) + `tests/test_investigator.py` + `scoring/test_tool_contract.py` +
`tests/test_file_grep_index.py` + `scoring/test_file_grep.py` = **113 passed**（file_grep 全走査経路は純追加、
既定 OFF 相当の 180s は非破壊）。
