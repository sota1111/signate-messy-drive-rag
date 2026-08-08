# SOT-2509 Final Report

## Summary

参照選択を質問契約と実ファイル構造から決定する経路へ変更した。版差分は `version_diff` による
全スライド/全シート比較を必須化し、解決値をモデルの再選択・言い換え・再送に依存せず直接確定する。
「明記」質問は引用条件と候補の同一セル/文証拠を commit gate で検証し、画像のみ PDF は質問別の
Gemini Vision 構造抽出へ送る。xlsx はフェーズ等のグルーピング列だけを保守的に前方補完し、通常の
空欄から担当者等を作らない。

## Improvement Cycles

| Cycle | idx0 | idx52 | idx89 | Decision |
| --- | --- | --- | --- | --- |
| 1 | スライド6追記へ到達（契約分類要修正） | 検索 timeout | T23（生 xlsx 空欄） | 分類・画像PDF・compute 正規化を修正 |
| 2 | 通信上の棄権 | `監視ダッシュボード構築（別契約）` の原文へ到達 | T27 正答 | 解決済み版差分の棄権禁止、Vision の役割限定を強化 |
| 3 | 差分解決後の再送 error | 対象原文へ到達（隣接項目を含む） | T27 正答 | 版差分を直接確定、Vision を最小候補の構造出力へ変更 |
| 4（再開後に許可） | 意味相当の決定論出力だが judge は Incorrect | 隣接項目を含み Incorrect | 初手の空応答を安全棄権 | 0 match / 1 abstain / 2 wrong、追加実行上限に到達 |
| 5（再度許可） | 共通見出しを含む追記表面へ正規化して match | Vision は正しい単一候補を返したが再探索が上限到達 | 空初手再誘導後に T27 正答 | 2 match / 1 abstain / 0 wrong、後続で単一候補直接 commit を追加 |
| 6（08-08再開 1/2） | match | caption_image に到達せず安全棄権 | match | 2 match / 1 abstain / 0 wrong、画像PDF参照の前段を修正 |
| 7（08-08再開 2/2） | match | 一意な報告書の単一 strict literal 候補を直接確定して match | match | 3 match / 0 abstain / 0 wrong、promote |

当初上限3 cycle後、Linear の明示指示で cycle 4 と cycle 5 を各1回追加した。生成済み回答の正式 focused 採点は cycle 1 が
match 0 / abstain 1 / wrong 2、cycle 2・3 がそれぞれ match 1 / abstain 1 / wrong 1 であり、
cycle 4 は match 0 / abstain 1 / wrong 2、cycle 5 は match 2 / abstain 1 / wrong 0 だった。cycle 5 では idx0 と
idx89 が match。idx52 は Vision ツールが正しい単一候補を返した後もモデルが再探索を続け、max_turns=12 の
`BUDGET_EXHAUSTED` で棄権した。08-08 の再開指示に従う cycle 6 では画像PDFへ到達する前に棄権したため、projectを
決定論解決し、一意な report PDF だけを Vision 検証する fail-closed 経路を追加した。cycle 7 は3問すべて match。
Gold-100 は最新指示により本Issueでは再実行せず、SOT-2527 の統合一回実測へ委譲した。

## Changed Files

- `src/rag/agent/{investigator,obligations,question_contract,routing}.py` — 必須版差分、literal証拠義務、空初手再誘導、単一 strict literal 直接 commit、画像PDF優先ルーティング。
- `src/rag/{archetype,diffpair}.py` — attached `old` 名の分類、全版構造比較、追加セクションの構造要約。
- `src/rag/extract/office.py`, `src/rag/tools/{compute_sandbox,extract_tools}.py` — xlsxグループ列補完と同一視覚行境界を持つ質問別Gemini Vision PDF抽出。
- `scoring/test_{compute,diffpair,tool_contract}.py`, `tests/test_{investigator,obligations,routing,office_xlsx}.py` — 決定論・誤選択防止・実コーパス回帰テスト。
- `scoring/gold_offline.py`, `scoring/test_gold_offline.py` — 登録済み棄権コード集合を動的に検証し、8番目のコード追加後も品質ゲートを安定化。
- `docs/ai/experiment_ledger.jsonl` — `deterministic-reference-selection` 軸の cycle 6/7 と promote 結果。

## Verification

- Focused contract/regression tests after cycle 7 fix: 160 passed.
- Formal focused scoring: cycle 1 = 0/1/2、cycle 2 = 1/1/1、cycle 3 = 1/1/1
  cycle 4 = 0/1/2、cycle 5 = 2/1/0、cycle 6 = 2/1/0、cycle 7 = 3/0/0（match / abstain / wrong）。
- Full pytest after cycle 7 fix: 896 passed, 7 non-fatal openpyxl WMF warnings.
- Python compile check (`src`, `scoring`, `backend`, `tests`): PASS.
- `git diff --check`: PASS.
- npm lint/typecheck/test/e2e: N/A（Python repository、`package.json` なし）。
- Gold-100: 本runでは未実行（human指示08-08によりSOT-2527へ委譲）。過去の一度限り実測は match 18 / wrong 4 / abstain 78。

## Acceptance Criteria

- [x] 版差分質問は回答確定前に全スライド/全シートの決定論比較を実行し、解決値をそのまま確定する。
- [x] 「明記」は条件語と候補が同一箇所に literal 共起しない候補を拒否する。
- [x] xlsx の結合/空欄グループ列を前方補完し、通常列の空欄から値を捏造しない。
- [x] idx0/52/89 の誤選択原因へ一般則を実装し、特定回答を production code にハードコードしていない。
- [x] Gemini-only investigator 経路を維持した。
- [x] Gold-100 非劣化の統合実測を SOT-2527 へ明示的に委譲した。
- [x] idx0/52/89 が同一 focused cycle で全件 match（再開指示の最大2 cycle内）。

## Remaining Issues

本Issueの focused 条件は達成済み。残る統合 Gold-100 非劣化確認は SOT-2527 が一度だけ実行する。

## Linear Report: POSTED

## Acceptance: PASS

## Next Action: READY_FOR_REVIEW
