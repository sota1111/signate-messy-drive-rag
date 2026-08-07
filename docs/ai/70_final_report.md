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

上限3 cycleで探索を終了した。生成済み回答の正式 focused 採点は cycle 1 が
match 0 / abstain 1 / wrong 2、cycle 2・3 がそれぞれ match 1 / abstain 1 / wrong 1 であり、
3問同時 match の受け入れ条件は未達だった。cycle 3 後の最終決定論出力は idx0 が
`スライド6 追加：4.1 データ理解・品質確認〜4.5 ガバナンス・監査対応 の各フェーズの作業内容を追加`
となり、idx89 は実 xlsx の phase 6 最終開始タスク `最終報告・成果物提出・検収会` を返すが、
上限を超える cycle 4 は実行していない。

## Changed Files

- `src/rag/agent/{investigator,obligations,question_contract,routing}.py` — 必須版差分、literal証拠義務、commit gate、画像PDF優先ルーティング。
- `src/rag/{archetype,diffpair}.py` — attached `old` 名の分類、全版構造比較、追加セクションの構造要約。
- `src/rag/extract/office.py`, `src/rag/tools/{compute_sandbox,extract_tools}.py` — xlsxグループ列補完と質問別Gemini Vision PDF抽出。
- `scoring/test_{compute,diffpair,tool_contract}.py`, `tests/test_{investigator,obligations,routing,office_xlsx}.py` — 決定論・誤選択防止・実コーパス回帰テスト。
- `docs/ai/experiment_ledger.jsonl` — `deterministic-reference-selection` 軸の採用結果。

## Verification

- Focused contract/regression tests: 134 passed.
- Formal focused scoring: cycle 1 = 0/1/2、cycle 2 = 1/1/1、cycle 3 = 1/1/1
  （match / abstain / wrong、受け入れ条件未達）。
- Full pytest: 748 passed, 7 non-fatal openpyxl WMF warnings.
- Python compile check (`src`, `scoring`, `tests`): PASS.
- `git diff --check`: PASS.
- npm lint/typecheck/test/e2e: N/A（Python repository、`package.json` なし）。
- Gold-100 investigator run: match 18 / wrong 4 / abstain 78 / cost $4.5951.
- Gate: match >= 18 PASS、wrong <= 13 PASS、SOT-2508 baseline の既存 match→wrong = 0 PASS。

## Acceptance Criteria

- [x] 版差分質問は回答確定前に全スライド/全シートの決定論比較を実行し、解決値をそのまま確定する。
- [x] 「明記」は条件語と候補が同一箇所に literal 共起しない候補を拒否する。
- [x] xlsx の結合/空欄グループ列を前方補完し、通常列の空欄から値を捏造しない。
- [x] idx0/52/89 の誤選択原因へ一般則を実装し、特定回答を production code にハードコードしていない。
- [x] Gemini-only investigator 経路を維持した。
- [x] Gold-100 品質閾値と既存 match 非誤答化を満たした。
- [ ] idx0/52/89 が同一 focused cycle で全件 match（3 cycle以内）。

## Remaining Issues

Gold-100 の idx52 は全件並列時に broad search が timeout して安全棄権した。最終変更では literal 質問を
`find_files → caption_image` 優先へ変更したが、focused 上限到達後のため再生成・再採点していない。
PR #73 は受け入れ未達のためマージせずクローズし、ブランチを保持する。次へ進むには、追加 focused cycle を
許可するか、全体 Gold 閾値通過をもって受け入れるかの人判断が必要。

## Linear Report: POSTED

## Acceptance: FAIL

## Next Action: NEEDS_USER_INPUT
