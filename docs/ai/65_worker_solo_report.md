# Worker Report — SOT-2719（solo / handoff continuation / honest 見送り）

## Summary
「ページ数」型設問のみ bare 番号へ決定論整形する狭いレバー（`RAG_PAGE_COUNT_BARE`, 既定OFF）を実装し、
`formatting.apply_page_count_bare` + Gemini serve-boundary hook `investigator._apply_page_count_bare` を
配線した。unit test 17件 PASS・回帰ガード（idx12/18/59 の Nページ/ページ目 保持）は focused で実証。
しかし **backend=gemini focused と引き継いだ full gold100 の双方で idx84 は回収できなかった**。full は
net96（96 match / 3 abstain / 1 wrong）で、唯一の wrong は idx84 のまま。目標 net97 と idx84 Perfect が未達のため、
main は net96 champion 据え置きで **honest 見送り**とする。PR/merge なし。

## idx84 が回収できなかった根本原因（framing churn）
- focused の idx84 実回答は **`スライド6（文書記載のページ番号: 5）`**、full の実回答は
  **`スライド6（文書に記載のページ番号: 5）`**。いずれも verdict **Incorrect**。
- 本レバーは「回答本文中の `<N>ページ(目)` トークンの N を bare 化」する text-strip 型。今回の churn 変種は
  正しい印字ページ番号 5 が「ページ番号: **5**」の位置（`ページ` の **後**）に出現し、先頭には distractor の
  「スライド**6**」が来るため、`<N>ページ` トークンが存在せず **lever は fail-closed で no-op**（誤って 6 を
  掴む事故を避けた点は正しい挙動）。
- すなわち Gemini の値表現が run 毎に「5ページ（スライド6）」「スライド6（…ページ番号: 5）」等へ揺れる
  framing churn 本体は、LLM 出力を後段整形する狭いルールでは決定論的に回収できない（正しい 5 が正規化可能な
  位置に来ない run が存在し、かつ distractor 数字 6 と混在する）。

## 回帰ガード（安全性は実証済み）
- focused: idx12=`2ページ` Perfect / idx18=`2ページ` Perfect / idx59=`13ページ` Perfect（**形式回帰ゼロ**）。
- Sonnet 番兵 10/10 Perfect（regressions=[]）。
- `RAG_PAGE_COUNT_BARE` 既定 OFF ⇒ serve byte-identical。main には本コード自体が無く真に無改変。
- gold100 全走査で「ページ数」トークンを含む設問は idx84 のみ ⇒ 12/18/59 は構造的に無改変。

## Changed Files（feature branch のみ・未 merge）
- `src/rag/agent/formatting.py` — `apply_page_count_bare` / `page_count_bare_enabled`（値保存・質問キーゲート）。
- `src/rag/agent/investigator.py` — `_apply_page_count_bare` serve-boundary hook（既定OFFで no-op）。
- `tests/test_page_count_bare.py` — 17件（idx84想定phrasing回収・churn variant・idx12/18/59保持・OFF byte-identical）。
- `scripts/gemini_focused_sot2719.sh` — focused gate（target=84,12,18,59＋番兵10, `RAG_PAGE_COUNT_BARE=1`）。
- 注: commit `dcd9625` は関数重複を含む（作業途中）。未 merge のため main へは波及しない。

## Commands / Verification
- `pytest tests/test_page_count_bare.py` → **17 passed**。
- focused（`artifacts/focused_gemini_sot2719.json`）: gate=sentinel PASS だが **target idx84=Incorrect**。
  idx12/18/59=Perfect・番兵10/10。
- full Gemini gold100（引き継ぎ済み PID 717419 を重複起動せず回収）: **match 96 / abstain 3 / wrong 1 / net96**。
  idx12=`2ページ` MATCH、idx18=`2ページ` MATCH、idx59=`13ページ` MATCH、idx84 のみ WRONG。

## Acceptance Criteria
- [ ] idx84 が focused で Perfect（bare `5`） — **未達**（framing churn で lever no-op）。
- [x] idx12/18/59 形式回帰ゼロ（focused 実証）。
- [x] Sonnet 番兵回帰ゼロ・OFF byte-identical。
- [ ] full で net97・idx84回収 — **未達**（net96、idx84 が唯一の wrong）。→ **honest 見送り**。

## Remaining Issues / 次サイクルの escalation 推奨
- 本 text-strip 型レバーでは churn を決定論回収できない。次サイクルは issue が示唆した **決定論ページロケータ
  直コミット**（`heading_page_store` 等で「東都人材の最終報告書で F1 ランキング表が印字されたページ N」を確定し、
  「ページ数」型設問に限り bare N を direct-commit。LLM の churny テキストに依存しない）を検討。ただし独自の
  retrieval/回帰面を持ち、+1 は gold100 単発ノイズ（±3-5）内で限界効用は小。過剰整形で idx12/18/59 を壊す
  リスクと天秤にかけ、慎重に focused 検証したうえでのみ導入すること。

## Linear Report: POSTED
## Acceptance: FAIL
## Next Action: NEEDS_DEBUG

---

# Worker Report — SOT-2720（solo / handoff continuation）

## Summary
質問文キーの小数精度丸め契約と Gemini no-items 裸形式 fold を実装した。引き継ぎ時の focused は
idx57/63 が deterministic early-return のため serve-boundary hook を迂回していたので、全回答経路へ最終整形を
適用する共通 wrapper を追加した。idx85 の実出力揺れ（句点付き括弧注記／「全6項目達成」証明）も、未達成項目を
問う設問に限定した fail-closed 規則で吸収した。

focused は idx57=`0.42396`、idx63=`0.15002`、idx85=`該当なし` が全て Perfect、回帰対象も全て
Perfect、番兵10/10、regressions=[]。ただし条件通過後に一度だけ実行した full Gemini gold100 は
match96 / abstain3 / wrong1、net=95 で、要求 net>=98 を満たさなかった。3対象の intervention は3/3 match
でありレバー自体の回帰は見られないが、aggregate acceptance 未達のため PR/merge は行わない。

## Changed Files
- `src/rag/agent/formatting.py` — explicit decimal ROUND_HALF_UP と evidence-bound no-items fold。
- `src/rag/agent/investigator.py` — deterministic / fact / Claude-MCP / Gemini 全回答経路の SOT-2720 最終整形。
- `tests/test_decimal_precision_none_bare_fold.py` — deterministic early-return と Gemini phrasing churn 回帰。
- `scripts/gemini_focused_sot2720.sh`, `scripts/gemini_gold100.sh` — focused/full 測定配線。
- `docs/ai/experiment_ledger.jsonl`, `docs/ai/gemini_gold100_history.jsonl` — inconclusive 測定記録。

## Verification
- `.venv/bin/pytest -q tests/test_decimal_precision_none_bare_fold.py` — 33 passed。
- `.venv/bin/pytest -q` — 2361 passed, 14 warnings。
- focused official Gemini — PASS、targets 57/63/85 Perfect、番兵10/10、回帰0。
- full official:false Gemini（1回）— match96 / abstain3 / wrong1 / net95、cost $1.450493。

## Acceptance Criteria
- [x] 質問文キーの小数第N位丸めで idx57/63 focused Perfect。
- [x] 指定なし idx36 は `0.09619112771492555` を維持し、回帰対象 Perfect。
- [x] idx85 裸 `該当なし` Perfect、idx9/38 回帰0。
- [x] 番兵10/10、フラグOFF byte-identical（default OFF unit coverage）。
- [ ] full net>=98 — 未達（net95）。

## Risks / Remaining
full の非対象 churn（abstain idx1/25/98、wrong idx29）により aggregate gate 未達。仕様どおり full の再実行は
せず、レバーは experiment ledger で inconclusive とした。PR/merge なし、Linear は In Progress のまま。

## Linear Report: POSTED
## Acceptance: FAIL
## Next Action: NEEDS_DEBUG
