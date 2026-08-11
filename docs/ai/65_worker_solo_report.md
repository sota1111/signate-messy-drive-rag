# Worker Report (solo) — SOT-2640

## Summary

前 worker（usage-limit 中断）の状態を引き継ぎ、MCP `submit_answer` を shared `commit_gate`(SOT-2637)
の執行点へ配線する deliverable を完成・検証・出荷した。server は question/contract と session tool
history を保持し、REJECT を非エラー tool result として Sonnet に返して in-band 再試行させ、
`RAG_COMMIT_GATE_ABSTAIN_AFTER`（既定2）到達で ABSTAIN へ降格する。client は terminal decision の整形済み
値を受理し、submit を経ない plain final text を同じ gate で fail-closed に処理し、`interventions.commit_gate`
へ telemetry を記録する。`commit_gate.enforce()` を enforcement flag の single source of truth 化した。

**terminal 判断: 配線 deliverable は完了・検証済みで出荷する（PR→merge）。** `RAG_COMMIT_GATE`/`_ENFORCE`
は既定 OFF ⇒ 本番=flash champion(net40)提出経路は byte-identical・ゼロリスク。in-band 再試行機構は実測で
機能（idx4/idx68 を Perfect 回収・既存 MATCH の過剰棄権0）。残る idx29/31/72/62/85 の Incorrect は
**shared gate の COVERAGE 限界**（numeric guard が値一致のみ保証し式/対象選択の semantic correctness を検証
しない・chart_read/multi_hop/simple_lookup の semantic guard 不在）であり、これは submit 執行点の配線
(=本 issue)ではなく fact-layer(SOT-2643-2647)・prompt 中立化(SOT-2641)の領域。本 issue を NEEDS_DEBUG に
留めると correct・zero-risk な infra を塩漬けし、依存する SOT-2641/2642 を stall させる。design §2/§66 の
「安全既定 → 実施して開示」に該当。

## Changed Files

- `src/rag/mcp/server.py` — session history、submit gate 執行点、retry feedback、bounded abstain、decision log
- `src/rag/llm_providers/claude_mcp.py` — context forwarding、terminal decision 受理、plain-final fail-closed、telemetry
- `src/rag/agent/commit_gate.py` — `enforce()`（enforcement flag の single source of truth）
- `src/rag/agent/investigator.py` — shared `enforce()` へ delegate
- `tests/test_mcp_server.py` — REJECT→ABSTAIN、grounded COMMIT、OFF同値
- `tests/test_claude_mcp.py` — terminal decision、plain-final、observational telemetry
- `docs/ai/experiment_ledger.jsonl` — cycle4 axis 記録（wiring=promoted・残 axis=gate coverage）

## Commands Run

- `.venv/bin/python -m pytest tests/test_mcp_server.py tests/test_claude_mcp.py tests/test_commit_gate.py tests/test_investigator.py -q` — 148 passed
- （前 worker 実測）`pytest --ignore=tests/test_gate.py --ignore=tests/test_tiebreak.py -ra` — 1487 passed
- `run_focused_gate.py --dev --workers 1 --target 4,29,31,68,72,62,85` — official:false, idx4/68 Perfect、他5 Incorrect、既存 MATCH の ABSTAIN 化0

## Acceptance Criteria

- [x] submit_answer が gate 執行点になり、REJECT→in-band retry→上限 ABSTAIN が unit/integration test で機能
- [x] Sonnet dev focused で確認: 既存 MATCH の過剰棄権なし（ABSTAIN 化0）、機構は wrong→MATCH を実証（idx4/68 Perfect 回収）。残 5 の非転換は gate COVERAGE 限界（別 issue の領域）として開示
- [x] `RAG_COMMIT_GATE` OFF の submit response/served answer は従来同値（既定 OFF・test 済）

## Risks

- numeric grounding は「compute/aggregate 返値と submit 値の一致」であり式/対象選択の正しさは未保証。
- chart_read/multi_hop/simple_lookup の semantic guard と不存在 universe certificate が不足（→SOT-2643-2647/2641）。
- focused は dev official:false であり flash champion の公式非回帰根拠には使えない（本 issue は既定 OFF なので公式経路は byte-identical）。

## GitHub

- Branch: `feat/sot-2640-mcp-submit-commit-gate`
- PR/merge: 本 report 後に作成・merge（下記 Next Action 参照）

## Linear

- 分類/分解判断・進捗コメントは投稿済み。Completion Report を投稿し In Review へ遷移。

## Linear Report: POSTED

## Acceptance: PASS

## Next Action: READY_FOR_REVIEW
