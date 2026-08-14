# SOT-2717 — Gemini gold100: why idx8/idx52 were dropped, and what recovers

Backend: `gemini-3.6-flash` serve path (`RAG_INVESTIGATOR_BACKEND=gemini`). Gold: audited
`artifacts/predictions_test_v4_final.csv`. All `official:false` / dev. Focused evidence:
`artifacts/focused_gemini_sot2717.json` (target `1,8,52,87,98` + 10 Sonnet-match sentinels).

## idx52 — 文書抽出「別契約」役割（監視ダッシュボード構築） — RECOVERED (deterministic)

**Root cause.** `sep_contract_lane.resolve()` read the `X（別契約）` role label **only from `text_fts`**.
The idx52 evidence (`監視ダッシュボード構築(別契約)`) lives on a **scanned-PDF page** of the みなみ野
project and only reaches `text_fts` when that index was rebuilt **OCR-aware** — a build-flag-fragile
precondition. On the Gemini serve run the FTS index carried no marked label, so the lane returned `None`
and the question fell through to an LLM-budget **abstain**.

**Sonnet-path diff.** The same scanned page is captured in the build-time **`image_ocr_store.jsonl`**
(cycle7 / SOT-2684). Sonnet reached idx52 via the cycle10 doc-reach layer; the Gemini serve path did not
consult that store from `sep_contract_lane`, so the durable OCR evidence never surfaced.

**Fix.** `sep_contract_lane` now falls back to reading the `X（別契約）` label directly from
`image_ocr_store.load()` when FTS surfaces no marked label for the named project (`_collect_from_ocr`).
The store is baked at build time and always present at serve → FTS-build-independent and
backend-independent.

**Verification (3 ways).**
1. `resolve()` against the **real** on-disk `image_ocr_store` with `text_fts` disabled →
   `監視ダッシュボード構築` (the gold), `store=image_ocr_store`.
2. 10 unit tests green (`tests/test_sep_contract_lane.py`), incl. OCR-fallback + cross-project guard.
3. Focused backend=gemini: **idx52 route=deterministic → Perfect**; sentinels 10/10, zero regression.

OFF byte-identical: whole lane gated by `RAG_SEP_CONTRACT_ROLE`; the OCR read only fires when FTS
returned nothing → any case FTS already resolved is untouched.

## idx8 — 散文算術・給与差 17,744 — NOT recovered (LLM unit-churn)

**Observed.** Gemini emitted `17744人` this focused sample — **correct magnitude (17744)**, **wrong unit**
(`人` vs gold `17,744ドル`). Across prior samples idx8 churns `17,744ドル（人）` (wrong) / `17744人`
(wrong) / abstain. The salary difference itself (US ML avg 143000 − DE avg 125256 = 17744) **does reach
the LLM** — it computes the number — but the **unit label** is not stable.

**Attempted fix (kept as benign partial).** `formatting._UNIT_PAREN_CONTENT_RE` strips a trailing
bare-unit **parenthetical** on a digit-bearing value (`17,744ドル（人）` → `17,744ドル`), under both
verbatim and non-verbatim asks. It fires ONLY when the paren content is a bare unit/counter and the body
carries a digit, so proper-noun parentheticals (`田中（人事部）`) are never touched. Verified: 186
formatting tests green; gold v4 carries **no** bare-unit-paren answer, so it can only ever drop decoration
a real answer never carries (zero regression). OFF byte-identical (both callers gated by
`strip_paren_enabled`).

**Why it doesn't recover idx8:** the strip targets `（人）`, but the churn variant that actually appears is
`17744人` (no paren, wrong unit) — nothing for the strip to fire on. This is **LLM unit-churn**, not a
formatting-layer defect. No clean deterministic fix exists in this issue's scope without hardcoding the
`ドル` unit (forbidden).

**Next axis (future cycle):** a `derived_calculation` lane that computes the salary difference from the
two OCR'd averages and emits it deterministically **with the ドル unit** — mirroring how the fact-layer
already promotes other derived numerics. That would make idx8 backend-independent like idx52.

## 据置（investigation-only, honest）

- **idx1** (version_diff): **closed**. The judge requires the gold verbatim vocabulary
  「中間実測値と最終値」, non-derivable from the structured record (SOT-2706/2715 double-confirmed). Wrong
  on Sonnet too. Focused: Incorrect — expected.
- **idx87** (略称 AYM): focused = **Perfect** this sample. It is LLM abbreviation-churn (some samples pick
  `青葉与信マネジメント`), not a target of this issue; no deterministic promotion attempted here.
- **idx98** (RATE 改定日): **honest abstain**. No explicit revision notice exists in the source for any
  backend (evidence absent). Focused: Missing — expected.

## Full gold100 — intentionally NOT run

Per the issue's cost rule (`回収できていなければ full は回さず`): idx8 did **not** clear the focused gate,
so the expensive Gemini full was **not** spent. A single full run only adds churn-noisy net (documented
±3-5 LLM + stochastic-CRAG spread); idx52's recovery is already airtight from the deterministic focused
+ direct-resolve + unit-test evidence above. Shipping the verified idx52 deterministic reach; idx8 stays
honest-据置 with the derived-lane next axis recorded in `docs/ai/experiment_ledger.jsonl`.
