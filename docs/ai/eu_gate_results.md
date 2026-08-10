# SOT-2589 — Expected-Utility Three-Tier Abstain Gate (results)

Final re-calibration of the abstain gate (SOT-2568 deep-research 実装順 7/7, P2). Replaces the
confidence-threshold commit decision (`GATE_COMMIT_CONFIDENCE=0.7`) with an **expected-utility** decision
aligned to the metric (Perfect +1 / Acceptable +0.5 / Missing 0 / Incorrect −1):

> commit iff **U = P(P) + 0.5·P(A) − P(I) > 0**

- **Module:** `src/rag/agent/eu_gate.py` — pure, network-free. `GateSignals` bundles the three signal
  groups; `decide()` returns one of **HARD ACCEPT** (deterministic lane + evidence complete + verifier
  agree) / **SOFT ACCEPT** (`U>0`) / **ABSTAIN** (hard epistemic blocker, or `U≤0`).
- **Wiring:** `src/rag/agent/gate.py` behind `RAG_EU_GATE` (default **OFF**). OFF ⇒ the confidence gate
  runs unchanged (byte-identical, regression 0). ON ⇒ a served resolution commits/abstains on `U`; the
  execution gate still applies on top; an abstained consensus is never resurrected.

## Signal groups (research comment)

- **Hard evidence signals — PRIMARY.** Positive-evidence (raise correctness, default "not obtained"):
  `canonical_doc_resolved` (SOT-2583), `evidence_slots_complete` (SOT-2584), `self_consistency_agrees`,
  `operand_sources_complete` (SOT-2586), `universe_exhaustively_scanned` (SOT-2587),
  `version_pair_resolved` (SOT-2588). Blockers (force ABSTAIN, default "no problem"): `parser_capable`,
  `execution_engines_agree`, `source_conflict_absent`, `explicit_file_constraint_satisfied`.
- **Retrieval signals — SECONDARY:** top1 score / top1-top2 margin / dense-sparse agreement / diversity.
- **Model signals — AUXILIARY:** answer verifier (strong); **verbal confidence demoted** to weight ≤0.05
  (RiskEval arXiv:2601.07767 / SABER arXiv:2605.18792: verbal confidence is not faithful to precision).
- **BUDGET_EXHAUSTED is excluded from the abstain criteria** — it is operational, not epistemic
  (`failure_taxonomy.is_operational`); the decision never consults it.

## gold100 diagnostics (`scripts/measure_eu_gate.py` → `artifacts/eu_gate_diagnostics.json`)

Offline / network-free. Diagnostic only — **not** an LB predictor (local proxy ↔ real LB ρ=−0.09);
promotion is real-LB gated.

| block | metric | value |
| --- | --- | --- |
| champion baseline (confidence gate) | coverage | 0.46 (46/100 answered) |
| | incorrect_rate_on_answered | 0.152 (7/46) |
| | expected_score | 0.32 |
| abstain taxonomy (54 abstains) | **operational (BUDGET_EXHAUSTED)** | **27** — EU gate excludes from abstain |
| | epistemic | 27 (corpus_absent 18 / not_retrieved 5 / parser 2 / ambiguous 1 / incomplete 1) |
| EU gate on answered pool (offline-proxy signals) | mean U (match / wrong) | 0.594 / 0.631 |
| | utility_discriminates | **False** |

### Reading

1. **The BUDGET_EXHAUSTED exclusion is the real lever.** 27 of 54 abstains (50%) are *operational*, not
   epistemic — the old "budget-exhausted ⇒ abstain" rule was leaving that coverage headroom on the table.
   The remaining 18 `corpus_absent` are correctly forced to ABSTAIN by the hard blocker.
2. **Offline-computable signals do not separate the 7 WRONG from the 39 MATCH** (mean U_wrong ≳ U_match).
   The WRONG answers had the same offline signals *and higher verbal confidence* than the correct ones —
   an empirical reproduction of the "verbal confidence is not faithful" finding that motivates demoting
   it. Discrimination has to come from **runtime** hard-evidence signals (`execution_engines_agree`,
   `source_conflict_absent`, `operand_sources_complete`) that an offline pass cannot reconstruct.
3. **Conclusion:** the gate structure removes the imagined failure mode (budget-as-abstain, confidence-as-
   precision) and hard-blocks the truly-absent, but a coverage *gain* on the answered pool needs the live
   serve-path signals. Promotion therefore awaits a human-gated live gold100 A/B (real LB primary KPI);
   this run records the diagnostics and ships the lane default-OFF.

Run: `RAG_EU_GATE=1 PYTHONPATH=. .venv/bin/python scripts/measure_eu_gate.py`
