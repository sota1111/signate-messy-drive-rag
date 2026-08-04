# Project status

_Last updated: 2026-08-04 (initial build)._

## What exists

| Layer | Status |
|---|---|
| GitHub repo (Private) | `sota1111/signate-messy-drive-rag` |
| GCP project | `signate-messy-drive-rag` (#204417478580, billing: NanoBanana) |
| Vertex models | `gemini-2.5-flash` / `gemini-2.5-pro` / `text-embedding-005` @ `us-central1` |
| Extraction | docx/xlsx/pptx (bold, highlight-HSV, tables) + pdf/csv/code/ipynb/json + PNG (Gemini vision); decrypts 2 password-protected files |
| Retrieval index | 2,931 chunks, 768-d embeddings + BM25 (incl. 54 captioned figures) |
| RAG | hybrid (RRF) retrieval + glossary expansion + project boost → Gemini answer, confidence-gated abstention |
| 3-gate scoring | gate1 (valid30, official GT) · gate2 (auto holdout + sealed AOBM) · gate3 (test100 → signate submit) |
| Backend | Cloud Run FastAPI `/ask`, Terraform IaC, deploy.sh |

## Current objective score (関門1, Gemini proxy judge)

**+0.100** over valid 30 — 11 Perfect / 11 Missing / 8 Incorrect (image-captioned index).
Metric: Perfect +1 / Acceptable +0.5 / Missing 0 / Incorrect −1, mean.
(Official score would be computed by SIGNATE's gpt-5.2 judge on submission.)

## SIGNATE submissions

| # | date | memo | official Public score | note |
|---|---|---|---|---|
| 1 | 2026-08-04 | baseline hybrid RAG + abstention | **−0.2333** (−7/30) | below blank; committed answers net-negative |
| 2 | 2026-08-04 | precision-first (high-conf + strict verify) | **−0.0667** (−2/30) | +0.167; 該当なし 19→2; still slightly net-neg |
| 3 | 2026-08-04 | + answer self-consistency gate | **−0.0333** (−1/30) | **best real submission**; 24 committed |
| 4 | 2026-08-04 | SOT-2407..2411 modules (diff/pivot/aggregate/enum) | **−0.1000** (−3/30) | **regressed** despite valid30 proxy +0.5833 |

### ⚠️ Key lesson — valid30 is NOT predictive (overfitting)
Submission #4's hard modules scored **+0.5833 on valid30 but −0.10 on the real public 30**. They were
developed against valid30's specific questions and commit confident-wrong answers on unseen questions
(each −1). **The local proxy AND valid30 both mislead.** Public LB shows each team's *best* submission,
so #3 (−0.0333) still stands as our recorded floor.

### Strategy pivot (2026-08-04, human decision): aim for the frontier, not ±0.x
1st place = 1.0 (30/30, answers everything correctly). The gap is **coverage × accuracy**, not abstention
tuning. Shift from precision-first (caps near 0) to answering the full spectrum correctly, gated by a
**generalization hold-out** (sealed project + independent synth) so improvements must *transfer*, not
overfit valid30. Driven by the opus autonomous loop: **SOT-2424** (hold-out + overfit detection, Urgent)
→ **SOT-2425** (per-archetype coverage×accuracy iteration).

**Public LB is 30 questions**; score = mean of Perfect+1/Acceptable+0.5/Missing0/Incorrect−1
(so a blank all-abstain submission = 0.0). Top ~11 teams = 30/30 = 1.0 → the task is fully saturable.

**Key learning — the real gpt-5.2 judge is ~0.2 harsher than the local Gemini proxy.** Baseline was
proxy ≈0 but real −0.23. Under Incorrect=−1, **precision beats recall**: abstain by default, commit
only high-confidence directly-readable facts. Optimize for **Incorrect→0**, not proxy mean.
The official score is on the SIGNATE page (CLI cannot read it back) — it is the only trustworthy signal.

## Live backend

- URL: `https://signate-messy-drive-rag-backend-4kvjtj6qvq-uc.a.run.app`
- `GET /health` · `POST /ask {"question": "...", "hard": false}`

## Commands

```bash
python -m src.rag.index                 # build index (add --no-images to skip vision captions)
python -m scoring.gate1                 # 関門1: run valid + objective score
python -m scoring.gate2                 # 関門2: auto-holdout + sealed-project transfer
python -m src.rag.run --split test      # build test predictions
python -m scoring.gate3 --submit --memo "…"   # 関門3: submit to SIGNATE
bash scripts/deploy.sh                   # build image + deploy Cloud Run
```

## Known follow-ups (score levers)

1. **Version-diff questions** (old vs `_final`/最新): implement structured doc-pair diffing
   (idx9-type). Currently weak.
2. **Cross-project aggregation** (消費税総額, 差額): add a compute step over extracted tables
   (idx3, idx8-type). Currently guesses.
3. **Answer conciseness / exact format**: post-process to match terse GT (idx17/23/28 were verbose).
4. **Figure reading** (marked words, histogram max count): stronger vision prompting +
   always-attach the referenced PNG (idx0/idx1 class).
5. **Judge reliability**: the Gemini proxy occasionally mis-scores exact matches (idx2). Add
   self-consistency (majority of N) for gate1/gate2, or supply an OpenAI key for gpt-5.2 parity.
6. Redeploy after index rebuilds to keep the served index current.
