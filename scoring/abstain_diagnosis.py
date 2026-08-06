"""SOT-2486 — both_abstain 律速診断: 検索欠損 (retrieval-miss) vs 導出欠損 (derivation-miss).

The gold-100 offline run abstains on 81/100 items. :mod:`scoring.abstain_breakdown` (SOT-2483) split
that pool by *reason* and proved the **recoverable** region (one side proposed a value) is EV-negative
to commit — over-abstention there is correct. The residual EV-positive frontier is the other region:
the **``both_abstain``** items (≈40) where investigator AND verifier *both* returned 無回答. Those are
真の無回答 — nothing was proposed, so relaxing the gate cannot help. To turn them into correct answers
the system must actually *find* the answer, and the actionable question is **why it didn't**:

- **retrieval_miss** — the gold-supporting evidence never surfaced in the retrieval top-k. The fix is
  retrieval expansion (more/better recall); no amount of reasoning helps if the chunk isn't present.
- **derivation_miss** — the evidence *was* in the retrieval top-k (or the answer isn't a lookup at all
  but a value to be computed/derived), yet the investigator failed to derive/commit it. The fix is
  tooling / reasoning / confidence, not retrieval.

This module classifies each ``both_abstain`` item into those two buckets **deterministically, with no
LLM** (受け入れ条件: 決定論・LLM最小), using two offline signals over the persisted index:

1. **Evidence localization (gold → file).** Distinctive "needles" (numbers, katakana/kanji runs,
   ascii identifiers) are extracted from the *gold* answer; a file's *coverage* is how many distinct
   needles literally appear in its chunks. The maximally-covering file(s) are the evidence location.
   When no file covers ≥ ``min_ratio`` of the needles the gold answer is **not literally locatable** —
   it must be derived/computed (e.g. a ``derived_calculation``), so retrieval cannot be blamed.
2. **Retrieval reachability (question → top-k, recall@k).** BM25 (the lexical component of the
   production dense+BM25 RRF retriever, :mod:`src.rag.retrieve`) ranks all chunks for the *question*;
   ``recall@k`` asks whether an evidence file appears within the top-k chunk files.

Classification (pure — :func:`diagnose_case`):

- gold not literally locatable                       → ``derivation_miss`` (subreason ``not_a_lookup``)
- evidence file present AND in retrieval top-k        → ``derivation_miss`` (subreason ``in_topk``)
- evidence file present but BELOW top-k               → ``retrieval_miss``  (subreason ``below_topk``)

**Proxy honesty.** The reachability signal is BM25-only (dense query embedding needs Vertex/network and
would break determinism). BM25 is a *subset* of production retrieval (dense+BM25 RRF), so a BM25 miss
may still be a dense hit: ``retrieval_miss`` here is an **upper bound** and ``derivation_miss`` a
**lower bound**. The per-case ``evidence_rank`` (the best rank at which an evidence file first appears
in the full BM25 ranking) shows how far below k a miss falls — a near-miss (rank just past k) is a
strong retrieval-expansion candidate, a deep miss is genuinely absent lexically.

This is a **diagnostic only** — it reads existing artifacts and never touches the production
gate/resolve path (関門2 自明に非劣化, byte 非破壊). The ledger attribution is
``both-abstain-retrieval-vs-derivation-diagnosis`` (result ``diagnostic``).
"""
from __future__ import annotations

import argparse
import json
import re
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
from rank_bm25 import BM25Okapi

from config import settings
from scoring.abstain_breakdown import (
    BOTH_ABSTAIN,
    classify_record,
    load_details,
    load_gold,
)
from src.rag import archetype, index

# Classification labels (stable identifiers used in the JSON report and by tests).
RETRIEVAL_MISS = "retrieval_miss"
DERIVATION_MISS = "derivation_miss"

# Subreasons — *why* a case landed in its bucket.
SUB_NOT_A_LOOKUP = "not_a_lookup"      # gold answer not literally locatable → derived/computed
SUB_IN_TOPK = "in_topk"                # evidence file surfaced in retrieval top-k, yet uncommitted
SUB_BELOW_TOPK = "below_topk"          # evidence file exists but ranked below top-k

DEFAULT_K = 16                         # production retrieve() default top-k (src/rag/retrieve.py)
DEFAULT_MIN_RATIO = 0.5                # a file must cover ≥ this fraction of needles to be "evidence"
_MAX_EVIDENCE_FILES = 4                # cap the reported evidence-file list (ties)

# --- needle extraction: distinctive tokens that anchor a gold answer to a source file ---
_NUM = re.compile(r"[0-9][0-9,.]*%?")           # 40% / 25,000 / 14,744 — strong numeric anchors
_KATA = re.compile(r"[ァ-ヶー]{3,}")             # katakana runs (product / entity names)
_KANJI = re.compile(r"[一-鿿]{3,}")              # kanji runs ≥3 (compound nouns; avoids common 1-2 char)
_ASCII = re.compile(r"[A-Za-z][A-Za-z0-9_-]{2,}")  # AUC-ROC / F1 / bmi — ascii identifiers ≥3 chars


def extract_needles(text: str) -> set[str]:
    """Distinctive tokens from a gold answer used to locate its evidence file (deterministic).

    Numbers, katakana runs (≥3), kanji runs (≥3) and ascii identifiers (≥3) — the tokens that, if the
    answer is a *lookup*, appear literally in the source document. Generic 1–2 char fragments are
    excluded so a coincidental single-character match can't ground a file.
    """
    out: set[str] = set()
    for rx in (_NUM, _KATA, _KANJI, _ASCII):
        out |= set(rx.findall(text or ""))
    return {n for n in (s.strip() for s in out) if len(n) >= 2}


@dataclass(frozen=True)
class EvidenceLoc:
    """Where the gold answer's evidence lives in the corpus (or that it isn't a literal lookup)."""

    files: tuple[str, ...]     # maximally-covering file(s) — the evidence location(s)
    n_needles: int
    best_coverage: int         # #distinct needles the best file covers
    coverage_ratio: float      # best_coverage / n_needles
    grounded: bool             # the gold answer is literally locatable (ratio ≥ min_ratio)


def locate_evidence(needles: set[str], chunks: list[dict], *,
                    min_ratio: float = DEFAULT_MIN_RATIO) -> EvidenceLoc:
    """Localize the gold answer to its source file(s) by literal needle coverage (deterministic).

    A file's coverage = the number of distinct ``needles`` that appear literally in *any* of its
    chunks. The maximally-covering file(s) are the evidence location; ``grounded`` is True when that
    coverage clears ``min_ratio`` of the needles, i.e. the answer is genuinely a lookup rather than a
    value to compute.
    """
    n = len(needles)
    if n == 0:
        return EvidenceLoc(files=(), n_needles=0, best_coverage=0, coverage_ratio=0.0, grounded=False)

    coverage: dict[str, int] = {}
    for c in chunks:
        f = c["file"]
        text = c["text"]
        hit = sum(1 for nd in needles if nd in text)
        if hit > coverage.get(f, 0):
            coverage[f] = hit

    best = max(coverage.values(), default=0)
    ratio = best / n if n else 0.0
    grounded = best >= 1 and ratio >= min_ratio
    files: tuple[str, ...] = ()
    if best >= 1:
        files = tuple(sorted(f for f, v in coverage.items() if v == best))[:_MAX_EVIDENCE_FILES]
    return EvidenceLoc(files=files, n_needles=n, best_coverage=best,
                       coverage_ratio=round(ratio, 4), grounded=grounded)


class Bm25FileRanker:
    """BM25 chunk ranker → per-file retrieval reachability (the lexical arm of production retrieve()).

    Uses the same :func:`src.rag.index.tokenize` and :class:`rank_bm25.BM25Okapi` as
    :class:`src.rag.retrieve.Retriever`, so the ranking mirrors the production sparse signal exactly.
    Deterministic and network-free (dense embeddings are intentionally omitted — see module docstring).
    """

    def __init__(self, chunks: list[dict]):
        self.files = [c["file"] for c in chunks]
        self.bm25 = BM25Okapi([index.tokenize(c["text"]) for c in chunks])

    def ranked_files(self, question: str) -> list[str]:
        """Per-chunk file list ordered by BM25 score for ``question`` — WITH repeats.

        Aligned to the BM25 chunk ranking (one entry per chunk, a file repeats once per chunk it owns).
        Kept chunk-level on purpose so ``ranked_files[:k]`` is exactly the file set of the top-k
        *chunks* — mirroring production ``retrieve(k=k)``, which returns k chunks (often several from
        the same file), not k distinct files.
        """
        scores = self.bm25.get_scores(index.tokenize(question))
        order = np.argsort(-scores)
        return [self.files[int(i)] for i in order]


@dataclass(frozen=True)
class CaseDiagnosis:
    """One classified ``both_abstain`` gold item."""

    index: int
    archetype: str
    classification: str          # retrieval_miss | derivation_miss
    subreason: str
    n_needles: int
    best_coverage: int
    coverage_ratio: float
    literal_grounded: bool
    evidence_files: list[str]
    retrieved_evidence_files: list[str]   # evidence files that landed within top-k
    evidence_rank: int | None    # best 1-based rank of an evidence file in the full BM25 order
    recall_hit: bool             # an evidence file is within top-k
    gold: str

    def to_dict(self) -> dict:
        return asdict(self)


def diagnose_case(*, idx: int, question: str, gold: str, archetype_label: str,
                  evidence: EvidenceLoc, ranked_files: list[str], k: int) -> CaseDiagnosis:
    """Pure classification of one ``both_abstain`` case from pre-computed evidence + ranking.

    Split out from IO so it is unit-testable with constructed inputs. ``ranked_files`` is the full
    per-chunk BM25 file order WITH repeats (from :meth:`Bm25FileRanker.ranked_files`), so
    ``ranked_files[:k]`` is the top-k *chunks*' file set (production-faithful recall@k); ``evidence``
    is the gold localization (from :func:`locate_evidence`). ``evidence_rank`` is the 1-based chunk
    rank at which an evidence file first appears.
    """
    topk = set(ranked_files[:k])
    ev_set = set(evidence.files)
    retrieved = sorted(ev_set & topk)
    recall_hit = bool(retrieved)

    # best rank (1-based) at which any evidence file first appears in the full ranking
    evidence_rank: int | None = None
    if ev_set:
        for rank, f in enumerate(ranked_files, start=1):
            if f in ev_set:
                evidence_rank = rank
                break

    if not evidence.grounded:
        classification, subreason = DERIVATION_MISS, SUB_NOT_A_LOOKUP
    elif recall_hit:
        classification, subreason = DERIVATION_MISS, SUB_IN_TOPK
    else:
        classification, subreason = RETRIEVAL_MISS, SUB_BELOW_TOPK

    return CaseDiagnosis(
        index=idx, archetype=archetype_label, classification=classification, subreason=subreason,
        n_needles=evidence.n_needles, best_coverage=evidence.best_coverage,
        coverage_ratio=evidence.coverage_ratio, literal_grounded=evidence.grounded,
        evidence_files=list(evidence.files), retrieved_evidence_files=retrieved,
        evidence_rank=evidence_rank, recall_hit=recall_hit, gold=gold,
    )


def both_abstain_indices(records: list[dict]) -> list[int]:
    """Indices of the ``both_abstain`` items (investigator AND verifier both abstained).

    Reuses :func:`scoring.abstain_breakdown.classify_record` as the single source of truth for what
    ``both_abstain`` means, so this diagnostic and the SOT-2483 breakdown never drift apart.
    """
    out = []
    for r in records:
        cls = classify_record(r)
        if cls is not None and cls.reason == BOTH_ABSTAIN:
            out.append(int(r.get("index", -1)))
    return out


def diagnose(records: list[dict], gold: dict[int, str], chunks: list[dict], *,
             k: int = DEFAULT_K, min_ratio: float = DEFAULT_MIN_RATIO,
             ranker: Bm25FileRanker | None = None) -> dict:
    """Full deterministic diagnosis of the ``both_abstain`` pool → the report payload.

    Builds the BM25 ranker (unless injected), classifies every ``both_abstain`` case, and aggregates
    counts by classification, by archetype, and the classification×archetype cross-tab.
    """
    if ranker is None:
        ranker = Bm25FileRanker(chunks)

    by_index = {int(r.get("index", -1)): r for r in records}
    cases: list[CaseDiagnosis] = []
    for idx in both_abstain_indices(records):
        rec = by_index.get(idx, {})
        question = str(rec.get("question") or "")
        g = gold.get(idx, "")
        evidence = locate_evidence(extract_needles(g), chunks, min_ratio=min_ratio)
        ranked = ranker.ranked_files(question)
        cases.append(diagnose_case(idx=idx, question=question, gold=g,
                                   archetype_label=archetype.classify(question),
                                   evidence=evidence, ranked_files=ranked, k=k))

    return _aggregate(cases, k=k, min_ratio=min_ratio)


def _aggregate(cases: list[CaseDiagnosis], *, k: int, min_ratio: float) -> dict:
    by_classification = {RETRIEVAL_MISS: 0, DERIVATION_MISS: 0}
    by_subreason: dict[str, int] = {}
    by_archetype: dict[str, dict] = {}
    for c in cases:
        by_classification[c.classification] = by_classification.get(c.classification, 0) + 1
        by_subreason[c.subreason] = by_subreason.get(c.subreason, 0) + 1
        b = by_archetype.setdefault(c.archetype or "unknown",
                                    {"total": 0, RETRIEVAL_MISS: 0, DERIVATION_MISS: 0})
        b["total"] += 1
        b[c.classification] += 1
    return {
        "schema_version": 1,
        "kind": "both_abstain_diagnosis",
        "issue": "SOT-2486",
        "k": k,
        "min_ratio": min_ratio,
        "retrieval_proxy": "bm25_lexical_only (dense omitted for determinism); "
                           "retrieval_miss is an upper bound, derivation_miss a lower bound",
        "n_both_abstain": len(cases),
        "by_classification": by_classification,
        "by_subreason": dict(sorted(by_subreason.items())),
        "by_archetype": dict(sorted(by_archetype.items())),
        "cases": [c.to_dict() for c in sorted(cases, key=lambda x: x.index)],
    }


def render(report: dict) -> str:
    lines = [
        f"# both_abstain 律速診断 (SOT-2486)  n={report['n_both_abstain']}  k={report['k']}",
        "",
        f"検索欠損 retrieval_miss : {report['by_classification'][RETRIEVAL_MISS]:>3}"
        "   (根拠が top-k に不在 → 検索拡張が効く)",
        f"導出欠損 derivation_miss: {report['by_classification'][DERIVATION_MISS]:>3}"
        "   (根拠は来ている/lookupでない → ツール・推論・確信度)",
        "",
        f"proxy: {report['retrieval_proxy']}",
        "",
        "## subreason 内訳",
    ]
    for sub, n in report["by_subreason"].items():
        lines.append(f"  {sub:<14} {n:>3}")
    lines += ["", "## 型別 (archetype) × 分類", "",
              f"| {'archetype':<20} | total | retrieval_miss | derivation_miss |",
              f"| {'-'*20} | ----- | -------------- | --------------- |"]
    for arch, b in report["by_archetype"].items():
        lines.append(f"| {arch:<20} | {b['total']:>5} | {b[RETRIEVAL_MISS]:>14} "
                     f"| {b[DERIVATION_MISS]:>15} |")
    lines += ["", "## per-case", "",
              f"| idx | archetype | class | subreason | ev_rank | needles(cov) | evidence file |",
              f"| --- | --------- | ----- | --------- | ------- | ------------ | ------------- |"]
    for c in report["cases"]:
        ev = c["evidence_files"][0] if c["evidence_files"] else "-"
        rank = c["evidence_rank"] if c["evidence_rank"] is not None else "-"
        lines.append(
            f"| {c['index']} | {c['archetype']} | {c['classification'].split('_')[0]} "
            f"| {c['subreason']} | {rank} | {c['best_coverage']}/{c['n_needles']} | {ev} |")
    return "\n".join(lines)


def _ledger_entry(report: dict) -> dict:
    ret = report["by_classification"][RETRIEVAL_MISS]
    der = report["by_classification"][DERIVATION_MISS]
    n = report["n_both_abstain"]
    top_arch = max(report["by_archetype"].items(),
                   key=lambda kv: kv[1]["total"], default=("-", {"total": 0}))
    return {
        "recordedAt": "REPLACE_ISO",
        "axis": "both-abstain-retrieval-vs-derivation-diagnosis",
        "result": "diagnostic",
        "cycle": 15,
        "hypothesis": "gold-100 の both_abstain=%d (真の無回答) の律速は検索欠損か導出欠損か。"
                      "決定論診断で切り分ければ、後続改善(検索拡張 vs ツール/推論拡張)を根拠付きで優先できる。" % n,
        "evidence": "決定論 (LLM無): gold needle の literal file-coverage で根拠を局在化し、BM25 (production "
                    "dense+BM25 retriever の lexical 成分) recall@%d で到達性を判定。both_abstain=%d を "
                    "retrieval_miss=%d / derivation_miss=%d に分類 (subreason %s)。最多型=%s。"
                    "BM25-only proxy のため retrieval_miss は上界・derivation_miss は下界。"
                    "本番 gate/resolve は不変 (byte 非破壊・関門2 非劣化)。成果物 "
                    "scoring/abstain_diagnosis.py, artifacts/both_abstain_diagnosis.{json,md}。"
                    % (report["k"], n, ret, der, report["by_subreason"], top_arch[0]),
    }


def append_ledger(report: dict, *, ledger_path: Path, now_iso: str) -> dict:
    """Append the diagnostic attribution entry to the experiment ledger (JSONL)."""
    entry = _ledger_entry(report)
    entry["recordedAt"] = now_iso
    ledger_path.parent.mkdir(parents=True, exist_ok=True)
    with ledger_path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(entry, ensure_ascii=False) + "\n")
    return entry


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        prog="scoring.abstain_diagnosis",
        description="Diagnose both_abstain gold-100 cases as retrieval-miss vs derivation-miss (SOT-2486).")
    ap.add_argument("--details", type=Path,
                    default=settings.ARTIFACTS_DIR / "predictions_test_resolve.details.jsonl",
                    help="resolve details.jsonl (per-item investigator/verifier/judge answers)")
    ap.add_argument("--gold", type=Path,
                    default=settings.ARTIFACTS_DIR / "predictions_test_v3_final.csv",
                    help="gold answers (headerless index,answer)")
    ap.add_argument("--k", type=int, default=DEFAULT_K, help="retrieval top-k for recall@k")
    ap.add_argument("--min-ratio", type=float, default=DEFAULT_MIN_RATIO,
                    help="min needle-coverage ratio for a gold answer to count as literally locatable")
    ap.add_argument("--out", type=Path,
                    default=settings.ARTIFACTS_DIR / "both_abstain_diagnosis.json",
                    help="write the JSON diagnosis here")
    ap.add_argument("--md-out", type=Path,
                    default=settings.ARTIFACTS_DIR / "both_abstain_diagnosis.md",
                    help="write the human-readable markdown here")
    ap.add_argument("--ledger", type=Path,
                    default=settings.REPO_ROOT / "docs" / "ai" / "experiment_ledger.jsonl",
                    help="append the diagnostic attribution entry here")
    ap.add_argument("--now", default=None,
                    help="ISO timestamp for the ledger entry (default: current UTC)")
    ap.add_argument("--no-ledger", action="store_true", help="skip the ledger attribution append")
    ap.add_argument("--json", dest="as_json", action="store_true",
                    help="print the full JSON report instead of the human summary")
    args = ap.parse_args(argv)

    records = load_details(args.details)
    gold = load_gold(args.gold)
    chunks = index.load_chunks()
    report = diagnose(records, gold, chunks, k=args.k, min_ratio=args.min_ratio)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    md = render(report)
    args.md_out.parent.mkdir(parents=True, exist_ok=True)
    args.md_out.write_text(md + "\n", encoding="utf-8")

    if not args.no_ledger:
        now_iso = args.now or _utc_now_iso()
        entry = append_ledger(report, ledger_path=args.ledger, now_iso=now_iso)
        print(f"ledger += {entry['axis']} ({entry['result']})  → {args.ledger}")

    print(json.dumps(report, ensure_ascii=False, indent=2) if args.as_json else md)
    print(f"\nreport: {args.out}\nmarkdown: {args.md_out}")
    return 0


def _utc_now_iso() -> str:
    from datetime import datetime, timezone
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


if __name__ == "__main__":
    raise SystemExit(main())
