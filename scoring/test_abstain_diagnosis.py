"""Network-free regressions for the both_abstain retrieval-vs-derivation diagnosis (SOT-2486).

All deterministic and LLM-free: needle extraction, evidence localization, the BM25 file ranker and
the pure :func:`diagnose_case` classifier are exercised over constructed chunks / records — no Vertex,
no gold judge. A tiny in-memory corpus stands in for the real index.
"""
from __future__ import annotations

import json

import pytest

from scoring import abstain_diagnosis as AD
from scoring.abstain_breakdown import BOTH_ABSTAIN


def _chunk(file, text, *, rel=None):
    return {"id": 0, "project": "p", "category": "c", "file": file,
            "rel": rel or f"dir/{file}", "kind": "body", "text": text}


def _rec(index, question, *, answer="わかりません", inv=None, ver=None):
    return {"index": index, "question": question, "answer": answer,
            "investigator_answer": inv if inv is not None else "わかりません",
            "verifier_answer": ver if ver is not None else "わかりません",
            "judge_answer": "わかりません", "status": "abstained",
            "verdict": {"category": ""}}


# --------------------------------------------------------------------------- needle extraction
def test_extract_needles_keeps_distinctive_tokens_drops_short_ones():
    nd = AD.extract_needles("着手金は40%、AUC-ROC と bmi、ノイズは の に")
    assert "40%" in nd and "AUC-ROC" in nd and "bmi" in nd
    # 1–2 char hiragana particles / fragments are excluded
    assert "の" not in nd and "に" not in nd


def test_extract_needles_empty():
    assert AD.extract_needles("") == set()
    assert AD.extract_needles(None) == set()


# --------------------------------------------------------------------------- evidence localization
def test_locate_evidence_grounds_on_covering_file():
    chunks = [
        _chunk("proposal.pptx", "着手金40% 検収金60% を 50% に変更"),
        _chunk("noise.docx", "無関係な内容"),
    ]
    needles = AD.extract_needles("着手金40%・検収金60%・50%")
    loc = AD.locate_evidence(needles, chunks, min_ratio=0.5)
    assert loc.grounded is True
    assert loc.files == ("proposal.pptx",)
    assert loc.best_coverage >= 2


def test_locate_evidence_not_grounded_when_answer_not_a_lookup():
    # gold answer "bmi" is a derived-column name that appears in no source chunk.
    chunks = [_chunk("data.xlsx", "height weight age という列がある")]
    loc = AD.locate_evidence(AD.extract_needles("bmi"), chunks, min_ratio=0.5)
    assert loc.grounded is False
    assert loc.files == ()


def test_locate_evidence_no_needles():
    loc = AD.locate_evidence(set(), [_chunk("f.txt", "x")], min_ratio=0.5)
    assert loc.grounded is False and loc.n_needles == 0


# --------------------------------------------------------------------------- BM25 ranker
def test_bm25_ranker_is_chunk_level_with_repeats_relevant_first():
    chunks = [
        _chunk("salary.docx", "機械学習エンジニア の 米国平均給与 は 高い", rel="a/salary.docx"),
        _chunk("salary.docx", "データサイエンティスト の 平均給与", rel="a/salary.docx"),
        _chunk("other.docx", "全く無関係な天気の話"),
    ]
    ranker = AD.Bm25FileRanker(chunks)
    files = ranker.ranked_files("機械学習エンジニア の 平均給与")
    assert files[0] == "salary.docx"
    # one entry per chunk (with repeats), aligned to production retrieve(k) returning k chunks
    assert len(files) == 3
    assert files.count("salary.docx") == 2
    # top-2 *chunks* are both salary.docx → recall@2 reaches salary.docx but not other.docx
    assert set(files[:2]) == {"salary.docx"}


# --------------------------------------------------------------------------- pure classification
def _loc(files, *, grounded, n=2, best=2):
    return AD.EvidenceLoc(files=tuple(files), n_needles=n, best_coverage=best,
                          coverage_ratio=best / n if n else 0.0, grounded=grounded)


def test_diagnose_case_retrieval_miss_when_evidence_below_topk():
    d = AD.diagnose_case(idx=1, question="q", gold="g", archetype_label="fact_lookup",
                         evidence=_loc(["ev.docx"], grounded=True),
                         ranked_files=["a", "b", "c", "ev.docx"], k=2)
    assert d.classification == AD.RETRIEVAL_MISS
    assert d.subreason == AD.SUB_BELOW_TOPK
    assert d.recall_hit is False
    assert d.evidence_rank == 4          # 1-based rank of ev.docx in the full order


def test_diagnose_case_derivation_miss_when_evidence_in_topk():
    d = AD.diagnose_case(idx=2, question="q", gold="g", archetype_label="fact_lookup",
                         evidence=_loc(["ev.docx"], grounded=True),
                         ranked_files=["ev.docx", "b", "c"], k=2)
    assert d.classification == AD.DERIVATION_MISS
    assert d.subreason == AD.SUB_IN_TOPK
    assert d.recall_hit is True
    assert d.retrieved_evidence_files == ["ev.docx"]
    assert d.evidence_rank == 1


def test_diagnose_case_derivation_miss_when_not_a_lookup():
    d = AD.diagnose_case(idx=3, question="q", gold="bmi", archetype_label="derived_calculation",
                         evidence=_loc([], grounded=False, n=1, best=0),
                         ranked_files=["a", "b"], k=2)
    assert d.classification == AD.DERIVATION_MISS
    assert d.subreason == AD.SUB_NOT_A_LOOKUP
    assert d.evidence_rank is None


# --------------------------------------------------------------------------- end-to-end (in-memory)
def test_diagnose_end_to_end_over_constructed_corpus():
    chunks = [
        _chunk("proposal.pptx", "着手金40% 検収金60% を それぞれ 50% に変更した"),
        _chunk("salary.docx", "機械学習エンジニア の 平均給与"),
        _chunk("filler.docx", "無関係"),
    ]
    records = [
        # both_abstain, gold literally in proposal.pptx which BM25 surfaces for the question → derivation
        _rec(0, "提案書の支払内訳の変更は？"),
        # both_abstain, gold not a lookup (computed) → derivation (not_a_lookup)
        _rec(1, "身長体重から算出した指標は？"),
        # NOT both_abstain (verifier proposed) → excluded from the pool
        _rec(2, "何か", ver="東京"),
        # committed → excluded
        _rec(3, "回答済み", answer="42"),
    ]
    gold = {0: "着手金50%・検収金50%", 1: "bmi", 2: "x", 3: "y"}

    report = AD.diagnose(records, gold, chunks, k=3)
    assert report["n_both_abstain"] == 2               # only idx 0 and 1
    assert {c["index"] for c in report["cases"]} == {0, 1}
    # both are derivation_miss in this tiny corpus (0 in_topk, 1 not_a_lookup)
    assert report["by_classification"][AD.DERIVATION_MISS] == 2
    assert report["by_classification"][AD.RETRIEVAL_MISS] == 0
    # cross-tab and subreasons populated
    assert set(report["by_subreason"]) <= {AD.SUB_IN_TOPK, AD.SUB_NOT_A_LOOKUP, AD.SUB_BELOW_TOPK}
    assert report["by_archetype"]


def test_both_abstain_indices_matches_breakdown_reason():
    records = [_rec(0, "q0"), _rec(1, "q1", ver="値"), _rec(2, "q2", answer="commit")]
    assert AD.both_abstain_indices(records) == [0]     # only the true both-abstain


# --------------------------------------------------------------------------- ledger append
def test_append_ledger_writes_diagnostic_entry(tmp_path):
    report = {
        "k": 16, "n_both_abstain": 2,
        "by_classification": {AD.RETRIEVAL_MISS: 1, AD.DERIVATION_MISS: 1},
        "by_subreason": {AD.SUB_BELOW_TOPK: 1, AD.SUB_NOT_A_LOOKUP: 1},
        "by_archetype": {"fact_lookup": {"total": 2, AD.RETRIEVAL_MISS: 1, AD.DERIVATION_MISS: 1}},
    }
    path = tmp_path / "experiment_ledger.jsonl"
    entry = AD.append_ledger(report, ledger_path=path, now_iso="2026-08-06T05:00:00Z")
    assert entry["axis"] == "both-abstain-retrieval-vs-derivation-diagnosis"
    assert entry["result"] == "diagnostic"
    written = [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]
    assert written[-1]["recordedAt"] == "2026-08-06T05:00:00Z"
    assert written[-1]["axis"] == entry["axis"]
