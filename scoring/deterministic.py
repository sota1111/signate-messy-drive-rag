"""Deterministic answer scorer — a noise-free stand-in for the LLM CRAG judge.

The real gpt-5.2 judge is unavailable locally and the Gemini proxy is too noisy to optimize
against. For SYNTHETIC benchmark items whose ground truth we extracted programmatically, we
can score exactly with typed comparators (numeric / set / string), mirroring the official
rubric (Perfect+1 / Acceptable+0.5 / Missing0 / Incorrect−1) without any model call.
"""
from __future__ import annotations

import re
import unicodedata

ABSTAIN_TOKENS = ("わかりません", "分かりません", "不明")
POINTS = {"Perfect": 1.0, "Acceptable": 0.5, "Missing": 0.0, "Incorrect": -1.0}


def _nfkc(s: str) -> str:
    return unicodedata.normalize("NFKC", str(s)).strip()


def is_abstain(pred: str) -> bool:
    p = _nfkc(pred)
    return (not p) or any(t in p for t in ABSTAIN_TOKENS)


def _numbers(s: str) -> list[float]:
    s = _nfkc(s).replace(",", "")
    return [float(x) for x in re.findall(r"-?\d+(?:\.\d+)?", s)]


def score_numeric(pred: str, truth: str, round_dp: int | None = None) -> str:
    if is_abstain(pred):
        return "Missing"
    tp, tt = _numbers(pred), _numbers(truth)
    if not tp or not tt:
        return "Incorrect"
    p, t = tp[0], tt[0]
    if abs(p - t) < 1e-9:
        return "Perfect"
    # Acceptable: matches when either side rounded to the ground truth's precision
    dp = round_dp if round_dp is not None else (len(truth.split(".")[1]) if "." in truth else 0)
    if round(p, dp) == round(t, dp):
        return "Acceptable"
    return "Incorrect"


def _tokens_set(s: str) -> set[str]:
    s = _nfkc(s)
    parts = re.split(r"[、,、/／・\n]+", s)
    return {_nfkc(p).lower() for p in parts if _nfkc(p)}


def score_set(pred: str, truth: str) -> str:
    """Enumeration: all elements must match (order-independent); missing/extra → Incorrect."""
    if is_abstain(pred):
        return "Missing"
    ps, ts = _tokens_set(pred), _tokens_set(truth)
    if not ts:
        return "Incorrect"
    # element-wise containment tolerance (pred element may include the truth element)
    def covered(t: str, pool: set[str]) -> bool:
        return any(t == x or t in x or x in t for x in pool)
    if all(covered(t, ps) for t in ts) and all(covered(p, ts) for p in ps):
        return "Perfect"
    return "Incorrect"


def score_string(pred: str, truth: str) -> str:
    if is_abstain(pred):
        return "Missing"
    p, t = _nfkc(pred).lower(), _nfkc(truth).lower()
    if not t:
        return "Incorrect"
    if p == t or t in p or p in t:
        return "Perfect"
    return "Incorrect"


SCORERS = {"numeric": score_numeric, "set": score_set, "string": score_string}


def score(pred: str, truth: str, kind: str) -> str:
    return SCORERS.get(kind, score_string)(pred, truth)
