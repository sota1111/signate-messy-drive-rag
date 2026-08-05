"""Deterministic, meaning-preserving answer normalization (SOT-2448 / B2).

The official metric is an LLM judge (gpt-5.2) that compares each answer to a model answer.
Observed leniency-gap analysis (SOT-2446) shows the real judge tips *verbose / off-format*
answers toward Incorrect — extra prose, redundant units, stray whitespace, polite copulas and
answer-labels that a concise ground-truth style would not carry. This layer canonicalises a
*committed* answer toward that concise ground-truth style **without changing its meaning**, so it
can only lift the judgment of an already-committed answer — it never answers a new question and
never removes a fact.

Safety contract — the single reason this is the lowest-risk score lever:
  * It is a pure post-processing reformat of an answer the pipeline already decided to commit. It is
    NEVER applied to an abstention, so it cannot turn a Missing (0) into an Incorrect (−1).
  * Every candidate reformat is gated by :func:`preserves_meaning`. A candidate is accepted ONLY when
    it preserves *every* number and *every* ASCII identifier verbatim and introduces *no* new
    (non-whitespace) character. If the check fails, the ORIGINAL answer is returned unchanged
    (identity fallback). So a normalization can only ever *drop* redundant formatting/prose that
    carries neither a number nor an identifier — it can never drop a value or fabricate one.

Because the transform is deterministic and value-preserving, applying it to a correct answer keeps
it correct under the deterministic scorer (proven in ``scoring.test_normalize``), so it cannot
regress the sealed hold-out or the real-style generalization axes (the two-axis adoption gate).
"""
from __future__ import annotations

import os
import re
import unicodedata

from config import settings

# Toggle (config/ is read-only in this repo, so this is an env flag read directly). Default ON.
_ENV_FLAG = "RAG_ANSWER_NORMALIZE"


def enabled() -> bool:
    return os.getenv(_ENV_FLAG, "1").strip().lower() not in {"0", "false", "no", "off", ""}


# Answers we must never touch — abstentions and the canonical "no match" reply. Normalising these
# risks nudging a Missing (0) toward an Incorrect (−1), the exact loss the abstention gate avoids.
_ABSTAIN_LIKE = {"", "該当なし", "不明", "N/A", "n/a", "なし", "-", "—"}


def _is_abstain(answer: str) -> bool:
    a = unicodedata.normalize("NFC", answer or "").strip()
    return a == settings.ABSTAIN or a in _ABSTAIN_LIKE


# ------------------------------- meaning-preservation primitives ------------------------------
_NUMBER = re.compile(r"\d[\d,]*(?:\.\d+)?")
_ASCII_TOKEN = re.compile(r"[A-Za-z0-9_]+")


def numbers(s: str) -> list[str]:
    """Every numeric literal, with thousands separators removed so 25,000 == 25000."""
    return sorted(m.group(0).replace(",", "") for m in _NUMBER.finditer(unicodedata.normalize("NFC", s)))


def identifiers(s: str) -> list[str]:
    """ASCII identifiers of length ≥2 that contain a letter (task/action IDs, param names, model
    names, column names). These carry answer meaning and must survive normalization verbatim."""
    toks = _ASCII_TOKEN.findall(unicodedata.normalize("NFC", s))
    return sorted(t for t in toks if len(t) >= 2 and any(c.isalpha() for c in t))


def _content_chars(s: str) -> set[str]:
    """Non-whitespace characters — used to forbid the introduction of any new character."""
    return {c for c in unicodedata.normalize("NFC", s) if not c.isspace()}


def preserves_meaning(original: str, candidate: str) -> bool:
    """True when ``candidate`` may safely replace ``original``.

    Requires: identical number multiset (order-insensitive, separators normalised), identical ASCII
    identifier set, no candidate character absent from the original (no fabrication), and non-empty.
    """
    if not candidate.strip():
        return False
    if numbers(candidate) != numbers(original):
        return False
    if identifiers(candidate) != identifiers(original):
        return False
    # Never introduce a character that was not already present (guards against fabrication; the
    # transform may only delete redundant formatting/prose, never add content).
    if not _content_chars(candidate) <= _content_chars(original):
        return False
    return True


# ---------------------------------- deterministic transforms ----------------------------------
_LABEL = re.compile(r"^(?:回答|答え|解答|結論|Answer|A)\s*[:：]\s*")
# Trailing polite copulas and fixed hedge *clauses* a concise ground-truth answer would not carry.
# Anchored to the end and limited to self-contained removable phrases — we deliberately do NOT strip
# a bare verb ending like "…します" (removing it would require a stem change and risk meaning).
_TRAILING_COPULA = re.compile(
    r"(?:で(?:す|した)|である|であった|と考えられます|と思われます|と言えます|とのことです)。?$"
)
# A run of ASCII/full-width spaces sitting between a digit and a Japanese counter/unit — Japanese
# never separates a number from its unit, so this is pure formatting noise ("100 万" -> "100万").
# Restricted to a curated unit set so it never fires on a digit that merely precedes prose/punctuation
# (e.g. a list marker "5. 業務" or "→ 5"), which the whitespace-strict deterministic scorer would treat
# as a different string.
_UNIT = ("万|億|兆|千|百|円|ドル|％|%|個|件|人|名|社|年|月|日|時間|時|分|秒|週|回|台|"
         "枚|本|点|割|倍|度|色|行|列|位|番|歳|㎡|平方メートル|パーセント|ポイント|割合")
_DIGIT_UNIT_SPACE = re.compile(rf"(?<=\d)[ 　]+(?=(?:{_UNIT}))")


def _candidate(answer: str) -> str:
    c = unicodedata.normalize("NFC", answer).strip()
    c = _LABEL.sub("", c)
    c = _TRAILING_COPULA.sub("", c).rstrip()
    c = re.sub(r"。$", "", c)               # trailing lone full stop on a value/list answer
    c = re.sub(r"[ \t　]+", " ", c)      # collapse redundant whitespace runs
    c = _DIGIT_UNIT_SPACE.sub("", c)
    c = re.sub(r" *、 *", "、", c)            # tidy ASCII spacing around the JP list delimiter
    c = re.sub(r"、+", "、", c).strip("、 ")   # collapse empty list fields
    return c.strip()


def normalize_answer(answer: str, *, question: str = "") -> str:
    """Return the meaning-preserving normalized answer, or ``answer`` unchanged.

    Never alters an abstention. Applies the deterministic reformat only when :func:`preserves_meaning`
    confirms every value/identifier is retained and nothing new is introduced; otherwise returns the
    original verbatim (identity fallback). ``question`` is accepted for future format-aware rules.
    """
    if not enabled() or _is_abstain(answer):
        return answer
    candidate = _candidate(answer)
    if candidate != answer and preserves_meaning(answer, candidate):
        return candidate
    return answer


def normalize_with_flag(answer: str, *, question: str = "") -> tuple[str, bool]:
    """Like :func:`normalize_answer` but also reports whether the answer was changed."""
    out = normalize_answer(answer, question=question)
    return out, out != answer
