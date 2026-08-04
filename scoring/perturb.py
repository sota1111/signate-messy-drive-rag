"""Deterministic meaning-preserving question perturbations and answer-stability scoring."""
from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass
from typing import Callable, Iterable

_ALIASES = (
    ("社内用語", "社内で使う略称"),
    ("正式名称", "正式な名称"),
    ("プロジェクト", "案件"),
    ("すべて", "全て"),
    ("答えてください", "教えてください"),
)


def variants(question: str, limit: int = 3) -> list[str]:
    """Return unique, deterministic paraphrases without changing requested facts."""
    q = question.strip()
    candidates: list[str] = []
    for old, new in _ALIASES:
        if old in q:
            candidates.append(q.replace(old, new))
            break
        if new in q:
            candidates.append(q.replace(new, old))
            break
    # Move a leading context ending in 「について、」 behind the request (word-order perturbation).
    m = re.match(r"^(.+?について)[、,](.+)$", q)
    if m:
        candidates.append(f"{m.group(2).rstrip('。')}。対象は{m.group(1)}です。")
    candidates.append(f"次の点を確認してください。{q}")
    out: list[str] = []
    for candidate in candidates:
        if candidate != q and candidate not in out:
            out.append(candidate)
        if len(out) >= limit:
            break
    return out


def normalize_answer(answer: object) -> str:
    text = unicodedata.normalize("NFKC", str(answer or "")).lower()
    return re.sub(r"[\s_、,。．・:：;；!?！？]+", "", text)


@dataclass(frozen=True)
class Robustness:
    stable: int
    total: int
    score: float
    unstable_ids: tuple[str, ...]


def score(answer_groups: Iterable[tuple[str, object, Iterable[object]]]) -> Robustness:
    """Measure questions for which every perturbed answer agrees with the original answer."""
    stable = total = 0
    unstable: list[str] = []
    for item_id, base, changed in answer_groups:
        answers = list(changed)
        if not answers:
            continue
        total += 1
        base_norm = normalize_answer(base)
        if base_norm and all(normalize_answer(x) == base_norm for x in answers):
            stable += 1
        else:
            unstable.append(item_id)
    return Robustness(stable, total, stable / total if total else 0.0, tuple(unstable))


def evaluate(items: Iterable[object], answer: Callable[[str], object], limit: int = 3) -> Robustness:
    groups = []
    for item in items:
        qs = variants(str(getattr(item, "question")), limit=limit)
        groups.append((str(getattr(item, "id")), answer(str(getattr(item, "question"))),
                       [answer(q) for q in qs]))
    return score(groups)
