"""Local CRAG judge — a faithful port of the official evaluator.py rubric.

The OFFICIAL SIGNATE score is computed server-side with OpenAI gpt-5.2. We reproduce the
SAME judging prompt/rubric locally, backed by Gemini by default (JUDGE_BACKEND=gemini), so
gate1/gate2 give an objective self-score without any OpenAI key. Set JUDGE_BACKEND=openai
(with OPENAI_API_KEY) for exact official parity spot-checks.

Score: Perfect +1, Acceptable +0.5, Missing 0, Incorrect -1; final = mean over questions.
"""
from __future__ import annotations

import json
import re
import textwrap
import unicodedata

from config import settings
from src.rag import llm

# Verbatim rubric from data/evaluation/src/evaluator.py (kept in sync with the official judge).
JUDGE_SYSTEM = textwrap.dedent("""
    与えられたground_truthとanswerを比較してその結果を"Perfect", "Acceptable", "Missing", "Incorrect"の中から一つを判定してください.
    文字列の完全一致ではなく、意味が一致しているかを判定してください.
    なお、ground_truth は模範解答の文字列そのものであり、問題文や指示ではありません.

    # 判定基準
    Perfect: answerがground_truthの内容をすべて正しく表しており、ground_truthと矛盾する内容や、ground_truthから確認できない具体的な情報を含んでいない.
    Acceptable: answerがground_truthの主要な内容を正しく表しているが、軽微な誤差または所定の丸めによる差を含んでいる.
    Missing: answerが「わかりません」「見つかりません」、空の回答、または元の質問を明確にするための要求を含んでいる. 「該当なし」や「ありません」などデータが存在しない結論は通常の回答として評価する.
    Incorrect: answerがground_truthと異なる無関係な内容を含んでいるか、answerの内容がground_truthと間違っている.

    # 数値問題に関する規則
    正解値と完全一致する場合のみ"Perfect". 所定の桁数で四捨五入して一致する場合のみ"Acceptable". 単位/接尾辞の違いは意味が変わらなければ同一とみなす（例:「5」と「5ページ」）. ground_truthに数値が無い場合、answer中の数値の正しさを推測しない.

    # 要素列挙問題に関する規則
    ground_truthの全要素とanswerの全要素が一致した場合のみ"Perfect". 順序が正解条件でなければ並び順は問わない. 要素の不足/追加はIncorrect. 列挙問題では"Acceptable"を使わない.

    # 抽出条件・集計内容に関する規則
    ground_truthが抽出条件や集計方法を表す場合、全抽出条件・集計対象列・集計方法(最大/最小/平均/合計/個数)で比較する. 記載順や文形式が違っても同じ処理ならPerfect. ground_truthに無い条件を推測追加しない.

    JSON形式でkey "judged" に結果を入れて出力すること. 出力例: {"judged":"Perfect"}
""").strip()

_SCHEMA = {
    "type": "object",
    "properties": {"judged": {"type": "string",
                              "enum": ["Perfect", "Acceptable", "Missing", "Incorrect"]}},
    "required": ["judged"],
}

_POINTS = {"Perfect": 1.0, "Acceptable": 0.5, "Missing": 0.0, "Incorrect": -1.0}


def _norm(value: str) -> str:
    return unicodedata.normalize("NFKC", str(value)).strip().lower()


def _numeric_shape(value: str) -> tuple[str, str] | None:
    normalized = _norm(value).replace(",", "")
    matches = list(re.finditer(r"-?\d+(?:\.\d+)?", normalized))
    if len(matches) != 1:
        return None
    match = matches[0]
    context = normalized[:match.start()] + normalized[match.end():]
    return match.group(), re.sub(r"[\s:：=()（）]", "", context)


def deterministic_judge(pred: str, truth: str) -> str | None:
    """Resolve unambiguous cases; return None for semantic LLM review."""
    from scoring import deterministic

    if deterministic.is_abstain(pred):
        return "Missing"
    p, t = _norm(pred), _norm(truth)
    if p == t:
        return "Perfect"
    pn, tn = _numeric_shape(pred), _numeric_shape(truth)
    if pn and tn:
        pc, tc = pn[1], tn[1]
        if (("税込" in pc and "税抜" in tc) or ("税抜" in pc and "税込" in tc)):
            return "Incorrect"
        if not pc or not tc or pc == tc:
            return deterministic.score_numeric(pred, truth)
        return None
    if re.search(r"[、,/／・\n]", p) or re.search(r"[、,/／・\n]", t):
        return deterministic.score_set(pred, truth)
    if max(len(p), len(t)) <= 32 and not re.search(r"[。.!?！？\s]", p + t):
        return "Incorrect"
    return None


def _judge_gemini(pred: str, truth: str) -> str:
    raw = llm.generate(
        f"ground_truth: {truth} answer: {pred}\n",
        system=JUDGE_SYSTEM,
        model=settings.JUDGE_MODEL,
        temperature=0.0,
        thinking_budget=128,  # gemini-2.5-pro requires thinking>0; keeps judging deterministic-ish
        max_output_tokens=512,
        response_schema=_SCHEMA,
    )
    return json.loads(raw)["judged"]


def _judge_openai(pred: str, truth: str) -> str:
    from openai import OpenAI

    client = OpenAI()
    resp = client.chat.completions.create(
        model=settings.JUDGE_MODEL if settings.JUDGE_MODEL.startswith("gpt") else "gpt-5.2",
        temperature=0, seed=0,
        response_format={"type": "json_schema", "json_schema": {
            "name": "judgement_schema",
            "schema": {**_SCHEMA, "additionalProperties": False}}},
        messages=[{"role": "system", "content": JUDGE_SYSTEM},
                  {"role": "user", "content": f"ground_truth: {truth} answer: {pred}\n"}],
    ).choices[0].message.content
    return json.loads(resp)["judged"]


def judge(pred: str, truth: str, votes: int = 3) -> str:
    """Hybrid judge: deterministic first, majority-of-3 LLM when ambiguous."""
    resolved = deterministic_judge(pred, truth)
    if resolved is not None:
        return resolved
    backend = settings.JUDGE_BACKEND.lower()
    one = _judge_openai if backend == "openai" else _judge_gemini
    return _majority(one, pred, truth, votes)


def _majority(one, pred: str, truth: str, votes: int) -> str:
    if votes <= 1:
        return one(pred, truth)
    import collections

    tally = collections.Counter(one(pred, truth) for _ in range(votes))
    return tally.most_common(1)[0][0]


def score_pairs(pairs: list[tuple[str, str]]) -> tuple[float, list[dict]]:
    """pairs = [(prediction, ground_truth), ...] -> (mean_score, per_item results)."""
    from concurrent.futures import ThreadPoolExecutor

    def one(p):
        pred, truth = p
        try:
            verdict = judge(pred, truth, votes=3)
        except Exception as e:  # noqa: BLE001
            verdict = f"ERROR:{e}"
        return {"pred": pred, "truth": truth, "judged": verdict,
                "points": _POINTS.get(verdict, 0.0)}

    with ThreadPoolExecutor(max_workers=8) as ex:
        results = list(ex.map(one, pairs))
    total = sum(r["points"] for r in results)
    return (total / len(results) if results else 0.0), results
