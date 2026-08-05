"""Local CRAG judge — a faithful port of the official evaluator.py rubric.

The OFFICIAL SIGNATE score is computed server-side with OpenAI gpt-5.2. We reproduce the
SAME judging prompt/rubric locally, backed by Gemini by default (JUDGE_BACKEND=gemini), so
gate1/gate2 give an objective self-score without any OpenAI key. Set JUDGE_BACKEND=openai
(with OPENAI_API_KEY) for exact official parity spot-checks.

Score: Perfect +1, Acceptable +0.5, Missing 0, Incorrect -1; final = mean over questions.
"""
from __future__ import annotations

import json
import os
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

# Strict addendum: the Gemini proxy over-awards "Perfect" (~0.2 more lenient than the real
# gpt-5.2 judge). These clauses tip verbose / format-mismatched answers away from Perfect.
JUDGE_STRICT_ADDENDUM = textwrap.dedent("""
    # 厳格化ルール（過大評価の抑制）
    - answerがground_truthより著しく冗長で、ground_truthから確認できない具体的情報（数値・固有名詞・日付・条件・単位）を追加している場合は"Perfect"にせず、追加情報が誤り・未確認なら"Incorrect"、無害な言い換え程度なら"Acceptable"とする.
    - 求められている書式（単位・記法・列挙形式・キー: 値の形）とanswerの書式が食い違う場合、意味が同じでも"Perfect"にはせず"Acceptable"以下とする.
    - ground_truthの主要要素をanswerが省略・言い換えで曖昧化している場合は"Perfect"にしない.
    - 判定に迷う場合は寛容側（Perfect）ではなく厳格側（Acceptable/Incorrect）に倒す.
""").strip()

_SCHEMA = {
    "type": "object",
    "properties": {"judged": {"type": "string",
                              "enum": ["Perfect", "Acceptable", "Missing", "Incorrect"]}},
    "required": ["judged"],
}

_POINTS = {"Perfect": 1.0, "Acceptable": 0.5, "Missing": 0.0, "Incorrect": -1.0}


def strict_enabled() -> bool:
    """Strict rubric is on by default; JUDGE_STRICT=0/false/no/off falls back to majority."""
    return os.getenv("JUDGE_STRICT", "1").strip().lower() not in ("0", "false", "no", "off", "")


def _system_prompt(strict: bool) -> str:
    return f"{JUDGE_SYSTEM}\n\n{JUDGE_STRICT_ADDENDUM}" if strict else JUDGE_SYSTEM


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


def strict_downgrade(pred: str, truth: str, verdict: str) -> str:
    """Deterministically knock a lenient "Perfect" down for verbose / format-mismatched
    free-text answers (the idx17/23/28 failure type). Only ever downgrades, never upgrades,
    and only fires on free-text pairs — short/numeric/enumeration answers are untouched.
    """
    if verdict != "Perfect":
        return verdict
    p, t = _norm(pred), _norm(truth)
    # Free-text only: at least one side reads like prose (sentence punctuation or long).
    prose = bool(re.search(r"[。．！？!?、,]", p + t)) or max(len(p), len(t)) > 24
    if not prose:
        return verdict
    # Verbosity: the answer is materially longer than the reference model answer.
    verbose = len(p) >= len(t) + 12 and len(p) >= 1.5 * len(t)
    # Extra specific claims the ground_truth cannot confirm: numbers present in the answer
    # but absent from the ground_truth (unconfirmable specifics → the rubric forbids Perfect).
    p_nums = set(re.findall(r"-?\d[\d,]*(?:\.\d+)?", p))
    t_nums = set(re.findall(r"-?\d[\d,]*(?:\.\d+)?", t))
    extra_numeric = bool(p_nums - t_nums)
    if extra_numeric and verbose:
        return "Incorrect"  # adds unverifiable numeric specifics on top of padding
    if verbose or extra_numeric:
        return "Acceptable"  # over-long or format-drifted paraphrase; not a clean Perfect
    return verdict


def _judge_gemini(pred: str, truth: str, strict: bool = True) -> str:
    raw = llm.generate(
        f"ground_truth: {truth} answer: {pred}\n",
        system=_system_prompt(strict),
        model=settings.JUDGE_MODEL,
        temperature=0.0,
        thinking_budget=128,  # gemini-2.5-pro requires thinking>0; keeps judging deterministic-ish
        max_output_tokens=512,
        response_schema=_SCHEMA,
    )
    return json.loads(raw)["judged"]


def _judge_openai(pred: str, truth: str, strict: bool = True) -> str:
    from openai import OpenAI

    client = OpenAI()
    resp = client.chat.completions.create(
        model=settings.JUDGE_MODEL if settings.JUDGE_MODEL.startswith("gpt") else "gpt-5.2",
        temperature=0, seed=0,
        response_format={"type": "json_schema", "json_schema": {
            "name": "judgement_schema",
            "schema": {**_SCHEMA, "additionalProperties": False}}},
        messages=[{"role": "system", "content": _system_prompt(strict)},
                  {"role": "user", "content": f"ground_truth: {truth} answer: {pred}\n"}],
    ).choices[0].message.content
    return json.loads(resp)["judged"]


def judge(pred: str, truth: str, votes: int = 3, strict: bool | None = None) -> str:
    """Hybrid judge: deterministic first, then LLM. Strict mode (default, JUDGE_STRICT)
    aggregates the votes by worst case rather than majority and applies a deterministic
    verbosity/format downgrade, narrowing the Gemini proxy's ~0.2 leniency gap."""
    strict = strict_enabled() if strict is None else strict
    resolved = deterministic_judge(pred, truth)
    if resolved is not None:
        return resolved
    backend = settings.JUDGE_BACKEND.lower()
    base = _judge_openai if backend == "openai" else _judge_gemini
    one = (lambda p, t: base(p, t, strict))
    verdict = _aggregate(one, pred, truth, votes, strict)
    return strict_downgrade(pred, truth, verdict) if strict else verdict


def _aggregate(one, pred: str, truth: str, votes: int, strict: bool) -> str:
    """Strict → worst-value aggregation (min points over the votes); else majority-of-N."""
    if votes <= 1:
        return one(pred, truth)
    verdicts = [one(pred, truth) for _ in range(votes)]
    if strict:
        return min(verdicts, key=lambda v: _POINTS.get(v, 0.0))
    import collections

    return collections.Counter(verdicts).most_common(1)[0][0]


def _majority(one, pred: str, truth: str, votes: int) -> str:
    """Backwards-compatible majority aggregation (non-strict path / callers)."""
    return _aggregate(one, pred, truth, votes, strict=False)


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
