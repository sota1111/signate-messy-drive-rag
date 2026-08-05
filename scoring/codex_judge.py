"""Codex CLI judge backend — local scoring on the SAME model family as the real grader.

The official SIGNATE score is computed with OpenAI gpt-5.2. SOT-2457 established that the
Gemini proxy judge is ~0.2 more lenient and its scores are uncorrelated with the real LB
(Spearman -0.09). The Codex CLI (`codex exec`, GPT-5.x) gives us the official judge's model
family WITHOUT an OpenAI API key, so `JUDGE_BACKEND=codex` routes the semantic-judgement
step here (the deterministic pre-pass in scoring.crag stays unchanged).

Mechanics: pairs are judged in BATCHES — one `codex exec` call scores a chunk of
(answer, ground_truth) items with the verbatim official rubric, `--sandbox read-only`,
`--output-schema` (structured verdicts) and `-o` (last-message file). A per-item CLI call
would pay ~10s of process startup per question; batching amortises it.

Env knobs (read directly from the environment — config/ is intentionally untouched):
- CODEX_JUDGE_MODEL   model id passed as `codex exec -m` (default: CLI default model)
- CODEX_JUDGE_BATCH   pairs per codex call (default 20)
- CODEX_JUDGE_VOTES   votes per pair (default 1; strict mode aggregates worst-case)
- CODEX_JUDGE_TIMEOUT seconds per codex call (default 600)
"""
from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

VERDICTS = ("Perfect", "Acceptable", "Missing", "Incorrect")

_OUTPUT_SCHEMA = {
    "type": "object",
    "properties": {
        "judgements": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "id": {"type": "integer"},
                    "judged": {"type": "string", "enum": list(VERDICTS)},
                },
                "required": ["id", "judged"],
                "additionalProperties": False,
            },
        }
    },
    "required": ["judgements"],
    "additionalProperties": False,
}


class CodexJudgeError(RuntimeError):
    """codex exec failed / timed out / returned unusable verdicts."""


def available() -> bool:
    return shutil.which("codex") is not None


def _int_env(key: str, default: int) -> int:
    try:
        return max(1, int(os.getenv(key, "") or default))
    except ValueError:
        return default


def votes() -> int:
    return _int_env("CODEX_JUDGE_VOTES", 1)


def batch_size() -> int:
    return _int_env("CODEX_JUDGE_BATCH", 20)


def timeout_s() -> int:
    return _int_env("CODEX_JUDGE_TIMEOUT", 600)


def _build_prompt(system_prompt: str, chunk: list[tuple[str, str]]) -> str:
    items = [{"id": i, "ground_truth": truth, "answer": pred}
             for i, (pred, truth) in enumerate(chunk)]
    return (
        f"{system_prompt}\n\n"
        "以下の items の各要素について、上記の判定基準で ground_truth と answer を比較し判定してください。\n"
        "ファイルやコマンドの実行は不要です。items のテキストのみで判定してください。\n"
        '出力は JSON オブジェクト {"judgements": [{"id": <id>, "judged": "<判定>"}, ...]} とし、'
        "全ての id をちょうど1回ずつ含めてください。\n\n"
        f"items = {json.dumps(items, ensure_ascii=False)}"
    )


def _extract_json(text: str) -> dict:
    text = text.strip()
    fenced = re.search(r"```(?:json)?\s*(\{.*\})\s*```", text, re.DOTALL)
    if fenced:
        text = fenced.group(1)
    start = text.find("{")
    if start < 0:
        raise CodexJudgeError(f"no JSON object in codex output: {text[:200]!r}")
    return json.loads(text[start:text.rfind("}") + 1])


def _run_codex_chunk(system_prompt: str, chunk: list[tuple[str, str]]) -> list[str]:
    """One `codex exec` call → verdict per pair (chunk order)."""
    if not available():
        raise CodexJudgeError("codex CLI not found on PATH")
    with tempfile.TemporaryDirectory(prefix="codex-judge-") as tmp:
        schema_path = Path(tmp) / "schema.json"
        out_path = Path(tmp) / "last_message.json"
        schema_path.write_text(json.dumps(_OUTPUT_SCHEMA), encoding="utf-8")
        cmd = ["codex", "exec",
               "--sandbox", "read-only", "--ephemeral", "--skip-git-repo-check",
               "--color", "never", "-C", tmp,
               "--output-schema", str(schema_path), "-o", str(out_path)]
        model = os.getenv("CODEX_JUDGE_MODEL", "").strip()
        if model:
            cmd += ["-m", model]
        cmd.append(_build_prompt(system_prompt, chunk))
        try:
            proc = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout_s())
        except subprocess.TimeoutExpired as e:
            raise CodexJudgeError(f"codex exec timed out after {timeout_s()}s") from e
        if proc.returncode != 0 or not out_path.exists():
            tail = (proc.stderr or proc.stdout or "").strip()[-400:]
            raise CodexJudgeError(
                f"codex exec failed (exit {proc.returncode}): {tail!r}")
        payload = _extract_json(out_path.read_text(encoding="utf-8"))
    by_id: dict[int, str] = {}
    for row in payload.get("judgements", []):
        if isinstance(row, dict) and row.get("judged") in VERDICTS:
            by_id.setdefault(int(row["id"]), row["judged"])
    missing = [i for i in range(len(chunk)) if i not in by_id]
    if missing:
        raise CodexJudgeError(f"codex verdicts missing ids {missing[:5]} "
                              f"(got {len(by_id)}/{len(chunk)})")
    return [by_id[i] for i in range(len(chunk))]


def judge_batch(pairs: list[tuple[str, str]], system_prompt: str,
                strict: bool = True, n_votes: int | None = None) -> list[str]:
    """Judge (pred, truth) pairs with the Codex CLI; returns one verdict per pair.

    votes>1 repeats each chunk and aggregates per item: strict = worst-case points
    (mirrors scoring.crag strict aggregation), non-strict = majority.
    """
    from scoring.crag import _POINTS

    n_votes = votes() if n_votes is None else max(1, n_votes)
    verdicts: list[str] = []
    for lo in range(0, len(pairs), batch_size()):
        chunk = pairs[lo:lo + batch_size()]
        rounds = [_run_codex_chunk(system_prompt, chunk) for _ in range(n_votes)]
        for i in range(len(chunk)):
            item_votes = [r[i] for r in rounds]
            if strict:
                verdicts.append(min(item_votes, key=lambda v: _POINTS.get(v, 0.0)))
            else:
                import collections
                verdicts.append(collections.Counter(item_votes).most_common(1)[0][0])
    return verdicts


def judge_one(pred: str, truth: str, system_prompt: str, strict: bool = True,
              n_votes: int | None = None) -> str:
    return judge_batch([(pred, truth)], system_prompt, strict, n_votes)[0]


def warn(msg: str) -> None:
    print(f"[codex_judge] {msg}", file=sys.stderr)
