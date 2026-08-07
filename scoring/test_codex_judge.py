"""Network-free regressions for the Codex CLI judge backend (SOT-2457).

No test here launches the real `codex exec` — subprocess/judge entry points are stubbed.
"""
import json
import subprocess
from pathlib import Path

import pytest

from scoring import codex_judge, crag


# ---- backend resolution ----

def test_default_backend_is_codex_when_cli_present(monkeypatch):
    monkeypatch.delenv("JUDGE_BACKEND", raising=False)
    monkeypatch.setattr(codex_judge, "available", lambda: True)
    assert crag.resolve_backend() == "codex"


def test_explicit_env_backend_wins_over_codex(monkeypatch):
    monkeypatch.setenv("JUDGE_BACKEND", "gemini")
    monkeypatch.setattr(codex_judge, "available", lambda: True)
    assert crag.resolve_backend() == "gemini"


def test_no_cli_raises_instead_of_silent_gemini(monkeypatch):
    # SOT-2457 directive: no automatic Gemini substitution — a missing codex CLI is an error.
    monkeypatch.delenv("JUDGE_BACKEND", raising=False)
    monkeypatch.setattr(codex_judge, "available", lambda: False)
    with pytest.raises(RuntimeError, match="codex CLI not found"):
        crag.resolve_backend()


# ---- codex exec invocation / parsing ----

def _fake_codex_run(verdicts_by_call):
    """subprocess.run stub: writes the -o last-message file like codex exec would."""
    calls = {"n": 0, "cmds": []}

    def fake_run(cmd, **kwargs):
        calls["cmds"].append(cmd)
        out_path = Path(cmd[cmd.index("-o") + 1])
        prompt = cmd[-1]
        items = json.loads(prompt[prompt.index("items = ") + len("items = "):])
        verdicts = verdicts_by_call[min(calls["n"], len(verdicts_by_call) - 1)]
        payload = {"judgements": [{"id": it["id"], "judged": verdicts[it["id"]]}
                                  for it in items]}
        out_path.write_text(json.dumps(payload), encoding="utf-8")
        calls["n"] += 1
        return subprocess.CompletedProcess(cmd, 0, stdout="", stderr="")

    return fake_run, calls


def test_batch_judges_all_pairs_in_one_call(monkeypatch):
    fake_run, calls = _fake_codex_run([["Perfect", "Incorrect", "Missing"]])
    monkeypatch.setattr(codex_judge, "available", lambda: True)
    monkeypatch.setattr(subprocess, "run", fake_run)
    out = codex_judge.judge_batch([("a", "b"), ("c", "d"), ("e", "f")], "SYS")
    assert out == ["Perfect", "Incorrect", "Missing"]
    assert calls["n"] == 1  # one CLI call for the whole chunk
    cmd = calls["cmds"][0]
    assert cmd[:2] == ["codex", "exec"]
    assert "read-only" in cmd and "--output-schema" in cmd


def test_batch_chunking_respects_batch_size(monkeypatch):
    fake_run, calls = _fake_codex_run([["Perfect"] * 20])
    monkeypatch.setattr(codex_judge, "available", lambda: True)
    monkeypatch.setattr(subprocess, "run", fake_run)
    monkeypatch.setenv("CODEX_JUDGE_BATCH", "2")
    out = codex_judge.judge_batch([("a", "b")] * 5, "SYS")
    assert out == ["Perfect"] * 5
    assert calls["n"] == 3  # 2 + 2 + 1


def test_votes_aggregate_worst_case_in_strict(monkeypatch):
    fake_run, _ = _fake_codex_run([["Perfect"], ["Incorrect"], ["Perfect"]])
    monkeypatch.setattr(codex_judge, "available", lambda: True)
    monkeypatch.setattr(subprocess, "run", fake_run)
    out = codex_judge.judge_batch([("a", "b")], "SYS", strict=True, n_votes=3)
    assert out == ["Incorrect"]


def test_missing_ids_raise(monkeypatch):
    def fake_run(cmd, **kwargs):
        out_path = Path(cmd[cmd.index("-o") + 1])
        out_path.write_text(json.dumps({"judgements": [{"id": 0, "judged": "Perfect"}]}),
                            encoding="utf-8")
        return subprocess.CompletedProcess(cmd, 0, stdout="", stderr="")

    monkeypatch.setattr(codex_judge, "available", lambda: True)
    monkeypatch.setattr(subprocess, "run", fake_run)
    with pytest.raises(codex_judge.CodexJudgeError):
        codex_judge.judge_batch([("a", "b"), ("c", "d")], "SYS")


def test_model_env_adds_m_flag(monkeypatch):
    fake_run, calls = _fake_codex_run([["Perfect"]])
    monkeypatch.setattr(codex_judge, "available", lambda: True)
    monkeypatch.setattr(subprocess, "run", fake_run)
    monkeypatch.setenv("CODEX_JUDGE_MODEL", "gpt-5.2")
    codex_judge.judge_batch([("a", "b")], "SYS")
    cmd = calls["cmds"][0]
    assert cmd[cmd.index("-m") + 1] == "gpt-5.2"


# ---- crag integration ----

_PROSE_PRED = "主要な内容を自然文で説明しています。"
_PROSE_TRUTH = "同じ意味を別の自然文で説明しています。"


def test_crag_judge_routes_to_codex(monkeypatch):
    monkeypatch.setenv("JUDGE_BACKEND", "codex")
    monkeypatch.setattr(codex_judge, "judge_one",
                        lambda p, t, sys_prompt, strict, n_votes=None: "Acceptable")
    assert crag.judge(_PROSE_PRED, _PROSE_TRUTH) == "Acceptable"


def test_crag_judge_codex_majority_keeps_strict_prompt(monkeypatch):
    monkeypatch.setenv("JUDGE_BACKEND", "codex")
    seen = {}

    def fake_one(pred, truth, sys_prompt, strict, n_votes=None):
        seen.update(sys_prompt=sys_prompt, strict=strict, n_votes=n_votes)
        return "Acceptable"

    monkeypatch.setattr(codex_judge, "judge_one", fake_one)
    assert crag.judge(_PROSE_PRED, _PROSE_TRUTH,
                      votes=3, majority=True) == "Acceptable"
    assert crag.JUDGE_STRICT_ADDENDUM in seen["sys_prompt"]
    assert seen["strict"] is False and seen["n_votes"] == 3


def test_crag_judge_codex_applies_strict_downgrade(monkeypatch):
    # Verbose free-text "Perfect" from the LLM must still be knocked down deterministically.
    truth = "未連絡"
    pred = "pdaysの値-1は、これまで一度も連絡実績がなく未連絡であり、該当件数は約40件です。"
    monkeypatch.setenv("JUDGE_BACKEND", "codex")
    monkeypatch.setattr(crag, "deterministic_judge", lambda *_a: None)
    monkeypatch.setattr(codex_judge, "judge_one",
                        lambda p, t, sys_prompt, strict, n_votes=None: "Perfect")
    assert crag.judge(pred, truth) == "Incorrect"


def test_crag_judge_codex_error_propagates_no_gemini_fallback(monkeypatch):
    # SOT-2457 directive: a codex failure stops the run; Gemini must never be called.
    monkeypatch.setenv("JUDGE_BACKEND", "codex")

    def boom(*_a, **_k):
        raise codex_judge.CodexJudgeError("usage limit")

    def gemini_forbidden(*_a, **_k):
        raise AssertionError("gemini judge must not be called")

    monkeypatch.setattr(codex_judge, "judge_one", boom)
    monkeypatch.setattr(crag, "_judge_gemini", gemini_forbidden)
    with pytest.raises(codex_judge.CodexJudgeError):
        crag.judge(_PROSE_PRED, _PROSE_TRUTH)


def test_score_pairs_codex_batches_only_nondeterministic(monkeypatch):
    monkeypatch.setenv("JUDGE_BACKEND", "codex")
    seen = {}

    def fake_batch(pairs, sys_prompt, strict, n_votes=None):
        seen["pairs"] = pairs
        return ["Acceptable"] * len(pairs)

    monkeypatch.setattr(codex_judge, "judge_batch", fake_batch)
    score, results = crag.score_pairs([
        ("Recall", "Recall"),            # deterministic Perfect — must not reach codex
        (_PROSE_PRED, _PROSE_TRUTH),     # semantic — codex
    ])
    assert seen["pairs"] == [(_PROSE_PRED, _PROSE_TRUTH)]
    assert [r["judged"] for r in results] == ["Perfect", "Acceptable"]
    assert score == pytest.approx((1.0 + 0.5) / 2)
    assert all(r["backend"] == "codex" for r in results)


def test_gold_style_score_pairs_uses_three_vote_majority(monkeypatch):
    monkeypatch.setenv("JUDGE_BACKEND", "codex")
    seen = {}

    def fake_batch(pairs, sys_prompt, strict, n_votes=None):
        seen.update(pairs=pairs, strict=strict, n_votes=n_votes)
        return ["Acceptable"] * len(pairs)

    monkeypatch.setattr(codex_judge, "judge_batch", fake_batch)
    _score, results = crag.score_pairs(
        [(_PROSE_PRED, _PROSE_TRUTH)], votes=3, majority=True)
    assert seen == {"pairs": [(_PROSE_PRED, _PROSE_TRUTH)],
                    "strict": False, "n_votes": 3}
    assert results[0]["judged"] == "Acceptable"


def test_score_pairs_codex_error_propagates_no_gemini_fallback(monkeypatch):
    monkeypatch.setenv("JUDGE_BACKEND", "codex")

    def boom(*_a, **_k):
        raise codex_judge.CodexJudgeError("timeout")

    def gemini_forbidden(*_a, **_k):
        raise AssertionError("gemini judge must not be called")

    monkeypatch.setattr(codex_judge, "judge_batch", boom)
    monkeypatch.setattr(crag, "_judge_gemini", gemini_forbidden)
    with pytest.raises(codex_judge.CodexJudgeError):
        crag.score_pairs([(_PROSE_PRED, _PROSE_TRUTH)])
