"""SOT-2563 — per-call wall-clock deadline + cooperative cancel for the file_grep full scan.

The investigator's ``timeout_s`` is only checked between model turns, so a single synchronous full
``_extract`` scan could overrun the budget 1.5–3.6× (280–658s) and melt the whole question budget in
one call. These tests pin the runtime guard:

  * :mod:`src.rag.tools.call_budget` — env cap / propagated-budget → per-call deadline maths;
  * ``file_grep`` cooperatively cancels its scan at the deadline, returns early with
    ``evidence.deadline_hit`` + a route-switch ``note``, and is otherwise byte-identical (precision
    non-regression) when the deadline is not reached.
"""
from __future__ import annotations

import time
import unicodedata
from itertools import count
from pathlib import Path

import pytest

from src.rag.tools import call_budget
from src.rag.tools.file_grep import file_grep
from src.rag.tools.contract import is_contract


def _write(p: Path, text: str) -> None:
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(text, encoding="utf-8")


@pytest.fixture()
def mini_corpus(tmp_path: Path) -> Path:
    """A tiny NFD-normalized share-drive-shaped corpus with the query word in several files."""
    root = tmp_path / "share_drive"
    nfd = lambda s: unicodedata.normalize("NFD", s)  # noqa: E731
    _write(root / nfd("プロジェクト/かえで総合病院/01.契約") / "契約メモ.md",
           "契約条件の概要\n特記事項: 更新条項は自動更新とする\n")
    _write(root / nfd("プロジェクト/あおば/03.データ") / "notes.txt", "契約の控え\n")
    _write(root / "社内管理" / "メモ.txt", "契約の社内用語メモ\n")
    return root


# --------------------------------------------------------------------------- call_budget maths
def test_max_scan_seconds_default_and_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("RAG_FILE_GREP_MAX_SCAN_S", raising=False)
    assert call_budget.max_scan_seconds() == 180.0            # conservative default > max legit 125s
    monkeypatch.setenv("RAG_FILE_GREP_MAX_SCAN_S", "42")
    assert call_budget.max_scan_seconds() == 42.0
    monkeypatch.setenv("RAG_FILE_GREP_MAX_SCAN_S", "0")       # disabled ⇒ unbounded
    assert call_budget.max_scan_seconds() == 0.0
    monkeypatch.setenv("RAG_FILE_GREP_MAX_SCAN_S", "bogus")   # unparseable ⇒ fall back to default
    assert call_budget.max_scan_seconds() == 180.0


def test_scan_deadline_unbounded_when_disabled_and_no_budget(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("RAG_FILE_GREP_MAX_SCAN_S", "0")
    assert call_budget.current_remaining() is None
    assert call_budget.scan_deadline() is None                # cap off + no propagated budget


def test_scan_deadline_is_smaller_of_cap_and_remaining(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("RAG_FILE_GREP_MAX_SCAN_S", "180")
    now = 1000.0
    clk = lambda: now  # noqa: E731
    # No propagated budget ⇒ bounded by the cap only.
    assert call_budget.scan_deadline(clk) == pytest.approx(now + 180.0)
    # Remaining budget smaller than the cap ⇒ bounded by the remaining budget.
    with call_budget.remaining_budget(30.0):
        assert call_budget.current_remaining() == 30.0
        assert call_budget.scan_deadline(clk) == pytest.approx(now + 30.0)
    # Remaining budget larger than the cap ⇒ bounded by the cap.
    with call_budget.remaining_budget(600.0):
        assert call_budget.scan_deadline(clk) == pytest.approx(now + 180.0)
    # Exhausted budget ⇒ deadline is now (an immediate cut, never a full pass it has no time for).
    with call_budget.remaining_budget(0.0):
        assert call_budget.scan_deadline(clk) == pytest.approx(now)
    assert call_budget.current_remaining() is None            # context restored on exit


# --------------------------------------------------------------------------- file_grep cancellation
def test_file_grep_cut_immediately_when_deadline_already_passed(mini_corpus: Path) -> None:
    out = file_grep("契約", corpus_dir=mini_corpus, deadline_s=time.monotonic() - 1.0)
    assert is_contract(out)
    assert out["value"] == []                                  # nothing scanned ⇒ early "not found"
    ev = out["evidence"]
    assert ev["deadline_hit"] is True and ev["partial"] is True
    assert ev["files_scanned"] == 0
    assert "別ルート" in ev["note"]                            # steers the model to switch tools


def test_file_grep_cut_mid_scan_returns_partial(mini_corpus: Path) -> None:
    # A fake clock that advances 10s per read: the deadline (15s) trips before the 2nd file, so
    # exactly one file is scanned and the scan stops cooperatively rather than extracting them all.
    ticks = count(10, 10)
    clk = lambda: next(ticks)  # noqa: E731
    out = file_grep("契約", corpus_dir=mini_corpus, deadline_s=15.0, clock=clk)
    assert out["evidence"]["deadline_hit"] is True
    assert out["evidence"]["files_scanned"] == 1               # cut after the first file


def test_file_grep_generous_deadline_is_byte_identical(mini_corpus: Path) -> None:
    baseline = file_grep("契約", corpus_dir=mini_corpus)                 # no deadline
    guarded = file_grep("契約", corpus_dir=mini_corpus,
                        deadline_s=time.monotonic() + 10_000.0)          # never reached
    assert guarded["evidence"]["deadline_hit"] is False
    assert "partial" not in guarded["evidence"] and "note" not in guarded["evidence"]
    # Precision non-regression: an un-tripped deadline leaves the hits exactly as before.
    assert guarded["value"] == baseline["value"]
    assert len({h["file"] for h in baseline["value"]}) == 3             # all three files matched


def test_file_grep_honours_propagated_budget(mini_corpus: Path) -> None:
    # An exhausted propagated budget (as the investigator publishes when timeout_s is spent) cuts the
    # scan even without an explicit deadline_s, so a single call cannot melt an already-spent budget.
    with call_budget.remaining_budget(0.0):
        out = file_grep("契約", corpus_dir=mini_corpus)
    assert out["evidence"]["deadline_hit"] is True
    assert out["evidence"]["files_scanned"] == 0
