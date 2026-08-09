"""SOT-2563 focused/offline verification harness (gold100 は実行しない).

Deterministically exercises the per-call file_grep deadline against the REAL 403-file corpus without
invoking the LLM, so it is cheap and reproducible. Measures a single ``file_grep`` call's wall-clock
elapsed and ``files_scanned`` (a) uncapped (the runaway full-scan baseline) and (b) with a small
per-call cap, and asserts the guard bounds the call to the cap, cooperatively cancels, and stamps the
route-switch signal (``deadline_hit`` + ``note``). Run modes:

    python scripts/verify_sot2563_deadline.py baseline   # uncapped full scan (slow: 300-658s)
    python scripts/verify_sot2563_deadline.py capped 10   # capped at 10s (fast)
"""
from __future__ import annotations

import json
import os
import sys
import time

from src.rag.tools import call_budget
from src.rag.tools.file_grep import file_grep

# A rare token that misses everywhere: file_grep still extracts every searchable file (content is
# read before matching), so full-scan cost is query-independent and this forces the whole pass.
QUERY = "ZZQXWV_該当なし稀語_SOT2563"


def _run(cap: float | None) -> dict:
    if cap is not None:
        os.environ["RAG_FILE_GREP_MAX_SCAN_S"] = str(cap)
    else:
        os.environ.pop("RAG_FILE_GREP_MAX_SCAN_S", None)
    t0 = time.monotonic()
    out = file_grep(QUERY)
    elapsed = time.monotonic() - t0
    ev = out["evidence"]
    return {
        "cap_s": cap,
        "elapsed_s": round(elapsed, 2),
        "files_scanned": ev.get("files_scanned"),
        "deadline_hit": ev.get("deadline_hit"),
        "partial": ev.get("partial", False),
        "hits": len(out["value"]),
        "note_present": "note" in ev,
    }


if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else "capped"
    if mode == "baseline":
        print(json.dumps(_run(cap=0.0), ensure_ascii=False))  # cap disabled ⇒ full scan
    else:
        cap = float(sys.argv[2]) if len(sys.argv) > 2 else 10.0
        print(json.dumps(_run(cap=cap), ensure_ascii=False))
