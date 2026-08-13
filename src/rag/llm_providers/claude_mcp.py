"""claude-mcp investigator backend (SOT-2627) — run one question's whole tool loop on flat-rate Sonnet.

Where :mod:`src.rag.llm_providers.claude_cli` (SOT-2606) only covers *tool-free text* roles, this module
delegates the investigator's **entire function-calling loop** — tool round-trips and the final answer —
to a flat-rate ``claude -p --mcp-config`` session driving the SOT-2626 stdio MCP server
(:mod:`src.rag.mcp.server`). The 2026-08-10 smoke proved ``LLM_PROVIDER=claude-cli`` alone runs the
gold100 loop with **zero** Sonnet calls (the loop is genai-fixed at ``investigator.py:2316``); the only
way to move that loop off Gemini metering is to let ``claude`` itself drive the tools over MCP.

**Dev-only.** Selected by ``RAG_INVESTIGATOR_BACKEND=claude-mcp`` (default ``gemini``). The production
answer path stays Gemini-only (SOT-2460); official measurement stays on flash 3.6 (SOT-2625). The
flat-rate usage limit is **shared account-wide with the autonomous workers**, so this throttles them —
run it only when workers are idle, at parallelism 1 (the investigator run default for this backend).

What it produces
----------------
A :class:`~src.rag.agent.investigator.Investigation` byte-compatible with the details.jsonl schema
(``index`` is added by :mod:`src.rag.run`), so ``scoring.gold_offline --answers/--run`` scores it exactly
like the Gemini backend. ``model`` is reported as ``"sonnet(claude-mcp)"`` and ``cost_usd`` is 0 (flat
rate — see :meth:`Usage.cost_usd`, which zeroes any ``(claude-mcp)`` model).

Budget & interruption
---------------------
``claude`` exposes no ``--max-turns`` (checked, CLI 2.1.x), so the per-question budget is enforced two
ways: a subprocess wall-clock ``timeout`` (= ``timeout_s``) and a server-side tool-call cap
(``RAG_MCP_MAX_TOOL_CALLS`` = ``max_turns``, passed through the generated mcp-config). On a detected
flat-rate **usage-limit** the run trips a process-wide latch so every remaining question short-circuits
to an abstain immediately (no further ``claude`` calls hammer the shared limit); answered questions are
persisted to a resume sidecar so a later re-run continues where it stopped.
"""
from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
import tempfile
import threading
import time
from pathlib import Path
from typing import Any, Mapping, Sequence

from src.rag.agent.investigator import (
    ABSTAIN,
    SUBMIT_ANSWER,
    SYSTEM_PROMPT,
    AgentTool,
    Answer,
    Investigation,
    Usage,
    _answer_from_args,
    fanout_finisher_budget,
    fanout_finisher_enabled,
    is_abstain,
    plan_fanout_budget,
    plan_fanout_enabled,
)

CLAUDE_BIN = "claude"
MCP_SERVER_NAME = "investigator"          # the --mcp-config key ⇒ tools surface as mcp__investigator__*
DEFAULT_MODEL = "sonnet"                   # flat-rate Sonnet; RAG_CLAUDE_MCP_MODEL overrides
_MODEL_TAG = "claude-mcp"                  # model label suffix ⇒ Usage.cost_usd zeroes the flat-rate cost

# Built-in claude tools we never want the investigator loop to use — the loop must go through the MCP
# corpus tools only (parity with the Gemini function-calling surface). Disallowed tools are auto-denied
# in non-interactive ``-p`` mode, so claude routes to the MCP tools instead.
_DISALLOWED_BUILTINS = (
    "Bash", "Read", "Edit", "Write", "NotebookEdit", "WebFetch", "WebSearch",
    "Glob", "Grep", "Task", "TodoWrite",
)

# --- process-wide usage-limit latch (SOT-2627 §4) -------------------------------------------------
# Once a flat-rate usage limit is seen, every remaining question in the batch short-circuits instead of
# issuing more ``claude`` calls that would only burn the shared account limit faster.
_USAGE_LIMIT = threading.Event()
_RESUME_LOCK = threading.Lock()


class UsageLimitError(RuntimeError):
    """Raised internally when the flat-rate Claude usage limit is detected (see :func:`_is_usage_limit`)."""


def usage_limit_tripped() -> bool:
    """Whether a flat-rate usage limit has been detected in this process (test/inspection hook)."""
    return _USAGE_LIMIT.is_set()


def reset_usage_limit() -> None:
    """Clear the process-wide usage-limit latch (used by tests / a deliberate fresh run)."""
    _USAGE_LIMIT.clear()


# --- resume sidecar (SOT-2627 §4) -----------------------------------------------------------------
def _resume_path() -> Path | None:
    """Resolve the resume-sidecar path, or ``None`` when resume is disabled.

    ``RAG_CLAUDE_MCP_RESUME`` unset ⇒ default ``<ARTIFACTS_DIR>/claude_mcp_resume.jsonl``; an explicit
    path redirects it; ``0``/``off``/``false``/``no``/empty disables the sidecar entirely.
    """
    raw = os.getenv("RAG_CLAUDE_MCP_RESUME")
    if raw is not None:
        low = raw.strip().lower()
        if low in {"", "0", "off", "false", "no"}:
            return None
        return Path(raw)
    from config import settings
    return settings.ARTIFACTS_DIR / "claude_mcp_resume.jsonl"


def _resume_key(question: str, model: str) -> str:
    return hashlib.sha256(f"{model}\x00{question}".encode("utf-8")).hexdigest()


def _load_resume(path: Path) -> dict[str, dict]:
    out: dict[str, dict] = {}
    if not path.exists():
        return out
    try:
        for line in path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            key = rec.get("key")
            if key:
                out[key] = rec
    except Exception:  # noqa: BLE001 — a corrupt sidecar must never break a run; just ignore it
        return out
    return out


def _append_resume(path: Path, key: str, question: str, inv: Investigation) -> None:
    rec = {"key": key, "question": question, "record": _investigation_to_record(inv)}
    try:
        with _RESUME_LOCK:
            path.parent.mkdir(parents=True, exist_ok=True)
            with path.open("a", encoding="utf-8") as fh:
                fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
    except Exception:  # noqa: BLE001 — persistence is best-effort; a write failure only costs resumability
        pass


def _investigation_to_record(inv: Investigation) -> dict[str, Any]:
    """Compact, self-contained serialization sufficient to rebuild the :class:`Investigation`."""
    return {
        "question": inv.question,
        "answer": inv.answer.answer,
        "confidence": inv.answer.confidence,
        "evidence": inv.answer.evidence,
        "method": inv.answer.method,
        "iterations": inv.iterations,
        "tool_calls": list(inv.tool_calls),
        "input_tokens": inv.usage.input_tokens,
        "output_tokens": inv.usage.output_tokens,
        "model": inv.model,
        "elapsed_s": inv.elapsed_s,
        "stop_reason": inv.stop_reason,
        "error": inv.error,
        "contract": inv.contract,
    }


def _record_to_investigation(rec: Mapping[str, Any]) -> Investigation:
    return Investigation(
        question=str(rec.get("question", "")),
        answer=Answer(
            answer=str(rec.get("answer", ABSTAIN) or ABSTAIN),
            confidence=float(rec.get("confidence", 0.0) or 0.0),
            evidence=str(rec.get("evidence", "") or ""),
            method=str(rec.get("method", "") or ""),
        ),
        iterations=int(rec.get("iterations", 0) or 0),
        tool_calls=list(rec.get("tool_calls", []) or []),
        usage=Usage(int(rec.get("input_tokens", 0) or 0), int(rec.get("output_tokens", 0) or 0)),
        model=str(rec.get("model", f"{DEFAULT_MODEL}({_MODEL_TAG})")),
        elapsed_s=float(rec.get("elapsed_s", 0.0) or 0.0),
        stop_reason=str(rec.get("stop_reason", "answered")),
        error=rec.get("error"),
        contract=rec.get("contract"),
    )


# --- claude invocation ----------------------------------------------------------------------------
def _repo_root() -> Path:
    # src/rag/llm_providers/claude_mcp.py → repo root is three parents up from the package dir.
    return Path(__file__).resolve().parents[3]


def _mcp_config(tool_log: str, max_tool_calls: int,
                question: str = "", contract: str | None = None,
                commit_gate_log: str | None = None) -> dict[str, Any]:
    """Build a --mcp-config dict launching the SOT-2626 stdio server with per-question budget/log env.

    SOT-2640: the question/contract are forwarded so the server-side commit gate (submit_answer execution
    point) has the commit context. They are inert unless RAG_COMMIT_GATE is set in the ambient env, which
    the launched server process inherits (the CLI merges this ``env`` over the parent environment)."""
    launcher = str(_repo_root() / "scripts" / "mcp_investigator_server.sh")
    env = {"RAG_MCP_TOOL_LOG": tool_log}
    if max_tool_calls and max_tool_calls > 0:
        env["RAG_MCP_MAX_TOOL_CALLS"] = str(int(max_tool_calls))
    # SOT-2664 — carry the bounded finisher config alongside the per-question budget so the launched
    # server enables it deterministically (does not rely on ambient-env inheritance). OFF ⇒ keys omitted
    # ⇒ byte-identical config.
    _fin = fanout_finisher_budget()
    if _fin > 0:
        env["RAG_FANOUT_FINISHER"] = "1"
        env["RAG_FANOUT_FINISHER_MAX"] = str(int(_fin))
    if question:
        env["RAG_MCP_QUESTION"] = question
    if contract:
        env["RAG_MCP_CONTRACT"] = str(contract)
    if commit_gate_log:
        env["RAG_MCP_COMMIT_GATE_LOG"] = commit_gate_log
    return {"mcpServers": {MCP_SERVER_NAME: {"command": "bash", "args": [launcher], "env": env}}}


def _allowed_tools(tools: Sequence[AgentTool]) -> list[str]:
    return [f"mcp__{MCP_SERVER_NAME}__{t.name}" for t in tools]


def _bare_answer_enabled() -> bool:
    """裸回答書式契約 (cycle4 調査: wrong 21 のうち 12 件が「値は正しいが説明文/注記/記法で judge 落ち」)。

    プロンプトのみの契約 — 値の後処理は一切行わない (値保存)。既定 OFF。
    """
    return os.getenv("RAG_BARE_ANSWER", "0").strip().lower() in ("1", "true", "yes", "on")


_BARE_ANSWER_CONTRACT = (
    "\n\n【回答書式契約】answer フィールドには質問が要求する値・語句そのものだけを書く(裸の値):"
    "理由・計算過程・出典・セル位置・補足説明・括弧書きの注記は answer に一切含めず、evidence フィールドへ書く。"
    "例:『差額は0円(…のため)』ではなく『0円』/『13ページ(スライド13)「8. 費用見積」』ではなく『13ページ』。"
    "数値は原文・計算結果の精度をそのまま書き、勝手に丸めたり『約』を付けたりしない。"
    "範囲は『A ~ B』の形式で書く(区間記法 (a, b] 等は使わない)。"
    "一覧を要求されたら項目のみを読点区切りで列挙し、各項目に説明を付けない。"
)


def _none_bare_enabled() -> bool:
    """SOT-2665 (cycle5 C2) — 「該当なし」裸形式契約。

    cycle5 の wrong のうち idx9/85 は gold=「該当なし」なのに、変更点を列挙(idx9)したり
    「なし(全6項目達成)」と達成状況を注記(idx85)して judge 落ちした。存在しないことが確定した回答は
    装飾なしの「該当なし」のみに縛る。プロンプトのみ(値の後処理なし)・既定 OFF。
    """
    return os.getenv("RAG_NONE_BARE", "0").strip().lower() in ("1", "true", "yes", "on")


_NONE_BARE_CONTRACT = (
    "\n\n【該当なし契約】質問が要求する項目・変更・値が確定的に存在しない/見つからない場合、"
    "answer は装飾なしで『該当なし』のみとする(『なし』『特になし』ではなく『該当なし』)。"
    "『なし(全6項目達成)』のような理由・達成状況・件数の注記や、"
    "存在した別項目の列挙・言い換えを answer に一切付けない(補足は evidence へ書く)。"
)


def _two_tier_answer_enabled() -> bool:
    """SOT-2670 — two-tier submit_answer schema (full_answer + bare_answer). Prompt-side mirror of the
    structural flag defined in ``investigator`` so the tool description AND the system prompt both steer
    the model toward bare_answer as the scored value. Default OFF."""
    return os.getenv("RAG_TWO_TIER_ANSWER", "0").strip().lower() in ("1", "true", "yes", "on")


_TWO_TIER_ANSWER_CONTRACT = (
    "\n\n【二段回答契約】submit_answer には full_answer と bare_answer を必ず両方渡す。"
    "full_answer には根拠・限定条件を含む完全な文を書いて『考える場所』とし、bare_answer には"
    "質問が要求する値・語句そのものだけを書く(採点対象は bare_answer)。bare_answer では"
    "ラベル・肩書・単位・記法を原文どおり完全に写し、切り詰め・言い換え・注記の付加をしない。"
    "存在しないことが確定した場合の bare_answer は『該当なし』のみとする。"
)


def _exact_label_enabled() -> bool:
    """SOT-2666 (cycle5 C3) — 完全ラベル写経・回答範囲限定契約。

    cycle5 の wrong のうち idx21/62/78 は値そのものは正しいのに、肩書を切り詰めたり(idx21=「部長」/
    gold「人材戦略部長」)、複数値をラベルなしで並べたり(idx62=「500 vs 300」/ gold「1位=500、2位=300」)、
    問われていない追加主張を付けたり(idx78=「上限は設けられていない」)して judge 落ちした。値保存の裸回答
    契約(RAG_BARE_ANSWER)を補完し、substance を成す語(肩書・順位ラベル・対応関係)は削らず、逆に
    問われた範囲を越える主張は足さないよう縛る。プロンプトのみ(値の後処理なし)・既定 OFF。
    """
    return os.getenv("RAG_EXACT_LABEL", "0").strip().lower() in ("1", "true", "yes", "on")


_EXACT_LABEL_CONTRACT = (
    "\n\n【完全ラベル契約】answer に載せる値は、それを構成する肩書・役職・部署・順位ラベル・単位を"
    "原文どおり完全に写経する — 一部だけを切り詰めない(『人材戦略部長』を『部長』にしない)。"
    "質問が複数の値を区別して問う場合(順位別・年度別・モデル別など)は、どの値がどれに対応するかが"
    "分かるラベル付き形式で全て併記する(『500 vs 300』ではなく『1位=500、2位=300』)。この対応ラベルは"
    "値の一部であり注記ではないので省かない。同時に、問われた範囲のみを答え、問われていない事実・"
    "上限有無・背景・評価といった追加の主張を answer に足さない(補足は evidence へ書く)。"
)


def _vdiff_normalize_enabled() -> bool:
    """SOT-2681 (cycle6 K5) — 版差分の意味正規化契約(prompt-only)。

    cycle5 の version_diff wrong (idx1/95) は値・セルは正しく特定できているのに、変更の *型* を取り違えて
    judge 落ちしていた: リスト追記(担当者「渡辺 遥」→「渡辺 遥 / 小林 直樹」)を「変更」と書く(idx95, gold
    「小林 直樹を追加」)、同域の削除＋挿入を「削除」とだけ書いて置換側を落とす(idx1, gold「…を削除し1行要約
    に置換」)。加えて離れた版(v1→v3)を中間版の逐次読み直しで予算切れ棄権(idx14)。値後処理はせず、Sonnet
    backend の system suffix に型正規化の言い回し契約を足すだけ(RAG_EXACT_LABEL と同型)・既定 OFF。
    """
    return os.getenv("RAG_VDIFF_NORMALIZE", "0").strip().lower() in ("1", "true", "yes", "on")


_VDIFF_NORMALIZE_CONTRACT = (
    "\n\n【版差分の意味正規化契約】旧版→新版の実質的変更を述べるときは、変更の型を意味に沿って正規化する。"
    "(1)リスト追記: 旧値が新値の部分集合で、新値＝旧値＋追加項目になっている場合(例: 担当者『渡辺 遥』→"
    "『渡辺 遥 / 小林 直樹』)は『変更』と書かず、追加された項目だけを主語にして『(対象/行)に<追加項目>を追加』"
    "と表現する(旧項目は温存されているので消えていない)。"
    "(2)同域の置換: 同一の箇所・表・節で、あるブロックの削除と別ブロックの挿入が対になっている場合(例: "
    "性能比較表を削除して1行要約を挿入)は、削除だけで止めず『<削除対象>を削除し、<挿入内容>に置換』と"
    "両側を1文で述べる。"
    "(3)離れた版: 旧版と新版が隣接しない(例 v1→v3)場合も、事前確定済みの版ペア差分(diff_lookup)を優先して"
    "1回で取得し、中間版を逐次読み直して予算を使い切らない。"
    "いずれも値・ラベルは原文どおり写経し、問われた範囲を越える主張は足さない(補足は evidence へ)。"
)


def _special_provision_enabled() -> bool:
    """SOT-2688 (cycle7 K5, idx78) — 「特別規定なし＋一般規定内容」合成回答契約 (prompt-only)。

    cycle6 の simple_lookup wrong idx78 は evidence に「ACTH/200時間超の特別な精算規定なし＋適用される
    一般規定 6.1〜6.3 条(25,000円/時・税別・月次精算…)」を完全に取得済みなのに answer を「該当なし」だけで
    終え、gold「特別規定なし＋一般規定の内容」と judge 不一致になった。『規定内容を答えよ』型で特定条件の
    専用規定が不在かつ一般規定を取得済みのとき、不在＋一般規定内容を1回答に合成させる。値後処理はせず、
    Sonnet backend の system suffix に契約を足すだけ(RAG_NONE_BARE と同型)・既定 OFF。RAG_NONE_BARE(該当なし
    裸形式)とは対象が異なる(あちらは要求物そのものが不在の問い)ため衝突しない。
    """
    return os.getenv("RAG_SPECIAL_PROVISION", "0").strip().lower() in ("1", "true", "yes", "on")


_SPECIAL_PROVISION_CONTRACT = (
    "\n\n【特別規定不在＋一般規定合成契約】ある特定条件(特定の項目・数値しきい値を超えた場合など)についての"
    "『規定内容を答えよ』型の問いで、(1)その条件に対する専用(特別)規定が確定的に存在せず、かつ(2)その場合に"
    "適用される一般規定・原則を証拠として取得済みのときは、answer を単なる『該当なし』で終えない。"
    "『(当該条件に対する)特別な規定はなく、適用される一般規定は〈一般規定の要点(数値・単位・条件は原文どおり)〉』"
    "の形で、特別規定の不在と一般規定の内容を1つの回答に合成する。一般規定の内容は問われた事項(精算方法など)"
    "に絞って自然な日本語で簡潔に述べ、生のフィールド名・内部ID(例 time_and_materials)や証拠の逐語コピー・"
    "冗長な言い換えは書かない(要点の条件だけを1〜2文で)。一般規定を取得できていない場合に限り従来どおり"
    "『該当なし』/棄権とする(内容を捏造しない)。要求物そのものが存在しない問い(該当なし契約の対象)は"
    "この契約の対象外。"
)


def _vdiff_classify_enabled() -> bool:
    """SOT-2695 (cycle8 C6) — 版差分の実質変更分類契約 (prompt-only, 既定 OFF)。

    diff_store 側の分類補正(`RAG_VDIFF_CLASSIFY`, diffpair)と対で使う言い回し契約。cycle7 wrong:
    idx9 は見出しラベル追加・スライド分割(体裁再編)を実質変更として挙げた(gold「該当なし」)、idx14 は
    データ列名のアンダースコア表記化(loan_status 等)を surface に落とし STEP 再編を実質変更として答えた
    (gold は列名変更のみ)。値後処理はせず、SUBSTANTIVE が無い体裁差は「該当なし」、列名/スキーマ語彙の
    変更は最優先の実質変更、と Sonnet backend に縛る。既定 OFF ⇒ 影響なし。"""
    return os.getenv("RAG_VDIFF_CLASSIFY", "0").strip().lower() in ("1", "true", "yes", "on")


_VDIFF_CLASSIFY_CONTRACT = (
    "\n\n【版差分の実質変更分類契約】旧版→新版で『案件遂行に関連する実質的変更』を問われたときは、"
    "事前確定済みの版ペア差分(diff_lookup)の intent を基準に分類する。"
    "(1)実質変更なしの即断: diff_lookup の evidence が substantive_change=false(または verdict_hint に"
    "『該当なし』)を返したら、変更が見出しラベル追加・スライド分割・体裁再構成だけで実質的変更が無い"
    "ということなので、それ以上ファイルを探索せず直ちに answer=『該当なし』で submit する。見出しへの"
    "ラベル付加(『5. 業務提言』→『5. 業務提言 ― クイックウィン（短期）』)やフェーズ別スライドへの分割は"
    "実質変更として挙げない。"
    "(2)列名・スキーマ語彙の変更を最優先: データ列名やスキーマ語彙の変更(スペース→アンダースコア表記化、"
    "identifier 特徴の SUBSTANTIVE で intent が最上位)が挙がっている場合は、それが案件遂行に関わる実質"
    "変更であり、STEP・節の統合/再構成(LAYOUT)より優先して回答の主題にする。回答は『データ列名を"
    "アンダースコア表記に修正：<変更後の列名(アンダースコア形)を全て読点区切りで列挙>』の形にし、変更後の"
    "列名を主語にする(『A → B』の羅列や STEP 再構成の説明を主にしない)。列名は diff_lookup の値を原文"
    "どおり写経し、問われた範囲を越える主張は足さない(補足は evidence へ)。"
)


def _report_series_enabled() -> bool:
    """SOT-2695 (cycle8 C6, idx16) — 報告資料の系列スコープ＋最小集合契約 (prompt-only, 既定 OFF)。

    cycle7 wrong idx16「中間報告資料の黄×赤」は、同じ 05.会議/報告資料 フォルダにある M01 キックオフ
    (2025-04-09, 4,620,000) と M02 中間レビュー(2025-04-29, 0.589) の両方を列挙した(gold は中間＝M02 の
    0.589 のみ)。各報告資料は自己申告のチェックポイント名(M01=キックオフ／M02=中間レビュー／『中間分析
    報告』『最終報告』)を本文冒頭に持つので、質問が指す系列を質問非依存に判定して系列外文書を除ける。
    値後処理はせず、Sonnet backend に系列スコープと最小集合を縛るだけ・既定 OFF ⇒ 影響なし。"""
    return os.getenv("RAG_REPORT_SERIES", "0").strip().lower() in ("1", "true", "yes", "on")


_REPORT_SERIES_CONTRACT = (
    "\n\n【報告資料の系列スコープ契約】『中間報告』『最終報告』『キックオフ』『定例』など特定の報告系列を"
    "指定して値を抽出する問いでは、対象を系列で先に絞る。各報告資料は本文冒頭に自己申告のチェックポイント/"
    "表題(例『チェックポイント: M01（キックオフ・前提確認会）』『対象チェックポイント: M02（中間レビュー会）』"
    "『…中間分析報告です』『最終報告』)を持つので、これを質問非依存の系列ラベルとして読む: キックオフ/M01 は"
    "中間ではない、中間レビュー/中間(分析)報告/M02 は中間、最終報告は最終。質問が『中間報告資料』を指すなら"
    "中間に該当する文書のみから抽出し、キックオフや定例など系列外の文書の値は含めない。該当文書が一意に"
    "定まるときは、その文書から抽出した最小集合だけを答える(別系列の文書や余分な値を足さない)。系列判定の"
    "根拠が本文から得られないときのみ従来どおり全該当を挙げる(値の捏造・gold への当て推量はしない)。"
)


def _harness_system_suffix() -> str:
    base = (
        "\n\n【実行環境(dev)】ツールは MCP 経由で `mcp__investigator__<ツール名>` として提供される"
        "(例: `mcp__investigator__file_grep`, `mcp__investigator__compute`)。これら以外のツールは使わない。"
        f"最終回答は必ず `mcp__investigator__{SUBMIT_ANSWER}` を1回だけ呼んで確定する"
        "(通常の文章では答えない)。根拠が得られない場合のみ answer=「" + ABSTAIN + "」で submit する。"
    )
    if _bare_answer_enabled():
        base += _BARE_ANSWER_CONTRACT
    if _none_bare_enabled():
        base += _NONE_BARE_CONTRACT
    if _two_tier_answer_enabled():
        base += _TWO_TIER_ANSWER_CONTRACT
    if _exact_label_enabled():
        base += _EXACT_LABEL_CONTRACT
    if _vdiff_normalize_enabled():
        base += _VDIFF_NORMALIZE_CONTRACT
    if _special_provision_enabled():
        base += _SPECIAL_PROVISION_CONTRACT
    if _vdiff_classify_enabled():
        base += _VDIFF_CLASSIFY_CONTRACT
    if _report_series_enabled():
        base += _REPORT_SERIES_CONTRACT
    return base


def _is_usage_limit(text: str) -> bool:
    """Heuristic detection of a flat-rate Claude usage/rate limit from CLI output."""
    t = (text or "").lower()
    needles = (
        "usage limit", "rate limit", "rate_limit", "limit reached", "too many requests",
        "429", "resets at", "quota", "overloaded_error", "usage_limit",
    )
    return any(n in t for n in needles)


def _run_claude(prompt: str, *, system: str, model: str, cfg_path: str,
                allowed: Sequence[str], timeout: float) -> tuple[str, str, int, bool]:
    """Run one ``claude -p`` session; return (stdout, stderr, returncode, timed_out)."""
    cmd = [
        CLAUDE_BIN, "-p",
        "--model", model,
        "--mcp-config", cfg_path,
        "--strict-mcp-config",
        "--output-format", "stream-json",
        "--verbose",
        "--append-system-prompt", system,
        "--allowedTools", *allowed,
        "--disallowedTools", *_DISALLOWED_BUILTINS,
    ]
    try:
        proc = subprocess.run(
            cmd, input=prompt, capture_output=True, text=True, timeout=timeout,
        )
        return proc.stdout or "", proc.stderr or "", proc.returncode, False
    except subprocess.TimeoutExpired as exc:
        out = exc.stdout or ""
        err = exc.stderr or ""
        if isinstance(out, bytes):
            out = out.decode("utf-8", "replace")
        if isinstance(err, bytes):
            err = err.decode("utf-8", "replace")
        return out, err, -1, True


def _parse_stream_json(stdout: str) -> dict[str, Any]:
    """Parse claude ``stream-json`` JSONL into the fields we need to rebuild an Investigation.

    Returns ``tool_calls`` (bare tool names, in order), ``submit_args`` (the last submit_answer input, or
    None), ``final_text`` (final assistant text), ``usage`` (input/output tokens), ``is_error``,
    ``subtype``, and ``result_text`` (the terminal result message text, if any).
    """
    tool_calls: list[str] = []
    submit_args: dict[str, Any] | None = None
    final_text = ""
    input_tokens = 0
    output_tokens = 0
    is_error = False
    subtype = ""
    result_text = ""
    prefix = f"mcp__{MCP_SERVER_NAME}__"
    for line in stdout.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            evt = json.loads(line)
        except json.JSONDecodeError:
            continue
        etype = evt.get("type")
        if etype == "assistant":
            msg = evt.get("message") or {}
            for block in msg.get("content") or []:
                if not isinstance(block, Mapping):
                    continue
                if block.get("type") == "tool_use":
                    name = str(block.get("name") or "")
                    # Record ONLY investigator MCP tools (the mcp__investigator__* surface). Claude's own
                    # built-ins (e.g. ToolSearch used for schema discovery) are excluded so tool_calls /
                    # iterations reflect the same investigator-tool loop the Gemini trace does — a fair
                    # reachability comparison, not a claude-harness artifact.
                    if not name.startswith(prefix):
                        continue
                    bare = name[len(prefix):]
                    tool_calls.append(bare)
                    if bare == SUBMIT_ANSWER:
                        inp = block.get("input")
                        submit_args = dict(inp) if isinstance(inp, Mapping) else {}
                elif block.get("type") == "text":
                    final_text = str(block.get("text") or "") or final_text
            usage = msg.get("usage") or {}
            input_tokens += int(usage.get("input_tokens", 0) or 0)
            output_tokens += int(usage.get("output_tokens", 0) or 0)
        elif etype == "result":
            is_error = bool(evt.get("is_error"))
            subtype = str(evt.get("subtype") or "")
            rt = evt.get("result")
            if isinstance(rt, str):
                result_text = rt
            usage = evt.get("usage") or {}
            # The result message carries cumulative usage; prefer it when present (avoids double count).
            if usage:
                input_tokens = int(usage.get("input_tokens", 0) or 0)
                output_tokens = int(usage.get("output_tokens", 0) or 0)
    return {
        "tool_calls": tool_calls,
        "submit_args": submit_args,
        "final_text": (result_text or final_text).strip(),
        "usage": Usage(input_tokens, max(0, output_tokens)),
        "is_error": is_error,
        "subtype": subtype,
    }


def _abstain_investigation(question: str, *, model: str, elapsed_s: float, stop_reason: str,
                           error: str | None, contract: str | None,
                           tool_calls: Sequence[str] = (), iterations: int = 0,
                           usage: Usage | None = None) -> Investigation:
    return Investigation(
        question=question,
        answer=Answer(answer=ABSTAIN, confidence=0.0, evidence="", method=f"claude-mcp: {stop_reason}"),
        iterations=iterations,
        tool_calls=list(tool_calls),
        usage=usage or Usage(),
        model=model,
        elapsed_s=elapsed_s,
        stop_reason=stop_reason,
        error=error,
        contract=contract,
    )


def _load_commit_gate_log(path: str) -> list[dict[str, Any]]:
    """Read the server-side decision stream. A missing/truncated telemetry file is non-fatal."""
    out: list[dict[str, Any]] = []
    try:
        with open(path, encoding="utf-8") as fh:
            for line in fh:
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError:
                    continue
                decision = rec.get("commit_gate") if isinstance(rec, Mapping) else None
                if isinstance(decision, Mapping):
                    out.append(dict(decision))
    except OSError:
        pass
    return out


def _plan_fanout_telemetry(tool_calls: Sequence[str], budget: int) -> dict[str, Any]:
    """SOT-2661 — per-stage telemetry for the planner→fan-out→synthesis flow (SOT-2629 form).

    Derives the stage accounting from the executed tool-call sequence:
    * ①計画/②収集 — ``search_first`` / ``search_calls`` (the unified fan-out call(s));
    * ③合成 — ``supplement_calls`` (the ≤2 additional narrowing probes after the first search).
    ``tool_turns`` is the total investigator-tool turns (submit_answer excluded), i.e. the段数 the
    3-stage flow is meant to hold at ≤ ``budget``.
    """
    from src.rag.tools import unified_search as _unified_search
    search_name = _unified_search.TOOL_NAME
    non_submit = [t for t in tool_calls if t != SUBMIT_ANSWER]
    search_calls = sum(1 for t in non_submit if t == search_name)
    first = non_submit[0] if non_submit else None
    # supplement = narrowing tool calls that are NOT the unified search (stage ③); when search never
    # fired, every non-submit call counts as a supplement probe (the flow degraded to individual tools).
    supplement = (len(non_submit) - search_calls) if search_calls else len(non_submit)
    # SOT-2664 — non-terminal tool USES the model emitted beyond the plan-fanout budget. This counts both
    # the reads the bounded finisher actually GRANTED (≤ finisher_max, targeted+resolved only) AND any
    # over-budget calls the server REFUSED with budget_exhausted (they still appear as tool_use blocks in
    # the stream), so it is an over-budget *attempt* count, not the granted count — the true granted count
    # is server-side (≤ finisher_max). When the finisher is OFF the server refuses every over-budget call,
    # but the model may still emit (and get refused on) such attempts, so read this together with
    # ``finisher_enabled``.
    finisher_max = fanout_finisher_budget()
    over_budget_calls = max(0, len(non_submit) - int(budget)) if budget and budget > 0 else 0
    return {
        "enabled": True,
        "budget": int(budget),
        "first_tool": first,
        "search_first": bool(first == search_name),
        "search_calls": int(search_calls),
        "supplement_calls": int(max(0, supplement)),
        "extra_searches": int(max(0, search_calls - 1)),
        "tool_turns": int(len(non_submit)),
        "finisher_enabled": bool(finisher_max > 0),
        "finisher_max": int(finisher_max),
        "over_budget_calls": int(over_budget_calls),
    }


def investigate_question(question: str, *, tools: Sequence[AgentTool],
                         system: str | None = None, contract: str | None = None,
                         preamble: str | None = None,
                         max_turns: int = 12, timeout_s: float = 180.0,
                         model: str | None = None) -> Investigation:
    """Run one question's whole tool loop on flat-rate Sonnet via claude CLI + MCP.

    Mirrors :func:`src.rag.agent.investigator.investigate`'s contract: always returns an
    :class:`Investigation` (never raises), abstaining with a coded ``stop_reason`` on any failure so a
    batch keeps going and the failure is visible in the details log.
    """
    mdl = model or os.getenv("RAG_CLAUDE_MCP_MODEL") or DEFAULT_MODEL
    model_label = f"{mdl}({_MODEL_TAG})"
    resume = _resume_path()

    # Resume/short-circuit gates (checked before any subprocess spend).
    key = _resume_key(question, model_label)
    if resume is not None:
        cached = _load_resume(resume).get(key)
        if cached is not None:
            return _record_to_investigation(cached.get("record") or {})
    if _USAGE_LIMIT.is_set():
        return _abstain_investigation(
            question, model=model_label, elapsed_s=0.0, stop_reason="usage_limit",
            error="flat-rate usage limit tripped earlier this run; skipped", contract=contract)

    if shutil.which(CLAUDE_BIN) is None:
        return _abstain_investigation(
            question, model=model_label, elapsed_s=0.0, stop_reason="model_error",
            error=f"{CLAUDE_BIN!r} not on PATH (claude-mcp backend requires the Claude Code CLI)",
            contract=contract)

    sys_prompt = (system or SYSTEM_PROMPT) + _harness_system_suffix()
    user_prompt = f"{preamble}\n\n---\n\n{question}" if preamble else question
    allowed = _allowed_tools(tools)
    # SOT-2661 — redefine the per-question budget to the 3-stage shape when RAG_PLAN_FANOUT is on
    # (≤5 non-terminal tool calls + the uncapped submit ⇒ ≤6 total). No-op (== max_turns) when the flag is off,
    # so the server-side RAG_MCP_MAX_TOOL_CALLS cap stays byte-identical.
    eff_max_turns = plan_fanout_budget(max_turns)

    started = time.monotonic()
    tmp = tempfile.NamedTemporaryFile("w", suffix=".mcp.json", delete=False, encoding="utf-8")
    log_fd, log_path = tempfile.mkstemp(suffix=".mcp_tool_calls.jsonl")
    os.close(log_fd)
    gate_fd, gate_log_path = tempfile.mkstemp(suffix=".mcp_commit_gate.jsonl")
    os.close(gate_fd)
    try:
        json.dump(_mcp_config(log_path, eff_max_turns, question=question, contract=contract,
                              commit_gate_log=gate_log_path),
                  tmp, ensure_ascii=False)
        tmp.flush()
        tmp.close()
        stdout, stderr, rc, timed_out = _run_claude(
            user_prompt, system=sys_prompt, model=mdl, cfg_path=tmp.name,
            allowed=allowed, timeout=timeout_s)
        gate_decisions = _load_commit_gate_log(gate_log_path)
    finally:
        for p in (tmp.name, log_path, gate_log_path):
            try:
                os.unlink(p)
            except OSError:
                pass
    elapsed = max(0.0, time.monotonic() - started)

    parsed = _parse_stream_json(stdout)
    tool_calls = parsed["tool_calls"]
    iterations = sum(1 for t in tool_calls if t != SUBMIT_ANSWER)
    usage = parsed["usage"]

    # Usage-limit detection: trip the latch so the rest of the batch short-circuits; do NOT persist this
    # abstain to the resume sidecar so a later re-run retries the question.
    combined = "\n".join((stderr, parsed["subtype"], parsed["final_text"]))
    if (timed_out is False and rc != 0 and _is_usage_limit(combined)) or (
            parsed["is_error"] and _is_usage_limit(combined)):
        _USAGE_LIMIT.set()
        return _abstain_investigation(
            question, model=model_label, elapsed_s=elapsed, stop_reason="usage_limit",
            error=f"claude usage limit: {combined.strip()[:400]}", contract=contract,
            tool_calls=tool_calls, iterations=iterations, usage=usage)

    if timed_out:
        inv = _abstain_investigation(
            question, model=model_label, elapsed_s=elapsed, stop_reason="timeout",
            error=f"claude -p exceeded timeout_s={timeout_s}", contract=contract,
            tool_calls=tool_calls, iterations=iterations, usage=usage)
    elif rc != 0 and parsed["submit_args"] is None and not parsed["final_text"]:
        inv = _abstain_investigation(
            question, model=model_label, elapsed_s=elapsed, stop_reason="model_error",
            error=f"claude -p exited {rc}: {(stderr or stdout)[:400]}", contract=contract,
            tool_calls=tool_calls, iterations=iterations, usage=usage)
    else:
        gate_tel: dict[str, Any] | None = None
        if parsed["submit_args"] is not None:
            answer = _answer_from_args(parsed["submit_args"])
            if gate_decisions:
                from src.rag.agent import commit_gate as _commit_gate
                last = gate_decisions[-1]
                gate_tel = dict(last.get("telemetry") or {})
                # The server is the submit execution point. Honor its formatted/abstained terminal value;
                # a trailing REJECT means Claude ended without a successful re-submit and must fail closed.
                if _commit_gate.enforce():
                    verdict = str(last.get("verdict") or "")
                    if verdict in {"COMMIT", "ABSTAIN"}:
                        final = str(last.get("final_answer") or ABSTAIN)
                    else:
                        final = ABSTAIN
                    answer = Answer(answer=final,
                                    confidence=(answer.confidence if verdict == "COMMIT" else 0.0),
                                    evidence=answer.evidence,
                                    method=f"claude-mcp: commit_gate {verdict or 'REJECT'}")
            stop_reason = "answered"
        elif parsed["final_text"]:
            # Plain final-text answer with no submit_answer call — accepted at confidence 0.0, mirroring
            # the Gemini loop's plain-final-text branch.
            answer = Answer(answer=parsed["final_text"], confidence=0.0, evidence="",
                            method="claude-mcp: plain final text (no submit_answer)")
            if is_abstain(answer.answer):
                answer = Answer(answer=ABSTAIN, confidence=0.0, evidence="", method=answer.method)
            # A final text bypasses submit_answer, so no in-band retry is possible. Apply the same gate
            # once client-side and fail closed on REJECT, as required by SOT-2640.
            from src.rag.agent import commit_gate as _commit_gate
            if _commit_gate.enabled():
                decision = _commit_gate.evaluate(question, contract, answer.answer)
                gate_tel = decision.telemetry
                if _commit_gate.enforce():
                    final = (decision.final_answer if decision.verdict == _commit_gate.COMMIT else ABSTAIN)
                    answer = Answer(answer=final,
                                    confidence=(answer.confidence if decision.verdict == _commit_gate.COMMIT else 0.0),
                                    evidence=answer.evidence,
                                    method=f"claude-mcp: plain final commit_gate {decision.verdict}")
            stop_reason = "answered"
        else:
            answer = Answer(answer=ABSTAIN, confidence=0.0, evidence="",
                            method="claude-mcp: no answer emitted")
            stop_reason = "max_turns"
        # SOT-2650 — 値を変えない括弧内付加情報の書式契約 (RAG_FORMAT_STRIP_PAREN, default OFF). Applied at
        # the backend boundary so it covers both the submit_answer and plain-final-text commits even when
        # the commit gate itself is off (the cycle2 dev lever set runs gate-less).
        strip_tel: dict[str, Any] | None = None
        if stop_reason == "answered" and not is_abstain(answer.answer):
            from src.rag.agent import formatting as _fmt
            if _fmt.strip_paren_enabled():
                stripped, fired = _fmt.strip_trailing_parenthetical(question, answer.answer)
                strip_tel = {"applied": bool(fired), "rules": fired}
                if fired:
                    answer = Answer(answer=stripped, confidence=answer.confidence,
                                    evidence=answer.evidence, method=answer.method + " +paren_strip")
            # SOT-2656 — 値保存回答正規化 (説明文/接頭辞/カウンタ) (RAG_FORMAT_VALUE_NORM, default OFF).
            if _fmt.value_norm_enabled():
                normed, nfired = _fmt.normalize_value_answer(question, answer.answer)
                if nfired:
                    strip_tel = {"applied": True,
                                 "rules": list((strip_tel or {}).get("rules", [])) + nfired}
                    answer = Answer(answer=normed, confidence=answer.confidence,
                                    evidence=answer.evidence, method=answer.method + " +value_norm")
            # SOT-2682 — 小数指定問の単位 strip (RAG_DECIMAL_UNIT_STRIP, default OFF). 値保存(書式のみ):
            # 「小数第N位で答えて」問への回答末尾の単位サフィックスだけを落とす (idx79「7.00時間/タスク」→「7.00」)。
            if _fmt.decimal_unit_strip_enabled():
                dstripped, dfired = _fmt.strip_decimal_spec_unit(question, answer.answer)
                if dfired:
                    strip_tel = {"applied": True,
                                 "rules": list((strip_tel or {}).get("rules", [])) + dfired}
                    answer = Answer(answer=dstripped, confidence=answer.confidence,
                                    evidence=answer.evidence, method=answer.method + " +decimal_unit_strip")
            # SOT-2688 — ビン範囲の区間記法→チルダ書式 (RAG_BIN_RANGE_FORMAT, default OFF). 値保存(書式のみ):
            # ビン/範囲を問う問いへの区間式回答 (6.088138, 6.288138] を gold の「6.088138 ~ 6.288138」へ整形 (idx29)。
            if _fmt.bin_range_format_enabled():
                branged, bfired = _fmt.naturalize_bin_range(question, answer.answer)
                if bfired:
                    strip_tel = {"applied": True,
                                 "rules": list((strip_tel or {}).get("rules", [])) + bfired}
                    answer = Answer(answer=branged, confidence=answer.confidence,
                                    evidence=answer.evidence, method=answer.method + " +bin_range")
        inv = Investigation(
            question=question, answer=answer, iterations=iterations, tool_calls=tool_calls,
            usage=usage, model=model_label, elapsed_s=elapsed, stop_reason=stop_reason,
            error=None, contract=contract)
        if gate_tel is not None:
            inv.interventions["commit_gate"] = gate_tel
        if strip_tel is not None:
            inv.interventions["format_strip_paren"] = strip_tel

    # SOT-2661 — always record the 3-stage flow telemetry when RAG_PLAN_FANOUT is on (answered OR
    # abstained: timeout/model_error paths still carry the tool sequence up to the failure). Off ⇒
    # interventions untouched (byte-identical).
    if plan_fanout_enabled():
        inv.interventions["plan_fanout"] = _plan_fanout_telemetry(tool_calls, eff_max_turns)

    if resume is not None and inv.stop_reason != "usage_limit":
        _append_resume(resume, key, question, inv)
    return inv
