"""SOT-2467 — Gemini-only early validation gate over 5 gold questions.

Purpose (parent SOT-2460 Step1)
-------------------------------
Confirm the *Gemini-only, tool-driven* answer path is worth building **before** the full Step2
investigation agent is written. We hand a Gemini model **only the generic, corpus-agnostic tools**
(file discovery / grep / Office extraction / decryption / pandas compute / chart & vision reads) and a
question — **no corpus-specific fact** (passwords, 略称, 書式規則) is injected. The model must
*self-discover* those facts through the tools (e.g. brute-forcing an encrypted file's password from its
filename date) and re-derive the answer, which is then checked against a fixed gold value.

Five gold questions, one per **answer archetype** so the gate exercises every tool family:

======== ================================================================= ============================
type      what it proves                                                    primary tool(s)
======== ================================================================= ============================
decrypt   self-discovers an Office password, decrypts, reads a value        read_office / decrypt
format    extracts *formatting* meaning (highlight / emphasis markers)      pdf_emphasis / pptx_pivot / grep
compute   runs a deterministic pandas aggregation (暗算禁止)                compute
enumerate lists every matching identifier from a document                  read_office / grep
chart     reads a chart's plotted value (numCache or vision)               read_chart_values / caption_image
======== ================================================================= ============================

GO decision: ``n_correct >= GATE_THRESHOLD`` (4 of 5). The run records, per question, the tool-call
iteration count and token cost so Step2's design (model choice, tool set, budget) is grounded in
measured evidence, not a guess (受け入れ条件②).

Design for testability
----------------------
The agent loop (:func:`run_question`) is a *pure* driver over an abstract :class:`Model` — it never
imports the Gemini SDK. Live runs inject :class:`GeminiModel`; unit tests inject a scripted fake. The
generic tools are the real deterministic ones from :mod:`src.rag.tools`, so an offline test can drive a
real tool (grep / compute) end-to-end without any network call.
"""
from __future__ import annotations

import argparse
import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Mapping, Protocol, Sequence

from config import settings
from src.rag.tools import contract as _contract
from src.rag.tools.extract_tools import (
    caption_figure,
    decrypt as _decrypt,
    extract_office,
    find_files,
)
from src.rag.tools.file_grep import file_grep
from src.rag.tools.chart_numcache import extract_chart_numcache
from src.rag.tools.compute_sandbox import run as compute_run
from src.rag.tools.emf_pivot import extract_pptx_pivots
from src.rag.tools.highlight_extract import highlight_extract
from src.rag.tools.pdf_faux_italic import emphasized_words
from src.rag.tools.profile import CorpusProfile

# --------------------------------------------------------------------------- gate configuration
GATE_THRESHOLD = 4            # GO when at least this many of the 5 gold answers match
DEFAULT_MAX_TURNS = 12        # hard cap on model turns per question (tool rounds + a final answer)
ABSTAIN = settings.ABSTAIN

# Vertex Gemini list price (USD per 1M tokens), (input, output). Estimates for cost bookkeeping only —
# the *relative* per-question cost is what feeds Step2 sizing, so approximate list price is sufficient.
PRICING: dict[str, tuple[float, float]] = {
    "gemini-2.5-pro": (1.25, 10.0),
    "gemini-2.5-flash": (0.30, 2.50),
    "gemini-2.5-flash-lite": (0.10, 0.40),
}


@dataclass(frozen=True)
class GoldQuestion:
    """One fixed early-gate probe: an ``id``, its answer ``type``, the ``question`` and the ``gold``."""

    id: str
    type: str
    question: str
    gold: str


# The 5 gold questions. Questions/answers are the manually-verified gold from the SIGNATE valid set,
# except ``decrypt`` which targets an *encrypted* file (no valid-set question does) — its gold is the
# 税込見込金額 read from the decrypted かえで contract. No password / alias is stored here: the agent
# must self-discover them (移植性の担保).
GOLD_QUESTIONS: tuple[GoldQuestion, ...] = (
    GoldQuestion(
        id="decrypt",
        type="decrypt",
        question=(
            "恒一会 かえで総合病院の暗号化された契約書ファイル（データ分析業務委託契約書）における、"
            "税込見込金額はいくらですか。"
        ),
        gold="3,850,000円",
    ),
    GoldQuestion(
        id="format",
        type="format",
        question=(
            "青潮モビリティサービスの最終報告における、モビリティ需要の要因分析のページで、"
            "マーカーされている単語をすべて抜き出してください。"
        ),
        gold="hr、weekday、weathersit、temp",
    ),
    GoldQuestion(
        id="compute",
        type="compute",
        question=(
            "青葉与信マネジメントの分析対象データにおいて、term=3 years、grade=B1、"
            "purpose=credit_card に該当する loan_amnt の平均を算出してください。"
            "四捨五入して整数値で出してください。"
        ),
        gold="1526",
    ),
    GoldQuestion(
        id="enumerate",
        type="enumerate",
        question=(
            "AYMのPLにおいて、探索的分析・仮説整理フェーズに一致するタスクIDをすべて挙げてください。"
        ),
        gold="T09、T10、T11、T12",
    ),
    GoldQuestion(
        id="chart",
        type="chart",
        question=(
            "KSSのfigure_06.pngにおいて、dayによる件数推移とあわせて表示されているTG平均が"
            "最も低い日は何日ですか。"
        ),
        gold="20日",
    ),
)

SYSTEM_PROMPT = (
    "あなたは社内ドライブの文書QAを行う調査エージェントです。与えられた汎用ツールだけを使って質問に答えます。\n"
    "厳守事項:\n"
    "1. 暗算・記憶・創作で答えない。必ずツールで根拠となる値を取得してから答える。\n"
    "2. パスワード・略称・書式規則などのコーパス固有事実は与えられていない。ツール(ファイル探索/grep/"
    "Office抽出/復号/pandas計算/チャート読取/画像説明)で自力発見する。暗号化ファイルは復号ツールが"
    "ファイル名等から鍵を推定して復号する。\n"
    "3. まず関連ファイルを探索し、必要なツールを反復呼び出しして値を確定する。ツールがエラー/空を返しても"
    "諦めず、原因(列名違い・ファイル違い・値表記違い)を切り分けて別のファイルや式で必ず再試行する。安易に"
    "棄権しない。\n"
    "4. 数値計算(平均・合計・件数など)は必ず compute ツールで行う。列名や絞り込み値が不明なときは、まず"
    "`df.columns.tolist()` や `df['列'].unique().tolist()` を compute で確認してから集計式を組む。生データは"
    "各案件の「03.データ」直下(train.csv 等)を優先する。\n"
    "5. チャート/図の質問は、まず read_chart_values(Office埋め込みチャートの厳密値)を試し、PNG等で使えない"
    "場合のみ caption_image(vision)で読む。系列とカテゴリ(例: day 別)を対応付けて最小/最大の該当ラベルを"
    "特定する。\n"
    "6. 書式(マーカー/強調/ハイライト)の質問は、まず highlight_extract(xlsx/pptx/docx/pdfのハイライト・"
    "マーカー語/セルを文書順で列挙、colorで色指定可)を使う。補助として pdf_emphasis・pptx_pivot・"
    "read_office の書式情報や対象語の file_grep も併用して該当語を特定する。\n"
    "7. 十分な根拠が得られたら、ツールを呼ばず最終回答テキストのみを簡潔に返す(余計な説明・前置きは不要、"
    "値/一覧のみ)。列挙は「、」区切り。金額は原文の表記(例: 3,850,000円)。\n"
    f"8. あらゆる手段を尽くしても根拠が得られない場合に限り「{ABSTAIN}」と答える。"
)


# --------------------------------------------------------------------------- usage / results
@dataclass
class Usage:
    """Token usage accumulator (output includes any 'thinking' tokens for cost realism)."""

    input_tokens: int = 0
    output_tokens: int = 0

    def __add__(self, other: "Usage") -> "Usage":
        return Usage(self.input_tokens + other.input_tokens, self.output_tokens + other.output_tokens)

    @property
    def total_tokens(self) -> int:
        return self.input_tokens + self.output_tokens

    def cost_usd(self, model: str) -> float:
        pin, pout = PRICING.get(model, PRICING["gemini-2.5-pro"])
        return self.input_tokens / 1e6 * pin + self.output_tokens / 1e6 * pout


@dataclass(frozen=True)
class Call:
    """A single function call the model requested."""

    name: str
    args: dict[str, Any]


@dataclass(frozen=True)
class ToolResponse:
    """The dispatched result for one :class:`Call`, fed back to the model."""

    name: str
    response: Any


@dataclass(frozen=True)
class Step:
    """One model turn: either ``function_calls`` to run, or a ``final_text`` answer (never both)."""

    function_calls: tuple[Call, ...] = ()
    final_text: str | None = None
    usage: Usage = field(default_factory=Usage)


@dataclass
class QuestionResult:
    """Outcome for one gold question."""

    id: str
    type: str
    question: str
    gold: str
    answer: str
    correct: bool
    iterations: int
    tool_calls: list[str]
    usage: Usage
    model: str
    error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "type": self.type,
            "question": self.question,
            "gold": self.gold,
            "answer": self.answer,
            "correct": self.correct,
            "iterations": self.iterations,
            "tool_calls": list(self.tool_calls),
            "input_tokens": self.usage.input_tokens,
            "output_tokens": self.usage.output_tokens,
            "total_tokens": self.usage.total_tokens,
            "cost_usd": round(self.usage.cost_usd(self.model), 6),
            "model": self.model,
            "error": self.error,
        }


# --------------------------------------------------------------------------- generic tool layer
@dataclass(frozen=True)
class AgentTool:
    """A generic, agent-callable tool: JSON-schema ``parameters`` + a Python ``fn``."""

    name: str
    description: str
    parameters: dict[str, Any]
    fn: Callable[..., Any]


def _obj(props: dict[str, dict[str, Any]], required: Sequence[str] = ()) -> dict[str, Any]:
    return {"type": "object", "properties": props, "required": list(required)}


_STR = {"type": "string"}
_BOOL = {"type": "boolean"}


def build_tools(profile: CorpusProfile) -> list[AgentTool]:
    """Wire the deterministic :mod:`src.rag.tools` into agent-callable tools bound to ``profile``.

    ``profile`` carries self-discovered secrets (passwords/aliases) across a single question's tool
    calls; it is never seeded with corpus facts here (移植性の担保).
    """
    return [
        AgentTool(
            "find_files",
            "コーパス内のファイルを名前/拡張子/プロジェクトで検索し、該当ファイル一覧を返す。",
            _obj({"query": _STR, "ext": _STR, "project": _STR}),
            lambda query=None, ext=None, project=None: find_files(query, ext=ext, project=project),
        ),
        AgentTool(
            "file_grep",
            "コーパス全体を全文/セル/ファイル名でgrepし、一致箇所(ファイル・行・抜粋)を返す。",
            _obj({"query": _STR, "ext": _STR, "project": _STR, "regex": _BOOL}, ["query"]),
            lambda query, ext=None, project=None, regex=False: file_grep(
                query, regex=bool(regex), ext=ext, project=project),
        ),
        AgentTool(
            "read_office",
            "docx/xlsx/pptxから書式保持テキストを抽出する。暗号化ファイルは鍵を自力推定して透過復号する。",
            _obj({"file": _STR}, ["file"]),
            lambda file: extract_office(file, profile=profile),
        ),
        AgentTool(
            "decrypt",
            "暗号化Officeファイルの復号可否と方式(根拠のみ、パスワードは返さない)を確認する。",
            _obj({"file": _STR}, ["file"]),
            lambda file: _decrypt(file, profile=profile),
        ),
        AgentTool(
            "compute",
            "csv/xlsxに対し単一のpandas式(dfを参照)を実行し、計算値と根拠(列・範囲)を返す。暗算の代替。",
            _obj({"file": _STR, "expr": _STR, "sheet": _STR}, ["file", "expr"]),
            lambda file, expr, sheet=None: compute_run(file, expr, sheet=sheet),
        ),
        AgentTool(
            "read_chart_values",
            "xlsx/pptx埋め込みチャートのnumCacheから系列名・カテゴリ・プロット値を厳密に読む(推測なし)。",
            _obj({"file": _STR}, ["file"]),
            lambda file: extract_chart_numcache(file),
        ),
        AgentTool(
            "caption_image",
            "図表PNG画像をvisionモデルで説明する(numCacheが無い画像チャート向け)。",
            _obj({"file": _STR}, ["file"]),
            lambda file: caption_figure(file),
        ),
        AgentTool(
            "pdf_emphasis",
            "PDF内の強調(疑似イタリック=行列シアー)された単語を抽出する。",
            _obj({"file": _STR}, ["file"]),
            lambda file: emphasized_words(file),
        ),
        AgentTool(
            "pptx_pivot",
            "pptxに埋め込まれたPivotTable(EMF)を復元し、表とハイライトされたセルを返す。",
            _obj({"file": _STR}, ["file"]),
            lambda file: extract_pptx_pivots(file),
        ),
        AgentTool(
            "highlight_extract",
            "xlsx/pptx/docx/pdfのハイライト・マーカー強調された語/セルを文書順で列挙する。"
            "colorに色名(黄/オレンジ/赤/青など)を渡すとその色だけに絞れる。書式型(マーカー語抽出・"
            "色付きセルの条件)の質問に使う。",
            _obj({"file": _STR, "color": _STR}, ["file"]),
            lambda file, color=None: highlight_extract(file, color=color, profile=profile),
        ),
    ]


def _jsonable(obj: Any, *, max_str: int = 4000, max_items: int = 60, _depth: int = 0) -> Any:
    """Best-effort JSON-safe, size-bounded view of a tool result (keeps token cost in check)."""
    if _contract.is_contract(obj):
        obj = _contract.ensure_contract(obj)
    if isinstance(obj, str):
        return obj if len(obj) <= max_str else obj[:max_str] + f"…(+{len(obj) - max_str}字)"
    if isinstance(obj, Mapping):
        return {str(k): _jsonable(v, max_str=max_str, max_items=max_items, _depth=_depth + 1)
                for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        head = list(obj)[:max_items]
        out = [_jsonable(v, max_str=max_str, max_items=max_items, _depth=_depth + 1) for v in head]
        if len(obj) > max_items:
            out.append(f"…(+{len(obj) - max_items}件)")
        return out
    if isinstance(obj, (int, float, bool)) or obj is None:
        return obj
    return str(obj)


def dispatch(tools_by_name: Mapping[str, AgentTool], name: str, args: Mapping[str, Any] | None) -> Any:
    """Run tool ``name`` with ``args``; return a JSON-safe result or an ``{"error": ...}`` mapping.

    Errors are returned (not raised) so the model can see the failure and try another approach — the
    agent loop must never crash on one bad tool call.
    """
    tool = tools_by_name.get(name)
    if tool is None:
        return {"error": f"unknown tool: {name}", "available": sorted(tools_by_name)}
    try:
        out = tool.fn(**dict(args or {}))
    except TypeError as e:
        return {"error": f"bad arguments for {name}: {e}"}
    except Exception as e:  # noqa: BLE001 — surface any tool failure back to the model
        return {"error": f"{type(e).__name__}: {e}"}
    return _jsonable(out)


# --------------------------------------------------------------------------- gold matching
_DELIMS = "、,，/／\n\t 　;；・"


def _norm(s: str) -> str:
    import unicodedata
    s = unicodedata.normalize("NFKC", str(s)).strip()
    for tail in ("です", "でした", "。"):
        if s.endswith(tail):
            s = s[: -len(tail)]
    return s.strip().lower()


def _numbers(s: str) -> list[str]:
    import re
    return sorted(m.group(0).replace(",", "") for m in re.finditer(r"-?\d[\d,]*(?:\.\d+)?", str(s)))


def _tokens(s: str) -> set[str]:
    import re
    parts = re.split(f"[{re.escape(_DELIMS)}]+", _norm(s))
    return {p for p in (t.strip() for t in parts) if p}


def is_abstain(answer: str) -> bool:
    a = _norm(answer)
    return not a or a == _norm(ABSTAIN) or "わかりません" in a or "不明" in a


def answers_match(pred: str, gold: str, qtype: str) -> bool:
    """True when ``pred`` matches ``gold`` for a question of ``qtype``.

    * ``enumerate`` / ``format`` → order-insensitive **set** equality of delimiter-split tokens.
    * ``compute`` / ``decrypt`` / ``chart`` (numeric answers) → equal number multiset, OR the gold
      string appears verbatim in the (normalized) prediction.
    * otherwise → normalized-equal or gold ⊆ pred.
    """
    if is_abstain(pred):
        return False
    np, ng = _norm(pred), _norm(gold)
    if np == ng:
        return True
    if qtype in ("enumerate", "format"):
        return _tokens(pred) == _tokens(gold)
    if qtype in ("compute", "decrypt", "chart"):
        gn = _numbers(gold)
        if gn and _numbers(pred) == gn:
            return True
        return ng in np
    return ng in np or np in ng


# --------------------------------------------------------------------------- agent loop (pure)
class Model(Protocol):
    """A model conversation for one question. ``next(None)`` starts it; ``next(responses)`` continues
    it after tool calls. Implementations hold their own history; the loop stays transport-agnostic."""

    def next(self, tool_responses: Sequence[ToolResponse] | None) -> Step: ...


def run_question(model: Model, question: GoldQuestion, tools: Sequence[AgentTool], *,
                 max_turns: int = DEFAULT_MAX_TURNS) -> QuestionResult:
    """Drive ``model`` through tool-calling until it returns a final answer (or ``max_turns`` hit).

    ``iterations`` counts tool rounds (turns that requested ≥1 function call). Token usage is summed
    across every turn. On max-turns with no final answer the question resolves to an abstention.
    """
    by_name = {t.name: t for t in tools}
    usage = Usage()
    tool_calls: list[str] = []
    iterations = 0
    responses: list[ToolResponse] | None = None
    answer = ABSTAIN
    error: str | None = None

    for _turn in range(max_turns):
        try:
            step = model.next(responses)
        except Exception as e:  # noqa: BLE001 — a transport failure ends the question, not the gate
            error = f"model error: {type(e).__name__}: {e}"
            break
        usage = usage + step.usage
        if not step.function_calls:
            answer = (step.final_text or "").strip() or ABSTAIN
            break
        iterations += 1
        responses = []
        for call in step.function_calls:
            tool_calls.append(call.name)
            out = dispatch(by_name, call.name, call.args)
            responses.append(ToolResponse(call.name, out))
    else:
        # loop exhausted without a final-answer turn
        error = error or f"max_turns={max_turns} reached without a final answer"

    correct = error is None and answers_match(answer, question.gold, question.type)
    model_name = getattr(model, "model_name", settings.GEN_MODEL_HARD)
    return QuestionResult(
        id=question.id, type=question.type, question=question.question, gold=question.gold,
        answer=answer, correct=correct, iterations=iterations, tool_calls=tool_calls,
        usage=usage, model=model_name, error=error,
    )


# --------------------------------------------------------------------------- live Gemini model
class GeminiModel:
    """Live Vertex Gemini function-calling conversation (imported lazily so tests stay offline)."""

    def __init__(self, question: str, genai_tools: Any, *, model: str | None = None,
                 system: str = SYSTEM_PROMPT, max_output_tokens: int = 1024,
                 thinking_budget: int = 1024):
        from google.genai import types

        from src.rag import llm

        self._types = types
        self._client = llm.client()
        self.model_name = model or settings.GEN_MODEL_HARD
        self._contents = [types.Content(role="user", parts=[types.Part.from_text(text=question)])]
        self._config = types.GenerateContentConfig(
            temperature=0.0, seed=0, system_instruction=system,
            max_output_tokens=max_output_tokens,
            thinking_config=types.ThinkingConfig(thinking_budget=thinking_budget),
            tools=genai_tools,
        )

    def next(self, tool_responses: Sequence[ToolResponse] | None) -> Step:
        types = self._types
        if tool_responses:
            self._contents.append(types.Content(role="user", parts=[
                types.Part.from_function_response(
                    name=r.name,
                    response=r.response if isinstance(r.response, Mapping) else {"result": r.response},
                ) for r in tool_responses]))
        resp = self._client.models.generate_content(
            model=self.model_name, contents=self._contents, config=self._config)
        cand = resp.candidates[0]
        parts = list(cand.content.parts or [])
        self._contents.append(cand.content)  # keep the model's own turn in history
        calls = tuple(
            Call(p.function_call.name, dict(p.function_call.args or {}))
            for p in parts if getattr(p, "function_call", None))
        text = "".join(p.text for p in parts if getattr(p, "text", None))
        um = getattr(resp, "usage_metadata", None)
        prompt_toks = getattr(um, "prompt_token_count", 0) or 0
        total_toks = getattr(um, "total_token_count", 0) or 0
        usage = Usage(prompt_toks, max(0, total_toks - prompt_toks))
        return Step(function_calls=calls, final_text=(None if calls else text), usage=usage)


def to_genai_tools(tools: Sequence[AgentTool]) -> Any:
    """Convert :class:`AgentTool` schemas into a google-genai ``Tool`` (live path only)."""
    from google.genai import types

    _TYPE = {"object": types.Type.OBJECT, "string": types.Type.STRING,
             "boolean": types.Type.BOOLEAN, "number": types.Type.NUMBER,
             "integer": types.Type.INTEGER, "array": types.Type.ARRAY}

    def _schema(d: Mapping[str, Any]) -> Any:
        kind = _TYPE.get(d.get("type", "string"), types.Type.STRING)
        props = d.get("properties")
        return types.Schema(
            type=kind,
            properties={k: _schema(v) for k, v in props.items()} if props else None,
            required=list(d.get("required", [])) or None,
        )

    decls = [types.FunctionDeclaration(name=t.name, description=t.description,
                                       parameters=_schema(t.parameters)) for t in tools]
    return [types.Tool(function_declarations=decls)]


def gemini_model_factory(question: GoldQuestion, tools: Sequence[AgentTool], *,
                         model: str | None = None) -> GeminiModel:
    """Fresh live conversation for one question (each question gets an isolated context)."""
    return GeminiModel(question.question, to_genai_tools(tools), model=model)


# --------------------------------------------------------------------------- gate driver
def run_gate(model_factory: Callable[[GoldQuestion, Sequence[AgentTool]], Model], *,
             questions: Sequence[GoldQuestion] = GOLD_QUESTIONS,
             profile_factory: Callable[[], CorpusProfile] | None = None,
             max_turns: int = DEFAULT_MAX_TURNS) -> dict[str, Any]:
    """Run every gold question through a fresh model+profile and aggregate a GO/NO-GO report.

    ``model_factory(question, tools) -> Model`` builds the (live or fake) conversation. Each question
    gets its own :class:`CorpusProfile` so a self-discovered secret never leaks the answer across
    questions. Returns a JSON-able summary (also the shape written to the artifact / ledger).
    """
    prof = profile_factory or (lambda: CorpusProfile())
    results: list[QuestionResult] = []
    for q in questions:
        tools = build_tools(prof())
        model = model_factory(q, tools)
        results.append(run_question(model, q, tools, max_turns=max_turns))

    n_correct = sum(r.correct for r in results)
    total_usage = Usage()
    for r in results:
        total_usage = total_usage + r.usage
    model_name = results[0].model if results else settings.GEN_MODEL_HARD
    return {
        "gate": "early_gate",
        "issue": "SOT-2467",
        "n_questions": len(results),
        "n_correct": n_correct,
        "threshold": GATE_THRESHOLD,
        "go": n_correct >= GATE_THRESHOLD,
        "total_iterations": sum(r.iterations for r in results),
        "total_input_tokens": total_usage.input_tokens,
        "total_output_tokens": total_usage.output_tokens,
        "total_cost_usd": round(total_usage.cost_usd(model_name), 6),
        "results": [r.to_dict() for r in results],
    }


def write_artifact(summary: Mapping[str, Any], path: str | Path) -> Path:
    """Write the gate summary as pretty JSON; returns the path."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return p


def append_ledger(summary: Mapping[str, Any], path: str | Path, *, recorded_at: str,
                  cycle: int = 1) -> None:
    """Append one experiment-ledger JSONL entry recording this gate's outcome."""
    entry = {
        "recordedAt": recorded_at,
        "axis": "gemini-only early validation gate (SOT-2467)",
        "result": "promoted" if summary.get("go") else "inconclusive",
        "cycle": cycle,
        "hypothesis": "汎用ツール+Geminiのfunction-callingでコーパス固有事実を自力発見し、型の異なる"
                      "ゴールド5問を4/5以上再導出できる(Gemini-onlyパスのStep2着手可否)。",
        "evidence": (f"n_correct={summary.get('n_correct')}/{summary.get('n_questions')} "
                     f"go={summary.get('go')} total_iterations={summary.get('total_iterations')} "
                     f"cost_usd={summary.get('total_cost_usd')}"),
    }
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(entry, ensure_ascii=False) + "\n")


def _print_report(summary: Mapping[str, Any]) -> None:
    verdict = "GO ✅" if summary["go"] else "NO-GO ❌"
    print(f"\n=== Gemini early gate: {verdict}  "
          f"({summary['n_correct']}/{summary['n_questions']}, threshold {summary['threshold']}) ===")
    for r in summary["results"]:
        mark = "○" if r["correct"] else "×"
        print(f" [{mark}] {r['id']:<9} iters={r['iterations']} "
              f"tokens={r['total_tokens']} cost=${r['cost_usd']:.4f}")
        print(f"       gold={r['gold']!r}  pred={r['answer'][:80]!r}")
        if r["error"]:
            print(f"       error: {r['error']}")
        print(f"       tools: {r['tool_calls']}")
    print(f" total: iters={summary['total_iterations']} "
          f"tokens={summary['total_input_tokens']}+{summary['total_output_tokens']} "
          f"cost=${summary['total_cost_usd']:.4f}\n")


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Gemini-only early validation gate (SOT-2467).")
    ap.add_argument("--model", default=None, help="Gemini model (default: settings.GEN_MODEL_HARD)")
    ap.add_argument("--max-turns", type=int, default=DEFAULT_MAX_TURNS)
    ap.add_argument("--artifact", default=str(settings.ARTIFACTS_DIR / "early_gate.json"))
    ap.add_argument("--ledger", default=str(settings.REPO_ROOT / "docs/ai/experiment_ledger.jsonl"))
    ap.add_argument("--no-ledger", action="store_true", help="do not append a ledger entry")
    args = ap.parse_args(argv)

    def factory(q: GoldQuestion, tools: Sequence[AgentTool]) -> Model:
        return gemini_model_factory(q, tools, model=args.model)

    summary = run_gate(factory, max_turns=args.max_turns)
    _print_report(summary)
    write_artifact(summary, args.artifact)
    print(f"artifact → {args.artifact}")
    if not args.no_ledger:
        append_ledger(summary, args.ledger,
                      recorded_at=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()))
    return 0 if summary["go"] else 1


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
