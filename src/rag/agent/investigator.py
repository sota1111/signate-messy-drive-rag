"""SOT-2468 — Gemini function-calling investigation loop (skeleton).

Parent SOT-2460 Step2. This is the **reusable production** answer path: one question is answered by a
*plan → tool iteration → structured answer* loop over Vertex Gemini function-calling. The model is given
only the corpus-agnostic Step1 tools (file discovery / grep / Office extraction / decryption / pandas
compute / chart & vision reads) — **no corpus-specific fact** (passwords, 略称, 書式規則) is injected;
the model self-discovers them through the tools. The loop terminates when the model calls the terminal
``submit_answer`` tool, returning the fixed answer schema::

    {answer, confidence, evidence, method}

Design for testability
----------------------
The loop (:func:`investigate`) is a *pure* driver over an abstract :class:`Model`; it never imports the
Gemini SDK. Live runs inject :class:`GeminiModel`; unit tests inject a scripted fake. The tools are the
real deterministic ones from :mod:`src.rag.tools`, so an offline test can drive a real tool end-to-end
without any network call.

Guardrails (受け入れ条件・実装内容)
-----------------------------------
* **反復上限** — ``max_turns`` caps model turns per question; on exhaustion the loop abstains.
* **タイムアウト** — ``timeout_s`` (wall-clock, checked between turns) bounds a single question.
* **コスト計上** — token usage is summed every turn and priced via :meth:`Usage.cost_usd`.

The Step1 early-validation gate :mod:`scoring.early_gate` (SOT-2467) shares the same agent-loop shape;
its consolidation onto this module (and wiring into the production answer path) is SOT-2469's scope, so
this change is intentionally self-contained and leaves the green gate untouched.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Callable, Mapping, Protocol, Sequence

from config import settings
from src.rag.tools import contract as _contract
from src.rag.tools.chart_numcache import extract_chart_numcache
from src.rag.tools.compute_sandbox import run as compute_run
from src.rag.tools.emf_pivot import extract_pptx_pivots
from src.rag.tools.extract_tools import (
    caption_figure,
    decrypt as _decrypt,
    extract_office,
    find_files,
)
from src.rag.tools.file_grep import file_grep
from src.rag.tools.highlight_extract import highlight_extract
from src.rag.tools.pdf_faux_italic import emphasized_words
from src.rag.tools.profile import CorpusProfile
from src.rag.tools.seating_chart import seating_lookup

# --------------------------------------------------------------------------- loop configuration
DEFAULT_MAX_TURNS = 12            # hard cap on model turns per question (tool rounds + the final answer)
DEFAULT_TIMEOUT_S = 180.0         # wall-clock budget per question (checked between turns)
ABSTAIN = settings.ABSTAIN
SUBMIT_ANSWER = "submit_answer"   # terminal tool name the model calls to finish

# Vertex Gemini list price (USD per 1M tokens), (input, output) — estimates for cost bookkeeping only.
PRICING: dict[str, tuple[float, float]] = {
    "gemini-2.5-pro": (1.25, 10.0),
    "gemini-2.5-flash": (0.30, 2.50),
    "gemini-2.5-flash-lite": (0.10, 0.40),
}

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
    "4. 数値計算(平均・合計・件数など)は必ず compute ツールで行う。train.xlsx/train.csv 等は案件ごとに"
    "同名で複数存在するので、質問が指す案件名を compute の project 引数(会社名の一部)で渡してファイルを"
    "特定する。曖昧エラーが返ったら、そのエラーが挙げる『存在プロジェクト』から該当案件を選んで project を"
    "付けて再試行する(棄権しない)。列名や絞り込み値が不明なときは、まず `df.columns.tolist()` や"
    "`df['列'].unique().tolist()` を compute で確認してから集計式を組む。\n"
    "5. 旧版(old版)と最新版の比較・変更点を問う質問は、grepで手作業比較せず version_diff ツールに質問文を"
    "そのまま渡す。決定論の構造diffが『変更前 → 変更後』を返すので、その value をそのまま回答にする。value が"
    "null のときのみ他手段(grep等)を検討する。\n"
    "6a. 内線番号/EXT/座席/『向かい・隣・同じ列』を問う質問は seating_lookup ツールを使う(座席表は画像1枚で"
    "grep/office抽出では読めない)。多段(案件→担当者→内線)では先に担当者の氏名を他ツールで特定し、その氏名を"
    "seating_lookup(name=…)に渡す。『Aさんの向かいの人のEXT』は seating_lookup(name='A', relation='向かい')。\n"
    "6. 十分な根拠が得られたら、最終回答は必ず submit_answer ツールを1回だけ呼んで返す(通常のテキストでは"
    "答えない)。submit_answer には次を渡す: answer=回答本文(値/一覧のみ、列挙は「、」区切り、金額は原文表記)、"
    "confidence=0.0〜1.0の自己確信度、evidence=根拠(参照ファイル・値・ツール結果)、method=導出手順の要約。\n"
    f"7. あらゆる手段を尽くしても根拠が得られない場合に限り answer=「{ABSTAIN}」・confidence=0.0 で submit_answer"
    "する。"
)


# --------------------------------------------------------------------------- usage / answer schema
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
class Answer:
    """The fixed structured answer schema every investigation returns."""

    answer: str
    confidence: float          # self-reported, clamped to [0.0, 1.0]
    evidence: str = ""         # 根拠: files / values / tool outputs the answer rests on
    method: str = ""           # how it was derived (which tools / steps)

    def to_dict(self) -> dict[str, Any]:
        return {"answer": self.answer, "confidence": self.confidence,
                "evidence": self.evidence, "method": self.method}


def _coerce_confidence(value: Any) -> float:
    """Best-effort parse of a model-supplied confidence into a clamped float in [0.0, 1.0]."""
    try:
        c = float(value)
    except (TypeError, ValueError):
        return 0.0
    if c != c:  # NaN
        return 0.0
    return max(0.0, min(1.0, c))


def _answer_from_args(args: Mapping[str, Any] | None) -> Answer:
    """Build an :class:`Answer` from the model's ``submit_answer`` arguments (robust to omissions)."""
    a = dict(args or {})
    text = str(a.get("answer", "") or "").strip() or ABSTAIN
    conf = _coerce_confidence(a.get("confidence"))
    if is_abstain(text):
        conf = 0.0
    return Answer(answer=text, confidence=conf,
                  evidence=str(a.get("evidence", "") or ""),
                  method=str(a.get("method", "") or ""))


def is_abstain(answer: str) -> bool:
    import unicodedata

    a = unicodedata.normalize("NFKC", str(answer)).strip().lower()
    return not a or a == unicodedata.normalize("NFKC", ABSTAIN).strip().lower() \
        or "わかりません" in a or "不明" in a


# --------------------------------------------------------------------------- transport data types
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
class Investigation:
    """Outcome for one investigated question."""

    question: str
    answer: Answer
    iterations: int             # tool rounds (turns that requested ≥1 tool call)
    tool_calls: list[str]
    usage: Usage
    model: str
    elapsed_s: float
    stop_reason: str            # "answered" | "max_turns" | "timeout" | "model_error"
    error: str | None = None

    @property
    def confidence(self) -> float:
        return self.answer.confidence

    def to_dict(self) -> dict[str, Any]:
        return {
            "question": self.question,
            **self.answer.to_dict(),
            "iterations": self.iterations,
            "tool_calls": list(self.tool_calls),
            "input_tokens": self.usage.input_tokens,
            "output_tokens": self.usage.output_tokens,
            "total_tokens": self.usage.total_tokens,
            "cost_usd": round(self.usage.cost_usd(self.model), 6),
            "model": self.model,
            "elapsed_s": round(self.elapsed_s, 3),
            "stop_reason": self.stop_reason,
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
_NUM = {"type": "number"}


def _version_diff(question: str) -> dict[str, Any]:
    """Deterministic version-diff solver, wrapped in the {value, evidence, method} contract.

    Wires the deterministic differ (:func:`src.rag.diffpair.answer_question_agent`, built on the
    SOT-2408 hold-out-validated solver) into the agent's function-calling loop so a 版差分 question is
    answered by an actual structural diff instead of the model guessing. Imported lazily so serve-time
    import of this module stays lean.

    ``value`` is the rendered "変更前 → 変更後" summary, or ``None`` when the differ abstains (not a
    diff question / no unambiguous pair / non-adjacent versions / unreadable / too many realigned
    changes). ``None`` is a
    legitimate contract value meaning "resolved nothing" — the agent then abstains as before, never a
    guess (precision 1.0 維持).
    """
    from src.rag import diffpair  # lazy: keeps serve-time import free of corpus/Office deps

    if not diffpair.is_diff_question(question):
        return _contract.make(
            None, engine="diffpair", evidence={"applicable": False},
            note="版差分質問ではない(旧版/old版/_r1_r2等の版参照+比較動詞が必要)")
    # precision-first agent entry: abstains on non-adjacent / unreadable / ambiguous pairs so a
    # wrong diff (−1) is never surfaced in place of an abstention (0).
    answer = diffpair.answer_question_agent(question)
    return _contract.make(
        answer, engine="diffpair",
        evidence={"applicable": True, "resolved": answer is not None},
        scheme="structural-version-diff",
        note=("隣接版(旧版→最新/vN→vN+1)の構造diff(セル/段落を整列し実質変更のみ)"
              if answer is not None
              else "版ペア不確定/非隣接/読取不能/大規模変更のため棄権(None)"))


# The terminal tool: calling it ends the investigation with the structured answer schema. Its ``fn`` is
# never dispatched (the loop intercepts the call), but a no-op keeps the tool uniform.
SUBMIT_ANSWER_TOOL = AgentTool(
    SUBMIT_ANSWER,
    "十分な根拠が得られたら最終回答を確定する。answer(回答本文)・confidence(0.0〜1.0)・evidence(根拠)・"
    "method(導出手順)を渡す。これを呼ぶと調査は終了する。",
    _obj({"answer": _STR, "confidence": _NUM, "evidence": _STR, "method": _STR}, ["answer"]),
    lambda answer=None, confidence=None, evidence=None, method=None: {"submitted": True},
)


def build_generic_tools(profile: CorpusProfile) -> list[AgentTool]:
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
            "csv/xlsxに対し単一のpandas式(dfを参照)を実行し、計算値と根拠(列・範囲)を返す。暗算の代替。"
            "train.xlsx/train.csv等は案件ごとに同名で複数存在するため、案件名を project(会社名の一部)で"
            "渡してファイルを特定する。曖昧エラー時は返された『存在プロジェクト』から project を選び再試行する。",
            _obj({"file": _STR, "expr": _STR, "sheet": _STR, "project": _STR}, ["file", "expr"]),
            lambda file, expr, sheet=None, project=None: compute_run(
                file, expr, sheet=sheet, project=project),
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
        AgentTool(
            "version_diff",
            "同一文書の旧版と最新版(または _r1/_r2・_v1/_v3 等の版)を構造diffし、変更点を"
            "『(項目)：変更前 → 変更後』で決定論的に返す。旧版/old版/前の版と最新版の比較・変更箇所を"
            "問う質問に使う。question に質問文をそのまま渡す(会社名・文書名・版指定を含めるほど特定精度が"
            "上がる)。値は決定論・推測なし。版ペアが一意に定まらない/読取不能/変更が大規模すぎる場合は"
            "value=null を返す(その場合は棄権のまま)。",
            _obj({"question": _STR}, ["question"]),
            lambda question: _version_diff(question),
        ),
        AgentTool(
            "seating_lookup",
            "座席表(フロアマップ)から 氏名⇄内線(EXT)⇄座席 を引く。内線/EXT/座席/『向かい・隣・同じ列』を"
            "問う質問に使う(座席図は画像1枚で他ツールでは読めない)。name=氏名(『〜さん』可)を渡すとその人の"
            "EXTを返す。relation に『向かい/隣/同じ列/同じ行』を渡すとその隣人のEXTを返す(例: 井上さんの"
            "向かいの人のEXT)。ext=内線から人物を、role=役割(Exec/PM/DS/BA/DE/QA)+pod で EXT を引くこともできる。"
            "多段質問(案件→担当者→内線)では、先に担当者の氏名を他ツールで特定し、その氏名を name で渡す。"
            "該当なし/曖昧なときは value=null(棄権)を返す。",
            _obj({"name": _STR, "ext": _STR, "role": _STR, "relation": _STR, "pod": _NUM}),
            lambda name=None, ext=None, role=None, relation=None, pod=None: seating_lookup(
                name=name, ext=ext, role=role, relation=relation,
                pod=int(pod) if isinstance(pod, (int, float)) else None),
        ),
    ]


def build_tools(profile: CorpusProfile) -> list[AgentTool]:
    """The full tool set exposed to the investigator: generic Step1 tools + the terminal answer tool."""
    return [*build_generic_tools(profile), SUBMIT_ANSWER_TOOL]


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


# --------------------------------------------------------------------------- agent loop (pure)
class Model(Protocol):
    """A model conversation for one question. ``next(None)`` starts it; ``next(responses)`` continues
    it after tool calls. Implementations hold their own history; the loop stays transport-agnostic."""

    def next(self, tool_responses: Sequence[ToolResponse] | None) -> Step: ...


def investigate(model: Model, question: str, tools: Sequence[AgentTool], *,
                max_turns: int = DEFAULT_MAX_TURNS, timeout_s: float = DEFAULT_TIMEOUT_S,
                clock: Callable[[], float] = time.monotonic) -> Investigation:
    """Drive ``model`` through tool-calling until it submits a structured answer.

    The loop ends on the first of: the model calls ``submit_answer`` (→ ``answered``); ``max_turns`` is
    reached (→ ``max_turns``, abstain); the wall-clock ``timeout_s`` is exceeded between turns
    (→ ``timeout``, abstain); or the transport raises (→ ``model_error``, abstain). A plain final-text
    turn (no ``submit_answer``) is accepted as the answer with confidence 0.0.

    ``iterations`` counts tool rounds (turns that requested ≥1 tool call). Token usage is summed across
    every turn so the caller can price the question.
    """
    by_name = {t.name: t for t in tools}
    usage = Usage()
    tool_calls: list[str] = []
    iterations = 0
    responses: list[ToolResponse] | None = None
    answer = Answer(answer=ABSTAIN, confidence=0.0)
    stop_reason = "max_turns"
    error: str | None = None
    start = clock()

    for _turn in range(max_turns):
        if clock() - start > timeout_s:
            stop_reason = "timeout"
            error = f"timeout_s={timeout_s} exceeded"
            break
        try:
            step = model.next(responses)
        except Exception as e:  # noqa: BLE001 — a transport failure ends the question, not the batch
            stop_reason = "model_error"
            error = f"model error: {type(e).__name__}: {e}"
            break
        usage = usage + step.usage

        if not step.function_calls:
            # The model answered in free text without the terminal tool: accept it, but we have no
            # self-reported confidence, so record 0.0 and note the shortcut in ``method``.
            text = (step.final_text or "").strip() or ABSTAIN
            answer = Answer(answer=text, confidence=0.0, method="(submit_answer未使用: 最終テキストを採用)")
            stop_reason = "answered"
            break

        responses = []
        submitted = False
        dispatched_tool = False
        for call in step.function_calls:
            tool_calls.append(call.name)
            if call.name == SUBMIT_ANSWER:
                answer = _answer_from_args(call.args)
                stop_reason = "answered"
                submitted = True
                break
            out = dispatch(by_name, call.name, call.args)
            responses.append(ToolResponse(call.name, out))
            dispatched_tool = True
        if dispatched_tool:
            # count only genuine tool rounds; the terminal submit_answer turn is not a round
            iterations += 1
        if submitted:
            break
    else:
        error = error or f"max_turns={max_turns} reached without a final answer"

    model_name = getattr(model, "model_name", settings.GEN_MODEL_HARD)
    return Investigation(
        question=question, answer=answer, iterations=iterations, tool_calls=tool_calls,
        usage=usage, model=model_name, elapsed_s=max(0.0, clock() - start),
        stop_reason=stop_reason, error=error,
    )


def investigate_batch(model_factory: Callable[[str, Sequence[AgentTool]], Model],
                      questions: Sequence[str], *,
                      profile_factory: Callable[[], CorpusProfile] | None = None,
                      max_turns: int = DEFAULT_MAX_TURNS,
                      timeout_s: float = DEFAULT_TIMEOUT_S) -> list[Investigation]:
    """Investigate each question with a fresh model + tools + profile; return one result per question.

    Each question gets its own :class:`CorpusProfile` so a self-discovered secret never leaks the answer
    across questions (移植性の担保).
    """
    prof = profile_factory or (lambda: CorpusProfile())
    out: list[Investigation] = []
    for q in questions:
        tools = build_tools(prof())
        model = model_factory(q, tools)
        out.append(investigate(model, q, tools, max_turns=max_turns, timeout_s=timeout_s))
    return out


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


def gemini_model_factory(question: str, tools: Sequence[AgentTool], *,
                         model: str | None = None) -> GeminiModel:
    """Fresh live conversation for one question (each question gets an isolated context)."""
    return GeminiModel(question, to_genai_tools(tools), model=model)


def answer_question(question: str, *, model: str | None = None,
                    profile: CorpusProfile | None = None,
                    max_turns: int = DEFAULT_MAX_TURNS,
                    timeout_s: float = DEFAULT_TIMEOUT_S) -> Investigation:
    """Convenience live entry point: investigate one ``question`` with a real Gemini conversation."""
    tools = build_tools(profile or CorpusProfile())
    model_obj = gemini_model_factory(question, tools, model=model)
    return investigate(model_obj, question, tools, max_turns=max_turns, timeout_s=timeout_s)
