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

import os
import re
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Mapping, Protocol, Sequence

from config import settings
from src.rag.tools import contract as _contract
from src.rag.tools.canonical_route import canonical_route
from src.rag.tools.chart_numcache import read_chart_values
from src.rag.tools.compute_sandbox import run as compute_run
from src.rag.tools.corpus_aggregate import corpus_aggregate
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
DIRECTIVE_MESSAGE = "__instruction__"  # plain user guidance, never a synthetic function response


def _bool_env(key: str, default: bool) -> bool:
    raw = os.getenv(key)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


# SOT-2521 — whether the convenience :func:`answer_question` entry point wires the loop-side
# deterministic first move (see :func:`investigate`'s ``first_move``). **Default OFF so the production
# answer path stays byte-identical.** The always-on variant regressed the single gold-100 measurement
# (match 21→19, wrong 6→9 vs the SOT-2510 baseline): seeding canonical evidence and telling the model to
# stop searching nudged premature *wrong* commits on questions the model would otherwise have kept
# investigating. So the mechanism ships behind this flag, OFF, pending a *narrowed* enablement whose
# net effect is measured by the single integrated SOT-2527 gold-100 run — mirroring the
# ``GATE_SLICE_CALIBRATE`` / ``GATE_COMMIT_ON_TIEBREAK`` default-OFF precedent (mechanism landed,
# relaxation deferred to a real non-regression confirmation). The ``investigate(first_move=…)`` parameter
# itself is unchanged; only this entry point's default wiring is gated.
FIRST_MOVE_ROUTING = _bool_env("RAG_FIRST_MOVE_ROUTING", False)

# SOT-2525 — whether :func:`answer_question` wires the loop-side deterministic tool fallback tried before
# an UNANSWERABLE abstain is accepted (see :func:`investigate`'s ``fallback``). **Default OFF so the
# production answer path stays byte-identical** (mirroring the ``RAG_FIRST_MOVE_ROUTING`` default-OFF
# precedent). When on, a question about to abstain forces one contract-typed deterministic tool
# (canonical_route / version_diff / file_grep, each self-resolving from the question) and only feeds real
# evidence back — never an answer. Its net effect on gold-100 is measured by the single integrated
# SOT-2527 run before any default flip.
UNANSWERABLE_FALLBACK = _bool_env("RAG_UNANSWERABLE_FALLBACK", False)

# SOT-2524 — whether :func:`answer_question` wires the budget-exhaustion boundary hook (see
# :func:`investigate`'s ``budget_boundary``). The obligation-driven local re-search director (SOT-2502)
# already fires on a *deliberate* abstain, but the dominant BUDGET_EXHAUSTED cause is the non-deliberate
# boundary — the model wanders through tools until ``max_turns``/``timeout_s`` is reached without ever
# committing or abstaining, so the director never runs. This hook gives it a bounded last push at the
# still-unmet obligations *at that boundary*, before the abstain is finalized. **Default ON**: it only
# extends the already-default-ON director to where most BUDGET abstains actually land, and it is EV-safe
# (the commit threshold is untouched — it only turns a would-be BUDGET abstain into either a grounded
# answer or a coded, history-bearing abstain). Set ``RAG_BUDGET_BOUNDARY_RESEARCH=0`` to restore the
# pre-hook boundary (abstain finalized immediately). Bounded by the director's ``max_rounds`` and by the
# existing ``timeout_s`` (a timeout-triggered boundary adds no model turns — it only records terminal).
BUDGET_BOUNDARY_RESEARCH = _bool_env("RAG_BUDGET_BOUNDARY_RESEARCH", True)

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
    "4b. 案件名+データ資産(train.xlsx/train.csv・分析コード/modeling.py・notebook/EDA・leaderboard・"
    "スケジュール等)を指すデータ/計算系の質問で、grep/検索で根拠ファイルが見つからないときは canonical_route "
    "ツールに質問文をそのまま渡す。用語集で案件を特定し canonical ファイルを直行解決して返すので(chunk検索を"
    "迂回)、返った先頭 rel を compute/read_office/read_chart_values に渡すか、canonical_route(question, "
    "expr='…') で計算値まで得る。例: 『京橋信用ソリューションズの分析コードの n_estimators』は"
    "canonical_route(question=…, kind='code') で modeling.py を得て read_office で読む。\n"
    "5. 旧版(old版)と最新版の比較・変更点を問う質問は、grepで手作業比較せず version_diff ツールに質問文を"
    "そのまま渡す。決定論の構造diffが『変更前 → 変更後』を返すので、その value をそのまま回答にする。value が"
    "null のときのみ他手段(grep等)を検討する。\n"
    "5b. PDFの検索・抽出結果が空で、ページが画像だけの場合は caption_image(file=..., question=質問全文) を"
    "使う。引用語や『明記』の質問では、条件語と候補が同じ原文行にある vision 出力だけを根拠にする。\n"
    "6a. 内線番号/EXT/座席/『向かい・隣・同じ列・Xから見て右側/左側』を問う質問は seating_lookup "
    "ツールを使う(座席表は画像1枚で"
    "grep/office抽出では読めない)。多段(案件→担当者→内線)では先に担当者の氏名を他ツールで特定し、その氏名を"
    "seating_lookup(name=…)に渡す。『Aさんの向かいの人のEXT』は seating_lookup(name='A', relation='向かい')。"
    "『Aさんから見て右側の人の名前をすべて』は seating_lookup(name='A', relation='右側', field='name')。\n"
    "6b. 『全体で/横断で/全案件で/最も〜な案件・人』のように複数案件をまたいで集計・比較する質問は"
    "corpus_aggregate ツールを使う(単一案件の compute では同名ファイルが複数で解けない)。例: 『最も多く案件に"
    "関わる人』=corpus_aggregate(metric='staff', op='count') の top、『着手金が最も高い案件』="
    "corpus_aggregate(metric='deposit', op='max')、『固定金額契約で1行あたり契約金額が最も高い案件』="
    "corpus_aggregate(metric='amount_per_row', op='max', fixed_only=true, round_up=true)、『契約期間が"
    "2025-08-15〜09-07 に重なり40日超の案件を主略称で』=corpus_aggregate(metric='period_days', op='filter', "
    "overlap_start='2025-08-15', overlap_end='2025-09-07', min_days=40)、『全案件のPP・契約書・PLAN・FRに"
    "役割付きで記載されたDA人物のユニオン人数』=corpus_aggregate(metric='staff_population', op='count')。"
    "案件特定後の多段(その案件の担当者→"
    "内線 等)は返り値 staff(ES/PM…)の氏名を seating_lookup に渡して解決する。\n"
    "6c. グラフの数値を問う場合は read_chart_values を必ず使う。系列/列名を column に、ヒストグラムの"
    "最多カウントなら operation='histogram_max_count' を渡す。numCacheが無い画像グラフも元データ列から"
    "決定論的に再集計される。caption_image はグラフの所在・軸ラベル確認にだけ使い、その画像説明中の数値を"
    "answer/evidence/methodへ採用してはならない。read_chart_values が値を返せなければ推測せず棄権する。\n"
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


def _reference_commit_rejection(question: str, candidate: Answer, contract: str | None, *,
                                tool_outputs: Sequence[Any],
                                version_diff_result: Mapping[str, Any] | None) -> dict[str, Any] | None:
    """Return a structured rejection when deterministic reference evidence is still incomplete."""
    if contract == "version_diff":
        if version_diff_result is None:
            return {
                "answer_rejected": True,
                "reason": "版差分契約で必須の version_diff ツールが未実行です。",
                "directive": (
                    "回答を確定せず version_diff(question=質問全文) を実行してください。"
                    "grep/目視による一部比較だけで回答してはなりません。"),
            }
        resolved = version_diff_result.get("value")
        if resolved is None:
            if is_abstain(candidate.answer):
                return None
            return {
                "answer_rejected": True,
                "reason": "全スライド/全シートの決定論的版差分が解決できていません。",
                "directive": "別の変更を推測せず、追加の版特定ができなければ棄権してください。",
            }
        if is_abstain(candidate.answer):
            return {
                "answer_rejected": True,
                "reason": "version_diff は全版差分を解決済みのため棄権できません。",
                "expected": str(resolved),
                "directive": "version_diff が返した value を省略・言い換えせずそのまま回答してください。",
            }
        if candidate.answer.strip() != str(resolved).strip():
            return {
                "answer_rejected": True,
                "reason": "回答が version_diff の決定論的 value と一致しません。",
                "expected": str(resolved),
                "directive": "version_diff が返した value を省略・言い換えせずそのまま回答してください。",
            }

    if is_abstain(candidate.answer):
        return None

    from src.rag.agent import obligations as _obligations

    literal_check = _obligations.validate_literal_evidence(
        question, candidate.answer, tool_outputs)
    if not literal_check.passed:
        return {
            "answer_rejected": True,
            "reason": "literal 一致の証拠義務が未充足です。",
            "issues": list(literal_check.issues),
            "directive": (
                "引用条件語が候補と同じセル/文に文字列として存在する箇所をツールで確認してください。"
                "共起しない候補は採用せず、確認不能なら棄権してください。"),
        }
    return None


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
    contract: str | None = None  # SOT-2498 — routing contract this question was classified as (or None)
    calc_record: dict[str, Any] | None = None  # SOT-2506 — in-memory record for the execution gate

    @property
    def confidence(self) -> float:
        return self.answer.confidence

    def to_dict(self) -> dict[str, Any]:
        return {
            "question": self.question,
            **self.answer.to_dict(),
            "contract": self.contract,
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
_INT = {"type": "integer"}


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
        evidence={"applicable": True, "resolved": answer is not None,
                  "coverage": "all-slides/all-sheets"},
        scheme="structural-version-diff",
        note=("隣接版(旧版→最新/vN→vN+1)の全スライド/全シート構造diff"
              "(セル/段落を整列し実質変更のみ)"
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
            "コーパス全体を全文/セル/ファイル名でgrepし、一致箇所(ファイル・行・抜粋)を返す。"
            "逆引き索引が有効な場合は所在(ファイル・シート・セル/段落)を即答し、未ヒット時のみ全走査にフォールバックする。",
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
            "canonical_route",
            "データ/計算系の質問を、案件(project)の canonical データファイル(train.xlsx/train.csv・"
            "分析コード modeling.py・notebook 01_eda.ipynb・leaderboard.csv 等)へ直行解決するツール。"
            "chunk検索で根拠が top-k に上がらない(索引で拾えない)データ質問の迂回ルート。question に質問文を"
            "そのまま渡すと、用語集で案件を特定し、質問の種別(train/code/notebook/leaderboard/schedule/"
            "contract)に対応する canonical ファイルを決定論で発見して {rel, project, category, ext, kind} の"
            "一覧(canonical本命が先頭)を返す。project(会社名の一部)/kind を明示して絞ることも可。返った先頭 rel を"
            "compute/read_office/read_chart_values に渡す。expr(単一pandas式)を渡すと先頭の表ファイルに対して"
            "その場で compute を実行し計算値まで返す。『京橋の分析コードの n_estimators』『〜のtrain.xlsxの…』の"
            "ように案件名+データ資産を指す質問に使う。該当なしは value=[](棄権のまま)。",
            _obj({"question": _STR, "project": _STR, "kind": _STR, "expr": _STR, "sheet": _STR},
                 ["question"]),
            lambda question, project=None, kind=None, expr=None, sheet=None: canonical_route(
                question, project=project, kind=kind, expr=expr, sheet=sheet),
        ),
        AgentTool(
            "read_chart_values",
            "xlsx/pptx埋め込みチャートの数値を厳密に読む。numCacheを最優先し、画像化されたxlsxヒストグラムは"
            "columnで指定した元データ列から再集計する。最多カウントはoperation='histogram_max_count'。"
            "vision数値は使用せず、厳密経路が無ければエラー=棄権。",
            _obj({"file": _STR, "column": _STR, "operation": _STR}, ["file"]),
            lambda file, column=None, operation=None: read_chart_values(
                file, column=column, operation=operation),
        ),
        AgentTool(
            "caption_image",
            "図表PNGを説明し、または画像のみのPDF全ページをvisionモデルで質問別に厳密転記する。"
            "PDFでは question に元の質問全文を渡す。引用語/『明記』は条件語と候補を同一原文行で返す。"
            "チャートでは所在・系列名・軸ラベル確認専用で、説明中の数値を回答根拠に使用してはならない。"
            "数値はread_chart_valuesで取得する。",
            _obj({"file": _STR, "question": _STR}, ["file"]),
            lambda file, question=None: caption_figure(file, question=question),
        ),
        AgentTool(
            "pdf_emphasis",
            "PDF内の強調(疑似イタリック=行列シアー)された単語を抽出する。",
            _obj({"file": _STR}, ["file"]),
            lambda file: emphasized_words(file),
        ),
        AgentTool(
            "pptx_pivot",
            "pptxに埋め込まれたPivotTable(EMF)を復元し、表とハイライトされたセルを返す。"
            "同一案件の元データと全表示セルを照合して semantics(行フィールド/列フィールド/対象列/集計方法)を"
            "意味解決し、各セルに filters・target_column・aggregation_label・semantic_summary を付ける。"
            "ピボット回答は生ラベルや値だけでなく semantic_summary の抽出条件+対象列+集計方法を必ず含める。",
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
            "value=null を返す(その場合は棄権のまま)。版差分契約では回答確定前の実行が必須で、返された"
            "value をそのまま回答する。",
            _obj({"question": _STR}, ["question"]),
            lambda question: _version_diff(question),
        ),
        AgentTool(
            "seating_lookup",
            "座席表(フロアマップ)から 氏名⇄内線(EXT)⇄座席 を引く。内線/EXT/座席/『向かい・隣・同じ列・"
            "Xから見て右側/左側』を"
            "問う質問に使う(座席図は画像1枚で他ツールでは読めない)。name=氏名(『〜さん』可)を渡すとその人の"
            "EXTを返す。relation に『向かい/隣/同じ列/同じ行』を渡すとその隣人のEXTを返す(例: 井上さんの"
            "向かいの人のEXT)。右側/左側は着席向き基準の複数候補を返す。名前が必要なら field='name'、"
            "座席レコードなら field='seat'、既定は field='ext'。ext=内線から人物を、"
            "role=役割(Exec/PM/DS/BA/DE/QA)+pod で EXT を引くこともできる。"
            "多段質問(案件→担当者→内線)では、先に担当者の氏名を他ツールで特定し、その氏名を name で渡す。"
            "該当なし/曖昧なときは value=null(棄権)を返す。",
            _obj({"name": _STR, "ext": _STR, "role": _STR, "relation": _STR, "pod": _NUM,
                  "field": _STR}),
            lambda name=None, ext=None, role=None, relation=None, pod=None, field="ext": seating_lookup(
                name=name, ext=ext, role=role, relation=relation,
                pod=int(pod) if isinstance(pod, (int, float)) else None, field=field),
        ),
        AgentTool(
            "corpus_aggregate",
            "全プロジェクトを横断して契約情報を集約する決定論ツール。単一案件のcompute/read_officeでは"
            "解けない『全体で/横断で/最も〜な案件・人』を扱う(train.xlsx/契約書は案件ごとに同名複数のため)。"
            "metric: contract_amount(契約金額税込)/deposit(着手金税込)/train_rows(学習データ行数)/"
            "amount_per_row(契約金額税込÷train行数)/period_days(契約期間日数)/staff(契約書の乙担当者)/"
            "staff_population(PP・契約書・PLAN・FR×全案件の役割付きDA人物union)。"
            "op: max/min(数値metricの極値案件を主略称abbrev+valueで返す。staff情報も同梱するので『最大案件のES』は"
            "その staff.ES を seating_lookup(name=…) に渡す)/count(staffの案件横断出現回数→最頻top=『最も多く"
            "案件に関わる人』、staff_population は count で閉包済み人数+人物union)/filter(契約期間フィルタ)/list。"
            "固定金額契約に絞るときは fixed_only=true。円単位で切り上げるときは round_up=true。"
            "契約期間フィルタは overlap_start/overlap_end(YYYY-MM-DD)で重なる案件、min_days でその日数超"
            "(『40日を超える』→min_days=40)に絞り、主略称の配列を返す。値は決定論(推測なし)、"
            "該当なしは value=null(棄権)。",
            _obj({"metric": _STR, "op": _STR, "fixed_only": _BOOL, "round_up": _BOOL,
                  "overlap_start": _STR, "overlap_end": _STR, "min_days": _INT}, ["metric"]),
            lambda metric, op="max", fixed_only=False, round_up=False,
            overlap_start=None, overlap_end=None, min_days=None: corpus_aggregate(
                metric, op=op, fixed_only=bool(fixed_only), round_up=bool(round_up),
                overlap_start=overlap_start, overlap_end=overlap_end,
                min_days=int(min_days) if isinstance(min_days, (int, float)) else None),
        ),
    ]


def build_tools(profile: CorpusProfile) -> list[AgentTool]:
    """The full tool set exposed to the investigator: generic Step1 tools + the terminal answer tool."""
    return [*build_generic_tools(profile), SUBMIT_ANSWER_TOOL]


def _jsonable(obj: Any, *, max_str: int = 8000, max_items: int = 60, _depth: int = 0) -> Any:
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


def _first_move_useful(out: Any) -> bool:
    """Whether a deterministic first-move tool actually reached evidence worth seeding (SOT-2521).

    A first move is only worth seeding when it resolved a concrete target: a ``{value, …}`` contract
    with a non-empty ``value`` (canonical_route records / a computed value), or a plain non-empty
    list. An error mapping, an empty ``value`` (no project/kind resolved), or an empty list means the
    deterministic route did not reach the needle — leave the first turn model-driven instead of
    seeding a misleading empty result.
    """
    if _contract.is_contract(out):
        value = _contract.ensure_contract(out).get("value")
        return value not in (None, "", [], {})
    if isinstance(out, Mapping):
        return "error" not in out and bool(out)
    if isinstance(out, (list, tuple)):
        return bool(out)
    return out not in (None, "")


def _budget_boundary_directive(director: Any, enabled: bool, tool_calls: Sequence[str],
                               answer: "Answer") -> str | None:
    """SOT-2524 — the re-search directive to emit at the max_turns budget boundary, or ``None`` to finalize.

    Returns ``None`` (finalize the abstain unchanged) when the boundary hook is disabled, no re-search
    director is active, or the pending answer is already a commit. Otherwise it asks the director — using
    the evidence it has already folded in via :meth:`ResearchDirector.observe` over the run's tool
    results — for the next targeted directive; the director returns one (keep searching) or ``None`` after
    recording its terminal (:data:`BUDGET`/:data:`UNANSWERABLE`) for the abstain ledger. It never inspects
    or changes the commit threshold — it only grows *search* at the boundary.
    """
    if not (enabled and director is not None and is_abstain(answer.answer)):
        return None
    evidence = " ".join(t for t in (answer.evidence, answer.method) if t).strip()
    non_submit = sum(1 for c in tool_calls if c != SUBMIT_ANSWER)
    return director.review(evidence, non_submit, at_boundary=True)


def _first_move_directive(name: str, out: Any) -> str:
    """The seed user-message describing a deterministic first-move result (SOT-2521).

    Carries the tool's own output (the same evidence the model would receive had it called the tool
    itself) and tells the model to build on it — compute / extract — rather than re-run search or the
    route from scratch. Injects no answer and no policy: the committed answer is still the model's.
    """
    import json

    try:
        payload = json.dumps(out, ensure_ascii=False, default=str)[:6000]
    except (TypeError, ValueError):
        payload = str(out)[:6000]
    return (
        f"初手として決定論ツール『{name}』を質問に対して実行しました"
        "（探索予算を canonical 証拠へ直結させるため、chunk 検索は省略済み）。\n"
        f"実行結果(JSON): {payload}\n"
        "この結果の primary/records を起点に、canonical_route の再実行や一律の chunk 検索を先に行わず、"
        "必要なら compute / read_office / read_chart_values で計算・抽出して回答してください。"
        "この初手で対象が得られていない場合や、より適切な根拠が別にある場合のみ、他ツールへ切り替えてください。")


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


def _has_deterministic_gantt_evidence(out: Any) -> bool:
    """Whether ``read_office`` returned at least one resolved native-shape Gantt span."""
    if not _contract.is_contract(out):
        return False
    value = str(out.get("value") or "")
    return (
        (out.get("method") or {}).get("engine") == "pptx"
        and "【ガント週グリッド:決定論】" in value
        and bool(re.search(r"\]\s*[^\n]+:\s*第\d+週目から第\d+週目", value))
    )


def _normalized_activity(text: str) -> str:
    """Normalize an activity label for question↔native-shape matching."""
    from src.rag.corpus import nfc

    return re.sub(r"[^0-9A-Za-z\u3040-\u30ff\u3400-\u9fff]+", "", nfc(text)).replace("の", "")


def _closure_satisfied(evidence: Mapping[str, Any]) -> bool:
    """Validate the four machine-readable closure conditions shared by enum/count fast paths."""
    closure = evidence.get("closure")
    return bool(
        isinstance(closure, Mapping)
        and closure.get("authoritative_population_resolved")
        and closure.get("inclusion_exclusion_recorded")
        and not closure.get("second_path_novel_candidates")
        and closure.get("enumeration_count") == closure.get("aggregate_count")
    )


def _deterministic_seating_side_answer(question: str) -> Answer | None:
    """Resolve occupant-relative left/right name enumerations from the hash-pinned seat frame."""
    if not (re.search(r"(?:右側|右手|左側|左手)", question)
            and re.search(r"名前|人物|人", question)
            and re.search(r"すべて|全て|全部", question)):
        return None
    subject = re.search(r"([一-龯々]{1,6})(?:さん|氏|様)?から見て", question)
    relation = re.search(r"(右側|右手|左側|左手)", question)
    if not subject or not relation:
        return None
    result = seating_lookup(name=subject.group(1), relation=relation.group(1), field="name")
    if not _contract.is_contract(result) or not _closure_satisfied(result.get("evidence") or {}):
        return None
    values = result.get("value")
    if not isinstance(values, list) or not values or not all(isinstance(v, str) for v in values):
        return None
    evidence = result["evidence"]
    return Answer(
        answer="、".join(values), confidence=1.0,
        evidence=(f"{evidence.get('file')} pixel_sha={evidence.get('pixel_sha')} / "
                  f"subject={subject.group(1)} / relation={relation.group(1)} / "
                  f"closure={len(values)}件一致"),
        method="pixel-hash pin済み座席アンカーを着席者の向きフレームへ変換し、右左半平面を完全列挙",
    )


def _deterministic_staff_population_answer(question: str) -> Answer | None:
    """Count the closed union of role-bound DA people across canonical PP/contract/PLAN/FR files."""
    from src.rag.agent import question_contract as _question_contract

    if not _question_contract.is_staff_population_question(question):
        return None
    result = corpus_aggregate("staff_population", op="count")
    if not _contract.is_contract(result) or not _closure_satisfied(result.get("evidence") or {}):
        return None
    value = result.get("value")
    if not isinstance(value, Mapping) or not isinstance(value.get("count"), int):
        return None
    evidence = result["evidence"]
    return Answer(
        answer=str(value["count"]), confidence=1.0,
        evidence=(f"PP/契約書/PLAN/FR canonical files={len(evidence.get('selected_files') or [])}; "
                  f"projects={len(evidence.get('projects') or [])}; union={value['count']}"),
        method="全案件×4文書型の正本列挙→役割付きDA人物抽出→表記正規化・重複解決→列挙件数と集計件数照合",
    )


def _deterministic_gantt_answer(question: str, profile: CorpusProfile) -> Answer | None:
    """Resolve a uniquely named PPTX activity from native Gantt geometry, without an LLM."""
    route = canonical_route(question)
    project = (route.get("evidence") or {}).get("project") if _contract.is_contract(route) else None
    filename_match = re.search(r"([^/\\、。\sの]+\.pptx)", question, re.I)
    target_match = re.search(r"(?:において[、,])(.+?)(?:の)?実行予定スケジュール", question)
    if not project or not filename_match or not target_match:
        return None
    found = find_files(filename_match.group(1), ext="pptx", project=str(project))
    files = found.get("value") if _contract.is_contract(found) else None
    if not isinstance(files, list) or len(files) != 1:
        return None
    office = extract_office(str(files[0]["rel"]), profile=profile)
    if not _has_deterministic_gantt_evidence(office):
        return None
    target = _normalized_activity(target_match.group(1))
    matches = []
    for match in re.finditer(
            r"^\[スライド(?P<slide>\d+)\]\s*(?P<activity>.+?):\s*"
            r"第(?P<start>\d+)週目から第(?P<end>\d+)週目", str(office["value"]), re.M):
        activity = _normalized_activity(match.group("activity"))
        if activity == target:
            matches.append(match)
    if len(matches) != 1:
        return None
    match = matches[0]
    return Answer(
        answer=f"第{match.group('start')}週目から第{match.group('end')}週目",
        confidence=1.0,
        evidence=f"{files[0]['rel']} スライド{match.group('slide')}",
        method="週ヘッダx座標とバーleft/widthの半開区間重なりによる決定論抽出",
    )


def _deterministic_regulation_answer(question: str, profile: CorpusProfile) -> Answer | None:
    """Render a complete fallback rule when the special condition is absent and all fields are explicit."""
    from src.rag.corpus import nfc

    route = canonical_route(question)
    project = (route.get("evidence") or {}).get("project") if _contract.is_contract(route) else None
    if not project:
        return None
    found = find_files("契約", ext="docx", project=str(project))
    files = found.get("value") if _contract.is_contract(found) else None
    if not isinstance(files, list) or len(files) != 1:
        return None
    office = extract_office(str(files[0]["rel"]), profile=profile)
    text = nfc(str(office.get("value") or ""))

    # Only conclude that no special clause exists when the question has distinctive threshold tokens
    # and none occurs in the authoritative contract.  Otherwise leave the judgment to normal research.
    condition_tokens = re.findall(r"[A-Za-z][A-Za-z0-9_]{1,}|\d+(?:\.\d+)?\s*(?:時間|件|円|%)", question)
    if not condition_tokens or any(nfc(token) in text for token in condition_tokens):
        return None

    rate = re.search(r"時間単価は\s*([\d,]+円)\s*[（(](?:消費税別|税別)[）)]", text)
    unit = re.search(r"計上単位は\s*(\d+分)", text)
    cycle = re.search(r"(月次|週次|日次)(?:で)?(?:甲に提出|精算)", text)
    tax = "消費税を加算" in text
    rounded = bool(re.search(r"端数は\s*\d+分単位に切り上げ", text))
    uncapped = "契約総額を固定するものではない" in text
    if not (rate and unit and cycle and tax and rounded and uncapped):
        return None
    subject_match = re.search(r"(?:において[、,])(.+?)の精算方法", question)
    subject = subject_match.group(1) if subject_match else "質問の条件"
    answer = (
        f"{subject}の特別な精算規定はなく、実績工数に時間単価{rate.group(1)}(税別)を乗じ"
        f"消費税を加算して{cycle.group(1)}で精算する({unit.group(1)}単位・切上げ、上限なし)。")
    return Answer(
        answer=answer,
        confidence=1.0,
        evidence=f"{files[0]['rel']} 報酬および支払条件",
        method="特別条件不在確認後、一般規定の単価・税・単位・丸め・周期・上限を機械抽出",
    )


# --------------------------------------------------------------------------- agent loop (pure)
class Model(Protocol):
    """A model conversation for one question. ``next(None)`` starts it; ``next(responses)`` continues
    it after tool calls. Implementations hold their own history; the loop stays transport-agnostic."""

    def next(self, tool_responses: Sequence[ToolResponse] | None) -> Step: ...


def investigate(model: Model, question: str, tools: Sequence[AgentTool], *,
                max_turns: int = DEFAULT_MAX_TURNS, timeout_s: float = DEFAULT_TIMEOUT_S,
                clock: Callable[[], float] = time.monotonic,
                ledger: "str | object | bool | None" = None,
                calc_ledger: "str | object | bool | None" = None,
                research: "bool | Mapping[str, Any] | object | None" = None,
                enumeration: "bool | Mapping[str, Any] | object | None" = None,
                contract: "str | None" = None,
                first_move: "tuple[str, Mapping[str, Any]] | None" = None,
                fallback: "object | None" = None,
                budget_boundary: bool = False) -> Investigation:
    """Drive ``model`` through tool-calling until it submits a structured answer.

    The loop ends on the first of: the model calls ``submit_answer`` (→ ``answered``); ``max_turns`` is
    reached (→ ``max_turns``, abstain); the wall-clock ``timeout_s`` is exceeded between turns
    (→ ``timeout``, abstain); or the transport raises (→ ``model_error``, abstain). A plain final-text
    turn (no ``submit_answer``) is accepted as the answer with confidence 0.0.

    ``iterations`` counts tool rounds (turns that requested ≥1 tool call). Token usage is summed across
    every turn so the caller can price the question.

    ``ledger`` (SOT-2492) enables the abstain ledger: when it is not ``None``/``False``, per-tool
    outcome signals are *observed* (never altered) during the loop, and if the final answer is an
    abstain a coded :class:`~src.rag.agent.abstain_ledger.AbstainRecord` is appended. ``True`` writes to
    the default path (``artifacts/abstain_ledger.jsonl``); a ``str``/``Path`` writes there instead. The
    default ``None`` is a pure no-op so the commit-vs-abstain decision and every existing caller are
    unchanged (受け入れ条件②).

    ``calc_ledger`` (SOT-2495/SOT-2506) is the commit-side twin: when enabled, derivation-tool contracts
    are *observed* (never altered) during the loop, and if the final answer is numeric—or is a label
    selected by a ``numeric`` contract—a typed :class:`~src.rag.agent.calc_ledger.CalcRecord` (raw_text /
    parsed_value / unit / source_range / formula + per-compute証跡) is appended (``True`` →
    ``artifacts/calc_ledger.jsonl``; a ``str``/``Path`` redirects it). Same observer-only invariant: the
    answer path is byte-identical with it on or off.

    ``research`` (SOT-2502) enables the obligation-driven local re-search loop: when it is not
    ``None``/``False``, a deliberate ``submit_answer`` abstain is not accepted immediately — the
    :class:`~src.rag.agent.research_loop.ResearchDirector` discharges the question's evidence obligations
    against what the tools actually found and, while budget remains, feeds a *targeted* re-search
    directive (unmet obligation + per-kind tactics) back to the model so it re-searches only the unmet
    obligation before it may abstain again. A ``Mapping`` supplies the budget (``max_rounds`` /
    ``max_tool_calls``). The commit threshold/confidence are never touched (SOT-2483 の軸とは別物): a
    re-search either reaches a grounded answer or yields a *coded, history-bearing* abstain
    (:data:`~src.rag.agent.abstain_ledger.BUDGET_EXHAUSTED` / ``UNANSWERABLE``). Off by default so the
    answer path stays byte-identical.

    ``enumeration`` (SOT-2500) enables the full-enumeration closure protocol: when it is not
    ``None``/``False`` and the question is a ``full_enumeration`` contract, a deliberate abstain is
    intercepted **once** and the :class:`~src.rag.agent.enumeration.EnumerationGate` feeds the closure
    *procedure* back to the model (権威的母集団の特定 → 各候補の包含/除外理由 → 別経路で新規候補ゼロ →
    列挙件数と集計件数の一致 → 列挙順序規則) so completeness is *proven* before the model may abstain again.
    The procedure injects no corpus fact (no 略称/member list — only the source category to read), so the
    no-fact-injection invariant holds. It only ever turns an enumeration abstain into either a
    closure-proven answer or a still-coded abstain — the commit threshold is never touched. Off by default
    so the answer path stays byte-identical; it composes with ``research`` (the enum procedure is tried
    first, the generic re-search after).

    ``contract`` (SOT-2498) is a pure *label* recorded onto the returned :class:`Investigation` (and, on
    an abstain, into the ledger) so the routing contract the question was classified as is captured
    per-question alongside the tools actually used. It never influences the loop — the routing *hint*
    itself is injected into the model's system prompt at construction (see :func:`answer_question`), not
    here — so passing it leaves the answer path byte-identical (default ``None``).

    ``first_move`` (SOT-2521) is an optional ``(tool_name, kwargs)`` plan the loop runs *deterministically
    before the model's first turn*, seeding the model with the tool's evidence (as a plain user directive)
    so it stops burning its turn budget wandering through chunk search on questions whose evidence is a
    canonical data-asset needle (BUDGET_EXHAUSTED, the top abstain cause). It only seeds when the tool
    actually reaches evidence (:func:`_first_move_useful`); an empty/error result falls back to the
    ordinary model-driven first turn. The seed steers the *first* move only — the model's later tool
    choice and the commit-vs-abstain decision are untouched — and injects no answer, so the answer path
    stays byte-identical when it is ``None`` (default). The plan is built by
    :func:`~src.rag.agent.routing.deterministic_first_move` in :func:`answer_question`.

    ``fallback`` (SOT-2525) is an optional one-shot gate
    (:class:`~src.rag.agent.question_contract.DeterministicFallbackGate`) tried when the model is about to
    abstain and neither the enumeration-closure nor the obligation re-search loop intervened. It forces
    the question's *contract-typed deterministic tool* (canonical_route / version_diff / file_grep — each
    self-resolving from the question text) to run exactly once before UNANSWERABLE is accepted, and — only
    when that tool actually reaches evidence (:func:`_first_move_useful`) — feeds the evidence back so the
    model may answer from it (subject to every existing commit guard, so no answer is injected). When the
    tool reaches nothing, the abstain stands unchanged (従来の安全動作を維持). It is one-shot so a still-
    abstaining model cannot loop, and injects no corpus fact, so the answer path is byte-identical when it
    is ``None`` (default). The gate is built in :func:`answer_question` behind ``RAG_UNANSWERABLE_FALLBACK``.

    ``budget_boundary`` (SOT-2524) enables the budget-exhaustion boundary hook, effective only together
    with ``research``. Without it the re-search director fires solely on a *deliberate* abstain; with it,
    when the loop is about to finalize a non-committed abstain because ``max_turns`` (or ``timeout_s``)
    was reached, the director gets a bounded last push at the still-unmet obligations: it emits one
    targeted re-search directive and up to ``budget.max_rounds`` extra model turns are granted to answer
    it — still inside ``timeout_s`` (so a timeout-triggered boundary grants no turns and only records the
    director's terminal). The commit threshold is untouched, so it only turns a would-be BUDGET abstain
    into either a grounded answer or a coded, history-bearing abstain. ``False`` by default so existing
    callers are byte-identical; :func:`answer_question` wires it from ``RAG_BUDGET_BOUNDARY_RESEARCH``.
    """
    record_enabled = ledger not in (None, False)
    if record_enabled:
        from src.rag.agent import abstain_ledger as _abstain_ledger
        signals = _abstain_ledger.AbstainSignals()
    else:
        _abstain_ledger = None
        signals = None

    calc_enabled = calc_ledger not in (None, False)
    if calc_enabled:
        from src.rag.agent import calc_ledger as _calc_ledger
        calc_signals = _calc_ledger.CalcSignals(question=question)
    else:
        _calc_ledger = None
        calc_signals = None

    research_enabled = research not in (None, False)
    if research_enabled:
        from src.rag.agent import research_loop as _research_loop
        director = _research_loop.ResearchDirector(question, budget=research)
    else:
        director = None

    enumeration_enabled = enumeration not in (None, False)
    if enumeration_enabled:
        from src.rag.agent import enumeration as _enumeration
        enum_gate = _enumeration.EnumerationGate(question)
    else:
        enum_gate = None

    by_name = {t.name: t for t in tools}
    usage = Usage()
    tool_calls: list[str] = []
    iterations = 0
    responses: list[ToolResponse] | None = None
    answer = Answer(answer=ABSTAIN, confidence=0.0)
    stop_reason = "max_turns"
    error: str | None = None
    strict_chart_evidence = False
    chart_guidance_sent = False
    simple_lookup_guidance_sent = False
    regulation_guidance_sent = False
    from src.rag.agent import question_contract as _question_contract
    gantt_question = contract == "chart_read" and _question_contract.is_gantt_week_question(question)
    tool_outputs: list[Any] = []
    version_diff_result: Mapping[str, Any] | None = None
    start = clock()

    # SOT-2521 — deterministic first move: run the routed first-move tool *before* the model's first
    # turn and seed its evidence, so the model starts from the canonical needle instead of wandering
    # through chunk search until BUDGET_EXHAUSTED. Only seeds on a useful (non-empty) result; the
    # observe-only ledgers see it exactly as a normal tool round would, and no answer is injected.
    if first_move is not None:
        fm_name, fm_args = first_move
        if fm_name in by_name:
            fm_out = dispatch(by_name, fm_name, dict(fm_args or {}))
            if _first_move_useful(fm_out):
                tool_calls.append(fm_name)
                tool_outputs.append(fm_out)
                if fm_name == "version_diff" and isinstance(fm_out, Mapping):
                    version_diff_result = fm_out
                if signals is not None:
                    signals.observe(fm_name, fm_out)
                if calc_signals is not None:
                    calc_signals.observe(fm_name, fm_out)
                if director is not None:
                    director.observe(fm_name, fm_out)
                iterations += 1
                responses = [ToolResponse(DIRECTIVE_MESSAGE, _first_move_directive(fm_name, fm_out))]

    # SOT-2524 — the model-turn cap is a mutable `turn_limit` (not a fixed range) so the budget-exhaustion
    # boundary hook can grant a bounded number of extra turns exactly once, when the cap is reached without
    # a committed answer. Each granted turn still passes the wall-clock check below, so a boundary that is
    # simultaneously past `timeout_s` runs no extra turn.
    turn = 0
    turn_limit = max_turns
    boundary_tried = False
    submitted = False
    while True:
        if turn >= turn_limit:
            # Turn budget exhausted. Give the re-search director one bounded targeted push at the still-
            # unmet obligations before finalizing the abstain; on a directive, extend the turn budget by
            # the director's max_rounds and run the re-search through the same commit gate below.
            if boundary_tried or submitted:
                break
            boundary_tried = True
            boundary_directive = _budget_boundary_directive(
                director, budget_boundary, tool_calls, answer)
            if boundary_directive is None:
                break
            responses = [ToolResponse(SUBMIT_ANSWER, {
                "abstain_rejected": True,
                "reason": "反復上限に達しました。棄権を確定する前に、未充足の証拠義務『だけ』を局所再探索してください。",
                "directive": boundary_directive,
            })]
            turn_limit += director.budget.max_rounds
        turn += 1
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
            if (contract == "simple_lookup" and not tool_calls and is_abstain(text)
                    and not simple_lookup_guidance_sent):
                # An empty first model turn is a transport/model omission, not evidence that a named
                # source is absent.  Give the ordinary lookup path one bounded mandatory-tool retry;
                # the second abstain remains terminal so this cannot loop or inflate every question.
                responses = [ToolResponse(DIRECTIVE_MESSAGE, (
                    "空の初回応答だけでは棄権できません。質問で指定された会社・ファイル名を使って "
                    "find_files を実行し、対象がOffice文書なら read_office、表データなら "
                    "canonical_route を実行してください。抽出結果から質問の行・グループ・最大/最後などの"
                    "条件を確認し、根拠が得られた場合だけ submit_answer してください。"
                    "必須ツールでも対象を解決できない場合のみ棄権してください。"))]
                simple_lookup_guidance_sent = True
                iterations += 1
                continue
            if contract == "chart_read" and not strict_chart_evidence and not chart_guidance_sent:
                # A first-turn prose answer/abstain is not evidence that the strict path is unavailable.
                # Feed one bounded mandatory-tool directive back; if the model still cannot obtain strict
                # evidence, its next abstain is accepted (never loop indefinitely).
                responses = [ToolResponse(DIRECTIVE_MESSAGE, (
                    (
                        "ガント週範囲はread_officeの決定論週グリッドを試す前に確定・棄権できません。"
                        if gantt_question else
                        "グラフ数値はread_chart_valuesの厳密経路を試す前に確定・棄権できません。")
                    + " " +
                    (
                        "対象pptxをfind_filesで特定し、read_office(file=...) の"
                        "【ガント週グリッド:決定論】から対象活動の開始週・終了週を採用してください。"
                        "抽出が曖昧な場合のみ棄権してください。"
                        if gantt_question else
                        "対象xlsxをfind_files/canonical_routeで特定し、read_chart_values(file=..., "
                        "column=..., operation='histogram_max_count')を実行してください。"
                        "失敗した場合のみ棄権してください。"
                    )))]
                chart_guidance_sent = True
                iterations += 1
                continue
            if not is_abstain(text):
                regulation_check = _question_contract.validate_regulation_answer(question, text)
                if not regulation_check.passed:
                    if not regulation_guidance_sent:
                        responses = [ToolResponse(
                            DIRECTIVE_MESSAGE,
                            "通常テキストでは回答を確定できません。特別規定が存在しない場合は、同じ契約の"
                            "一般規定を再確認し、単価・税処理・課金単位・丸め・精算周期・上限をすべて含む"
                            "回答をsubmit_answerで返してください。原文で確定できなければ棄権してください。")]
                        regulation_guidance_sent = True
                        iterations += 1
                        continue
                    text = ABSTAIN
            if contract == "chart_read" and not strict_chart_evidence and not is_abstain(text):
                text = ABSTAIN
            candidate = Answer(answer=text, confidence=0.0,
                               method="(submit_answer未使用: 最終テキストを採用)")
            rejection = _reference_commit_rejection(
                question, candidate, contract, tool_outputs=tool_outputs,
                version_diff_result=version_diff_result)
            if rejection is not None:
                responses = [ToolResponse(SUBMIT_ANSWER, rejection)]
                iterations += 1
                continue
            answer = candidate
            stop_reason = "answered"
            break

        responses = []
        submitted = False
        dispatched_tool = False
        for call in step.function_calls:
            tool_calls.append(call.name)
            if call.name == SUBMIT_ANSWER:
                candidate = _answer_from_args(call.args)
                # SOT-2507 — chart pixels are never numeric authority.  A non-abstain chart answer may
                # commit only after read_chart_values returned numCache or source-cell recomputation.
                if (contract == "chart_read" and not is_abstain(candidate.answer)
                        and not strict_chart_evidence):
                    responses.append(ToolResponse(SUBMIT_ANSWER, {
                        "answer_rejected": True,
                        "reason": "グラフ数値の厳密証拠がありません。vision値では回答を確定できません。",
                        "directive": (
                            "read_office(file=...) の【ガント週グリッド:決定論】を実行し、"
                            "週ヘッダx座標とバーleft/widthの結果だけを採用してください。"
                            "決定論抽出が曖昧な場合は棄権してください。"
                            if gantt_question else
                            "read_chart_values(file=..., column=..., operation=...) を実行し、"
                            "numCacheまたは元データ再集計のresultだけを採用してください。"
                            "厳密経路が失敗した場合は棄権してください。"
                        ),
                    }))
                    dispatched_tool = True
                    break
                # SOT-2511 — "no special regulation" is only an intermediate finding for a question
                # asking what the regulation says.  Reject a premature terminal answer until the
                # governing fallback rule's rate/tax treatment, billing unit/rounding, cycle and cap are
                # all present.  The guard embeds no policy values; it only enforces semantic coverage.
                if not is_abstain(candidate.answer):
                    from src.rag.agent import question_contract as _question_contract

                    regulation_check = _question_contract.validate_regulation_answer(
                        question, candidate.answer)
                    if not regulation_check.passed:
                        responses.append(ToolResponse(SUBMIT_ANSWER, {
                            "answer_rejected": True,
                            "reason": "規定内容回答のfallback一般規定が不完全です。",
                            "missing": list(regulation_check.missing),
                            "directive": (
                                "『特別規定は存在しない』だけで確定せず、同じ契約の一般規定を局所再探索し、"
                                "単価・税処理・課金単位・丸め・精算周期・上限の有無を回答本文にすべて含めてください。"
                                "値を原文で確定できない場合は推測せず棄権してください。"
                            ),
                        }))
                        dispatched_tool = True
                        break
                rejection = _reference_commit_rejection(
                    question, candidate, contract, tool_outputs=tool_outputs,
                    version_diff_result=version_diff_result)
                if rejection is not None:
                    responses.append(ToolResponse(SUBMIT_ANSWER, rejection))
                    dispatched_tool = True
                    break
                # SOT-2508 — a numeric answer is not complete merely because a number was computed.
                # Enforce the question's quantity definition / unit / rounding contract against the
                # observed compute trail *before* accepting the terminal answer.  This catches the
                # classic ``XのうちYの割合`` denominator swap (e.g. using Y as the denominator) even
                # when the arithmetic itself is internally consistent and high-confidence.  The check
                # is active on the production path where the calc ledger is enabled; callers that
                # deliberately disable calc observation retain the historical byte-identical loop.
                if (contract == "numeric" and calc_signals is not None
                        and not is_abstain(candidate.answer)):
                    from src.rag.agent import question_contract as _question_contract

                    numeric_check = _question_contract.validate_numeric_answer(
                        question, candidate.answer, calc_signals.steps)
                    if not numeric_check.passed:
                        responses.append(ToolResponse(SUBMIT_ANSWER, {
                            "answer_rejected": True,
                            "reason": "量の定義・単位・丸めの証拠義務が未充足です。",
                            "issues": list(numeric_check.issues),
                            "directive": (
                                "回答を確定せず、質問が要求する量を分子・分母・母集団へ書き下してください。"
                                "『XのうちYの割合』ではXだけを分母として別computeで件数を取得し、"
                                "分子件数/分母件数/未丸め値/指定単位/最後の丸めをmethodに明記して再計算すること。"
                                "再計算できなければ推測せず棄権してください。"
                            ),
                        }))
                        dispatched_tool = True
                        break
                # SOT-2500 — full-enumeration closure protocol: for a full_enumeration contract, a
                # deliberate abstain is intercepted once and the closure *procedure* (権威的母集団の特定 →
                # 閉包条件ゲート → 列挙順序) is fed back so completeness is proven before an abstain is
                # accepted. Tried before the generic re-search; one-shot so it cannot loop.
                if enum_gate is not None and is_abstain(candidate.answer):
                    enum_directive = enum_gate.review()
                    if enum_directive is not None:
                        responses.append(ToolResponse(SUBMIT_ANSWER, {
                            "abstain_rejected": True,
                            "reason": "完全列挙の閉包が未確認です。棄権の前に権威的母集団の特定と閉包条件の確認を行ってください。",
                            "directive": enum_directive,
                        }))
                        dispatched_tool = True  # count the guided round; send the procedure next turn
                        break
                # SOT-2502 — obligation-driven local re-search: a deliberate abstain is not accepted
                # immediately. While budget remains and unmet obligations exist, feed a *targeted*
                # re-search directive back so the model re-searches only the unmet obligation. The
                # commit threshold is never touched — a committed (non-abstain) answer finalizes as-is.
                if director is not None and is_abstain(candidate.answer):
                    ev = " ".join(t for t in (candidate.evidence, candidate.method) if t).strip()
                    non_submit = sum(1 for c in tool_calls if c != SUBMIT_ANSWER)
                    directive = director.review(ev, non_submit)
                    if directive is not None:
                        responses.append(ToolResponse(SUBMIT_ANSWER, {
                            "abstain_rejected": True,
                            "reason": "未充足の証拠義務が残っています。棄権の前に局所再探索を実行してください。",
                            "directive": directive,
                        }))
                        dispatched_tool = True  # count the re-search round; send the directive next turn
                        break
                # SOT-2525 — deterministic tool fallback before concluding UNANSWERABLE: when the model
                # is about to abstain and neither the enumeration closure nor the obligation re-search
                # intervened, force ONE contract-typed deterministic tool (canonical_route / version_diff
                # / file_grep, each self-resolving from the question) and — only if it reached concrete
                # evidence — feed that evidence back so the model may answer from it. One-shot; when the
                # tool reaches nothing the abstain proceeds unchanged (従来の安全動作を維持).
                if fallback is not None and is_abstain(candidate.answer):
                    fb_plan = fallback.plan()
                    if fb_plan is not None and fb_plan[0] in by_name:
                        fb_name, fb_args = fb_plan
                        fb_out = dispatch(by_name, fb_name, dict(fb_args or {}))
                        tool_calls.append(fb_name)
                        tool_outputs.append(fb_out)
                        if fb_name == "version_diff" and isinstance(fb_out, Mapping):
                            version_diff_result = fb_out
                        if signals is not None:
                            signals.observe(fb_name, fb_out)
                        if calc_signals is not None:
                            calc_signals.observe(fb_name, fb_out)
                        if director is not None:
                            director.observe(fb_name, fb_out)
                        if _first_move_useful(fb_out):
                            responses.append(ToolResponse(SUBMIT_ANSWER, {
                                "abstain_rejected": True,
                                "reason": "UNANSWERABLE と確定する前に契約型の決定論ツールを実行しました。",
                                "directive": fallback.directive(fb_name, fb_out),
                            }))
                            dispatched_tool = True  # count the forced fallback round; re-prompt next turn
                            break
                answer = candidate
                if director is not None and not is_abstain(answer.answer):
                    director.note_answered()
                stop_reason = "answered"
                submitted = True
                break
            out = dispatch(by_name, call.name, call.args)
            if call.name == "read_chart_values" and _contract.is_contract(out):
                method = out.get("method") or {}
                strict_chart_evidence = (
                    method.get("engine") in {"chart_numcache", "chart_source_compute"}
                    and method.get("numeric_authority") is True
                    and method.get("vision_used") is False
                    and out.get("value") not in (None, "", [], {})
                )
            elif call.name == "read_office" and gantt_question:
                strict_chart_evidence = _has_deterministic_gantt_evidence(out)
            responses.append(ToolResponse(call.name, out))
            tool_outputs.append(out)
            if call.name == "version_diff" and isinstance(out, Mapping):
                version_diff_result = out
            dispatched_tool = True
            if signals is not None:
                # observe-only: fold the tool outcome into the abstain signals without touching `out`
                signals.observe(call.name, out)
            if calc_signals is not None:
                # observe-only: capture derivation証跡 (compute/aggregate/chart) for the calc ledger
                calc_signals.observe(call.name, out)
            if director is not None:
                # observe-only: collect successful tool evidence so unmet obligations reflect coverage
                director.observe(call.name, out)
            if call.name == "caption_image" and isinstance(out, Mapping):
                # A question-specific Vision extraction has already enforced literal co-occurrence,
                # visual-line locality and the smallest directly modified candidate.  When exactly one
                # such candidate remains, it is deterministic evidence: commit it directly instead of
                # asking the model to re-select (or repeatedly rediscover) neighbouring cell items.
                from src.rag.agent import obligations as _obligations

                literal = _obligations.literal_terms(question)
                value = out.get("value")
                candidates = value if isinstance(value, list) else []
                if literal and len(candidates) == 1 and isinstance(candidates[0], Mapping):
                    candidate = str(candidates[0].get("candidate", "") or "").strip()
                    conditioned_text = str(
                        candidates[0].get("conditioned_text", "") or "")
                    literal_check = _obligations.validate_literal_evidence(
                        question, candidate, [out])
                    strict_line = (
                        candidates[0].get("same_visual_line") is True
                        and candidate in conditioned_text
                        and all(term in conditioned_text for term in literal)
                    )
                    if candidate and strict_line and literal_check.passed:
                        answer = Answer(
                            answer=candidate, confidence=1.0,
                            evidence=str(out.get("evidence", "")),
                            method="同一視覚行の単一 literal 候補をそのまま採用",
                        )
                        stop_reason = "answered"
                        submitted = True
                        if director is not None:
                            director.note_answered()
                        break
            if (contract == "version_diff" and call.name == "version_diff"
                    and isinstance(out, Mapping) and out.get("value") is not None):
                # The deterministic full-document differ is the authority for this contract. Once it
                # resolves, committing its exact value directly avoids a second model turn that could
                # paraphrase, select a different change, abstain, or fail in transport.
                answer = Answer(
                    answer=str(out["value"]), confidence=1.0,
                    evidence=str(out.get("evidence", "")),
                    method="version_diff の全スライド/全シート構造差分をそのまま採用",
                )
                stop_reason = "answered"
                submitted = True
                if director is not None:
                    director.note_answered()
                break
        if dispatched_tool:
            # count only genuine tool rounds; the terminal submit_answer turn is not a round
            iterations += 1
        if submitted:
            break
    if not submitted and stop_reason == "max_turns":
        error = error or f"max_turns={max_turns} reached without a final answer"

    model_name = getattr(model, "model_name", settings.GEN_MODEL_HARD)
    investigation = Investigation(
        question=question, answer=answer, iterations=iterations, tool_calls=tool_calls,
        usage=usage, model=model_name, elapsed_s=max(0.0, clock() - start),
        stop_reason=stop_reason, error=error, contract=contract,
    )

    # SOT-2492 — abstain ledger: purely post-decision. The answer above is already final; here we only
    # *record* it. Every abstain path (self-abstain / max_turns / timeout / model_error) flows through
    # `record_abstain`, which always assigns a state code, so a code-less abstain is impossible.
    if record_enabled and is_abstain(investigation.answer.answer):
        signals.stop_reason = stop_reason
        signals.evidence_text = " ".join(
            t for t in (answer.evidence, answer.method, answer.answer) if t).strip()
        if director is not None:
            # SOT-2502 — record the local re-search history + why it terminated (BUDGET/UNANSWERABLE)
            # so 探索履歴 always remains on the abstain and 即棄権 leaves an auditable trail.
            signals.research_trace = director.trace()
            signals.research_terminal = director.terminal
        _abstain_ledger.record_abstain(
            investigation, signals, path=(None if ledger is True else ledger))

    # SOT-2495/SOT-2506 — calc ledger: record literal numbers AND answers (e.g. a column label such as
    # ``bmi``) produced under the numeric/derived-calculation contract.  The latter matters for argmax
    # statistics: the served answer is text, but it is still the output of a replayable computation.
    # Keep the record on the Investigation as well as persisting it, avoiding racy question lookups in
    # the shared JSONL when gate_question runs concurrently over gold-100.
    if calc_enabled and not is_abstain(investigation.answer.answer) and (
            _calc_ledger.is_numeric_answer(investigation.answer.answer) or contract == "numeric"):
        record = _calc_ledger.record_calc(
            investigation, calc_signals, path=(None if calc_ledger is True else calc_ledger))
        investigation.calc_record = record.to_dict()

    return investigation


def investigate_batch(model_factory: Callable[[str, Sequence[AgentTool]], Model],
                      questions: Sequence[str], *,
                      profile_factory: Callable[[], CorpusProfile] | None = None,
                      shared_profile: CorpusProfile | None = None,
                      max_turns: int = DEFAULT_MAX_TURNS,
                      timeout_s: float = DEFAULT_TIMEOUT_S) -> list[Investigation]:
    """Investigate each question with a fresh model + tools; return one result per question.

    Profile modes (SOT-2528):
    - default / ``profile_factory``: each question gets its OWN :class:`CorpusProfile` (a fresh one, or
      the factory's) so nothing carries across questions — the historical isolation behaviour.
    - ``shared_profile``: every question reuses the SAME profile instance, so a password/alias/format
      discovered on one question is reused on the next instead of being re-derived. These are corpus
      facts (not per-question answers), so sharing only removes rediscovery cost — answers are unchanged.
      ``shared_profile`` takes precedence over ``profile_factory`` when both are given.
    """
    prof = profile_factory or (lambda: CorpusProfile())
    out: list[Investigation] = []
    for q in questions:
        tools = build_tools(shared_profile if shared_profile is not None else prof())
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
                (types.Part.from_text(text=str(r.response)) if r.name == DIRECTIVE_MESSAGE else
                 types.Part.from_function_response(
                     name=r.name,
                     response=(r.response if isinstance(r.response, Mapping)
                               else {"result": r.response})))
                for r in tool_responses]))
        resp = self._client.models.generate_content(
            model=self.model_name, contents=self._contents, config=self._config)
        cand = resp.candidates[0]
        parts = list(cand.content.parts or [])
        # Vertex occasionally omits ``content.role`` on a function-call response.  Reusing that object
        # verbatim makes the next request fail with "Please use a valid role: user, model".  Preserve
        # the returned parts while normalizing the documented model role.
        self._contents.append(types.Content(role="model", parts=parts))
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
                         model: str | None = None,
                         system: str | None = None) -> GeminiModel:
    """Fresh live conversation for one question (each question gets an isolated context).

    ``system`` overrides the default :data:`SYSTEM_PROMPT` — used by :func:`answer_question` to inject
    the SOT-2498 contract routing hint for this question (default keeps the base prompt).
    """
    return GeminiModel(question, to_genai_tools(tools), model=model,
                       system=system if system is not None else SYSTEM_PROMPT)


def answer_question(question: str, *, model: str | None = None,
                    profile: CorpusProfile | None = None,
                    max_turns: int = DEFAULT_MAX_TURNS,
                    timeout_s: float = DEFAULT_TIMEOUT_S,
                    ledger: "str | object | bool | None" = True,
                    calc_ledger: "str | object | bool | None" = True,
                    research: "bool | Mapping[str, Any] | object | None" = True,
                    enumeration: "bool | Mapping[str, Any] | object | None" = True,
                    routing: "bool | object | None" = True,
                    contract_flash: "Callable[[str], str | None] | None" = None) -> Investigation:
    """Convenience live entry point: investigate one ``question`` with a real Gemini conversation.

    The abstain ledger (SOT-2492) is **on by default** here so the production answer path records a
    coded diagnosis for every abstain (``artifacts/abstain_ledger.jsonl``). The calc ledger (SOT-2495)
    is likewise **on by default** so every committed numeric answer records its typed calculation証跡
    (``artifacts/calc_ledger.jsonl``). The obligation-driven local re-search loop (SOT-2502) is **on by
    default** so a deliberate abstain re-searches its unmet evidence obligations before it is accepted
    (即棄権が構造上不可能). The budget-exhaustion boundary hook (SOT-2524) is likewise **on by default**
    (``RAG_BUDGET_BOUNDARY_RESEARCH``) so that re-search also fires at the ``max_turns``/``timeout``
    boundary — where the model wandered out of turns without ever deliberately abstaining, the dominant
    BUDGET_EXHAUSTED cause. The full-enumeration closure protocol (SOT-2500) is **on by default** so a
    ``full_enumeration`` question proves its completeness (権威的母集団の特定 + 閉包条件) before it may
    abstain. Pass ``ledger``/``calc_ledger``/``research``/``enumeration`` ``=False``/``None`` to disable,
    or a path (ledgers) / budget mapping (research) to configure.

    Contract routing (SOT-2498) is **on by default** here: the question is classified into its
    :class:`~src.rag.agent.question_contract.QuestionContract` and a corpus-fact-free *routing hint*
    (推奨初手ツール優先順 + 完了条件) is appended to the system prompt so the agent's first move is
    steered toward the tool most likely to reach the evidence — data/横断集計/数値 →
    ``canonical_route``/``compute``/``corpus_aggregate`` first, 書式/グラフ/空間/版差分 → the specialised
    tool first, 単純検索 → the fast retrieval path (Adaptive-RAG). The hint is *advisory only* (final tool
    choice stays with the model) and the classified contract is recorded on the returned
    :class:`Investigation`. Pass ``routing=False``/``None`` to answer with the base prompt (no hint, no
    contract label); ``contract_flash`` injects a live flash arbiter for genuinely ambiguous questions
    (default: deterministic classification only, so this path stays reproducible).
    """
    profile_obj = profile or CorpusProfile()
    tools = build_tools(profile_obj)
    system: str | None = None
    contract: str | None = None
    first_move: "tuple[str, Mapping[str, Any]] | None" = None
    fallback: "object | None" = None
    if routing not in (None, False):
        from src.rag.agent import question_contract as _question_contract
        from src.rag.agent import routing as _routing  # lazy: keeps import free of the classifier deps
        qc = _routing.classify_for_routing(question, flash=contract_flash)
        system = _routing.routed_system_prompt(SYSTEM_PROMPT, qc, question)
        contract = qc.contract
        # SOT-2521 — the loop-side deterministic first move (see investigate ``first_move``). Reuses the
        # same first-tool decision as the prompt hint so they can never disagree. Gated OFF by default
        # (``FIRST_MOVE_ROUTING``): the always-on variant regressed gold-100, so the wiring stays dormant
        # here — the answer path is byte-identical — until SOT-2527 measures a narrowed enablement.
        if FIRST_MOVE_ROUTING:
            first_move = _routing.deterministic_first_move(qc, question)
        # SOT-2525 — deterministic tool fallback before UNANSWERABLE (see investigate ``fallback``).
        # Gated OFF by default (``UNANSWERABLE_FALLBACK``): the wiring stays dormant so the answer path is
        # byte-identical until SOT-2527 measures its net effect on gold-100.
        if UNANSWERABLE_FALLBACK:
            fallback = _question_contract.DeterministicFallbackGate(question, contract)
        started = time.monotonic()
        deterministic: Answer | None = None
        deterministic_tools: list[str] = []
        deterministic = _deterministic_seating_side_answer(question)
        if deterministic is not None:
            deterministic_tools = ["seating_lookup", "enumeration_closure"]
        elif _question_contract.is_staff_population_question(question):
            deterministic = _deterministic_staff_population_answer(question)
            deterministic_tools = ["corpus_aggregate", "enumeration_closure"]
        elif contract == "chart_read" and _question_contract.is_gantt_week_question(question):
            deterministic = _deterministic_gantt_answer(question, profile_obj)
            deterministic_tools = ["canonical_route", "find_files", "read_office"]
        elif (contract == "simple_lookup"
              and _question_contract.is_regulation_content_question(question)):
            deterministic = _deterministic_regulation_answer(question, profile_obj)
            deterministic_tools = ["canonical_route", "find_files", "read_office"]
        if deterministic is not None:
            return Investigation(
                question=question,
                answer=deterministic,
                iterations=len(deterministic_tools),
                tool_calls=deterministic_tools,
                usage=Usage(),
                model="deterministic",
                elapsed_s=max(0.0, time.monotonic() - started),
                stop_reason="answered",
                contract=contract,
            )
        # Ratio contracts need separate numerator/denominator computations plus final unit/rounding
        # validation.  Preserve the global cap for all other questions, but allow four bounded extra
        # turns when the caller kept the default so a correct re-computation can finish after one
        # rejected contract-violating submission (SOT-2508 focused cycle 1 root cause).
        if (contract == "numeric" and max_turns == DEFAULT_MAX_TURNS
                and _question_contract.numeric_requirements(question).ratio):
            max_turns += 4
        if (timeout_s == DEFAULT_TIMEOUT_S and (
                _question_contract.is_regulation_content_question(question)
                or _question_contract.is_gantt_week_question(question))):
            # A whole-section read / PPTX shape extraction plus guarded resubmission can exceed the
            # ordinary budget.  This remains bounded and applies only to the two deterministic paths.
            timeout_s = 300.0
    model_obj = gemini_model_factory(question, tools, model=model, system=system)
    return investigate(model_obj, question, tools, max_turns=max_turns, timeout_s=timeout_s,
                       ledger=ledger, calc_ledger=calc_ledger, research=research,
                       enumeration=enumeration, contract=contract, first_move=first_move,
                       fallback=fallback, budget_boundary=BUDGET_BOUNDARY_RESEARCH)
