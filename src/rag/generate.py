"""Answer generation over retrieved evidence, with confidence-gated abstention.

Because the official metric scores Incorrect = -1 but Missing = 0, a low-confidence guess
is strictly worse than abstaining. The model returns {answer, confidence, ...}; on low
confidence we emit the abstention string. When a question concerns a figure/chart/highlight
and a matching PNG is retrieved, the raw image is attached for multimodal reading.
"""
from __future__ import annotations

import json
import re

import tiktoken

from config import settings
from src.rag import archetype, compute, diffpair, enumeration, llm, normalize, pivotcond, retrieve
from src.rag.corpus import nfc

_ENC = tiktoken.get_encoding("cl100k_base")

# Trust map (config/archetype_trust.json) produced by scoring.selfimprove: which question
# archetypes the RAG answers reliably. The gate is strictly *additive* — it can only turn a
# would-be answer into an abstention for a measured-unreliable archetype, so it never raises the
# Incorrect rate. Unknown archetypes and a missing map leave behaviour unchanged.
_TRUST_PATH = settings.REPO_ROOT / "config" / "archetype_trust.json"


def _load_trust() -> dict[str, dict]:
    try:
        data = json.loads(_TRUST_PATH.read_text(encoding="utf-8"))
        return data.get("archetypes", data) if isinstance(data, dict) else {}
    except Exception:
        return {}


def _trust_blocks(question: str) -> bool:
    """True when the question's archetype was *measured* and marked untrusted (→ abstain)."""
    arch = archetype.classify(question)
    if arch == "unknown":
        return False
    entry = _load_trust().get(arch)
    return bool(entry is not None and entry.get("trust") is False)


def _holdout_validated(arch: str) -> bool:
    """True when the archetype was *proven on the sealed hold-out slice* (config/archetype_trust.json,
    ``holdout_validated``). Gates whether a deterministic hard module (diffpair / compute / pivot /
    enumeration) may DIRECTLY commit its answer. Until an archetype earns this, its module is advisory:
    its extraction is injected as a hint and the LLM + consistency + verify gates decide (abstain-
    leaning), so a module that overfit the visible projects cannot commit a confident wrong answer on
    unseen test data (the #4 −0.1 regression). A missing map / entry ⇒ False ⇒ advisory."""
    entry = _load_trust().get(arch)
    return bool(entry is not None and entry.get("holdout_validated") is True)

SYSTEM = """あなたは社内共有ドライブの資料に基づいて質問へ回答するRAGアシスタントです。
以下を厳守してください。
- 回答は必ず日本語。提供された『根拠資料』のみを使用し、外部知識や推測で補わない。
- 質問文が指定する形式・単位・小数桁・丸め方・並び順・主略称/通常表現・抽出対象の表記に従う。
- 資料内で定義されたタスクID/アクションID/列名/パラメータ名などの識別子は資料の表記どおりに書く。
- 要素を列挙する問題は、指定された順序（ID昇順・座席表順・文書出現順など）で過不足なく「、」区切りで列挙する。
- 条件に該当する対象が資料内に**明確に存在しないと確認できる**場合のみ「該当なし」と答える。
  対象が存在するか自体が不明なとき「該当なし」と断定してはいけない（それは誤答=-1になる）→ confidence="low"。
- 根拠資料から答えを特定できない、または確信が持てない場合は confidence を "low" にする。
  誤答は0点より悪い(-1点)ため、確信が持てないときは推測せず低確信とすること。
- 特に、複数資料をまたぐ集計・計算・差分は、根拠から数値を一つ一つ確認できないなら confidence="low"。
- answer は **問われた値・要素そのものだけ** を書く。説明文・前置き・言い換え・根拠の再掲・
  単位の重複・「〜です」等の冗長表現を付けない。ground_truthに無い追加情報を足すと誤りになる。"""

# Second pass: distill to the minimal exact-format answer AND strictly re-verify support.
_VERIFY_SYSTEM = """あなたは回答の最終検証・整形を行う**厳格な**レビュアです。誤答は0点より悪い(-1点)ため、
確実に正しいと言い切れない場合は不採用(supported=false)にしてください。JSONで返す。
- final: 質問が求める値・要素だけに最小化した最終回答（説明・前置き・冗長語・根拠の再掲・余分な語を
  全て除去、列挙は「、」区切り、指定の単位/桁/表記・並び順に従う）。
- supported の判定（**疑わしきは false**）:
  * 答えの値・要素が根拠（テキストまたは画像）に**明示的に現れており**そのまま読み取れる → true。
  * 次はすべて false にする:
    - 複数資料をまたぐ**計算・集計・合計・差分・平均**で、その数値が根拠に直接書かれていない。
    - **列挙問題**で、条件に一致する要素を**すべて**列挙できたと根拠から確認しきれない（一部のみは false）。
    - 「該当なし」だが、対象が本当に存在しないと根拠で確認できない（見つからなかっただけは false）。
    - 下書きに余分な語・言い換えがあり、正解表記と一致する確証がない。
    - 根拠が弱い/断片的で、答えが推測を含む。"""

_VERIFY_SCHEMA = {
    "type": "object",
    "properties": {
        "supported": {"type": "boolean"},
        "final": {"type": "string"},
    },
    "required": ["supported", "final"],
}

_RESPONSE_SCHEMA = {
    "type": "object",
    "properties": {
        "reasoning": {"type": "string", "description": "根拠に基づく短い思考（日本語）"},
        "answer": {"type": "string", "description": "最終回答（日本語, 簡潔）"},
        "confidence": {"type": "string", "enum": ["high", "medium", "low"]},
    },
    "required": ["answer", "confidence"],
}

_FIGURE_HINT = re.compile(
    r"図|ヒストグラム|グラフ|チャート|プロット|散布|ヒートマップ|マーカー|ハイライト|"
    r"色|太字|塗り|\.png|figure|カウント|棒グラフ|強調"
)
_PNG_NAME = re.compile(r"([A-Za-z0-9_\-]+\.png)")


def _gather_images(question: str, evidence: list[dict], limit: int = 16) -> list["llm.Image"]:
    """Attach PNGs referenced by the question or present as image chunks in the evidence."""
    from src.rag import corpus
    from src.rag.extract import vision

    if not _FIGURE_HINT.search(nfc(question)):
        return []
    refs = {r.rel: r for r in corpus.walk()}
    if not refs:  # corpus not present (e.g. serving from a lean container) → text-only
        return []
    wanted = {nfc(n).lower() for n in _PNG_NAME.findall(question)}
    target_company = retrieve.get().glossary.company_of(question)

    picks: list = []
    # explicit filename match first
    for rel, r in refs.items():
        if r.ext == "png" and nfc(r.name).lower() in wanted:
            if not target_company or nfc(r.project) == nfc(target_company):
                picks.append(r)
    # Image-only reports contain one embedded raster per page. Attach the complete report so a
    # page-title question cannot silently read an unrelated retrieved PNG.
    pdf_picks = []
    for c in evidence:
        if c.get("kind") == "pdf":
            r = refs.get(c["rel"])
            if r is not None and (not target_company or nfc(r.project) == nfc(target_company)):
                pdf_picks.append(r)
    if pdf_picks:
        imgs = []
        for r in pdf_picks:
            try:
                imgs.extend(llm.Image(data=data, mime_type=mime)
                            for data, mime in vision.pdf_page_images(r.path))
            except Exception:
                continue
            if imgs:
                return imgs[:limit]
    # else image chunks that surfaced in retrieval
    if not picks:
        for c in evidence:
            if c.get("kind") == "image":
                r = refs.get(c["rel"])
                if r is not None:
                    picks.append(r)
    seen, imgs = set(), []
    for r in picks:
        if r.rel in seen:
            continue
        seen.add(r.rel)
        try:
            imgs.append(llm.Image(data=vision.load_image_bytes(r.path), mime_type="image/png"))
        except Exception:
            pass
        if len(imgs) >= limit:
            break
    return imgs


def _format_evidence(evidence: list[dict], max_chars: int = 16000) -> str:
    out, total = [], 0
    for i, c in enumerate(evidence, 1):
        # Fact rows (SOT-2449) are deterministic extractions of a single high-signal artifact
        # (a highlighted cell / bold term / Pivot condition / code parameter). Flag them so the
        # reader weights them as directly-readable evidence rather than incidental chunk text.
        tag = "【高信頼・抽出済fact】" if c.get("kind") == "fact" else ""
        block = f"--- 根拠{i}{tag} ---\n{c['text']}\n"
        if total + len(block) > max_chars:
            break
        out.append(block)
        total += len(block)
    return "\n".join(out)


def _clip_tokens(text: str, max_tokens: int = settings.MAX_ANSWER_TOKENS - 20) -> str:
    toks = _ENC.encode(text)
    if len(toks) <= max_tokens:
        return text
    return _ENC.decode(toks[:max_tokens])


def _norm_answer(a: str) -> str:
    """Normalize for consistency comparison: keep alphanumerics + JP, drop separators/units-ish."""
    a = nfc(a).lower()
    return re.sub(r"[\s、,。.\-_/:：;；()（）「」『』【】\"'`円%％]", "", a)


def _draft(prompt: str, system: str, model: str, images, temperature: float, thinking: int) -> dict:
    raw = llm.generate(prompt, system=system, model=model, images=images,
                       temperature=temperature, thinking_budget=thinking,
                       max_output_tokens=2048, response_schema=_RESPONSE_SCHEMA)
    try:
        return json.loads(raw)
    except Exception:
        return {"answer": raw.strip(), "confidence": "low"}


def deterministic_front(question: str, *, apply_trust_gate: bool = True) -> tuple[dict | None, list[str]]:
    """Trust gate + deterministic hard modules, shared by every generation backend.

    Returns ``(result, advisory)``: ``result`` is a finished answer dict when a gate/module
    resolves the question outright (direct-commit or abstain), else None; ``advisory`` collects
    unvalidated module extractions to inject as hints into the LLM prompt.
    """
    # Preserve the additive trust policy even for deterministic routes: an explicitly measured and
    # disabled archetype remains disabled, while an unmapped cross-aggregate proceeds to computation.
    if apply_trust_gate and _trust_blocks(question):
        return _result(question, settings.ABSTAIN, "untrusted-archetype", "", [], [], verified=False), []

    # Hard-module advisory gate (SOT-2424). The deterministic modules below (compute / enumeration /
    # diffpair / pivotcond) DIRECTLY commit a confident answer only for archetypes proven to generalize
    # on the sealed hold-out slice (`_holdout_validated`). For an unproven archetype the module runs in
    # *advisory* mode: its extraction is stashed in ``advisory`` and injected as a hint into the LLM
    # prompt, where consistency + verify decide (abstain-leaning) — so a module that overfit the
    # visible projects cannot commit a confident wrong answer on unseen test (#4 −0.1 regression).
    # ``measuring`` (selfimprove passes apply_trust_gate=False) keeps the ORIGINAL direct-commit
    # behaviour so the loop can measure each module's true deterministic precision per slice.
    measuring = not apply_trust_gate
    advisory: list[str] = []

    # Deterministic UNCONDITIONAL CSV column statistic ("train.csv の {列} の 最大値/平均値/最小値").
    # Pure pandas over the project's own train.csv, so it generalizes to the sealed hold-out companies
    # with no overfit — unlike the conditional _csv_aggregate below it needs no row filter. Direct-commit
    # only when the classified archetype (csv_column_max / csv_column_mean) is hold-out validated;
    # otherwise the value is an advisory hint for the LLM+verify gates. It defers (returns None) on any
    # conditional question, so it never answers a filtered question with a whole-column value.
    _csv_arch = archetype.classify(question)
    if _csv_arch in ("csv_column_max", "csv_column_mean"):
        stat = compute.column_stat_answer(question)
        if stat is not None:
            if measuring or _holdout_validated(_csv_arch):
                return _result(question, _clip_tokens(stat), "high", stat, [], [], verified=True), advisory
            advisory.append(stat)

    if compute.is_compute_question(question):
        computed = compute.answer_question(question)
        if computed is not None:
            if measuring or _holdout_validated("cross_aggregate"):
                return _result(question, _clip_tokens(computed), "high", computed, [], [], verified=True), advisory
            advisory.append(computed)
        elif measuring or _holdout_validated("cross_aggregate"):
            return _result(question, settings.ABSTAIN, "compute-unresolved", "", [], [], verified=False), advisory

    if enumeration.is_enumeration_question(question):
        enum_answer = enumeration.answer_question(question)
        if enum_answer is not None:
            enum_arch = archetype.classify(question)  # enum_set / highlight_set
            if measuring or _holdout_validated(enum_arch):
                return _result(question, _clip_tokens(enum_answer), "high", enum_answer, [], [], verified=True), advisory
            advisory.append(enum_answer)

    # Version-diff questions ("old版と最新版の変更点") are answered by a deterministic structural
    # diff of the two versions, not by retrieval+LLM (the two files are near-identical and the single
    # changed value is buried). When the pair/diff resolves uniquely and version_diff is hold-out
    # validated we commit it; otherwise the diff becomes an advisory hint. When it's clearly a diff
    # question but unresolved, we abstain only in the direct-commit path (Missing 0 beats Incorrect -1).
    if diffpair.is_diff_question(question):
        try:
            diff_ans = diffpair.answer_question(question)
        except Exception:
            diff_ans = None
        if diff_ans:
            if measuring or _holdout_validated("version_diff"):
                return _result(question, _clip_tokens(diff_ans), "high", diff_ans, [], [], verified=True), advisory
            advisory.append(diff_ans)
        elif measuring or _holdout_validated("version_diff"):
            return _result(question, settings.ABSTAIN, "diff-unresolved", "", [], [], verified=False), advisory

    # PivotTable / AutoFilter extraction-condition questions ("PivotTableの抽出条件", "フィルターで
    # 抽出されている条件") are answered by reading the workbook's pivot definition / autofilter XML and
    # recomputing the condition deterministically (valid idx6/idx11/idx21), not by retrieval+LLM (the
    # condition is not in the flattened cell text). Direct-commit only when pivot_condition is hold-out
    # validated; otherwise advisory. Unresolved abstains only in the direct-commit path.
    if pivotcond.is_pivot_condition_question(question):
        try:
            pv_ans = pivotcond.answer_question(question)
        except Exception:
            pv_ans = None
        if pv_ans:
            if measuring or _holdout_validated("pivot_condition"):
                return _result(question, _clip_tokens(pv_ans), "high", pv_ans, [], [], verified=True), advisory
            advisory.append(pv_ans)
        elif measuring or _holdout_validated("pivot_condition"):
            return _result(question, settings.ABSTAIN, "pivot-unresolved", "", [], [], verified=False), advisory

    return None, advisory


def advisory_block(advisory: list[str]) -> str:
    """Prompt block offering unvalidated hard-module extractions as unverified hints."""
    if not advisory:
        return ""
    return (
        "\n\n【未検証の自動抽出候補（hold-out未実証。参考情報に過ぎない。"
        "根拠資料から独立に確認できる場合のみ採用し、確認できなければ採用せず confidence=\"low\"）】\n"
        + "\n".join(f"- {a}" for a in advisory))


def answer_question(question: str, *, k: int = 16, hard: bool = False, verify: bool = True,
                    consistency: bool = True, apply_trust_gate: bool = True) -> dict:
    resolved, advisory = deterministic_front(question, apply_trust_gate=apply_trust_gate)
    if resolved is not None:
        return resolved

    # Retrieval + LLM path (reached when no hard module direct-committed above). The additive trust
    # gate already abstained measured-unreliable archetypes up front; scoring.selfimprove passes
    # apply_trust_gate=False so this loop can *measure* trust.
    r = retrieve.get()
    evidence = r.retrieve(question, k=k)
    images = _gather_images(question, evidence)
    # Advisory hints from unvalidated hard modules: offered as *unverified* candidates only. The verify
    # pass (疑わしきは false) requires the answer to be independently supported by the evidence, so a
    # hint that isn't corroborated leads to abstention rather than a confident commit.
    prompt = (
        f"質問:\n{question}\n\n"
        f"根拠資料:\n{_format_evidence(evidence)}{advisory_block(advisory)}\n\n"
        "上記の根拠のみに基づき、質問へ回答してください。JSONで reasoning, answer, confidence を返す。"
    )
    model = settings.GEN_MODEL_HARD if (hard or images) else settings.GEN_MODEL
    think = 2048 if (hard or images) else 512

    # Self-consistency: draft twice with diversity. Unstable answers (computed/ambiguous guesses)
    # disagree across samples → abstain. Stable facts agree → candidate to commit.
    obj = _draft(prompt, SYSTEM, model, images, 0.0, think)
    ans = (obj.get("answer") or "").strip()
    conf = obj.get("confidence", "low")
    if not ans or conf != "high":
        return _result(question, settings.ABSTAIN, conf, ans, evidence, images, verified=False)

    if consistency:
        obj2 = _draft(prompt, SYSTEM, model, images, 0.6, think)
        ans2 = (obj2.get("answer") or "").strip()
        if not ans2 or _norm_answer(ans) != _norm_answer(ans2):
            return _result(question, settings.ABSTAIN, "inconsistent", ans, evidence, images,
                           verified=False)

    # second pass: distill to exact format + strict support check
    if verify:
        try:
            vraw = llm.generate(
                f"質問:\n{question}\n\n根拠資料:\n{_format_evidence(evidence, max_chars=12000)}\n\n"
                f"下書き回答: {ans}\n\n上記を検証・整形してJSONで supported, final を返す。",
                system=_VERIFY_SYSTEM, model=model, images=images, temperature=0.0,
                thinking_budget=512, max_output_tokens=1024, response_schema=_VERIFY_SCHEMA,
            )
            vobj = json.loads(vraw)
            if not vobj.get("supported", False):
                return _result(question, settings.ABSTAIN, "low", ans, evidence, images, verified=True)
            final_ans = (vobj.get("final") or ans).strip() or ans
            return _result(question, _clip_tokens(final_ans), conf, ans, evidence, images, verified=True)
        except Exception:
            pass  # fall through to the un-verified answer

    return _result(question, _clip_tokens(ans), conf, ans, evidence, images, verified=False)


def _result(question, answer, conf, raw, evidence, images, verified) -> dict:
    # Final commit sink for every answering path. The meaning-preserving normalization layer
    # (SOT-2448) canonicalises a *committed* answer toward the concise ground-truth style so the
    # gpt-5.2 judge does not penalise verbose/off-format phrasing; it is a no-op on abstentions and
    # falls back to the original whenever a value/identifier would not be preserved (zero new risk).
    return {
        "question": question,
        "answer": normalize.normalize_answer(answer, question=question),
        "confidence": conf,
        "raw_answer": raw,
        "verified": verified,
        "evidence_files": [c["rel"] for c in evidence[:6]],
        "used_images": len(images),
    }
