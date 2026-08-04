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
from src.rag import llm, retrieve
from src.rag.corpus import nfc

_ENC = tiktoken.get_encoding("cl100k_base")

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
_VERIFY_SYSTEM = """あなたは回答の整形と最終検証を行うレビュアです。
質問・根拠資料（画像を含む場合あり）・下書き回答を読み、JSONで返す。
- final: 質問が求める値・要素だけに最小化した最終回答（説明・前置き・冗長語・根拠の再掲を全て除去、
  列挙は「、」区切り、指定の単位/桁/表記・並び順に従う）。該当が無いと確認できる場合は「該当なし」。
- supported: 原則 true。下書きが根拠（テキストまたは画像）に整合していれば true とする。
  false にするのは、**複数資料をまたぐ計算・集計・差分**で数値を根拠から確認できない、
  または答えが根拠に全く存在しない/明らかな当て推量の場合のみ。図表・ハイライト・単一資料の
  読み取りは、画像や根拠と整合していれば supported=true とすること。"""

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


def _gather_images(question: str, evidence: list[dict], limit: int = 3) -> list["llm.Image"]:
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
        block = f"--- 根拠{i} ---\n{c['text']}\n"
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


def answer_question(question: str, *, k: int = 16, hard: bool = False, verify: bool = False) -> dict:
    r = retrieve.get()
    evidence = r.retrieve(question, k=k)
    images = _gather_images(question, evidence)
    prompt = (
        f"質問:\n{question}\n\n"
        f"根拠資料:\n{_format_evidence(evidence)}\n\n"
        "上記の根拠のみに基づき、質問へ回答してください。JSONで reasoning, answer, confidence を返す。"
    )
    model = settings.GEN_MODEL_HARD if (hard or images) else settings.GEN_MODEL
    raw = llm.generate(
        prompt,
        system=SYSTEM,
        model=model,
        images=images,
        temperature=0.0,
        thinking_budget=2048 if (hard or images) else 512,
        max_output_tokens=2048,
        response_schema=_RESPONSE_SCHEMA,
    )
    try:
        obj = json.loads(raw)
    except Exception:
        obj = {"answer": raw.strip(), "confidence": "low"}

    ans = (obj.get("answer") or "").strip()
    conf = obj.get("confidence", "low")

    # confidence-gated abstention (Incorrect=-1 → don't guess)
    if not ans or conf == "low":
        return _result(question, settings.ABSTAIN, conf, ans, evidence, images, verified=False)

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
    return {
        "question": question,
        "answer": answer,
        "confidence": conf,
        "raw_answer": raw,
        "verified": verified,
        "evidence_files": [c["rel"] for c in evidence[:6]],
        "used_images": len(images),
    }
