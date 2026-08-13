"""版ペア差分 record summary の direct-commit レーン（SOT-2706 / cycle10, idx1・idx22）.

SOT-2700 が実証したとおり、diff store の record summary が gold と byte 一致していても、plan_fanout の
LLM が言い換える経路では逐語が脱落して Incorrect になる（idx22: 『（Attr1〜64は同一）』脱落、idx1: 意味枠脱落）。
本レーンはその **direct-commit 昇格** — version_diff 型の質問で diff store の対象ペアが一意に解決し、rank0 が
SUBSTANTIVE で summary を持つとき、その **record summary を逐語で回答としてコミット**する（plan_fanout の
言い換えを経由しない）。

precision-first（SOT-2601 の発火緩和 fail の教訓）:
  * 版ペアは質問から一意に束縛できる時のみ（明示ファイル名一致 → project stem 一致の順、複数該当は棄権）。
  * rank0 が唯一の summary 保持 SUBSTANTIVE 変更である時のみ（多重変更・ペア不確定は従来経路へ委譲）。
  * 質問が変更点を問うている（is_diff_question or 変更/比較 等の語）時のみ。

``RAG_VDIFF_DIRECT_COMMIT`` 既定 OFF ⇒ :func:`resolve` は None を返し serve path は byte-identical。
RAG_FACT_LAYER の下位レーンとして fact_layer.resolve の末尾に後置される（投資者ツールサーフェスは非改変）。
gold 値ハードコードなし・summary はストア（質問非依存 build）由来。
"""
from __future__ import annotations

import os
import re
from typing import Any

_ON = {"1", "true", "yes", "on"}

# 法人格接頭辞（案件バインドの stem 照合で除去する。質問は接頭辞を落として呼ぶ — memory sot2707）。
_CORP_PREFIX = re.compile(r"(株式会社|医療法人社団|医療法人|一般社団法人|社会福祉法人|有限会社|合同会社)")
# 変更点を問う質問の語（is_diff_question が False の notebook 系 idx22 もこれで拾う）。
_CHANGE_CUES = ("変更", "変わ", "修正", "更新", "差分", "変化", "比較", "変えた", "変わっ")


def enabled() -> bool:
    """True when the serve path should direct-commit a diff-store record summary (default OFF)."""
    return os.getenv("RAG_VDIFF_DIRECT_COMMIT", "0").strip().lower() in _ON


def _norm(value: Any) -> str:
    return re.sub(r"\s+", "", str(value or ""))


def _change_intent(question: str) -> bool:
    """質問が「何が変わったか」を問うているか（is_diff_question OR 変更/比較語）。"""
    try:
        from src.rag import diffpair
        if diffpair.is_diff_question(question):
            return True
    except Exception:  # noqa: BLE001
        pass
    qn = _norm(question)
    return any(cue in qn for cue in _CHANGE_CUES)


def _resolve_unique_record(diff_store, question: str) -> "dict[str, Any] | None":
    """質問から diff store の版ペアを一意に束縛する（明示ファイル名 → project stem の順、複数該当は None）。"""
    try:
        rows = diff_store.load()
    except Exception:  # noqa: BLE001
        return None
    if not rows:
        return None
    qn = _norm(question)
    file_hits: list[dict[str, Any]] = []
    proj_hits: list[dict[str, Any]] = []
    for r in rows:
        on, nn = _norm(r.get("old_name")), _norm(r.get("new_name"))
        if on and nn and on in qn and nn in qn:
            file_hits.append(r)
        proj = _norm(r.get("project"))
        stem = _norm(_CORP_PREFIX.sub("", r.get("project") or ""))
        if (proj and proj in qn) or (stem and stem in qn):
            proj_hits.append(r)
    # 明示ファイル名一致を優先（同一 project に複数版ペアがあっても正しいペアを選べる）。無ければ project 一致。
    pool = file_hits if file_hits else proj_hits
    return pool[0] if len(pool) == 1 else None


def resolve(question: str) -> "dict[str, Any] | None":
    """rank0 が summary 付き SUBSTANTIVE の一意ペアに束縛できる時だけ、その summary を逐語 commit する contract。

    それ以外（OFF / 変更意図なし / ペア不確定 / rank0 非 SUBSTANTIVE / summary 無し / 複数 summary で曖昧）は
    None を返し従来の LLM ループへ委譲する（回答数を減らさない・wrong を増やさない）。fail-open。
    """
    if not enabled():
        return None
    try:
        if not _change_intent(question):
            return None
        from src.rag.index import diff_store
        rec = _resolve_unique_record(diff_store, question)
        if rec is None:
            return None
        changes = rec.get("changes") or []
        if not changes:
            return None
        rank0 = changes[0]
        if rank0.get("intent") != "SUBSTANTIVE":
            return None
        summary = str(rank0.get("summary") or "").strip()
        if not summary:
            return None
        # 曖昧回避: summary を持つ SUBSTANTIVE 変更が rank0 ただ一つである時のみ commit（多重変更は従来経路）。
        summarized = [c for c in changes
                      if str(c.get("summary") or "").strip() and c.get("intent") == "SUBSTANTIVE"]
        if len(summarized) != 1 or summarized[0] is not rank0:
            return None
    except Exception:  # noqa: BLE001 — a broken read must fall back, never break the answer path
        return None

    evidence = {
        "old_file": rec.get("old_rel"),
        "new_file": rec.get("new_rel"),
        "version_basis": rec.get("basis"),
        "structural_location": rank0.get("structural_location"),
        "rank": rank0.get("rank"),
        "change_kind": rank0.get("kind"),
        "store": "diff_store",
        "provenance": "precomputed version-pair diff record summary (verbatim, plan_fanout非経由)",
    }
    method = {
        "engine": "diff_store",
        "contract": "version_diff",
        "selection": "record_summary_direct_commit",
        "naturalize": False,
        "verified_operand": True,
        "confidence": 1.0,
    }
    return {"value": summary, "evidence": evidence, "method": method}
