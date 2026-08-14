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


def _subflag(name: str) -> bool:
    """SOT-2712 — a direct-commit target-class extension is gated by its own sub-flag (default OFF).

    ``enabled()`` (``RAG_VDIFF_DIRECT_COMMIT``) still gates the whole lane; these sub-flags only turn on
    the extra classes so that with only the base flag set the serve path stays byte-identical to SOT-2706.
    """
    return os.getenv(name, "0").strip().lower() in _ON


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


# --------------------------------------------------------------------------- SOT-2712 拡張の束縛
# 版種別トークン（doc-kind）を new/old 名から質問非依存に導出する（project 語だけでは同一 project 内の
# 複数版ペアを区別できない — idx9 は 提案書 v1→v3 と 最終報告 old→new が同 project）。バージョン/old マーカーと
# 拡張子を落とし、残った素の版種別語（例「最終報告」/「提案書」）が質問に出るペアだけを残す。
_VERSION_MARKER = re.compile(r"_?(v\d+|r\d+|rev\d*|old|new|final|latest|最新|最終版|旧|新)", re.I)
_EXT_RE = re.compile(r"\.(pptx|docx|xlsx|pdf|ipynb|csv|md)$", re.I)


def _doc_kind(rec: "dict[str, Any]") -> str:
    """The version-pair's document-kind token (project / corp / version / ext stripped), whitespace-free."""
    name = _norm(rec.get("new_name") or rec.get("old_name") or "")
    s = _EXT_RE.sub("", name)
    projfull = _norm(rec.get("project") or "")
    projstem = _norm(_CORP_PREFIX.sub("", rec.get("project") or ""))
    for token in (projfull, projstem):
        if token:
            s = s.replace(token, "")
    s = _VERSION_MARKER.sub("", s)
    return s.strip("_／/・.-")


def _bind_record(diff_store, question: str) -> "dict[str, Any] | None":
    """一意束縛（拡張版）: 明示ファイル名 → project、project が複数該当なら doc-kind 一致で 1 件に絞る。

    :func:`_resolve_unique_record` と最初の 2 tier は同一。差分は、project 一致が複数（同 project 内に複数版
    ペア）で従来 None になっていたケースを **doc-kind トークンが質問に出る 1 ペア**に限って束縛する点のみ。
    これは新サブフラグ・レーン専用（既存 summary commit 経路は :func:`_resolve_unique_record` のまま不変）。
    """
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
    if len(file_hits) == 1:
        return file_hits[0]
    if file_hits:
        return None  # 明示ファイル名が複数該当 = 曖昧、従来経路へ
    if len(proj_hits) == 1:
        return proj_hits[0]
    if len(proj_hits) > 1:
        narrowed = [r for r in proj_hits if (_doc_kind(r) and _doc_kind(r) in qn)]
        if len(narrowed) == 1:
            return narrowed[0]
    return None


def _summary_commit(rec: "dict[str, Any]") -> "dict[str, Any] | None":
    """SOT-2706 の逐語 summary commit（idx1/22）— rank0 が summary 付き SUBSTANTIVE 唯一の時だけ commit。"""
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


def _resolve_column_rename(rec: "dict[str, Any]") -> "dict[str, Any] | None":
    """SOT-2712 (idx14) — 列名変更クラスの direct-commit。rank0 が schema_name_change の SUBSTANTIVE modify の時、
    全 schema_name_change 変更の old→new を質問非依存に集約して summary を合成する（gold 文言のハードコードなし）。

    語彙は実物の old/new のみから導出: 全て「空白→アンダースコア」変換なら「アンダースコア表記へ修正」、それ以外は
    素の「修正」。列名の見出しは store の structural_location（例『データ列名』）を採用。
    """
    changes = rec.get("changes") or []
    if not changes:
        return None
    rank0 = changes[0]
    if rank0.get("intent") != "SUBSTANTIVE" or "schema_name_change" not in (rank0.get("attributes") or []):
        return None
    news: list[str] = []
    olds: list[str] = []
    for c in changes:
        if (c.get("intent") == "SUBSTANTIVE" and c.get("kind") == "modify"
                and "schema_name_change" in (c.get("attributes") or [])):
            n, o = str(c.get("new") or "").strip(), str(c.get("old") or "").strip()
            if n and o and n not in news:
                news.append(n)
                olds.append(o)
    if not news:
        return None
    # 変換記述を実データから導出（gold 文言ではなく old→new の実変換）: 全て「空白→アンダースコア」か否か。
    underscore = all(" " in o and n == o.replace(" ", "_") for o, n in zip(olds, news))
    label = str(rank0.get("structural_location") or "").strip() or "データ列名"
    verb = "アンダースコア表記へ修正" if underscore else "表記を修正"
    value = f"{label}を{verb}：{'、'.join(news)}"
    evidence = {
        "old_file": rec.get("old_rel"),
        "new_file": rec.get("new_rel"),
        "version_basis": rec.get("basis"),
        "structural_location": rank0.get("structural_location"),
        "renamed_columns": [f"{o}→{n}" for o, n in zip(olds, news)],
        "store": "diff_store",
        "provenance": "precomputed schema_name_change class (verbatim old/new, plan_fanout非経由)",
    }
    method = {
        "engine": "diff_store",
        "contract": "version_diff",
        "selection": "column_rename_class_direct_commit",
        "naturalize": False,
        "verified_operand": True,
        "confidence": 1.0,
    }
    return {"value": value, "evidence": evidence, "method": method}


def _resolve_no_change(rec: "dict[str, Any]") -> "dict[str, Any] | None":
    """SOT-2712 (idx9) — 実質変更ゼロの「該当なし」verdict の direct-commit。

    整合が取れた（alignment_ok）版ペアで **SUBSTANTIVE 変更が 1 件も無い**（全て LAYOUT/SURFACE/BOILERPLATE の
    再配置・体裁変更）時だけ、裸形式「該当なし」を commit する（SOT-2665 裸形式教訓）。整合失敗ペアには claim
    しない（欠測を「該当なし」と偽装しない）。
    """
    if not rec.get("alignment_ok", False):
        return None
    changes = rec.get("changes") or []
    if any(c.get("intent") == "SUBSTANTIVE" for c in changes):
        return None
    evidence = {
        "old_file": rec.get("old_rel"),
        "new_file": rec.get("new_rel"),
        "version_basis": rec.get("basis"),
        "substantive_change": False,
        "change_count": rec.get("change_count"),
        "store": "diff_store",
        "provenance": "precomputed diff record: no SUBSTANTIVE change (layout/surface only)",
    }
    method = {
        "engine": "diff_store",
        "contract": "version_diff",
        "selection": "no_change_verdict_direct_commit",
        "naturalize": False,
        "verified_operand": True,
        "confidence": 1.0,
    }
    return {"value": "該当なし", "evidence": evidence, "method": method}


def resolve(question: str) -> "dict[str, Any] | None":
    """rank0 が summary 付き SUBSTANTIVE の一意ペアに束縛できる時だけ、その summary を逐語 commit する contract。

    SOT-2712 でサブフラグ既定 OFF の対象クラス拡張を後置: (a) 列名変更クラス（idx14, ``RAG_VDIFF_DC_COLRENAME``）と
    (b) 実質変更ゼロの「該当なし」verdict（idx9, ``RAG_VDIFF_DC_NOCHANGE``）。逐語 summary commit（idx1/22）が
    束縛・発火しない時のみ試す。どのサブフラグも OFF の従来運用では逐語 commit と完全に同一（byte-identical）。

    それ以外（OFF / 変更意図なし / ペア不確定 / 発火条件不成立）は None を返し従来の LLM ループへ委譲する
    （回答数を減らさない・wrong を増やさない）。fail-open。
    """
    if not enabled():
        return None
    try:
        if not _change_intent(question):
            return None
        from src.rag.index import diff_store
        # 既存の逐語 summary commit（idx1/22）— 束縛/発火条件は SOT-2706 のまま不変。
        rec = _resolve_unique_record(diff_store, question)
        if rec is not None:
            committed = _summary_commit(rec)
            if committed is not None:
                return committed
        # SOT-2712 拡張は、逐語 commit が発火しなかった時のみ（サブフラグ OFF なら以降は全て None）。
        if not (_subflag("RAG_VDIFF_DC_COLRENAME") or _subflag("RAG_VDIFF_DC_NOCHANGE")):
            return None
        rec2 = _bind_record(diff_store, question)
        if rec2 is None:
            return None
        if _subflag("RAG_VDIFF_DC_COLRENAME"):
            result = _resolve_column_rename(rec2)
            if result is not None:
                return result
        if _subflag("RAG_VDIFF_DC_NOCHANGE"):
            result = _resolve_no_change(rec2)
            if result is not None:
                return result
        return None
    except Exception:  # noqa: BLE001 — a broken read must fall back, never break the answer path
        return None
