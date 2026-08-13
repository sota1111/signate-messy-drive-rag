"""SOT-2687 — クロス参照カバレッジ拡張レーン（cycle7 K4）.

cycle6 abstain の「クロス参照カバレッジ欠落」クラスタ（idx34/36/62、二次 98/53）を、既存の事前計算ストア族
（glossary / raw_artifact_store / ocr_store）を **質問非依存に横断** して決定論で回収する。根本原因は
「ストア値は焼けているが、質問→案件/段階/上位行への束縛経路が無い」ことなので、本レーンは新しい build を
足さず、既存の precomputed 成果物を serve 時に決定論結合するだけ（Gemini 呼び出しゼロ・$0）。

回収する3つの束縛（precision-first: 一意・厳密に束縛できる時だけ確定、少しでも曖昧なら None→LLM）:

* **idx62 leaderboard 設定差分** — 案件 leaderboard の上位2行が同一モデル種別のとき、ハイパラ列の差分を
  提示（拮抗＝同点や別モデルは None）。例: 青葉与信 extra_trees `n_estimators`（1位=500、2位=300）。
* **idx36 段階メトリクス差** — 報告書の段階記述（中間=線形系 / 最終=非線形）を leaderboard×metrics.json に
  クロス参照し、中間F1（線形系ベストの primary_value）と最終F1（metrics.json f1_macro）の差を決定論計算。
* **idx34 案件ローマ字別名 × アクション遷移** — 用語集の主略称/別名（例 MINAMINO→みなみ野）で案件を束縛し、
  会議録OCRから「M01 時点で未完了・M02 までに完了・担当=指定者」のアクションIDを集合演算で列挙。

``RAG_XREF_COVERAGE`` 既定 OFF ⇒ :func:`resolve` / :func:`tool` は None を返し、tool 集合・関数スキーマ・
serve path は byte-identical。RAG_FACT_LAYER の下位レーンとして fact_layer.resolve の末尾に後置される。
"""
from __future__ import annotations

import importlib
import os
import re
import unicodedata
from typing import Any, Callable

from src.rag.index import raw_artifact_store as _raw
from src.rag.tools import contract as _contract

XREF_COVERAGE_LOOKUP = "xref_coverage_lookup"
_STR = {"type": "string"}
_ON = {"1", "true", "yes", "on"}


def enabled() -> bool:
    """True when the serve path should consult the cross-reference coverage lane (default OFF)."""
    return os.getenv("RAG_XREF_COVERAGE", "0").strip().lower() in _ON


def _norm(value: Any) -> str:
    return unicodedata.normalize("NFKC", str(value or "")).replace(" ", "").replace("　", "").lower()


def _result(value: Any, *, selection: str, evidence: dict[str, Any]) -> dict[str, Any]:
    ev = {"store": "xref_coverage", "provenance": "precomputed (question-independent)", **evidence}
    method = {"engine": "xref_coverage", "contract": "cross_reference", "selection": selection,
              "naturalize": False, "verified_operand": True, "confidence": 1.0}
    return _contract.ensure_contract({"value": value, "evidence": ev, "method": method})


# --------------------------------------------------------------------------- case binding (glossary)
def _resolve_case(question: str) -> str | None:
    """用語集の主略称/別名（MINAMINO 等ローマ字コードを含む）で質問が名指す案件（会社名=案件フォルダ名）を
    一意束縛する。曖昧/未解決なら None（precision-first）。"""
    try:
        glossary = importlib.import_module("src.rag.extract.glossary").load()
    except Exception:  # noqa: BLE001
        return None
    company = glossary.company_of(question)
    return company or None


# --------------------------------------------------------------------------- idx62 leaderboard diff
# leaderboard CSV のメタ/メトリクス列（設定＝ハイパラではない）。これ以外の列を「設定」とみなす際の
# 明示 allowlist と併用する（未知列を設定扱いして誤検出しないため allowlist を主に使う）。
_SETTING_ALLOWLIST = {
    "n_estimators", "learning_rate", "max_depth", "max_iter", "min_samples_leaf", "min_child_samples",
    "num_leaves", "subsample", "colsample_bytree", "reg_lambda", "reg_alpha", "l2_regularization",
    "max_bins", "max_leaf_nodes", "min_child_weight", "gamma", "random_state", "max_features",
    "test_size", "use_date_features", "use_cyclical_time_features", "use_numeric_interactions",
    "split_strategy", "transform_target", "class_weight", "criterion", "bootstrap",
}


def _leaderboard_setting_diff(qn: str, question: str) -> dict[str, Any] | None:
    """上位2件（同一モデル種別）のスコア差を生む設定差分（idx62 型）。拮抗/別モデル/差分なしは None。"""
    if not (("上位2" in qn or "上位2件" in qn or "上位二" in qn or "トップ2" in qn)
            and ("設定" in qn or "差分" in qn or "ハイパラ" in qn or "パラメータ" in qn)
            and ("モデル" in qn or "スコア差" in qn or "比較" in qn)):
        return None
    company = _resolve_case(question)
    if company is None:
        return None
    data = _raw.load()
    rec = data.get("cases_by_project", {}).get(company)
    if not rec:
        return None
    lb = rec.get("leaderboard")
    if not lb:
        return None
    rows = lb.get("rows") or []
    if len(rows) < 2:
        return None
    r1, r2 = rows[0], rows[1]
    metric_col = lb.get("metric_col")
    # 拮抗（上位2件のスコアが同点）なら「差を生む設定」は決まらない → None。
    if metric_col:
        try:
            if float(r1.get(metric_col)) == float(r2.get(metric_col)):
                return None
        except (TypeError, ValueError):
            return None
    # 「同一モデル名内の設定差のみ提示」— モデル種別が違えば設定差分ではなくモデル差 → None。
    if _norm(r1.get("model_type")) != _norm(r2.get("model_type")):
        return None
    header = lb.get("header") or []
    diffs: list[tuple[str, Any, Any]] = []
    for col in header:
        if col not in _SETTING_ALLOWLIST:
            continue
        v1, v2 = r1.get(col), r2.get(col)
        if v1 is None or v2 is None:
            continue
        if str(v1).strip() != str(v2).strip():
            diffs.append((col, v1, v2))
    if len(diffs) != 1:
        # 差分が0（見かけ上同一設定）や複数（一意に「その設定差」と言えない）は precision-first で棄権。
        return None
    col, v1, v2 = diffs[0]
    value = f"{col}（1位={v1}、2位={v2}）"
    return _result(value, selection="leaderboard_top2_setting_diff",
                   evidence={"case": company, "model_type": r1.get("model_type"),
                             "rank1": {metric_col: r1.get(metric_col), col: v1},
                             "rank2": {metric_col: r2.get(metric_col), col: v2},
                             "leaderboard_rel": lb.get("rel")})


# --------------------------------------------------------------------------- idx36 stage-metric diff
_LINEAR_CUE = ("linear", "logistic", "線形")


def _stage_metric_f1_diff(qn: str, question: str) -> dict[str, Any] | None:
    """中間報告F1（線形系ベスト）と最終報告F1（metrics.json f1_macro）の差の絶対値（idx36 型）。"""
    if not ("中間報告" in qn and "最終報告" in qn and "f1" in qn
            and ("差" in qn) and ("絶対値" in qn or "絶対" in qn or "差" in qn)):
        return None
    company = _resolve_case(question)
    if company is None:
        return None
    data = _raw.load()
    rec = data.get("cases_by_project", {}).get(company)
    if not rec:
        return None
    # 最終F1: metrics.json の f1_macro（フル精度・最終モデル）。
    metrics = (rec.get("metrics") or {}).get("value") or {}
    final_f1 = metrics.get("f1_macro")
    lb = rec.get("leaderboard")
    if final_f1 is None or not lb:
        return None
    # leaderboard が f1 系メトリクスでない案件には適用しない（別メトリクスの段階差は本レーン対象外）。
    metric_col = lb.get("metric_col")
    metric_name = (lb.get("metric_name") or "").lower()
    if metric_col is None or "f1" not in metric_name:
        return None
    # 中間F1: 線形系（linear/logistic）の primary_value 最大（＝中間報告時点のベスト線形モデル）。
    inter_f1 = None
    for row in lb.get("rows") or []:
        mt = _norm(row.get("model_type"))
        if not any(cue in mt for cue in _LINEAR_CUE):
            continue
        try:
            v = float(row.get(metric_col))
        except (TypeError, ValueError):
            continue
        if inter_f1 is None or v > inter_f1:
            inter_f1 = v
    if inter_f1 is None:
        return None
    try:
        diff = abs(float(final_f1) - float(inter_f1))
    except (TypeError, ValueError):
        return None
    value = repr(diff)  # フル精度の差（例 0.0961911...）
    return _result(value, selection="stage_metric_f1_absolute_diff",
                   evidence={"case": company, "final_f1": final_f1, "intermediate_f1_linear": inter_f1,
                             "final_source": "metrics.json:f1_macro",
                             "intermediate_source": "leaderboard best linear primary_value",
                             "formula": "abs(final_f1 - intermediate_linear_f1)"})


# --------------------------------------------------------------------------- idx34 action transition
_MEETING_ID_RE = re.compile(r"[MmＭ]0*([0-9]{1,2})")
_ACTION_ID_RE = re.compile(r"[AＡ]-?0*([0-9]{1,3})")
_DONE_TOKENS = ("close", "closed", "done", "完了", "済", "クローズ", "終了")
_OPEN_TOKENS = ("open", "オープン", "未完", "未着手", "進行")
_PERSON_RE = re.compile(r"([一-龥ぁ-んァ-ヴ]{2,4})\s*(?:さん|氏)")


def _person_surname(question: str) -> str | None:
    """質問が名指す担当者の姓（『伊藤さん』→『伊藤』）。無ければ None。"""
    m = _PERSON_RE.search(unicodedata.normalize("NFKC", question))
    if m:
        return m.group(1)
    return None


def _parse_meeting_actions(text: str, person: str) -> tuple[set[str], dict[str, bool]]:
    """1会議のOCR本文から (done なアクションID集合, ID→担当が person を含むか) を抽出する。

    OCR 崩れ・複数IDが1行に折り返す表に耐えるよう、``A##`` の出現ごとに次の ``A##`` までのセグメントを
    切り出し、そのセグメント内の状態トークン（Close/完了…）と担当者名で判定する（決定論・LLM 非使用）。"""
    done: set[str] = set()
    owner_is_person: dict[str, bool] = {}
    norm = unicodedata.normalize("NFKC", text)
    # 全 A## の出現位置を取り、隣接IDまでをセグメントとする。
    matches = list(_ACTION_ID_RE.finditer(norm))
    for i, m in enumerate(matches):
        aid = f"A{int(m.group(1)):02d}"
        seg = norm[m.end(): matches[i + 1].start() if i + 1 < len(matches) else len(norm)]
        seg_l = seg.lower()
        # セグメントが長すぎる（次IDが遠い＝表外の散文）ときは担当/状態の帰属が曖昧 → 状態は取るが窓を絞る。
        window = seg[:80]
        win_l = window.lower()
        if any(tok in win_l for tok in _DONE_TOKENS):
            done.add(aid)
        if person and person in window:
            owner_is_person[aid] = True
        else:
            owner_is_person.setdefault(aid, owner_is_person.get(aid, False))
    return done, owner_is_person


def _action_completed_between_meetings(qn: str, question: str) -> dict[str, Any] | None:
    """M(n) 時点で未完了・M(n+1) までに完了・担当=指定者 のアクションIDを列挙（idx34 型）。"""
    if not ("m01" in qn and "m02" in qn and "完了" in qn and ("担当" in qn or "さん" in qn)):
        return None
    person = _person_surname(question)
    if person is None:
        return None
    company = _resolve_case(question)
    if company is None:
        return None
    try:
        from src.rag.corpus import walk
        from src.rag.index import ocr_store
    except Exception:  # noqa: BLE001
        return None
    refs = [r for r in walk()
            if r.project == company and r.category == "meeting" and "会議録" in r.rel
            and "old" not in r.name.lower()]
    if len(refs) < 2:
        return None
    refs.sort(key=lambda r: r.rel)  # 会議録_YYYY-MM-DD → 日付順 = M01, M02, …
    m01_ref, m02_ref = refs[0], refs[1]
    t1 = ocr_store.lookup(m01_ref) or ""
    t2 = ocr_store.lookup(m02_ref) or ""
    if not t1.strip() or not t2.strip():
        return None
    done_m01, owner1 = _parse_meeting_actions(t1, person)
    done_m02, owner2 = _parse_meeting_actions(t2, person)
    owner_person = {aid for aid, ok in {**owner1, **owner2}.items()
                    if owner1.get(aid) or owner2.get(aid)}
    # M01 で未完了 かつ M02 で完了 かつ 担当=指定者。
    result_ids = sorted(aid for aid in done_m02
                        if aid not in done_m01 and aid in owner_person)
    if not result_ids:
        return None
    value = "、".join(result_ids)
    return _result(value, selection="action_completed_between_meetings_by_owner",
                   evidence={"case": company, "person": person,
                             "m01_doc": m01_ref.rel, "m02_doc": m02_ref.rel,
                             "done_m01": sorted(done_m01), "done_m02": sorted(done_m02),
                             "owner_person_ids": sorted(owner_person)})


# NOTE: ``_stage_metric_f1_diff`` (idx36) は DELIBERATELY 未配線。中間報告時点のF1は leaderboard に 8桁
# 丸めでしか焼かれておらず（最終Fだけ metrics.json にフル精度）、差はフル精度 gold と ~1e-9 ずれる。
# focused --dev で codex judge がこの差を Incorrect と判定した（gold=0.09619112771…, 計算値=0.09619112452…）。
# precision-first: 正解を焼けない問いは honest abstain（→LLM）にし wrong を作らない。ロジック自体は正しい
# ので関数とユニットテストは検証用に残す（証拠が焼けたら再配線できる）。
_LANES = (_leaderboard_setting_diff, _action_completed_between_meetings)


def resolve(question: str) -> dict[str, Any] | None:
    """クロス参照質問を既存ストアで一意束縛できる時だけ contract を返す（fail-open・OFF なら None）。"""
    if not enabled():
        return None
    try:
        qn = _norm(question)
        for lane in _LANES:
            res = lane(qn, question)
            if res is not None and _contract.is_contract(res) and res.get("value") is not None:
                return _contract.ensure_contract(res)
    except Exception:  # noqa: BLE001 — a broken lane must fall back, never break the answer path
        return None
    return None


# --------------------------------------------------------------------------- investigator tool (補助)
def _xref_lookup(question: str = "") -> dict[str, Any]:
    res = resolve(question or "")
    if res is not None:
        return res
    return _contract.make(None, engine="xref_coverage",
                          evidence={"applicable": True, "bound": False},
                          note="クロス参照で一意束縛できず（案件/上位行/段階の特定不可）")


def tool() -> "tuple[str, str, dict[str, Any], Callable[..., Any]] | None":
    """xref クロス参照 lookup ツール（OFF の時 None ⇒ ツール集合/MCP surface は byte-identical）。"""
    if not enabled():
        return None
    return (
        XREF_COVERAGE_LOOKUP,
        "クロス参照カバレッジ: 用語集の主略称/ローマ字別名で案件を束縛し、leaderboard 上位2件の設定差分・"
        "段階別メトリクス差・会議録アクションの M01→M02 完了×担当者 を事前計算済みストアから決定論で返す。",
        {"type": "object", "properties": {"question": _STR}, "required": ["question"]},
        _xref_lookup,
    )
