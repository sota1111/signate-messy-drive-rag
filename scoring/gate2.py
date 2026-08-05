"""関門2 — 汎化ゲート (封印AOBM + synth holdout, 新エージェント配線 SOT-2476).

過適合ガードの主指標。関門1(教師30問)への過学習を検知するため、同じ共有ドライブから
**機械的に** 新しい質問/正解ペアを生成し(=正解はコードで抽出したものなので信頼できる)、
それを **本番Gemini investigatorエージェント経路** (`src.rag.run` の ``investigator`` /
``resolve`` チェーン、SOT-2469) で解いて採点する。旧 ``generate.answer_question`` (legacy
text-only) ではなく、本番と同一の解答経路を使うことで「本番システムの汎化」を測る。

さらに **封印(sealed)案件** の設問だけを別集計し「未知案件」への転移(=汎化スコア)を得る。
封印案件集合は単一の真実源 :func:`scoring.selfimprove.sealed_companies` から取得し、seen/sealed
分割の **隔離不変条件を厳守** する(:func:`check_isolation` が違反時に :class:`IsolationError` を
送出。会社名の NFC/NFD 正規化揺れによる漏洩もここで検知する)。

出力は昇格判定に使える **機械可読の汎化スコア** (overall / seen / sealed(汎化) / gap / skill別
内訳 / ``usable`` フラグ)で、``--json`` / ``--out`` で JSON も出せる。

生成される設問スキル (30問と同系だが別インスタンス):
  - 契約書の太字箇所の列挙        (office bold)
  - modeling.py 等のパラメータ値   (code assignment)
  - train.csv の集計 (平均)        (pandas)
  - synth ベンチ (config/metrics/csv/glossary/version_diff/pivot/cross)  (scoring.synth)

    python -m scoring.gate2                 # generate holdout, solve with agent, score
    python -m scoring.gate2 --show          # just print the generated holdout Q/A
    python -m scoring.gate2 --gen resolve   # solve with the full investigator→verifier→tiebreak chain
    python -m scoring.gate2 --json --out artifacts/gate2.json   # machine-readable generalization score
"""
from __future__ import annotations

import argparse
import json
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

import pandas as pd

from config import settings
from src.rag import corpus
from src.rag.corpus import nfc
from src.rag.extract import extract, glossary

# The representative sealed project used as the visible "unseen case" exemplar; the AUTHORITATIVE
# sealed company set is :func:`scoring.selfimprove.sealed_companies` (single source of truth).
SEALED_PROJECT_CODE = "AOBM"  # 株式会社青葉バイオメディカル機器

# question -> answer text. Default = the production Gemini investigator agent (see make_answerer).
Answerer = Callable[[str], str]
# pairs=[(pred, truth), ...] -> (mean_score, per-item results); default = scoring.crag.score_pairs.
Judge = Callable[[list], tuple]


class IsolationError(RuntimeError):
    """Raised when the sealed-project isolation invariant is violated (a company appears in both the
    seen and sealed slices — e.g. a NFC/NFD normalization leak, or a sealed company misrouted into
    the development slice)."""


@dataclass
class QA:
    question: str
    answer: str
    skill: str
    company: str
    sealed: bool


def _company(code: str) -> str:
    g = glossary.load()
    return g.case_to_company.get(code, "")


def sealed_companies() -> set[str]:
    """The authoritative NFC-normalized sealed company set (delegates to scoring.selfimprove)."""
    from scoring.selfimprove import sealed_companies as _sc

    return {nfc(c) for c in _sc()}


def _is_sealed(company: str, sealed: set[str]) -> bool:
    return nfc(company) in sealed


# --------------------------------------------------------------------------- holdout generation
def _bold_terms_qa(refs, sealed: set[str]) -> list[QA]:
    out = []
    for r in refs:
        if r.category != "contract" or r.ext != "docx":
            continue
        doc = extract(r)
        first = doc.text.split("\n", 1)[0]
        if not first.startswith("【太字箇所】"):
            continue
        terms = [t.strip() for t in first.replace("【太字箇所】", "").split("/") if t.strip()]
        terms = [t for t in terms if not re.fullmatch(r"[0-9,./-]+", t)][:6]
        if len(terms) >= 2:
            out.append(QA(
                question=f"{r.project} の契約書で太字（強調）で記載されている項目のうち、"
                         f"日付や純粋な数値を除いたものを挙げてください。",
                answer="、".join(terms),
                skill="bold", company=r.project, sealed=_is_sealed(r.project, sealed)))
    return out


_ASSIGN = re.compile(r"^\s*([A-Za-z_][A-Za-z0-9_]*)\s*=\s*([0-9]+(?:\.[0-9]+)?)\s*$")


def _code_param_qa(refs, sealed: set[str]) -> list[QA]:
    out = []
    for r in refs:
        if r.ext != "py" or "modeling" not in r.name:
            continue
        text = r.path.read_text(encoding="utf-8", errors="ignore")
        for m in _ASSIGN.finditer(text):
            name, val = m.group(1), m.group(2)
            if name in ("random_state", "n_estimators", "max_depth", "test_size", "n_splits",
                        "learning_rate", "max_iter"):
                out.append(QA(
                    question=f"{r.project} の modeling.py において、{name} に設定されている値は何ですか。数値で答えてください。",
                    answer=val, skill="code_param", company=r.project,
                    sealed=_is_sealed(r.project, sealed)))
                break
    return out


def _csv_aggregate_qa(refs, sealed: set[str]) -> list[QA]:
    out = []
    for r in refs:
        if r.ext != "csv" or r.name != "train.csv" or r.category != "data":
            continue
        try:
            df = pd.read_csv(r.path)
        except Exception:
            continue
        num = df.select_dtypes("number")
        if num.shape[1] == 0:
            continue
        col = num.columns[0]
        mean = round(float(num[col].mean()), 2)
        out.append(QA(
            question=f"{r.project} の train.csv において、列「{col}」の平均値を小数第2位で四捨五入して答えてください。",
            answer=str(mean), skill="csv_agg", company=r.project,
            sealed=_is_sealed(r.project, sealed)))
    return out


def _synth_qa(sealed: set[str]) -> list[QA]:
    """The machine-generated synth bench (scoring.synth), tagged seen/sealed by company."""
    from scoring import synth

    return [QA(question=it.question, answer=it.truth, skill=it.archetype, company=it.company,
               sealed=_is_sealed(it.company, sealed))
            for it in synth.build()]


def generate_holdout() -> list[QA]:
    """封印AOBM + synth holdout。会社ごとに seen/sealed を付与し、隔離不変条件を検証して返す。"""
    sealed = sealed_companies()
    refs = corpus.walk()
    qa: list[QA] = []
    qa += _bold_terms_qa(refs, sealed)
    qa += _code_param_qa(refs, sealed)
    qa += _csv_aggregate_qa(refs, sealed)
    qa += _synth_qa(sealed)
    check_isolation(qa, sealed)
    return qa


# --------------------------------------------------------------------------- sealed isolation guard
def check_isolation(holdout: list[QA], sealed: set[str] | None = None) -> dict:
    """Enforce 封印projectの隔離厳守 and return isolation diagnostics.

    Invariants (raise :class:`IsolationError` on violation):
      1. every company carries ONE consistent sealed label across its items — a company that shows
         up as both sealed and seen means a normalization leak (NFC vs NFD folder name);
      2. no company in the authoritative ``sealed`` set appears in the seen slice.
    A vacuous sealed slice is NOT an error here (the corpus may not contain a sealed project in a
    test fixture); it is surfaced via the returned ``sealed_n`` / the report's ``usable`` flag.
    """
    sealed = sealed_companies() if sealed is None else sealed
    labels: dict[str, set[bool]] = {}
    for qa in holdout:
        labels.setdefault(nfc(qa.company), set()).add(qa.sealed)
    mixed = sorted(c for c, ls in labels.items() if len(ls) > 1)
    if mixed:
        raise IsolationError(
            f"company labeled both sealed and seen (NFC/NFD normalization leak?): {mixed}")
    seen_companies = {c for c, ls in labels.items() if ls == {False}}
    leaked = sorted(seen_companies & sealed)
    if leaked:
        raise IsolationError(f"sealed company leaked into the seen slice: {leaked}")
    return {
        "sealed_companies": sorted(sealed),
        "seen_companies": sorted(seen_companies),
        "sealed_present": sorted(c for c, ls in labels.items() if ls == {True}),
    }


# --------------------------------------------------------------------------- answer backend wiring
def make_answerer(gen: str = "investigator", hard: bool = False) -> Answerer:
    """Wire the holdout solver to the production answer path (SOT-2469). Reuses ``run.make_worker``
    so gate2 shares the exact backend selection with the real submission pipeline. ``gen`` defaults
    to ``investigator`` (Gemini tool-agent, production); ``resolve`` runs the full
    investigator→verifier→tiebreak chain."""
    from src.rag.run import make_worker

    work = make_worker(gen, hard)

    def answerer(question: str) -> str:
        return work(0, question)[1]["answer"]

    return answerer


def solve(holdout: list[QA], answerer: Answerer, workers: int = 8) -> list[str]:
    """Answer every holdout question with ``answerer`` (parallel), preserving input order."""
    preds: list[str | None] = [None] * len(holdout)
    with ThreadPoolExecutor(max_workers=max(1, workers)) as ex:
        futs = {ex.submit(answerer, qa.question): i for i, qa in enumerate(holdout)}
        for fut in as_completed(futs):
            preds[futs[fut]] = fut.result()
    return [p if p is not None else settings.ABSTAIN for p in preds]


# --------------------------------------------------------------------------- report
@dataclass
class Slice:
    n: int
    score: float

    def to_dict(self) -> dict:
        return {"n": self.n, "score": None if self.score != self.score else round(self.score, 4)}


@dataclass
class GateReport:
    """The machine-readable generalization score used for promotion判定 (acceptance #1)."""

    items: list[dict] = field(default_factory=list)  # {skill, company, sealed, judged, points, pred, gt}
    isolation: dict = field(default_factory=dict)

    @staticmethod
    def _mean(rows: list[dict]) -> float:
        return sum(r["points"] for r in rows) / len(rows) if rows else float("nan")

    def to_dict(self) -> dict:
        seen = [r for r in self.items if not r["sealed"]]
        sealed = [r for r in self.items if r["sealed"]]
        overall = self._mean(self.items)
        seen_s, sealed_s = self._mean(seen), self._mean(sealed)
        gap = (seen_s - sealed_s) if (seen_s == seen_s and sealed_s == sealed_s) else None

        by_skill: dict[str, list[dict]] = {}
        for r in self.items:
            by_skill.setdefault(r["skill"], []).append(r)
        verdicts: dict[str, int] = {}
        for r in self.items:
            verdicts[r["judged"]] = verdicts.get(r["judged"], 0) + 1

        return {
            "n": len(self.items),
            "overall_score": None if overall != overall else round(overall, 4),
            "seen": Slice(len(seen), seen_s).to_dict(),
            "sealed": Slice(len(sealed), sealed_s).to_dict(),  # <- 汎化 (未知案件への転移)
            # positive gap = the system does worse on unseen sealed cases than on seen ones = overfit.
            "generalization_gap": None if gap is None else round(gap, 4),
            "by_skill": {s: Slice(len(rs), self._mean(rs)).to_dict()
                         for s, rs in sorted(by_skill.items())},
            "verdicts": dict(sorted(verdicts.items())),
            # A generalization score is only usable for a promotion decision when BOTH slices are
            # non-empty (otherwise seen/sealed comparison is undefined).
            "usable": bool(seen) and bool(sealed),
            "isolation": self.isolation,
        }

    def render(self) -> str:
        d = self.to_dict()

        def fmt(x):
            return "  n/a " if x is None else f"{x:+.4f}"

        lines = [
            "==================== 関門2 (汎化ゲート: 封印 + synth holdout) ====================",
            f"overall score : {fmt(d['overall_score'])}  (n={d['n']})",
            f"  未封印(seen) : {fmt(d['seen']['score'])}  (n={d['seen']['n']})",
            f"  封印(sealed) : {fmt(d['sealed']['score'])}  (n={d['sealed']['n']})   <- 汎化(未知案件への転移)",
            f"  汎化gap      : {fmt(d['generalization_gap'])}   (seen - sealed; +ほど過適合)",
            f"  usable(昇格判定に使用可): {d['usable']}",
            "  by skill:",
        ]
        for s, sl in d["by_skill"].items():
            lines.append(f"    {s:<16} n={sl['n']:>2}  score={fmt(sl['score'])}")
        lines.append(f"  verdicts: {d['verdicts']}")
        return "\n".join(lines)


def evaluate(holdout: list[QA], answerer: Answerer, judge: Judge | None = None,
             workers: int = 8) -> GateReport:
    """Solve ``holdout`` with ``answerer``, score with ``judge`` (default scoring.crag.score_pairs),
    and build the generalization :class:`GateReport`. ``answerer`` / ``judge`` are injectable so the
    report machinery is testable without any network (see scoring/test_gate2.py)."""
    if judge is None:
        from scoring import crag
        judge = crag.score_pairs

    isolation = check_isolation(holdout)
    preds = solve(holdout, answerer, workers=workers)
    pairs = list(zip(preds, [qa.answer for qa in holdout]))
    _score, results = judge(pairs)
    items = [
        {"skill": qa.skill, "company": qa.company, "sealed": qa.sealed,
         "judged": r["judged"], "points": r["points"], "pred": r["pred"], "gt": qa.answer}
        for qa, r in zip(holdout, results)
    ]
    return GateReport(items=items, isolation=isolation)


# --------------------------------------------------------------------------- CLI
def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--show", action="store_true", help="just print the generated holdout Q/A")
    ap.add_argument("--gen", default="investigator",
                    help="answer backend (SOT-2469): investigator (production, default) | resolve "
                         "(full investigator→verifier→tiebreak chain) | gated | gemini/opus (legacy)")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--limit", type=int, default=None, help="solve only the first N holdout items (smoke)")
    ap.add_argument("--json", dest="as_json", action="store_true",
                    help="print the JSON generalization report instead of the human summary")
    ap.add_argument("--out", type=Path, default=None, help="also write the JSON report to this path")
    args = ap.parse_args(argv)

    holdout = generate_holdout()
    if args.limit:
        holdout = holdout[: args.limit]
    n_sealed = sum(qa.sealed for qa in holdout)
    print(f"generated {len(holdout)} holdout Q/A ({n_sealed} sealed / {len(holdout) - n_sealed} seen; "
          f"sealed set = {sorted(sealed_companies())})")

    if args.show:
        for i, qa in enumerate(holdout):
            tag = "🔒" if qa.sealed else "  "
            print(f"{tag}[{i}] ({qa.skill}) {qa.question}\n      GT: {qa.answer}")
        return 0

    report = evaluate(holdout, make_answerer(args.gen), workers=args.workers)
    payload = report.to_dict()
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"wrote {args.out}")
    print(json.dumps(payload, ensure_ascii=False, indent=2) if args.as_json else report.render())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
