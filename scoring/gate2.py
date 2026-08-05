"""関門2 — 教師データから推測した未知. 汎化を測る自作ホールドアウト.

30問(関門1)への過学習を検知するため、同じ共有ドライブから **機械的に** 新しい
質問/正解ペアを生成し(=正解はコードで抽出したものなので信頼できる)、RAGを採点する。
さらに **1案件を封印(sealed)** し、その案件由来の設問だけを別集計して「未知案件」への
転移を観察する。

生成される設問スキル (30問と同系だが別インスタンス):
  - 契約書の太字箇所の列挙        (office bold)
  - modeling.py 等のパラメータ値   (code assignment)
  - train.csv の集計 (平均/最大)   (pandas)
  - xlsx でハイライトされた行/セル (fill color)

    python -m scoring.gate2            # generate holdout, run RAG, score (+sealed breakdown)
    python -m scoring.gate2 --show     # just print the generated holdout Q/A
"""
from __future__ import annotations

import argparse
import re
from dataclasses import dataclass

import pandas as pd

from config import settings
from src.rag import corpus
from src.rag.extract import extract, glossary

# One project's questions are treated as a pure "unseen case" holdout.
SEALED_PROJECT_CODE = "AOBM"  # 株式会社青葉バイオメディカル機器


@dataclass
class QA:
    question: str
    answer: str
    skill: str
    project: str
    sealed: bool


def _company(code: str) -> str:
    g = glossary.load()
    return g.case_to_company.get(code, "")


def _bold_terms_qa(refs, sealed) -> list[QA]:
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
                skill="bold", project=r.project, sealed=r.project == sealed))
    return out


_ASSIGN = re.compile(r"^\s*([A-Za-z_][A-Za-z0-9_]*)\s*=\s*([0-9]+(?:\.[0-9]+)?)\s*$")


def _code_param_qa(refs, sealed) -> list[QA]:
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
                    answer=val, skill="code_param", project=r.project,
                    sealed=r.project == sealed))
                break
    return out


def _csv_aggregate_qa(refs, sealed) -> list[QA]:
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
            answer=str(mean), skill="csv_agg", project=r.project, sealed=r.project == sealed))
    return out


def generate_holdout() -> list[QA]:
    refs = corpus.walk()
    sealed = _company(SEALED_PROJECT_CODE)
    qa: list[QA] = []
    qa += _bold_terms_qa(refs, sealed)
    qa += _code_param_qa(refs, sealed)
    qa += _csv_aggregate_qa(refs, sealed)
    return qa


def main(show: bool) -> None:
    holdout = generate_holdout()
    print(f"generated {len(holdout)} holdout Q/A (sealed={_company(SEALED_PROJECT_CODE)})")
    # Adoption is decided on TWO axes now (SOT-2447): this sealed hold-out AND the real-style
    # transcription bench (`scoring.realstyle`). The sealed hold-out alone let the #5 regression
    # through; a change must clear both. See `scoring.overfit_check.assess_two_axis`.
    from scoring import realstyle

    rs = realstyle.build()
    print(f"real-style bench (2nd adoption axis): {len(rs)} items → judged by "
          f"`python -m scoring.overfit_check` (two-axis gate)")
    if show:
        for i, qa in enumerate(holdout):
            tag = "🔒" if qa.sealed else "  "
            print(f"{tag}[{i}] ({qa.skill}) {qa.question}\n      GT: {qa.answer}")
        print("\n---- real-style transcription bench (test100-style, deterministic GT) ----")
        for it in rs:
            print(f"[{it.mode:9}] ({it.archetype}) {it.question}\n      GT: {it.truth}")
        return

    from src.rag import generate as gen
    from scoring import crag

    preds = [gen.answer_question(qa.question)["answer"] for qa in holdout]
    pairs = list(zip(preds, [qa.answer for qa in holdout]))
    score, results = crag.score_pairs(pairs)

    seen = [r for r, qa in zip(results, holdout) if not qa.sealed]
    sealed_r = [r for r, qa in zip(results, holdout) if qa.sealed]

    def mean(rs):
        return sum(x["points"] for x in rs) / len(rs) if rs else float("nan")

    print("\n==================== 関門2 (auto-holdout) ====================")
    print(f"overall score: {score:+.4f}  (n={len(results)})")
    print(f"  未封印案件 : {mean(seen):+.4f}  (n={len(seen)})")
    print(f"  封印案件   : {mean(sealed_r):+.4f}  (n={len(sealed_r)})   <- 未知案件への転移")
    print("-" * 60)
    for qa, r in zip(holdout, results):
        tag = "🔒" if qa.sealed else "  "
        print(f"{tag} {r['judged']:11} [{qa.skill}] pred={r['pred'][:30]!r} gt={qa.answer[:24]!r}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--show", action="store_true")
    args = ap.parse_args()
    main(args.show)
