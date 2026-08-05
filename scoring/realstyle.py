"""Real-style transcription benchmark — the SECOND generalization axis (SOT-2447 / A2).

The sealed hold-out (``scoring.selfimprove``) and valid30 (``scoring.gate1``) are both scored with
the *synthetic* question phrasings the RAG was tuned against, so a change can climb them while still
failing on the real test100 wording — exactly the #4/#5 regressions (valid/hold-out up, real Public
score down). This bench closes that gap: it transcribes the **real test100 question STYLE** (public,
GT-less) onto **known corpus facts** whose ground truth we extract deterministically, so we can score
transfer-to-production exactly, with no LLM/GCP.

Design invariants:
  * **Deterministic GT only** — every truth comes from a machine reader (bold-marker extraction, JSON
    config/metrics, pandas CSV stats, the 用語集 maps, structural version diff). Each item self-scores
    Perfect (validated in ``scoring.test_realstyle`` and ``selfimprove.self_test``).
  * **No valid30 leakage** — valid-anchored / cross-aggregate synth items are deliberately NOT drawn
    here (valid30 is the burnt-out dev set, isolated from the adoption decision — see
    ``scoring.overfit_check.adoption_axes``).
  * **Production phrasing** — questions use test100-derived templates, a *different surface form* from
    ``scoring.synth`` on purpose, so the bench measures style transfer rather than re-measuring synth.
  * **test100 distribution** — items are balanced to test100's answer-mode mixture
    (計算/比較/抽出/参照 ≈ calculate/compare/extract/lookup) and cover the 8 core archetypes.

    python -m scoring.realstyle            # build + write artifacts/realstyle_qa.jsonl, print mix
    python -m scoring.realstyle --show     # print the generated Q/A
"""
from __future__ import annotations

import argparse
import collections
import functools
import json
import re
from dataclasses import asdict, dataclass
from pathlib import Path

from config import settings
from scoring import dist_match, synth
from src.rag import corpus
from src.rag.corpus import nfc

REALSTYLE_PATH = settings.ARTIFACTS_DIR / "realstyle_qa.jsonl"

# Target answer-mode mixture, taken from the real test100 distribution (計算34/比較5/抽出35/参照26).
# The bench is balanced toward these fractions (capped by the deterministic-GT pool per family) and,
# when scored, aggregated under the same production weighting so the axis reflects the real mixture.
TEST100_FAMILY_FRACTIONS = {"calculate": 0.34, "compare": 0.05, "extract": 0.35, "lookup": 0.26}
DEFAULT_TARGET_N = 60  # extract pool ceiling (~21) / 0.35 ≈ 60 keeps the 35% extract share honest


@dataclass
class RealStyleItem:
    id: str
    archetype: str
    kind: str          # numeric | set | string (deterministic comparator)
    mode: str          # calculate | compare | extract | lookup (test100 answer-mode family)
    question: str      # test100-STYLE phrasing (distinct surface form from scoring.synth)
    truth: str
    company: str
    source: str        # corpus-relative provenance
    style_ref: str     # which test100 question style this transcribes


# ============================ test100-style phrasing templates ==================================
# Each transcribes a real test100 question shape onto a known fact. The surface form is intentionally
# different from scoring.synth so the bench measures transfer to the production wording.
def _phrase(archetype: str, company: str, param: str) -> tuple[str, str] | None:
    """Return (question, style_ref) for a re-phrased fact, or None if the archetype isn't re-phrased."""
    c = company
    if archetype == "config_model_type":  # test100 idx5/idx35 style
        return (f"{c}の分析において最良モデルとして採用されているモデルの種類は何ですか。",
                "test100:model-type")
    if archetype == "config_hyperparam":  # idx5 "...max_depthはいくらに設定されていますか。"
        return (f"{c}の最終報告で最良モデルとしているモデルにおいて、{param} はいくつに設定されていますか。"
                "数値で答えてください。", "test100:idx5-hyperparam")
    if archetype == "metric_score":  # idx35/idx36 "...F1スコア...はいくつですか"
        return (f"{c}の最終報告に記載されている {param} の値を小数第4位まで答えてください。",
                "test100:idx35-metric")
    if archetype == "data_shape":  # idx24-class data-shape lookup
        return (f"{c}の分析対象データにおける {param} はいくつですか。数値で答えてください。",
                "test100:data-shape")
    if archetype == "csv_column_mean":  # idx-class csv stat, terse
        return (f"{c}のtrain.csvにおいて、{param} 列の平均値を小数第2位まで求めてください。",
                "test100:csv-mean")
    if archetype == "csv_column_max":
        return (f"{c}のtrain.csvにおいて、{param} 列の最大値はいくつですか。",
                "test100:csv-max")
    if archetype == "glossary_formal":  # idx21/idx43 heavy abbreviation use
        return (f"社内用語集に照らして、社内略称「{param}」の正式名称を答えてください。",
                "test100:idx21-abbrev")
    if archetype == "glossary_abbrev":
        return (f"社内用語集において、「{param}」に対応する主略称（社内用語）を答えてください。",
                "test100:idx24-shortcode")
    if archetype == "version_diff":  # idx0/idx1/idx9/idx74/idx95 style
        return (f"{c}の{param}について、旧版から最新版への案件遂行に関連する実質的な変更を、"
                "変更前と変更後で挙げてください。", "test100:idx9-versiondiff")
    return None


def _param_from_id(archetype: str, item_id: str) -> str:
    """Recover the re-phrasing parameter (hyperparam name / column / abbreviation / file base)."""
    head = item_id.split("::")[0]
    if archetype in ("config_hyperparam", "metric_score", "data_shape"):
        return head.split("_", 1)[1] if "_" in head else head
    if archetype == "csv_column_mean":
        return head[len("csvmean_"):]
    if archetype == "csv_column_max":
        return head[len("csvmax_"):]
    if archetype in ("glossary_formal", "glossary_abbrev"):
        return item_id.split("::", 1)[1]
    if archetype == "version_diff":
        parts = item_id.split("::")
        return parts[2] if len(parts) >= 3 else ""
    return ""


# ============================ deterministic extract-family generator ============================
_BOLD_MARKER = "【太字箇所】"
# document category → a natural test100-style question for its bold-emphasis extraction.
_BOLD_STYLE = {
    "contract": ("{c}との契約書において、太字で記載されている箇所のうち、日付や純粋な数値を除いたものを"
                 "すべて抽出してください。", "test100:idx3-contract-bold"),
    "proposal": ("{c}の提案資料において、太字で強調されている項目を、日付や純粋な数値を除いてすべて"
                 "抽出してください。", "test100:idx11-bold-emphasis"),
    "meeting":  ("{c}の会議・報告資料において、太字で強調されている箇所を、日付や純粋な数値を除いて"
                 "すべて抽出してください。", "test100:idx71-bold-emphasis"),
    "internal": ("社内管理資料「{name}」において、太字で記載されている項目を、日付や純粋な数値を除いて"
                 "すべて挙げてください。", "test100:internal-bold"),
}
_NUMERIC_ONLY = re.compile(r"[0-9,.　/\-—〜~%％]+")


def _bold_terms(ref) -> list[str]:
    from src.rag.extract import extract as _extract

    try:
        doc = _extract(ref)
    except Exception:
        return []
    first = doc.text.split("\n", 1)[0]
    if not first.startswith(_BOLD_MARKER):
        return []
    terms = [t.strip() for t in first.replace(_BOLD_MARKER, "").split("/") if t.strip()]
    # drop pure date/number tokens (test100 idx3 explicitly excludes them) and de-duplicate in order.
    seen: dict[str, None] = {}
    for t in terms:
        if not re.fullmatch(_NUMERIC_ONLY, t):
            seen.setdefault(nfc(t), None)
    return list(seen)[:6]


@functools.lru_cache(maxsize=1)
def gen_document_extract() -> list[RealStyleItem]:
    # Cached: walks + extracts every docx (expensive); callers must treat the result as read-only.
    out: list[RealStyleItem] = []
    for ref in corpus.walk():
        if ref.ext != "docx":
            continue
        terms = _bold_terms(ref)
        if len(terms) < 2:
            continue
        company = synth._company_of(ref.path)
        tpl = _BOLD_STYLE.get(ref.category)
        if not tpl:
            continue
        q = tpl[0].format(c=company, name=Path(ref.rel).stem)
        out.append(RealStyleItem(
            id=f"rs_bold::{synth._rel(ref.path)}", archetype="document_extract", kind="set",
            mode="extract", question=q, truth="、".join(terms), company=company,
            source=synth._rel(ref.path), style_ref=tpl[1]))
    return out


# ============================ re-phrased deterministic-GT items ==================================
# valid30-derived synth archetypes are EXCLUDED so valid30 stays isolated from the adoption decision.
_VALID_DERIVED = {"cross_aggregate", "contract_amount", "highlight_set", "enum_set", "pivot_condition"}


@functools.lru_cache(maxsize=1)
def gen_rephrased() -> list[RealStyleItem]:
    # Cached (synth.build re-extracts the whole corpus); callers must treat the result as read-only.
    out: list[RealStyleItem] = []
    for it in synth.build():
        if it.archetype in _VALID_DERIVED:
            continue
        if it.source in ("valid official ground truth", "valid deterministic ground truth"):
            continue
        param = _param_from_id(it.archetype, it.id)
        phrased = _phrase(it.archetype, it.company, param)
        if phrased is None:
            continue
        q, ref = phrased
        out.append(RealStyleItem(
            id=f"rs::{it.id}", archetype=it.archetype, kind=it.kind,
            mode=dist_match.family(it.archetype), question=q, truth=it.truth,
            company=it.company, source=it.source, style_ref=ref))
    return out


# ============================ distribution-balanced assembly ====================================
def _target_counts(total: int) -> dict[str, int]:
    counts = {fam: round(frac * total) for fam, frac in TEST100_FAMILY_FRACTIONS.items()}
    # nudge the largest family so the rounded counts sum exactly to `total`.
    drift = total - sum(counts.values())
    if drift:
        biggest = max(counts, key=counts.get)
        counts[biggest] += drift
    return counts


def _take_round_robin(items: list[RealStyleItem], want: int) -> list[RealStyleItem]:
    """Pick `want` items spread evenly across the archetypes present, so a family's target is not
    swallowed by a single archetype (preserves the 8-archetype coverage the issue requires)."""
    by_arch: dict[str, list[RealStyleItem]] = collections.defaultdict(list)
    for it in sorted(items, key=lambda x: x.id):
        by_arch[it.archetype].append(it)
    order = sorted(by_arch)  # deterministic archetype order
    out: list[RealStyleItem] = []
    while len(out) < want and any(by_arch[a] for a in order):
        for a in order:
            if by_arch[a] and len(out) < want:
                out.append(by_arch[a].pop(0))
    return out


def build(total: int = DEFAULT_TARGET_N) -> list[RealStyleItem]:
    """Assemble the bench: deterministic candidates balanced to the test100 answer-mode mixture.

    A family short of its target contributes its whole pool (its production weight compensates in
    ``scoring.selfimprove.production_weighted_score``); never fabricates items to hit a count."""
    pools: dict[str, list[RealStyleItem]] = collections.defaultdict(list)
    for it in gen_document_extract() + gen_rephrased():
        pools[it.mode].append(it)
    for fam in pools:
        pools[fam].sort(key=lambda x: x.id)  # deterministic selection order

    targets = _target_counts(total)
    chosen: list[RealStyleItem] = []
    for fam, want in targets.items():
        chosen.extend(_take_round_robin(pools.get(fam, []), want))
    # If a shortfall dropped us below the ≥50 floor, backfill from the richest remaining pools.
    if len(chosen) < 50:
        picked = {i.id for i in chosen}
        rest = sorted((i for fam in pools for i in pools[fam] if i.id not in picked),
                      key=lambda x: (x.mode != "extract", x.id))
        chosen.extend(rest[:50 - len(chosen)])
    return chosen


def write(items: list[RealStyleItem], path: Path = REALSTYLE_PATH) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for it in items:
            f.write(json.dumps(asdict(it), ensure_ascii=False) + "\n")


def load(path: Path = REALSTYLE_PATH) -> list[RealStyleItem]:
    out: list[RealStyleItem] = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                out.append(RealStyleItem(**json.loads(line)))
    return out


def main(show: bool) -> None:
    items = build()
    write(items)
    by_mode = collections.Counter(i.mode for i in items)
    by_arch = collections.Counter(i.archetype for i in items)
    print(f"built {len(items)} real-style items → {REALSTYLE_PATH}")
    print("answer-mode mix:", dict(by_mode))
    print("target mix     :", _target_counts(len(items)))
    print("archetypes     :", dict(by_arch))
    if show:
        for it in items:
            print(f"[{it.mode:9}] ({it.archetype}) {it.question}\n      GT: {it.truth}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--show", action="store_true")
    args = ap.parse_args()
    main(args.show)
