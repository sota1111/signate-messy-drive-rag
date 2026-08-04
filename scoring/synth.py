"""Synthesize labelled Q/A benchmark items with programmatically-extracted ground truth.

The official gpt-5.2 judge is unavailable locally and the Gemini proxy judge is too noisy to
optimize against. Instead we build a benchmark whose answers we *know* — extracted deterministically
from machine-readable corpus files — so `scoring.deterministic` can score exactly, per archetype:

  * ``config_model_type`` / ``config_hyperparam`` — values in each project's ``project_config.json``
  * ``metric_score`` / ``data_shape``           — numbers in each project's ``metrics.json``
  * ``csv_column_mean`` / ``csv_column_max``     — pandas stats over each project's ``train.csv``
  * ``glossary_formal`` / ``glossary_abbrev``    — the 社内用語集 abbreviation ↔ formal-name maps

Every item carries ``kind`` (numeric/set/string) so the deterministic scorer knows the comparator.
GT correctness is self-validated in `scoring.selfimprove` (each truth scored against itself → Perfect).

    python -m scoring.synth                      # build + write artifacts/synth_qa.jsonl, print counts
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import pandas as pd

from config import settings
from src.rag.corpus import nfc

SYNTH_PATH = settings.ARTIFACTS_DIR / "synth_qa.jsonl"
# On-disk Japanese path segments are NFD; compare after NFC-normalizing (never hardcode an NFC
# segment into a Path — rglob would fail to match). See src.rag.corpus.nfc.
_PROJECT_MARKER = nfc("プロジェクト")


@dataclass
class SynthItem:
    id: str
    archetype: str
    kind: str  # numeric | set | string
    question: str
    truth: str
    company: str
    source: str  # corpus-relative source file


def _company_of(path: Path) -> str:
    """Company folder name (the segment directly under プロジェクト), else the parent dir name.

    Filesystem segments may be NFD-normalized, so match/return via NFC (`corpus.nfc`)."""
    parts = [nfc(p) for p in path.parts]
    if _PROJECT_MARKER in parts:
        i = parts.index(_PROJECT_MARKER)
        if i + 1 < len(parts):
            return parts[i + 1]
    return nfc(path.parent.name)


def _rel(path: Path) -> str:
    try:
        return str(path.relative_to(settings.CORPUS_DIR))
    except ValueError:
        return str(path)


def _fmt_num(x: float, dp: int) -> str:
    return f"{round(float(x), dp):.{dp}f}"


def _load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8", errors="replace"))


# --- one canonical file per company (avoid duplicate configs with divergent values) -------------
def _canonical_by_company(name: str) -> list[Path]:
    best: dict[str, Path] = {}
    # rglob from CORPUS_DIR (an ASCII-safe root); an NFC-hardcoded subdir Path would not match NFD.
    for p in sorted(settings.CORPUS_DIR.rglob(name)):
        if _PROJECT_MARKER not in [nfc(x) for x in p.parts]:
            continue  # only project files (skip 社内管理 etc.)
        c = _company_of(p)
        # prefer the shortest path (the primary copy directly under 04.分析/analysis_outputs)
        if c not in best or len(p.parts) < len(best[c].parts):
            best[c] = p
    return [best[c] for c in sorted(best)]


# ================================ generators ====================================================
def gen_config(items: list[SynthItem]) -> None:
    for p in _canonical_by_company("project_config.json"):
        try:
            cfg = _load_json(p)
        except Exception:
            continue
        company, rel = _company_of(p), _rel(p)
        mt = cfg.get("model_type")
        if isinstance(mt, str) and mt:
            items.append(SynthItem(
                f"cfg_mt::{company}", "config_model_type", "string",
                f"{company}のproject_config.jsonで指定されている model_type は何ですか。",
                nfc(mt), company, rel))
        for key in ("random_state", "test_size"):
            v = cfg.get(key)
            if isinstance(v, (int, float)) and not isinstance(v, bool):
                dp = 0 if float(v).is_integer() else 2
                items.append(SynthItem(
                    f"cfg_{key}::{company}", "config_hyperparam", "numeric",
                    f"{company}のproject_config.jsonの {key} の値はいくつですか。",
                    _fmt_num(v, dp), company, rel))


def gen_metrics(items: list[SynthItem]) -> None:
    score_keys = ("accuracy", "f1_macro", "auc_roc")
    shape_keys = ("row_count", "feature_count", "train_rows", "test_rows")
    for p in _canonical_by_company("metrics.json"):
        try:
            m = _load_json(p)
        except Exception:
            continue
        company, rel = _company_of(p), _rel(p)
        for key in score_keys:
            v = m.get(key)
            if isinstance(v, (int, float)) and not isinstance(v, bool):
                items.append(SynthItem(
                    f"met_{key}::{company}", "metric_score", "numeric",
                    f"{company}の metrics.json における {key} の値を小数第4位まで答えてください。",
                    _fmt_num(v, 4), company, rel))
        for key in shape_keys:
            v = m.get(key)
            if isinstance(v, (int, float)) and not isinstance(v, bool):
                items.append(SynthItem(
                    f"shp_{key}::{company}", "data_shape", "numeric",
                    f"{company}の metrics.json における {key} はいくつですか。",
                    _fmt_num(v, 0), company, rel))


def gen_csv(items: list[SynthItem], per_project_cols: int = 5) -> None:
    for p in _canonical_by_company("train.csv"):
        try:
            df = pd.read_csv(p)
        except Exception:
            continue
        company, rel = _company_of(p), _rel(p)
        num = [c for c in df.columns
               if pd.api.types.is_numeric_dtype(df[c]) and str(c).lower() not in ("id", "index")]
        # skip binary/target-like columns (uninformative mean/max); prefer higher-cardinality cols
        num = [c for c in num if df[c].nunique(dropna=True) > 3]
        for c in num[:per_project_cols]:
            mean_v = df[c].mean()
            max_v = df[c].max()
            if pd.notna(mean_v):
                items.append(SynthItem(
                    f"csvmean_{c}::{company}", "csv_column_mean", "numeric",
                    f"{company}の train.csv の {c} 列の平均値を小数第2位まで答えてください。",
                    _fmt_num(mean_v, 2), company, rel))
            if pd.notna(max_v):
                dp = 0 if float(max_v).is_integer() else 2
                items.append(SynthItem(
                    f"csvmax_{c}::{company}", "csv_column_max", "numeric",
                    f"{company}の train.csv の {c} 列の最大値はいくつですか。",
                    _fmt_num(max_v, dp), company, rel))


def gen_glossary(items: list[SynthItem], limit: int = 25) -> None:
    from src.rag.extract import glossary as G

    try:
        g = G.load()
    except Exception:
        return
    # abbreviation -> formal
    for i, (ab, formal) in enumerate(sorted(g.abbrev_to_formal.items())):
        if i >= limit:
            break
        if ab and formal:
            items.append(SynthItem(
                f"gl_formal::{ab}", "glossary_formal", "string",
                f"社内用語集で、社内用語「{ab}」の正式名称は何ですか。",
                nfc(formal), "社内管理", "社内管理/社内用語集"))
    # formal -> abbreviation
    for i, (formal, ab) in enumerate(sorted(g.formal_to_abbrev.items())):
        if i >= limit:
            break
        if ab and formal:
            items.append(SynthItem(
                f"gl_abbrev::{formal}", "glossary_abbrev", "string",
                f"社内用語集で、正式名称「{formal}」に対応する社内用語（略称）は何ですか。",
                nfc(ab), "社内管理", "社内管理/社内用語集"))


def gen_version_diff(items: list[SynthItem]) -> None:
    """One item per resolvable version pair (old→latest structural diff).

    Ground truth is the deterministic structural diff itself (``src.rag.diffpair``), so scoring a RAG
    run against it measures whether the generator correctly *routes* a diff question to the structural
    differ and reproduces the "変更前 → 変更後" answer — the same path that answers valid idx9."""
    from src.rag import diffpair

    try:
        pairs = diffpair.find_pairs()
    except Exception:
        return
    for p in pairs:
        company = nfc(p.new.project) or nfc(p.old.project)
        base = nfc(p.base)
        if not company or not base:
            continue
        q = (f"{company}の{base}について、旧版と最新版を比較し、"
             "変更された箇所を変更前と変更後で答えてください。")
        try:
            truth = diffpair.answer_question(q)
        except Exception:
            truth = None
        if not truth:  # unresolvable / ambiguous / no substantive change → skip (not a benchmark item)
            continue
        items.append(SynthItem(
            f"vdiff::{company}::{base}", "version_diff", "string",
            q, truth, company, _rel(p.new.path)))


def gen_pivot_condition(items: list[SynthItem]) -> None:
    """One item per PivotTable measure and per active AutoFilter in the corpus.

    Ground truth is the deterministic ``src.rag.pivotcond`` answer, so scoring a RAG run against it
    measures whether the generator routes a condition question to the pivot/filter reader and
    reproduces the answer — the same path that answers valid idx6 / idx11 / idx21."""
    from src.rag import pivotcond

    try:
        bench = pivotcond.benchmark_items()
    except Exception:
        return
    for n, b in enumerate(bench):
        if not b.truth:
            continue
        items.append(SynthItem(
            f"pivotcond::{b.company}::{b.source}::{n}",
            "pivot_condition", "string", b.question, nfc(b.truth), b.company, b.source))


def gen_cross_aggregate(items: list[SynthItem]) -> None:
    """Known-valid cross-document calculations, answered by the deterministic compute route."""
    import pandas as pd
    questions = pd.read_csv(settings.QUESTIONS_VALID).set_index("index")
    truths = pd.read_csv(settings.VALID_GROUND_TRUTH, header=None, index_col=0)
    for idx in (3, 8, 13):
        q = str(questions.at[idx, "question"])
        truth = str(truths.at[idx, 1])
        items.append(SynthItem(f"cross::{idx}", "cross_aggregate", "numeric", q, truth,
                               "横断", "valid deterministic ground truth"))


# valid index -> (archetype, kind, company). Ground truth is the official valid_txt.csv answer (known
# correct), so these enrich enum/highlight/contract coverage without risking a fabricated GT. The
# company is the real owning project so sealing that project moves the item into the hold-out slice.
_VALID_ANCHORED: dict[int, tuple[str, str, str]] = {
    0:  ("highlight_set", "set",     "株式会社青潮モビリティサービス"),
    23: ("highlight_set", "set",     "株式会社青潮モビリティサービス"),
    25: ("highlight_set", "set",     "株式会社東都人材プラットフォーム"),
    20: ("enum_set",      "set",     "青葉与信マネジメント株式会社"),
    26: ("enum_set",      "set",     "株式会社青葉バイオメディカル機器"),
    15: ("enum_set",      "set",     "横断"),
    12: ("contract_amount", "numeric", "京橋信用ソリューションズ株式会社"),
}


def gen_valid_anchored(items: list[SynthItem]) -> None:
    """Enum / highlight / contract-amount items with official (known-correct) valid ground truth.

    These archetypes have no clean machine-GT generator across every project, so we anchor a handful
    to the official valid answers. They score Perfect against themselves (self-test invariant) and,
    labelled with their real owning company, participate in the hold-out split like any other item."""
    try:
        questions = pd.read_csv(settings.QUESTIONS_VALID).set_index("index")
        truths = pd.read_csv(settings.VALID_GROUND_TRUTH, header=None, index_col=0)
    except Exception:
        return
    for idx, (arch, kind, company) in _VALID_ANCHORED.items():
        try:
            q = str(questions.at[idx, "question"])
            truth = nfc(str(truths.at[idx, 1]))
        except Exception:
            continue
        if not q or not truth:
            continue
        items.append(SynthItem(f"{arch}::valid{idx}", arch, kind, q, truth, company,
                               "valid official ground truth"))


def build() -> list[SynthItem]:
    items: list[SynthItem] = []
    gen_config(items)
    gen_metrics(items)
    gen_csv(items)
    gen_glossary(items)
    gen_version_diff(items)
    gen_pivot_condition(items)
    gen_cross_aggregate(items)
    gen_valid_anchored(items)
    return items


def write(items: list[SynthItem], path: Path = SYNTH_PATH) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for it in items:
            f.write(json.dumps(asdict(it), ensure_ascii=False) + "\n")


def load(path: Path = SYNTH_PATH) -> list[SynthItem]:
    out: list[SynthItem] = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                out.append(SynthItem(**json.loads(line)))
    return out


if __name__ == "__main__":
    import collections

    its = build()
    write(its)
    by_arch = collections.Counter(i.archetype for i in its)
    print(f"built {len(its)} synthetic items → {SYNTH_PATH}")
    for arch in sorted(by_arch):
        print(f"  {arch:20} {by_arch[arch]:3}")
