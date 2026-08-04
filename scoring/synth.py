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


def gen_csv(items: list[SynthItem], per_project_cols: int = 2) -> None:
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


def gen_glossary(items: list[SynthItem], limit: int = 15) -> None:
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


def build() -> list[SynthItem]:
    items: list[SynthItem] = []
    gen_config(items)
    gen_metrics(items)
    gen_csv(items)
    gen_glossary(items)
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
