"""canonical_route — data/計算系質問を canonical データファイルへ *直行* させる取得ルート (SOT-2494).

The SOT-2486 診断で確認した通り、``both_abstain`` の ``retrieval_miss`` 群の多くは、答えの根拠が
案件フォルダの canonical なデータ資産 — ``train.xlsx`` / ``train.csv`` / 分析コード (``modeling.py``
等) / ノートブック (``01_eda.ipynb``) / ``leaderboard.csv`` — に在るのに、その根拠 chunk の BM25 ランクが
数百位まで沈み、chunk 検索(dense+BM25 RRF)では top-k に上がってこない。索引側 (rrf4/rerank/fact-index)
の改善は rejected 済み・かつ rank 600 超は索引改善では拾えない。

このモジュールは索引を *改善* するのではなく **迂回** する: 質問が指す **案件 (project)** を用語集で
解決し、質問の要求する **種別 (kind)** の canonical ファイルを **実行時にコーパスを歩いて発見** して直接
返す — chunk 検索を一切経由しない決定論ルートである。解決したファイルはそのまま既存の compute /
read_office / chart 系ツールへ渡せる (``expr`` を渡せばこの場で :mod:`compute_sandbox` を実行して値まで
返す)。

移植性 (受け入れ条件②: 秘密/コーパス固有値のハードコード無し)
------------------------------------------------------------------
* **案件解決** は用語集 (:mod:`src.rag.extract.glossary`, 社内用語集.docx を実行時に読む適応層) に委譲する。
  会社名・別名・主略称は一切コード内に埋め込まない。
* **ファイル発見** は :func:`src.rag.corpus.walk` を実行時に歩いて行う。canonical ファイル名を列挙して
  ハードコードしない。種別マッチャは *汎用のデータサイエンス/業務語彙* (``train`` / ``leaderboard`` /
  ``modeling`` / ``eda`` + コーパス共通の 03.データ/04.分析 カテゴリ番号 = :data:`src.rag.corpus.CATEGORY_BY_PREFIX`)
  だけを使い、この特定コーパスの秘密値 (パスワード・略称の答え) には一切依存しない。別コーパスに移しても
  「案件フォルダ配下の train / 分析コード / ノートブック / leaderboard」という構造さえ同じなら動く。

スコープ (追加のみ)
------------------
investigator の tool set に *追加* するのみ。既存ツール・既存の優先順位/ルーティングは変更しない
(優先ルーティングは後続 Issue)。純粋・決定論・ネットワーク非依存 (``expr`` 実行時も compute_sandbox の
restricted pandas のみ)。
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from src.rag.corpus import FileRef, nfc, walk
from src.rag.extract import glossary as _glossary
from src.rag.extract.glossary import Glossary
from src.rag.tools import contract
from src.rag.tools.compute_sandbox import run as compute_run

# --------------------------------------------------------------------------- canonical kinds
# A canonical file "kind" is (a) a *generic* keyword vocabulary that lets us infer what the question
# is after, and (b) a *structural* matcher (extension + generic stem token + corpus category tag) that
# discovers the real file at run time. No corpus-specific value (company/alias/password) appears here —
# only conventional DS/business filenames and the corpus-wide 00..06 folder taxonomy.

_ANALYSIS = "analysis"   # corpus.CATEGORY_BY_PREFIX["04"] — 04.分析
_DATA = "data"           # corpus.CATEGORY_BY_PREFIX["03"] — 03.データ
_CONTRACT = "contract"   # corpus.CATEGORY_BY_PREFIX["01"] — 01.契約
_PLAN = "plan"           # corpus.CATEGORY_BY_PREFIX["02"] — 02.計画


@dataclass(frozen=True)
class Kind:
    """One canonical-file kind: how to detect it in a question and how to find its file(s)."""

    name: str
    keywords: tuple[str, ...]          # generic vocabulary that signals this kind in a question (NFC)
    exts: tuple[str, ...]              # acceptable extensions (lowercase, no dot)
    stem_tokens: tuple[str, ...]       # generic filename stems that identify the file (lowercase); () = any
    categories: tuple[str, ...]        # corpus category tags to scope the search; () = any
    prefer: tuple[str, ...] = ()       # rel-substrings that rank the *canonical* copy first (NFC)


# Ordered by specificity: earlier kinds win when a question matches more than one keyword set.
KINDS: tuple[Kind, ...] = (
    # 分析コード / モデル / ハイパーパラメータ — the analysis source (.py), modeling.py is the canonical one.
    Kind(
        name="code",
        keywords=("分析コード", "コード", "スクリプト", "モデル", "ハイパーパラメータ", "ハイパラ",
                  "n_estimators", "learning_rate", "random_state", "max_depth", ".py", "実装",
                  "勾配ブースティング", "パイプライン"),
        exts=("py",),
        stem_tokens=(),
        categories=(_ANALYSIS,),
        prefer=("modeling", "/src/", "run_train"),
    ),
    # ノートブック / EDA
    Kind(
        name="notebook",
        keywords=("ノートブック", "notebook", "ipynb", "eda", "探索的"),
        exts=("ipynb",),
        stem_tokens=(),
        categories=(_ANALYSIS,),
        prefer=("notebooks/",),
    ),
    # leaderboard / 実験結果 / スコア表
    Kind(
        name="leaderboard",
        keywords=("leaderboard", "リーダーボード", "実験結果", "スコア表", "experiments"),
        exts=("csv",),
        stem_tokens=("leaderboard",),
        categories=(),
        prefer=("experiments/",),
    ),
    # スケジュール / WBS / ガント (計画フォルダの表)
    Kind(
        name="schedule",
        keywords=("スケジュール", "wbs", "ガント", "工程表"),
        exts=("xlsx", "xlsm"),
        stem_tokens=("スケジュール", "schedule", "wbs"),
        categories=(_PLAN, _DATA, ""),
        prefer=("02.", "計画"),
    ),
    # train データ (xlsx を優先。csv/xlsx 双方 canonical)
    Kind(
        name="train",
        keywords=("train.xlsx", "train.csv", "train", "学習データ", "分析データ", "データセット",
                  "学習用データ", "訓練データ"),
        exts=("xlsx", "csv", "xlsm"),
        stem_tokens=("train",),
        categories=(),
        prefer=("03.", "データ"),
    ),
    # 契約書 (契約情報)
    Kind(
        name="contract",
        keywords=("契約書", "契約金額", "着手金", "契約情報"),
        exts=("docx",),
        stem_tokens=("契約",),
        categories=(_CONTRACT,),
        prefer=("01.", "契約"),
    ),
)

# Explicit extension hints in a question narrow an otherwise multi-ext kind (e.g. train.xlsx vs train.csv).
# A plain ``\b`` fails to close after Japanese text (``train.xlsx において`` — 'x'→'に' is not a word
# boundary in Python's unicode ``\w``), so gate on "not an ascii alnum" instead.
_EXT_HINT = re.compile(r"\.(xlsx|xlsm|csv|ipynb|py|docx)(?![a-z0-9])", re.IGNORECASE)

# A notebook's markdown may describe a statistic incorrectly while its executed output (or backing
# table) is authoritative.  These generic terms identify notebook questions whose claim is derived
# from tabular data.  In that case :func:`infer_kinds` also exposes the project's ``train`` asset so
# ``canonical_route(question, expr=...)`` can take the canonical_route→compute path directly.
_DERIVED_COMPUTE_RE = re.compile(
    r"相関|平均|合計|総和|最大|最小|中央値|標準偏差|分散|件数|比率|割合|統計|集計|"
    r"correlation|corr\b|mean\b|sum\b|median\b|std\b|variance\b|idxmax\b|idxmin\b",
    re.IGNORECASE,
)


class CanonicalRouteError(ValueError):
    """Raised only for a malformed *explicit* kind argument; question-driven resolution never raises."""


# --------------------------------------------------------------------------- project resolution
def resolve_project(question: str | None, project: str | None, *,
                    corpus_dir: Path | None = None, glossary: Glossary | None = None) -> str | None:
    """Resolve the corpus project folder a question/hint refers to (NFC), or ``None``.

    An explicit ``project`` hint wins: it is matched (NFC, symmetric substring) against the walked
    corpus project folders, so a partial company name resolves to the full legal folder name and
    vice-versa. Otherwise the project is derived from ``question`` via the glossary adaptation layer
    (:meth:`Glossary.company_of`, self-discovered from 社内用語集.docx — no name is hard-coded here),
    then mapped to the matching corpus folder. Returns the canonical folder name or ``None`` when no
    single project can be determined (e.g. a cross-project aggregate or a non-project question).

    ``corpus_dir`` / ``glossary`` are dependency-injection seams (mirroring ``corpus_aggregate``): a
    test can point both at a synthetic mini-drive; production leaves them ``None`` (real corpus + the
    cached glossary).
    """
    projects = sorted({r.project for r in walk(corpus_dir) if r.project})
    if project:
        hint = nfc(project)
        hits = [p for p in projects if hint in nfc(p) or nfc(p) in hint]
        return hits[0] if hits else None
    if not question:
        return None
    g = glossary if glossary is not None else _glossary.load()
    company = g.company_of(question)
    if not company:
        return None
    c = nfc(company)
    hits = [p for p in projects if c in nfc(p) or nfc(p) in c]
    return hits[0] if hits else None


# --------------------------------------------------------------------------- kind inference
def _keyword_hit(keyword: str, q: str, ql: str) -> bool:
    """Membership test: case-insensitive for ascii tokens (F1/n_estimators), exact for Japanese."""
    return keyword.lower() in ql if keyword.isascii() else keyword in q


def infer_kinds(question: str) -> list[Kind]:
    """The canonical-file kinds a question is asking about, most-specific first (may be empty)."""
    q = nfc(question)
    ql = q.lower()
    kinds = [k for k in KINDS if any(_keyword_hit(kw, q, ql) for kw in k.keywords)]
    # A derived notebook claim needs both the document and its canonical backing data.  Append rather
    # than replace so ordinary discovery still returns the notebook first; an ``expr`` call below picks
    # the first runnable table as its execution primary.
    if any(k.name == "notebook" for k in kinds) and _DERIVED_COMPUTE_RE.search(q):
        train = next(k for k in KINDS if k.name == "train")
        if train not in kinds:
            kinds.append(train)
    return kinds


# --------------------------------------------------------------------------- file discovery
def _matches_kind(ref: FileRef, kind: Kind, ext_hint: str | None) -> bool:
    ext = ref.ext.lower()
    exts = (ext_hint,) if ext_hint and ext_hint in kind.exts else kind.exts
    if ext not in exts:
        return False
    if kind.categories and ref.category not in kind.categories:
        return False
    if kind.stem_tokens:
        stem = nfc(ref.name).lower()
        if not any(tok in stem for tok in kind.stem_tokens):
            return False
    return True


def _rank_key(ref: FileRef, kind: Kind) -> tuple:
    """Deterministic ordering: canonical (preferred-path) copies first, then shortest rel, then rel.

    ``kind.prefer`` is an *ordered* priority list — a file is ranked by the index of the first prefer
    token its rel contains (so ``modeling`` before ``run_train`` for the ``code`` kind), files matching
    no prefer token sort last among their kind.
    """
    rel = nfc(ref.rel)
    rel_l = rel.lower()
    prefer_rank = next((i for i, p in enumerate(kind.prefer) if p.lower() in rel_l), len(kind.prefer))
    # de-prioritize obvious non-canonical variants (old / draft / preview / backup copies)
    variant = 1 if re.search(r"(_old|old\.|draft|preview|backup|copy|コピー|旧)", rel_l) else 0
    return (prefer_rank, variant, rel.count("/"), rel)


def _discover_live(project: str | None, kind: Kind, *, ext_hint: str | None = None,
                   corpus_dir: Path | None = None) -> list[FileRef]:
    """Live walk+match implementation of :func:`discover` (bypasses the precomputed manifest).

    Pure over :func:`src.rag.corpus.walk`; the canonical copy (preferred path, non-variant) sorts first.
    The index-time manifest builder (SOT-2530) calls this directly so it precomputes from the true
    corpus scan rather than recursing into its own cache.
    """
    hits = [r for r in walk(corpus_dir)
            if (project is None or r.project == project) and _matches_kind(r, kind, ext_hint)]
    return sorted(hits, key=lambda r: _rank_key(r, kind))


def discover(project: str | None, kind: Kind, *, ext_hint: str | None = None,
             corpus_dir: Path | None = None) -> list[FileRef]:
    """All corpus files of ``kind`` in ``project`` (or corpus-wide when ``project`` is None), ranked.

    Fast path (SOT-2530 / 事前処理 #3): when the precomputed canonical manifest is enabled
    (``RAG_CANONICAL_MANIFEST``), the project→kind→rel resolution is an O(1) dict lookup over the
    index-time manifest instead of re-walking the corpus and running ``_matches_kind`` on every
    file per call (find_files/canonical_route run ~1,119× on the BUDGET questions). The manifest
    stores the *ext_hint-agnostic* ranked rel list for each (project, kind); an ``ext_hint`` narrows
    it here exactly as :func:`_matches_kind` would (only ``kind.exts`` members are ever a valid hint),
    so the result is byte-identical to the live scan. Any miss — a manifest that is disabled, absent,
    stale (corpus/KINDS signature drift), or missing this project — falls back to the live walk, so
    correctness/非劣化 is unchanged and only the slow path is removed. Default OFF ⇒ byte-identical.
    """
    if project is not None:
        from src.rag.index import canonical_manifest

        if canonical_manifest.enabled():
            rels = canonical_manifest.lookup_rels(project, kind.name, corpus_dir=corpus_dir)
            if rels is not None:
                by_rel = canonical_manifest.rel_index(corpus_dir)
                refs = [by_rel[r] for r in rels if r in by_rel]
                if ext_hint and ext_hint in kind.exts:
                    refs = [r for r in refs if r.ext.lower() == ext_hint]
                return refs
    return _discover_live(project, kind, ext_hint=ext_hint, corpus_dir=corpus_dir)


# --------------------------------------------------------------------------- record + tool
def _record(ref: FileRef, kind_name: str) -> dict[str, Any]:
    return {"rel": ref.rel, "project": ref.project, "category": ref.category,
            "ext": ref.ext, "kind": kind_name}


def _kind_by_name(name: str) -> Kind:
    for k in KINDS:
        if k.name == name:
            return k
    raise CanonicalRouteError(
        f"unknown kind {name!r} (expected one of {[k.name for k in KINDS]})")


def canonical_route(question: str | None = None, *, project: str | None = None,
                    kind: str | None = None, expr: str | None = None,
                    sheet: int | str | None = None, limit: int = 12,
                    corpus_dir: Path | None = None,
                    glossary: Glossary | None = None) -> dict[str, Any]:
    """Resolve a data/計算系 question to its canonical corpus file(s) — bypassing chunk retrieval.

    Parameters
    ----------
    question:
        The natural-language question. Its project is resolved via the glossary and its canonical-file
        kind(s) inferred from generic vocabulary. Optional when both ``project`` and ``kind`` are given.
    project:
        Optional project/company hint (NFC substring) overriding question-derived project resolution —
        the same disambiguation the ``compute`` tool takes (SOT-2487).
    kind:
        Optional explicit kind (``code`` / ``notebook`` / ``leaderboard`` / ``schedule`` / ``train`` /
        ``contract``) overriding keyword inference. Unknown values raise.
    expr:
        Optional single pandas expression. When given AND the primary resolved file is a csv/xlsx, the
        expression is executed against it via :mod:`compute_sandbox` and the compute contract is returned
        (enriched with the routing evidence) — the "直接 compute 接続". Ignored for non-tabular primaries.
    sheet:
        Optional xlsx sheet selector forwarded to compute (only relevant with ``expr``).
    limit:
        Cap on the number of resolved-file records returned in ``value`` (canonical copies first).

    Returns
    -------
    The unified ``{value, evidence, method}`` contract. Without ``expr``: ``value`` is a list of
    ``{rel, project, category, ext, kind}`` records (canonical copy first) — hand the top ``rel`` to
    ``compute`` / ``read_office`` / ``read_chart_values``. With a runnable ``expr``: the compute result.

    The route is strictly project-scoped: when no single project resolves (a cross-project aggregate or a
    non-project question) ``value`` is ``[]`` (caller stays abstained / uses ``corpus_aggregate``).
    """
    proj = resolve_project(question, project, corpus_dir=corpus_dir, glossary=glossary)
    ext_hint = None
    if question:
        m = _EXT_HINT.search(question)
        ext_hint = m.group(1).lower() if m else None

    if kind is not None:
        kinds = [_kind_by_name(nfc(kind).strip())]
    elif question:
        kinds = infer_kinds(question)
    else:
        kinds = list(KINDS)

    # The route is strictly project-scoped: its whole purpose is 案件解決 → その案件の canonical file 直行.
    # When no single project resolves (a cross-project aggregate like 『最も着手金が高い案件』, or a
    # non-project question), decline (value=[]) and leave it to corpus_aggregate / the other tools — never
    # fan a bare keyword out corpus-wide, which would hand back an ambiguous 全案件 file list.
    records: list[dict[str, Any]] = []
    primary: FileRef | None = None
    primary_kind: str | None = None
    tabular_primary: FileRef | None = None
    tabular_primary_kind: str | None = None
    if proj is not None:
        for k in kinds:
            for ref in discover(proj, k, ext_hint=ext_hint, corpus_dir=corpus_dir):
                rec = _record(ref, k.name)
                if rec in records:
                    continue
                records.append(rec)
                if primary is None:
                    primary, primary_kind = ref, k.name
                if tabular_primary is None and ref.ext.lower() in {"csv", "xlsx", "xlsm"}:
                    tabular_primary, tabular_primary_kind = ref, k.name
                if len(records) >= limit:
                    break
            if len(records) >= limit:
                break

    # When an expression is requested, the execution primary must be a table.  This lets a question
    # that explicitly names a notebook still re-run the notebook's statistic over its canonical train
    # data instead of silently ignoring the expression because the first discovered file is .ipynb.
    if expr and tabular_primary is not None:
        primary, primary_kind = tabular_primary, tabular_primary_kind

    evidence = {
        "project": proj,
        "project_hint": project,
        "resolved_kinds": [k.name for k in kinds],
        "ext_hint": ext_hint,
        "n_files": len(records),
        "primary": primary.rel if primary is not None else None,
        "route": "canonical-direct (chunk検索を迂回)",
    }
    method_common = {"question": question, "kind_arg": kind}

    # Direct compute connection: run the expression against the primary tabular file.
    if expr and primary is not None and primary.ext.lower() in {"csv", "xlsx", "xlsm"}:
        result = compute_run(primary, expr, sheet=sheet, project=proj)
        result["evidence"]["canonical_route"] = evidence
        result["method"]["route"] = "canonical_route→compute"
        result["method"]["primary_kind"] = primary_kind
        return result

    return contract.make(
        records, engine="canonical_route", evidence=evidence,
        **method_common,
    )
