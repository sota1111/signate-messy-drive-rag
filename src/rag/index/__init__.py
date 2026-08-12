"""Build and persist a hybrid retrieval index over the extracted corpus.

Each document is chunked (provenance header + body), embedded with Vertex text-embedding,
and stored alongside a BM25 lexical index. Persisted under INDEX_DIR:
  chunks.jsonl   — one chunk record per line (id, project, category, file, rel, kind, text)
  embeddings.npy — float32 matrix aligned to chunks.jsonl
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, asdict
from pathlib import Path

import numpy as np

from config import settings
from src.rag import corpus, llm
# NOTE: extract/* (office→openpyxl/pptx, passwords→msoffcrypto) is imported lazily inside build()
# so the SERVE path (generate→retrieve→index) needs none of the heavy extraction deps.

CHUNK_CHARS = 1200
CHUNK_OVERLAP = 180
MAX_CHUNKS_PER_DOC = 60


@dataclass
class Chunk:
    id: int
    project: str
    category: str
    file: str
    rel: str
    kind: str
    text: str


def _split(text: str, size: int = CHUNK_CHARS, overlap: int = CHUNK_OVERLAP) -> list[str]:
    text = text.strip()
    if len(text) <= size:
        return [text] if text else []
    # prefer splitting on blank lines / newlines near the boundary
    chunks, start = [], 0
    while start < len(text):
        end = min(start + size, len(text))
        if end < len(text):
            nl = text.rfind("\n", start + size // 2, end)
            if nl != -1:
                end = nl
        chunks.append(text[start:end].strip())
        if end >= len(text):
            break
        start = max(end - overlap, start + 1)
    return [c for c in chunks if c]


def _header(ref: corpus.FileRef) -> str:
    cat = ref.category or "-"
    proj = ref.project or "-"
    return f"[案件: {proj} | 区分: {cat} | ファイル: {ref.rel}]"


# ---- lexical tokenizer: ascii words + Japanese char bigrams ----
_ASCII = re.compile(r"[A-Za-z0-9_]+")
_JP = re.compile(r"[぀-ヿ一-鿿]")


def tokenize(text: str) -> list[str]:
    t = text.lower()
    toks = _ASCII.findall(t)
    jp = _JP.findall(t)
    toks += ["".join(pair) for pair in zip(jp, jp[1:])]  # bigrams
    toks += jp  # unigrams for rare single-char terms
    return toks


def build(caption_images: bool = True, verbose: bool = True) -> None:
    from src.rag import facts  # deterministic fact-row distillation (SOT-2449)
    from src.rag.extract import extract  # heavy deps, only needed for building
    from src.rag.index import evidence_index  # typed inverted evidence store (SOT-2531 / #4a)

    refs = corpus.walk()
    chunks: list[Chunk] = []
    cid = 0
    n_facts = 0
    # Typed inverted evidence store (SOT-2531 / #4a). Populated from the SAME extraction pass
    # below (no second extraction), then written to its own artifact. Guarded end-to-end so a
    # failure here can never corrupt the chunks/embeddings the serve path depends on.
    ev_entries: list[evidence_index.EvidenceEntry] = []
    for i, ref in enumerate(refs):
        doc = extract(ref, caption_images=caption_images)
        if not doc.text.strip():
            continue
        try:
            ev_entries.extend(evidence_index.scan_doc(ref, doc.text))
        except Exception:  # additive; never break the primary index build
            pass
        body_chunks = _split(doc.text)[:MAX_CHUNKS_PER_DOC]
        header = _header(ref)
        for bc in body_chunks:
            chunks.append(Chunk(cid, ref.project, ref.category, ref.name, ref.rel, doc.kind,
                                f"{header}\n{bc}"))
            cid += 1
        # Fact-level index (SOT-2449): high-signal extracted artifacts (highlights / bold /
        # Pivot conditions / code params) as independent 1-fact-per-line embedded rows. The raw
        # chunks above stay in place (BM25 + dense); fact rows are additive. Opt-in — a
        # structural no-op unless RAG_FACT_INDEX=1.
        if facts.enabled():
            for fr in facts.fact_rows(ref, doc.text):
                chunks.append(Chunk(cid, ref.project, ref.category, ref.name, ref.rel, "fact", fr))
                cid += 1
                n_facts += 1
        if verbose and (i + 1) % 50 == 0:
            print(f"  extracted {i + 1}/{len(refs)} files, {len(chunks)} chunks ({n_facts} facts)")
    if verbose:
        print(f"  fact rows indexed: {n_facts}")

    # Persist the typed evidence store next to (but independent of) the retrieval index.
    try:
        ev_counts = evidence_index.write_index(ev_entries)
        if verbose:
            print(f"  evidence index: {ev_counts['entries']} entries "
                  f"from {ev_counts['files']} files -> {evidence_index.default_out_path()}")
    except Exception as e:  # additive; never break the primary index build
        if verbose:
            print(f"  evidence index skipped: {type(e).__name__}: {e}")

    # Deterministic structure pre-store (SOT-2533 / #4b sibling): precompute the runtime-expensive
    # structure extractions (highlights / chart numCache / pivot conditions / seating directory /
    # version diffs) once, so the runtime tools read a persisted answer instead of re-parsing per
    # question. Additive and fully guarded — a failure here can never corrupt the retrieval index,
    # and runtime consultation is opt-in (RAG_STRUCTURE_STORE) so the champion serve path is
    # byte-identical by default.
    try:
        from src.rag.index import structure_store
        st_counts = structure_store.build(refs)
        if verbose:
            print(f"  structure store: {st_counts['files']} files "
                  f"(highlights={st_counts['highlights']}, charts={st_counts['charts']}, "
                  f"pivots={st_counts['pivots']}, seating={st_counts['seating']}, "
                  f"version_pairs={st_counts['version_pairs']}) -> {structure_store.default_out_path()}")
    except Exception as e:  # additive; never break the primary index build
        if verbose:
            print(f"  structure store skipped: {type(e).__name__}: {e}")

    # Canonical-route manifest (SOT-2530 / 事前処理 #3): precompute project→kind→ranked-rel once so
    # canonical_route.discover is an O(1) lookup at run time instead of re-walking the corpus and
    # running _matches_kind per call. Additive and fully guarded — a failure here can never corrupt
    # the retrieval index, and runtime consultation is opt-in (RAG_CANONICAL_MANIFEST) so the champion
    # serve path is byte-identical by default.
    try:
        from src.rag.index import canonical_manifest
        cm_counts = canonical_manifest.build(refs)
        if verbose:
            print(f"  canonical manifest: {cm_counts['projects']} projects "
                  f"({cm_counts['kind_entries']} kind entries, {cm_counts['files']} files) "
                  f"-> {canonical_manifest.default_out_path()}")
    except Exception as e:  # additive; never break the primary index build
        if verbose:
            print(f"  canonical manifest skipped: {type(e).__name__}: {e}")

    # Document registry (SOT-2583 / 事前処理 #1): one deterministic record per corpus file
    # (doc_id / case_id / normalized filename + aliases / version family / entities / structure), so
    # the serve path can resolve *which document* a question refers to — and hard-abstain on an
    # explicitly-named file it cannot find (idx12) — before the answer loop starts, instead of burning
    # the turn budget re-discovering the corpus. Additive and fully guarded — a failure here can never
    # corrupt the retrieval index, and runtime consultation is opt-in (RAG_DOCUMENT_REGISTRY) so the
    # champion serve path is byte-identical by default.
    try:
        from src.rag.index import document_registry
        dr_counts = document_registry.build(refs)
        if verbose:
            print(f"  document registry: {dr_counts['files']} files "
                  f"(versioned={dr_counts['versioned']}, with_predecessor={dr_counts['with_predecessor']}, "
                  f"with_sheets={dr_counts['with_sheets']}, warnings={dr_counts['warnings']}) "
                  f"-> {document_registry.default_out_path()}")
    except Exception as e:  # additive; never break the primary index build
        if verbose:
            print(f"  document registry skipped: {type(e).__name__}: {e}")

    # Pre-index decrypt cache (SOT-2529 / 事前処理 #2): resolve every encrypted Office file's
    # password ONCE here and persist it into corpus_profile.json, so the run-wide shared profile
    # (SOT-2528) is pre-warmed and the first runtime decrypt/extract_office hits the cache instead
    # of brute-forcing sibling docs × dates × formats per question. Additive and fully guarded — a
    # failure here can never corrupt the retrieval index; secrets only reach the gitignored runtime
    # profile; runtime consultation stays opt-in (RAG_SHARE_CORPUS_PROFILE) so the champion serve
    # path is byte-identical by default.
    try:
        from src.rag.index import precomputed_passwords
        pw_counts = precomputed_passwords.build(refs)
        if verbose:
            print(f"  pre-index decrypt: {pw_counts['encrypted']} encrypted "
                  f"(resolved={pw_counts['resolved']}, cached={pw_counts['cached']}, "
                  f"unresolved={pw_counts['unresolved']}) -> {pw_counts['path']}")
    except Exception as e:  # additive; never break the primary index build
        if verbose:
            print(f"  pre-index decrypt skipped: {type(e).__name__}: {e}")

    # OCR persistence store (SOT-2650 / 事前計算事実層 追補): transcribe every effectively-image-only
    # PDF once at build time (Gemini vision — 前処理限定の使用) and persist the text, so the serve path
    # can read scanned 会議録/報告書/最終報告 under RAG_OCR_STORE with no genai call (Sonnet dev runs set
    # RAG_FORBID_GEMINI=1 and therefore silently lose live OCR — the store is their only route to these
    # documents). Unlike the LLM-free sibling stores this step SPENDS vision calls, so it additionally
    # gates the build itself behind RAG_OCR_STORE_BUILD (default OFF ⇒ skipped, prior artifact kept).
    try:
        from src.rag.index import ocr_store
        if ocr_store.build_enabled():
            ocs = ocr_store.build(refs)
            if verbose:
                rep = ocs["report"]
                print(f"  ocr store: {ocs['records']} scanned PDFs "
                      f"(transcribed={rep['transcribed']}, reused={rep['reused']}, "
                      f"pages={rep['pages']}) -> {ocr_store.default_out_path()}")
        elif verbose:
            print("  ocr store: build skipped (RAG_OCR_STORE_BUILD off; existing artifact kept)")
    except Exception as e:  # additive; never break the primary index build
        if verbose:
            print(f"  ocr store skipped: {type(e).__name__}: {e}")

    # Case master table (SOT-2643 / 事前計算事実層 1/5): precompute one deterministic row per 案件
    # (全案件 × 標準属性 with per-cell provenance), so cross-corpus enumeration / comparison questions
    # (enum hard core idx38/67/87 …) reduce to a single filter over a self-evidently complete universe
    # (row count == 案件数) instead of runtime evidence-search that exhausts the turn budget. Additive
    # and fully guarded — a failure here can never corrupt the retrieval index, and runtime consultation
    # is opt-in (RAG_CASE_MASTER) so the champion serve path is byte-identical by default.
    try:
        from src.rag.index import case_master
        cmst = case_master.build(refs)
        if verbose:
            cert = cmst["report"]["certificate"]
            print(f"  case master: {cmst['cases']} cases "
                  f"(universe_ok={cert['row_count_equals_universe']}) "
                  f"-> {case_master.default_out_path()}")
    except Exception as e:  # additive; never break the primary index build
        if verbose:
            print(f"  case master skipped: {type(e).__name__}: {e}")

    # ID master reverse-lookup (SOT-2644 / 事前計算事実層 2/5): precompute every ID-code occurrence
    # (タスクID/アクションID/マイルストーン/チェックポイント/WBS/AI番号 …) with its full 原文行 + 隣接列 +
    # locator, so an "ID→内容そのまま抜き出す" question resolves to an O(1) reverse-lookup instead of the
    # file_grep repetition that exhausts the turn budget (idx98: 11連発). Additive and fully guarded — a
    # failure here can never corrupt the retrieval index, and runtime consultation is opt-in
    # (RAG_ID_MASTER) so the champion serve path is byte-identical by default.
    try:
        from src.rag.index import id_master
        idm = id_master.build(refs)
        if verbose:
            print(f"  id master: {idm['occurrences']} occurrences "
                  f"({idm['distinct_ids']} distinct ids over {idm['files']} files) "
                  f"-> {id_master.default_out_path()}")
    except Exception as e:  # additive; never break the primary index build
        if verbose:
            print(f"  id master skipped: {type(e).__name__}: {e}")

    # Derived-metrics store (SOT-2645 / 事前計算事実層 3/5): precompute standard derived quantities for
    # every 案件 × 全数値列 (column stats / percentiles / correlations / histograms / ratios / OLS
    # predictions / F1 threshold sweep) with independent dual verification (fail-closed), so
    # derived_calculation questions (idx50/57/63 …) reduce to an O(1) lookup instead of the run-time
    # compute guess-and-check that exhausts the turn budget (idx57: 8連発). Question-independent &
    # LLM-free. Additive and fully guarded — a failure here can never corrupt the retrieval index, and
    # runtime consultation is opt-in (RAG_DERIVED_METRICS) so the champion serve path is byte-identical.
    try:
        from src.rag.index import derived_metrics
        dm = derived_metrics.build(refs)
        if verbose:
            cov = dm["report"]["coverage"]
            vs = dm["report"]["verification_summary"]
            print(f"  derived metrics: {dm['cases']} cases "
                  f"(model={cov['cases_with_model']}, sweep={cov['cases_with_threshold_sweep']}, "
                  f"mismatches_dropped={vs['mismatches_dropped']}/{vs['metrics_computed']}) "
                  f"-> {derived_metrics.default_out_path()}")
    except Exception as e:  # additive; never break the primary index build
        if verbose:
            print(f"  derived metrics skipped: {type(e).__name__}: {e}")

    # Version-pair diff store (SOT-2646 / 事前計算事実層 4/5): precompute every version-family pair's
    # structural diff + SUBSTANTIVE-ranked candidates (+ per-change exclusion attributes) once, so a
    # version_diff question reads a pre-confirmed result instead of re-running the runtime diff→LLM
    # formatting that flaked run-to-run (cycle3 version 系 MATCH↔WRONG). Pair enumeration unions the
    # filename rules with the document-registry families so the SOT-2588 gap (_old×unversioned latest,
    # idx0/idx1) is filled. Question-independent & LLM-free (diff logic itself unchanged — it only calls
    # diffpair.rank_changes). Additive and fully guarded — a failure here can never corrupt the retrieval
    # index, and runtime consultation is opt-in (RAG_DIFF_STORE) so the champion serve path is byte-identical.
    try:
        from src.rag.index import diff_store
        ds = diff_store.build(refs)
        if verbose:
            cov = ds["report"]["version_hard_core_coverage"]
            covered = sum(1 for c in cov.values() if c["enumerated"] and c["alignment_ok"])
            print(f"  diff store: {ds['pairs']} version pairs "
                  f"({ds['changes']} ranked changes, alignment_failures={ds['alignment_failures']}, "
                  f"hard_core_covered={covered}/{len(cov)}) -> {diff_store.default_out_path()}")
    except Exception as e:  # additive; never break the primary index build
        if verbose:
            print(f"  diff store skipped: {type(e).__name__}: {e}")

    # Report attribute store (SOT-2655 / 事前計算事実層 追補 — cycle4 クラスタD): precompute per-案件 numeric
    # attributes from 最終報告/調査系 文書 + 分析成果物 — 最良モデルのハイパーパラメータ表 (metrics.json
    # model_params), 将来フェーズの想定工数 (報告本文の フェーズX/工数 行), 高影響特徴量とターゲット相関 —
    # so scanned-PDF numeric abstains (idx5 max_depth / idx28 相関最大特徴量 / idx64 フェーズ工数合計,
    # BUDGET_EXHAUSTED) reduce to an O(1) lookup instead of file_grep that never reaches a scan's body.
    # Question-independent & LLM-free (reads persisted OCR + text-layer + analysis JSON + train tables).
    # Additive and fully guarded; runtime consultation is opt-in (RAG_REPORT_ATTR_STORE) so the champion
    # serve path is byte-identical.
    try:
        from src.rag.index import report_attr_store
        ras = report_attr_store.build(refs)
        if verbose:
            rep = ras["report"]
            print(f"  report attr store: {ras['cases']} cases "
                  f"(model={rep['with_model']}, phases={rep['with_phases']}, "
                  f"features={rep['with_features']}) -> {report_attr_store.default_out_path()}")
    except Exception as e:  # additive; never break the primary index build
        if verbose:
            print(f"  report attr store skipped: {type(e).__name__}: {e}")

    # Case finance/effort store (SOT-2654): all-case operands and cross-case derived tables. Runtime
    # consumption is separately opt-in, preserving the default answer path byte-for-byte.
    try:
        from src.rag.index import case_finance_store
        cfs = case_finance_store.build(refs)
        if verbose:
            cert = cfs["report"]["certificate"]
            print(f"  case finance store: {cfs['cases']} cases "
                  f"(universe_ok={cert['row_count_equals_universe']}) "
                  f"-> {case_finance_store.default_out_path()}")
    except Exception as e:  # additive; never break the primary index build
        if verbose:
            print(f"  case finance store skipped: {type(e).__name__}: {e}")

    if verbose:
        print(f"embedding {len(chunks)} chunks...")
    texts = [c.text for c in chunks]
    vecs = llm.embed(texts, task_type="RETRIEVAL_DOCUMENT")
    emb = np.asarray(vecs, dtype=np.float32)
    emb /= (np.linalg.norm(emb, axis=1, keepdims=True) + 1e-8)

    settings.INDEX_DIR.mkdir(parents=True, exist_ok=True)
    with open(settings.INDEX_DIR / "chunks.jsonl", "w", encoding="utf-8") as f:
        for c in chunks:
            f.write(json.dumps(asdict(c), ensure_ascii=False) + "\n")
    np.save(settings.INDEX_DIR / "embeddings.npy", emb)
    if verbose:
        print(f"index built: {len(chunks)} chunks, emb {emb.shape} -> {settings.INDEX_DIR}")


def load_chunks() -> list[dict]:
    path = settings.INDEX_DIR / "chunks.jsonl"
    return [json.loads(l) for l in open(path, encoding="utf-8")]


def load_embeddings() -> np.ndarray:
    return np.load(settings.INDEX_DIR / "embeddings.npy")
