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
