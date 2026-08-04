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
    from src.rag.extract import extract  # heavy deps, only needed for building

    refs = corpus.walk()
    chunks: list[Chunk] = []
    cid = 0
    for i, ref in enumerate(refs):
        doc = extract(ref, caption_images=caption_images)
        if not doc.text.strip():
            continue
        body_chunks = _split(doc.text)[:MAX_CHUNKS_PER_DOC]
        header = _header(ref)
        for bc in body_chunks:
            chunks.append(Chunk(cid, ref.project, ref.category, ref.name, ref.rel, doc.kind,
                                f"{header}\n{bc}"))
            cid += 1
        if verbose and (i + 1) % 50 == 0:
            print(f"  extracted {i + 1}/{len(refs)} files, {len(chunks)} chunks")

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


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--no-images", action="store_true", help="skip Gemini PNG captioning")
    args = ap.parse_args()
    build(caption_images=not args.no_images)
