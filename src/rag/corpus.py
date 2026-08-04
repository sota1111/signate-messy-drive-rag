"""Corpus walking + file classification (NFC-aware).

The corpus is a messy shared drive whose Japanese directory names are NFD-normalized
(macOS origin). Everything here normalizes to NFC so downstream matching works.
"""
from __future__ import annotations

import functools
import unicodedata
from dataclasses import dataclass
from pathlib import Path

from config import settings


def nfc(s: str) -> str:
    return unicodedata.normalize("NFC", s)


# Fixed per-project folder taxonomy → a coarse category tag used for retrieval boosts.
CATEGORY_BY_PREFIX = {
    "00": "proposal",   # 提案
    "01": "contract",   # 契約
    "02": "plan",       # 計画/スケジュール
    "03": "data",       # データ
    "04": "analysis",   # 分析 (ML repo)
    "05": "meeting",    # 会議
    "06": "report",     # 報告書
}

# Office temp/lock files to always skip.
def _is_junk(name: str) -> bool:
    n = nfc(name)
    return n.startswith("~$") or n == ".DS_Store" or n.endswith(".pyc")


@dataclass
class FileRef:
    path: Path
    project: str            # NFC company name, or "社内管理" / ""
    category: str           # proposal/contract/.../analysis/meeting/report/internal/""
    rel: str                # NFC path relative to corpus root
    name: str               # NFC filename
    ext: str                # lowercase extension without dot

    @property
    def stem(self) -> str:
        return self.name.rsplit(".", 1)[0] if "." in self.name else self.name


def _category_of(parts: list[str]) -> str:
    for p in parts:
        pre = p[:2]
        if pre in CATEGORY_BY_PREFIX:
            return CATEGORY_BY_PREFIX[pre]
    if any(nfc(p) == "社内管理" for p in parts):
        return "internal"
    return ""


@functools.lru_cache(maxsize=4)
def walk(corpus_dir: Path | None = None) -> list[FileRef]:
    """Return every content file in the corpus as a classified FileRef (memoized; static corpus)."""
    root = corpus_dir or settings.CORPUS_DIR
    if not root.exists():
        return []
    proj_dir_name = {nfc(d.name): d for d in root.iterdir() if d.is_dir()}
    refs: list[FileRef] = []
    for f in root.rglob("*"):
        if not f.is_file() or _is_junk(f.name):
            continue
        rel_parts = [nfc(p) for p in f.relative_to(root).parts]
        # project = second path component when under プロジェクト/, else the top folder
        project = ""
        if rel_parts and rel_parts[0] == "プロジェクト" and len(rel_parts) >= 2:
            project = rel_parts[1]
        elif rel_parts and rel_parts[0] == "社内管理":
            project = "社内管理"
        ext = f.suffix.lower().lstrip(".")
        refs.append(
            FileRef(
                path=f,
                project=project,
                category=_category_of(rel_parts),
                rel="/".join(rel_parts),
                name=nfc(f.name),
                ext=ext,
            )
        )
    return refs
