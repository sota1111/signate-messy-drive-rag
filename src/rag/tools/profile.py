"""Corpus adaptation layer — a runtime cache of self-discovered corpus facts.

The tools discover固有 facts about *this specific* messy drive at run time: which password
actually decrypted an Office file, which alias a project turned out to be known by, which colour
a highlight resolved to, the real data-sheet name inside a chart-fronted xlsx. Re-deriving those
every call is slow (password brute force scans sibling docs) and, worse, non-idempotent across a
long run. :class:`CorpusProfile` persists them to ``corpus_profile.json`` so a later tool call can
*reuse* the discovery instead of repeating it.

Security invariant (受け入れ条件②)
--------------------------------
Raw secrets — decrypted passwords, the literal answer strings behind 略称 — are **never bundled in
code**. They live only in this *runtime* JSON, which is written under a gitignored artifacts dir and
is explicitly gitignored by name, so it is never committed. The schema (this module + the committed
``docs/corpus_profile.schema.json``) is secret-free; only a live run populates the values.

Schema (``corpus_profile.json``)::

    {
      "version": 1,
      "passwords": { "<corpus-relative file>": "<decrypted password>" },
      "aliases":   { "<canonical company>": ["<alias>", ...] },
      "formats":   { "<key>": <json value> }   # e.g. xlsx data-sheet name, resolved highlight colour
    }
"""
from __future__ import annotations

import json
import os
import tempfile
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from config import settings

SCHEMA_VERSION = 1

# Runtime cache lives under the gitignored artifacts dir; never committed (holds raw secrets).
DEFAULT_PROFILE_PATH = settings.ARTIFACTS_DIR / "corpus_profile.json"

_ON = {"1", "true", "yes", "on"}


def share_enabled() -> bool:
    """True when a run should share ONE persisted profile across all its questions (default OFF).

    The discovered facts held by a :class:`CorpusProfile` (which password decrypted a file, which
    alias a project is known by, the real data-sheet name inside a chart-fronted xlsx) are *corpus*
    facts, not per-question answers, so reusing them from question to question only removes rediscovery
    cost — it never changes an answer (受け入れ条件③/SOT-2528). Re-deriving them every question is the
    dominant 反復上限 (12ターン/180秒) cost: 探索の 73% は発見系で、暗号化ブルートフォースや略称解決を
    同じファイルでも毎問やり直している。

    Gated behind ``RAG_SHARE_CORPUS_PROFILE`` and default OFF so the champion serve path stays
    byte-identical unless the flag is set — mirroring the ``RAG_EVIDENCE_INDEX`` /
    ``RAG_STRUCTURE_STORE`` / ``RAG_CANONICAL_MANIFEST`` conventions. The end-to-end effect (平均
    elapsed_s と BUDGET_EXHAUSTED 減少, 回答不変) is measured with the flag ON in the SOT-2527 gold100.
    """
    return os.getenv("RAG_SHARE_CORPUS_PROFILE", "0").strip().lower() in _ON


@dataclass
class CorpusProfile:
    """In-memory view of ``corpus_profile.json`` with read/write/roundtrip helpers."""

    passwords: dict[str, str] = field(default_factory=dict)
    aliases: dict[str, list[str]] = field(default_factory=dict)
    formats: dict[str, Any] = field(default_factory=dict)
    version: int = SCHEMA_VERSION
    path: Path = DEFAULT_PROFILE_PATH
    # Guards read-modify-write mutations so a run-wide shared instance stays consistent when written
    # by concurrent worker threads (``src/rag/run.py`` answers questions in a ThreadPoolExecutor).
    # Excluded from equality/repr so it never affects (de)serialization or test comparisons.
    _lock: threading.RLock = field(
        default_factory=threading.RLock, compare=False, repr=False)

    # ------------------------------------------------------------------ (de)serialize
    def to_dict(self) -> dict[str, Any]:
        with self._lock:
            return {
                "version": self.version,
                "passwords": dict(self.passwords),
                "aliases": {k: list(v) for k, v in self.aliases.items()},
                "formats": dict(self.formats),
            }

    @classmethod
    def from_dict(cls, obj: dict[str, Any], *, path: Path | None = None) -> "CorpusProfile":
        return cls(
            passwords=dict(obj.get("passwords") or {}),
            aliases={k: list(v) for k, v in (obj.get("aliases") or {}).items()},
            formats=dict(obj.get("formats") or {}),
            version=int(obj.get("version") or SCHEMA_VERSION),
            path=Path(path) if path is not None else DEFAULT_PROFILE_PATH,
        )

    @classmethod
    def load(cls, path: str | Path | None = None) -> "CorpusProfile":
        """Load the profile, returning an empty one when the file is missing/corrupt (fail-open)."""
        p = Path(path) if path is not None else DEFAULT_PROFILE_PATH
        try:
            obj = json.loads(p.read_text(encoding="utf-8"))
        except (FileNotFoundError, json.JSONDecodeError, OSError):
            return cls(path=p)
        prof = cls.from_dict(obj, path=p)
        return prof

    def save(self, path: str | Path | None = None) -> Path:
        """Atomically write the profile to disk and return the path written."""
        p = Path(path) if path is not None else self.path
        p.parent.mkdir(parents=True, exist_ok=True)
        text = json.dumps(self.to_dict(), ensure_ascii=False, indent=2, sort_keys=True)
        # atomic replace so a concurrent reader never sees a half-written file
        fd, tmp = tempfile.mkstemp(dir=str(p.parent), prefix=".corpus_profile.", suffix=".tmp")
        try:
            with open(fd, "w", encoding="utf-8") as fh:
                fh.write(text)
            Path(tmp).replace(p)
        except BaseException:
            Path(tmp).unlink(missing_ok=True)
            raise
        self.path = p
        return p

    # ------------------------------------------------------------------ passwords
    def get_password(self, file_rel: str) -> str | None:
        return self.passwords.get(file_rel)

    def set_password(self, file_rel: str, password: str) -> None:
        with self._lock:
            self.passwords[file_rel] = password

    # ------------------------------------------------------------------ aliases
    def get_aliases(self, company: str) -> list[str]:
        return list(self.aliases.get(company, []))

    def add_alias(self, company: str, alias: str) -> None:
        with self._lock:
            bucket = self.aliases.setdefault(company, [])
            if alias and alias not in bucket:
                bucket.append(alias)

    # ------------------------------------------------------------------ formats
    def get_format(self, key: str, default: Any = None) -> Any:
        return self.formats.get(key, default)

    def set_format(self, key: str, value: Any) -> None:
        with self._lock:
            self.formats[key] = value

    # ------------------------------------------------------------------ merge
    def merge(self, other: "CorpusProfile") -> None:
        """Fold another profile's discoveries into this one (existing keys win; additive only).

        Used to seed a run-wide shared profile from a per-question one, or to combine a freshly loaded
        on-disk profile with in-memory discoveries without dropping either side.
        """
        with self._lock:
            for k, v in other.passwords.items():
                self.passwords.setdefault(k, v)
            for company, aliases in other.aliases.items():
                bucket = self.aliases.setdefault(company, [])
                for a in aliases:
                    if a and a not in bucket:
                        bucket.append(a)
            for k, v in other.formats.items():
                self.formats.setdefault(k, v)


def load(path: str | Path | None = None) -> CorpusProfile:
    """Module-level convenience mirroring :meth:`CorpusProfile.load`."""
    return CorpusProfile.load(path)
