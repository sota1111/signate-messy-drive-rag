#!/usr/bin/env python3
"""SOT-2624 — flag-manifest reconciliation for integration-measurement scripts.

Adversarial-review hole **H3 (構成漏れ)** from the cycle-2 gold100 post-mortem
(``docs/ai/cycle2_adversarial_review.md``): a serve-path flag
(``RAG_DERIVED_FORMAT_CONTRACTS``, default OFF, ``src/rag/agent/formatting.py``) that the
consolidated measurement *assumed* was active was never ``export``ed by
``scripts/sot_cycle2_gold100.sh``, so the whole gold100 ran with it silently OFF — and the
run was a one-shot promotion gate, so 3 rescuable near-misses stayed WRONG with no re-run.
Nothing checked, **before** the run, that the flags the code reads and the flags the script
sets actually agree.

This is that pre-run machine check. It statically scans the implementation for every
``RAG_*`` flag the code reads (with its default), extracts the ``export RAG_*=...`` set from a
given integration script, and reconciles the two:

* **(a) impl-but-not-exported** — a flag the code reads that the script never sets, so it runs
  on its *default*. Listed with the default and definition site. This is the H3 class. It is a
  **warning** by default (many defaults are intentionally left implicit) and an **error** under
  ``--strict``.
* **(b) exported-but-not-in-impl** — a ``RAG_*`` the script sets that no code reads: a typo or a
  retired flag. Always an **error** (exit 2), because it silently does nothing.
* **(c) effective config** — the full table of every implementation flag with its resolved value
  (exported value, or ``default``).

Scope is deliberately ``RAG_*``-only: the measurement scripts also export non-flag env
(``GEN_MODEL``/``VERTEX_LOCATION``/``PYTHONPATH``…) that this checker must not flag.

Serve path is untouched — this only *reads* source text and a shell script.

Usage::

    python scripts/check_flag_manifest.py scripts/sot_cycle2_gold100.sh          # (b) fails
    python scripts/check_flag_manifest.py scripts/sot_cycle2_gold100.sh --strict # (a) also fails
    python scripts/check_flag_manifest.py <script.sh> --json                     # machine-readable

Integration-script template (put at the TOP of a new ``sot_*_gold100.sh``, before the run)::

    # --- flag-manifest preflight (SOT-2624): abort if this script sets a RAG_* no code reads ---
    .venv/bin/python scripts/check_flag_manifest.py "$0" || exit 1
    # (add --strict to also abort when a code-read flag is left on its default)
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path

# Only flags with this prefix are reconciled (the measurement scripts also export unrelated env).
FLAG_PREFIX = "RAG_"

_REPO_ROOT = Path(__file__).resolve().parent.parent

# A flag name: the prefix + upper snake.
_NAME = FLAG_PREFIX + r"[A-Z0-9_]*"

# A *reader*: any call (or registry tuple) whose first argument is a ``RAG_*`` string literal, with an
# optional 2nd-arg default. The optional dotted callee makes this cover every wrapper uniformly —
# ``_env_flag(...)`` / ``_bool_env(...)`` / ``os.getenv(...)`` / ``os.environ.get(...)`` — and, with no
# callee, the ``("RAG_X", True)`` registry tuples (``det_pipeline._WAVE_FLAGS``). This is the set that
# carries a default, so it drives (a) and the effective-config table.
_READER_RE = re.compile(
    r"""(?:[A-Za-z_][\w.]*)?\(\s*["'](?P<name>""" + _NAME + r""")["']"""
    r"""(?:\s*,\s*(?P<default>"[^"]*"|'[^']*'|[^),]+))?\s*\)"""
)

# ``os.environ["RAG_X"]`` — required (no default; a KeyError if unset).
_ENVIRON_ITEM_RE = re.compile(r"""os\.environ\[\s*["'](?P<name>""" + _NAME + r""")["']\s*\]""")

# Every ``RAG_*`` string literal, regardless of context (calls, dict values, comments-as-strings). This
# is the *superset* used to decide (b): an export is "unknown" only if its name appears nowhere in src,
# so a real flag read via any pattern this checker does not model can never be a false (b) error.
_LITERAL_RE = re.compile(r"""["'](?P<name>""" + _NAME + r""")["']""")

# Best-effort dynamic-name detector: a flag read whose first arg is NOT a plain string literal
# (variable, f-string, concatenation). We only warn when the call text mentions the prefix to cut noise.
_DYNAMIC_RE = re.compile(
    r"""(?:os\.getenv|os\.environ\.get|_env_flag|_bool_env)\(\s*(?P<arg>[^)\n]*)"""
)


@dataclass
class FlagDef:
    """One implementation flag: its resolved default plus every place the code reads it."""

    name: str
    default: str
    locations: list[str] = field(default_factory=list)
    defaults: set[str] = field(default_factory=set)

    @property
    def default_conflict(self) -> bool:
        return len(self.defaults) > 1


def _iter_py_files(src_root: Path):
    for p in sorted(src_root.rglob("*.py")):
        if "__pycache__" in p.parts:
            continue
        yield p


def _line_of(text: str, offset: int) -> int:
    return text.count("\n", 0, offset) + 1


def _norm_default(raw: str | None) -> str:
    """Normalize a captured default expression to a readable value string."""
    if raw is None:
        return "None (unset)"
    raw = raw.strip()
    if (len(raw) >= 2) and raw[0] in "\"'" and raw[-1] == raw[0]:
        return raw[1:-1]
    return raw


def extract_impl_flags(src_root: Path | None = None) -> dict[str, FlagDef]:
    """Scan ``src_root`` for every ``FLAG_PREFIX`` flag the code *reads* (with a default), + locations.

    These are the reader call-sites, so each entry has a resolved default; this set drives the (a)
    impl-but-not-exported list and the effective-config table. See :func:`extract_known_flag_names`
    for the broader typo-detection superset.
    """
    src_root = Path(src_root) if src_root is not None else (_REPO_ROOT / "src")
    flags: dict[str, FlagDef] = {}

    def _record(name: str, default: str, rel: str, line: int) -> None:
        if not name.startswith(FLAG_PREFIX):
            return
        fd = flags.get(name)
        if fd is None:
            fd = flags[name] = FlagDef(name=name, default=default)
        fd.locations.append(f"{rel}:{line}")
        fd.defaults.add(default)

    for path in _iter_py_files(src_root):
        text = path.read_text(encoding="utf-8", errors="replace")
        try:
            rel = str(path.relative_to(_REPO_ROOT))
        except ValueError:
            rel = str(path)
        for m in _READER_RE.finditer(text):
            _record(m["name"], _norm_default(m["default"]), rel, _line_of(text, m.start()))
        for m in _ENVIRON_ITEM_RE.finditer(text):
            _record(m["name"], "None (required)", rel, _line_of(text, m.start()))
    return flags


def extract_known_flag_names(src_root: Path | None = None) -> set[str]:
    """Every ``FLAG_PREFIX`` name appearing as a string literal anywhere in ``src_root``.

    Superset of :func:`extract_impl_flags` — includes flags referenced only in registries or read via
    a dynamically-built name. Used solely to keep (b) (exported-but-unknown) conservative: a name that
    appears somewhere in the source is a real flag, never a typo error.
    """
    src_root = Path(src_root) if src_root is not None else (_REPO_ROOT / "src")
    names: set[str] = set()
    for path in _iter_py_files(src_root):
        text = path.read_text(encoding="utf-8", errors="replace")
        for m in _LITERAL_RE.finditer(text):
            names.add(m["name"])
    return names


def extract_dynamic_reads(src_root: Path | None = None) -> list[str]:
    """Best-effort: locate flag reads with a non-literal first arg that mentions RAG (warn-only)."""
    src_root = Path(src_root) if src_root is not None else (_REPO_ROOT / "src")
    out: list[str] = []
    for path in _iter_py_files(src_root):
        text = path.read_text(encoding="utf-8", errors="replace")
        try:
            rel = str(path.relative_to(_REPO_ROOT))
        except ValueError:
            rel = str(path)
        for m in _DYNAMIC_RE.finditer(text):
            arg = m["arg"].lstrip()
            if not arg or arg[0] in "\"'":
                continue  # plain string literal — handled by the concrete extractors
            if "RAG" in arg.upper():
                out.append(f"{rel}:{_line_of(text, m.start())}  {arg.strip()[:60]}")
    return out


def extract_script_exports(script_path: Path) -> dict[str, str]:
    """Extract ``export RAG_*=VALUE`` assignments from a shell script (prefix-scoped)."""
    exports: dict[str, str] = {}
    text = Path(script_path).read_text(encoding="utf-8", errors="replace")
    assign_re = re.compile(r"(?P<name>" + _NAME + r")=(?P<val>\"[^\"]*\"|'[^']*'|\S+)")
    for line in text.splitlines():
        stripped = line.lstrip()
        if not stripped.startswith("export "):
            continue
        body = stripped[len("export "):]
        body = _strip_unquoted_comment(body)
        for m in assign_re.finditer(body):
            name = m["name"]
            if name.startswith(FLAG_PREFIX):
                exports[name] = _norm_default(m["val"])
    return exports


def _strip_unquoted_comment(s: str) -> str:
    """Drop a trailing ``# ...`` comment that is not inside quotes."""
    in_single = in_double = False
    for i, ch in enumerate(s):
        if ch == "'" and not in_double:
            in_single = not in_single
        elif ch == '"' and not in_single:
            in_double = not in_double
        elif ch == "#" and not in_single and not in_double:
            return s[:i]
    return s


@dataclass
class Report:
    impl: dict[str, FlagDef]
    exports: dict[str, str]
    dynamic: list[str]
    known: set[str] = field(default_factory=set)  # every RAG_* literal in src (typo-check superset)
    not_exported: list[str] = field(default_factory=list)  # (a) impl but not exported
    unknown_exports: list[str] = field(default_factory=list)  # (b) exported but no code reads
    conflicts: list[str] = field(default_factory=list)  # multiple distinct defaults for one flag

    def effective_config(self) -> list[tuple[str, str, str]]:
        """(name, value, source) for every reader flag AND every exported flag (union)."""
        rows: list[tuple[str, str, str]] = []
        for name in sorted(set(self.impl) | set(self.exports)):
            if name in self.exports:
                rows.append((name, self.exports[name], "export"))
            else:
                rows.append((name, self.impl[name].default, "default"))
        return rows


def reconcile(
    impl: dict[str, FlagDef],
    exports: dict[str, str],
    dynamic: list[str] | None = None,
    known: set[str] | None = None,
) -> Report:
    # A flag known to the source (readers ∪ any literal) is never a typo; fall back to impl keys when
    # the literal superset is not supplied so a bare reconcile() call still behaves sensibly.
    known = set(known) if known is not None else set(impl)
    rep = Report(impl=impl, exports=exports, dynamic=list(dynamic or []), known=known)
    rep.not_exported = sorted(n for n in impl if n not in exports)
    rep.unknown_exports = sorted(n for n in exports if n not in known)
    rep.conflicts = sorted(n for n, fd in impl.items() if fd.default_conflict)
    return rep


def _render(rep: Report, script_path: str, strict: bool) -> str:
    lines: list[str] = []
    lines.append(f"# Flag-manifest reconciliation — {script_path}")
    lines.append(f"impl RAG_* flags: {len(rep.impl)}   script RAG_* exports: {len(rep.exports)}")
    lines.append("")

    lines.append(f"## (b) EXPORTED BUT UNKNOWN TO SOURCE — ERROR ({len(rep.unknown_exports)})")
    if rep.unknown_exports:
        for n in rep.unknown_exports:
            lines.append(f"  ✗ {n}={rep.exports[n]}   (typo or retired flag — silently does nothing)")
    else:
        lines.append("  (none)")
    lines.append("")

    tag = "ERROR" if strict else "WARN"
    lines.append(f"## (a) CODE READS IT BUT SCRIPT DOES NOT EXPORT — runs on default [{tag}] ({len(rep.not_exported)})")
    if rep.not_exported:
        for n in rep.not_exported:
            fd = rep.impl[n]
            lines.append(f"  • {n}  default={fd.default}  ({fd.locations[0]})")
    else:
        lines.append("  (none)")
    lines.append("")

    if rep.conflicts:
        lines.append(f"## default conflicts (same flag, differing defaults) ({len(rep.conflicts)})")
        for n in rep.conflicts:
            fd = rep.impl[n]
            lines.append(f"  ! {n}: {sorted(fd.defaults)}  @ {', '.join(fd.locations)}")
        lines.append("")

    if rep.dynamic:
        lines.append(f"## dynamic / non-literal flag reads (best-effort, warn-only) ({len(rep.dynamic)})")
        for d in rep.dynamic:
            lines.append(f"  ? {d}")
        lines.append("")

    lines.append("## (c) effective configuration")
    for name, value, source in rep.effective_config():
        mark = " " if source == "export" else "~"
        lines.append(f"  {mark} {name} = {value}  [{source}]")
    return "\n".join(lines)


def _to_json(rep: Report) -> dict:
    return {
        "impl_count": len(rep.impl),
        "export_count": len(rep.exports),
        "unknown_exports": {n: rep.exports[n] for n in rep.unknown_exports},
        "not_exported": {n: rep.impl[n].default for n in rep.not_exported},
        "conflicts": {n: sorted(rep.impl[n].defaults) for n in rep.conflicts},
        "dynamic": rep.dynamic,
        "effective": [
            {"name": n, "value": v, "source": s} for n, v, s in rep.effective_config()
        ],
    }


def build_report(script_path: Path, src_root: Path | None = None) -> Report:
    impl = extract_impl_flags(src_root)
    known = extract_known_flag_names(src_root)
    exports = extract_script_exports(script_path)
    dynamic = extract_dynamic_reads(src_root)
    return reconcile(impl, exports, dynamic, known)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Reconcile implementation RAG_* flags with a script's exports.")
    ap.add_argument("script", help="integration measurement shell script to check")
    ap.add_argument("--src-root", default=None, help="source root to scan (default: <repo>/src)")
    ap.add_argument("--strict", action="store_true", help="also fail on (a) impl-but-not-exported")
    ap.add_argument("--json", action="store_true", help="emit machine-readable JSON instead of text")
    args = ap.parse_args(argv)

    script_path = Path(args.script)
    if not script_path.is_file():
        print(f"error: script not found: {script_path}", file=sys.stderr)
        return 2

    src_root = Path(args.src_root) if args.src_root else None
    rep = build_report(script_path, src_root)

    if args.json:
        print(json.dumps(_to_json(rep), ensure_ascii=False, indent=2))
    else:
        print(_render(rep, str(script_path), args.strict))

    # (b) is always fatal; (a) is fatal only under --strict.
    if rep.unknown_exports:
        return 2
    if args.strict and rep.not_exported:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
