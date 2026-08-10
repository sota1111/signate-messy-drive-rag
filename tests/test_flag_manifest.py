"""SOT-2624 — offline tests for the flag-manifest reconciliation checker.

Network-free: the checker only statically scans source text and a shell script. The historical
cycle-2 accident (``RAG_DERIVED_FORMAT_CONTRACTS`` read by the code but never exported by
``scripts/sot_cycle2_gold100.sh`` ⇒ silently OFF for the one-shot gold100 gate; adversarial-review
hole H3) is pinned here as a regression test: the real script must still surface that flag in the
(a) impl-but-not-exported list.
"""
from __future__ import annotations

import importlib
from pathlib import Path

import pytest

cfm = importlib.import_module("scripts.check_flag_manifest")

_REPO_ROOT = Path(__file__).resolve().parent.parent
_CYCLE2_SCRIPT = _REPO_ROOT / "scripts" / "sot_cycle2_gold100.sh"


# --------------------------------------------------------------------------- implementation extraction


def test_reader_extraction_covers_known_flag_wrappers():
    """The reader scan must find flags read through every wrapper (``_env_flag`` / ``_bool_env`` /
    ``os.getenv``) and the registry tuples — a regression guard against a wrapper going unmodelled."""
    impl = cfm.extract_impl_flags()
    for name in (
        "RAG_DET_PIPELINE_ROUTER",   # _env_flag
        "RAG_FORMAT_EVENTS",         # os.getenv
        "RAG_SPIN_PIVOT",            # _bool_env
        "RAG_SEARCH_CAP",            # _bool_env
        "RAG_DERIVED_FORMAT_CONTRACTS",
        "RAG_DET_PIPELINE_B1",       # ("RAG_...", True) registry tuple
    ):
        assert name in impl, f"{name} not extracted as a reader flag"
    # every recorded flag carries its prefix, a (possibly empty-string) default, and a definition site
    for name, fd in impl.items():
        assert name.startswith(cfm.FLAG_PREFIX)
        assert isinstance(fd.default, str)  # "" is a legitimate default (RAG_OPERAND_PREFILL_MAX)
        assert fd.locations


def test_known_names_is_superset_of_readers():
    impl = cfm.extract_impl_flags()
    known = cfm.extract_known_flag_names()
    assert set(impl).issubset(known)
    assert len(known) >= len(impl)


def test_derived_format_contracts_default_off_from_formatting():
    impl = cfm.extract_impl_flags()
    fd = impl["RAG_DERIVED_FORMAT_CONTRACTS"]
    assert fd.default == "False"
    assert any("formatting.py" in loc for loc in fd.locations)


# --------------------------------------------------------------------------- shell export extraction


def test_script_export_extraction_prefix_scoped():
    exports = cfm.extract_script_exports(_CYCLE2_SCRIPT)
    # RAG_* flags set by the real script are captured...
    assert exports.get("RAG_DET_PIPELINE_ROUTER") == "1"
    assert exports.get("RAG_DET_PIPELINE_B2") == "0"
    assert exports.get("RAG_FORMAT_EVENTS") == "1"
    # ...and non-RAG env (and the flag NOT set) are out of scope
    assert "GEN_MODEL" not in exports
    assert "VERTEX_LOCATION" not in exports
    assert "GATE_EXEC_CORRECT" not in exports
    assert "RAG_DERIVED_FORMAT_CONTRACTS" not in exports


def test_inline_comment_stripped_from_export(tmp_path):
    script = tmp_path / "s.sh"
    script.write_text(
        "#!/usr/bin/env bash\n"
        'export RAG_DET_PIPELINE_B1=1   # Wave B1 default ON\n'
        'export RAG_A=1 RAG_B=0  # two on one line\n',
        encoding="utf-8",
    )
    exports = cfm.extract_script_exports(script)
    assert exports == {"RAG_DET_PIPELINE_B1": "1", "RAG_A": "1", "RAG_B": "0"}


# --------------------------------------------------------------------------- historical-bug regression


def test_cycle2_script_surfaces_derived_format_contracts_gap():
    """THE regression: the real one-shot gold100 script must flag the H3 accident before a re-run."""
    rep = cfm.build_report(_CYCLE2_SCRIPT)
    assert "RAG_DERIVED_FORMAT_CONTRACTS" in rep.not_exported, (
        "the checker must detect that the code-read RAG_DERIVED_FORMAT_CONTRACTS was never exported"
    )
    # It was read but not set, so it ran on its default (OFF) — the exact silent-OFF that lost idx8/37/64.
    assert rep.impl["RAG_DERIVED_FORMAT_CONTRACTS"].default == "False"


def test_cycle2_script_has_no_unknown_exports():
    """Every RAG_* the real script exports is a real flag ⇒ no false (b) errors ⇒ default run passes."""
    rep = cfm.build_report(_CYCLE2_SCRIPT)
    assert rep.unknown_exports == []


def test_main_default_exit_zero_on_real_script(capsys):
    rc = cfm.main([str(_CYCLE2_SCRIPT)])
    assert rc == 0  # (b) empty ⇒ non-strict run passes even though (a) is non-empty
    assert "RAG_DERIVED_FORMAT_CONTRACTS" in capsys.readouterr().out


def test_main_strict_fails_when_flags_run_on_default(capsys):
    rc = cfm.main([str(_CYCLE2_SCRIPT), "--strict"])
    assert rc == 1  # (a) non-empty ⇒ strict fails


# --------------------------------------------------------------------------- typo / obsolete export


def test_unknown_export_is_error(tmp_path, capsys):
    script = tmp_path / "fake.sh"
    script.write_text(
        "#!/usr/bin/env bash\n"
        "export RAG_DET_PIPELINE_ROUTER=1\n"
        "export RAG_TOTALLY_FAKE_FLAG=1\n",
        encoding="utf-8",
    )
    rc = cfm.main([str(script)])
    out = capsys.readouterr().out
    assert rc == 2  # (b) is always fatal
    assert "RAG_TOTALLY_FAKE_FLAG" in out


def test_reconcile_uses_known_superset_not_impl_keys():
    """A flag known to source but not a modelled reader must NOT be reported as an unknown export."""
    impl = {"RAG_READER": cfm.FlagDef(name="RAG_READER", default="0", locations=["x.py:1"], defaults={"0"})}
    known = {"RAG_READER", "RAG_REGISTRY_ONLY"}
    exports = {"RAG_REGISTRY_ONLY": "1", "RAG_TYPO": "1"}
    rep = cfm.reconcile(impl, exports, known=known)
    assert rep.unknown_exports == ["RAG_TYPO"]           # only the truly-unknown one
    assert "RAG_REGISTRY_ONLY" not in rep.unknown_exports  # known literal ⇒ not a typo
    assert rep.not_exported == ["RAG_READER"]            # reader, not exported


def test_missing_script_returns_error(capsys):
    rc = cfm.main([str(_REPO_ROOT / "scripts" / "no_such_script.sh")])
    assert rc == 2
    assert "not found" in capsys.readouterr().err
