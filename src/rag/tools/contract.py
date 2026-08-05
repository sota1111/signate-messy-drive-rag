"""Unified tool output contract — every agent-callable tool returns ``{value, evidence, method}``.

A single, explicit shape lets the generation layer treat *any* tool result uniformly: the
answer is ``value``, its provenance (which file / range / rows it came from) is ``evidence``,
and *how* it was produced (engine, code, model, scheme name — never a raw secret) is ``method``.
This turns "the LLM guessed" into "a tool computed X from file Y using method Z", which is the
whole point of tool use on this corpus (暗算・創作の禁止 → 計算と根拠の証跡化).

Contract fields
---------------
``value``
    The answer payload. Any JSON-serializable Python value (str / number / bool / list / dict /
    None). ``None`` is a legitimate value meaning "the tool ran but resolved nothing".
``evidence``
    A mapping describing *where* the value came from — at minimum a ``file`` (or ``source``)
    key, plus tool-specific provenance (rows, columns_used, A1 range, page, sheet, ...).
``method``
    A mapping describing *how* the value was produced — at minimum an ``engine`` key naming the
    mechanism (``pandas`` / ``docx`` / ``xlsx`` / ``glossary`` / ``msoffcrypto`` / ``vision`` ...).
    It records the recipe (code, model, decryption *scheme name*) but **never a raw secret** —
    passwords, alias answer strings and other固有秘密 belong in the runtime adaptation layer, not
    in a tool's returned method (nor in source).

The class is intentionally tiny: existing tools (``compute_sandbox``) already emit a bare dict of
this shape, so :func:`is_contract` accepts both a :class:`ToolResult` and a plain dict, and
:func:`ensure_contract` normalizes either into the canonical dict.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping

CONTRACT_KEYS = ("value", "evidence", "method")


@dataclass
class ToolResult:
    """The unified ``{value, evidence, method}`` result every tool returns.

    ``evidence`` and ``method`` default to empty dicts so a tool can build them incrementally.
    Use :meth:`to_dict` to get the plain, JSON-serializable contract mapping.
    """

    value: Any
    evidence: dict[str, Any] = field(default_factory=dict)
    method: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {"value": self.value, "evidence": dict(self.evidence), "method": dict(self.method)}

    @classmethod
    def from_dict(cls, obj: Mapping[str, Any]) -> "ToolResult":
        if not is_contract(obj):
            raise ContractError(f"not a valid tool contract: {obj!r}")
        return cls(value=obj["value"], evidence=dict(obj["evidence"]), method=dict(obj["method"]))


class ContractError(ValueError):
    """Raised when a value does not conform to the tool contract."""


def is_contract(obj: Any) -> bool:
    """True iff ``obj`` is a valid contract: exactly the 3 keys, dict evidence/method.

    Accepts a :class:`ToolResult` or a plain mapping. ``value`` may be anything (incl. ``None``);
    ``evidence`` and ``method`` must be mappings.
    """
    if isinstance(obj, ToolResult):
        obj = obj.to_dict()
    if not isinstance(obj, Mapping):
        return False
    if set(obj.keys()) != set(CONTRACT_KEYS):
        return False
    return isinstance(obj.get("evidence"), Mapping) and isinstance(obj.get("method"), Mapping)


def ensure_contract(obj: Any) -> dict[str, Any]:
    """Normalize a :class:`ToolResult` or contract mapping into the canonical plain dict.

    Raises :class:`ContractError` if ``obj`` is not a valid contract — use this at tool
    boundaries so a malformed result fails loudly instead of poisoning the answer path.
    """
    if isinstance(obj, ToolResult):
        return obj.to_dict()
    if is_contract(obj):
        return {"value": obj["value"], "evidence": dict(obj["evidence"]), "method": dict(obj["method"])}
    raise ContractError(f"not a valid tool contract: {obj!r}")


def make(value: Any, *, engine: str, evidence: Mapping[str, Any] | None = None,
         **method: Any) -> dict[str, Any]:
    """Convenience builder for the canonical contract dict.

    ``engine`` is the mandatory ``method.engine`` tag; extra keyword args are folded into
    ``method``; ``evidence`` seeds the evidence mapping. Keeps wrappers to one line each.
    """
    m: dict[str, Any] = {"engine": engine}
    m.update(method)
    return {"value": value, "evidence": dict(evidence or {}), "method": m}
