"""Minimal exception helpers for the local RDDL+ parser fork."""

from __future__ import annotations

import warnings

try:
    import termcolor
except Exception:  # pragma: no cover - optional dependency
    termcolor = None


def raise_warning(message: str, color: str = "yellow") -> None:
    """Emit a parser warning with optional terminal coloring."""

    if termcolor is not None:
        message = termcolor.colored(message, color)
    warnings.warn(message)


class RDDLParseError(SyntaxError):
    """Raised when the local RDDL+ parser encounters invalid syntax."""


__all__ = ["RDDLParseError", "raise_warning"]
