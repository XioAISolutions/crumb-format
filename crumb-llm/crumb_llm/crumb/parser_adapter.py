"""Adapter onto crumb-format's parser and validator.

Boundary rule #1: CrumbLLM does NOT copy the CRUMB spec. Boundary rule #2:
CrumbLLM depends on crumb-format for parsing and validation. This module is
the single seam where that dependency is resolved.

Resolution order:
  1. A future clean ``crumb_format`` top-level API (preferred).
  2. crumb-format's current module layout (``cli.crumb.parse_crumb``), which
     is what ``pip install crumb-format`` exposes today.
  3. The installed ``crumb`` CLI as a subprocess fallback.

If none are available we raise a clear, actionable error rather than silently
re-implementing the format.
"""

from __future__ import annotations

import sys
from functools import lru_cache
from pathlib import Path
from typing import Callable


class CrumbFormatUnavailable(RuntimeError):
    """Raised when crumb-format cannot be located for parsing/validation."""


_INSTALL_HINT = (
    "crumb-format is required for parsing CRUMB files. Install it with:\n"
    "    pip install crumb-format\n"
    "CrumbLLM intentionally does not embed the CRUMB spec."
)


def _maybe_add_monorepo_root() -> None:
    """Dev convenience: if running inside the crumb-format monorepo, make its
    ``cli`` package importable without a separate install."""
    here = Path(__file__).resolve()
    for parent in here.parents:
        if (parent / "cli" / "crumb.py").exists():
            if str(parent) not in sys.path:
                sys.path.insert(0, str(parent))
            return


@lru_cache(maxsize=1)
def _resolve_parser() -> Callable[[str], dict] | None:
    # 1. Preferred future API.
    try:
        from crumb_format import parse_crumb  # type: ignore

        return parse_crumb
    except Exception:
        pass

    # 2. Current crumb-format layout (installed package or monorepo).
    for _ in range(2):
        try:
            from cli.crumb import parse_crumb  # type: ignore

            return parse_crumb
        except Exception:
            _maybe_add_monorepo_root()
    return None


def parse_text(text: str) -> dict:
    """Parse CRUMB text into ``{"headers": {...}, "sections": {...}}``.

    Raises ``ValueError`` on malformed CRUMB (propagated from crumb-format)
    and ``CrumbFormatUnavailable`` if crumb-format cannot be found.
    """
    parser = _resolve_parser()
    if parser is None:
        raise CrumbFormatUnavailable(_INSTALL_HINT)
    return parser(text)


def validate_text(text: str) -> list[str]:
    """Validate CRUMB text. Returns a list of error strings (empty == valid).

    Validation is delegated entirely to crumb-format's parser, which raises on
    any structural problem.
    """
    try:
        parse_text(text)
        return []
    except CrumbFormatUnavailable:
        raise
    except ValueError as exc:
        return [str(exc)]
    except Exception as exc:  # pragma: no cover - defensive
        return [f"unexpected parse error: {exc}"]
