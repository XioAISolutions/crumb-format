"""Load CRUMB files into :class:`CrumbDoc` objects."""

from __future__ import annotations

from pathlib import Path

from crumb_llm.crumb.parser_adapter import parse_text
from crumb_llm.models import CrumbDoc


def load_crumb_text(text: str, path: str | None = None) -> CrumbDoc:
    """Parse CRUMB text (via crumb-format) into a :class:`CrumbDoc`."""
    parsed = parse_text(text)
    return CrumbDoc(
        path=path,
        headers=dict(parsed.get("headers", {})),
        sections={k: list(v) for k, v in parsed.get("sections", {}).items()},
        raw_text=text,
    )


def load_crumb(path: str | Path) -> CrumbDoc:
    """Read and parse a ``.crumb`` file from disk."""
    p = Path(path)
    text = p.read_text(encoding="utf-8")
    return load_crumb_text(text, path=str(p))
