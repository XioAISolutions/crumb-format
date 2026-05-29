"""Write CrumbLLM outputs to a file or stdout."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any


def write_output(content: str, out: str | None) -> None:
    """Write ``content`` to ``out`` (a path) or to stdout when ``out`` is None.

    The path ``-`` also means stdout.
    """
    if out is None or out == "-":
        sys.stdout.write(content)
        if not content.endswith("\n"):
            sys.stdout.write("\n")
        return
    p = Path(out)
    if p.parent and not p.parent.exists():
        p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(content, encoding="utf-8")


def dumps_json(obj: Any) -> str:
    return json.dumps(obj, indent=2, ensure_ascii=False)
