"""JSON validity gate."""

from __future__ import annotations

import json
import re
from typing import Any

_FENCE_RE = re.compile(r"```(?:json)?\s*(.*?)```", re.DOTALL)


def _strip_fences(text: str) -> str:
    match = _FENCE_RE.search(text or "")
    return match.group(1).strip() if match else (text or "").strip()


def check_json(text: str) -> tuple[bool, Any, list[str]]:
    """Validate that ``text`` is JSON when JSON output was requested.

    Returns ``(ok, parsed_or_None, warnings)``. Tolerates a single ```json```
    code fence around the payload, since models love to add them.
    """
    candidate = _strip_fences(text)
    if not candidate:
        return False, None, ["expected JSON output but got an empty response"]
    try:
        parsed = json.loads(candidate)
        return True, parsed, []
    except (ValueError, TypeError) as exc:
        return False, None, [f"output was requested as JSON but did not parse: {exc}"]
