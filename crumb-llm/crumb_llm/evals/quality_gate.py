"""Top-level quality gate that composes the individual checks."""

from __future__ import annotations

from crumb_llm.evals.hallucination_check import check_hallucinated_paths
from crumb_llm.evals.json_check import check_json

# Phrases that signal a generic, content-free "I can't help" style answer.
_GENERIC_MARKERS = (
    "i'm sorry",
    "i am sorry",
    "i cannot",
    "i can't help",
    "as an ai",
    "no information",
    "i don't have enough",
    "unable to determine",
)

_MIN_USEFUL_LEN = 12


def check_generic_empty(text: str) -> list[str]:
    """Warn when output is empty or a generic non-answer."""
    stripped = (text or "").strip()
    if not stripped:
        return ["model returned an empty response"]
    if len(stripped) < _MIN_USEFUL_LEN:
        return ["model response is suspiciously short"]
    lowered = stripped.lower()
    for marker in _GENERIC_MARKERS:
        if marker in lowered:
            return [f"model returned a generic non-answer (matched {marker!r})"]
    return []


def run_quality_gates(
    text: str,
    source_paths: set[str] | list[str] | None = None,
    expect_json: bool = False,
) -> tuple[object | None, list[str]]:
    """Run all gates over a raw model response.

    Returns ``(parsed_json_or_None, warnings)``. ``parsed_json_or_None`` is the
    decoded object when ``expect_json`` is set and parsing succeeded.
    """
    warnings: list[str] = []
    parsed: object | None = None

    warnings.extend(check_generic_empty(text))

    if expect_json:
        ok, parsed, json_warnings = check_json(text)
        warnings.extend(json_warnings)

    if source_paths is not None:
        warnings.extend(check_hallucinated_paths(text, source_paths))

    return parsed, warnings
