"""Hallucinated file-path detection.

A common failure mode: the model invents file paths that do not appear in the
source CRUMB. We compare every path-like token in the output against the set of
paths present in the source and warn on any that are not grounded.
"""

from __future__ import annotations

from crumb_llm.models import extract_file_paths


def _basename(path: str) -> str:
    return path.rsplit("/", 1)[-1]


def check_hallucinated_paths(
    output_text: str,
    source_paths: set[str] | list[str],
) -> list[str]:
    """Return warnings for output paths not grounded in ``source_paths``.

    A path is considered grounded if it matches a source path exactly, or if
    its basename matches a source basename (to tolerate relative-prefix drift
    like ``./src/x.py`` vs ``src/x.py``).
    """
    source = set(source_paths)
    source_basenames = {_basename(p) for p in source}
    warnings: list[str] = []
    seen: set[str] = set()

    for path in extract_file_paths(output_text):
        if path in source or _basename(path) in source_basenames:
            continue
        if path in seen:
            continue
        seen.add(path)
        warnings.append(
            f"possible hallucinated path: {path!r} is not referenced in the "
            "source CRUMB"
        )
    return warnings
