"""Canonical agent failure-mode names for the ``[checks]`` section.

This module exposes the closed-list vocabulary defined in
``docs/v1.4/agent-failure-modes.md`` as runtime data, plus a small
suggestion heuristic for ad-hoc names that look like they should map
to a canonical one.

Used by ``crumb lint --check-failure-modes`` to surface:
  - canonical names with their statuses (informational)
  - ad-hoc names with a probable canonical replacement (suggestion)

This module never raises on user input. Unknown names continue to
validate per the existing free-form annotation rule.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Iterable, Iterator, Optional

# The closed-list vocabulary. Order matches the doc; this is what the
# spec calls "canonical names". Anything else is "ad-hoc".
CANONICAL_NAMES: tuple[str, ...] = (
    "hallucinated_tool_call",
    "refusal_loop",
    "tool_error_unhandled",
    "semantic_drift",
    "token_budget_exceeded",
    "invalid_handoff_target",
    "circular_reference",
    "truncated_output",
    "prompt_injection_suspected",
    "unauthorized_tool_call",
)

CANONICAL_SET = frozenset(CANONICAL_NAMES)


# Heuristic mappings. Keys are normalized (lowercased, separators stripped)
# substrings; matches suggest the canonical name. Order matters — first
# match wins, so put more specific patterns first.
_SUGGESTIONS: tuple[tuple[str, str], ...] = (
    ("hallucinatedtoolcall", "hallucinated_tool_call"),
    ("hallucinated", "hallucinated_tool_call"),
    ("hallucination", "hallucinated_tool_call"),
    ("madeuptool", "hallucinated_tool_call"),
    ("nonexistenttool", "hallucinated_tool_call"),
    ("refusalloop", "refusal_loop"),
    ("modelrefused", "refusal_loop"),
    ("repeatedrefusal", "refusal_loop"),
    ("refusal", "refusal_loop"),
    ("toolerror", "tool_error_unhandled"),
    ("tooluncaught", "tool_error_unhandled"),
    ("toolfail", "tool_error_unhandled"),
    ("offtopic", "semantic_drift"),
    ("driftingoffgoal", "semantic_drift"),
    ("contextlost", "semantic_drift"),
    ("budgetexceeded", "token_budget_exceeded"),
    ("oversize", "token_budget_exceeded"),
    ("toomanytokens", "token_budget_exceeded"),
    ("invalidhandoff", "invalid_handoff_target"),
    ("unknownhandoff", "invalid_handoff_target"),
    ("badhandoffrecipient", "invalid_handoff_target"),
    ("cycle", "circular_reference"),
    ("circularref", "circular_reference"),
    ("loopinrefs", "circular_reference"),
    ("truncated", "truncated_output"),
    ("cutoff", "truncated_output"),
    ("incompleteoutput", "truncated_output"),
    ("promptinjection", "prompt_injection_suspected"),
    ("injectionsuspected", "prompt_injection_suspected"),
    ("jailbreak", "prompt_injection_suspected"),
    ("unauthorized", "unauthorized_tool_call"),
    ("policyviolation", "unauthorized_tool_call"),
    ("guardrailviolation", "unauthorized_tool_call"),
)


_NORMALIZE_RE = re.compile(r"[^a-z0-9]")


def _normalize(name: str) -> str:
    return _NORMALIZE_RE.sub("", name.lower())


def is_canonical(name: str) -> bool:
    return name in CANONICAL_SET


def suggest_canonical(name: str) -> Optional[str]:
    """Return the most likely canonical name for an ad-hoc input, or None.

    Performs case-insensitive, separator-insensitive substring matching
    against a hand-curated patterns table. Returns ``None`` when nothing
    matches confidently — the caller should not falsely "correct" an
    unrecognized name to a vocabulary entry.
    """
    if is_canonical(name):
        return None  # already canonical; no suggestion needed
    norm = _normalize(name)
    if not norm:
        return None
    for pattern, target in _SUGGESTIONS:
        if pattern in norm:
            return target
    return None


# ── Lint surface ────────────────────────────────────────────────────────


@dataclass(frozen=True)
class FailureModeFinding:
    line_no: int       # 1-indexed within the [checks] section
    raw_line: str
    name: str
    code: str          # "canonical_failure_mode" | "ad_hoc_with_suggestion"
    message: str


_CHECK_LINE_RE = re.compile(
    r"^\s*-\s*(?P<name>[a-zA-Z_][a-zA-Z0-9_.-]*)\s*::\s*(?P<status>\S+)"
)


def check_failure_mode_lines(checks_lines: Iterable[str]) -> Iterator[FailureModeFinding]:
    """Walk a ``[checks]`` section's lines and yield findings.

    Two finding kinds:

    - ``canonical_failure_mode`` (INFO): the line uses a canonical name.
      Surfaced so consumers can grep for them; not a problem.
    - ``ad_hoc_with_suggestion`` (INFO): the name isn't canonical but
      heuristically maps to one. Caller can suggest a rename.

    Lines that don't match the ``- name :: status`` shape are skipped
    silently — they may be ``@type:`` annotations or prose.
    """
    for line_no, raw in enumerate(checks_lines, start=1):
        line = raw.strip()
        if not line:
            continue
        match = _CHECK_LINE_RE.match(line)
        if not match:
            continue
        name = match.group("name")
        status = match.group("status")

        if is_canonical(name):
            yield FailureModeFinding(
                line_no=line_no,
                raw_line=line,
                name=name,
                code="canonical_failure_mode",
                message=f"canonical failure mode {name!r} :: {status}",
            )
            continue

        suggestion = suggest_canonical(name)
        if suggestion:
            yield FailureModeFinding(
                line_no=line_no,
                raw_line=line,
                name=name,
                code="ad_hoc_with_suggestion",
                message=(
                    f"check name {name!r} looks like the canonical "
                    f"{suggestion!r}; consider renaming for cross-tool "
                    f"portability (see docs/v1.4/agent-failure-modes.md)"
                ),
            )
