"""Strict ISO-8601 parsing for `[handoff] deadline=` annotations.

Implements the v1.4 deadlines draft (`docs/v1.4/handoff-deadlines.md`).
Each function in this module corresponds to one corner case Codex caught
during review of that draft — the comments call out which one.

Two accepted forms (exclusive):
  - Date-only: ``YYYY-MM-DD``                  → receiver-local
  - Datetime: ``YYYY-MM-DDTHH:MM:SS<tz>``      → tz required
                                                 (``Z`` or ``±HH:MM``)

Anything else is malformed. Callers (validator + lint) treat malformed
values as warnings, never as parse-rejecting errors — a v1.3 free-form
``deadline=Friday`` continues to validate. See ``check_deadline_lines``
for the lint-side surface.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import date, datetime, timezone
from typing import Optional


# Strict regexes. The grammar in the v1.4 draft does NOT permit:
#   - missing seconds (`HH:MM`)
#   - fractional seconds (`HH:MM:SS.fff`)
#   - second-precision offsets (`±HH:MM:SS`)
# So we don't lean on stdlib parser leniency — we anchor a strict regex
# first, then dispatch to stdlib for the actual date math.
DATE_ONLY_RE = re.compile(r"^(\d{4})-(\d{2})-(\d{2})$")
DATETIME_RE = re.compile(
    r"^(\d{4})-(\d{2})-(\d{2})T(\d{2}):(\d{2}):(\d{2})(Z|[+-]\d{2}:\d{2})$"
)


class DeadlineParseError(ValueError):
    """Raised by :func:`parse_deadline` when the value doesn't match the v1.4 grammar."""


@dataclass(frozen=True)
class ParsedDeadline:
    """The result of a successful :func:`parse_deadline` call."""

    kind: str  # "date" or "datetime"
    value: object  # date or aware datetime


def parse_deadline(value: str) -> ParsedDeadline:
    """Parse a ``deadline=`` annotation per the v1.4 grammar.

    Returns a :class:`ParsedDeadline` on success; raises
    :class:`DeadlineParseError` on any deviation from the grammar.

    The dispatch is form-first: the presence of a literal ``T`` selects
    the datetime branch. This matters because Python 3.11+'s
    ``datetime.fromisoformat`` happily accepts a bare ``YYYY-MM-DD`` and
    returns a tz-naive ``datetime`` — a blanket "must have tzinfo" check
    on that result would falsely reject valid date-only deadlines.
    """
    if not isinstance(value, str) or not value:
        raise DeadlineParseError("deadline value must be a non-empty string")

    if "T" in value:
        return _parse_datetime(value)
    return _parse_date(value)


def _parse_date(value: str) -> ParsedDeadline:
    """Date-only branch.

    Spec: receiver-local. We hand off to ``date.fromisoformat`` which
    already round-trip-validates calendar correctness (Feb 30 raises;
    month 13 raises). We re-check the regex first because
    ``date.fromisoformat`` in 3.11+ also accepts loose forms like
    ``20260430`` that the v1.4 grammar does not.
    """
    if not DATE_ONLY_RE.match(value):
        raise DeadlineParseError(
            f"date-only deadline must be YYYY-MM-DD; got {value!r}"
        )
    try:
        parsed = date.fromisoformat(value)
    except ValueError as exc:
        # Out-of-range calendar (Feb 30, month 13). Already shaped right
        # per the regex; just translate the stdlib error.
        raise DeadlineParseError(
            f"date-only deadline {value!r} is not a real calendar date: {exc}"
        ) from exc
    return ParsedDeadline(kind="date", value=parsed)


def _parse_datetime(value: str) -> ParsedDeadline:
    """Datetime branch.

    Strict regex match on the grammar (no missing seconds, no fractional
    seconds, no second-precision offsets). Then normalize the trailing
    ``Z`` to ``+00:00`` because Python 3.10's ``datetime.fromisoformat``
    rejects ``Z``; ``requires-python = ">=3.10"`` is in scope. Finally,
    require ``parsed.tzinfo is not None`` — the regex guarantees this is
    true, but we double-check defensively.
    """
    if not DATETIME_RE.match(value):
        raise DeadlineParseError(
            f"datetime deadline must be YYYY-MM-DDTHH:MM:SS with Z or ±HH:MM; "
            f"got {value!r}"
        )

    # Python 3.10 fromisoformat doesn't accept "Z"; normalize.
    parse_value = value
    if parse_value.endswith("Z"):
        parse_value = parse_value[:-1] + "+00:00"

    try:
        parsed = datetime.fromisoformat(parse_value)
    except ValueError as exc:
        # Out-of-range calendar (Feb 30, month 13, hour 25). Regex
        # accepted the shape; stdlib catches the calendar invalidity.
        raise DeadlineParseError(
            f"datetime deadline {value!r} is not a real timestamp: {exc}"
        ) from exc

    if parsed.tzinfo is None:
        # Defensive: the regex requires Z or ±HH:MM, so we shouldn't
        # reach here, but if the regex ever drifts this catches it.
        raise DeadlineParseError(
            f"datetime deadline {value!r} parsed without a timezone "
            f"(internal: regex/stdlib disagreement)"
        )

    return ParsedDeadline(kind="datetime", value=parsed)


def is_overdue(parsed: ParsedDeadline, now: Optional[datetime] = None) -> bool:
    """Return True if the deadline is in the past relative to ``now``.

    For date-only deadlines, "in the past" means the date is strictly
    before today (receiver-local). For datetime deadlines, the comparison
    is at instant precision in UTC.
    """
    if now is None:
        now = datetime.now(tz=timezone.utc)

    if parsed.kind == "date":
        # Receiver-local: compare against today in the receiver's local zone,
        # which is what `now.astimezone()` gives without an explicit tz arg.
        today_local = now.astimezone().date()
        return parsed.value < today_local

    # datetime: aware comparison in UTC.
    deadline = parsed.value
    if deadline.tzinfo is None:
        # Should not happen — parse_deadline guarantees tzinfo. But if a
        # caller hand-builds a ParsedDeadline, treat naive as not overdue.
        return False
    return deadline < now


# ── Lint surface ────────────────────────────────────────────────────────
# Used by `crumb lint --check-deadlines`. Returns warnings; never raises.

@dataclass(frozen=True)
class DeadlineFinding:
    line_no: int        # 1-indexed line within the [handoff] section
    raw_line: str
    code: str           # "malformed_deadline" | "overdue_deadline"
    message: str


_KV_RE = re.compile(r"\b([a-zA-Z_][a-zA-Z0-9_]*)=(\S+)")


def check_deadline_lines(handoff_lines, now: Optional[datetime] = None):
    """Walk a `[handoff]` section's lines and yield :class:`DeadlineFinding`s.

    ``handoff_lines`` is the list of strings (with or without leading "- ").
    Lines without a ``deadline=`` annotation are silently skipped.
    Malformed deadlines emit a finding; well-formed past deadlines also
    emit a finding. Future deadlines are silent.
    """
    for line_no, raw in enumerate(handoff_lines, start=1):
        line = raw.strip()
        if not line:
            continue
        kv = {m.group(1): m.group(2) for m in _KV_RE.finditer(line)}
        deadline_value = kv.get("deadline")
        if deadline_value is None:
            continue

        try:
            parsed = parse_deadline(deadline_value)
        except DeadlineParseError as exc:
            yield DeadlineFinding(
                line_no=line_no,
                raw_line=line,
                code="malformed_deadline",
                message=str(exc),
            )
            continue

        if is_overdue(parsed, now=now):
            yield DeadlineFinding(
                line_no=line_no,
                raw_line=line,
                code="overdue_deadline",
                message=f"deadline {deadline_value!r} is in the past",
            )
