"""Tests for the structured error taxonomy introduced in 0.7.0.

Pins two contracts:

1. Each structural failure raises the *specific* ``CrumbError`` subclass
   with the correct stable ``code`` attribute.
2. Every such exception is still a ``ValueError`` — callers written against
   the pre-0.7.0 parser must continue to work unchanged.
"""

from __future__ import annotations

import pytest

from cli import crumb
from cli.errors import (
    ALL_ERROR_CODES,
    BadVersionError,
    CrumbError,
    E_BAD_VERSION,
    E_EMPTY_SECTION,
    E_INVALID_HEADER_LINE,
    E_MISSING_END_MARKER,
    E_MISSING_HEADER,
    E_MISSING_MARKER,
    E_MISSING_SECTION,
    E_MISSING_SEPARATOR,
    E_ORPHAN_BODY,
    E_UNKNOWN_KIND,
    EmptySectionError,
    InvalidHeaderLineError,
    MissingEndMarkerError,
    MissingHeaderError,
    MissingMarkerError,
    MissingSectionError,
    MissingSeparatorError,
    OrphanBodyError,
    UnknownKindError,
)


# ---------------------------------------------------------------------------
# Base-class invariants
# ---------------------------------------------------------------------------

def test_crumb_error_is_value_error_subclass():
    """Backwards-compatibility: CrumbError must be a ValueError."""
    assert issubclass(CrumbError, ValueError)


@pytest.mark.parametrize(
    "subclass",
    [
        MissingMarkerError,
        MissingEndMarkerError,
        MissingSeparatorError,
        InvalidHeaderLineError,
        MissingHeaderError,
        BadVersionError,
        UnknownKindError,
        OrphanBodyError,
        MissingSectionError,
        EmptySectionError,
    ],
)
def test_every_subclass_is_crumb_error(subclass):
    assert issubclass(subclass, CrumbError)
    assert issubclass(subclass, ValueError)


def test_all_error_codes_unique_and_populated():
    assert len(set(ALL_ERROR_CODES)) == len(ALL_ERROR_CODES)
    assert all(isinstance(code, str) and code for code in ALL_ERROR_CODES)


# ---------------------------------------------------------------------------
# Specific parse-time errors raise the right subclass + code
# ---------------------------------------------------------------------------

def test_missing_begin_marker():
    with pytest.raises(MissingMarkerError) as exc:
        crumb.parse_crumb("v=1.1\nkind=task\nsource=a\n---\n[goal]\nx\nEND CRUMB\n")
    assert exc.value.code == E_MISSING_MARKER


def test_missing_end_marker():
    with pytest.raises(MissingEndMarkerError) as exc:
        crumb.parse_crumb("BEGIN CRUMB\nv=1.1\nkind=task\nsource=a\n---\n[goal]\nx\n")
    assert exc.value.code == E_MISSING_END_MARKER


def test_missing_separator():
    with pytest.raises(MissingSeparatorError) as exc:
        crumb.parse_crumb(
            "BEGIN CRUMB\nv=1.1\nkind=task\nsource=a\n[goal]\nx\nEND CRUMB\n"
        )
    assert exc.value.code == E_MISSING_SEPARATOR


def test_invalid_header_line():
    with pytest.raises(InvalidHeaderLineError) as exc:
        crumb.parse_crumb(
            "BEGIN CRUMB\nv=1.1\nthis-has-no-equals\nsource=a\n---\n[goal]\nx\nEND CRUMB\n"
        )
    assert exc.value.code == E_INVALID_HEADER_LINE


def test_missing_required_header():
    with pytest.raises(MissingHeaderError) as exc:
        # missing kind=
        crumb.parse_crumb(
            "BEGIN CRUMB\nv=1.1\nsource=a\n---\n[goal]\nx\nEND CRUMB\n"
        )
    assert exc.value.code == E_MISSING_HEADER
    assert "kind" in str(exc.value)


def test_bad_version():
    with pytest.raises(BadVersionError) as exc:
        crumb.parse_crumb(
            "BEGIN CRUMB\nv=2.0\nkind=task\nsource=a\n---\n[goal]\nx\nEND CRUMB\n"
        )
    assert exc.value.code == E_BAD_VERSION


def test_unknown_kind():
    with pytest.raises(UnknownKindError) as exc:
        crumb.parse_crumb(
            "BEGIN CRUMB\nv=1.1\nkind=banana\nsource=a\n---\n[goal]\nx\nEND CRUMB\n"
        )
    assert exc.value.code == E_UNKNOWN_KIND


def test_orphan_body_before_first_section():
    with pytest.raises(OrphanBodyError) as exc:
        crumb.parse_crumb(
            "BEGIN CRUMB\nv=1.1\nkind=task\nsource=a\n---\n"
            "stray text\n[goal]\nx\n[context]\n- y\n[constraints]\n- z\nEND CRUMB\n"
        )
    assert exc.value.code == E_ORPHAN_BODY


def test_missing_required_section():
    # task requires goal + context + constraints; omit [constraints]
    with pytest.raises(MissingSectionError) as exc:
        crumb.parse_crumb(
            "BEGIN CRUMB\nv=1.1\nkind=task\nsource=a\n---\n"
            "[goal]\nx\n[context]\n- y\nEND CRUMB\n"
        )
    assert exc.value.code == E_MISSING_SECTION
    assert "constraints" in str(exc.value)


def test_empty_required_section():
    with pytest.raises(EmptySectionError) as exc:
        crumb.parse_crumb(
            "BEGIN CRUMB\nv=1.1\nkind=task\nsource=a\n---\n"
            "[goal]\n\n[context]\n- y\n[constraints]\n- z\nEND CRUMB\n"
        )
    assert exc.value.code == E_EMPTY_SECTION


# ---------------------------------------------------------------------------
# Backwards-compatibility: old code catching bare ValueError still works
# ---------------------------------------------------------------------------

def test_catching_value_error_still_works():
    """Pre-0.7.0 callers catching ValueError must still see the error."""
    with pytest.raises(ValueError):
        crumb.parse_crumb("not a crumb at all")


def test_error_messages_unchanged():
    """Error message strings are part of the user-visible surface.

    Freezing these prevents accidental churn in CLI output or log entries.
    """
    with pytest.raises(CrumbError) as exc:
        crumb.parse_crumb("not a crumb at all")
    assert str(exc.value) == "missing BEGIN CRUMB marker"
