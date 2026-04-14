"""Structured error taxonomy for CRUMB parsing.

Introduced in v0.7.0 as a stable public API surface. All error classes
subclass :class:`ValueError` so existing code that catches ``ValueError``
from :func:`cli.crumb.parse_crumb` continues to work unchanged.

Each error carries a stable machine-readable ``code`` attribute.
The string codes are part of the tooling-stability surface and will
not be renamed within the ``crumb-format`` 0.x line. New codes may be
added in minor releases; existing codes are not removed or renumbered
before a major bump.

Usage::

    from cli.errors import CrumbError, E_MISSING_MARKER
    try:
        parse_crumb(text)
    except CrumbError as exc:
        print(exc.code, exc)

Or, for callers that prefer to dispatch on exception type::

    from cli.errors import MissingMarkerError, BadVersionError
    try:
        parse_crumb(text)
    except BadVersionError:
        ...
    except MissingMarkerError:
        ...
"""

# ---------------------------------------------------------------------------
# Stable error code constants
# ---------------------------------------------------------------------------
# These strings are part of the public surface — third-party tooling may
# compare against them. Do not rename or reuse them across releases.

E_MISSING_MARKER = "E_MISSING_MARKER"            # BEGIN CRUMB missing
E_MISSING_END_MARKER = "E_MISSING_END_MARKER"    # END CRUMB missing
E_MISSING_SEPARATOR = "E_MISSING_SEPARATOR"      # --- missing between headers and body
E_INVALID_HEADER_LINE = "E_INVALID_HEADER_LINE"  # header has no '=' separator
E_MISSING_HEADER = "E_MISSING_HEADER"            # required header not present
E_BAD_VERSION = "E_BAD_VERSION"                  # v= header not supported
E_UNKNOWN_KIND = "E_UNKNOWN_KIND"                # kind= header not one of the six
E_ORPHAN_BODY = "E_ORPHAN_BODY"                  # body content before first [section]
E_MISSING_SECTION = "E_MISSING_SECTION"          # required section for kind not present
E_EMPTY_SECTION = "E_EMPTY_SECTION"              # required section present but empty

ALL_ERROR_CODES = (
    E_MISSING_MARKER,
    E_MISSING_END_MARKER,
    E_MISSING_SEPARATOR,
    E_INVALID_HEADER_LINE,
    E_MISSING_HEADER,
    E_BAD_VERSION,
    E_UNKNOWN_KIND,
    E_ORPHAN_BODY,
    E_MISSING_SECTION,
    E_EMPTY_SECTION,
)


# ---------------------------------------------------------------------------
# Exception hierarchy
# ---------------------------------------------------------------------------

class CrumbError(ValueError):
    """Base class for all structural errors raised by :func:`parse_crumb`.

    Subclasses ``ValueError`` for backwards compatibility with callers
    written against the pre-0.7.0 parser, which raised bare ``ValueError``.
    Each instance carries a ``code`` attribute drawn from :data:`ALL_ERROR_CODES`.
    """

    code: str = ""

    def __init__(self, message: str, *, code: str = ""):
        super().__init__(message)
        if code:
            self.code = code


class MissingMarkerError(CrumbError):
    """BEGIN CRUMB marker is missing or misplaced."""
    code = E_MISSING_MARKER


class MissingEndMarkerError(CrumbError):
    """END CRUMB marker is missing or misplaced."""
    code = E_MISSING_END_MARKER


class MissingSeparatorError(CrumbError):
    """The `---` separator between headers and body is missing."""
    code = E_MISSING_SEPARATOR


class InvalidHeaderLineError(CrumbError):
    """A header line is present but has no `key=value` form."""
    code = E_INVALID_HEADER_LINE


class MissingHeaderError(CrumbError):
    """A required header (``v``, ``kind``, ``source``) is missing."""
    code = E_MISSING_HEADER


class BadVersionError(CrumbError):
    """The ``v=`` header is not a supported version."""
    code = E_BAD_VERSION


class UnknownKindError(CrumbError):
    """The ``kind=`` header is not one of the recognised kinds."""
    code = E_UNKNOWN_KIND


class OrphanBodyError(CrumbError):
    """Body content appears before the first ``[section]`` marker."""
    code = E_ORPHAN_BODY


class MissingSectionError(CrumbError):
    """A section required for the declared kind is missing."""
    code = E_MISSING_SECTION


class EmptySectionError(CrumbError):
    """A required section is present but has no content."""
    code = E_EMPTY_SECTION


__all__ = [
    # Base
    "CrumbError",
    # Subclasses
    "MissingMarkerError",
    "MissingEndMarkerError",
    "MissingSeparatorError",
    "InvalidHeaderLineError",
    "MissingHeaderError",
    "BadVersionError",
    "UnknownKindError",
    "OrphanBodyError",
    "MissingSectionError",
    "EmptySectionError",
    # Code constants
    "E_MISSING_MARKER",
    "E_MISSING_END_MARKER",
    "E_MISSING_SEPARATOR",
    "E_INVALID_HEADER_LINE",
    "E_MISSING_HEADER",
    "E_BAD_VERSION",
    "E_UNKNOWN_KIND",
    "E_ORPHAN_BODY",
    "E_MISSING_SECTION",
    "E_EMPTY_SECTION",
    "ALL_ERROR_CODES",
]
