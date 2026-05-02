"""Bridge between OpenTelemetry-style trace JSONL and CRUMB ``kind=log`` crumbs.

Used by ``crumb from-otel`` (generic) and ``crumb from-halo`` (HALO-flavored
defaults; see https://github.com/context-labs/halo). Both consume the same
underlying parser since HALO traces are standard OTEL.

Design notes
------------
- We don't grow the wire format. A trace becomes a ``kind=log`` crumb with
  one bullet per span in ``[entries]``. Inventing a new ``kind=trace``
  would violate the minimal-basis filter (`docs/v1.4-scoping.md`) — adding
  a primitive when an existing one fits is tech debt.
- Permissive parsing. OTEL JSONL shapes vary across vendors and SDK
  versions. We extract well-known fields (``name``, ``traceId``, ``spanId``,
  ``startTimeUnixNano``, status, attributes) and degrade gracefully on
  missing keys rather than rejecting the line.
- One bullet per span, ordered by start time. Tool calls and errors are
  surfaced inline with short prefixes so a downstream AI can grep them.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, Iterator, List, Optional


@dataclass
class Span:
    """Subset of OTEL span fields we surface in a CRUMB log entry."""

    name: str = ""
    trace_id: str = ""
    span_id: str = ""
    parent_span_id: str = ""
    start_unix_nano: int = 0
    end_unix_nano: int = 0
    status_code: str = ""        # "OK" / "ERROR" / "" if unknown
    status_message: str = ""
    attributes: dict = field(default_factory=dict)
    events: List[dict] = field(default_factory=list)
    raw: dict = field(default_factory=dict)


def _canonicalize_status_code(raw) -> str:
    """Normalize an OTEL status code to ``""``, ``"OK"``, or ``"ERROR"``.

    OTLP spec encodes status as an enum: 0=UNSET, 1=OK, 2=ERROR. Real-world
    exporters emit any of: the integer, the string of the integer, the
    short name (``"OK"``/``"ERROR"``), or the prefixed form
    (``"STATUS_CODE_ERROR"``). Anything else collapses to ``""`` so
    ``summarize`` doesn't double-count an unknown shape as an error.
    """
    if raw is None or raw == "":
        return ""
    # Numeric path (int, or a string of digits).
    try:
        as_int = int(raw)
    except (TypeError, ValueError):
        as_int = None
    if as_int is not None:
        return {0: "", 1: "OK", 2: "ERROR"}.get(as_int, "")
    # String path.
    text = str(raw).upper().replace("STATUS_CODE_", "")
    if text in ("UNSET", ""):
        return ""
    if text in ("OK", "ERROR"):
        return text
    return text  # unknown short string — pass through, conservative


def _coerce_int(value) -> int:
    """OTEL nano timestamps come as int, str, or float depending on the SDK."""
    if value is None:
        return 0
    try:
        return int(value)
    except (TypeError, ValueError):
        try:
            return int(float(value))
        except (TypeError, ValueError):
            return 0


def _flatten_attributes(raw_attrs) -> dict:
    """OTEL attributes are sometimes a flat dict, sometimes a list of
    {key, value: {stringValue|intValue|...}} entries. Normalize both
    into a flat string-valued dict.
    """
    if isinstance(raw_attrs, dict):
        return {k: str(v) for k, v in raw_attrs.items() if v is not None}
    out = {}
    if isinstance(raw_attrs, list):
        for entry in raw_attrs:
            if not isinstance(entry, dict):
                continue
            key = entry.get("key")
            value = entry.get("value", entry)
            if isinstance(value, dict):
                # OTLP shape: pick whatever typed-value field is present.
                for k in ("stringValue", "intValue", "doubleValue",
                         "boolValue", "bytesValue"):
                    if k in value:
                        value = value[k]
                        break
            if key:
                out[str(key)] = str(value) if value is not None else ""
    return out


def parse_span(record: dict) -> Span:
    """Parse one JSONL record into a :class:`Span` permissively."""
    name = (
        record.get("name")
        or record.get("operation")
        or record.get("operationName")
        or ""
    )

    # Real-world OTEL JSONL has scattered the `status` field across at
    # least three shapes: a dict ({"code": ..., "message": ...}), a bare
    # string ("OK"/"ERROR"), and a bare scalar (the int enum, e.g. 2).
    # Permissive parsing handles all of them; anything else collapses
    # to "" so we don't AttributeError on .get for non-dicts.
    raw_status = record.get("status")
    if isinstance(raw_status, dict):
        raw_code = raw_status.get("code", "")
        status_message = str(raw_status.get("message", ""))
    elif isinstance(raw_status, (str, int, float)) and not isinstance(raw_status, bool):
        raw_code = raw_status
        status_message = ""
    else:
        raw_code = ""
        status_message = ""
    status_code = _canonicalize_status_code(raw_code)

    return Span(
        name=str(name),
        trace_id=str(record.get("traceId") or record.get("trace_id") or ""),
        span_id=str(record.get("spanId") or record.get("span_id") or ""),
        parent_span_id=str(
            record.get("parentSpanId") or record.get("parent_span_id") or ""
        ),
        start_unix_nano=_coerce_int(
            record.get("startTimeUnixNano") or record.get("start_time_unix_nano")
        ),
        end_unix_nano=_coerce_int(
            record.get("endTimeUnixNano") or record.get("end_time_unix_nano")
        ),
        status_code=status_code,
        status_message=status_message,
        attributes=_flatten_attributes(record.get("attributes")),
        # Guard against non-list `events` payloads (e.g. {"events": 1}
        # from flattened or hand-rolled exports). list(scalar) raises
        # TypeError, which would abort the whole read_otel_jsonl pass.
        events=list(record["events"]) if isinstance(record.get("events"), list) else [],
        raw=record,
    )


def _expand_otlp_envelope(record: dict) -> Iterator[dict]:
    """Yield individual span dicts from an OTLP-shaped envelope record.

    Standard OTLP JSONL often wraps spans in nested envelopes — anywhere
    from one line per span (HALO-style flat JSONL) up to one line per
    *batch* per *resource* per *scope*. The shapes we accept:

      - ``{"resourceSpans": [{"scopeSpans": [{"spans": [...]}, ...]}, ...]}``
      - ``{"scopeSpans": [{"spans": [...]}, ...]}``
      - ``{"spans": [...]}``  (a bare batch)
      - ``{"name": "...", ...}``  (already a single span — no envelope)

    Anything that doesn't match a known envelope shape is yielded as-is
    (treated as a single span). This keeps backward compat with the
    HALO flat form while correctly handling OTLP-spec exporters.
    """
    if "resourceSpans" in record and isinstance(record["resourceSpans"], list):
        for rs in record["resourceSpans"]:
            if isinstance(rs, dict):
                yield from _expand_otlp_envelope(rs)
        return
    if "scopeSpans" in record and isinstance(record["scopeSpans"], list):
        for ss in record["scopeSpans"]:
            if isinstance(ss, dict):
                yield from _expand_otlp_envelope(ss)
        return
    # OTLP renamed "instrumentationLibrarySpans" → "scopeSpans" at v1.0,
    # but older SDKs (and traces archived before the rename) still emit
    # the legacy name. Same nested shape; just a different key.
    if "instrumentationLibrarySpans" in record and isinstance(record["instrumentationLibrarySpans"], list):
        for ils in record["instrumentationLibrarySpans"]:
            if isinstance(ils, dict):
                yield from _expand_otlp_envelope(ils)
        return
    if "spans" in record and isinstance(record["spans"], list):
        for span in record["spans"]:
            if isinstance(span, dict):
                yield span
        return
    # Plain span (HALO-flat or already-flattened export).
    yield record


def read_otel_jsonl(path: str | Path) -> Iterator[Span]:
    """Yield :class:`Span` objects from a JSONL trace file.

    Skips blank lines and lines that don't parse as JSON objects (HALO
    occasionally emits debug log lines mixed in with trace records;
    we don't crash on those). OTLP envelope records are expanded to
    their nested spans before parsing.
    """
    with open(path, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            if not isinstance(record, dict):
                continue
            for span_dict in _expand_otlp_envelope(record):
                if not isinstance(span_dict, dict):
                    continue
                # Belt-and-suspenders: parse_span is hardened against
                # known-bad shapes (scalar status, non-list events,
                # etc.), but real-world JSONL has surprised us before.
                # A single span that triggers a new corner case
                # shouldn't abort the whole file — drop it and
                # continue, matching the bridge's stated permissive
                # contract.
                try:
                    yield parse_span(span_dict)
                except (TypeError, AttributeError, ValueError):
                    continue


def _format_duration_ms(span: Span) -> str:
    if span.start_unix_nano and span.end_unix_nano > span.start_unix_nano:
        return f"{(span.end_unix_nano - span.start_unix_nano) // 1_000_000}ms"
    return ""


def _format_span_bullet(span: Span) -> str:
    """One bullet line for a span — name, status, duration, salient attrs."""
    parts = [f"- {span.name}" if span.name else "- (unnamed span)"]
    if span.status_code and span.status_code != "OK":
        parts.append(f":: {span.status_code.lower()}")
    duration = _format_duration_ms(span)
    if duration:
        parts.append(f"duration={duration}")
    # Surface a few well-known agent-y attributes inline; everything else
    # is dropped to keep the [entries] readable.
    for key in ("model", "model_name", "tool.name", "tool_name", "agent_name"):
        value = span.attributes.get(key)
        if value:
            parts.append(f"{key.replace('.', '_')}={value}")
    if span.status_message:
        parts.append(f"note={span.status_message[:80]!r}")
    return "  ".join(parts)


@dataclass
class TraceSummary:
    """Aggregate stats over a list of spans, surfaced as crumb headers."""

    span_count: int = 0
    error_count: int = 0
    trace_id: str = ""
    started_at: str = ""
    ended_at: str = ""
    total_duration_ms: int = 0


def summarize(spans: List[Span]) -> TraceSummary:
    if not spans:
        return TraceSummary()
    start_min = min((s.start_unix_nano for s in spans if s.start_unix_nano), default=0)
    end_max = max((s.end_unix_nano for s in spans if s.end_unix_nano), default=0)

    def _iso(ns: int) -> str:
        # datetime.fromtimestamp raises ValueError or OverflowError on
        # nano timestamps that are negative, far in the past, or far in
        # the future. A single bad timestamp shouldn't abort the whole
        # summary — return an empty string instead, matching the same
        # permissive posture as the rest of the bridge.
        if not ns:
            return ""
        try:
            return datetime.fromtimestamp(
                ns / 1_000_000_000, tz=timezone.utc
            ).isoformat(timespec="seconds")
        except (ValueError, OverflowError, OSError):
            return ""

    return TraceSummary(
        span_count=len(spans),
        error_count=sum(1 for s in spans if s.status_code == "ERROR"),
        # First span's traceId is canonical when all spans share one.
        trace_id=spans[0].trace_id,
        started_at=_iso(start_min),
        ended_at=_iso(end_max),
        total_duration_ms=(end_max - start_min) // 1_000_000 if end_max > start_min else 0,
    )


def spans_to_log_crumb(
    spans: Iterable[Span],
    *,
    title: str = "OTEL trace",
    source: str = "otel",
    project: str = "",
) -> str:
    """Build a complete CRUMB ``kind=log`` document from a list of spans.

    Produces a minimal v=1.3 log crumb: required headers (`v`, `kind`,
    `source`, `title`), optional `project`, plus computed trace-summary
    headers (``trace_id``, ``span_count``, ``error_count``,
    ``started_at``, ``ended_at``, ``total_duration_ms``). One bullet
    per span in `[entries]`, ordered by start time.

    Empty input still produces a valid crumb (``[entries]`` gets a
    `(no spans)` placeholder so the section isn't empty — empty
    sections fail validation per SPEC §6).
    """
    spans = sorted(spans, key=lambda s: s.start_unix_nano)
    summary = summarize(spans)

    headers = [
        f"v=1.3",
        f"kind=log",
        f"title={title}",
        f"source={source}",
    ]
    if project:
        headers.append(f"project={project}")
    if summary.trace_id:
        headers.append(f"trace_id={summary.trace_id}")
    if summary.started_at:
        headers.append(f"started_at={summary.started_at}")
    if summary.ended_at:
        headers.append(f"ended_at={summary.ended_at}")
    if summary.total_duration_ms:
        headers.append(f"total_duration_ms={summary.total_duration_ms}")
    headers.append(f"span_count={summary.span_count}")
    # Always emit error_count, even when zero. Consumers parsing the
    # header schema need to distinguish "all-OK trace" from "field
    # missing/unknown"; conditional emission breaks that contract.
    headers.append(f"error_count={summary.error_count}")

    if spans:
        entries = [_format_span_bullet(s) for s in spans]
    else:
        entries = ["- (no spans)"]

    return "\n".join(
        [
            "BEGIN CRUMB",
            *headers,
            "---",
            "[entries]",
            *entries,
            "END CRUMB",
            "",
        ]
    )


def jsonl_to_log_crumb(
    path: str | Path,
    *,
    title: str = "OTEL trace",
    source: str = "otel",
    project: str = "",
) -> str:
    """Read OTEL JSONL from disk and produce a kind=log crumb."""
    spans = list(read_otel_jsonl(path))
    return spans_to_log_crumb(spans, title=title, source=source, project=project)
