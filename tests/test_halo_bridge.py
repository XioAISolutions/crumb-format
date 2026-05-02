"""Tests for cli/halo_bridge — OTEL JSONL → kind=log crumb.

Covers happy path, malformed-input tolerance, OTLP-style attributes,
empty-trace fallback, end-to-end CLI for both `from-otel` and `from-halo`,
and a final round-trip check that the generated crumb actually validates
against the existing parser (so we never emit a malformed crumb).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "cli"))
sys.path.insert(0, str(REPO_ROOT))

from cli import crumb as crumb_cli  # noqa: E402
from cli.halo_bridge import (  # noqa: E402
    Span,
    jsonl_to_log_crumb,
    parse_span,
    read_otel_jsonl,
    spans_to_log_crumb,
    summarize,
)


FIXTURE = REPO_ROOT / "tests" / "fixtures" / "halo-traces.jsonl"


# ── parse_span ─────────────────────────────────────────────────────────


class TestParseSpan:
    def test_basic_span(self):
        s = parse_span({
            "name": "tool.call",
            "traceId": "t1",
            "spanId": "s1",
            "startTimeUnixNano": 1700000001000000000,
            "endTimeUnixNano": 1700000002000000000,
            "status": {"code": "OK"},
            "attributes": {"tool.name": "grep"},
        })
        assert s.name == "tool.call"
        assert s.trace_id == "t1"
        assert s.span_id == "s1"
        assert s.status_code == "OK"
        assert s.attributes["tool.name"] == "grep"

    def test_status_code_strips_otlp_prefix(self):
        # Some OTLP exporters emit "STATUS_CODE_ERROR" rather than "ERROR".
        s = parse_span({"status": {"code": "STATUS_CODE_ERROR", "message": "boom"}})
        assert s.status_code == "ERROR"
        assert s.status_message == "boom"

    def test_status_as_bare_string(self):
        s = parse_span({"status": "OK"})
        assert s.status_code == "OK"

    def test_numeric_status_code_canonicalizes_to_error(self):
        # OTLP wire format encodes status.code as an enum: 2 = ERROR.
        # Real-world exporters often emit the integer (or its string form).
        # Codex finding: a stringified "2" was missing the ERROR branch
        # downstream, so traces with real failures looked healthy.
        s = parse_span({"status": {"code": 2, "message": "boom"}})
        assert s.status_code == "ERROR"
        s2 = parse_span({"status": {"code": "2"}})
        assert s2.status_code == "ERROR"
        s3 = parse_span({"status": {"code": 1}})
        assert s3.status_code == "OK"
        s4 = parse_span({"status": {"code": 0}})
        assert s4.status_code == ""  # UNSET → empty (not counted as error)

    def test_otlp_attribute_list_shape(self):
        # OTLP wire format: attributes are a list of {key, value: {stringValue: ...}}
        s = parse_span({
            "attributes": [
                {"key": "model", "value": {"stringValue": "claude"}},
                {"key": "tokens", "value": {"intValue": 142}},
                {"key": "cost", "value": {"doubleValue": 0.014}},
            ],
        })
        assert s.attributes == {"model": "claude", "tokens": "142", "cost": "0.014"}

    def test_snake_case_keys_accepted(self):
        # Some SDKs emit snake_case; we accept both.
        s = parse_span({
            "trace_id": "t-snake",
            "span_id": "s-snake",
            "parent_span_id": "p-snake",
            "start_time_unix_nano": "1700000000000000000",
            "end_time_unix_nano": "1700000001000000000",
        })
        assert s.trace_id == "t-snake"
        assert s.span_id == "s-snake"
        assert s.parent_span_id == "p-snake"
        assert s.start_unix_nano == 1700000000000000000

    def test_string_nano_timestamps_coerced(self):
        s = parse_span({"startTimeUnixNano": "1700000000000000000"})
        assert s.start_unix_nano == 1700000000000000000

    def test_missing_fields_are_empty_not_crash(self):
        s = parse_span({})
        assert s.name == ""
        assert s.start_unix_nano == 0


# ── read_otel_jsonl ────────────────────────────────────────────────────


class TestReadJsonl:
    def test_skips_blank_and_garbage_lines(self, tmp_path):
        path = tmp_path / "t.jsonl"
        path.write_text(
            json.dumps({"name": "first"}) + "\n"
            + "\n"  # blank
            + "this is not json\n"  # garbage
            + json.dumps({"name": "second"}) + "\n"
            + "\n",
            encoding="utf-8",
        )
        spans = list(read_otel_jsonl(path))
        assert [s.name for s in spans] == ["first", "second"]

    def test_skips_non_object_lines(self, tmp_path):
        path = tmp_path / "t.jsonl"
        path.write_text(
            json.dumps([1, 2, 3]) + "\n"  # array, not object
            + json.dumps({"name": "ok"}) + "\n"
            + '"a string"\n',
            encoding="utf-8",
        )
        spans = list(read_otel_jsonl(path))
        assert [s.name for s in spans] == ["ok"]

    def test_fixture_loads_five_spans(self):
        # The synthetic fixture has 5 trace records + 1 garbage line.
        spans = list(read_otel_jsonl(FIXTURE))
        assert len(spans) == 5
        names = [s.name for s in spans]
        assert "agent.session.start" in names
        assert "agent.refusal" in names


# ── summarize ──────────────────────────────────────────────────────────


class TestSummarize:
    def test_empty(self):
        s = summarize([])
        assert s.span_count == 0
        assert s.error_count == 0

    def test_counts_errors(self):
        spans = list(read_otel_jsonl(FIXTURE))
        s = summarize(spans)
        assert s.span_count == 5
        # Two ERROR-status spans in the fixture (hallucinated tool, refusal).
        assert s.error_count == 2

    def test_trace_id_from_first_span(self):
        spans = list(read_otel_jsonl(FIXTURE))
        assert summarize(spans).trace_id == "trace-abc123"

    def test_iso_timestamps(self):
        spans = list(read_otel_jsonl(FIXTURE))
        s = summarize(spans)
        assert s.started_at.startswith("2023-")
        assert s.ended_at.startswith("2023-")


# ── spans_to_log_crumb ─────────────────────────────────────────────────


class TestSpansToLogCrumb:
    def test_empty_spans_produces_valid_crumb(self):
        text = spans_to_log_crumb([], title="empty", source="test")
        assert "BEGIN CRUMB" in text and "END CRUMB" in text
        assert "[entries]" in text
        # Sections may not be empty per spec — placeholder bullet lives in.
        assert "(no spans)" in text
        # Round-trip: the empty-trace crumb must still validate.
        crumb_cli.parse_crumb(text)

    def test_required_headers(self):
        spans = list(read_otel_jsonl(FIXTURE))
        text = spans_to_log_crumb(spans, title="t", source="otel")
        parsed = crumb_cli.parse_crumb(text)
        assert parsed["headers"]["v"] == "1.3"
        assert parsed["headers"]["kind"] == "log"
        assert parsed["headers"]["source"] == "otel"
        assert parsed["headers"]["title"] == "t"

    def test_summary_headers_emitted(self):
        spans = list(read_otel_jsonl(FIXTURE))
        text = spans_to_log_crumb(spans)
        parsed = crumb_cli.parse_crumb(text)
        assert parsed["headers"]["trace_id"] == "trace-abc123"
        assert parsed["headers"]["span_count"] == "5"
        assert parsed["headers"]["error_count"] == "2"

    def test_entries_ordered_by_start_time(self):
        spans = list(read_otel_jsonl(FIXTURE))
        text = spans_to_log_crumb(spans)
        parsed = crumb_cli.parse_crumb(text)
        entries = parsed["sections"]["entries"]
        # First fixture span by start time is agent.session.start; last is agent.session.end.
        assert "agent.session.start" in entries[0]
        assert "agent.session.end" in entries[-1]

    def test_error_status_surfaced_in_entry(self):
        spans = list(read_otel_jsonl(FIXTURE))
        text = spans_to_log_crumb(spans)
        # The hallucinated-tool span has ERROR status with a "tool not found" message.
        assert ":: error" in text
        assert "tool not found" in text


# ── CLI entry points ───────────────────────────────────────────────────


class TestCliFromOtel:
    def test_from_otel_writes_valid_crumb(self, tmp_path, capsys):
        out = tmp_path / "log.crumb"
        crumb_cli.main(
            ["from-otel", str(FIXTURE), "-o", str(out)]
        )
        text = out.read_text(encoding="utf-8")
        crumb_cli.parse_crumb(text)  # validates clean
        assert "kind=log" in text
        assert "source=otel" in text

    def test_from_otel_to_stdout(self, capsys):
        crumb_cli.main(["from-otel", str(FIXTURE)])
        captured = capsys.readouterr()
        assert "BEGIN CRUMB" in captured.out
        assert "kind=log" in captured.out

    def test_from_otel_custom_title_and_project(self, tmp_path):
        out = tmp_path / "log.crumb"
        crumb_cli.main(
            ["from-otel", str(FIXTURE), "--title", "my run",
             "--project", "alpha", "-o", str(out)]
        )
        parsed = crumb_cli.parse_crumb(out.read_text(encoding="utf-8"))
        assert parsed["headers"]["title"] == "my run"
        assert parsed["headers"]["project"] == "alpha"


class TestCliFromHalo:
    def test_from_halo_default_source_label(self, tmp_path):
        out = tmp_path / "log.crumb"
        crumb_cli.main(["from-halo", str(FIXTURE), "-o", str(out)])
        parsed = crumb_cli.parse_crumb(out.read_text(encoding="utf-8"))
        assert parsed["headers"]["source"] == "halo"
        assert "HALO trace from" in parsed["headers"]["title"]

    def test_from_halo_writes_valid_crumb(self, tmp_path):
        out = tmp_path / "log.crumb"
        crumb_cli.main(["from-halo", str(FIXTURE), "-o", str(out)])
        # Round-trip: must validate against the live parser.
        crumb_cli.parse_crumb(out.read_text(encoding="utf-8"))


# ── jsonl_to_log_crumb (the public combined helper) ────────────────────


class TestJsonlToLogCrumb:
    def test_smoke(self):
        text = jsonl_to_log_crumb(FIXTURE, source="halo")
        crumb_cli.parse_crumb(text)
        assert "trace_id=trace-abc123" in text
